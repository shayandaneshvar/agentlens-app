"""
Reproduce AgentLens scores with the paper's hyperparameters: k=5, seed=42.

Unlike reproduce_paper.py (which reads the shipped annotation column), this
RE-RUNS the scoring pipeline end-to-end:
  for each task, pick 5 donors (seed=42) from the released passing trajectories,
  merge them into a k=5 PTA, then score every remaining trajectory with
  match.run + quality_assessment, apply build_dataset.revised_tier, and
  aggregate tier distribution + model comparison.

Caveat: the ORIGINAL 5 donors per task were excluded from the release (verified:
GT num_traces == released_passing + 5 for all 47 tasks), so these k=5 references
use DIFFERENT donors than the paper. Numbers are therefore close-but-not-exact;
this validates the methodology, not bit-identical scores.
"""
import logging
import random
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sdk" / "src"))
warnings.filterwarnings("ignore")
logging.getLogger("swe_trace_sdk").setLevel(logging.ERROR)

from swe_trace_sdk import io, match, trace as trace_api  # noqa: E402

K, SEED = 5, 42
IDEAL_MIN, LUCKY_MAX = 70, 47
TRAJ = ROOT / "agentlens-bench" / "trajectories"

MODEL_RE = re.compile(r"-logs-(.+?)-(?:pass|fail)-")


def model_of(name: str) -> str:
    m = MODEL_RE.search(name)
    return m.group(1) if m else "unknown"


def revised_tier(score: int, passed: bool) -> str:
    if not passed:
        return "partial_fail" if score >= 40 else "off_track"
    if score >= IDEAL_MIN:
        return "ideal"
    if score < LUCKY_MAX:
        return "lucky"
    return "solid"


def main():
    rng = random.Random(SEED)  # one shared RNG, tasks in sorted order (as build_dataset)
    task_dirs = sorted(d for d in TRAJ.iterdir() if d.is_dir())

    records = []
    skipped = []
    for td in task_dirs:
        task = td.name
        passing = sorted(f for f in td.glob("*.json") if "-pass-" in f.name)
        failing = sorted(f for f in td.glob("*.json") if "-fail-" in f.name)
        if len(passing) < K:
            skipped.append((task, len(passing)))
            continue
        donors = passing.copy()
        rng.shuffle(donors)
        donor_set = set(donors[:K])
        try:
            merged = trace_api.merge([io.load_saved_trace(d) for d in donor_set], use_llm=False)
        except Exception as e:
            skipped.append((task, f"merge_error: {e}"))
            continue

        for f in passing + failing:
            if f in donor_set:
                continue
            passed = "-pass-" in f.name
            try:
                cand = io.load_saved_trace(f)
                res = match.run(cand, merged)
                qr = match.quality_assessment(res, cand, merged, passed=passed)
                records.append({
                    "task": task, "model": model_of(f.name), "passed": passed,
                    "score": qr.quality_score,
                    "tier": revised_tier(qr.quality_score, passed),
                    "coherence": res.metrics.coherence_score,
                })
            except Exception:
                pass
        print(f"  {task}: scored {sum(1 for r in records if r['task']==task)} "
              f"(merged {len(merged.states)} states)", flush=True)

    # ---------- aggregate ----------
    passing = [r for r in records if r["passed"]]
    failing = [r for r in records if not r["passed"]]
    np_ = len(passing)

    def pct(n, d):
        return round(100.0 * n / d, 1) if d else 0.0

    tc = defaultdict(int)
    for r in passing:
        tc[r["tier"]] += 1

    print("\n" + "=" * 74)
    print(f"k={K}, seed={SEED} RE-SCORED  ({len(records)} trajectories, "
          f"{np_} passing, {len(failing)} failing)")
    print("=" * 74)
    print("Quality tier distribution among PASSING (paper: 20.2 / 69.1 / 10.7):")
    for t in ("ideal", "solid", "lucky"):
        print(f"  {t:<8} {tc[t]:>4}  ({pct(tc[t], np_)}%)")

    print("\nModel comparison (mean QS of passing, Lucky%) — paper Table 2 in parens:")
    paper = {"sonnet-4.5": (67.4, 1.0), "opus-4.5": (66.2, 0.5), "gpt-4o": (63.4, 4.1),
             "gemini-2.5-pro": (59.2, 7.6), "gpt-5.3-codex": (58.3, 15.3),
             "opus-4.6": (56.7, 18.7), "gpt-5.2-codex": (56.1, 19.4), "gpt-4.1": (54.7, 23.2)}
    by_model = defaultdict(list)
    for r in passing:
        by_model[r["model"]].append(r)
    rows = []
    for model, rs in by_model.items():
        qs = sum(x["score"] for x in rs) / len(rs)
        lucky = pct(sum(1 for x in rs if x["tier"] == "lucky"), len(rs))
        rows.append((model, qs, lucky, len(rs)))
    rows.sort(key=lambda x: -x[1])
    print(f"  {'model':<16}{'QS':>7}{'Lucky%':>8}{'n':>6}   | paper(QS,Lucky%)")
    for model, qs, lucky, n in rows:
        p = paper.get(model)
        ptxt = f"{p[0]},{p[1]}" if p else "-"
        print(f"  {model:<16}{qs:>7.1f}{lucky:>8.1f}{n:>6}   | {ptxt}")

    if skipped:
        print(f"\nSkipped {len(skipped)} task(s) with <{K} released passing: "
              f"{[t for t, _ in skipped]}")


if __name__ == "__main__":
    main()
