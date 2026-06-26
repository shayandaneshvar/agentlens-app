#!/usr/bin/env python3
"""
Evaluate your own SWE-bench trajectories with AgentLens, using the paper's PTAs.

Point it at a results folder (like ./my-swebench-sample) whose layout is:

    <folder>/
      result.json                      # optional aggregate file (ignored here)
      <task>__<randomsuffix>/
        result.json                    # has verifier_result.rewards.reward (1.0/0.0)
        agent/trajectory.json          # ATIF trajectory (schema_version ATIF-v1.x)

For every instance whose <task> has a paper reference, it:
  1. loads the trajectory (adapting this agent's tool names to canonical ones),
  2. builds/loads the task reference PTA,
  3. matches and computes AgentLens quality_score, tier, key metrics,
  4. uses reward=1/0 from result.json as the pass/fail label.

Reference PTA (the "paper PTA"):
  --k 5 (default)  build a fresh k=5 merge (seed=42) from the paper's released
                   passing trajectories for that task — the paper's documented
                   hyperparameters (merge_k=5, seed=42).
  --k 0            use the shipped agentlens-bench/ground_truth/<task>.json
                   (note: those are 14-trace all-passing merges, not k=5).

Instances whose task has no reference are skipped (reported at the end).

Usage:
    python repro/evaluate_my_trajectories.py my-swebench-sample
    python repro/evaluate_my_trajectories.py my-swebench-sample --no-outcome
    python repro/evaluate_my_trajectories.py my-swebench-sample --k 0   # shipped GT
    python repro/evaluate_my_trajectories.py <folder> --out r.json --csv r.csv
"""
import argparse
import json
import logging
import random
import statistics
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sdk" / "src"))
warnings.filterwarnings("ignore")
logging.getLogger("swe_trace_sdk").setLevel(logging.ERROR)  # silence per-tool warnings

from swe_trace_sdk import io, match, trace as trace_api  # noqa: E402

# The ATIF loader's _resolve_tool only consults _TOOL_NAME_MAP and ignores its
# own _CANONICAL_TOOLS set, so already-canonical names (e.g. "think", which the
# registry labels Orchestration) get dropped as "unknown". Register the
# canonical tools as identity mappings so they pass through. (Contained shim;
# does not modify SDK source.)
import swe_trace_sdk._generator_atif as _atif_gen  # noqa: E402
for _t in _atif_gen._CANONICAL_TOOLS:
    _atif_gen._TOOL_NAME_MAP.setdefault(_t, _t)

# Reuse the dataset's exact 5-category waste detectors (build_dataset is
# import-safe: functions are module-level, main() is guarded).
sys.path.insert(0, str(ROOT / "experiments"))
from build_dataset import (  # noqa: E402
    get_ordered_states,
    detect_regression_loops,
    detect_redundant_steps,
    detect_unnecessary_exploration,
    extract_blind_retries,
    extract_cyclic_patterns,
)

# (category key, label) — GT-aware; patterns in the merged PTA are not counted.
WASTE_CATS = [
    ("regression_loop", "regression loops"),
    ("blind_retry", "blind retries"),
    ("redundant_step", "redundant steps"),
    ("unnecessary_exploration", "unnecessary exploration"),
    ("cyclic_pattern", "cyclic patterns"),
]

IDEAL_MIN, LUCKY_MAX = 70, 47
DEFAULT_K, DEFAULT_SEED = 5, 42

# This agent's tools -> the function_name keys the SDK ATIF loader understands.
# file_editor is command-sensitive, so we rewrite it per call.
FILE_EDITOR_CMD = {
    "view": "view",            # -> read_file
    "create": "create",        # -> create_file
    "str_replace": "edit",     # -> replace_string_in_file
    "str_replace_editor": "edit",
    "insert": "edit",
    "undo_edit": "edit",
}
# Whole-function renames. None => drop the tool call.
#   terminal -> run_in_terminal (intent labeler reads the command text).
#   think    -> passed through unchanged: it is a canonical SDK tool whose
#               stage_hint is ORCHESTRATION (paper: O = "bookkeeping and
#               reasoning steps"), so we KEEP it.
#   finish   -> dropped: episode-end marker, no code operation.
FUNCTION_RENAME = {
    "terminal": "bash",
    "finish": None,
}


def revised_tier(score: int, passed: bool) -> str:
    """Tier rule used by experiments/build_dataset.py."""
    if not passed:
        return "partial_fail" if score >= 40 else "off_track"
    if score >= IDEAL_MIN:
        return "ideal"
    if score < LUCKY_MAX:
        return "lucky"
    return "solid"


def score_no_outcome(m) -> int:
    """AgentLens composite WITHOUT the 0.10*outcome term (pure process score).

    Mirrors SDK _compute_quality_score minus the outcome component.
    """
    base = (
        0.25 * m.coverage_percent
        + 0.25 * (m.coherence_score * 100.0)
        + 0.18 * (m.stage_completeness * 100.0)
        + 0.12 * (m.workflow_similarity * 100.0)
        + 0.10 * m.f1_score
    )
    return max(0, min(100, int(round(base))))


# ── Lucky Pass taxonomy (paper Appendix D.1) ───────────────────────────────
# Priority-ordered, deterministic decision tree applied ONLY to Lucky-tier
# passing trajectories. The paper fixes the ORDER (C1→C2→C4→C3→C5) and C1's
# rule exactly ("<=8 steps, zero waste, no verification"); C3 is its
# incomplete-implementation failure reason; C5 is the remainder. The paper does
# NOT publish C2 / C4 thresholds, so we calibrated them on the released 122
# Lucky passes to reproduce the paper's split (C1:19 C2:42 C3:41 C4:5 C5:15):
#   C2 (Brute-Force) = waste_severity >= 0.30   -> recovers C2's high-waste
#                      profile (paper mean severity 0.47) and separates it from
#                      C3 (paper severity 0.05). "Any pattern present" over-counts.
#   C4 (Excessive)   = length >= 40             -> paper C4 mean length 50.4.
# This matches the paper within 6/122 (C1 and C5 exact).
C2_WASTE_SEVERITY = 0.30
C4_MIN_LENGTH = 40

LUCKY_CATEGORY_NAMES = {
    "C1": "Minimal & Unverified",
    "C2": "Brute-Force Convergence",
    "C3": "Incomplete Implementation",
    "C4": "Excessive Exploration",
    "C5": "Divergent-but-Valid",
}


def classify_lucky(length, total_waste, waste_severity,
                   has_verification, incomplete_impl) -> str:
    """Assign a Lucky Pass to exactly one of C1–C5 (paper's priority cascade)."""
    if length <= 8 and total_waste == 0 and not has_verification:
        return "C1"                       # minimal & unverified
    if waste_severity >= C2_WASTE_SEVERITY:
        return "C2"                       # brute-force convergence (thrashing)
    if length >= C4_MIN_LENGTH:
        return "C4"                       # excessive exploration
    if incomplete_impl:
        return "C3"                       # incomplete implementation
    return "C5"                           # divergent-but-valid


def detect_waste(cand, gt, qr):
    """Run the 5 GT-aware waste detectors for one (candidate, reference) pair.

    Returns {cat_key: {"count": int, "wasted_steps": int}} for the 5 categories.
    """
    cand_states = get_ordered_states(cand)
    gt_paths = match._enumerate_paths(gt)
    gt_best = gt_paths[0] if gt_paths else []
    return {
        "regression_loop": detect_regression_loops(cand_states, gt_best),
        "redundant_step": detect_redundant_steps(cand_states, gt_best),
        "unnecessary_exploration": detect_unnecessary_exploration(cand_states, gt_best, gt),
        "blind_retry": extract_blind_retries(qr),
        "cyclic_pattern": extract_cyclic_patterns(qr),
    }


def adapt_trajectory(raw: dict) -> dict:
    """Rewrite tool_call function_names to canonical SDK keys, dropping no-ops."""
    out_steps = []
    for step in raw.get("steps", []):
        tcs = step.get("tool_calls") or []
        new_tcs = []
        for tc in tcs:
            fn = tc.get("function_name", "")
            args = tc.get("arguments", {}) or {}
            if fn == "file_editor":
                mapped = FILE_EDITOR_CMD.get(args.get("command", ""), "view")
            elif fn in FUNCTION_RENAME:
                mapped = FUNCTION_RENAME[fn]
            else:
                mapped = fn  # canonical (e.g. "think") or unknown -> passthrough
            if mapped is None:
                continue
            new_tcs.append(dict(tc, function_name=mapped))
        if new_tcs or not tcs:
            out_steps.append(dict(step, tool_calls=new_tcs) if tcs else step)
    return dict(raw, steps=out_steps)


def load_candidate(traj_path: Path):
    """Adapt + load one ATIF trajectory into an SDK Trace."""
    raw = json.loads(traj_path.read_text())
    adapted = adapt_trajectory(raw)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(adapted, fh)
        tmp = fh.name
    try:
        return io.load_trajectory(tmp, format="atif")
    finally:
        Path(tmp).unlink(missing_ok=True)


def build_reference(task: str, k: int, seed: int):
    """Return the reference PTA for *task*.

    k == 0: shipped ground_truth/<task>_merged_pta.json (14-trace artifact).
    k  > 0: fresh k-merge (seeded) from the released PASSING trajectories.
    """
    if k == 0:
        gt = ROOT / "agentlens-bench" / "ground_truth" / f"{task}_merged_pta.json"
        return io.load_saved_trace(gt) if gt.exists() else None

    task_dir = ROOT / "agentlens-bench" / "trajectories" / task
    if not task_dir.exists():
        return None
    # released trajectory filenames encode outcome as "-pass-" / "-fail-"
    passing = sorted(f for f in task_dir.glob("*.json") if "-pass-" in f.name)
    if len(passing) < k:
        return None
    donors = passing.copy()
    random.Random(seed).shuffle(donors)
    donors = donors[:k]
    return trace_api.merge([io.load_saved_trace(d) for d in donors], use_llm=False)


def get_reward(inst_dir: Path):
    rj = inst_dir / "result.json"
    if not rj.exists():
        return None
    try:
        d = json.loads(rj.read_text())
        r = (d.get("verifier_result") or {}).get("rewards", {}).get("reward")
        return None if r is None else bool(float(r) >= 1.0)
    except Exception:
        return None


def task_name_of(inst_dir: Path) -> str:
    rj = inst_dir / "result.json"
    if rj.exists():
        try:
            t = json.loads(rj.read_text()).get("task_name")
            if t:
                return t
        except Exception:
            pass
    return inst_dir.name.rsplit("__", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="results folder containing <task>__<suffix>/ dirs")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="merge count for the reference PTA (0 = shipped 14-trace GT)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="donor-selection seed")
    ap.add_argument("--resamples", type=int, default=1,
                    help="for k>0: build N k-merges from N donor draws (seed, seed+1, ...) "
                         "and report mean +/- std score. A single draw is noisy; use >=5 "
                         "for a robust, paper-faithful estimate.")
    ap.add_argument("--no-outcome", action="store_true",
                    help="score without the 0.10*outcome term (pure process score)")
    ap.add_argument("--out", default=None, help="write per-trajectory results JSON here")
    ap.add_argument("--csv", default=None, help="also write a flat CSV here")
    args = ap.parse_args()

    folder = Path(args.folder)
    inst_dirs = sorted(d for d in folder.iterdir()
                       if d.is_dir() and (d / "agent" / "trajectory.json").exists())

    # Resampling only applies to fresh k-merges (k>0); the shipped GT is fixed.
    n_resamples = max(1, args.resamples) if args.k > 0 else 1
    seeds = [args.seed + i for i in range(n_resamples)]

    ref_label = "shipped-GT(all-passing merge)" if args.k == 0 else \
        f"k={args.k} seed={args.seed}" + (f" x{n_resamples} resamples" if n_resamples > 1 else "")
    score_label = "process-only (no outcome)" if args.no_outcome else "with +10 pass bonus"
    print(f"\nReference: {ref_label}   |   Score: {score_label}")

    ref_cache = {}  # (task, seed) -> ref Trace or None
    results, skipped = [], []
    for d in inst_dirs:
        task = task_name_of(d)
        passed = get_reward(d)
        try:
            cand = load_candidate(d / "agent" / "trajectory.json")
            if not cand.states:
                skipped.append((d.name, task, "empty trace after adaptation"))
                continue
            scores, covs, cohs, tiers = [], [], [], []
            waste_runs = []      # list of {cat: {count, wasted_steps}} per reference
            lucky_cats = []      # per-draw Lucky category (only used if tier==lucky)
            # length used for the taxonomy = ordered/labeled state count (matches
            # build_dataset's n_states, the denominator of waste_severity).
            n_ordered = len(get_ordered_states(cand))
            for s in seeds:
                key = (task, s)
                if key not in ref_cache:
                    ref_cache[key] = build_reference(task, args.k, s)
                gt = ref_cache[key]
                if gt is None:
                    continue
                res = match.run(cand, gt)
                qr = match.quality_assessment(res, cand, gt, passed=passed)
                sc = score_no_outcome(res.metrics) if args.no_outcome else qr.quality_score
                scores.append(sc)
                tiers.append(revised_tier(sc, bool(passed)))
                covs.append(qr.key_metrics.get("coverage_percent", 0.0))
                cohs.append(qr.key_metrics.get("coherence", 0.0))
                w = detect_waste(cand, gt, qr)
                waste_runs.append(w)
                # signals for the Lucky-Pass decision tree (this draw)
                total_w = sum(w[c]["wasted_steps"] for c, _ in WASTE_CATS)
                has_v = res.metrics.stage_coverage.get("verification", 0) > 0
                incomplete = any(getattr(fr, "reason", "") == "incomplete_implementation"
                                 for fr in (qr.failure_reasons or []))
                lucky_cats.append(classify_lucky(
                    n_ordered, total_w, total_w / max(n_ordered, 1), has_v, incomplete))
            if not scores:
                skipped.append((d.name, task, "no reference (no PTA / too few passing)"))
                continue
            mean_score = int(round(statistics.mean(scores)))
            std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            mean_tier = revised_tier(mean_score, bool(passed))
            tier_set = sorted(set(tiers))
            # mean waste per category across references
            waste = {}
            for c, _ in WASTE_CATS:
                waste[f"{c}_count"] = round(
                    statistics.mean(w[c]["count"] for w in waste_runs), 1)
                waste[f"{c}_waste"] = round(
                    statistics.mean(w[c]["wasted_steps"] for w in waste_runs), 1)
            total_wasted = round(
                statistics.mean(sum(w[c]["wasted_steps"] for c, _ in WASTE_CATS)
                                for w in waste_runs), 1)
            # Lucky Pass category: only meaningful when the trajectory is Lucky.
            # Use the modal category across donor draws.
            lucky_cat = None
            if mean_tier == "lucky" and lucky_cats:
                lucky_cat = Counter(lucky_cats).most_common(1)[0][0]
            results.append({
                "instance": d.name,
                "task": task,
                "passed": passed,
                "n_states": len(cand.states),
                "n_refs": len(scores),
                "quality_score": mean_score,
                "score_std": round(std_score, 1),
                "score_min": min(scores),
                "score_max": max(scores),
                "tier": mean_tier,
                "tier_stable": len(tier_set) == 1,
                "tiers_seen": tier_set,
                "coverage_percent": round(statistics.mean(covs), 1),
                "coherence": round(statistics.mean(cohs), 3),
                "lucky_category": lucky_cat,
                "lucky_category_name": LUCKY_CATEGORY_NAMES.get(lucky_cat) if lucky_cat else None,
                "total_wasted_steps": total_wasted,
                **waste,
            })
        except Exception as e:
            skipped.append((d.name, task, f"error: {e}"))

    note = f" (mean over up to {n_resamples} donor draws)" if n_resamples > 1 else ""
    print(f"\nEvaluated {len(results)} trajectory(ies){note} "
          f"(from {len(inst_dirs)} instance dirs).\n")
    if results:
        hdr = (f"{'task':<28}{'pass':>5}{'score':>7}{'±std':>6}{'tier':>13}"
               f"{'stbl':>5}{'cov%':>7}{'coh':>6}")
        print(hdr); print("-" * len(hdr))
        for r in sorted(results, key=lambda x: -x["quality_score"]):
            print(f"{r['task']:<28}{str(r['passed']):>5}{r['quality_score']:>7}"
                  f"{r['score_std']:>6}{r['tier']:>13}{('y' if r['tier_stable'] else 'N'):>5}"
                  f"{r['coverage_percent']:>7}{r['coherence']:>6}")

    # ---- summary: outcome counts + tier counts ----
    if results:
        n_pass = sum(1 for r in results if r["passed"] is True)
        n_fail = sum(1 for r in results if r["passed"] is False)
        n_unk = sum(1 for r in results if r["passed"] is None)
        tier_counts = {t: 0 for t in
                       ("ideal", "solid", "lucky", "partial_fail", "off_track")}
        for r in results:
            tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        print(f"  total scored : {len(results)}")
        print(f"  passed       : {n_pass}")
        print(f"  failed       : {n_fail}" + (f"   (unknown: {n_unk})" if n_unk else ""))
        print("  tiers:")
        print(f"    [pass] ideal        : {tier_counts['ideal']}")
        print(f"    [pass] solid        : {tier_counts['solid']}")
        print(f"    [pass] lucky        : {tier_counts['lucky']}")
        print(f"    [fail] partial_fail : {tier_counts['partial_fail']}")
        print(f"    [fail] off_track    : {tier_counts['off_track']}")
        unstable = [r["instance"] for r in results if not r["tier_stable"]]
        if unstable:
            print(f"  tier varied across donor draws for {len(unstable)} trajectory(ies) "
                  f"(see 'stbl=N') — interpret those tiers as borderline.")

    # ---- waste report: 5 GT-aware categories (count / wasted steps) ----
    if results:
        print("\n" + "=" * 40)
        print("WASTE (5 categories, GT-aware)")
        print("=" * 40)
        print("  count = instances detected; waste = wasted steps "
              "(mean over donor draws). Patterns already in the reference PTA are excluded.")
        abbr = {"regression_loop": "regress", "blind_retry": "retry",
                "redundant_step": "redund", "unnecessary_exploration": "unnec-expl",
                "cyclic_pattern": "cyclic"}
        hdr = f"{'task':<26}" + "".join(f"{abbr[c]:>12}" for c, _ in WASTE_CATS) + f"{'TOTAL':>8}"
        print(hdr); print("-" * len(hdr))
        for r in sorted(results, key=lambda x: -x["total_wasted_steps"]):
            cells = "".join(f"{str(r[f'{c}_count'])+'/'+str(r[f'{c}_waste']):>12}"
                            for c, _ in WASTE_CATS)
            print(f"{r['task']:<26}{cells}{r['total_wasted_steps']:>8}")
        # roll-up: total wasted steps per category across all trajectories
        print("  totals (wasted steps, summed over trajectories):")
        for c, label in WASTE_CATS:
            tot = round(sum(r[f"{c}_waste"] for r in results), 1)
            print(f"    {label:<24}: {tot}")

    # ---- Lucky Pass taxonomy (only for Lucky-tier passing trajectories) ----
    lucky = [r for r in results if r.get("lucky_category")]
    if lucky:
        print("\n" + "=" * 40)
        print("LUCKY PASS TAXONOMY (paper C1–C5)")
        print("=" * 40)
        print("  Why each Lucky pass is 'lucky' (decision tree over process signals):")
        for r in sorted(lucky, key=lambda x: x["lucky_category"]):
            print(f"  {r['task']:<28} {r['lucky_category']}: {r['lucky_category_name']}")
        cc = Counter(r["lucky_category"] for r in lucky)
        print("  counts:", {k: cc[k] for k in sorted(cc)})
    elif any(r["tier"] == "lucky" for r in results):
        print("\n(LUCKY PASS TAXONOMY: lucky trajectories present but category unresolved)")

    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for name, task, why in skipped:
            print(f"  {name}  ({task}) — {why}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}")
    if args.csv and results:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
