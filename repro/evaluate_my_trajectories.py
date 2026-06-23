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
import sys
import tempfile
import warnings
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
    ap.add_argument("--no-outcome", action="store_true",
                    help="score without the 0.10*outcome term (pure process score)")
    ap.add_argument("--out", default=None, help="write per-trajectory results JSON here")
    ap.add_argument("--csv", default=None, help="also write a flat CSV here")
    args = ap.parse_args()

    folder = Path(args.folder)
    inst_dirs = sorted(d for d in folder.iterdir()
                       if d.is_dir() and (d / "agent" / "trajectory.json").exists())

    ref_label = "shipped-GT(14-trace)" if args.k == 0 else f"k={args.k} seed={args.seed}"
    score_label = "process-only (no outcome)" if args.no_outcome else "with +10 pass bonus"
    print(f"\nReference: {ref_label}   |   Score: {score_label}")

    ref_cache = {}
    results, skipped = [], []
    for d in inst_dirs:
        task = task_name_of(d)
        if task not in ref_cache:
            ref_cache[task] = build_reference(task, args.k, args.seed)
        gt = ref_cache[task]
        if gt is None:
            skipped.append((d.name, task, "no reference (no PTA / too few passing)"))
            continue
        passed = get_reward(d)
        try:
            cand = load_candidate(d / "agent" / "trajectory.json")
            if not cand.states:
                skipped.append((d.name, task, "empty trace after adaptation"))
                continue
            res = match.run(cand, gt)
            qr = match.quality_assessment(res, cand, gt, passed=passed)
            score = score_no_outcome(res.metrics) if args.no_outcome else qr.quality_score
            km = qr.key_metrics
            results.append({
                "instance": d.name,
                "task": task,
                "passed": passed,
                "n_states": len(cand.states),
                "quality_score": score,
                "tier": revised_tier(score, bool(passed)),
                "sdk_verdict": qr.verdict,
                "coverage_percent": round(km.get("coverage_percent", 0.0), 1),
                "coherence": round(km.get("coherence", 0.0), 3),
                "stage_completeness": round(km.get("stage_completeness", 0.0), 3),
                "workflow_similarity": round(km.get("workflow_similarity", 0.0), 3),
                "divergence_step": getattr(qr.divergence_point, "step", None)
                if qr.divergence_point else None,
            })
        except Exception as e:
            skipped.append((d.name, task, f"error: {e}"))

    print(f"\nEvaluated {len(results)} trajectory(ies) (from {len(inst_dirs)} instance dirs).\n")
    if results:
        hdr = f"{'task':<28}{'pass':>5}{'score':>7}{'tier':>13}{'cov%':>7}{'coh':>6}{'div':>5}"
        print(hdr); print("-" * len(hdr))
        for r in sorted(results, key=lambda x: -x["quality_score"]):
            print(f"{r['task']:<28}{str(r['passed']):>5}{r['quality_score']:>7}"
                  f"{r['tier']:>13}{r['coverage_percent']:>7}{r['coherence']:>6}"
                  f"{str(r['divergence_step']):>5}")
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
