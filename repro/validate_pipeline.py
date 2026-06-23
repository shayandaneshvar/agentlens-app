"""
Validate that the AgentLens SDK scoring code reproduces the shipped dataset.

For each trajectory: load its per-trajectory PTA (candidate) and the task's
merged k=5 ground-truth PTA, run match.run + quality_assessment, and compare
the recomputed quality_score / quality_tier against the annotation.

Usage: python repro/validate_pipeline.py [--per-task N] [--tasks T]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sdk" / "src"))
warnings.filterwarnings("ignore")

from swe_trace_sdk import io, match  # noqa: E402

BENCH = ROOT / "agentlens-bench"
IDEAL_MIN, LUCKY_MAX = 70, 47


def revised_tier(score: int, passed: bool) -> str:
    """Exact tier rule used by experiments/build_dataset.py."""
    if not passed:
        return "partial_fail" if score >= 40 else "off_track"
    if score >= IDEAL_MIN:
        return "ideal"
    if score < LUCKY_MAX:
        return "lucky"
    return "solid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=8,
                    help="max trajectories sampled per task (0 = all)")
    ap.add_argument("--tasks", type=int, default=12,
                    help="number of tasks to sample (0 = all 47)")
    args = ap.parse_args()

    ann = pd.read_parquet(BENCH / "annotations" / "trajectories.parquet")
    ann = ann.set_index("trajectory_id")

    task_dirs = sorted((BENCH / "trajectories").iterdir())
    if args.tasks:
        # evenly spaced sample across the 47 tasks for diversity
        idx = np.linspace(0, len(task_dirs) - 1, args.tasks).astype(int)
        task_dirs = [task_dirs[i] for i in sorted(set(idx))]

    rng = np.random.default_rng(42)
    recs = []
    for td in task_dirs:
        task = td.name
        gt_path = BENCH / "ground_truth" / f"{task}_merged_pta.json"
        if not gt_path.exists():
            continue
        gt = io.load_saved_trace(gt_path)
        files = sorted(td.glob("*.json"))
        if args.per_task and len(files) > args.per_task:
            files = [files[i] for i in sorted(rng.choice(len(files), args.per_task, replace=False))]
        for fp in files:
            tid = fp.stem
            if tid not in ann.index:
                continue
            row = ann.loc[tid]
            cand = io.load_saved_trace(fp)
            res = match.run(cand, gt)
            qr = match.quality_assessment(res, cand, gt, passed=bool(row["passed"]))
            recs.append({
                "task": task, "tid": tid,
                "score_ann": int(row["quality_score"]),
                "score_repro": int(qr.quality_score),
                "tier_ann": row["quality_tier"],
                "tier_repro": revised_tier(int(qr.quality_score), bool(row["passed"])),
            })
        print(f"  scored {task} ({len([r for r in recs if r['task']==task])} trajs)", flush=True)

    r = pd.DataFrame(recs)
    signed = r["score_repro"] - r["score_ann"]
    abs_err = signed.abs()
    tier_match = (r["tier_repro"] == r["tier_ann"]).mean()
    print(f"\n  Mean signed diff (repro-ann): {signed.mean():+.2f}  (bias direction)")

    print("\n" + "=" * 70)
    print(f"PIPELINE VALIDATION  ({len(r)} trajectories, {r['task'].nunique()} tasks)")
    print("=" * 70)
    print(f"  Mean |score diff|         : {abs_err.mean():.2f}")
    print(f"  Median |score diff|       : {abs_err.median():.1f}")
    print(f"  Within +/-2 points        : {(abs_err <= 2).mean()*100:.1f}%")
    print(f"  Within +/-5 points        : {(abs_err <= 5).mean()*100:.1f}%")
    print(f"  Exact tier match          : {tier_match*100:.1f}%")
    print(f"  Score correlation (Pearson): {r['score_ann'].corr(r['score_repro']):.4f}")
    # tier distribution comparison on the sample (passing only)
    print("\n  Tier distribution on sampled passing trajectories:")
    passing = r[r["tid"].isin(ann[ann["passed"]].index)]
    for col, label in [("tier_ann", "annotated"), ("tier_repro", "reproduced")]:
        vc = passing[col].value_counts()
        n = len(passing)
        print(f"    {label:<11}", {k: f"{v} ({100*v/n:.0f}%)" for k, v in vc.items()})


if __name__ == "__main__":
    main()
