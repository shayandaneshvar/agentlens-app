"""
Quality Analysis Experiment
============================
For each eligible task (≥5 passing trajectories):
  1. Build a fresh k=5 merged PTA from randomly selected passing trajectories
  2. Run quality_assessment() on all remaining trajectories (excluding the 5 donors)
  3. Record quality_score and quality_tier per trajectory

Produces:
  - Model ranking by mean quality score (contrasted with pass-rate ranking)
  - Lucky pass prevalence statistics (the paper's headline finding)
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add SDK to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdk" / "src"))

from swe_trace_sdk import trace as trace_api
from swe_trace_sdk import match

# ============================================================================
# Configuration
# ============================================================================

MERGE_K = 5              # Number of passing trajectories to merge
MAX_PTA_STATES = 9999    # Effectively no limit (path enumeration is capped)
SEED = 42                # Random seed for reproducibility
TASK_TIMEOUT = 300       # Max seconds per task (skip remaining if exceeded)
TRAJ_TIMEOUT = 60        # Max seconds per trajectory (unused in in-process mode)

# ============================================================================
# Helpers
# ============================================================================

def parse_pta_filename(filename: str, task_name: str) -> Optional[Tuple[str, bool, str]]:
    """Parse a PTA filename to extract model name, pass/fail, and run ID.
    
    Expected format: {task_name}-logs-{model_name}-{pass|fail}-{run_id}_pta.json
    
    Returns:
        (model_name, passed, run_id) or None if parsing fails.
    """
    if filename.endswith("_merged_pta.json"):
        return None
    if not filename.endswith("_pta.json"):
        return None
    base = filename[:-len("_pta.json")]
    prefix = f"{task_name}-logs-"
    if not base.startswith(prefix):
        return None
    rest = base[len(prefix):]
    m = re.match(r'^(.+)-(pass|fail)-(\d+)$', rest)
    if not m:
        return None
    return m.group(1), m.group(2) == "pass", m.group(3)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to both console and file."""
    logger = logging.getLogger("quality_analysis")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    log_file = output_dir / "quality_analysis.log"
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


# ============================================================================
# Main Analysis
# ============================================================================

def run_quality_analysis(
    experiment_outputs_dir: Path,
    output_dir: Path,
    merge_k: int = MERGE_K,
    max_pta_states: int = MAX_PTA_STATES,
    task_timeout: int = TASK_TIMEOUT,
    traj_timeout: int = TRAJ_TIMEOUT,
    seed: int = SEED,
) -> Dict:
    """Run quality assessment on all eligible tasks."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("Quality Analysis Experiment")
    logger.info(f"  merge_k={merge_k}, max_pta_states={max_pta_states}, seed={seed}, task_timeout={task_timeout}s")
    logger.info("=" * 70)
    
    start_time = time.time()
    rng = random.Random(seed)
    
    # Discover tasks
    task_dirs = sorted([
        d for d in experiment_outputs_dir.iterdir()
        if d.is_dir() and (d / "pta_outputs").is_dir()
    ])
    logger.info(f"Found {len(task_dirs)} tasks with PTA outputs")
    
    # Results collection
    all_results = []
    task_summaries = []
    skipped_tasks_insufficient = []
    skipped_tasks_too_large = []
    
    for task_idx, task_dir in enumerate(task_dirs, 1):
        task_name = task_dir.name
        pta_dir = task_dir / "pta_outputs"
        pta_files = sorted(pta_dir.glob("*.json"))
        
        passing_ptas = []
        failing_ptas = []
        
        for f in pta_files:
            parsed = parse_pta_filename(f.name, task_name)
            if parsed is None:
                continue
            model_name, passed, run_id = parsed
            entry = {
                "path": f,
                "model": model_name,
                "passed": passed,
                "run_id": run_id,
                "trajectory_id": f.stem,
            }
            if passed:
                passing_ptas.append(entry)
            else:
                failing_ptas.append(entry)
        
        total_trajs = len(passing_ptas) + len(failing_ptas)
        
        # Check eligibility: need at least merge_k passing trajectories
        if len(passing_ptas) < merge_k:
            skipped_tasks_insufficient.append((task_name, len(passing_ptas)))
            logger.info(f"[{task_idx}/{len(task_dirs)}] {task_name}: SKIP (only {len(passing_ptas)} passing, need {merge_k})")
            continue
        
        # Select merge donors (shuffle + first k)
        passing_shuffled = passing_ptas.copy()
        rng.shuffle(passing_shuffled)
        donors = passing_shuffled[:merge_k]
        donor_ids = {d["trajectory_id"] for d in donors}
        
        # Build merged PTA
        logger.info(f"[{task_idx}/{len(task_dirs)}] {task_name}: merging {merge_k} PTAs...")
        try:
            donor_traces = [trace_api.load(str(d["path"]), format="trace") for d in donors]
            merged_pta = trace_api.merge(donor_traces, use_llm=False)
        except Exception as e:
            logger.warning(f"  Merge failed: {e}")
            skipped_tasks_too_large.append((task_name, f"merge_error: {e}"))
            continue
        
        num_states = len(merged_pta.states)
        num_transitions = len(merged_pta.transitions)
        if num_states >= max_pta_states:
            skipped_tasks_too_large.append((task_name, f"{num_states} states"))
            logger.info(f"  SKIP: merged PTA has {num_states} states (>= {max_pta_states})")
            continue
        
        logger.info(f"  Merged PTA: {num_states} states, {num_transitions} transitions. Assessing {total_trajs - merge_k} trajectories...")
        
        # Assess all non-donor trajectories (in-process, safe with path limit in SDK)
        task_results = []
        assessed_count = 0
        error_count = 0
        timeout_count = 0
        task_start = time.time()
        
        all_candidates = passing_ptas + failing_ptas
        for entry in all_candidates:
            if entry["trajectory_id"] in donor_ids:
                continue
            
            # Task-level timeout
            if time.time() - task_start > task_timeout:
                remaining = sum(1 for e in all_candidates if e["trajectory_id"] not in donor_ids) - assessed_count - error_count - timeout_count
                logger.warning(f"  Task timeout ({task_timeout}s) reached. Skipping {remaining} remaining.")
                timeout_count += remaining
                break
            
            try:
                candidate = trace_api.load(str(entry["path"]), format="trace")
                result = match.run(candidate, merged_pta, use_llm=False)
                report = match.quality_assessment(result, candidate, merged_pta, passed=entry["passed"])
                
                record = {
                    "task": task_name,
                    "model": entry["model"],
                    "trajectory_id": entry["trajectory_id"],
                    "passed": entry["passed"],
                    "quality_score": report.quality_score,
                    "quality_tier": report.quality_tier,
                }
                task_results.append(record)
                all_results.append(record)
                assessed_count += 1
                
            except (MemoryError, RecursionError) as e:
                logger.warning(f"  MemoryError/RecursionError on {entry['trajectory_id']}: {e}")
                error_count += 1
            except Exception as e:
                logger.warning(f"  Error on {entry['trajectory_id']}: {e}")
                error_count += 1
        
        task_summaries.append({
            "task": task_name,
            "merged_states": num_states,
            "assessed": assessed_count,
            "errors": error_count,
            "timeouts": timeout_count,
            "passing_assessed": sum(1 for r in task_results if r["passed"]),
            "failing_assessed": sum(1 for r in task_results if not r["passed"]),
        })
        
        logger.info(f"  Done: {assessed_count} assessed, {error_count} errors, {timeout_count} timeouts (task time: {time.time()-task_start:.1f}s)")
    
    elapsed = time.time() - start_time
    
    # ========================================================================
    # Compute Aggregates
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    
    # Per-model aggregates
    model_stats = defaultdict(lambda: {
        "total": 0, "passed": 0, "failed": 0,
        "quality_scores": [], "tiers": defaultdict(int)
    })
    
    for r in all_results:
        model = r["model"]
        model_stats[model]["total"] += 1
        if r["passed"]:
            model_stats[model]["passed"] += 1
            model_stats[model]["quality_scores"].append(r["quality_score"])
            model_stats[model]["tiers"][r["quality_tier"]] += 1
        else:
            model_stats[model]["failed"] += 1
    
    # Model ranking table
    model_table = []
    for model, stats in model_stats.items():
        pass_count = stats["passed"]
        total = stats["total"]
        pass_rate = pass_count / total if total > 0 else 0
        mean_qs = (sum(stats["quality_scores"]) / len(stats["quality_scores"])) if stats["quality_scores"] else 0
        model_table.append({
            "model": model,
            "pass_rate": pass_rate,
            "pass_count": pass_count,
            "total": total,
            "mean_quality_score": mean_qs,
            "ideal_count": stats["tiers"].get("ideal", 0),
            "solid_count": stats["tiers"].get("solid", 0),
            "lucky_count": stats["tiers"].get("lucky", 0),
        })
    
    by_pass_rate = sorted(model_table, key=lambda x: -x["pass_rate"])
    by_quality = sorted(model_table, key=lambda x: -x["mean_quality_score"])
    for rank, entry in enumerate(by_pass_rate, 1):
        entry["pass_rate_rank"] = rank
    for rank, entry in enumerate(by_quality, 1):
        entry["quality_rank"] = rank
    for e in model_table:
        e["rank_delta"] = e["pass_rate_rank"] - e["quality_rank"]
    
    # Print results
    logger.info("\n--- MODEL RANKING COMPARISON ---")
    logger.info(f"{'Model':<20} {'PassRate':>8} {'PR Rank':>8} {'MeanQS':>8} {'QS Rank':>8} {'Delta':>6} {'Ideal':>6} {'Solid':>6} {'Lucky':>6}")
    logger.info("-" * 90)
    for e in sorted(model_table, key=lambda x: x["quality_rank"]):
        logger.info(
            f"{e['model']:<20} {e['pass_rate']*100:>7.1f}% {e['pass_rate_rank']:>8} "
            f"{e['mean_quality_score']:>8.1f} {e['quality_rank']:>8} {e['rank_delta']:>+6} "
            f"{e['ideal_count']:>6} {e['solid_count']:>6} {e['lucky_count']:>6}"
        )
    
    all_passing = [r for r in all_results if r["passed"]]
    total_passing = len(all_passing)
    tier_counts = defaultdict(int)
    for r in all_passing:
        tier_counts[r["quality_tier"]] += 1
    
    logger.info(f"\n--- PASS QUALITY DISTRIBUTION (n={total_passing}) ---")
    logger.info(f"{'Tier':<15} {'Count':>8} {'Percentage':>12}")
    logger.info("-" * 40)
    for tier in ["ideal", "solid", "lucky"]:
        count = tier_counts.get(tier, 0)
        pct = (count / total_passing * 100) if total_passing > 0 else 0
        logger.info(f"{tier:<15} {count:>8} {pct:>11.1f}%")
    
    all_failing = [r for r in all_results if not r["passed"]]
    total_failing = len(all_failing)
    fail_tier_counts = defaultdict(int)
    for r in all_failing:
        fail_tier_counts[r["quality_tier"]] += 1
    
    logger.info(f"\n--- FAIL QUALITY DISTRIBUTION (n={total_failing}) ---")
    logger.info(f"{'Tier':<15} {'Count':>8} {'Percentage':>12}")
    logger.info("-" * 40)
    for tier in ["partial_fail", "off_track"]:
        count = fail_tier_counts.get(tier, 0)
        pct = (count / total_failing * 100) if total_failing > 0 else 0
        logger.info(f"{tier:<15} {count:>8} {pct:>11.1f}%")
    
    logger.info(f"\n--- SUMMARY ---")
    logger.info(f"Tasks processed: {len(task_summaries)}")
    logger.info(f"Tasks skipped (insufficient passing): {len(skipped_tasks_insufficient)}")
    logger.info(f"Tasks skipped (merged PTA too large): {len(skipped_tasks_too_large)}")
    logger.info(f"Total trajectories assessed: {len(all_results)}")
    logger.info(f"  Passing: {total_passing}")
    logger.info(f"  Failing: {total_failing}")
    if total_passing > 0:
        lucky_pct = tier_counts.get("lucky", 0) / total_passing * 100
        logger.info(f"\n*** HEADLINE: {lucky_pct:.1f}% of passes are Lucky Passes ***")
    logger.info(f"\nTotal time: {elapsed:.1f}s")
    
    # Save Results
    output_data = {
        "config": {
            "merge_k": merge_k,
            "max_pta_states": max_pta_states,
            "task_timeout": task_timeout,
            "traj_timeout": traj_timeout,
            "seed": seed,
        },
        "summary": {
            "tasks_processed": len(task_summaries),
            "tasks_skipped_insufficient": len(skipped_tasks_insufficient),
            "tasks_skipped_too_large": len(skipped_tasks_too_large),
            "total_assessed": len(all_results),
            "total_passing_assessed": total_passing,
            "total_failing_assessed": total_failing,
            "elapsed_seconds": elapsed,
        },
        "pass_quality_distribution": {
            tier: {"count": tier_counts.get(tier, 0), "pct": (tier_counts.get(tier, 0) / total_passing * 100) if total_passing > 0 else 0}
            for tier in ["ideal", "solid", "lucky"]
        },
        "fail_quality_distribution": {
            tier: {"count": fail_tier_counts.get(tier, 0), "pct": (fail_tier_counts.get(tier, 0) / total_failing * 100) if total_failing > 0 else 0}
            for tier in ["partial_fail", "off_track"]
        },
        "model_ranking": sorted(model_table, key=lambda x: x["quality_rank"]),
        "task_summaries": task_summaries,
        "skipped_tasks_insufficient": skipped_tasks_insufficient,
        "skipped_tasks_too_large": skipped_tasks_too_large,
        "trajectory_results": all_results,
    }
    
    results_file = output_dir / "quality_analysis_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_file}")
    return output_data


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Quality Analysis Experiment")
    parser.add_argument("experiment_outputs", type=Path, help="Path to experiment_outputs directory")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("new_research_experiments/quality_analysis_outputs"))
    parser.add_argument("--merge-k", type=int, default=MERGE_K)
    parser.add_argument("--max-states", type=int, default=MAX_PTA_STATES)
    parser.add_argument("--task-timeout", type=int, default=TASK_TIMEOUT)
    parser.add_argument("--traj-timeout", type=int, default=TRAJ_TIMEOUT)
    parser.add_argument("--seed", type=int, default=SEED)
    
    args = parser.parse_args()
    
    if not args.experiment_outputs.is_dir():
        print(f"ERROR: {args.experiment_outputs} is not a directory")
        sys.exit(1)
    
    run_quality_analysis(
        experiment_outputs_dir=args.experiment_outputs,
        output_dir=args.output_dir,
        merge_k=args.merge_k,
        max_pta_states=args.max_states,
        task_timeout=args.task_timeout,
        traj_timeout=args.traj_timeout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
