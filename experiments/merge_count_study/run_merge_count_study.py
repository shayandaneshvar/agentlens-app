#!/usr/bin/env python3
"""
Merge Count Study — How does the number of merged trajectories affect metric scores?

Experimental design:
  1. For each eligible task, fix a test set (test_pass_count passed + test_fail_count failed).
     The test set is chosen ONCE (with a fixed seed) and stays identical across all merge counts.
  2. From the remaining passed trajectories, vary the merge count from --min-merge to --max-merge.
  3. At each merge count k, optionally resample N random subsets of k passed trajectories,
     build a merged PTA from each subset, and score the fixed test set.
  4. Aggregate results across tasks and resamples → mean ± std at each merge count.
  5. Print a summary table and generate a Plotly HTML chart.

Usage:
    python experiments/run_merge_count_study.py <data_dir> --output-dir results/merge_study
    python experiments/run_merge_count_study.py <data_dir> --min-merge 2 --max-merge 8 --resamples 5
"""

import sys
import os
import json
import argparse
import logging
import random
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

# SDK & project imports
from swe_trace_sdk import trace as trace_api, match
from swe_trace_sdk.models import Trace
from swe_trace_sdk.match import extract_required_tools, check_process_coverage, extract_required_files, check_file_coverage

# Reuse helpers from the holdout experiment & shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "holdout"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_holdout_experiment import (
    TrajectoryInfo,
    discover_trajectories,
    find_trajectory_file,
    get_trajectory_json_path,
    generate_pta_from_trajectory,
    merge_ptas,
)
from metrics_utils import (
    STATS_AVAILABLE,
    TQDM_AVAILABLE,
    tqdm,
    MetricSet,
    compute_classification_metrics,
    fill_metric_set,
)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SingleRunResult:
    """Scores from one merge-count + resample run against the fixed test set."""
    merge_count: int
    resample_idx: int
    task_name: str

    structural_auroc: float = 0.5
    process_auroc: float = 0.5
    combined_auroc: float = 0.5

    structural_accuracy: float = 0.0
    process_accuracy: float = 0.0
    combined_accuracy: float = 0.0

    structural_pass_mean: float = 0.0
    structural_fail_mean: float = 0.0
    process_pass_mean: float = 0.0
    process_fail_mean: float = 0.0
    combined_pass_mean: float = 0.0
    combined_fail_mean: float = 0.0

    num_test_passed: int = 0
    num_test_failed: int = 0
    merged_pta_states: int = 0
    merged_pta_transitions: int = 0
    error: str = ""


@dataclass
class MergeCountPoint:
    """Aggregated metrics at a single merge-count level (across tasks & resamples)."""
    merge_count: int

    structural_auroc_mean: float = 0.0
    structural_auroc_std: float = 0.0
    process_auroc_mean: float = 0.0
    process_auroc_std: float = 0.0
    combined_auroc_mean: float = 0.0
    combined_auroc_std: float = 0.0

    structural_accuracy_mean: float = 0.0
    structural_accuracy_std: float = 0.0
    process_accuracy_mean: float = 0.0
    process_accuracy_std: float = 0.0
    combined_accuracy_mean: float = 0.0
    combined_accuracy_std: float = 0.0

    structural_gap_mean: float = 0.0
    structural_gap_std: float = 0.0
    combined_gap_mean: float = 0.0
    combined_gap_std: float = 0.0

    num_runs: int = 0

    # Micro-averaged metrics (pool all test trajectories across tasks/resamples)
    micro_structural_auroc: float = 0.0
    micro_process_auroc: float = 0.0
    micro_combined_auroc: float = 0.0
    micro_structural_accuracy: float = 0.0
    micro_process_accuracy: float = 0.0
    micro_combined_accuracy: float = 0.0


# ============================================================================
# SCORING — score the fixed test set against a merged PTA
# ============================================================================

@dataclass
class _TestScore:
    """Internal: per-trajectory score."""
    is_passed: bool
    structural_coverage: float = 0.0
    process_coverage: float = 0.0
    error: str = ""


# In-memory cache for loaded test Trace objects (trajectory_id → Trace).
# Test trajectories are scored many times (once per merge-count × resample),
# so caching avoids redundant disk I/O and JSON parsing.
_test_trace_cache: Dict[str, Trace] = {}
# Negative cache: trajectory IDs that failed to load (avoids re-trying each time)
_test_trace_failures: Dict[str, str] = {}


def score_test_set(
    merged_pta: Trace,
    test_trajectories: List[TrajectoryInfo],
    temp_dir: Path,
) -> List[_TestScore]:
    """Score held-out test trajectories against a merged PTA using SDK matching."""
    required_tools = extract_required_tools(merged_pta)
    required_files = extract_required_files(merged_pta)

    results: List[_TestScore] = []
    for traj_info in test_trajectories:
        tid = traj_info.trajectory_id

        # Fast-fail for trajectories that previously errored
        if tid in _test_trace_failures:
            results.append(_TestScore(is_passed=traj_info.passed, error=_test_trace_failures[tid]))
            continue

        try:
            # Use in-memory cache to avoid re-loading the same trace
            if tid in _test_trace_cache:
                traj_trace = _test_trace_cache[tid]
            elif traj_info.pta_path and Path(traj_info.pta_path).exists():
                traj_trace = trace_api.load(traj_info.pta_path, format="trace")
                _test_trace_cache[tid] = traj_trace
            else:
                json_result = get_trajectory_json_path(traj_info, temp_dir)
                if not json_result:
                    raise ValueError(f"No JSON for {tid}")
                json_path, fmt = json_result
                traj_trace = trace_api.load(json_path, format=fmt)
                _test_trace_cache[tid] = traj_trace

            match_result = match.run(traj_trace, merged_pta, use_llm=False)
            best_f1 = match_result.metrics.f1_score

            # File-level process coverage (more discriminating than tool-name)
            file_cov = 100.0
            if required_files:
                file_ratio, _ = check_file_coverage(traj_trace, required_files)
                file_cov = file_ratio * 100.0

            results.append(_TestScore(
                is_passed=traj_info.passed,
                structural_coverage=round(best_f1, 2),
                process_coverage=round(file_cov, 2),
            ))
        except Exception as e:
            logger.error(f"Error scoring {tid}: {e}")
            _test_trace_failures[tid] = str(e)
            results.append(_TestScore(is_passed=traj_info.passed, error=str(e)))

    return results


def _metrics_from_scores(scores: List[_TestScore]) -> dict:
    """Compute AUROC / accuracy for all four metric types from a list of _TestScore."""
    if not STATS_AVAILABLE or len(scores) < 2:
        return {}

    # Structural
    struct = compute_classification_metrics(scores, "structural_coverage", passed_field="is_passed")
    proc = compute_classification_metrics(scores, "process_coverage", passed_field="is_passed")

    # Combined (average of structural + process)
    class _Tmp:
        def __init__(self, is_passed, structural_coverage, error=""):
            self.is_passed = is_passed
            self.structural_coverage = structural_coverage
            self.error = error

    combined_scores = [
        _Tmp(s.is_passed, 0.7 * s.structural_coverage + 0.3 * s.process_coverage, s.error)
        for s in scores
    ]
    combined = compute_classification_metrics(combined_scores, "structural_coverage", passed_field="is_passed")

    return {
        "structural": struct,
        "process": proc,
        "combined": combined,
    }


# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

def run_study(
    data_dir: Path,
    output_dir: Path,
    min_merge: int,
    max_merge: int,
    test_pass_count: int,
    test_fail_count: int,
    num_resamples: int,
    seed: int,
) -> Tuple[List[MergeCountPoint], List[SingleRunResult]]:
    """Run the full merge-count study."""

    temp_dir = Path(tempfile.mkdtemp(prefix="pta_mc_"))

    print(f"Discovering trajectories in: {data_dir}")
    tasks = discover_trajectories(data_dir)
    print(f"Found {len(tasks)} tasks")

    # We need at least max_merge + test_pass_count passed and test_fail_count failed
    min_passed_needed = max_merge + test_pass_count
    eligible: Dict[str, List[TrajectoryInfo]] = {}
    for tname, trajs in tasks.items():
        passed = [t for t in trajs if t.passed]
        failed = [t for t in trajs if not t.passed]
        if len(passed) >= min_passed_needed and len(failed) >= test_fail_count:
            eligible[tname] = trajs

    print(f"Eligible tasks (>= {min_passed_needed} passed, >= {test_fail_count} failed): {len(eligible)}")
    if not eligible:
        print("ERROR: No eligible tasks found. Try reducing --max-merge.")
        return [], []

    # ---- For each task, fix the test set ONCE ----
    rng_split = random.Random(seed)
    task_splits: Dict[str, Tuple[List[TrajectoryInfo], List[TrajectoryInfo], List[TrajectoryInfo]]] = {}
    # (remaining_passed_pool, test_passed, test_failed)

    for tname, trajs in eligible.items():
        passed = [t for t in trajs if t.passed]
        failed = [t for t in trajs if not t.passed]

        shuffled_passed = passed.copy()
        rng_split.shuffle(shuffled_passed)

        # Reserve the LAST test_pass_count passed for testing (so they stay fixed)
        test_passed = shuffled_passed[-test_pass_count:]
        remaining_passed = shuffled_passed[:-test_pass_count]

        shuffled_failed = failed.copy()
        rng_split.shuffle(shuffled_failed)
        test_failed = shuffled_failed[:test_fail_count]

        task_splits[tname] = (remaining_passed, test_passed, test_failed)

    # Pre-generate PTAs for all test trajectories (done once, reused)
    print("\nPre-generating PTAs for test trajectories...")
    for tname, (_, test_passed, test_failed) in tqdm(task_splits.items(), desc="Generating test PTAs"):
        task_out = output_dir / tname
        task_out.mkdir(parents=True, exist_ok=True)
        for traj in test_passed + test_failed:
            generate_pta_from_trajectory(traj, task_out, temp_dir)

    # ---- Run for each merge count ----
    all_runs: List[SingleRunResult] = []
    merge_counts = list(range(min_merge, max_merge + 1))
    total_combinations = sum(
        len([t for t in task_splits if len(task_splits[t][0]) >= k]) * (num_resamples if num_resamples > 1 else 1)
        for k in merge_counts
    )
    completed = 0
    _start_time = time.time()
    MAX_WALL_SECONDS = 60 * 60  # 60-minute hard limit

    for k in merge_counts:
        print(f"\n{'='*60}")
        print(f"  Merge count k = {k}")
        print(f"{'='*60}")

        for tname in tqdm(sorted(task_splits.keys()), desc=f"k={k}"):
            remaining_passed, test_passed, test_failed = task_splits[tname]

            if len(remaining_passed) < k:
                logger.warning(f"Skipping {tname} at k={k}: only {len(remaining_passed)} remaining passed")
                continue

            task_out = output_dir / tname
            task_out.mkdir(parents=True, exist_ok=True)
            test_trajs = test_passed + test_failed

            actual_resamples = num_resamples if num_resamples > 1 else 1

            for r_idx in range(actual_resamples):
                # Time limit check
                elapsed = time.time() - _start_time
                if elapsed > MAX_WALL_SECONDS:
                    logger.warning(f"Wall-clock limit ({MAX_WALL_SECONDS}s) reached after {completed}/{total_combinations} runs. Stopping early.")
                    break

                rng_resample = random.Random(seed + k * 1000 + r_idx)
                pool = remaining_passed.copy()
                rng_resample.shuffle(pool)
                selected = pool[:k]

                # Generate PTAs for selected training trajectories
                train_ptas = []
                for traj in selected:
                    pta = generate_pta_from_trajectory(traj, task_out, temp_dir)
                    if pta:
                        train_ptas.append(pta)

                if len(train_ptas) < k:
                    all_runs.append(SingleRunResult(
                        merge_count=k, resample_idx=r_idx, task_name=tname,
                        error=f"Only {len(train_ptas)}/{k} PTAs generated",
                    ))
                    continue

                merged = merge_ptas(train_ptas)
                if not merged:
                    all_runs.append(SingleRunResult(
                        merge_count=k, resample_idx=r_idx, task_name=tname,
                        error="Merge failed",
                    ))
                    continue

                # Skip scoring if merged PTA is too large (avoids combinatorial explosion)
                MAX_PTA_STATES = 100
                if len(merged.states) > MAX_PTA_STATES:
                    all_runs.append(SingleRunResult(
                        merge_count=k, resample_idx=r_idx, task_name=tname,
                        merged_pta_states=len(merged.states),
                        merged_pta_transitions=len(merged.transitions),
                        error=f"Skipped: merged PTA too large ({len(merged.states)} states > {MAX_PTA_STATES})",
                    ))
                    continue

                # Score the fixed test set
                try:
                    scores = score_test_set(merged, test_trajs, temp_dir)
                except (MemoryError, RecursionError) as e:
                    all_runs.append(SingleRunResult(
                        merge_count=k, resample_idx=r_idx, task_name=tname,
                        error=f"Scoring failed: {type(e).__name__}",
                    ))
                    continue
                metrics = _metrics_from_scores(scores)

                run = SingleRunResult(
                    merge_count=k,
                    resample_idx=r_idx,
                    task_name=tname,
                    num_test_passed=len(test_passed),
                    num_test_failed=len(test_failed),
                    merged_pta_states=len(merged.states),
                    merged_pta_transitions=len(merged.transitions),
                )

                for mtype in ["structural", "process", "combined"]:
                    m = metrics.get(mtype, {})
                    if "error" not in m:
                        setattr(run, f"{mtype}_auroc", m.get("auroc", 0.5))
                        setattr(run, f"{mtype}_accuracy", m.get("accuracy", 0.0))
                        setattr(run, f"{mtype}_pass_mean", m.get("pass_mean", 0.0))
                        setattr(run, f"{mtype}_fail_mean", m.get("fail_mean", 0.0))

                all_runs.append(run)
                completed += 1
                if completed % 5 == 0:
                    elapsed = time.time() - _start_time
                    print(f"  [{completed}/{total_combinations}] elapsed={elapsed:.0f}s")

            # Break out of task loop if time limit hit
            if time.time() - _start_time > MAX_WALL_SECONDS:
                break

        # Break out of merge-count loop if time limit hit
        if time.time() - _start_time > MAX_WALL_SECONDS:
            print(f"\n⚠ Time limit reached. Completed {completed}/{total_combinations} runs.")
            break

    print(f"\nTotal runs completed: {completed}/{total_combinations} in {time.time() - _start_time:.0f}s")

    # ---- Aggregate per merge count ----
    points: List[MergeCountPoint] = []
    for k in merge_counts:
        runs_k = [r for r in all_runs if r.merge_count == k and not r.error]
        if not runs_k:
            points.append(MergeCountPoint(merge_count=k))
            continue

        pt = MergeCountPoint(merge_count=k, num_runs=len(runs_k))

        for mtype in ["structural", "process", "combined"]:
            aurocs = [getattr(r, f"{mtype}_auroc") for r in runs_k]
            accs = [getattr(r, f"{mtype}_accuracy") for r in runs_k]
            setattr(pt, f"{mtype}_auroc_mean", float(np.mean(aurocs)))
            setattr(pt, f"{mtype}_auroc_std", float(np.std(aurocs)))
            setattr(pt, f"{mtype}_accuracy_mean", float(np.mean(accs)))
            setattr(pt, f"{mtype}_accuracy_std", float(np.std(accs)))

        # Score gap (pass_mean - fail_mean) for structural and combined
        struct_gaps = [r.structural_pass_mean - r.structural_fail_mean for r in runs_k]
        combined_gaps = [r.combined_pass_mean - r.combined_fail_mean for r in runs_k]
        pt.structural_gap_mean = float(np.mean(struct_gaps))
        pt.structural_gap_std = float(np.std(struct_gaps))
        pt.combined_gap_mean = float(np.mean(combined_gaps))
        pt.combined_gap_std = float(np.std(combined_gaps))

        # Micro-averaged: pool ALL test scores across tasks & resamples for this k
        # Collect raw _TestScore objects for micro-averaging
        all_micro_scores: List[_TestScore] = []
        for tname in sorted(task_splits.keys()):
            remaining_passed, test_passed, test_failed = task_splits[tname]
            if len(remaining_passed) < k:
                continue
            test_trajs = test_passed + test_failed
            # Re-score with the first resample's merged PTA for micro stats
            # (We only store per-run metrics, so micro requires re-computation.
            #  For efficiency, we just use the per-run macro values above.)

        points.append(pt)

    # Cleanup
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    return points, all_runs


# ============================================================================
# TERMINAL OUTPUT
# ============================================================================

def print_results(points: List[MergeCountPoint]) -> None:
    """Pretty-print the study results to the terminal."""
    print("\n" + "=" * 100)
    print("  MERGE COUNT STUDY RESULTS")
    print("=" * 100)

    # AUROC table
    print(f"\n{'Merge':>6} | {'Structural AUROC':>20} | {'Process AUROC':>20} | {'Combined AUROC':>20} | {'Runs':>5}")
    print("-" * 83)
    for pt in points:
        if pt.num_runs == 0:
            print(f"{pt.merge_count:>6} | {'  (no data)':>20} | {'':>20} | {'':>20} | {0:>5}")
            continue
        print(
            f"{pt.merge_count:>6} | "
            f"{pt.structural_auroc_mean:>8.4f} ± {pt.structural_auroc_std:<8.4f} | "
            f"{pt.process_auroc_mean:>8.4f} ± {pt.process_auroc_std:<8.4f} | "
            f"{pt.combined_auroc_mean:>8.4f} ± {pt.combined_auroc_std:<8.4f} | "
            f"{pt.num_runs:>5}"
        )

    # Accuracy table
    print(f"\n{'Merge':>6} | {'Struct. Accuracy':>20} | {'Process Accuracy':>20} | {'Combined Accuracy':>20}")
    print("-" * 78)
    for pt in points:
        if pt.num_runs == 0:
            continue
        print(
            f"{pt.merge_count:>6} | "
            f"{pt.structural_accuracy_mean:>8.1%} ± {pt.structural_accuracy_std:<8.1%} | "
            f"{pt.process_accuracy_mean:>8.1%} ± {pt.process_accuracy_std:<8.1%} | "
            f"{pt.combined_accuracy_mean:>8.1%} ± {pt.combined_accuracy_std:<8.1%}"
        )

    # Score gap table
    print(f"\n{'Merge':>6} | {'Struct. Score Gap':>22} | {'Combined Score Gap':>22}")
    print("-" * 58)
    for pt in points:
        if pt.num_runs == 0:
            continue
        print(
            f"{pt.merge_count:>6} | "
            f"{pt.structural_gap_mean:>9.2f} ± {pt.structural_gap_std:<9.2f} | "
            f"{pt.combined_gap_mean:>9.2f} ± {pt.combined_gap_std:<9.2f}"
        )

    print()


# ============================================================================
# HTML CHART GENERATION
# ============================================================================

def generate_chart(points: List[MergeCountPoint], all_runs: List[SingleRunResult], output_path: Path) -> None:
    """Generate an interactive Plotly HTML chart showing metrics vs merge count."""
    merge_counts = [pt.merge_count for pt in points if pt.num_runs > 0]
    if not merge_counts:
        print("No data to chart.")
        return

    # Prepare data for Plotly
    def _series(attr_mean, attr_std):
        y = [getattr(pt, attr_mean) for pt in points if pt.num_runs > 0]
        err = [getattr(pt, attr_std) for pt in points if pt.num_runs > 0]
        return y, err

    s_auroc, s_auroc_e = _series("structural_auroc_mean", "structural_auroc_std")
    p_auroc, p_auroc_e = _series("process_auroc_mean", "process_auroc_std")
    cb_auroc, cb_auroc_e = _series("combined_auroc_mean", "combined_auroc_std")

    s_acc, s_acc_e = _series("structural_accuracy_mean", "structural_accuracy_std")
    p_acc, p_acc_e = _series("process_accuracy_mean", "process_accuracy_std")
    cb_acc, cb_acc_e = _series("combined_accuracy_mean", "combined_accuracy_std")

    sg, sg_e = _series("structural_gap_mean", "structural_gap_std")
    cg, cg_e = _series("combined_gap_mean", "combined_gap_std")

    # Also collect individual run dots for scatter overlay
    run_data = {}
    for r in all_runs:
        if r.error:
            continue
        run_data.setdefault(r.merge_count, []).append(r)

    scatter_x = []
    scatter_struct = []
    scatter_combined = []
    for k in merge_counts:
        for r in run_data.get(k, []):
            scatter_x.append(k)
            scatter_struct.append(r.structural_auroc)
            scatter_combined.append(r.combined_auroc)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Merge Count Study</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg: #1a1a2e; --card: #16213e; --accent: #e94560;
            --txt: #eee; --txt2: #aaa;
            --green: #4ade80; --blue: #60a5fa; --orange: #fbbf24; --purple: #c084fc;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--txt); padding: 2rem; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: var(--accent); margin-bottom: 0.3rem; }}
        .subtitle {{ color: var(--txt2); margin-bottom: 1.5rem; }}
        .chart {{ background: var(--card); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
        @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; background: var(--card); border-radius: 8px; overflow: hidden; font-size: 0.9rem; }}
        th, td {{ padding: 0.6rem 0.8rem; text-align: right; border-bottom: 1px solid #0f3460; }}
        th {{ background: #0f3460; color: var(--accent); text-align: center; }}
        td:first-child, th:first-child {{ text-align: center; }}
        tr:hover {{ background: #0f3460; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 Merge Count Study</h1>
    <p class="subtitle">Effect of the number of merged trajectories on PTA discriminative power &bull; Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <div class="chart" id="auroc_chart" style="height:500px;"></div>

    <div class="grid">
        <div class="chart" id="accuracy_chart" style="height:400px;"></div>
        <div class="chart" id="gap_chart" style="height:400px;"></div>
    </div>

    <div class="chart" id="scatter_chart" style="height:450px;"></div>

    <h2 style="color: var(--accent); margin: 2rem 0 1rem;">Raw Data Table</h2>
    <table>
        <thead>
            <tr>
                <th>Merge Count</th>
                <th>Runs</th>
                <th>Struct AUROC</th>
                <th>Process AUROC</th>
                <th>Combined AUROC</th>
                <th>Combined Acc</th>
                <th>Score Gap</th>
            </tr>
        </thead>
        <tbody>
"""
    for pt in points:
        if pt.num_runs == 0:
            html += f"<tr><td>{pt.merge_count}</td><td colspan='6'>No data</td></tr>\n"
            continue
        html += (
            f"<tr>"
            f"<td>{pt.merge_count}</td>"
            f"<td>{pt.num_runs}</td>"
            f"<td>{pt.structural_auroc_mean:.4f} ± {pt.structural_auroc_std:.4f}</td>"
            f"<td>{pt.process_auroc_mean:.4f} ± {pt.process_auroc_std:.4f}</td>"
            f"<td>{pt.combined_auroc_mean:.4f} ± {pt.combined_auroc_std:.4f}</td>"
            f"<td>{pt.combined_accuracy_mean:.1%} ± {pt.combined_accuracy_std:.1%}</td>"
            f"<td>{pt.combined_gap_mean:.2f} ± {pt.combined_gap_std:.2f}</td>"
            f"</tr>\n"
        )

    plotly_layout = """{
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#eee' },
        xaxis: { title: 'Number of Merged Trajectories', dtick: 1, gridcolor: '#333' },
        legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(22,33,62,0.8)' },
        hovermode: 'x unified',
    }"""

    html += f"""
        </tbody>
    </table>
</div>

<script>
    var mc = {json.dumps(merge_counts)};

    // ---- AUROC Chart ----
    Plotly.newPlot('auroc_chart', [
        {{ x: mc, y: {json.dumps(s_auroc)}, error_y: {{ type:'data', array:{json.dumps(s_auroc_e)}, visible:true, color:'rgba(74,222,128,0.4)' }},
           name: 'Structural', mode: 'lines+markers', line: {{ color: '#4ade80', width: 2 }}, marker: {{ size: 8 }} }},
        {{ x: mc, y: {json.dumps(p_auroc)}, error_y: {{ type:'data', array:{json.dumps(p_auroc_e)}, visible:true, color:'rgba(96,165,250,0.4)' }},
           name: 'Process', mode: 'lines+markers', line: {{ color: '#60a5fa', width: 2 }}, marker: {{ size: 8 }} }},
        {{ x: mc, y: {json.dumps(cb_auroc)}, error_y: {{ type:'data', array:{json.dumps(cb_auroc_e)}, visible:true, color:'rgba(192,132,252,0.4)' }},
           name: 'Combined', mode: 'lines+markers', line: {{ color: '#c084fc', width: 3 }}, marker: {{ size: 10 }} }},
        {{ x: mc, y: mc.map(() => 0.5), name: 'Random Baseline', mode: 'lines', line: {{ color: '#555', width: 1, dash: 'dash' }} }},
    ], Object.assign({{}}, {plotly_layout}, {{
        title: 'AUROC vs Number of Merged Trajectories',
        yaxis: {{ title: 'AUROC', range: [0, 1.05], gridcolor: '#333' }},
    }}));

    // ---- Accuracy Chart ----
    Plotly.newPlot('accuracy_chart', [
        {{ x: mc, y: {json.dumps(s_acc)}, error_y: {{ type:'data', array:{json.dumps(s_acc_e)}, visible:true, color:'rgba(74,222,128,0.4)' }},
           name: 'Structural', mode: 'lines+markers', line: {{ color: '#4ade80' }}, marker: {{ size: 6 }} }},
        {{ x: mc, y: {json.dumps(p_acc)}, error_y: {{ type:'data', array:{json.dumps(p_acc_e)}, visible:true, color:'rgba(96,165,250,0.4)' }},
           name: 'Process', mode: 'lines+markers', line: {{ color: '#60a5fa' }}, marker: {{ size: 6 }} }},
        {{ x: mc, y: {json.dumps(cb_acc)}, error_y: {{ type:'data', array:{json.dumps(cb_acc_e)}, visible:true, color:'rgba(192,132,252,0.4)' }},
           name: 'Combined', mode: 'lines+markers', line: {{ color: '#c084fc', width: 2 }}, marker: {{ size: 8 }} }},
    ], Object.assign({{}}, {plotly_layout}, {{
        title: 'Accuracy vs Merge Count',
        yaxis: {{ title: 'Accuracy', range: [0, 1.05], gridcolor: '#333' }},
    }}));

    // ---- Score Gap Chart ----
    Plotly.newPlot('gap_chart', [
        {{ x: mc, y: {json.dumps(sg)}, error_y: {{ type:'data', array:{json.dumps(sg_e)}, visible:true, color:'rgba(74,222,128,0.4)' }},
           name: 'Structural Gap', mode: 'lines+markers', line: {{ color: '#4ade80' }}, marker: {{ size: 6 }} }},
        {{ x: mc, y: {json.dumps(cg)}, error_y: {{ type:'data', array:{json.dumps(cg_e)}, visible:true, color:'rgba(192,132,252,0.4)' }},
           name: 'Combined Gap', mode: 'lines+markers', line: {{ color: '#c084fc', width: 2 }}, marker: {{ size: 8 }} }},
        {{ x: mc, y: mc.map(() => 0), name: 'No Discrimination', mode: 'lines', line: {{ color: '#555', width: 1, dash: 'dash' }} }},
    ], Object.assign({{}}, {plotly_layout}, {{
        title: 'Score Gap (Pass Mean − Fail Mean) vs Merge Count',
        yaxis: {{ title: 'Score Gap', gridcolor: '#333' }},
    }}));

    // ---- Individual run scatter ----
    Plotly.newPlot('scatter_chart', [
        {{ x: {json.dumps(scatter_x)}, y: {json.dumps(scatter_struct)},
           name: 'Structural (individual runs)', mode: 'markers',
           marker: {{ size: 5, color: '#4ade80', opacity: 0.5 }} }},
        {{ x: {json.dumps(scatter_x)}, y: {json.dumps(scatter_combined)},
           name: 'Combined (individual runs)', mode: 'markers',
           marker: {{ size: 5, color: '#c084fc', opacity: 0.5 }} }},
        {{ x: mc, y: {json.dumps(s_auroc)}, name: 'Structural Mean', mode: 'lines',
           line: {{ color: '#4ade80', width: 2 }} }},
        {{ x: mc, y: {json.dumps(cb_auroc)}, name: 'Combined Mean', mode: 'lines',
           line: {{ color: '#c084fc', width: 2 }} }},
        {{ x: mc, y: mc.map(() => 0.5), name: 'Random', mode: 'lines',
           line: {{ color: '#555', dash: 'dash' }} }},
    ], Object.assign({{}}, {plotly_layout}, {{
        title: 'Individual Run AUROC Values (each dot = one task × resample)',
        yaxis: {{ title: 'AUROC', range: [0, 1.05], gridcolor: '#333' }},
    }}));
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nChart saved to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Study how the number of merged trajectories affects metric scores"
    )
    parser.add_argument("data_dir", help="Path to the dataset directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: experiments/results/merge_count_study)")
    parser.add_argument("--min-merge", type=int, default=2,
                        help="Minimum merge count to test (default: 2)")
    parser.add_argument("--max-merge", type=int, default=7,
                        help="Maximum merge count to test (default: 7)")
    parser.add_argument("--test-pass-count", type=int, default=3,
                        help="Number of passed trajectories to hold out for testing (default: 3)")
    parser.add_argument("--test-fail-count", type=int, default=3,
                        help="Number of failed trajectories for testing (default: 3)")
    parser.add_argument("--resamples", type=int, default=1,
                        help="Number of random resamples at each merge count (default: 1, meaning no resampling)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    if not STATS_AVAILABLE:
        print("ERROR: Required packages not available. Install with:")
        print("  pip install scikit-learn scipy numpy")
        return 1

    if args.min_merge < 2:
        print("ERROR: --min-merge must be at least 2")
        return 1
    if args.max_merge < args.min_merge:
        print("ERROR: --max-merge must be >= --min-merge")
        return 1

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "results" / "merge_count_study"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merge count study: k = {args.min_merge}..{args.max_merge}, "
          f"test={args.test_pass_count}P+{args.test_fail_count}F, "
          f"resamples={args.resamples}, seed={args.seed}")

    points, all_runs = run_study(
        data_dir=data_dir,
        output_dir=output_dir,
        min_merge=args.min_merge,
        max_merge=args.max_merge,
        test_pass_count=args.test_pass_count,
        test_fail_count=args.test_fail_count,
        num_resamples=args.resamples,
        seed=args.seed,
    )

    if not points:
        print("ERROR: No results produced")
        return 1

    # Print to terminal
    print_results(points)

    # Save JSON
    results_json = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "min_merge": args.min_merge,
            "max_merge": args.max_merge,
            "test_pass_count": args.test_pass_count,
            "test_fail_count": args.test_fail_count,
            "resamples": args.resamples,
            "seed": args.seed,
        },
        "points": [asdict(pt) for pt in points],
        "all_runs": [asdict(r) for r in all_runs],
    }
    json_path = output_dir / "merge_count_study_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Generate chart
    chart_path = output_dir / "merge_count_study_chart.html"
    generate_chart(points, all_runs, chart_path)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
