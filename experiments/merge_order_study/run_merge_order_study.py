#!/usr/bin/env python3
"""
Merge Order Study — How do different subsets and orderings of merged trajectories
affect the final matching scores?

Building on the merge-count study (which varies k), this experiment fixes k and
investigates two sources of variance:

  1. **Combination variance**: different subsets (combinations) of size k drawn
     from the pool of passing trajectories.
  2. **Permutation variance**: different sequential merge orderings
     (permutations) of the same subset.

This is designed as a **case-study** experiment: you point it at a single
task directory containing pass/fail trajectory folders (or zips).

Experimental design:
  1. Fix a test set (test_pass_count passed + test_fail_count failed).
     The test set is chosen ONCE (with a fixed seed) and stays identical
     across all runs.
  2. From the remaining passed trajectories, enumerate (or sample) all
     combinations of size k.
  3. For each combination, enumerate (or sample) different orderings
     (permutations).
  4. Merge PTAs *sequentially* in the given ordering and score the fixed test
     set.
  5. Aggregate results — quantify within-combination (ordering) variance vs
     between-combination (subset) variance.
  6. Print a summary table and generate an interactive Plotly HTML report.

Usage:
    python experiments/run_merge_order_study.py <task_dir> --k 4
    python experiments/run_merge_order_study.py data/evaluation platform-vscbench/python_refactor --k 3
    python experiments/run_merge_order_study.py <task_dir> --k 4 --max-combinations 10 --max-permutations 6
"""

import sys
import os
import json
import argparse
import logging
import random
import re
import shutil
import tempfile
import time
import itertools
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

# ---------------------------------------------------------------------------
# SDK & project imports
# ---------------------------------------------------------------------------

from swe_trace_sdk import trace as trace_api, match
from swe_trace_sdk.models import Trace

# Reuse helpers from the holdout experiment & shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "holdout"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_holdout_experiment import (
    TrajectoryInfo,
    discover_trajectories,
    find_trajectory_file,
    get_trajectory_json_path,
    generate_pta_from_trajectory,
)
from metrics_utils import (
    STATS_AVAILABLE,
    TQDM_AVAILABLE,
    tqdm,
    compute_classification_metrics,
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
# SINGLE-TASK DISCOVERY
# ============================================================================

def discover_single_task(task_dir: Path) -> Dict[str, List[TrajectoryInfo]]:
    """Discover trajectories inside a *single* task directory.

    Unlike ``discover_trajectories`` (which expects a parent directory
    containing multiple task sub-folders), this function treats
    *task_dir* itself as the task and looks for trajectory items
    (folders or zips with ``-logs-`` in the name) directly inside it.

    Returns
    -------
    dict
        ``{task_name: [TrajectoryInfo, ...]}``.  Empty dict if no
        trajectories were found.
    """
    task_name = task_dir.name
    trajectories: List[TrajectoryInfo] = []

    for item in task_dir.iterdir():
        if item.name.startswith(".") or item.name.startswith("_"):
            continue

        name = item.stem if item.suffix == ".zip" else item.name

        if "-logs-" not in name:
            continue

        parts = name.split("-logs-")
        if len(parts) != 2:
            continue

        rest = parts[1]
        m = re.match(r"^(.+)-(pass|fail)(?:-\d+)?$", rest)
        if not m:
            continue
        model_name = m.group(1)
        passed = m.group(2) == "pass"

        if item.is_dir():
            traj_file = find_trajectory_file(item)
            if not traj_file:
                continue
            traj_path = traj_file
        elif item.suffix == ".zip":
            traj_path = str(item)
        else:
            continue

        trajectories.append(TrajectoryInfo(
            trajectory_id=name,
            task_name=task_name,
            model_name=model_name,
            passed=passed,
            trajectory_path=traj_path,
        ))

    if trajectories:
        return {task_name: trajectories}
    return {}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SingleOrderRunResult:
    """Scores from one combination × permutation run against the fixed test set."""
    task_name: str
    k: int
    combination_idx: int
    permutation_idx: int
    trajectory_ids: List[str]           # ordered list of trajectory IDs used
    combination_key: str                # sorted, comma-joined IDs (identifies subset)

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
class CombinationSummary:
    """Aggregated metrics for a single combination (across its permutations)."""
    task_name: str
    combination_idx: int
    combination_key: str
    num_permutations: int = 0

    structural_auroc_mean: float = 0.0
    structural_auroc_std: float = 0.0
    combined_auroc_mean: float = 0.0
    combined_auroc_std: float = 0.0
    structural_accuracy_mean: float = 0.0
    combined_accuracy_mean: float = 0.0


@dataclass
class TaskSummary:
    """Aggregated metrics for a single task across all combinations & orderings."""
    task_name: str
    k: int
    num_combinations: int = 0
    total_runs: int = 0

    # Between-combination variance (mean of combination means ± std)
    between_structural_auroc_mean: float = 0.0
    between_structural_auroc_std: float = 0.0
    between_combined_auroc_mean: float = 0.0
    between_combined_auroc_std: float = 0.0

    # Within-combination variance (mean of per-combination stds)
    within_structural_auroc_std_mean: float = 0.0
    within_combined_auroc_std_mean: float = 0.0

    # Overall (across all runs)
    overall_structural_auroc_mean: float = 0.0
    overall_structural_auroc_std: float = 0.0
    overall_combined_auroc_mean: float = 0.0
    overall_combined_auroc_std: float = 0.0


# ============================================================================
# SEQUENTIAL MERGE — merge traces one-by-one in the given order
# ============================================================================

def merge_sequential(traces: List[Trace], use_llm: bool = False) -> Optional[Trace]:
    """Merge traces sequentially in the order given.

    This wraps ``trace_api.merge`` while preserving the caller's ordering,
    since the SDK merger processes traces in list order.
    """
    if not traces:
        return None
    if len(traces) == 1:
        return traces[0]
    try:
        return trace_api.merge(traces, use_llm=use_llm)
    except Exception as e:
        logger.error(f"Sequential merge failed: {e}")
        return None


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
_test_trace_cache: Dict[str, Trace] = {}


def score_test_set(
    merged_pta: Trace,
    test_trajectories: List[TrajectoryInfo],
    temp_dir: Path,
) -> List[_TestScore]:
    """Score held-out test trajectories against a merged PTA using the SDK."""
    required_files = match.extract_required_files(merged_pta)

    results: List[_TestScore] = []
    for traj_info in test_trajectories:
        try:
            tid = traj_info.trajectory_id
            # Use in-memory cache to avoid re-loading the same trace
            if tid in _test_trace_cache:
                traj_trace = _test_trace_cache[tid]
            elif traj_info.pta_path and Path(traj_info.pta_path).exists():
                traj_trace = trace_api.load(traj_info.pta_path, format="trace")
                _test_trace_cache[tid] = traj_trace
            else:
                json_path = get_trajectory_json_path(traj_info, temp_dir)
                if not json_path:
                    raise ValueError(f"No JSON for {tid}")
                traj_trace = trace_api.load(json_path, format="chatlog")
                _test_trace_cache[tid] = traj_trace

            # Structural coverage via SDK matcher (F1 score)
            match_result = match.run(traj_trace, merged_pta, use_llm=False)
            best_f1 = match_result.metrics.f1_score

            # File-level process coverage
            file_cov = 100.0
            if required_files:
                file_ratio, _ = match.check_file_coverage(traj_trace, required_files)
                file_cov = file_ratio * 100.0

            results.append(_TestScore(
                is_passed=traj_info.passed,
                structural_coverage=round(best_f1, 2),
                process_coverage=round(file_cov, 2),
            ))
        except Exception as e:
            logger.error(f"Error scoring {traj_info.trajectory_id}: {e}")
            results.append(_TestScore(is_passed=traj_info.passed, error=str(e)))

    return results


def _metrics_from_scores(scores: List[_TestScore]) -> dict:
    """Compute AUROC / accuracy for structural, process, and combined."""
    if not STATS_AVAILABLE or len(scores) < 2:
        return {}

    struct = compute_classification_metrics(scores, "structural_coverage", passed_field="is_passed")
    proc = compute_classification_metrics(scores, "process_coverage", passed_field="is_passed")

    # Combined (average of structural + process)
    @dataclass
    class _Tmp:
        is_passed: bool
        structural_coverage: float
        error: str = ""

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
    tasks: Dict[str, List[TrajectoryInfo]],
    output_dir: Path,
    k: int,
    test_pass_count: int,
    test_fail_count: int,
    max_combinations: int,
    max_permutations: int,
    seed: int,
) -> Tuple[List[TaskSummary], List[SingleOrderRunResult]]:
    """Run the full merge-order study.

    Parameters
    ----------
    tasks : dict
        ``{task_name: [TrajectoryInfo, ...]}``.  Typically produced by
        :func:`discover_single_task`.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="pta_mo_"))

    print(f"Tasks provided: {list(tasks.keys())}")

    # Need at least k + test_pass_count passed and test_fail_count failed
    min_passed = k + test_pass_count
    eligible: Dict[str, List[TrajectoryInfo]] = {}
    for tname, trajs in tasks.items():
        passed = [t for t in trajs if t.passed]
        failed = [t for t in trajs if not t.passed]
        if len(passed) >= min_passed and len(failed) >= test_fail_count:
            eligible[tname] = trajs

    print(f"Eligible tasks (>= {min_passed} passed, >= {test_fail_count} failed): {len(eligible)}")
    if not eligible:
        print("ERROR: No eligible tasks. Try reducing --k, --test-pass-count, or --test-fail-count.")
        return [], []

    # ---- Fix test set per task (identical to merge-count study) ----
    rng_split = random.Random(seed)
    task_splits: Dict[str, Tuple[List[TrajectoryInfo], List[TrajectoryInfo], List[TrajectoryInfo]]] = {}

    for tname, trajs in eligible.items():
        passed = [t for t in trajs if t.passed]
        failed = [t for t in trajs if not t.passed]

        shuffled_passed = passed.copy()
        rng_split.shuffle(shuffled_passed)
        test_passed = shuffled_passed[-test_pass_count:]
        remaining_passed = shuffled_passed[:-test_pass_count]

        shuffled_failed = failed.copy()
        rng_split.shuffle(shuffled_failed)
        test_failed = shuffled_failed[:test_fail_count]

        task_splits[tname] = (remaining_passed, test_passed, test_failed)

    # Pre-generate PTAs for test trajectories (done once)
    print("\nPre-generating PTAs for test trajectories...")
    for tname, (_, test_passed, test_failed) in tqdm(task_splits.items(), desc="Test PTAs"):
        task_out = output_dir / tname
        task_out.mkdir(parents=True, exist_ok=True)
        for traj in test_passed + test_failed:
            generate_pta_from_trajectory(traj, task_out, temp_dir)

    # Pre-generate PTAs for all pool trajectories
    print("Pre-generating PTAs for pool (train) trajectories...")
    for tname, (pool, _, _) in tqdm(task_splits.items(), desc="Pool PTAs"):
        task_out = output_dir / tname
        for traj in pool:
            generate_pta_from_trajectory(traj, task_out, temp_dir)

    # ---- Main loop: combinations × permutations ----
    all_runs: List[SingleOrderRunResult] = []
    task_summaries: List[TaskSummary] = []
    _start_time = time.time()
    MAX_WALL_SECONDS = 10 * 60  # 10-minute hard limit
    _time_exceeded = False

    for tname in sorted(task_splits.keys()):
        if _time_exceeded:
            break
        remaining_passed, test_passed, test_failed = task_splits[tname]
        test_trajs = test_passed + test_failed
        task_out = output_dir / tname

        pool_size = len(remaining_passed)
        total_combos = math.comb(pool_size, k)

        # Decide how many combinations to evaluate
        if total_combos <= max_combinations:
            combos = list(itertools.combinations(range(pool_size), k))
            sampled_combos = False
        else:
            rng_combo = random.Random(seed + hash(tname))
            combo_set: set = set()
            while len(combo_set) < max_combinations:
                c = tuple(sorted(rng_combo.sample(range(pool_size), k)))
                combo_set.add(c)
            combos = sorted(combo_set)
            sampled_combos = True

        print(f"\n{'='*60}")
        print(f"  Task: {tname}  |  pool={pool_size}  |  C({pool_size},{k})={total_combos}"
              f"  |  evaluating {len(combos)} combo(s)"
              f"{'  (sampled)' if sampled_combos else ''}")
        print(f"{'='*60}")

        combo_summaries: List[CombinationSummary] = []

        for c_idx, combo in enumerate(tqdm(combos, desc=f"{tname} combos")):
            if time.time() - _start_time > MAX_WALL_SECONDS:
                logger.warning(f"Wall-clock limit ({MAX_WALL_SECONDS}s) reached. Stopping early.")
                _time_exceeded = True
                break
            selected_trajs = [remaining_passed[i] for i in combo]
            combo_key = ",".join(sorted(t.trajectory_id for t in selected_trajs))

            # Load traces for this combination (order-independent)
            combo_traces: Dict[int, Trace] = {}
            load_ok = True
            for idx in combo:
                traj = remaining_passed[idx]
                tid = traj.trajectory_id
                # Use in-memory cache
                if tid in _test_trace_cache:
                    combo_traces[idx] = _test_trace_cache[tid]
                elif traj.pta_path and Path(traj.pta_path).exists():
                    try:
                        combo_traces[idx] = trace_api.load(traj.pta_path, format="trace")
                        _test_trace_cache[tid] = combo_traces[idx]
                    except Exception as e:
                        logger.error(f"Failed to load trace {tid}: {e}")
                        load_ok = False
                        break
                else:
                    json_path = get_trajectory_json_path(traj, temp_dir)
                    if not json_path:
                        load_ok = False
                        break
                    try:
                        combo_traces[idx] = trace_api.load(json_path, format="chatlog")
                        _test_trace_cache[tid] = combo_traces[idx]
                    except Exception as e:
                        logger.error(f"Failed to load trace {tid}: {e}")
                        load_ok = False
                        break

            if not load_ok or len(combo_traces) < k:
                all_runs.append(SingleOrderRunResult(
                    task_name=tname, k=k, combination_idx=c_idx,
                    permutation_idx=0, trajectory_ids=[t.trajectory_id for t in selected_trajs],
                    combination_key=combo_key,
                    error=f"Could not load all {k} traces",
                ))
                continue

            # Decide how many permutations to evaluate
            total_perms = math.factorial(k)
            if total_perms <= max_permutations:
                perms = list(itertools.permutations(combo))
            else:
                rng_perm = random.Random(seed + c_idx * 10000 + hash(tname))
                perm_set: set = set()
                # Always include the canonical (sorted) ordering
                perm_set.add(combo)
                while len(perm_set) < max_permutations:
                    p = list(combo)
                    rng_perm.shuffle(p)
                    perm_set.add(tuple(p))
                perms = sorted(perm_set)

            for p_idx, perm in enumerate(perms):
                ordered_traces = [combo_traces[i] for i in perm]
                ordered_ids = [remaining_passed[i].trajectory_id for i in perm]

                merged = merge_sequential(ordered_traces)
                if not merged:
                    all_runs.append(SingleOrderRunResult(
                        task_name=tname, k=k, combination_idx=c_idx,
                        permutation_idx=p_idx, trajectory_ids=ordered_ids,
                        combination_key=combo_key, error="Merge failed",
                    ))
                    continue

                scores = score_test_set(merged, test_trajs, temp_dir)
                metrics = _metrics_from_scores(scores)

                run = SingleOrderRunResult(
                    task_name=tname, k=k, combination_idx=c_idx,
                    permutation_idx=p_idx, trajectory_ids=ordered_ids,
                    combination_key=combo_key,
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

            # Summarise this combination
            combo_runs = [r for r in all_runs
                          if r.task_name == tname and r.combination_key == combo_key
                          and not r.error]
            cs = CombinationSummary(task_name=tname, combination_idx=c_idx,
                                    combination_key=combo_key, num_permutations=len(combo_runs))
            if combo_runs and NUMPY_AVAILABLE:
                s_aurocs = [r.structural_auroc for r in combo_runs]
                c_aurocs = [r.combined_auroc for r in combo_runs]
                cs.structural_auroc_mean = float(np.mean(s_aurocs))
                cs.structural_auroc_std = float(np.std(s_aurocs))
                cs.combined_auroc_mean = float(np.mean(c_aurocs))
                cs.combined_auroc_std = float(np.std(c_aurocs))
                cs.structural_accuracy_mean = float(np.mean([r.structural_accuracy for r in combo_runs]))
                cs.combined_accuracy_mean = float(np.mean([r.combined_accuracy for r in combo_runs]))
            combo_summaries.append(cs)

        # ---- Task-level summary ----
        task_runs = [r for r in all_runs if r.task_name == tname and not r.error]
        ts = TaskSummary(task_name=tname, k=k,
                         num_combinations=len(combos),
                         total_runs=len(task_runs))

        if task_runs and NUMPY_AVAILABLE:
            all_s = [r.structural_auroc for r in task_runs]
            all_c = [r.combined_auroc for r in task_runs]
            ts.overall_structural_auroc_mean = float(np.mean(all_s))
            ts.overall_structural_auroc_std = float(np.std(all_s))
            ts.overall_combined_auroc_mean = float(np.mean(all_c))
            ts.overall_combined_auroc_std = float(np.std(all_c))

            # Between-combination: std of combination means
            valid_cs = [c for c in combo_summaries if c.num_permutations > 0]
            if valid_cs:
                cs_s_means = [c.structural_auroc_mean for c in valid_cs]
                cs_c_means = [c.combined_auroc_mean for c in valid_cs]
                ts.between_structural_auroc_mean = float(np.mean(cs_s_means))
                ts.between_structural_auroc_std = float(np.std(cs_s_means))
                ts.between_combined_auroc_mean = float(np.mean(cs_c_means))
                ts.between_combined_auroc_std = float(np.std(cs_c_means))

                # Within-combination: mean of per-combination stds
                cs_s_stds = [c.structural_auroc_std for c in valid_cs]
                cs_c_stds = [c.combined_auroc_std for c in valid_cs]
                ts.within_structural_auroc_std_mean = float(np.mean(cs_s_stds))
                ts.within_combined_auroc_std_mean = float(np.mean(cs_c_stds))

        task_summaries.append(ts)

    # Cleanup
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    return task_summaries, all_runs


# ============================================================================
# TERMINAL OUTPUT
# ============================================================================

def print_results(task_summaries: List[TaskSummary], k: int) -> None:
    """Pretty-print the study results to the terminal."""
    print("\n" + "=" * 110)
    print(f"  MERGE ORDER STUDY RESULTS  (k = {k})")
    print("=" * 110)

    print(f"\n{'Task':<30} | {'Combos':>6} | {'Runs':>5} | "
          f"{'Overall Struct AUROC':>22} | {'Overall Comb AUROC':>22}")
    print("-" * 110)
    for ts in task_summaries:
        if ts.total_runs == 0:
            print(f"{ts.task_name:<30} | {ts.num_combinations:>6} | {0:>5} | {'(no data)':>22} | {'':>22}")
            continue
        print(
            f"{ts.task_name:<30} | {ts.num_combinations:>6} | {ts.total_runs:>5} | "
            f"{ts.overall_structural_auroc_mean:>8.4f} ± {ts.overall_structural_auroc_std:<8.4f} | "
            f"{ts.overall_combined_auroc_mean:>8.4f} ± {ts.overall_combined_auroc_std:<8.4f}"
        )

    # Variance decomposition table
    print(f"\n{'Task':<30} | {'Between-Combo std (Struct)':>25} | {'Within-Combo std (Struct)':>25} | "
          f"{'Between-Combo std (Comb)':>25} | {'Within-Combo std (Comb)':>25}")
    print("-" * 140)
    for ts in task_summaries:
        if ts.total_runs == 0:
            continue
        print(
            f"{ts.task_name:<30} | "
            f"{ts.between_structural_auroc_std:>25.4f} | "
            f"{ts.within_structural_auroc_std_mean:>25.4f} | "
            f"{ts.between_combined_auroc_std:>25.4f} | "
            f"{ts.within_combined_auroc_std_mean:>25.4f}"
        )

    print()


# ============================================================================
# HTML CHART GENERATION
# ============================================================================

def generate_chart(
    task_summaries: List[TaskSummary],
    all_runs: List[SingleOrderRunResult],
    k: int,
    output_path: Path,
) -> None:
    """Generate an interactive Plotly HTML report."""
    valid_runs = [r for r in all_runs if not r.error]
    if not valid_runs:
        print("No valid data to chart.")
        return

    # Group runs by task
    tasks_with_data = sorted({r.task_name for r in valid_runs})

    # --- Build per-combination box-plot data for each task ---
    # For the scatter: x = combination_idx, y = AUROC, color = metric type
    task_chart_blocks = []
    for tname in tasks_with_data:
        t_runs = [r for r in valid_runs if r.task_name == tname]
        combo_keys = sorted({r.combination_key for r in t_runs})
        combo_label_map = {ck: f"C{i}" for i, ck in enumerate(combo_keys)}

        combo_labels = [combo_label_map[r.combination_key] for r in t_runs]
        struct_vals = [r.structural_auroc for r in t_runs]
        combined_vals = [r.combined_auroc for r in t_runs]

        task_chart_blocks.append({
            "task": tname,
            "combo_labels": combo_labels,
            "struct_vals": struct_vals,
            "combined_vals": combined_vals,
            "num_combos": len(combo_keys),
            "num_runs": len(t_runs),
        })

    # Variance decomposition bar chart data
    bar_tasks = [ts.task_name for ts in task_summaries if ts.total_runs > 0]
    bar_between_s = [ts.between_structural_auroc_std for ts in task_summaries if ts.total_runs > 0]
    bar_within_s = [ts.within_structural_auroc_std_mean for ts in task_summaries if ts.total_runs > 0]
    bar_between_c = [ts.between_combined_auroc_std for ts in task_summaries if ts.total_runs > 0]
    bar_within_c = [ts.within_combined_auroc_std_mean for ts in task_summaries if ts.total_runs > 0]

    # HTML generation
    plotly_layout_base = """{
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#eee' },
        hovermode: 'closest',
        legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(22,33,62,0.8)' },
    }"""

    # Per-task chart divs and JS
    task_divs = ""
    task_js = ""
    for i, blk in enumerate(task_chart_blocks):
        div_id = f"task_chart_{i}"
        task_divs += f'<div class="chart" id="{div_id}" style="height:400px;"></div>\n'
        task_js += f"""
        Plotly.newPlot('{div_id}', [
            {{ x: {json.dumps(blk['combo_labels'])}, y: {json.dumps(blk['struct_vals'])},
               name: 'Structural AUROC', type: 'box', boxpoints: 'all', jitter: 0.4, pointpos: -1.5,
               marker: {{ color: '#4ade80', size: 5 }}, line: {{ color: '#4ade80' }} }},
            {{ x: {json.dumps(blk['combo_labels'])}, y: {json.dumps(blk['combined_vals'])},
               name: 'Combined AUROC', type: 'box', boxpoints: 'all', jitter: 0.4, pointpos: 1.5,
               marker: {{ color: '#c084fc', size: 5 }}, line: {{ color: '#c084fc' }} }},
        ], Object.assign({{}}, {plotly_layout_base}, {{
            title: '{blk["task"]}  ({blk["num_combos"]} combos, {blk["num_runs"]} runs)',
            xaxis: {{ title: 'Combination', gridcolor: '#333' }},
            yaxis: {{ title: 'AUROC', range: [0, 1.05], gridcolor: '#333' }},
            boxmode: 'group',
        }}));
        """

    # Raw data table
    raw_rows = ""
    for ts in task_summaries:
        if ts.total_runs == 0:
            raw_rows += f"<tr><td>{ts.task_name}</td><td colspan='6'>No data</td></tr>\n"
            continue
        raw_rows += (
            f"<tr>"
            f"<td>{ts.task_name}</td>"
            f"<td>{ts.num_combinations}</td>"
            f"<td>{ts.total_runs}</td>"
            f"<td>{ts.overall_structural_auroc_mean:.4f} ± {ts.overall_structural_auroc_std:.4f}</td>"
            f"<td>{ts.overall_combined_auroc_mean:.4f} ± {ts.overall_combined_auroc_std:.4f}</td>"
            f"<td>{ts.between_structural_auroc_std:.4f} / {ts.within_structural_auroc_std_mean:.4f}</td>"
            f"<td>{ts.between_combined_auroc_std:.4f} / {ts.within_combined_auroc_std_mean:.4f}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Merge Order Study (k={k})</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg: #1a1a2e; --card: #16213e; --accent: #e94560;
            --txt: #eee; --txt2: #aaa;
            --green: #4ade80; --blue: #60a5fa; --orange: #fbbf24; --purple: #c084fc;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--txt); padding: 2rem; }}
        .container {{ max-width: 1500px; margin: 0 auto; }}
        h1 {{ color: var(--accent); margin-bottom: 0.3rem; }}
        h2 {{ color: var(--accent); margin: 2rem 0 1rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.4rem; }}
        .subtitle {{ color: var(--txt2); margin-bottom: 1.5rem; }}
        .chart {{ background: var(--card); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
        @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; background: var(--card); border-radius: 8px; overflow: hidden; font-size: 0.9rem; }}
        th, td {{ padding: 0.6rem 0.8rem; text-align: right; border-bottom: 1px solid #0f3460; }}
        th {{ background: #0f3460; color: var(--accent); text-align: center; }}
        td:first-child, th:first-child {{ text-align: left; }}
        tr:hover {{ background: #0f3460; }}
        .insight {{ background: var(--card); border-left: 4px solid var(--accent); padding: 1rem 1.5rem; margin: 1rem 0; border-radius: 0 8px 8px 0; }}
        .insight strong {{ color: var(--orange); }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔀 Merge Order Study</h1>
    <p class="subtitle">
        Effect of subset choice and merge ordering on PTA scores &bull; k = {k}
        &bull; Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </p>

    <div class="insight">
        <strong>What this measures:</strong> For a fixed number of merged trajectories (k={k}),
        we vary <em>which</em> trajectories are chosen (combinations) and <em>in what order</em>
        they are merged (permutations).  <strong>Between-combination σ</strong> captures how
        much the subset choice matters; <strong>Within-combination σ</strong> captures how
        much ordering matters for a given subset.
    </div>

    <h2>Variance Decomposition</h2>
    <div class="grid">
        <div class="chart" id="var_structural" style="height:400px;"></div>
        <div class="chart" id="var_combined" style="height:400px;"></div>
    </div>

    <h2>Per-Task: AUROC by Combination (box = permutations)</h2>
    {task_divs}

    <h2>Summary Table</h2>
    <table>
        <thead>
            <tr>
                <th>Task</th>
                <th>Combos</th>
                <th>Runs</th>
                <th>Structural AUROC</th>
                <th>Combined AUROC</th>
                <th>Struct σ (between / within)</th>
                <th>Comb σ (between / within)</th>
            </tr>
        </thead>
        <tbody>
            {raw_rows}
        </tbody>
    </table>
</div>

<script>
    var layout = {plotly_layout_base};

    // ---- Variance decomposition (Structural) ----
    Plotly.newPlot('var_structural', [
        {{ x: {json.dumps(bar_tasks)}, y: {json.dumps(bar_between_s)},
           name: 'Between-Combo σ', type: 'bar', marker: {{ color: '#60a5fa' }} }},
        {{ x: {json.dumps(bar_tasks)}, y: {json.dumps(bar_within_s)},
           name: 'Within-Combo σ (ordering)', type: 'bar', marker: {{ color: '#fbbf24' }} }},
    ], Object.assign({{}}, layout, {{
        title: 'Structural AUROC — Variance Decomposition',
        barmode: 'group',
        xaxis: {{ gridcolor: '#333' }},
        yaxis: {{ title: 'Std Dev of AUROC', gridcolor: '#333' }},
    }}));

    // ---- Variance decomposition (Combined) ----
    Plotly.newPlot('var_combined', [
        {{ x: {json.dumps(bar_tasks)}, y: {json.dumps(bar_between_c)},
           name: 'Between-Combo σ', type: 'bar', marker: {{ color: '#60a5fa' }} }},
        {{ x: {json.dumps(bar_tasks)}, y: {json.dumps(bar_within_c)},
           name: 'Within-Combo σ (ordering)', type: 'bar', marker: {{ color: '#fbbf24' }} }},
    ], Object.assign({{}}, layout, {{
        title: 'Combined AUROC — Variance Decomposition',
        barmode: 'group',
        xaxis: {{ gridcolor: '#333' }},
        yaxis: {{ title: 'Std Dev of AUROC', gridcolor: '#333' }},
    }}));

    // ---- Per-task box plots ----
    {task_js}
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
        description="Case-study: how subset choice and merge ordering affect PTA matching scores"
    )
    parser.add_argument("task_dir",
                        help="Path to a single task directory containing trajectory "
                             "folders/zips (e.g. data/evaluation platform-vscbench/python_refactor)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: experiments/results/merge_order_study)")
    parser.add_argument("--k", type=int, default=4,
                        help="Fixed merge count (default: 4)")
    parser.add_argument("--test-pass-count", type=int, default=3,
                        help="Number of passed trajectories held out for testing (default: 3)")
    parser.add_argument("--test-fail-count", type=int, default=3,
                        help="Number of failed trajectories for testing (default: 3)")
    parser.add_argument("--max-combinations", type=int, default=10,
                        help="Max number of subset combinations to evaluate per task "
                             "(default: 10; use 0 for all)")
    parser.add_argument("--max-permutations", type=int, default=6,
                        help="Max number of orderings per combination "
                             "(default: 6; use 0 for all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    if not STATS_AVAILABLE:
        print("ERROR: Required packages not available. Install with:")
        print("  pip install scikit-learn scipy numpy")
        return 1

    if args.k < 2:
        print("ERROR: --k must be at least 2")
        return 1

    task_dir = Path(args.task_dir)
    if not task_dir.exists():
        print(f"ERROR: Task directory not found: {task_dir}")
        return 1

    # Try multi-task discovery first (OpenHands-style), fall back to single task
    tasks = discover_trajectories(task_dir)
    if not tasks:
        tasks = discover_single_task(task_dir)
    if not tasks:
        print(f"ERROR: No trajectories found in {task_dir}.")
        print("  Expected folders/zips matching <task>-logs-<model>-pass|fail")
        return 1

    for tname, trajs in tasks.items():
        n_pass = sum(1 for t in trajs if t.passed)
        n_fail = sum(1 for t in trajs if not t.passed)
        print(f"Task '{tname}': {n_pass} passed, {n_fail} failed trajectories")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "results" / "merge_order_study"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Treat 0 as "unlimited"
    max_combos = args.max_combinations if args.max_combinations > 0 else 10**9
    max_perms = args.max_permutations if args.max_permutations > 0 else 10**9

    print(f"Merge order study: k={args.k}, "
          f"test={args.test_pass_count}P+{args.test_fail_count}F, "
          f"max_combos={args.max_combinations}, max_perms={args.max_permutations}, "
          f"seed={args.seed}")

    task_summaries, all_runs = run_study(
        tasks=tasks,
        output_dir=output_dir,
        k=args.k,
        test_pass_count=args.test_pass_count,
        test_fail_count=args.test_fail_count,
        max_combinations=max_combos,
        max_permutations=max_perms,
        seed=args.seed,
    )

    if not task_summaries:
        print("ERROR: No results produced")
        return 1

    # Print to terminal
    print_results(task_summaries, args.k)

    # Save JSON
    results_json = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "k": args.k,
            "test_pass_count": args.test_pass_count,
            "test_fail_count": args.test_fail_count,
            "max_combinations": args.max_combinations,
            "max_permutations": args.max_permutations,
            "seed": args.seed,
        },
        "task_summaries": [asdict(ts) for ts in task_summaries],
        "all_runs": [asdict(r) for r in all_runs],
    }
    json_path = output_dir / "merge_order_study_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Generate chart
    chart_path = output_dir / "merge_order_study_chart.html"
    generate_chart(task_summaries, all_runs, args.k, chart_path)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
