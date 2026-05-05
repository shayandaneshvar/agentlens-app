#!/usr/bin/env python3
"""
Baseline Experiment: Individual Matching vs Merged PTA Matching.

Compares two scoring approaches:
  1. Individual Matching (baseline): Match candidate against each training
     trace individually, then average the scores.
  2. Merged PTA Matching (proposed): Merge training traces into one ground-
     truth PTA, then match candidate against the merged PTA.

Uses the same holdout split (train/test) for both approaches so results
are directly comparable.

Usage:
    python run_baseline_experiment.py <data_dir>
    python run_baseline_experiment.py <data_dir> --merge-count 3 --test-pass-count 1 --test-fail-count 1
    python run_baseline_experiment.py <data_dir> --output-dir <output_dir> -v
"""

import sys
import os
import json
import argparse
import logging
import random
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from swe_trace_sdk import trace as trace_api, match
from swe_trace_sdk.models import Trace
from swe_trace_sdk.match import extract_required_tools, check_process_coverage, extract_required_files, check_file_coverage

# Import shared experiment infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metrics_utils import (
    STATS_AVAILABLE,
    tqdm,
    MetricSet,
    TrajectoryScore,
    compute_classification_metrics,
    compute_verdict,
    fill_metric_set,
    get_html_styles,
    generate_confusion_matrix_html,
    generate_histogram_js,
    format_auroc_color,
)

# Import trajectory discovery / loading utilities from holdout experiment
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "holdout"))
from run_holdout_experiment import (
    TrajectoryInfo,
    discover_trajectories,
    find_trajectory_file,
    get_trajectory_json_path,
    split_trajectories,
    generate_pta_from_trajectory,
    merge_ptas,
    extract_zip,
)

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# RESULT DATACLASSES
# ============================================================================


@dataclass
class BaselineTrajectoryScore:
    """Per-trajectory scores from both approaches."""

    trajectory_id: str
    task_name: str
    model_name: str
    is_passed: bool  # ground truth

    # Individual matching (baseline) — averaged across training traces
    individual_structural: float = 0.0
    individual_process: float = 0.0
    individual_terminal_match_ratio: float = 0.0
    individual_scores_detail: List[Dict] = field(default_factory=list)

    # Merged PTA matching (proposed)
    merged_structural: float = 0.0
    merged_process: float = 0.0
    merged_terminal_match: bool = False

    error: str = ""

    @property
    def individual_combined(self) -> float:
        return 0.7 * self.individual_structural + 0.3 * self.individual_process

    @property
    def merged_combined(self) -> float:
        return 0.7 * self.merged_structural + 0.3 * self.merged_process


@dataclass
class TaskBaselineResult:
    """Per-task comparison results."""

    task_name: str
    num_train_passed: int
    num_test_passed: int
    num_failed: int
    merged_pta_states: int = 0
    merged_pta_transitions: int = 0

    # Individual matching metrics
    individual_structural: MetricSet = field(default_factory=MetricSet)
    individual_process: MetricSet = field(default_factory=MetricSet)
    individual_combined: MetricSet = field(default_factory=MetricSet)

    # Merged matching metrics
    merged_structural: MetricSet = field(default_factory=MetricSet)
    merged_process: MetricSet = field(default_factory=MetricSet)
    merged_combined: MetricSet = field(default_factory=MetricSet)

    trajectory_results: List[Dict] = field(default_factory=list)
    error: str = ""


@dataclass
class AggregateBaselineResult:
    """Aggregate comparison across all tasks."""

    total_tasks: int = 0
    total_train_passed: int = 0
    total_test_passed: int = 0
    total_failed: int = 0

    # Individual matching — micro-averaged
    individual_structural: MetricSet = field(default_factory=MetricSet)
    individual_process: MetricSet = field(default_factory=MetricSet)
    individual_combined: MetricSet = field(default_factory=MetricSet)

    # Merged matching — micro-averaged
    merged_structural: MetricSet = field(default_factory=MetricSet)
    merged_process: MetricSet = field(default_factory=MetricSet)
    merged_combined: MetricSet = field(default_factory=MetricSet)

    # Macro-averaged AUROC for quick comparison
    macro_individual_structural_auroc: float = 0.0
    macro_merged_structural_auroc: float = 0.0
    macro_individual_process_auroc: float = 0.0
    macro_merged_process_auroc: float = 0.0


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================


def score_individual_matching(
    candidate_trace: Trace,
    training_traces: List[Trace],
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Score a candidate by matching against each training trace individually
    and averaging the results.

    Returns dict with averaged scores and per-trace detail.
    """
    structural_scores = []
    process_scores = []
    terminal_matches = []
    detail = []

    for i, train_trace in enumerate(training_traces):
        try:
            result = match.run(candidate_trace, train_trace, use_llm=use_llm)

            coverage = result.metrics.f1_score
            terminal = result.metrics.terminal_state_match

            # File-level process coverage against this single trace
            req_files = extract_required_files(train_trace)
            if req_files:
                file_cov, _ = check_file_coverage(candidate_trace, req_files)
                proc_cov = file_cov * 100.0
            else:
                proc_cov = 100.0

            structural_scores.append(coverage)
            process_scores.append(proc_cov)
            terminal_matches.append(terminal)
            detail.append(
                {
                    "train_index": i,
                    "structural_coverage": round(coverage, 2),
                    "process_coverage": round(proc_cov, 2),
                    "terminal_match": terminal,
                }
            )
        except Exception as e:
            logger.warning(f"Individual match against trace {i} failed: {e}")
            detail.append({"train_index": i, "error": str(e)})

    if not structural_scores:
        return {"error": "All individual matches failed"}

    avg_structural = sum(structural_scores) / len(structural_scores)
    avg_process = sum(process_scores) / len(process_scores)
    terminal_ratio = sum(1 for t in terminal_matches if t) / len(terminal_matches)

    return {
        "avg_structural": round(avg_structural, 2),
        "avg_process": round(avg_process, 2),
        "terminal_match_ratio": round(terminal_ratio, 2),
        "detail": detail,
    }


def score_merged_matching(
    candidate_trace: Trace,
    merged_pta: Trace,
    required_tools: List[str],
    required_files: List[str] = None,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Score a candidate against the merged PTA.

    Returns dict with scores.
    """
    result = match.run(candidate_trace, merged_pta, use_llm=use_llm)

    coverage = result.metrics.f1_score
    terminal = result.metrics.terminal_state_match

    # File-level process coverage
    if required_files:
        file_cov, _ = check_file_coverage(candidate_trace, required_files)
        proc_cov = file_cov * 100.0
    else:
        proc_cov = 100.0

    return {
        "structural_coverage": round(coverage, 2),
        "process_coverage": round(proc_cov, 2),
        "terminal_match": terminal,
    }


# ============================================================================
# PER-TASK EXPERIMENT
# ============================================================================


def run_task_baseline(
    task_name: str,
    trajectories: List[TrajectoryInfo],
    output_dir: Path,
    temp_dir: Path,
    merge_count: int,
    test_pass_count: int,
    test_fail_count: int,
    use_llm: bool = False,
    seed: int = 42,
) -> Optional[TaskBaselineResult]:
    """Run the baseline comparison for a single task."""

    # Split trajectories — same split as holdout
    train_passed, test_passed, failed = split_trajectories(
        trajectories,
        merge_count=merge_count,
        test_pass_count=test_pass_count,
        seed=seed,
    )

    # Validate data availability
    if len(train_passed) < merge_count:
        logger.warning(
            f"Skipping {task_name}: not enough passed for merging "
            f"({len(train_passed)} < {merge_count})"
        )
        return None
    if len(test_passed) < test_pass_count:
        logger.warning(
            f"Skipping {task_name}: not enough held-out passed "
            f"({len(test_passed)} < {test_pass_count})"
        )
        return None
    if len(failed) < test_fail_count:
        logger.warning(
            f"Skipping {task_name}: not enough failed "
            f"({len(failed)} < {test_fail_count})"
        )
        return None

    # Limit failed to test_fail_count for balanced evaluation
    rng = random.Random(seed)
    failed_shuffled = failed.copy()
    rng.shuffle(failed_shuffled)
    failed = failed_shuffled[:test_fail_count]

    logger.info(
        f"Task {task_name}: train={len(train_passed)}, "
        f"test_passed={len(test_passed)}, failed={len(failed)}"
    )

    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate PTAs for training set ──────────────────────────────────
    train_traces: List[Trace] = []
    for traj in train_passed:
        pta = generate_pta_from_trajectory(traj, task_dir, temp_dir)
        if pta is None:
            logger.warning(
                f"Skipping {task_name}: failed to load training "
                f"trajectory {traj.trajectory_id}"
            )
            return None
        train_traces.append(pta)

    if len(train_traces) < merge_count:
        logger.warning(f"Skipping {task_name}: not enough training PTAs")
        return None

    # ── Merge training traces (for merged approach) ─────────────────────
    merged_pta = merge_ptas(train_traces, use_llm=use_llm)
    if not merged_pta:
        logger.warning(f"Skipping {task_name}: failed to merge PTAs")
        return None

    merged_path = task_dir / f"{task_name}_baseline_merged_pta.json"
    merged_pta.save(str(merged_path))

    merged_required_tools = extract_required_tools(merged_pta)
    merged_required_files = extract_required_files(merged_pta)

    # ── Generate PTAs for test set ──────────────────────────────────────
    test_trajectories = test_passed + failed
    for traj in test_trajectories:
        pta = generate_pta_from_trajectory(traj, task_dir, temp_dir)
        if pta is None:
            logger.warning(
                f"Skipping {task_name}: failed to load test "
                f"trajectory {traj.trajectory_id}"
            )
            return None

    # ── Score each test candidate with BOTH approaches ──────────────────
    all_scores: List[BaselineTrajectoryScore] = []

    for traj in test_trajectories:
        try:
            # Load candidate trace
            if traj.pta_path and Path(traj.pta_path).exists():
                candidate_trace = trace_api.load(traj.pta_path, format="trace")
            else:
                result = get_trajectory_json_path(traj, temp_dir)
                if not result:
                    raise ValueError(f"Could not find JSON for {traj.trajectory_id}")
                json_path, fmt = result
                candidate_trace = trace_api.load(json_path, format=fmt)

            # 1) Individual matching (baseline)
            indiv = score_individual_matching(
                candidate_trace, train_traces, use_llm=use_llm
            )

            # 2) Merged PTA matching (proposed)
            merged = score_merged_matching(
                candidate_trace, merged_pta, merged_required_tools,
                required_files=merged_required_files, use_llm=use_llm
            )

            score = BaselineTrajectoryScore(
                trajectory_id=traj.trajectory_id,
                task_name=traj.task_name,
                model_name=traj.model_name,
                is_passed=traj.passed,
                # Individual
                individual_structural=indiv.get("avg_structural", 0.0),
                individual_process=indiv.get("avg_process", 0.0),
                individual_terminal_match_ratio=indiv.get(
                    "terminal_match_ratio", 0.0
                ),
                individual_scores_detail=indiv.get("detail", []),
                # Merged
                merged_structural=merged["structural_coverage"],
                merged_process=merged["process_coverage"],
                merged_terminal_match=merged["terminal_match"],
                error=indiv.get("error", ""),
            )
            all_scores.append(score)

            logger.debug(
                f"  [{traj.trajectory_id}] passed={traj.passed} | "
                f"individual_struct={score.individual_structural:.1f} "
                f"merged_struct={score.merged_structural:.1f}"
            )

        except Exception as e:
            logger.error(f"Error scoring {traj.trajectory_id}: {e}")
            all_scores.append(
                BaselineTrajectoryScore(
                    trajectory_id=traj.trajectory_id,
                    task_name=traj.task_name,
                    model_name=traj.model_name,
                    is_passed=traj.passed,
                    error=str(e),
                )
            )

    # ── Compute classification metrics for both approaches ──────────────
    task_result = TaskBaselineResult(
        task_name=task_name,
        num_train_passed=len(train_passed),
        num_test_passed=len(test_passed),
        num_failed=len(failed),
        merged_pta_states=len(merged_pta.states),
        merged_pta_transitions=len(merged_pta.transitions),
        trajectory_results=[asdict(s) for s in all_scores],
    )

    _compute_task_metrics(task_result, all_scores)

    return task_result


def _compute_task_metrics(
    result: TaskBaselineResult, scores: List[BaselineTrajectoryScore]
) -> None:
    """Fill in MetricSet fields on a TaskBaselineResult for both approaches."""

    # Helper: create lightweight objects that compute_classification_metrics can handle
    @dataclass
    class _Tmp:
        is_passed: bool
        structural_coverage: float
        error: str = ""

    # ── Individual approach ─────────────────────────────────────────────
    indiv_struct = [
        _Tmp(s.is_passed, s.individual_structural, s.error) for s in scores
    ]
    fill_metric_set(
        result.individual_structural,
        compute_classification_metrics(indiv_struct, "structural_coverage"),
    )

    indiv_proc = [_Tmp(s.is_passed, s.individual_process, s.error) for s in scores]
    fill_metric_set(
        result.individual_process,
        compute_classification_metrics(indiv_proc, "structural_coverage"),
    )

    indiv_comb = [_Tmp(s.is_passed, s.individual_combined, s.error) for s in scores]
    fill_metric_set(
        result.individual_combined,
        compute_classification_metrics(indiv_comb, "structural_coverage"),
    )

    # ── Merged approach ─────────────────────────────────────────────────
    merged_struct = [
        _Tmp(s.is_passed, s.merged_structural, s.error) for s in scores
    ]
    fill_metric_set(
        result.merged_structural,
        compute_classification_metrics(merged_struct, "structural_coverage"),
    )

    merged_proc = [_Tmp(s.is_passed, s.merged_process, s.error) for s in scores]
    fill_metric_set(
        result.merged_process,
        compute_classification_metrics(merged_proc, "structural_coverage"),
    )

    merged_comb = [_Tmp(s.is_passed, s.merged_combined, s.error) for s in scores]
    fill_metric_set(
        result.merged_combined,
        compute_classification_metrics(merged_comb, "structural_coverage"),
    )


# ============================================================================
# AGGREGATE METRICS
# ============================================================================


def compute_aggregate_metrics(
    task_results: List[TaskBaselineResult],
    all_scores: List[BaselineTrajectoryScore],
) -> AggregateBaselineResult:
    """Compute micro- and macro-averaged metrics across all tasks."""

    valid_tasks = [t for t in task_results if not t.error]

    agg = AggregateBaselineResult(
        total_tasks=len(task_results),
        total_train_passed=sum(t.num_train_passed for t in task_results),
        total_test_passed=sum(t.num_test_passed for t in task_results),
        total_failed=sum(t.num_failed for t in task_results),
    )

    if not valid_tasks or not STATS_AVAILABLE:
        return agg

    # Macro-averaged AUROC
    indiv_struct_aurocs = [
        t.individual_structural.auroc for t in valid_tasks if not t.individual_structural.error
    ]
    merged_struct_aurocs = [
        t.merged_structural.auroc for t in valid_tasks if not t.merged_structural.error
    ]
    indiv_proc_aurocs = [
        t.individual_process.auroc for t in valid_tasks if not t.individual_process.error
    ]
    merged_proc_aurocs = [
        t.merged_process.auroc for t in valid_tasks if not t.merged_process.error
    ]

    if indiv_struct_aurocs:
        agg.macro_individual_structural_auroc = float(np.mean(indiv_struct_aurocs))
    if merged_struct_aurocs:
        agg.macro_merged_structural_auroc = float(np.mean(merged_struct_aurocs))
    if indiv_proc_aurocs:
        agg.macro_individual_process_auroc = float(np.mean(indiv_proc_aurocs))
    if merged_proc_aurocs:
        agg.macro_merged_process_auroc = float(np.mean(merged_proc_aurocs))

    # Micro-averaged (pool all test scores)
    @dataclass
    class _Tmp:
        is_passed: bool
        structural_coverage: float
        error: str = ""

    # Individual structural
    indiv_struct_tmp = [
        _Tmp(s.is_passed, s.individual_structural, s.error) for s in all_scores
    ]
    fill_metric_set(
        agg.individual_structural,
        compute_classification_metrics(indiv_struct_tmp, "structural_coverage"),
    )

    # Individual process
    indiv_proc_tmp = [
        _Tmp(s.is_passed, s.individual_process, s.error) for s in all_scores
    ]
    fill_metric_set(
        agg.individual_process,
        compute_classification_metrics(indiv_proc_tmp, "structural_coverage"),
    )

    # Individual combined
    indiv_comb_tmp = [
        _Tmp(s.is_passed, s.individual_combined, s.error) for s in all_scores
    ]
    fill_metric_set(
        agg.individual_combined,
        compute_classification_metrics(indiv_comb_tmp, "structural_coverage"),
    )

    # Merged structural
    merged_struct_tmp = [
        _Tmp(s.is_passed, s.merged_structural, s.error) for s in all_scores
    ]
    fill_metric_set(
        agg.merged_structural,
        compute_classification_metrics(merged_struct_tmp, "structural_coverage"),
    )

    # Merged process
    merged_proc_tmp = [
        _Tmp(s.is_passed, s.merged_process, s.error) for s in all_scores
    ]
    fill_metric_set(
        agg.merged_process,
        compute_classification_metrics(merged_proc_tmp, "structural_coverage"),
    )

    # Merged combined
    merged_comb_tmp = [
        _Tmp(s.is_passed, s.merged_combined, s.error) for s in all_scores
    ]
    fill_metric_set(
        agg.merged_combined,
        compute_classification_metrics(merged_comb_tmp, "structural_coverage"),
    )

    return agg


# ============================================================================
# HTML COMPARISON REPORT
# ============================================================================


def _comparison_table_html(
    indiv_s: MetricSet,
    indiv_p: MetricSet,
    indiv_c: MetricSet,
    merged_s: MetricSet,
    merged_p: MetricSet,
    merged_c: MetricSet,
) -> str:
    """Generate side-by-side comparison table HTML."""

    def _delta(m_val: float, i_val: float) -> str:
        d = m_val - i_val
        color = "var(--success)" if d > 0 else ("var(--accent)" if d < 0 else "var(--text-secondary)")
        sign = "+" if d > 0 else ""
        return f'<span style="color:{color}">{sign}{d:.4f}</span>'

    rows = ""
    metrics = [
        ("AUROC", "auroc", ".4f"),
        ("KS-Statistic", "ks_statistic", ".4f"),
        ("Accuracy", "accuracy", ".1%"),
        ("F1 Score", "f1", ".4f"),
        ("Precision", "precision", ".4f"),
        ("Recall", "recall", ".4f"),
    ]

    for label, attr, fmt in metrics:
        iv_s = getattr(indiv_s, attr)
        iv_p = getattr(indiv_p, attr)
        iv_c = getattr(indiv_c, attr)
        mv_s = getattr(merged_s, attr)
        mv_p = getattr(merged_p, attr)
        mv_c = getattr(merged_c, attr)

        rows += f"""<tr>
            <td><strong>{label}</strong></td>
            <td>{iv_s:{fmt}}</td><td>{mv_s:{fmt}}</td><td>{_delta(mv_s, iv_s)}</td>
            <td>{iv_p:{fmt}}</td><td>{mv_p:{fmt}}</td><td>{_delta(mv_p, iv_p)}</td>
            <td>{iv_c:{fmt}}</td><td>{mv_c:{fmt}}</td><td>{_delta(mv_c, iv_c)}</td>
        </tr>"""

    return f"""
    <table>
        <thead>
            <tr>
                <th rowspan="2">Metric</th>
                <th colspan="3" style="text-align:center">Structural Coverage</th>
                <th colspan="3" style="text-align:center">Process Coverage</th>
                <th colspan="3" style="text-align:center">Combined Score</th>
            </tr>
            <tr>
                <th>Individual</th><th>Merged</th><th>Delta</th>
                <th>Individual</th><th>Merged</th><th>Delta</th>
                <th>Individual</th><th>Merged</th><th>Delta</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>"""


def generate_comparison_report(
    aggregate: AggregateBaselineResult,
    task_results: List[TaskBaselineResult],
    all_scores: List[BaselineTrajectoryScore],
    output_path: str,
) -> None:
    """Generate an HTML comparison report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    styles = get_html_styles()

    # Summary cards
    def card(value: str, label: str, css_class: str = "") -> str:
        cls = f'class="metric-value {css_class}"' if css_class else 'class="metric-value"'
        return f'<div class="metric-card"><div {cls}>{value}</div><div class="metric-label">{label}</div></div>'

    summary_cards = "".join(
        [
            card(str(aggregate.total_tasks), "Tasks"),
            card(str(aggregate.total_train_passed), "Train Passed"),
            card(str(aggregate.total_test_passed), "Test Passed"),
            card(str(aggregate.total_failed), "Test Failed"),
        ]
    )

    # Headline metrics
    ms = aggregate.merged_structural
    iis = aggregate.individual_structural
    headline_cards = "".join(
        [
            card(f"{iis.auroc:.3f}", "Individual AUROC (Struct)"),
            card(f"{ms.auroc:.3f}", "Merged AUROC (Struct)", "good" if ms.auroc > iis.auroc else ""),
            card(f"{iis.f1:.3f}", "Individual F1"),
            card(f"{ms.f1:.3f}", "Merged F1", "good" if ms.f1 > iis.f1 else ""),
        ]
    )

    # Comparison table
    comp_table = _comparison_table_html(
        aggregate.individual_structural,
        aggregate.individual_process,
        aggregate.individual_combined,
        aggregate.merged_structural,
        aggregate.merged_process,
        aggregate.merged_combined,
    )

    # Confusion matrices
    cm_html = f"""
    <div class="charts-grid">
        {generate_confusion_matrix_html(aggregate.individual_structural, "Individual — Structural")}
        {generate_confusion_matrix_html(aggregate.merged_structural, "Merged — Structural")}
    </div>
    <div class="charts-grid">
        {generate_confusion_matrix_html(aggregate.individual_combined, "Individual — Combined")}
        {generate_confusion_matrix_html(aggregate.merged_combined, "Merged — Combined")}
    </div>"""

    # Score distributions
    valid = [s for s in all_scores if not s.error]
    indiv_struct_pass = [s.individual_structural for s in valid if s.is_passed]
    indiv_struct_fail = [s.individual_structural for s in valid if not s.is_passed]
    merged_struct_pass = [s.merged_structural for s in valid if s.is_passed]
    merged_struct_fail = [s.merged_structural for s in valid if not s.is_passed]
    indiv_comb_pass = [s.individual_combined for s in valid if s.is_passed]
    indiv_comb_fail = [s.individual_combined for s in valid if not s.is_passed]
    merged_comb_pass = [s.merged_combined for s in valid if s.is_passed]
    merged_comb_fail = [s.merged_combined for s in valid if not s.is_passed]

    hist_js = "\n".join(
        [
            generate_histogram_js(
                "hist_indiv_struct",
                indiv_struct_pass,
                indiv_struct_fail,
                "Individual — Structural Coverage",
                aggregate.individual_structural.optimal_threshold,
            ),
            generate_histogram_js(
                "hist_merged_struct",
                merged_struct_pass,
                merged_struct_fail,
                "Merged — Structural Coverage",
                aggregate.merged_structural.optimal_threshold,
            ),
            generate_histogram_js(
                "hist_indiv_comb",
                indiv_comb_pass,
                indiv_comb_fail,
                "Individual — Combined Score",
                aggregate.individual_combined.optimal_threshold,
            ),
            generate_histogram_js(
                "hist_merged_comb",
                merged_comb_pass,
                merged_comb_fail,
                "Merged — Combined Score",
                aggregate.merged_combined.optimal_threshold,
            ),
        ]
    )

    # Per-task table
    task_rows = ""
    for tr in sorted(task_results, key=lambda t: t.task_name):
        if tr.error:
            task_rows += f'<tr><td>{tr.task_name}</td><td colspan="8" style="color:var(--accent)">{tr.error}</td></tr>'
            continue
        is_auroc = tr.individual_structural.auroc
        ms_auroc = tr.merged_structural.auroc
        delta = ms_auroc - is_auroc
        delta_color = "var(--success)" if delta > 0 else ("var(--accent)" if delta < 0 else "var(--text-secondary)")
        task_rows += f"""<tr>
            <td>{tr.task_name}</td>
            <td>{tr.num_train_passed}</td>
            <td>{tr.num_test_passed + tr.num_failed}</td>
            <td style="color:{format_auroc_color(is_auroc)}">{is_auroc:.4f}</td>
            <td style="color:{format_auroc_color(ms_auroc)}">{ms_auroc:.4f}</td>
            <td style="color:{delta_color}">{delta:+.4f}</td>
            <td>{tr.individual_structural.f1:.3f}</td>
            <td>{tr.merged_structural.f1:.3f}</td>
            <td>{tr.merged_pta_states}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Baseline Comparison: Individual vs Merged PTA</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>{styles}</style>
</head><body>
<div class="container">
    <h1>Baseline Comparison: Individual vs Merged PTA Matching</h1>
    <p class="timestamp">Generated: {timestamp}</p>

    <h2>Experiment Summary</h2>
    <div class="summary-grid">{summary_cards}</div>

    <h2>Headline: Individual vs Merged</h2>
    <div class="summary-grid">{headline_cards}</div>

    <h2>Full Comparison (Micro-Averaged)</h2>
    <p>Delta = Merged &minus; Individual. <span style="color:var(--success)">Green</span> = merged is better.</p>
    {comp_table}

    <h2>Confusion Matrices</h2>
    {cm_html}

    <h2>Score Distributions</h2>
    <div class="charts-grid">
        <div class="chart-container"><div id="hist_indiv_struct"></div></div>
        <div class="chart-container"><div id="hist_merged_struct"></div></div>
    </div>
    <div class="charts-grid">
        <div class="chart-container"><div id="hist_indiv_comb"></div></div>
        <div class="chart-container"><div id="hist_merged_comb"></div></div>
    </div>

    <h2>Per-Task Breakdown</h2>
    <table>
        <thead><tr>
            <th>Task</th><th>Train</th><th>Test</th>
            <th>Indiv AUROC</th><th>Merged AUROC</th><th>Delta</th>
            <th>Indiv F1</th><th>Merged F1</th><th>Merged States</th>
        </tr></thead>
        <tbody>{task_rows}</tbody>
    </table>

    <h2>Methodology</h2>
    <div style="background:var(--bg-secondary);padding:1.5rem;border-radius:12px;margin:1rem 0">
        <h3>Individual Matching (Baseline)</h3>
        <p>Each test candidate is matched against each training trace independently using
        the same greedy subsequence-coverage algorithm. The structural coverage, process
        coverage, and terminal match are averaged across all N training traces.</p>

        <h3 style="margin-top:1rem">Merged PTA Matching (Proposed)</h3>
        <p>All N training traces are merged into a single ground-truth PTA (a DAG that
        represents the valid solution space). Each test candidate is then matched once
        against this merged PTA. The merged PTA can capture multiple valid solution
        paths, so a correct candidate that follows any valid path receives high coverage.</p>

        <h3 style="margin-top:1rem">Why Merging Should Win</h3>
        <ul style="margin:0.5rem 0 0 1.5rem;color:var(--text-secondary)">
            <li><strong>Single-path bias:</strong> An individual trace only represents one valid approach.
                Averaging inflates scores for candidates that happen to match one trace well, but gives
                low scores to candidates that follow a different — equally valid — approach.</li>
            <li><strong>Solution-space coverage:</strong> The merged PTA creates a DAG where any valid
                path leads to high coverage, regardless of which specific solution strategy was used.</li>
            <li><strong>Better discrimination:</strong> Because the merged PTA captures the full solution space,
                failed trajectories (that go off-track) get systematically low coverage, while passed
                trajectories (following any valid path) get high coverage.</li>
        </ul>
    </div>
</div>
<script>{hist_js}</script>
</body></html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    logger.info(f"HTML report saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: Individual matching vs merged PTA matching"
    )
    parser.add_argument("data_dir", help="Directory containing trajectory data")
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: data_dir/__baseline_experiment)",
    )
    parser.add_argument(
        "--merge-count",
        type=int,
        required=True,
        help="Number of passed trajectories for training (>= 2)",
    )
    parser.add_argument(
        "--test-pass-count",
        type=int,
        required=True,
        help="Number of held-out passed trajectories for testing (>= 1)",
    )
    parser.add_argument(
        "--test-fail-count",
        type=int,
        required=True,
        help="Number of failed trajectories for testing (>= 1)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for equivalence checks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    # Validate
    if args.merge_count < 2:
        print("ERROR: --merge-count must be at least 2")
        return 1
    if args.test_pass_count < 1:
        print("ERROR: --test-pass-count must be at least 1")
        return 1
    if args.test_fail_count < 1:
        print("ERROR: --test-fail-count must be at least 1")
        return 1

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not STATS_AVAILABLE:
        print("ERROR: Required packages not available. Install with:")
        print("  pip install scikit-learn scipy numpy")
        return 1

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else data_dir / "__baseline_experiment"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Temp directory for ZIP extraction (short path to avoid Windows MAX_PATH)
    temp_dir = Path(tempfile.mkdtemp(prefix="pta_baseline_"))

    print(f"Discovering trajectories in: {data_dir}")
    tasks = discover_trajectories(data_dir)
    print(f"Found {len(tasks)} tasks")

    # Filter eligible tasks
    min_passed = args.merge_count + args.test_pass_count
    eligible = {}
    for name, trajs in tasks.items():
        n_pass = sum(1 for t in trajs if t.passed)
        n_fail = sum(1 for t in trajs if not t.passed)
        if n_pass >= min_passed and n_fail >= args.test_fail_count:
            eligible[name] = trajs

    print(
        f"Eligible tasks (>= {min_passed} passed "
        f"[{args.merge_count} merge + {args.test_pass_count} test], "
        f">= {args.test_fail_count} failed): {len(eligible)}"
    )

    if not eligible:
        print("ERROR: No eligible tasks found")
        return 1

    # ── Run experiment for each task ────────────────────────────────────
    task_results: List[TaskBaselineResult] = []
    all_scores: List[BaselineTrajectoryScore] = []

    print(
        f"\nRunning baseline experiment "
        f"(merge_count={args.merge_count}, "
        f"test_pass_count={args.test_pass_count}, "
        f"test_fail_count={args.test_fail_count})..."
    )

    for task_name in tqdm(sorted(eligible.keys()), desc="Processing tasks"):
        result = run_task_baseline(
            task_name=task_name,
            trajectories=eligible[task_name],
            output_dir=output_dir,
            temp_dir=temp_dir,
            merge_count=args.merge_count,
            test_pass_count=args.test_pass_count,
            test_fail_count=args.test_fail_count,
            use_llm=args.use_llm,
            seed=args.seed,
        )

        if result:
            task_results.append(result)
            for tr_dict in result.trajectory_results:
                all_scores.append(BaselineTrajectoryScore(**{
                    k: v for k, v in tr_dict.items()
                    if k in BaselineTrajectoryScore.__dataclass_fields__
                }))

    if not task_results:
        print("ERROR: No tasks completed successfully")
        return 1

    print(f"\nCompleted {len(task_results)} tasks")

    # ── Aggregate metrics ───────────────────────────────────────────────
    aggregate = compute_aggregate_metrics(task_results, all_scores)

    # ── Print summary ───────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("BASELINE COMPARISON: Individual Matching vs Merged PTA Matching")
    print("=" * 90)
    print(f"\nDataset: {aggregate.total_tasks} tasks")
    print(f"  Train (passed): {aggregate.total_train_passed}")
    print(f"  Test (passed):  {aggregate.total_test_passed}")
    print(f"  Test (failed):  {aggregate.total_failed}")

    iis = aggregate.individual_structural
    ms = aggregate.merged_structural
    iip = aggregate.individual_process
    mp = aggregate.merged_process
    iic = aggregate.individual_combined
    mc = aggregate.merged_combined

    print(f"\n{'=' * 90}")
    print("MICRO-AVERAGED COMPARISON")
    print(f"{'=' * 90}")
    print(
        f"\n{'Metric':<20} {'Indiv Struct':>14} {'Merged Struct':>14} "
        f"{'Indiv Proc':>14} {'Merged Proc':>14} "
        f"{'Indiv Comb':>14} {'Merged Comb':>14}"
    )
    print("-" * 104)
    print(
        f"{'AUROC':<20} {iis.auroc:>14.4f} {ms.auroc:>14.4f} "
        f"{iip.auroc:>14.4f} {mp.auroc:>14.4f} "
        f"{iic.auroc:>14.4f} {mc.auroc:>14.4f}"
    )
    print(
        f"{'KS-Statistic':<20} {iis.ks_statistic:>14.4f} {ms.ks_statistic:>14.4f} "
        f"{iip.ks_statistic:>14.4f} {mp.ks_statistic:>14.4f} "
        f"{iic.ks_statistic:>14.4f} {mc.ks_statistic:>14.4f}"
    )
    print(
        f"{'Accuracy':<20} {iis.accuracy:>13.1%} {ms.accuracy:>13.1%} "
        f"{iip.accuracy:>13.1%} {mp.accuracy:>13.1%} "
        f"{iic.accuracy:>13.1%} {mc.accuracy:>13.1%}"
    )
    print(
        f"{'F1 Score':<20} {iis.f1:>14.4f} {ms.f1:>14.4f} "
        f"{iip.f1:>14.4f} {mp.f1:>14.4f} "
        f"{iic.f1:>14.4f} {mc.f1:>14.4f}"
    )

    print(f"\n{'=' * 90}")
    print("MACRO-AVERAGED AUROC (mean across tasks)")
    print(f"{'=' * 90}")
    print(
        f"  Structural: Individual={aggregate.macro_individual_structural_auroc:.4f}  "
        f"Merged={aggregate.macro_merged_structural_auroc:.4f}"
    )
    print(
        f"  Process:    Individual={aggregate.macro_individual_process_auroc:.4f}  "
        f"Merged={aggregate.macro_merged_process_auroc:.4f}"
    )

    # ── Save results ────────────────────────────────────────────────────
    results_json = {
        "generated_at": datetime.now().isoformat(),
        "experiment_type": "baseline_comparison",
        "experiment_config": {
            "data_dir": str(data_dir),
            "merge_count": args.merge_count,
            "test_pass_count": args.test_pass_count,
            "test_fail_count": args.test_fail_count,
            "seed": args.seed,
            "use_llm": args.use_llm,
        },
        "aggregate_metrics": asdict(aggregate),
        "task_results": [asdict(t) for t in task_results],
    }

    results_path = output_dir / "baseline_experiment_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Generate HTML report
    html_path = output_dir / "baseline_comparison_report.html"
    generate_comparison_report(aggregate, task_results, all_scores, str(html_path))
    print(f"Generated HTML report: {html_path}")

    # Cleanup
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Failed to clean up temp directory: {e}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    sys.exit(main() or 0)
