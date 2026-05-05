#!/usr/bin/env python3
"""
Compute Correlation Metrics - Analyze how well merged PTA discriminates pass/fail trajectories.

This script computes classification metrics to evaluate whether the merged PTA
(built from successful trajectories) can distinguish between passing and failing runs.

Metrics computed:
- AUROC: Area Under ROC Curve (0.5 = random, 1.0 = perfect discrimination)
- KS-Statistic: Maximum separation between pass/fail score distributions
- Optimal Threshold: Threshold that maximizes Youden's J (TPR - FPR)
- Accuracy/F1: Classification metrics at optimal threshold

Usage:
    python compute_correlation_metrics.py <experiment_outputs_dir>
    python compute_correlation_metrics.py <experiment_outputs_dir> --output-dir <output_dir>
    python compute_correlation_metrics.py <experiment_outputs_dir> --use-llm
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from swe_trace_sdk import trace as trace_api, match
from swe_trace_sdk.models import Trace
from swe_trace_sdk.match import extract_required_tools, check_process_coverage
from swe_trace_sdk.intent import label_trace_intents

# Import shared utilities
from metrics_utils import (
    STATS_AVAILABLE,
    TQDM_AVAILABLE,
    tqdm,
    MetricSet,
    TrajectoryScore,
    compute_classification_metrics,
    compute_verdict,
    fill_metric_set,
    compute_all_metric_sets,
    get_html_styles,
    generate_confusion_matrix_html,
    generate_ablation_table_html,
    generate_histogram_js,
    extract_score_distributions,
    format_auroc_color,
)

# For backward compatibility, also expose these
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Note: TrajectoryScore is now imported from metrics_utils


@dataclass
class TaskMetrics:
    """Classification metrics for a single task.
    
    Uses a dict-based structure so new score types added to the SDK
    (e.g. weighted_score, stage_completeness) are picked up automatically.
    """
    task_name: str
    num_passed: int
    num_failed: int
    
    # Per-score-type metrics: {score_name: {metric_name: value}}
    # e.g. score_metrics['structural']['auroc'] = 0.82
    score_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    error: str = ""
    
    # ── Convenience accessors for backward compatibility ──────────
    @property
    def structural_auroc(self) -> float:
        return self.score_metrics.get('structural', {}).get('auroc', 0.0)
    
    @property
    def structural_ks_statistic(self) -> float:
        return self.score_metrics.get('structural', {}).get('ks_statistic', 0.0)
    
    @property
    def structural_accuracy(self) -> float:
        return self.score_metrics.get('structural', {}).get('accuracy', 0.0)
    
    @property
    def structural_f1(self) -> float:
        return self.score_metrics.get('structural', {}).get('f1', 0.0)
    
    @property
    def weighted_auroc(self) -> float:
        return self.score_metrics.get('weighted', {}).get('auroc', 0.0)
    
    @property
    def combined_auroc(self) -> float:
        return self.score_metrics.get('combined', {}).get('auroc', 0.0)
    
    @property
    def process_auroc(self) -> float:
        return self.score_metrics.get('process', {}).get('auroc', 0.0)


# All score types to evaluate — maps label → (attribute, needs_scaling)
# needs_scaling=True means the attribute is 0-1 and should be × 100 for threshold logic
SCORE_TYPES: Dict[str, Tuple[str, bool]] = {
    'structural':         ('structural_coverage',  False),
    'process':            ('process_coverage',     False),
    'weighted':           ('weighted_score',       False),
    'stage_completeness': ('stage_completeness',   True),   # 0-1 → 0-100
    'workflow_similarity':('workflow_similarity',   True),   # 0-1 → 0-100
    'coherence':          ('coherence_score',      True),   # 0-1 → 0-100
    'temporal_profile':   ('temporal_profile_score', True), # 0-1 → 0-100
    'bottleneck':         ('bottleneck_coverage',   False),  # already 0-100
    'combined':           ('combined_score',       False),
}


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all tasks.
    
    Uses dict-based storage so it automatically adapts when new score
    types are added to the SDK or SCORE_TYPES mapping.
    """
    total_tasks: int
    total_trajectories: int
    total_passed: int
    total_failed: int
    
    # Macro-averaged metrics per score type: {score_name: {metric_name: value}}
    macro_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Micro-averaged metrics per score type: {score_name: {metric_name: value}}
    micro_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # ── Convenience accessors for backward compatibility ──────────
    @property
    def micro_structural_auroc(self) -> float:
        return self.micro_metrics.get('structural', {}).get('auroc', 0.0)
    
    @property
    def micro_structural_ks_statistic(self) -> float:
        return self.micro_metrics.get('structural', {}).get('ks_statistic', 0.0)
    
    @property
    def micro_structural_optimal_threshold(self) -> float:
        return self.micro_metrics.get('structural', {}).get('optimal_threshold', 0.0)
    
    @property
    def micro_structural_accuracy(self) -> float:
        return self.micro_metrics.get('structural', {}).get('accuracy', 0.0)
    
    @property
    def micro_structural_f1(self) -> float:
        return self.micro_metrics.get('structural', {}).get('f1', 0.0)
    
    @property
    def micro_structural_precision(self) -> float:
        return self.micro_metrics.get('structural', {}).get('precision', 0.0)
    
    @property
    def micro_structural_recall(self) -> float:
        return self.micro_metrics.get('structural', {}).get('recall', 0.0)
    
    @property
    def micro_structural_ks_pvalue(self) -> float:
        return self.micro_metrics.get('structural', {}).get('ks_pvalue', 1.0)
    
    @property
    def micro_tp(self) -> int:
        return self.micro_metrics.get('structural', {}).get('tp', 0)
    
    @property
    def micro_fp(self) -> int:
        return self.micro_metrics.get('structural', {}).get('fp', 0)
    
    @property
    def micro_tn(self) -> int:
        return self.micro_metrics.get('structural', {}).get('tn', 0)
    
    @property
    def micro_fn(self) -> int:
        return self.micro_metrics.get('structural', {}).get('fn', 0)


def compute_similarity_scores(
    merged_pta_path: str,
    trajectory_results: List[Dict[str, Any]],
    use_llm: bool = False,
    llm_prefix: str = "DEFAULT"
) -> List[TrajectoryScore]:
    """
    Compute similarity scores for all trajectories against merged PTA.
    
    Args:
        merged_pta_path: Path to merged PTA JSON
        trajectory_results: List of trajectory info dicts from experiment summary
        use_llm: Whether to use LLM for equivalence checks
        llm_prefix: LLM config prefix
        
    Returns:
        List of TrajectoryScore objects
    """
    scores = []
    
    # Load merged trace using SDK
    try:
        merged_trace = trace_api.load(merged_pta_path, format="trace")
        
        # Extract required tools
        required_tools = extract_required_tools(merged_trace)
        
    except Exception as e:
        logger.error(f"Failed to load merged PTA: {e}")
        return scores
    
    # Process each trajectory
    for traj_info in trajectory_results:
        traj_id = traj_info.get('trajectory_id', 'unknown')
        task_name = traj_info.get('task_name', 'unknown')
        model_name = traj_info.get('model_name', 'unknown')
        passed = traj_info.get('pass_fail', 'failed') == 'passed'
        pta_file = traj_info.get('pta_file', '')
        
        if not pta_file or not Path(pta_file).exists():
            scores.append(TrajectoryScore(
                trajectory_id=traj_id,
                task_name=task_name,
                model_name=model_name,
                passed=passed,
                structural_coverage=0.0,
                process_coverage=0.0,
                terminal_match=False,
                matched_states=0,
                total_states=0,
                predicted_verdict="ERROR",
                error=f"PTA file not found: {pta_file}"
            ))
            continue
        
        try:
            # Load trajectory trace using SDK
            traj_trace = trace_api.load(pta_file, format="trace")
            
            # Use SDK match — returns all metrics (coverage, weighted, stages, workflow)
            label_trace_intents(traj_trace)  # Label candidate intent stages for workflow similarity
            match_result = match.run(traj_trace, merged_trace, use_llm=use_llm)
            m = match_result.metrics  # MatchMetrics
            
            # Compute process coverage (% of required tools present)
            proc_cov = 100.0  # Default to 100% if no required tools
            missing_tools = []
            if required_tools:
                proc_cov_ratio, missing_tools = check_process_coverage(traj_trace, required_tools)
                proc_cov = proc_cov_ratio * 100.0
            
            # Compute predicted verdict using all available metrics
            result_for_verdict = {
                'coverage_percent': m.coverage_percent,
                'terminal_state_match': m.terminal_state_match,
                'process_coverage': proc_cov / 100.0,
                'missing_tools': missing_tools,
                'weighted_score': m.weighted_score,
                'stage_completeness': m.stage_completeness,
                'workflow_similarity': m.workflow_similarity,
            }
            verdict = compute_verdict(result_for_verdict)
            
            scores.append(TrajectoryScore(
                trajectory_id=traj_id,
                task_name=task_name,
                model_name=model_name,
                passed=passed,
                structural_coverage=round(m.coverage_percent, 2),
                process_coverage=round(proc_cov, 2),
                weighted_score=round(m.weighted_score, 2),
                stage_completeness=round(m.stage_completeness, 4),
                workflow_similarity=round(m.workflow_similarity, 4),
                stage_coverage=m.stage_coverage,
                terminal_match=m.terminal_state_match,
                matched_states=m.matched_count,
                total_states=m.total_ground_truth_states,
                predicted_verdict=verdict,
            ))
            
        except Exception as e:
            logger.error(f"Error processing {traj_id}: {e}")
            scores.append(TrajectoryScore(
                trajectory_id=traj_id,
                task_name=task_name,
                model_name=model_name,
                passed=passed,
                structural_coverage=0.0,
                process_coverage=0.0,
                terminal_match=False,
                matched_states=0,
                total_states=0,
                predicted_verdict="ERROR",
                error=str(e)
            ))
    
    return scores


# Note: compute_classification_metrics is now imported from metrics_utils


def _compute_score_metrics(
    task_scores: List[Any],
    attr: str,
    needs_scaling: bool,
) -> Dict[str, Any]:
    """Compute classification metrics for a single score type.
    
    If *needs_scaling* is True, the attribute values are 0-1 and will be
    scaled to 0-100 before computing metrics (so thresholds are in %).
    """
    if not needs_scaling:
        # For combined_score (a property), we need temp objects
        if attr == 'combined_score':
            @dataclass
            class _Tmp:
                passed: bool
                combined_score: float
                error: str = ""
            temps = [_Tmp(getattr(s, 'passed', False), s.combined_score, getattr(s, 'error', ''))
                     for s in task_scores]
            return compute_classification_metrics(temps, 'combined_score')
        return compute_classification_metrics(task_scores, attr)
    
    # Scale 0-1 → 0-100
    @dataclass
    class _Scaled:
        passed: bool
        scaled_val: float
        error: str = ""
    
    scaled = [_Scaled(getattr(s, 'passed', False),
                      getattr(s, attr, 0.0) * 100.0,
                      getattr(s, 'error', ''))
              for s in task_scores]
    return compute_classification_metrics(scaled, 'scaled_val')


def compute_task_metrics(task_scores: List[TrajectoryScore]) -> TaskMetrics:
    """Compute classification metrics for a single task across all score types."""
    task_name = task_scores[0].task_name if task_scores else "unknown"
    num_passed = sum(1 for s in task_scores if s.passed)
    num_failed = sum(1 for s in task_scores if not s.passed)
    
    metrics = TaskMetrics(
        task_name=task_name,
        num_passed=num_passed,
        num_failed=num_failed,
    )
    
    # Need both classes for classification metrics
    if num_passed == 0 or num_failed == 0:
        metrics.error = "Single class only (all passed or all failed)"
        return metrics
    
    # Compute metrics for every score type (data-driven)
    for label, (attr, needs_scaling) in SCORE_TYPES.items():
        result = _compute_score_metrics(task_scores, attr, needs_scaling)
        if 'error' not in result:
            metrics.score_metrics[label] = result
    
    return metrics


def compute_aggregate_metrics(
    all_scores: List[TrajectoryScore],
    task_metrics: List[TaskMetrics]
) -> AggregateMetrics:
    """Compute aggregate metrics across all tasks for all score types."""
    
    valid_tasks = [t for t in task_metrics if not t.error]
    
    aggregate = AggregateMetrics(
        total_tasks=len(task_metrics),
        total_trajectories=len(all_scores),
        total_passed=sum(1 for s in all_scores if s.passed),
        total_failed=sum(1 for s in all_scores if not s.passed),
    )
    
    if not valid_tasks:
        return aggregate
    
    # ── Macro-averaged (mean across tasks) ────────────────────────
    if NUMPY_AVAILABLE:
        for label in SCORE_TYPES:
            aurocs = [t.score_metrics.get(label, {}).get('auroc', 0.5)
                      for t in valid_tasks if label in t.score_metrics]
            ks_vals = [t.score_metrics.get(label, {}).get('ks_statistic', 0.0)
                       for t in valid_tasks if label in t.score_metrics]
            if aurocs:
                aggregate.macro_metrics[label] = {
                    'auroc': float(np.mean(aurocs)),
                    'ks_statistic': float(np.mean(ks_vals)) if ks_vals else 0.0,
                }
    
    # ── Micro-averaged (pool all trajectories) ────────────────────
    for label, (attr, needs_scaling) in SCORE_TYPES.items():
        result = _compute_score_metrics(all_scores, attr, needs_scaling)
        if 'error' not in result:
            aggregate.micro_metrics[label] = result
    
    return aggregate


def generate_html_report(
    aggregate: AggregateMetrics,
    task_metrics: List[TaskMetrics],
    all_scores: List[TrajectoryScore],
    output_path: str
) -> None:
    """Generate interactive HTML report with visualizations for all score types."""
    
    # Extract distributions for all score types
    dists = extract_score_distributions(all_scores)
    
    # Pretty labels for score types
    LABELS = {
        'structural': '📐 Structural Coverage',
        'process': '⚙️ Process Coverage',
        'weighted': '⚖️ Weighted Score',
        'stage_completeness': '🎯 Stage Completeness',
        'workflow_similarity': '🔄 Workflow Similarity',
        'combined': '🧮 Combined Score',
    }
    
    # Build comparison table rows from micro metrics
    comparison_rows = ""
    for metric_name in ('auroc', 'ks_statistic', 'optimal_threshold', 'accuracy', 'f1'):
        display = {
            'auroc': 'AUROC', 'ks_statistic': 'KS-Statistic',
            'optimal_threshold': 'Threshold', 'accuracy': 'Accuracy', 'f1': 'F1 Score',
        }[metric_name]
        comparison_rows += f"<tr><td><strong>{display}</strong></td>"
        for label in SCORE_TYPES:
            val = aggregate.micro_metrics.get(label, {}).get(metric_name, 0.0)
            if metric_name == 'accuracy':
                comparison_rows += f"<td>{val:.1%}</td>"
            elif metric_name == 'optimal_threshold':
                comparison_rows += f"<td>{val:.1f}%</td>"
            else:
                color = 'var(--success)' if metric_name == 'auroc' and val >= 0.7 else \
                        'var(--warning)' if metric_name == 'auroc' and val >= 0.6 else 'inherit'
                comparison_rows += f'<td style="color: {color}">{val:.3f}</td>'
        comparison_rows += "</tr>\n"
    
    # Build per-task table rows
    task_rows = ""
    for tm in sorted(task_metrics, key=lambda x: x.structural_auroc, reverse=True):
        if tm.error:
            task_rows += f'<tr><td>{tm.task_name}</td><td>{tm.num_passed}/{tm.num_failed}</td>'
            task_rows += f'<td colspan="{len(SCORE_TYPES) + 2}" style="color: var(--text-secondary)">{tm.error}</td></tr>\n'
        else:
            task_rows += f'<tr><td>{tm.task_name}</td><td>{tm.num_passed}/{tm.num_failed}</td>'
            for label in SCORE_TYPES:
                auroc = tm.score_metrics.get(label, {}).get('auroc', 0.0)
                color = 'var(--success)' if auroc >= 0.7 else 'var(--warning)' if auroc >= 0.6 else 'inherit'
                task_rows += f'<td style="color: {color}">{auroc:.3f}</td>'
            task_rows += f'<td>{tm.structural_ks_statistic:.3f}</td>'
            task_rows += f'<td>{tm.structural_accuracy:.1%}</td>'
            task_rows += "</tr>\n"
    
    # Column headers for score types
    score_th = "".join(f"<th>{LABELS.get(l, l)}</th>" for l in SCORE_TYPES)
    
    # Build chart JS (one histogram per score type)
    chart_divs = ""
    chart_js = ""
    for label in SCORE_TYPES:
        div_id = f"hist_{label}"
        chart_divs += f'<div class="chart-container"><div id="{div_id}"></div></div>\n'
        pass_vals = json.dumps(dists.get(label, {}).get('pass', []))
        fail_vals = json.dumps(dists.get(label, {}).get('fail', []))
        threshold = aggregate.micro_metrics.get(label, {}).get('optimal_threshold', 50.0)
        chart_js += f"""
        Plotly.newPlot('{div_id}', [
            {{ x: {pass_vals}, name: 'Passed', type: 'histogram', opacity: 0.7, marker: {{ color: '#4ade80' }}, xbins: {{ size: 5 }} }},
            {{ x: {fail_vals}, name: 'Failed', type: 'histogram', opacity: 0.7, marker: {{ color: '#e94560' }}, xbins: {{ size: 5 }} }}
        ], {{
            title: '{LABELS.get(label, label)} Distribution',
            barmode: 'overlay',
            xaxis: {{ title: 'Score %', range: [0, 100] }},
            yaxis: {{ title: 'Count' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {{ color: '#eee' }},
            shapes: [{{ type: 'line', x0: {threshold}, x1: {threshold}, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#fbbf24', width: 2, dash: 'dash' }} }}]
        }});
"""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PTA Correlation Metrics Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>{get_html_styles()}</style>
</head>
<body>
    <div class="container">
        <h1>🎯 PTA Correlation Metrics Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>📊 Dataset Summary</h2>
        <div class="summary-grid">
            <div class="metric-card"><div class="metric-value">{aggregate.total_tasks}</div><div class="metric-label">Tasks</div></div>
            <div class="metric-card"><div class="metric-value">{aggregate.total_trajectories}</div><div class="metric-label">Trajectories</div></div>
            <div class="metric-card"><div class="metric-value" style="color:var(--success)">{aggregate.total_passed}</div><div class="metric-label">Passed</div></div>
            <div class="metric-card"><div class="metric-value" style="color:var(--accent)">{aggregate.total_failed}</div><div class="metric-label">Failed</div></div>
        </div>
        
        <h2>🎯 All Score Types — Micro-Averaged Metrics</h2>
        <table>
            <thead><tr><th>Metric</th>{score_th}<th></th></tr></thead>
            <tbody>{comparison_rows}</tbody>
        </table>
        
        <h2>📈 Confusion Matrix (Weighted Score @ {aggregate.micro_metrics.get('weighted', {}).get('optimal_threshold', 0):.1f}%)</h2>
        <div class="confusion-matrix">
            <div class="cm-cell cm-header"></div><div class="cm-cell cm-header">Pred Pass</div><div class="cm-cell cm-header">Pred Fail</div>
            <div class="cm-cell cm-header">Actual Pass</div>
            <div class="cm-cell cm-tp">TP: {aggregate.micro_metrics.get('weighted', {}).get('tp', 0)}</div>
            <div class="cm-cell cm-fn">FN: {aggregate.micro_metrics.get('weighted', {}).get('fn', 0)}</div>
            <div class="cm-cell cm-header">Actual Fail</div>
            <div class="cm-cell cm-fp">FP: {aggregate.micro_metrics.get('weighted', {}).get('fp', 0)}</div>
            <div class="cm-cell cm-tn">TN: {aggregate.micro_metrics.get('weighted', {}).get('tn', 0)}</div>
        </div>
        
        <h2>📊 Score Distributions</h2>
        <div class="charts-grid">
            {chart_divs}
        </div>
        
        <h2>📋 Per-Task AUROC by Score Type</h2>
        <table>
            <thead><tr><th>Task</th><th>P/F</th>{score_th}<th>KS</th><th>Acc</th></tr></thead>
            <tbody>{task_rows}</tbody>
        </table>
        
        <h2>📝 Metric Definitions</h2>
        <table>
            <tr><th>Score Type</th><th>Description</th></tr>
            <tr><td><strong>Structural Coverage</strong></td><td>% of merged PTA states matched (in order) via subsequence coverage.</td></tr>
            <tr><td><strong>Process Coverage</strong></td><td>% of required tools present in trajectory.</td></tr>
            <tr><td><strong>Weighted Score</strong></td><td>Stage-importance-weighted coverage. Implementation=3×, Verification=2×, Exploration=1×, Orchestration=0.5×.</td></tr>
            <tr><td><strong>Stage Completeness</strong></td><td>Fraction of ground-truth intent stages with ≥1 match (0-100%).</td></tr>
            <tr><td><strong>Workflow Similarity</strong></td><td>LCS-ratio similarity between candidate and GT intent-stage transition fingerprints.</td></tr>
            <tr><td><strong>Combined</strong></td><td>50% weighted + 25% process + 25% workflow similarity.</td></tr>
        </table>
    </div>
    <script>{chart_js}</script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"Generated HTML report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute correlation metrics between merged PTA and pass/fail trajectories"
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to experiment outputs directory (containing experiment_summary.json)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (default: same as experiment_dir)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for semantic equivalence checks"
    )
    parser.add_argument(
        "--llm-prefix",
        default="DEFAULT",
        help="Environment variable prefix for LLM config"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check dependencies
    if not STATS_AVAILABLE or not NUMPY_AVAILABLE:
        print("ERROR: Missing required packages. Install with:")
        print("  pip install scikit-learn scipy numpy")
        return 1
    
    # Load experiment summary
    experiment_dir = Path(args.experiment_dir)
    summary_path = experiment_dir / "experiment_summary.json"
    
    if not summary_path.exists():
        print(f"ERROR: experiment_summary.json not found in {experiment_dir}")
        return 1
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print(f"Loaded experiment summary: {summary.get('total_tasks', 0)} tasks, {summary.get('total_trajectories', 0)} trajectories")
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each task
    all_scores: List[TrajectoryScore] = []
    task_metrics_list: List[TaskMetrics] = []
    
    task_results = summary.get('task_results', [])
    
    print(f"\nComputing similarity scores...")
    for task_info in tqdm(task_results, desc="Processing tasks"):
        task_name = task_info.get('task_name', 'unknown')
        merged_pta_path = task_info.get('merged_pta_file', '')
        trajectory_results = task_info.get('trajectory_results', [])
        
        if not merged_pta_path or not Path(merged_pta_path).exists():
            logger.warning(f"Skipping {task_name}: no merged PTA")
            continue
        
        if not trajectory_results:
            logger.warning(f"Skipping {task_name}: no trajectories")
            continue
        
        # Compute scores for this task
        task_scores = compute_similarity_scores(
            merged_pta_path,
            trajectory_results,
            use_llm=args.use_llm,
            llm_prefix=args.llm_prefix
        )
        
        if task_scores:
            all_scores.extend(task_scores)
            
            # Compute task-level metrics
            tm = compute_task_metrics(task_scores)
            task_metrics_list.append(tm)
    
    if not all_scores:
        print("ERROR: No scores computed. Check that PTA files exist.")
        return 1
    
    print(f"\nComputed scores for {len(all_scores)} trajectories across {len(task_metrics_list)} tasks")
    
    # Compute aggregate metrics
    aggregate = compute_aggregate_metrics(all_scores, task_metrics_list)
    
    # Print summary — data-driven across all score types
    score_labels = list(SCORE_TYPES.keys())
    col_width = 16
    
    print("\n" + "="*70)
    print("PTA CORRELATION METRICS SUMMARY")
    print("="*70)
    print(f"\nDataset: {aggregate.total_trajectories} trajectories ({aggregate.total_passed} passed, {aggregate.total_failed} failed)")
    
    print(f"\n{'='*70}")
    print("MICRO-AVERAGED METRICS (pooled across all trajectories)")
    print(f"{'='*70}")
    
    header = f"{'Metric':<16}" + "".join(f"{l:>{col_width}}" for l in score_labels)
    print(f"\n{header}")
    print("-" * len(header))
    
    for metric_name, fmt in [('auroc', '.4f'), ('ks_statistic', '.4f'),
                              ('optimal_threshold', '.1f'), ('accuracy', '.1%'), ('f1', '.4f')]:
        display = {'auroc': 'AUROC', 'ks_statistic': 'KS-Statistic',
                   'optimal_threshold': 'Opt. Threshold', 'accuracy': 'Accuracy', 'f1': 'F1 Score'}[metric_name]
        row = f"{display:<16}"
        for label in score_labels:
            val = aggregate.micro_metrics.get(label, {}).get(metric_name, 0.0)
            if fmt == '.1%':
                row += f"{val:{col_width}.1%}"
            elif fmt == '.1f':
                row += f"{val:>{col_width - 1}.1f}%"
            else:
                row += f"{val:>{col_width}{fmt}}"
        print(row)
    
    # Structural details
    s = aggregate.micro_metrics.get('structural', {})
    print(f"\nStructural Coverage Details:")
    print(f"  Precision: {s.get('precision', 0):.4f}")
    print(f"  Recall:    {s.get('recall', 0):.4f}")
    print(f"  KS p-value: {s.get('ks_pvalue', 1):.2e}")
    
    print(f"\n{'='*70}")
    print("MACRO-AVERAGED METRICS (averaged across tasks)")
    print(f"{'='*70}")
    
    header = f"{'Metric':<16}" + "".join(f"{l:>{col_width}}" for l in score_labels)
    print(f"\n{header}")
    print("-" * len(header))
    for metric_name in ('auroc', 'ks_statistic'):
        display = {'auroc': 'AUROC', 'ks_statistic': 'KS-Statistic'}[metric_name]
        row = f"{display:<16}"
        for label in score_labels:
            val = aggregate.macro_metrics.get(label, {}).get(metric_name, 0.0)
            row += f"{val:>{col_width}.4f}"
        print(row)
    
    # Save results
    results = {
        "generated_at": datetime.now().isoformat(),
        "experiment_dir": str(experiment_dir),
        "score_types": score_labels,
        "aggregate_metrics": {
            "total_tasks": aggregate.total_tasks,
            "total_trajectories": aggregate.total_trajectories,
            "total_passed": aggregate.total_passed,
            "total_failed": aggregate.total_failed,
            "macro_metrics": aggregate.macro_metrics,
            "micro_metrics": aggregate.micro_metrics,
        },
        "task_metrics": [
            {"task_name": tm.task_name, "num_passed": tm.num_passed,
             "num_failed": tm.num_failed, "error": tm.error,
             "score_metrics": tm.score_metrics}
            for tm in task_metrics_list
        ],
        "trajectory_scores": [asdict(s) for s in all_scores],
    }
    
    results_path = output_dir / "correlation_metrics.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")
    
    # Generate HTML report
    html_path = output_dir / "correlation_metrics_report.html"
    generate_html_report(aggregate, task_metrics_list, all_scores, str(html_path))
    print(f"Generated HTML report: {html_path}")
    
    print("\n" + "="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
