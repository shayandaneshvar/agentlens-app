#!/usr/bin/env python3
"""
Shared utilities for computing PTA correlation metrics.

This module provides common functionality used by both:
- compute_correlation_metrics.py (full dataset evaluation)
- run_holdout_experiment.py (train/test split validation)
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# OPTIONAL IMPORTS WITH FALLBACKS
# ============================================================================

try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, accuracy_score, f1_score,
        precision_score, recall_score, confusion_matrix
    )
    import numpy as np
    from scipy.stats import ks_2samp
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    np = None
    logger.warning("Install sklearn, scipy, numpy for metrics: pip install scikit-learn scipy numpy")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# SHARED DATACLASSES
# ============================================================================

@dataclass
class MetricSet:
    """Complete set of classification metrics for a single score type."""
    auroc: float = 0.0
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    optimal_threshold: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    pass_mean: float = 0.0
    pass_std: float = 0.0
    fail_mean: float = 0.0
    fail_std: float = 0.0
    error: str = ""


@dataclass
class TrajectoryScore:
    """Score for a single trajectory against merged PTA."""
    trajectory_id: str
    task_name: str
    model_name: str
    passed: bool  # Ground truth (is_passed)
    
    # The two core metrics from PTA matcher
    structural_coverage: float  # LCS-based path coverage
    process_coverage: float     # % of required tools present
    
    # New SDK metrics (from MatchMetrics)
    weighted_score: float = 0.0        # Stage-importance-weighted coverage (0-100)
    stage_completeness: float = 0.0    # Fraction of GT stages covered (0-1) → stored as 0-100%
    workflow_similarity: float = 0.0   # Fingerprint similarity (0-1) → stored as 0-100%
    coherence_score: float = 0.0       # Trajectory coherence (0-1) → stored as 0-100%
    temporal_profile_score: float = 0.0  # Temporal stage profile similarity (0-1) → stored as 0-100%
    bottleneck_coverage: float = 0.0    # Min per-stage coverage (0-100)
    stage_coverage: Dict[str, float] = field(default_factory=dict)  # Per-stage coverage %
    
    # Additional metrics
    terminal_match: bool = False
    matched_states: int = 0
    total_states: int = 0
    
    # Predicted verdict from matcher
    predicted_verdict: str = ""
    
    # For holdout experiments
    is_train: bool = False
    
    error: str = ""
    
    @property
    def is_passed(self) -> bool:
        """Alias for consistency with holdout experiment."""
        return self.passed
    
    @property
    def combined_score(self) -> float:
        """Weighted combination of all available metrics (0-100 scale).

        The weights are optimised via grid search on the holdout
        experiment.  After the intent-labeling improvements (terminal
        commands are no longer over-classified as verification),
        **coherence** and **temporal profile** become the strongest
        discriminators — they capture forward-progress patterns and
        temporal stage alignment which differ meaningfully between
        passing and failing trajectories.

        ``coherence_score`` and ``temporal_profile_score`` are stored
        on 0-1 scale; they are converted to 0-100 here.
        """
        return (
            0.20 * self.structural_coverage +
            0.15 * self.process_coverage +
            0.30 * (self.coherence_score * 100.0) +
            0.35 * (self.temporal_profile_score * 100.0)
        )


# ============================================================================
# VERDICT COMPUTATION
# ============================================================================

def compute_verdict(result: Dict[str, Any]) -> str:
    """Compute a pass/fail verdict from matching scores.

    Uses the new SDK metrics (weighted_score, stage_completeness,
    workflow_similarity) when available, falling back to legacy
    coverage-based heuristics for backward compatibility.

    Args:
        result: Dict with keys from MatchMetrics.to_dict(), plus optional
                'process_coverage' and 'missing_tools'.

    Returns:
        One of 'PASS', 'LIKELY PASS', 'UNCERTAIN', 'LIKELY FAIL', 'FAIL'.
    """
    weighted = result.get("weighted_score", 0.0)
    completeness = result.get("stage_completeness", 0.0)  # 0-1 scale
    wf_sim = result.get("workflow_similarity", 0.0)        # 0-1 scale
    coverage = result.get("coverage_percent", 0.0)
    terminal = result.get("terminal_state_match", False)
    process_coverage = result.get("process_coverage", 1.0)
    missing_tools = result.get("missing_tools", [])

    # Use intent-stage-aware scoring when available (weighted > 0 implies
    # intent labels exist on the ground-truth trace).
    if weighted > 0 or completeness > 0:
        # Primary signal: weighted_score (0-100) incorporates stage importance
        if weighted >= 75 and completeness >= 0.75:
            return "PASS"
        if weighted >= 60 and completeness >= 0.5:
            return "LIKELY PASS" if terminal else "UNCERTAIN"
        if weighted < 30 or completeness < 0.25:
            return "FAIL"
        if weighted < 50:
            return "LIKELY FAIL"
        return "UNCERTAIN"

    # Legacy fallback for traces without intent-stage labels
    if coverage < 40 and len(missing_tools) >= 2:
        return "FAIL"

    if process_coverage < 1.0:
        if len(missing_tools) >= 2:
            return "LIKELY FAIL"
        elif coverage < 60:
            return "LIKELY FAIL"
        else:
            return "UNCERTAIN"

    if coverage >= 80:
        return "PASS"

    if coverage >= 60:
        return "LIKELY PASS" if terminal else "UNCERTAIN"

    if coverage >= 40:
        return "UNCERTAIN"

    return "LIKELY FAIL"


# ============================================================================
# CORE METRICS COMPUTATION
# ============================================================================

def compute_classification_metrics(
    results: List[Any],
    score_field: str,
    passed_field: str = "passed",
    fixed_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute classification metrics for a list of scored results.
    
    Args:
        results: List of objects with score and passed/is_passed attributes
        score_field: Name of the score field to use as predictor
        passed_field: Name of the boolean field indicating ground truth
        fixed_threshold: If provided, use this threshold instead of calculating optimal.
                        Value should be between 0-100 (percentage).
        
    Returns:
        Dict with AUROC, KS-stat, threshold, accuracy, F1, confusion matrix, etc.
    """
    if not STATS_AVAILABLE:
        return {"error": "Required packages (sklearn, scipy, numpy) not available"}
    
    # Filter out errors
    valid = [r for r in results if not getattr(r, 'error', None)]
    
    if len(valid) < 2:
        return {"error": "Not enough valid results"}
    
    # Extract values - handle both 'passed' and 'is_passed' field names
    def get_passed(r):
        if hasattr(r, passed_field):
            return getattr(r, passed_field)
        if hasattr(r, 'is_passed'):
            return r.is_passed
        if hasattr(r, 'passed'):
            return r.passed
        return False
    
    y_true = np.array([1 if get_passed(r) else 0 for r in valid])
    y_scores = np.array([getattr(r, score_field) for r in valid])
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        return {"error": "Only one class present (all passed or all failed)"}
    
    # Separate scores by class
    pass_scores = y_scores[y_true == 1]
    fail_scores = y_scores[y_true == 0]
    
    metrics = {}
    
    # AUROC
    try:
        metrics['auroc'] = float(roc_auc_score(y_true, y_scores))
    except Exception as e:
        metrics['auroc'] = 0.5
        metrics['auroc_error'] = str(e)
    
    # KS-Statistic
    try:
        ks_stat, ks_pvalue = ks_2samp(pass_scores, fail_scores)
        metrics['ks_statistic'] = float(ks_stat)
        metrics['ks_pvalue'] = float(ks_pvalue)
    except Exception as e:
        metrics['ks_statistic'] = 0.0
        metrics['ks_pvalue'] = 1.0
        metrics['ks_error'] = str(e)
    
    # ROC curve and optimal threshold (Youden's J)
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        auto_threshold = float(thresholds[optimal_idx])
        # Clamp to valid range (avoid inf)
        auto_threshold = min(max(auto_threshold, 0.0), 100.0)
        metrics['auto_threshold'] = auto_threshold
        metrics['optimal_tpr'] = float(tpr[optimal_idx])
        metrics['optimal_fpr'] = float(fpr[optimal_idx])
        
        # Use fixed threshold if provided, otherwise use auto-calculated
        if fixed_threshold is not None:
            optimal_threshold = fixed_threshold
            metrics['optimal_threshold'] = optimal_threshold
            metrics['threshold_mode'] = 'fixed'
        else:
            optimal_threshold = auto_threshold
            metrics['optimal_threshold'] = optimal_threshold
            metrics['threshold_mode'] = 'auto'
    except Exception as e:
        optimal_threshold = fixed_threshold if fixed_threshold is not None else 50.0
        metrics['optimal_threshold'] = optimal_threshold
        metrics['threshold_mode'] = 'fixed' if fixed_threshold is not None else 'fallback'
        metrics['threshold_error'] = str(e)
    
    # Classification metrics at optimal threshold
    try:
        y_pred = (y_scores >= optimal_threshold).astype(int)
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['tp'] = int(tp)
        metrics['fp'] = int(fp)
        metrics['tn'] = int(tn)
        metrics['fn'] = int(fn)
    except Exception as e:
        metrics['accuracy'] = 0.0
        metrics['f1'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['classification_error'] = str(e)
    
    # Score distribution stats
    metrics['pass_mean'] = float(np.mean(pass_scores))
    metrics['pass_std'] = float(np.std(pass_scores))
    metrics['fail_mean'] = float(np.mean(fail_scores))
    metrics['fail_std'] = float(np.std(fail_scores))
    metrics['num_passed'] = int(len(pass_scores))
    metrics['num_failed'] = int(len(fail_scores))
    
    return metrics


def fill_metric_set(metric_set: MetricSet, metrics: Dict[str, Any]) -> None:
    """Populate a MetricSet from computed metrics dict."""
    if 'error' in metrics:
        metric_set.error = metrics['error']
        return
    
    metric_set.auroc = metrics.get('auroc', 0.0)
    metric_set.ks_statistic = metrics.get('ks_statistic', 0.0)
    metric_set.ks_pvalue = metrics.get('ks_pvalue', 1.0)
    metric_set.optimal_threshold = metrics.get('optimal_threshold', 0.0)
    metric_set.accuracy = metrics.get('accuracy', 0.0)
    metric_set.f1 = metrics.get('f1', 0.0)
    metric_set.precision = metrics.get('precision', 0.0)
    metric_set.recall = metrics.get('recall', 0.0)
    metric_set.tp = metrics.get('tp', 0)
    metric_set.fp = metrics.get('fp', 0)
    metric_set.tn = metrics.get('tn', 0)
    metric_set.fn = metrics.get('fn', 0)
    metric_set.pass_mean = metrics.get('pass_mean', 0.0)
    metric_set.pass_std = metrics.get('pass_std', 0.0)
    metric_set.fail_mean = metrics.get('fail_mean', 0.0)
    metric_set.fail_std = metrics.get('fail_std', 0.0)


def compute_all_metric_sets(
    results: List[TrajectoryScore]
) -> Dict[str, MetricSet]:
    """
    Compute MetricSet for all score types.
    
    Args:
        results: List of TrajectoryScore objects
        
    Returns:
        Dict with keys 'structural', 'process', 'weighted', 'stage_completeness',
        'workflow_similarity', 'combined'
    """
    metric_sets = {}
    
    # Direct fields — compute_classification_metrics works on any attribute name
    for key in ('structural_coverage', 'process_coverage', 'weighted_score',
                'stage_completeness', 'workflow_similarity', 'combined_score'):
        short = key.replace('_coverage', '').replace('_score', '').replace('_similarity', '')
        # stage_completeness → stage_completeness, combined_score → combined, etc.
        label = {
            'structural_coverage': 'structural',
            'process_coverage': 'process',
            'weighted_score': 'weighted',
            'stage_completeness': 'stage_completeness',
            'workflow_similarity': 'workflow_similarity',
            'combined_score': 'combined',
        }[key]
        
        metric_sets[label] = MetricSet()
        
        if key == 'combined_score':
            # combined_score is a property — need temp objects with it as an attribute
            @dataclass
            class _Tmp:
                passed: bool
                combined_score: float
                error: str = ""
            
            temps = [_Tmp(r.passed, r.combined_score, r.error) for r in results]
            metrics = compute_classification_metrics(temps, "combined_score")
        elif key == 'stage_completeness':
            # stage_completeness is 0-1; scale to 0-100 for consistent threshold semantics
            @dataclass
            class _TmpSC:
                passed: bool
                stage_completeness_pct: float
                error: str = ""
            
            temps_sc = [_TmpSC(r.passed, r.stage_completeness * 100.0, r.error) for r in results]
            metrics = compute_classification_metrics(temps_sc, "stage_completeness_pct")
        elif key == 'workflow_similarity':
            # workflow_similarity is 0-1; scale to 0-100 for consistent threshold semantics
            @dataclass
            class _TmpWF:
                passed: bool
                workflow_similarity_pct: float
                error: str = ""
            
            temps_wf = [_TmpWF(r.passed, r.workflow_similarity * 100.0, r.error) for r in results]
            metrics = compute_classification_metrics(temps_wf, "workflow_similarity_pct")
        else:
            metrics = compute_classification_metrics(results, key)
        
        fill_metric_set(metric_sets[label], metrics)
    
    return metric_sets


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def get_html_styles() -> str:
    """Return common CSS styles for HTML reports."""
    return """
        :root {
            --bg-primary: #1a1a2e; --bg-secondary: #16213e; --bg-card: #0f3460;
            --text-primary: #eee; --text-secondary: #aaa;
            --accent: #e94560; --success: #4ade80; --warning: #fbbf24; --info: #60a5fa;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg-primary); color: var(--text-primary); line-height: 1.6; padding: 2rem; }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { font-size: 2rem; color: var(--accent); margin-bottom: 0.5rem; }
        h2 { font-size: 1.5rem; margin: 2rem 0 1rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; }
        h3 { font-size: 1.2rem; margin: 1.5rem 0 0.5rem; color: var(--info); }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; }
        .metric-card { background: var(--bg-card); padding: 1.2rem; border-radius: 12px; text-align: center; }
        .metric-value { font-size: 2rem; font-weight: bold; color: var(--accent); }
        .metric-value.good { color: var(--success); }
        .metric-label { font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.3rem; }
        .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 1rem; margin: 1rem 0; }
        .charts-grid-2x2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1rem 0; }
        .chart-container { background: var(--bg-secondary); padding: 1rem; border-radius: 12px; }
        table { width: 100%; border-collapse: collapse; margin: 1rem 0; background: var(--bg-secondary); border-radius: 8px; overflow: hidden; font-size: 0.9rem; }
        th, td { padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid var(--bg-card); }
        th { background: var(--bg-card); color: var(--accent); }
        tr:hover { background: var(--bg-card); }
        .timestamp { color: var(--text-secondary); font-size: 0.9rem; }
        .confusion-matrix { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2px; max-width: 350px; margin: 1rem auto; background: var(--bg-card); padding: 1rem; border-radius: 8px; }
        .cm-cell { padding: 0.8rem; text-align: center; background: var(--bg-secondary); }
        .cm-header { font-weight: bold; color: var(--accent); }
        .cm-tp { background: rgba(74, 222, 128, 0.3); }
        .cm-tn { background: rgba(74, 222, 128, 0.2); }
        .cm-fp { background: rgba(233, 69, 96, 0.3); }
        .cm-fn { background: rgba(233, 69, 96, 0.2); }
        .tab-container { margin: 1rem 0; }
        .tab-buttons { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
        .tab-btn { padding: 0.6rem 1.2rem; background: var(--bg-card); border: none; color: var(--text-secondary); cursor: pointer; border-radius: 6px; transition: all 0.2s; font-size: 0.9rem; }
        .tab-btn:hover { background: var(--bg-secondary); color: var(--text-primary); }
        .tab-btn.active { background: var(--accent); color: var(--bg-primary); }
        .tab-content { display: none; animation: fadeIn 0.3s ease; }
        .tab-content.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    """


def generate_confusion_matrix_html(metric_set: MetricSet, title: str) -> str:
    """Generate HTML for a single confusion matrix."""
    return f"""
        <div>
            <h3 style="text-align: center;">{title}</h3>
            <p style="text-align: center; color: #fbbf24; font-size: 0.85rem; margin: 0.25rem 0;">Threshold: {metric_set.optimal_threshold:.1f}%</p>
            <div class="confusion-matrix">
                <div class="cm-cell cm-header"></div>
                <div class="cm-cell cm-header">Pred P</div>
                <div class="cm-cell cm-header">Pred F</div>
                <div class="cm-cell cm-header">Act P</div>
                <div class="cm-cell cm-tp">{metric_set.tp}</div>
                <div class="cm-cell cm-fn">{metric_set.fn}</div>
                <div class="cm-cell cm-header">Act F</div>
                <div class="cm-cell cm-fp">{metric_set.fp}</div>
                <div class="cm-cell cm-tn">{metric_set.tn}</div>
            </div>
        </div>
    """


def generate_ablation_table_html(
    structural: MetricSet,
    process: MetricSet,
    combined: MetricSet
) -> str:
    """Generate HTML table comparing all three score types."""
    return f"""
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Structural</th>
                    <th>Process</th>
                    <th>Combined</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>AUROC</strong></td>
                    <td>{structural.auroc:.4f}</td>
                    <td>{process.auroc:.4f}</td>
                    <td>{combined.auroc:.4f}</td>
                </tr>
                <tr>
                    <td><strong>KS-Statistic</strong></td>
                    <td>{structural.ks_statistic:.4f}</td>
                    <td>{process.ks_statistic:.4f}</td>
                    <td>{combined.ks_statistic:.4f}</td>
                </tr>
                <tr>
                    <td><strong>Opt. Threshold</strong></td>
                    <td>{structural.optimal_threshold:.1f}%</td>
                    <td>{process.optimal_threshold:.1f}%</td>
                    <td>{combined.optimal_threshold:.1f}%</td>
                </tr>
                <tr>
                    <td><strong>Accuracy</strong></td>
                    <td>{structural.accuracy:.1%}</td>
                    <td>{process.accuracy:.1%}</td>
                    <td>{combined.accuracy:.1%}</td>
                </tr>
                <tr>
                    <td><strong>F1 Score</strong></td>
                    <td>{structural.f1:.4f}</td>
                    <td>{process.f1:.4f}</td>
                    <td>{combined.f1:.4f}</td>
                </tr>
                <tr>
                    <td><strong>Precision</strong></td>
                    <td>{structural.precision:.4f}</td>
                    <td>{process.precision:.4f}</td>
                    <td>{combined.precision:.4f}</td>
                </tr>
                <tr>
                    <td><strong>Recall</strong></td>
                    <td>{structural.recall:.4f}</td>
                    <td>{process.recall:.4f}</td>
                    <td>{combined.recall:.4f}</td>
                </tr>
            </tbody>
        </table>
    """


def generate_histogram_js(
    div_id: str,
    pass_scores: List[float],
    fail_scores: List[float],
    title: str,
    threshold: float = None
) -> str:
    """Generate Plotly.js code for a histogram."""
    shapes = ""
    if threshold is not None:
        shapes = f"""
            shapes: [{{ type: 'line', x0: {threshold}, x1: {threshold}, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#fbbf24', width: 2, dash: 'dash' }} }}]
        """
    
    return f"""
        Plotly.newPlot('{div_id}', [
            {{ x: {json.dumps(pass_scores)}, name: 'Passed', type: 'histogram', opacity: 0.7, marker: {{ color: '#4ade80' }}, xbins: {{ size: 5 }} }},
            {{ x: {json.dumps(fail_scores)}, name: 'Failed', type: 'histogram', opacity: 0.7, marker: {{ color: '#e94560' }}, xbins: {{ size: 5 }} }}
        ], {{
            title: '{title}',
            barmode: 'overlay',
            xaxis: {{ title: 'Score %', range: [0, 100] }},
            yaxis: {{ title: 'Count' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {{ color: '#eee' }},
            {shapes}
        }});
    """


def extract_score_distributions(results: List[TrajectoryScore]) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract score distributions for all score types.
    
    Returns:
        Dict with 'structural', 'process', 'weighted', 'stage_completeness',
        'workflow_similarity', 'combined' keys, each containing 'pass' and 'fail' lists.
    """
    valid = [r for r in results if not r.error]
    
    return {
        'structural': {
            'pass': [r.structural_coverage for r in valid if r.passed],
            'fail': [r.structural_coverage for r in valid if not r.passed],
        },
        'process': {
            'pass': [r.process_coverage for r in valid if r.passed],
            'fail': [r.process_coverage for r in valid if not r.passed],
        },
        'weighted': {
            'pass': [r.weighted_score for r in valid if r.passed],
            'fail': [r.weighted_score for r in valid if not r.passed],
        },
        'stage_completeness': {
            'pass': [r.stage_completeness * 100.0 for r in valid if r.passed],
            'fail': [r.stage_completeness * 100.0 for r in valid if not r.passed],
        },
        'workflow_similarity': {
            'pass': [r.workflow_similarity * 100.0 for r in valid if r.passed],
            'fail': [r.workflow_similarity * 100.0 for r in valid if not r.passed],
        },
        'combined': {
            'pass': [r.combined_score for r in valid if r.passed],
            'fail': [r.combined_score for r in valid if not r.passed],
        },
    }


def format_auroc_color(auroc: float) -> str:
    """Return CSS color based on AUROC value."""
    if auroc >= 0.7:
        return "var(--success)"
    elif auroc >= 0.6:
        return "var(--warning)"
    else:
        return "var(--accent)"
