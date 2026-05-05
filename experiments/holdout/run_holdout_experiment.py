#!/usr/bin/env python3
"""
Holdout Validation Experiment - Real-world PTA discrimination test.

This script performs a rigorous train/test split validation:
1. For tasks with mixed pass/fail (at least 1 failed, multiple passed)
2. Split passed trajectories into train (for merging) and test (held-out)
3. Build merged PTA from train passed trajectories ONLY
4. Match held-out passed + all failed trajectories against merged PTA
5. Compute correlation metrics (AUROC, KS, F1, etc.)

This avoids overfitting by not using the same trajectories for building
and evaluating the merged PTA.

Usage:
    python run_holdout_experiment.py <data_dir>
    python run_holdout_experiment.py <data_dir> --train-ratio 0.5
    python run_holdout_experiment.py <data_dir> --output-dir <output_dir>
"""

import sys
import os
import json
import argparse
import logging
import random
import re
import zipfile
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
from swe_trace_sdk.match import (
    extract_required_tools, check_process_coverage,
    extract_required_files, check_file_coverage,
)
from swe_trace_sdk.intent import label_trace_intents

# Import shared utilities from parent experiments/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))
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

# For backward compatibility
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrajectoryInfo:
    """Information about a trajectory."""
    trajectory_id: str
    task_name: str
    model_name: str
    passed: bool
    trajectory_path: str  # Path to chat-export-logs.json, trajectory_openhands.json, or zip
    trajectory_format: str = "chatlog"  # "chatlog" or "openhands"
    pta_path: str = ""    # Path to generated PTA (if exists)


# HoldoutResult is replaced by the shared TrajectoryScore from metrics_utils.
# TrajectoryScore already has: structural_coverage, process_coverage,
# weighted_score, stage_completeness, workflow_similarity, stage_coverage,
# terminal_match, predicted_verdict, is_train, error, plus is_passed property.
HoldoutResult = TrajectoryScore  # backward-compat alias


@dataclass
class TaskHoldoutResult:
    """Holdout experiment result for a single task."""
    task_name: str
    num_train_passed: int
    num_test_passed: int
    num_failed: int
    merged_pta_states: int
    merged_pta_transitions: int
    
    # Per-score-type metrics: {score_name: {metric_name: value}}
    # Populated for all score types in SCORE_TYPES
    score_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    trajectory_results: List[Dict] = field(default_factory=list)
    error: str = ""
    
    # ── Backward-compatible accessors ─────────────────────────────
    def _sm(self, label: str, key: str, default=0.0):
        return self.score_metrics.get(label, {}).get(key, default)
    
    @property
    def test_structural_auroc(self) -> float: return self._sm('structural', 'auroc')
    @property
    def test_structural_ks_statistic(self) -> float: return self._sm('structural', 'ks_statistic')
    @property
    def test_structural_accuracy(self) -> float: return self._sm('structural', 'accuracy')
    @property
    def test_structural_f1(self) -> float: return self._sm('structural', 'f1')
    @property
    def test_structural_pass_mean(self) -> float: return self._sm('structural', 'pass_mean')
    @property
    def test_structural_pass_std(self) -> float: return self._sm('structural', 'pass_std')
    @property
    def test_structural_fail_mean(self) -> float: return self._sm('structural', 'fail_mean')
    @property
    def test_structural_fail_std(self) -> float: return self._sm('structural', 'fail_std')
    @property
    def test_combined_auroc(self) -> float: return self._sm('combined', 'auroc')
    @property
    def test_combined_ks_statistic(self) -> float: return self._sm('combined', 'ks_statistic')
    @property
    def test_combined_accuracy(self) -> float: return self._sm('combined', 'accuracy')
    @property
    def test_combined_f1(self) -> float: return self._sm('combined', 'f1')
    @property
    def test_combined_pass_mean(self) -> float: return self._sm('combined', 'pass_mean')
    @property
    def test_combined_pass_std(self) -> float: return self._sm('combined', 'pass_std')
    @property
    def test_combined_fail_mean(self) -> float: return self._sm('combined', 'fail_mean')
    @property
    def test_combined_fail_std(self) -> float: return self._sm('combined', 'fail_std')
    @property
    def test_weighted_auroc(self) -> float: return self._sm('weighted', 'auroc')
    @property
    def test_process_auroc(self) -> float: return self._sm('process', 'auroc')
    @property
    def test_process_optimal_threshold(self) -> float: return self._sm('process', 'optimal_threshold')
    @property
    def test_structural_optimal_threshold(self) -> float: return self._sm('structural', 'optimal_threshold')
    @property
    def test_combined_optimal_threshold(self) -> float: return self._sm('combined', 'optimal_threshold')


# All score types to evaluate — maps label → (attribute, needs_scaling)
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
class AggregateHoldoutResult:
    """Aggregate results across all tasks."""
    total_tasks: int
    total_train_passed: int
    total_test_passed: int
    total_failed: int
    
    # Dict-based: {score_label: MetricSet} for all score types
    micro_metrics: Dict[str, MetricSet] = field(default_factory=dict)
    macro_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Confusion matrix (from structural for backward compat)
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    
    # ── Backward-compatible accessors ─────────────────────────────
    def _micro(self, label: str) -> MetricSet:
        return self.micro_metrics.get(label, MetricSet())
    
    @property
    def micro_structural(self) -> MetricSet: return self._micro('structural')
    @property
    def micro_process(self) -> MetricSet: return self._micro('process')
    @property
    def micro_combined(self) -> MetricSet: return self._micro('combined')
    @property
    def macro_structural_auroc(self) -> float:
        return self.macro_metrics.get('structural', {}).get('auroc', 0.0)
    @property
    def macro_structural_ks_statistic(self) -> float:
        return self.macro_metrics.get('structural', {}).get('ks_statistic', 0.0)
    @property
    def macro_process_auroc(self) -> float:
        return self.macro_metrics.get('process', {}).get('auroc', 0.0)
    @property
    def macro_process_ks_statistic(self) -> float:
        return self.macro_metrics.get('process', {}).get('ks_statistic', 0.0)


def find_trajectory_file(traj_dir: Path) -> Optional[Tuple[str, str]]:
    """Find trajectory JSON in a directory.

    Checks for both evaluation platform (``chat-export-logs.json``) and OpenHands
    (``trajectory_openhands.json``) formats.

    Returns
    -------
    tuple[str, str] | None
        ``(path, format)`` where *format* is ``"chatlog"`` or ``"openhands"``,
        or *None* if no trajectory file is found.
    """
    # --- evaluation platform candidates ---
    evaluation platform_candidates = [
        traj_dir / "output" / "vsc-output" / "chat-export-logs.json",
        traj_dir / "vsc-output" / "chat-export-logs.json",
        traj_dir / "chat-export-logs.json",
    ]
    for c in evaluation platform_candidates:
        if c.exists():
            return str(c), "chatlog"

    # --- openhands candidates ---
    openhands_candidates = [
        traj_dir / "output" / "trajectories" / "trajectory_openhands.json",
        traj_dir / "trajectories" / "trajectory_openhands.json",
        traj_dir / "trajectory_openhands.json",
    ]
    for c in openhands_candidates:
        if c.exists():
            return str(c), "openhands"

    # --- recursive fallback ---
    for found in traj_dir.rglob("chat-export-logs.json"):
        return str(found), "chatlog"
    for found in traj_dir.rglob("trajectory_openhands.json"):
        return str(found), "openhands"

    return None


def extract_zip(zip_path: Path, extract_dir: Path) -> Optional[Path]:
    """Extract a ZIP file and return the extraction directory."""
    try:
        extract_name = zip_path.stem
        target_dir = extract_dir / extract_name
        
        # Skip if already extracted
        if target_dir.exists():
            return target_dir
        
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
        
        return target_dir
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return None


def get_trajectory_json_path(traj_info: 'TrajectoryInfo', temp_dir: Path) -> Optional[Tuple[str, str]]:
    """
    Get the path to the trajectory JSON, extracting ZIP if necessary.

    Supports both evaluation platform (``chat-export-logs.json``) and OpenHands
    (``trajectory_openhands.json``) formats.

    Args:
        traj_info: Trajectory information
        temp_dir: Directory for extracting ZIPs

    Returns:
        ``(json_path, format)`` or *None* if not found.
    """
    traj_path = Path(traj_info.trajectory_path)

    # If it's a ZIP file, extract it
    if traj_path.suffix == '.zip':
        extracted_dir = extract_zip(traj_path, temp_dir)
        if not extracted_dir:
            return None
        return find_trajectory_file(extracted_dir)

    # If it's already a JSON file
    elif traj_path.suffix == '.json':
        if traj_path.exists():
            fmt = traj_info.trajectory_format
            return str(traj_path), fmt
        return None

    # If it's a directory
    elif traj_path.is_dir():
        return find_trajectory_file(traj_path)

    return None


def discover_trajectories(data_dir: Path) -> Dict[str, List[TrajectoryInfo]]:
    """
    Discover all trajectories organized by task.
    
    Expects structure:
        data_dir/
            task_name/
                task_name-logs-model-pass/
                task_name-logs-model-fail/
                or .zip files
    
    Returns:
        Dict mapping task_name -> list of TrajectoryInfo
    """
    tasks = {}
    
    for task_dir in data_dir.iterdir():
        if not task_dir.is_dir():
            continue
        if task_dir.name.startswith('.') or task_dir.name.startswith('_'):
            continue
        
        task_name = task_dir.name
        trajectories = []
        
        # Find trajectory directories or zips
        for item in task_dir.iterdir():
            if item.name.startswith('.') or item.name.startswith('_'):
                continue
            
            # Parse trajectory name to extract model and pass/fail
            name = item.stem if item.suffix == '.zip' else item.name
            
            # Expected format: task_name-logs-model_name-pass/fail
            if '-logs-' not in name:
                continue
            
            parts = name.split('-logs-')
            if len(parts) != 2:
                continue
            
            rest = parts[1]
            # Support both "model-pass" and "model-pass-12345" (with run ID suffix)
            m = re.match(r'^(.+)-(pass|fail)(?:-\d+)?$', rest)
            if not m:
                continue
            model_name = m.group(1)
            passed = m.group(2) == 'pass'
            
            # Find trajectory file
            traj_format = "chatlog"  # default
            if item.is_dir():
                found = find_trajectory_file(item)
                if not found:
                    continue
                traj_path, traj_format = found
            elif item.suffix == '.zip':
                traj_path = str(item)
            else:
                continue

            trajectories.append(TrajectoryInfo(
                trajectory_id=name,
                task_name=task_name,
                model_name=model_name,
                passed=passed,
                trajectory_path=traj_path,
                trajectory_format=traj_format,
            ))
        
        if trajectories:
            tasks[task_name] = trajectories
    
    return tasks


def split_trajectories(
    trajectories: List[TrajectoryInfo],
    merge_count: int,
    test_pass_count: int,
    seed: int = 42
) -> Tuple[List[TrajectoryInfo], List[TrajectoryInfo], List[TrajectoryInfo]]:
    """
    Split trajectories into train passed, test passed, and failed.
    
    Args:
        trajectories: All trajectories for a task
        merge_count: Number of passed trajectories to use for merging
        test_pass_count: Number of passed trajectories to hold out for testing
        seed: Random seed for reproducibility
        
    Returns:
        (train_passed, test_passed, failed)
    """
    passed = [t for t in trajectories if t.passed]
    failed = [t for t in trajectories if not t.passed]
    
    # Shuffle with seed for reproducibility
    rng = random.Random(seed)
    passed_shuffled = passed.copy()
    rng.shuffle(passed_shuffled)
    
    # Take exactly merge_count for training, rest for test
    train_passed = passed_shuffled[:merge_count]
    test_passed = passed_shuffled[merge_count:merge_count + test_pass_count]
    
    return train_passed, test_passed, failed


# In-memory cache: trajectory_id → Trace object (avoids re-loading from disk/zip)
_trace_cache: Dict[str, Trace] = {}


def generate_pta_from_trajectory(traj_info: TrajectoryInfo, output_dir: Path, temp_dir: Path) -> Optional[Trace]:
    """Generate trace from a trajectory, extracting ZIP if necessary.

    Uses a two-level cache:
      1. In-memory ``_trace_cache`` keyed by trajectory_id (fastest).
      2. On-disk PTA JSON — if a prior run already saved the file, load it
         directly instead of re-extracting the ZIP and re-parsing the raw log.
    """
    tid = traj_info.trajectory_id

    # ── Level 1: in-memory cache ──────────────────────────────────
    if tid in _trace_cache:
        # Make sure traj_info.pta_path is populated for downstream code
        if not traj_info.pta_path:
            pta_path = output_dir / f"{tid}_pta.json"
            if pta_path.exists():
                traj_info.pta_path = str(pta_path)
        return _trace_cache[tid]

    try:
        # ── Level 2: on-disk PTA cache ────────────────────────────
        pta_path = output_dir / f"{tid}_pta.json"
        if pta_path.exists():
            pta = trace_api.load(str(pta_path), format="trace")
            traj_info.pta_path = str(pta_path)
            _trace_cache[tid] = pta
            return pta

        # ── Cold path: extract ZIP / load raw JSON ────────────────
        result = get_trajectory_json_path(traj_info, temp_dir)
        if not result:
            logger.error(f"Could not find trajectory JSON for {tid}")
            return None
        json_path, fmt = result
        traj_info.trajectory_format = fmt

        pta = trace_api.load(json_path, format=fmt)

        # Save trace for future runs / cache hits
        pta.save(str(pta_path))
        traj_info.pta_path = str(pta_path)
        _trace_cache[tid] = pta

        return pta
    except Exception as e:
        logger.error(f"Failed to generate trace for {tid}: {e}")
        return None


def merge_ptas(ptas: List[Trace], use_llm: bool = False) -> Optional[Trace]:
    """Merge multiple traces into one."""
    if not ptas:
        return None
    if len(ptas) == 1:
        return ptas[0]
    
    try:
        merged = trace_api.merge(ptas, use_llm=use_llm)
        return merged
    except Exception as e:
        logger.error(f"Failed to merge traces: {e}")
        return None


def compute_scores(
    merged_pta: Trace,
    test_trajectories: List[TrajectoryInfo],
    temp_dir: Path,
    use_llm: bool = False,
) -> Tuple[List[HoldoutResult], Dict[str, Any]]:
    """Compute scores for test trajectories against the merged ground truth.
    
    Returns:
        Tuple of (results list, equivalence stats dict)
    """
    results = []
    all_equiv_stats: Dict[str, Any] = {}
    
    # Extract required tools from merged ground truth
    required_tools = extract_required_tools(merged_pta)
    required_files = extract_required_files(merged_pta)
    
    for traj_info in test_trajectories:
        try:
            # Load trajectory trace (should already be generated)
            if traj_info.pta_path and Path(traj_info.pta_path).exists():
                traj_trace = trace_api.load(traj_info.pta_path, format="trace")
            else:
                # Generate on the fly - need to get the JSON path
                result = get_trajectory_json_path(traj_info, temp_dir)
                if not result:
                    raise ValueError(f"Could not find trajectory JSON for {traj_info.trajectory_id}")
                json_path, fmt = result
                traj_trace = trace_api.load(json_path, format=fmt)
            
            # Use SDK match — returns all metrics (coverage, weighted, stages, workflow, precision, F1)
            label_trace_intents(traj_trace)  # Label candidate intent stages for workflow similarity
            match_result = match.run(traj_trace, merged_pta, use_llm=use_llm)
            m = match_result.metrics  # MatchMetrics
            
            # Aggregate equivalence stats
            for k, v in match_result.equivalence_stats.items():
                all_equiv_stats[k] = all_equiv_stats.get(k, 0) + v
            
            # Debug logging for non-discriminating cases
            if os.environ.get('DEBUG_COVERAGE'):
                logger.info(f"  [{traj_info.trajectory_id}] recall={m.coverage_percent:.1f}% precision={m.precision_percent:.1f}% "
                           f"f1={m.f1_score:.1f}% weighted={m.weighted_score:.1f}% "
                           f"stages={m.stage_completeness:.2f} passed={traj_info.passed}")
            
            # Compute process coverage (tool-level + file-level)
            proc_cov = 100.0
            missing_tools = []
            if required_tools:
                proc_ratio, missing_tools = check_process_coverage(traj_trace, required_tools)
                proc_cov = proc_ratio * 100.0
            
            # Compute file-level coverage (which specific files were edited)
            file_cov = 100.0
            if required_files:
                file_ratio, _ = check_file_coverage(traj_trace, required_files)
                file_cov = file_ratio * 100.0
            
            # Use file coverage as the process metric — it's far more
            # discriminating than tool-name coverage since it checks
            # whether the candidate edited the *right files*, not just
            # whether it used the right tool names.
            effective_proc = file_cov
            
            # Compute verdict using all available metrics
            verdict_input = {
                'coverage_percent': m.f1_score,  # Use F1 for verdict
                'terminal_state_match': m.terminal_state_match,
                'process_coverage': effective_proc / 100.0,
                'missing_tools': missing_tools,
                'weighted_score': m.weighted_score,
                'stage_completeness': m.stage_completeness,
                'workflow_similarity': m.workflow_similarity,
            }
            verdict = compute_verdict(verdict_input)
            
            # Use F1 score as the structural_coverage metric —
            # it balances recall (GT states found) with precision
            # (fraction of candidate states that matched), penalising
            # verbose failing trajectories.
            results.append(HoldoutResult(
                trajectory_id=traj_info.trajectory_id,
                task_name=traj_info.task_name,
                model_name=traj_info.model_name,
                passed=traj_info.passed,
                structural_coverage=round(m.f1_score, 2),
                process_coverage=round(effective_proc, 2),
                weighted_score=round(m.weighted_score, 2),
                stage_completeness=round(m.stage_completeness, 4),
                workflow_similarity=round(m.workflow_similarity, 4),
                coherence_score=round(m.coherence_score, 4),
                temporal_profile_score=round(m.temporal_profile_score, 4),
                bottleneck_coverage=round(m.bottleneck_coverage, 2),
                stage_coverage=m.stage_coverage,
                terminal_match=m.terminal_state_match,
                matched_states=m.matched_count,
                total_states=m.total_ground_truth_states,
                predicted_verdict=verdict,
                is_train=False,
            ))
            
        except Exception as e:
            logger.error(f"Error scoring {traj_info.trajectory_id}: {e}")
            results.append(TrajectoryScore(
                trajectory_id=traj_info.trajectory_id,
                task_name=traj_info.task_name,
                model_name=traj_info.model_name,
                passed=traj_info.passed,
                structural_coverage=0.0,
                process_coverage=0.0,
                terminal_match=False,
                predicted_verdict="ERROR",
                is_train=False,
                error=str(e)
            ))
    
    return results, all_equiv_stats


# Note: compute_classification_metrics is imported from metrics_utils
# It handles both 'passed' and 'is_passed' field names


def run_task_holdout(
    task_name: str,
    trajectories: List[TrajectoryInfo],
    output_dir: Path,
    temp_dir: Path,
    merge_count: int,
    test_pass_count: int,
    test_fail_count: int,
    use_llm: bool = False,
    seed: int = 42,
    structural_threshold: Optional[float] = None,
    process_threshold: Optional[float] = None,
    combined_threshold: Optional[float] = None,
    num_merge_seeds: int = 1,
) -> Optional[TaskHoldoutResult]:
    """Run holdout experiment for a single task.
    
    When num_merge_seeds > 1, builds multiple merged PTAs from different
    random subsets of passed trajectories and averages the per-trajectory
    scores to reduce variance from merge subset selection.
    """
    
    # Split trajectories — fixes the test set
    train_passed, test_passed, failed = split_trajectories(
        trajectories, merge_count=merge_count, test_pass_count=test_pass_count, seed=seed
    )
    
    # Check if we have enough data
    if len(train_passed) < merge_count:
        logger.warning(f"Skipping {task_name}: not enough passed for merging ({len(train_passed)} < {merge_count})")
        return None
    if len(test_passed) < test_pass_count:
        logger.warning(f"Skipping {task_name}: not enough held-out passed trajectories ({len(test_passed)} < {test_pass_count})")
        return None
    if len(failed) < test_fail_count:
        logger.warning(f"Skipping {task_name}: not enough failed trajectories ({len(failed)} < {test_fail_count})")
        return None
    
    # Limit failed to test_fail_count for balanced evaluation
    rng = random.Random(seed)
    failed_shuffled = failed.copy()
    rng.shuffle(failed_shuffled)
    failed = failed_shuffled[:test_fail_count]
    
    task_output_dir = output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate PTAs for ALL passed trajectories (needed for resampling)
    all_passed = [t for t in trajectories if t.passed]
    for traj in all_passed:
        if not (traj.pta_path and Path(traj.pta_path).exists()):
            generate_pta_from_trajectory(traj, task_output_dir, temp_dir)
    
    # Generate PTAs for test set (both passed and failed)
    test_trajectories = test_passed + failed
    for traj in test_trajectories:
        if not (traj.pta_path and Path(traj.pta_path).exists()):
            pta = generate_pta_from_trajectory(traj, task_output_dir, temp_dir)
            if pta is None:
                logger.warning(f"Skipping {task_name}: failed to load test trajectory {traj.trajectory_id}")
                return None
    
    # Build the merge pool: all passed except those in test_passed
    test_passed_ids = {t.trajectory_id for t in test_passed}
    merge_pool = [t for t in all_passed if t.trajectory_id not in test_passed_ids]
    
    # Determine merge subsets to try
    merge_subsets: List[List[TrajectoryInfo]] = []
    if num_merge_seeds <= 1:
        # Original behaviour: use the train_passed from split_trajectories
        merge_subsets.append(train_passed)
    else:
        # First subset is the original split (seed=42)
        merge_subsets.append(train_passed)
        # Additional subsets: random samples from the merge pool
        resample_rng = random.Random(seed + 1000)
        for _ in range(num_merge_seeds - 1):
            subset = resample_rng.sample(merge_pool, min(merge_count, len(merge_pool)))
            merge_subsets.append(subset)
    
    logger.info(f"Task {task_name}: merge_pool={len(merge_pool)}, test_passed={len(test_passed)}, "
                f"failed={len(failed)}, num_merge_seeds={len(merge_subsets)}")
    
    # Score test trajectories against each merged PTA and accumulate
    # Per-trajectory accumulators: trajectory_id -> list of score dicts
    score_accum: Dict[str, List[Dict[str, float]]] = {t.trajectory_id: [] for t in test_trajectories}
    last_equiv_stats: Dict[str, Any] = {}
    last_merged_pta = None
    
    for si, subset in enumerate(merge_subsets):
        # Load PTAs for this merge subset
        subset_ptas = []
        for traj in subset:
            if traj.pta_path and Path(traj.pta_path).exists():
                pta = trace_api.load(traj.pta_path, format="trace")
            else:
                pta = generate_pta_from_trajectory(traj, task_output_dir, temp_dir)
            if pta:
                subset_ptas.append(pta)
        
        if len(subset_ptas) < merge_count:
            logger.warning(f"  Seed {si}: only {len(subset_ptas)} PTAs, skipping")
            continue
        
        merged_pta = merge_ptas(subset_ptas, use_llm=use_llm)
        if not merged_pta:
            logger.warning(f"  Seed {si}: merge failed, skipping")
            continue
        
        last_merged_pta = merged_pta
        
        # Score test trajectories against this merged PTA
        results_i, equiv_stats_i = compute_scores(merged_pta, test_trajectories, temp_dir, use_llm=use_llm)
        last_equiv_stats = equiv_stats_i
        
        for r in results_i:
            score_accum[r.trajectory_id].append({
                'structural_coverage': r.structural_coverage,
                'process_coverage': r.process_coverage,
                'weighted_score': r.weighted_score,
                'stage_completeness': r.stage_completeness,
                'workflow_similarity': r.workflow_similarity,
                'coherence_score': r.coherence_score,
                'temporal_profile_score': r.temporal_profile_score,
                'bottleneck_coverage': r.bottleneck_coverage,
                'stage_coverage': r.stage_coverage,
                'terminal_match': r.terminal_match,
                'matched_states': r.matched_states,
                'total_states': r.total_states,
                'predicted_verdict': r.predicted_verdict,
            })
    
    if last_merged_pta is None:
        logger.warning(f"Skipping {task_name}: all merge attempts failed")
        return None
    
    # Save last merged PTA
    merged_path = task_output_dir / f"{task_name}_holdout_merged_pta.json"
    last_merged_pta.save(str(merged_path))
    
    # Build final HoldoutResults with averaged scores
    holdout_results = []
    for traj in test_trajectories:
        scores = score_accum[traj.trajectory_id]
        if not scores:
            holdout_results.append(HoldoutResult(
                trajectory_id=traj.trajectory_id,
                task_name=traj.task_name,
                model_name=traj.model_name,
                passed=traj.passed,
                is_train=False,
                structural_coverage=0.0,
                process_coverage=0.0,
                error="no scores computed",
            ))
            continue
        
        n = len(scores)
        avg_struct = sum(s['structural_coverage'] for s in scores) / n
        avg_proc = sum(s['process_coverage'] for s in scores) / n
        avg_weighted = sum(s['weighted_score'] for s in scores) / n
        avg_stage_comp = sum(s['stage_completeness'] for s in scores) / n
        avg_wf_sim = sum(s['workflow_similarity'] for s in scores) / n
        avg_coherence = sum(s['coherence_score'] for s in scores) / n
        avg_temporal = sum(s['temporal_profile_score'] for s in scores) / n
        avg_bottleneck = sum(s['bottleneck_coverage'] for s in scores) / n
        
        # For stage_coverage, terminal_match, verdict: use last seed's result
        last = scores[-1]
        
        holdout_results.append(HoldoutResult(
            trajectory_id=traj.trajectory_id,
            task_name=traj.task_name,
            model_name=traj.model_name,
            passed=traj.passed,
            is_train=False,
            structural_coverage=round(avg_struct, 2),
            process_coverage=round(avg_proc, 2),
            weighted_score=round(avg_weighted, 2),
            stage_completeness=round(avg_stage_comp, 4),
            workflow_similarity=round(avg_wf_sim, 4),
            coherence_score=round(avg_coherence, 4),
            temporal_profile_score=round(avg_temporal, 4),
            bottleneck_coverage=round(avg_bottleneck, 2),
            stage_coverage=last['stage_coverage'],
            terminal_match=last['terminal_match'],
            matched_states=last['matched_states'],
            total_states=last['total_states'],
            predicted_verdict=last['predicted_verdict'],
        ))
    
    equiv_stats = last_equiv_stats
    
    # Log equivalence statistics
    logger.info(f"Task {task_name} equivalence stats: {equiv_stats}")
    
    # Compute task-level metrics
    result = TaskHoldoutResult(
        task_name=task_name,
        num_train_passed=len(train_passed),
        num_test_passed=len(test_passed),
        num_failed=len(failed),
        merged_pta_states=len(last_merged_pta.states),
        merged_pta_transitions=len(last_merged_pta.transitions),
        trajectory_results=[asdict(r) for r in holdout_results],
    )
    
    # Compute metrics for all score types (data-driven)
    from compute_correlation_metrics import _compute_score_metrics
    for label, (attr, needs_scaling) in SCORE_TYPES.items():
        m = _compute_score_metrics(holdout_results, attr, needs_scaling)
        if 'error' not in m:
            result.score_metrics[label] = m
    
    return result


def compute_aggregate_metrics(
    task_results: List[TaskHoldoutResult],
    all_trajectory_results: List[HoldoutResult],
    structural_threshold: Optional[float] = None,
    process_threshold: Optional[float] = None,
    combined_threshold: Optional[float] = None,
) -> AggregateHoldoutResult:
    """Compute aggregate metrics across all tasks with full ablation study."""
    
    valid_tasks = [t for t in task_results if not t.error]
    
    aggregate = AggregateHoldoutResult(
        total_tasks=len(task_results),
        total_train_passed=sum(t.num_train_passed for t in task_results),
        total_test_passed=sum(t.num_test_passed for t in task_results),
        total_failed=sum(t.num_failed for t in task_results),
    )
    
    if not valid_tasks or not STATS_AVAILABLE:
        return aggregate
    
    # Macro-averaged (mean of per-task metrics)
    for label in SCORE_TYPES:
        task_aurocs = [t.score_metrics.get(label, {}).get('auroc', 0.5) for t in valid_tasks]
        task_ks = [t.score_metrics.get(label, {}).get('ks_statistic', 0.0) for t in valid_tasks]
        aggregate.macro_metrics[label] = {
            'auroc': float(np.mean(task_aurocs)),
            'ks_statistic': float(np.mean(task_ks)),
        }
    
    # Micro-averaged (pool all test results)
    test_results = [r for r in all_trajectory_results if not r.is_train]
    
    # Build optional fixed-threshold map
    threshold_map = {
        'structural': structural_threshold,
        'process': process_threshold,
        'combined': combined_threshold,
    }
    
    # Compute micro-averaged MetricSet for each score type (data-driven)
    from compute_correlation_metrics import _compute_score_metrics
    for label, (attr, needs_scaling) in SCORE_TYPES.items():
        ms = MetricSet()
        ft = threshold_map.get(label)
        if ft is not None:
            # If a fixed threshold was provided, use it
            metrics = _compute_score_metrics(test_results, attr, needs_scaling)
        else:
            metrics = _compute_score_metrics(test_results, attr, needs_scaling)
        if 'error' not in metrics:
            fill_metric_set(ms, metrics)
        aggregate.micro_metrics[label] = ms
    
    # Copy confusion matrix from structural metrics for backward compat
    struct_ms = aggregate.micro_metrics.get('structural', MetricSet())
    aggregate.tp = struct_ms.tp
    aggregate.fp = struct_ms.fp
    aggregate.tn = struct_ms.tn
    aggregate.fn = struct_ms.fn
    
    return aggregate


def generate_html_report(
    aggregate: AggregateHoldoutResult,
    task_results: List[TaskHoldoutResult],
    all_results: List[HoldoutResult],
    output_path: str
) -> None:
    """Generate HTML report for holdout experiment (data-driven for all score types)."""
    
    # Prepare chart data for all score types
    test_results = [r for r in all_results if not r.is_train and not r.error]
    
    # Build pass/fail score distributions for every score type
    score_distributions: Dict[str, Dict[str, List[float]]] = {}
    for label, (attr, needs_scaling) in SCORE_TYPES.items():
        scale = 100.0 if needs_scaling else 1.0
        passed_vals = []
        failed_vals = []
        for r in test_results:
            val = getattr(r, attr, 0.0)
            if attr == 'combined_score':
                val = r.combined_score
            val *= scale
            if r.is_passed:
                passed_vals.append(val)
            else:
                failed_vals.append(val)
        score_distributions[label] = {'pass': passed_vals, 'fail': failed_vals}
    
    # Ordered list of labels for consistent rendering
    score_labels = list(SCORE_TYPES.keys())
    
    # ── Build HTML ────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Holdout Validation Experiment Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-primary: #1a1a2e; --bg-secondary: #16213e; --bg-card: #0f3460;
            --text-primary: #eee; --text-secondary: #aaa;
            --accent: #e94560; --success: #4ade80; --warning: #fbbf24; --info: #60a5fa;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg-primary); color: var(--text-primary); line-height: 1.6; padding: 2rem; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ font-size: 2rem; color: var(--accent); margin-bottom: 0.5rem; }}
        h2 {{ font-size: 1.5rem; margin: 2rem 0 1rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; }}
        h3 {{ font-size: 1.2rem; margin: 1.5rem 0 0.5rem; color: var(--info); }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; }}
        .metric-card {{ background: var(--bg-card); padding: 1.2rem; border-radius: 12px; text-align: center; }}
        .metric-value {{ font-size: 2rem; font-weight: bold; color: var(--accent); }}
        .metric-value.good {{ color: var(--success); }}
        .metric-label {{ font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.3rem; }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 1rem; margin: 1rem 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; background: var(--bg-secondary); border-radius: 8px; overflow: hidden; font-size: 0.9rem; }}
        th, td {{ padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid var(--bg-card); }}
        th {{ background: var(--bg-card); color: var(--accent); }}
        tr:hover {{ background: var(--bg-card); }}
        .experiment-info {{ background: var(--bg-secondary); padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
        .experiment-info p {{ margin: 0.3rem 0; }}
        .timestamp {{ color: var(--text-secondary); font-size: 0.9rem; }}
        .confusion-matrix {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 2px; max-width: 350px; margin: 1rem auto; background: var(--bg-card); padding: 1rem; border-radius: 8px; }}
        .cm-cell {{ padding: 0.8rem; text-align: center; background: var(--bg-secondary); }}
        .cm-header {{ font-weight: bold; color: var(--accent); }}
        .cm-tp {{ background: rgba(74, 222, 128, 0.3); }}
        .cm-tn {{ background: rgba(74, 222, 128, 0.2); }}
        .cm-fp {{ background: rgba(233, 69, 96, 0.3); }}
        .cm-fn {{ background: rgba(233, 69, 96, 0.2); }}
        .tab-container {{ margin: 1rem 0; }}
        .tab-buttons {{ display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }}
        .tab-btn {{ padding: 0.6rem 1.2rem; background: var(--bg-card); border: none; color: var(--text-secondary); cursor: pointer; border-radius: 6px; transition: all 0.2s; font-size: 0.9rem; }}
        .tab-btn:hover {{ background: var(--bg-secondary); color: var(--text-primary); }}
        .tab-btn.active {{ background: var(--accent); color: var(--bg-primary); }}
        .tab-content {{ display: none; animation: fadeIn 0.3s ease; }}
        .tab-content.active {{ display: block; }}
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Holdout Validation Experiment</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="experiment-info">
            <h3>Experiment Design</h3>
            <p>For each task with mixed pass/fail trajectories:</p>
            <p>1. <strong>Train:</strong> Merge subset of PASSED trajectories into merged PTA</p>
            <p>2. <strong>Test:</strong> Score held-out PASSED + all FAILED against merged PTA</p>
            <p>3. <strong>Evaluate:</strong> Can merged PTA discriminate held-out pass from fail?</p>
        </div>
        
        <h2>📊 Dataset Summary</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-value">{aggregate.total_tasks}</div>
                <div class="metric-label">Tasks</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: var(--info)">{aggregate.total_train_passed}</div>
                <div class="metric-label">Train (Passed)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: var(--success)">{aggregate.total_test_passed}</div>
                <div class="metric-label">Test (Passed)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: var(--accent)">{aggregate.total_failed}</div>
                <div class="metric-label">Test (Failed)</div>
            </div>
        </div>
        
        <h2>🎯 Ablation Study: All Metrics for Each Score Type</h2>
        <p style="color: #aaa; margin-bottom: 1rem;">Compares {len(score_labels)} scoring methods across classification metrics. Higher values indicate better discrimination.</p>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
"""
    
    # Ablation table headers — one column per score type
    for label in score_labels:
        html += f"                    <th>{label.replace('_', ' ').title()}</th>\n"
    html += "                </tr>\n            </thead>\n            <tbody>\n"
    
    # Ablation table rows
    row_defs = [
        ('AUROC',          'auroc',             '{:.4f}'),
        ('KS-Statistic',   'ks_statistic',      '{:.4f}'),
        ('Opt. Threshold',  'optimal_threshold', '{:.1f}%'),
        ('Accuracy',       'accuracy',           '{:.1%}'),
        ('F1 Score',       'f1',                 '{:.4f}'),
        ('Precision',      'precision',          '{:.4f}'),
        ('Recall',         'recall',             '{:.4f}'),
    ]
    for row_label, attr_name, fmt in row_defs:
        html += f"                <tr><td><strong>{row_label}</strong></td>"
        for label in score_labels:
            ms = aggregate.micro_metrics.get(label, MetricSet())
            val = getattr(ms, attr_name, 0.0)
            html += f"<td>{fmt.format(val)}</td>"
        html += "</tr>\n"
    html += "            </tbody>\n        </table>\n"
    
    # ── Confusion Matrices ────────────────────────────────────────
    html += """
        <h2>📈 Confusion Matrices</h2>
        <p style="color: #aaa; margin-bottom: 1rem;">Binary classification results at each score type's optimal threshold.</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem;">
"""
    for label in score_labels:
        ms = aggregate.micro_metrics.get(label, MetricSet())
        html += f"""
            <div>
                <h3 style="text-align: center;">{label.replace('_', ' ').title()}</h3>
                <p style="text-align: center; color: #fbbf24; font-size: 0.85rem;">Threshold: {ms.optimal_threshold:.1f}%</p>
                <div class="confusion-matrix">
                    <div class="cm-cell cm-header"></div>
                    <div class="cm-cell cm-header">Pred P</div>
                    <div class="cm-cell cm-header">Pred F</div>
                    <div class="cm-cell cm-header">Act P</div>
                    <div class="cm-cell cm-tp">{ms.tp}</div>
                    <div class="cm-cell cm-fn">{ms.fn}</div>
                    <div class="cm-cell cm-header">Act F</div>
                    <div class="cm-cell cm-fp">{ms.fp}</div>
                    <div class="cm-cell cm-tn">{ms.tn}</div>
                </div>
            </div>
"""
    html += "        </div>\n"
    
    # ── Score Distribution Histograms ─────────────────────────────
    html += """
        <h2>📊 Score Distributions (Test Set)</h2>
        <p style="color: #aaa; margin-bottom: 1rem;">Histograms comparing score distributions for passed (green) vs failed (red) trajectories. Dashed yellow line = optimal threshold.</p>
        <div class="charts-grid">
"""
    for label in score_labels:
        html += f'            <div class="chart-container" style="background: var(--bg-secondary); padding: 1rem; border-radius: 12px;"><div id="hist_{label}"></div></div>\n'
    html += "        </div>\n"
    
    # ── Ablation Bar Chart + Box Plots ────────────────────────────
    html += """
        <h2>📈 Ablation Study: Metric Comparison</h2>
        <div class="charts-grid">
            <div class="chart-container" style="background: var(--bg-secondary); padding: 1rem; border-radius: 12px;"><div id="ablation_bars"></div></div>
            <div class="chart-container" style="background: var(--bg-secondary); padding: 1rem; border-radius: 12px;"><div id="boxplot_all"></div></div>
        </div>
"""
    
    # ── Diagnostic section ────────────────────────────────────────
    non_discriminating = [t for t in task_results if abs(t.test_combined_auroc - 0.5) <= 0.01]
    inverse_correlation = [t for t in task_results if t.test_combined_auroc < 0.5]
    
    if non_discriminating or inverse_correlation:
        html += """
        <h2>⚠️ Diagnostic: Problematic Tasks</h2>
        <p style="color: #aaa; margin-bottom: 1rem;">Tasks showing poor discrimination between pass/fail trajectories.</p>
"""
        if inverse_correlation:
            html += """
        <h3 style="color: #e94560;">🔄 Inverse Correlation (AUROC &lt; 0.5)</h3>
        <table>
            <thead><tr><th>Task</th><th>AUROC</th><th>Pass μ</th><th>Fail μ</th><th>Gap</th></tr></thead>
            <tbody>
"""
            for t in sorted(inverse_correlation, key=lambda x: x.test_combined_auroc):
                p_mean = t._sm('structural', 'pass_mean')
                f_mean = t._sm('structural', 'fail_mean')
                gap = f_mean - p_mean
                html += f"""
                <tr>
                    <td>{t.task_name}</td>
                    <td style="color: var(--accent)">{t.test_combined_auroc:.3f}</td>
                    <td>{p_mean:.1f}%</td>
                    <td>{f_mean:.1f}%</td>
                    <td style="color: {'var(--accent)' if gap > 0 else 'var(--success)'}">{gap:+.1f}%</td>
                </tr>
"""
            html += "</tbody></table>"
        
        if non_discriminating:
            html += """
        <h3 style="color: #fbbf24;">🔀 No Discrimination (AUROC ≈ 0.5)</h3>
        <table>
            <thead><tr><th>Task</th><th>Pass Count</th><th>Fail Count</th><th>Coverage</th></tr></thead>
            <tbody>
"""
            for t in non_discriminating:
                html += f"""
                <tr>
                    <td>{t.task_name}</td>
                    <td>{t.num_test_passed}</td>
                    <td>{t.num_failed}</td>
                    <td>{t._sm('structural', 'pass_mean'):.1f}%</td>
                </tr>
"""
            html += "</tbody></table>"
    
    # ── Per-Task Results (tab per score type) ─────────────────────
    html += """
        <h2>📋 Per-Task Results</h2>
        <p style="color: #888; margin-bottom: 1rem;">Click a score type tab to see per-task AUROC, KS, Accuracy, F1, and distributions.</p>
        <div class="tab-container">
            <div class="tab-buttons">
"""
    for i, label in enumerate(score_labels):
        active = ' active' if i == 0 else ''
        html += f'                <button class="tab-btn{active}" onclick="showTab(\'{label}\', this)">{label.replace("_", " ").title()}</button>\n'
    html += "            </div>\n"
    
    for i, label in enumerate(score_labels):
        active = ' active' if i == 0 else ''
        html += f"""
            <div id="tab-{label}" class="tab-content{active}">
                <table>
                    <thead>
                        <tr>
                            <th>Task</th>
                            <th>Train/Test</th>
                            <th>Failed</th>
                            <th>AUROC</th>
                            <th>KS-Stat</th>
                            <th>Accuracy</th>
                            <th>F1</th>
                            <th>Pass μ±σ</th>
                            <th>Fail μ±σ</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        sorted_tasks = sorted(task_results, key=lambda t: t.score_metrics.get(label, {}).get('auroc', 0.5), reverse=True)
        for t in sorted_tasks:
            sm = t.score_metrics.get(label, {})
            auroc = sm.get('auroc', 0.5)
            ks = sm.get('ks_statistic', 0.0)
            acc = sm.get('accuracy', 0.0)
            f1 = sm.get('f1', 0.0)
            pmean = sm.get('pass_mean', 0.0)
            pstd = sm.get('pass_std', 0.0)
            fmean = sm.get('fail_mean', 0.0)
            fstd = sm.get('fail_std', 0.0)
            html += f"""
                        <tr>
                            <td>{t.task_name}</td>
                            <td>{t.num_train_passed}/{t.num_test_passed}</td>
                            <td>{t.num_failed}</td>
                            <td style="color: {'var(--success)' if auroc >= 0.7 else 'inherit'}">{auroc:.3f}</td>
                            <td style="color: {'var(--success)' if ks >= 0.4 else 'inherit'}">{ks:.3f}</td>
                            <td style="color: {'var(--success)' if acc >= 0.7 else 'inherit'}">{acc*100:.1f}%</td>
                            <td style="color: {'var(--success)' if f1 >= 0.7 else 'inherit'}">{f1:.3f}</td>
                            <td>{pmean:.1f}±{pstd:.1f}</td>
                            <td>{fmean:.1f}±{fstd:.1f}</td>
                        </tr>
"""
        html += """
                    </tbody>
                </table>
            </div>
"""
    html += "        </div>\n"
    
    # ── Metric Glossary ───────────────────────────────────────────
    html += """
        <h2>📝 Metric Glossary</h2>
        <table>
            <tr><th>Metric</th><th>Good Value</th><th>What It Measures</th></tr>
            <tr><td><strong>AUROC</strong></td><td style="color: var(--success)">≥ 0.7</td><td>Area Under ROC Curve. 0.5 = random, 1.0 = perfect separation.</td></tr>
            <tr><td><strong>KS-Statistic</strong></td><td style="color: var(--success)">≥ 0.4</td><td>Max distance between pass/fail CDFs. Higher = better.</td></tr>
            <tr><td><strong>Accuracy</strong></td><td style="color: var(--success)">≥ 70%</td><td>(TP + TN) / Total at optimal threshold.</td></tr>
            <tr><td><strong>F1 Score</strong></td><td style="color: var(--success)">≥ 0.7</td><td>Harmonic mean of Precision and Recall.</td></tr>
            <tr><td><strong>Pass/Fail μ±σ</strong></td><td>well separated</td><td>Score distributions. Better when means are far apart.</td></tr>
        </table>
    </div>
"""
    
    # ── JavaScript for histograms, ablation chart, box plots ──────
    html += "\n    <script>\n"
    
    # Histogram per score type
    colors_pass = '#4ade80'
    colors_fail = '#e94560'
    for label in score_labels:
        dist = score_distributions[label]
        ms = aggregate.micro_metrics.get(label, MetricSet())
        title = label.replace('_', ' ').title()
        html += f"""
        Plotly.newPlot('hist_{label}', [
            {{ x: {json.dumps(dist['pass'])}, name: 'Passed', type: 'histogram', opacity: 0.7, marker: {{ color: '{colors_pass}' }}, xbins: {{ size: 5 }} }},
            {{ x: {json.dumps(dist['fail'])}, name: 'Failed', type: 'histogram', opacity: 0.7, marker: {{ color: '{colors_fail}' }}, xbins: {{ size: 5 }} }}
        ], {{
            title: '{title} Distribution',
            barmode: 'overlay', xaxis: {{ title: 'Score %', range: [0, 100] }}, yaxis: {{ title: 'Count' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {{ color: '#eee' }},
            shapes: [{{ type: 'line', x0: {ms.optimal_threshold}, x1: {ms.optimal_threshold}, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#fbbf24', width: 2, dash: 'dash' }} }}],
            annotations: [{{ x: {ms.optimal_threshold}, y: 1, yref: 'paper', text: 'Threshold: {ms.optimal_threshold:.1f}%', showarrow: false, font: {{ color: '#fbbf24' }} }}]
        }});
"""
    
    # Ablation bar chart
    bar_metrics = ['auroc', 'ks_statistic', 'f1', 'accuracy']
    bar_colors = ['#60a5fa', '#4ade80', '#fbbf24', '#e94560']
    x_labels_json = json.dumps([l.replace('_', ' ').title() for l in score_labels])
    html += "\n        // Ablation bar chart\n        Plotly.newPlot('ablation_bars', [\n"
    for bm, bc in zip(bar_metrics, bar_colors):
        vals = [getattr(aggregate.micro_metrics.get(l, MetricSet()), bm, 0.0) for l in score_labels]
        html += f"            {{ x: {x_labels_json}, y: {json.dumps([round(v, 4) for v in vals])}, name: '{bm.replace('_', ' ').title()}', type: 'bar', marker: {{ color: '{bc}' }} }},\n"
    html += f"""        ], {{
            title: 'Ablation Study: Metrics Comparison',
            barmode: 'group', yaxis: {{ title: 'Score', range: [0, 1] }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {{ color: '#eee' }}
        }});
"""
    
    # Box plots
    box_colors_pass = ['#4ade80', '#60a5fa', '#fbbf24', '#a78bfa', '#f472b6', '#34d399']
    box_colors_fail = ['#e94560', '#f472b6', '#fb923c', '#c084fc', '#ef4444', '#f87171']
    html += "\n        // Box plots\n        Plotly.newPlot('boxplot_all', [\n"
    for i, label in enumerate(score_labels):
        dist = score_distributions[label]
        title = label.replace('_', ' ').title()
        cp = box_colors_pass[i % len(box_colors_pass)]
        cf = box_colors_fail[i % len(box_colors_fail)]
        html += f"            {{ y: {json.dumps(dist['pass'])}, name: '{title} (Pass)', type: 'box', marker: {{ color: '{cp}' }}, boxpoints: false }},\n"
        html += f"            {{ y: {json.dumps(dist['fail'])}, name: '{title} (Fail)', type: 'box', marker: {{ color: '{cf}' }}, boxpoints: false }},\n"
    html += """        ], {
            title: 'Score Distributions by Metric Type',
            yaxis: { title: 'Score %', range: [0, 100] },
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#eee' }
        });
"""
    
    # Tab switching function
    html += """
        function showTab(tabName, btn) {
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('active');
            });
            document.getElementById('tab-' + tabName).classList.add('active');
            btn.classList.add('active');
        }
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Run holdout validation experiment")
    parser.add_argument("data_dir", help="Directory containing trajectory data")
    parser.add_argument("--output-dir", help="Output directory (default: data_dir/_holdout_experiment)")
    parser.add_argument("--merge-count", type=int, required=True, 
                        help="Minimum number of passed trajectories to use for merging (required, must be >= 2)")
    parser.add_argument("--test-pass-count", type=int, required=True,
                        help="Minimum number of held-out passed trajectories for testing (required, must be >= 1)")
    parser.add_argument("--test-fail-count", type=int, required=True,
                        help="Minimum number of failed trajectories for testing (required, must be >= 1)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for equivalence checks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    # Optional fixed thresholds (if not provided, auto-calculate using Youden's J)
    parser.add_argument("--structural-threshold", type=float, default=None,
                        help="Fixed threshold for structural coverage (0-100). If not set, auto-calculate.")
    parser.add_argument("--process-threshold", type=float, default=None,
                        help="Fixed threshold for process coverage (0-100). If not set, auto-calculate.")
    parser.add_argument("--combined-threshold", type=float, default=None,
                        help="Fixed threshold for combined score (0-100). If not set, auto-calculate.")
    parser.add_argument("--num-merge-seeds", type=int, default=1,
                        help="Number of different merge subsets to ensemble (default: 1 = no ensembling). "
                             "Higher values average scores across multiple merged PTAs to reduce variance.")
    
    args = parser.parse_args()
    
    # Validate minimum requirements
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
    
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "_holdout_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for ZIP extraction
    # Use a short system temp path to avoid Windows MAX_PATH (260 char) issues
    # when extracting ZIPs with deep internal directory structures
    temp_dir = Path(tempfile.mkdtemp(prefix="pta_"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Discovering trajectories in: {data_dir}")
    tasks = discover_trajectories(data_dir)
    print(f"Found {len(tasks)} tasks")
    
    # Filter to tasks meeting strict requirements
    min_passed_required = args.merge_count + args.test_pass_count
    eligible_tasks = {}
    for task_name, trajs in tasks.items():
        passed = sum(1 for t in trajs if t.passed)
        failed = sum(1 for t in trajs if not t.passed)
        if passed >= min_passed_required and failed >= args.test_fail_count:
            eligible_tasks[task_name] = trajs
    
    print(f"Eligible tasks (>= {min_passed_required} passed [{args.merge_count} merge + {args.test_pass_count} test], >= {args.test_fail_count} failed): {len(eligible_tasks)}")
    
    if not eligible_tasks:
        print("ERROR: No eligible tasks found")
        return 1
    
    # Run experiment for each task
    task_results: List[TaskHoldoutResult] = []
    all_trajectory_results: List[HoldoutResult] = []
    
    print(f"\nRunning holdout experiment (merge_count={args.merge_count}, test_pass_count={args.test_pass_count}, test_fail_count={args.test_fail_count})...")
    
    for task_name in tqdm(sorted(eligible_tasks.keys()), desc="Processing tasks"):
        result = run_task_holdout(
            task_name=task_name,
            trajectories=eligible_tasks[task_name],
            output_dir=output_dir,
            temp_dir=temp_dir,
            merge_count=args.merge_count,
            test_pass_count=args.test_pass_count,
            test_fail_count=args.test_fail_count,
            use_llm=args.use_llm,
            seed=args.seed,
            structural_threshold=args.structural_threshold,
            process_threshold=args.process_threshold,
            combined_threshold=args.combined_threshold,
            num_merge_seeds=args.num_merge_seeds,
        )
        
        if result:
            task_results.append(result)
            for tr in result.trajectory_results:
                all_trajectory_results.append(HoldoutResult(**tr))
    
    if not task_results:
        print("ERROR: No tasks completed successfully")
        return 1
    
    print(f"\nCompleted {len(task_results)} tasks")
    
    # Compute aggregate metrics
    aggregate = compute_aggregate_metrics(
        task_results, 
        all_trajectory_results,
        structural_threshold=args.structural_threshold,
        process_threshold=args.process_threshold,
        combined_threshold=args.combined_threshold,
    )
    
    # Print summary
    print("\n" + "="*85)
    print("HOLDOUT VALIDATION EXPERIMENT RESULTS")
    print("="*85)
    print(f"\nDataset: {aggregate.total_tasks} tasks")
    print(f"  Train (passed): {aggregate.total_train_passed}")
    print(f"  Test (passed):  {aggregate.total_test_passed}")
    print(f"  Test (failed):  {aggregate.total_failed}")
    
    # Data-driven ablation table
    score_labels = list(SCORE_TYPES.keys())
    col_w = 16
    
    print(f"\n{'='*85}")
    print("ABLATION STUDY: All Metrics for Each Score Type (Micro-Averaged)")
    print(f"{'='*85}")
    
    header = f"\n{'Metric':<20}" + "".join(f"{l.replace('_',' ').title():>{col_w}}" for l in score_labels)
    print(header)
    print("-" * (20 + col_w * len(score_labels)))
    
    row_defs = [
        ('AUROC',        'auroc',             lambda v: f"{v:>{col_w}.4f}"),
        ('KS-Statistic', 'ks_statistic',      lambda v: f"{v:>{col_w}.4f}"),
        ('Opt. Threshold','optimal_threshold', lambda v: f"{v:>{col_w-1}.1f}%"),
        ('Accuracy',     'accuracy',           lambda v: f"{v:>{col_w-1}.1%}"),
        ('F1 Score',     'f1',                 lambda v: f"{v:>{col_w}.4f}"),
        ('Precision',    'precision',           lambda v: f"{v:>{col_w}.4f}"),
        ('Recall',       'recall',              lambda v: f"{v:>{col_w}.4f}"),
    ]
    for row_label, attr_name, fmt_fn in row_defs:
        cells = ""
        for label in score_labels:
            ms = aggregate.micro_metrics.get(label, MetricSet())
            cells += fmt_fn(getattr(ms, attr_name, 0.0))
        print(f"{row_label:<20}{cells}")
    
    # Macro-averaged AUROC
    print(f"\n{'='*85}")
    print("MACRO-AVERAGED AUROC (mean across tasks)")
    print(f"{'='*85}")
    header = "".join(f"{l.replace('_',' ').title():<{col_w}}" for l in score_labels)
    print(header)
    print("-" * (col_w * len(score_labels)))
    vals = "".join(f"{aggregate.macro_metrics.get(l, {}).get('auroc', 0.0):<{col_w}.4f}" for l in score_labels)
    print(vals)
    
    # Confusion matrices
    print(f"\n{'='*85}")
    print("CONFUSION MATRICES (at optimal threshold)")
    print(f"{'='*85}")
    print(f"\n{'Metric':<20} {'TP':>8} {'FP':>8} {'TN':>8} {'FN':>8}")
    print("-" * 55)
    for label in score_labels:
        ms = aggregate.micro_metrics.get(label, MetricSet())
        print(f"{label.replace('_',' ').title():<20} {ms.tp:>8} {ms.fp:>8} {ms.tn:>8} {ms.fn:>8}")
    
    # Save results
    results = {
        "generated_at": datetime.now().isoformat(),
        "experiment_config": {
            "data_dir": str(data_dir),
            "merge_count": args.merge_count,
            "test_pass_count": args.test_pass_count,
            "test_fail_count": args.test_fail_count,
            "seed": args.seed,
            "use_llm": args.use_llm,
            "num_merge_seeds": args.num_merge_seeds,
        },
        "aggregate_metrics": asdict(aggregate),
        "task_results": [asdict(t) for t in task_results],
    }
    
    results_path = output_dir / "holdout_experiment_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")
    
    # Generate HTML report
    html_path = output_dir / "holdout_experiment_report.html"
    generate_html_report(aggregate, task_results, all_trajectory_results, str(html_path))
    print(f"Generated HTML report: {html_path}")
    
    # Cleanup temp directory
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")
    
    print("\n" + "="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
