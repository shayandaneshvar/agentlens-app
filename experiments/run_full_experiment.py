#!/usr/bin/env python3
"""
Full Scale PTA Experiment Runner

This script processes all trajectory data from the evaluation platform dataset:
1. Iterates through all task folders
2. Extracts and generates PTAs from each trajectory ZIP file
3. Merges passed PTAs within each task to create ground truth
4. Generates comprehensive logs and summary report

Usage:
    python run_full_experiment.py <data_root> [options]
    
Examples:
    # Run full experiment on all tasks
    python run_full_experiment.py "C:\\path\\to\\evaluation platform-trajectory-data"
    
    # Run on specific tasks
    python run_full_experiment.py "C:\\path\\to\\evaluation platform-trajectory-data" --tasks python_refactor,chat_mode_simple
    
    # Quick test on first 3 tasks
    python run_full_experiment.py "C:\\path\\to\\evaluation platform-trajectory-data" --limit 3
    
    # Skip merging, only generate individual PTAs
    python run_full_experiment.py "C:\\path\\to\\evaluation platform-trajectory-data" --no-merge
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from swe_trace_sdk import trace as trace_api
from swe_trace_sdk.models import Trace, State, Transition

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class TrajectoryResult:
    """Result of processing a single trajectory."""
    trajectory_id: str
    task_name: str
    model_name: str
    pass_fail: str
    success: bool
    num_states: int = 0
    num_transitions: int = 0
    pta_file: str = ""
    error: str = ""
    processing_time: float = 0.0


@dataclass
class TaskResult:
    """Result of processing a task (multiple trajectories)."""
    task_name: str
    total_trajectories: int = 0
    passed_trajectories: int = 0
    failed_trajectories: int = 0
    successful_pta_generations: int = 0
    failed_pta_generations: int = 0
    merge_success: bool = False
    merged_pta_states: int = 0
    merged_pta_transitions: int = 0
    merged_pta_file: str = ""
    trajectory_results: List[TrajectoryResult] = field(default_factory=list)
    processing_time: float = 0.0
    error: str = ""
    # Equivalence stats from merging
    equivalence_stats: Dict[str, int] = field(default_factory=dict)


@dataclass 
class ExperimentSummary:
    """Overall experiment summary."""
    experiment_name: str
    start_time: str
    end_time: str
    total_duration_seconds: float
    data_root: str
    output_root: str
    
    # Overall counts
    total_tasks: int = 0
    total_trajectories: int = 0
    total_passed: int = 0
    total_failed: int = 0
    
    # Success counts
    successful_pta_generations: int = 0
    failed_pta_generations: int = 0
    successful_merges: int = 0
    failed_merges: int = 0
    
    # Per-model stats
    model_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Equivalence stats (aggregated across all tasks)
    equivalence_stats: Dict[str, int] = field(default_factory=dict)
    
    # Task results
    task_results: List[TaskResult] = field(default_factory=list)
    
    # Errors
    errors: List[str] = field(default_factory=list)


# ============================================================================
# Helper Functions
# ============================================================================

def setup_logging(log_file: Path, verbose: bool = False) -> logging.Logger:
    """Set up logging to file only (no console output)."""
    # Suppress all console logging from root and other loggers
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # File handler for all debug output
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Configure specific module loggers to write to file but not console
    for name in ['swe_trace_sdk', 'swe_trace_sdk.trace', 'swe_trace_sdk.match', 'swe_trace_sdk.equivalence', 'swe_trace_sdk._generator']:
        mod_logger = logging.getLogger(name)
        mod_logger.setLevel(logging.DEBUG)  # Allow debug for detailed tracing
        mod_logger.handlers.clear()
        mod_logger.addHandler(file_handler)  # Write to same file
        mod_logger.propagate = False
    
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False  # Don't propagate to root logger
    
    logger.addHandler(file_handler)
    
    logger.addHandler(file_handler)
    
    return logger


def parse_trajectory_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse trajectory ZIP filename to extract task, model, and pass/fail status.
    
    Supports both formats:
      {task}-logs-{model}-{pass|fail}.zip            (evaluation platform)
      {task}-logs-{model}-{pass|fail}-{run_id}.zip   (openhands)
    
    Returns: (task_name, model_name, pass_fail_status)
    """
    # Remove .zip extension
    name = filename.replace('.zip', '')
    
    # Split by '-logs-' to separate task from model+status
    if '-logs-' not in name:
        return name, 'unknown', 'unknown'
    
    parts = name.split('-logs-', 1)
    task_name = parts[0]
    rest = parts[1] if len(parts) > 1 else ''
    
    # Match model-pass/fail with optional run ID suffix
    m = re.match(r'^(.+)-(pass|fail)(?:-\d+)?$', rest)
    if m:
        model_name = m.group(1)
        pass_fail = 'passed' if m.group(2) == 'pass' else 'failed'
    else:
        model_name = rest or 'unknown'
        pass_fail = 'unknown'
    
    return task_name, model_name, pass_fail


def discover_tasks(data_root: Path) -> List[str]:
    """Discover all task folders in the data root."""
    tasks = []
    for item in data_root.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it contains ZIP files
            zips = list(item.glob("*.zip"))
            if zips:
                tasks.append(item.name)
    return sorted(tasks)


def discover_trajectories(task_dir: Path) -> List[Path]:
    """Discover all trajectory ZIP files in a task directory."""
    return sorted(task_dir.glob("*.zip"))


def extract_zip(zip_path: Path, extract_dir: Path) -> Optional[Path]:
    """Extract a ZIP file and return the extraction directory."""
    try:
        extract_name = zip_path.stem
        target_dir = extract_dir / extract_name
        
        if target_dir.exists():
            return target_dir
            
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
        
        return target_dir
    except Exception as e:
        return None


def find_trajectory_json(extracted_dir: Path) -> Optional[Tuple[Path, str]]:
    """Find trajectory JSON file and detect format.

    Returns (path, format) where format is ``"chatlog"`` or ``"openhands"``,
    or *None* if no recognised file is found.
    """
    # evaluation platform candidates
    for candidate in [
        extracted_dir / "output" / "vsc-output" / "chat-export-logs.json",
        extracted_dir / "vsc-output" / "chat-export-logs.json",
        extracted_dir / "chat-export-logs.json",
    ]:
        if candidate.exists():
            return candidate, "chatlog"

    # OpenHands candidates
    for candidate in [
        extracted_dir / "output" / "trajectories" / "trajectory_openhands.json",
        extracted_dir / "trajectories" / "trajectory_openhands.json",
        extracted_dir / "trajectory_openhands.json",
    ]:
        if candidate.exists():
            return candidate, "openhands"

    # Recursive fallback
    for path in extracted_dir.rglob("chat-export-logs.json"):
        return path, "chatlog"
    for path in extracted_dir.rglob("trajectory_openhands.json"):
        return path, "openhands"

    return None


# ============================================================================
# Core Processing Functions
# ============================================================================

def process_trajectory(
    zip_path: Path,
    output_dir: Path,
    temp_dir: Path,
    logger: logging.Logger
) -> TrajectoryResult:
    """Process a single trajectory ZIP file."""
    start_time = time.time()
    
    filename = zip_path.name
    task_name, model_name, pass_fail = parse_trajectory_filename(filename)
    
    result = TrajectoryResult(
        trajectory_id=filename,
        task_name=task_name,
        model_name=model_name,
        pass_fail=pass_fail,
        success=False
    )
    
    try:
        # Extract ZIP
        extracted_dir = extract_zip(zip_path, temp_dir)
        if not extracted_dir:
            result.error = "Failed to extract ZIP"
            logger.error(f"  Failed to extract: {filename}")
            return result
        
        # Find trajectory JSON (evaluation platform or openhands)
        found = find_trajectory_json(extracted_dir)
        if not found:
            result.error = "No trajectory JSON found (evaluation platform or openhands)"
            logger.error(f"  No trajectory JSON in: {filename}")
            return result
        logs_file, traj_format = found
        
        # Generate Trace from trajectory using SDK
        pta = trace_api.load(str(logs_file), format=traj_format)
        
        # Add metadata
        pta.metadata["trajectory_id"] = filename
        pta.metadata["task_name"] = task_name
        pta.metadata["model_name"] = model_name
        pta.metadata["pass_fail_status"] = pass_fail
        
        # Save PTA
        safe_name = filename.replace('.zip', '')
        pta_file = output_dir / f"{safe_name}_pta.json"
        pta.save(str(pta_file))
        
        result.success = True
        result.num_states = len(pta.states)
        result.num_transitions = len(pta.transitions)
        result.pta_file = str(pta_file)
        
        logger.debug(f"  Generated PTA: {filename} ({result.num_states} states)")
        
    except Exception as e:
        result.error = str(e)
        logger.error(f"  Error processing {filename}: {e}")
    
    result.processing_time = time.time() - start_time
    return result


def merge_task_ptas(
    task_name: str,
    pta_files: List[Path],
    output_dir: Path,
    logger: logging.Logger,
    use_llm: bool = False
) -> Tuple[bool, int, int, str, Dict[str, int]]:
    """
    Merge multiple PTAs from passed trajectories.
    
    Returns: (success, num_states, num_transitions, merged_file_path, equivalence_stats)
    """
    if len(pta_files) < 2:
        logger.warning(f"  Not enough PTAs to merge for {task_name} ({len(pta_files)})")
        return False, 0, 0, "", {}
    
    try:
        # Load Traces
        ptas = []
        for pf in pta_files:
            pta = trace_api.load(str(pf), format="trace")
            ptas.append((pf.stem, pta))
        
        # Merge using SDK
        merged_pta = trace_api.merge([p[1] for p in ptas], use_llm=use_llm)
        
        # Equivalence stats (SDK merge does not expose internal stats)
        equiv_stats = merged_pta.metadata.get("merge_info", {})
        
        # Save merged PTA
        merged_file = output_dir / f"{task_name}_merged_pta.json"
        merged_pta.save(str(merged_file))
        
        logger.info(f"  Merged {len(ptas)} PTAs -> {len(merged_pta.states)} states")
        logger.info(f"  Equivalence stats: checks={equiv_stats.get('total_checks', 0)}, "
                   f"llm_calls={equiv_stats.get('llm_calls', 0)}, "
                   f"llm_skipped={equiv_stats.get('llm_skipped', 0)}, "
                   f"cache_hits={equiv_stats.get('cache_hits', 0)}")
        
        return True, len(merged_pta.states), len(merged_pta.transitions), str(merged_file), equiv_stats
        
    except Exception as e:
        logger.error(f"  Merge failed for {task_name}: {e}")
        return False, 0, 0, "", {}


def process_task(
    task_dir: Path,
    output_root: Path,
    temp_root: Path,
    logger: logging.Logger,
    do_merge: bool = True,
    use_llm: bool = False
) -> TaskResult:
    """Process all trajectories in a task directory."""
    start_time = time.time()
    task_name = task_dir.name
    
    result = TaskResult(task_name=task_name)
    
    # Create output directory for this task
    task_output_dir = output_root / task_name / "pta_outputs"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for this task
    task_temp_dir = temp_root / task_name
    task_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover trajectories
    trajectories = discover_trajectories(task_dir)
    result.total_trajectories = len(trajectories)
    
    logger.info(f"Processing task: {task_name} ({len(trajectories)} trajectories)")
    
    # Process each trajectory
    passed_pta_files = []
    
    for zip_path in trajectories:
        traj_result = process_trajectory(
            zip_path, task_output_dir, task_temp_dir, logger
        )
        result.trajectory_results.append(traj_result)
        
        if traj_result.success:
            result.successful_pta_generations += 1
            if traj_result.pass_fail == 'passed':
                result.passed_trajectories += 1
                passed_pta_files.append(Path(traj_result.pta_file))
            else:
                result.failed_trajectories += 1
        else:
            result.failed_pta_generations += 1
    
    # Merge passed PTAs
    if do_merge and len(passed_pta_files) >= 2:
        logger.info(f"  Merging {len(passed_pta_files)} passed PTAs...")
        success, states, transitions, merged_file, equiv_stats = merge_task_ptas(
            task_name, passed_pta_files, task_output_dir, logger, use_llm
        )
        result.merge_success = success
        result.merged_pta_states = states
        result.merged_pta_transitions = transitions
        result.merged_pta_file = merged_file
        result.equivalence_stats = equiv_stats
    
    result.processing_time = time.time() - start_time
    
    # Clean up temp directory for this task
    try:
        shutil.rmtree(task_temp_dir)
    except:
        pass
    
    return result


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment(
    data_root: Path,
    output_root: Path,
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    do_merge: bool = True,
    use_llm: bool = False,
    verbose: bool = False
) -> ExperimentSummary:
    """Run the full experiment."""
    
    start_time = datetime.now()
    
    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = output_root / f"experiment_{start_time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file, verbose)
    
    logger.info("=" * 60)
    logger.info("FULL SCALE PTA EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Start time: {start_time}")
    
    # Create temp directory
    temp_root = Path(tempfile.mkdtemp(prefix="pta_experiment_"))
    logger.info(f"Temp directory: {temp_root}")
    
    # Discover tasks
    all_tasks = discover_tasks(data_root)
    logger.info(f"Discovered {len(all_tasks)} tasks")
    
    # Filter tasks if specified
    if tasks:
        all_tasks = [t for t in all_tasks if t in tasks]
        logger.info(f"Filtered to {len(all_tasks)} specified tasks")
    
    # Limit if specified
    if limit:
        all_tasks = all_tasks[:limit]
        logger.info(f"Limited to first {limit} tasks")
    
    # Initialize summary
    summary = ExperimentSummary(
        experiment_name="full_scale_pta_experiment",
        start_time=start_time.isoformat(),
        end_time="",
        total_duration_seconds=0,
        data_root=str(data_root),
        output_root=str(output_root),
        total_tasks=len(all_tasks)
    )
    
    # Process tasks
    logger.info(f"\nProcessing {len(all_tasks)} tasks...")
    
    # Print header
    print(f"\n{'='*60}")
    print(f"PTA EXPERIMENT")
    print(f"{'='*60}")
    print(f"Data: {data_root}")
    print(f"Tasks: {len(all_tasks)} | LLM: {'enabled' if use_llm else 'disabled'}")
    print(f"{'='*60}\n")
    
    # Create progress bar
    if HAS_TQDM:
        task_pbar = tqdm(all_tasks, desc="Processing tasks", unit="task", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        task_pbar = all_tasks
        print("Processing tasks (install tqdm for progress bar)...")
    
    for task_name in task_pbar:
        if HAS_TQDM:
            task_pbar.set_postfix_str(task_name[:30])
        task_dir = data_root / task_name
        
        try:
            task_result = process_task(
                task_dir, output_root, temp_root, logger,
                do_merge=do_merge, use_llm=use_llm
            )
            summary.task_results.append(task_result)
            
            # Update summary counts
            summary.total_trajectories += task_result.total_trajectories
            summary.total_passed += task_result.passed_trajectories
            summary.total_failed += task_result.failed_trajectories
            summary.successful_pta_generations += task_result.successful_pta_generations
            summary.failed_pta_generations += task_result.failed_pta_generations
            
            if task_result.merge_success:
                summary.successful_merges += 1
            elif do_merge and task_result.passed_trajectories >= 2:
                summary.failed_merges += 1
            
            # Aggregate equivalence stats
            for key, value in task_result.equivalence_stats.items():
                if key not in summary.equivalence_stats:
                    summary.equivalence_stats[key] = 0
                summary.equivalence_stats[key] += value
            
            # Update model stats
            for traj in task_result.trajectory_results:
                model = traj.model_name
                if model not in summary.model_stats:
                    summary.model_stats[model] = {
                        'total': 0, 'passed': 0, 'failed': 0,
                        'pta_success': 0, 'pta_failed': 0
                    }
                summary.model_stats[model]['total'] += 1
                if traj.pass_fail == 'passed':
                    summary.model_stats[model]['passed'] += 1
                else:
                    summary.model_stats[model]['failed'] += 1
                if traj.success:
                    summary.model_stats[model]['pta_success'] += 1
                else:
                    summary.model_stats[model]['pta_failed'] += 1
            
        except Exception as e:
            error_msg = f"Task {task_name} failed: {e}"
            logger.error(error_msg)
            summary.errors.append(error_msg)
    
    # Finalize summary
    end_time = datetime.now()
    summary.end_time = end_time.isoformat()
    summary.total_duration_seconds = (end_time - start_time).total_seconds()
    
    # Clean up temp directory
    try:
        shutil.rmtree(temp_root)
        logger.info(f"Cleaned up temp directory")
    except:
        pass
    
    # Save summary
    summary_file = output_root / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")
    
    # Print final summary
    print_summary(summary)
    
    return summary


def print_summary(summary: ExperimentSummary):
    """Print a formatted summary of the experiment."""
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Duration: {summary.total_duration_seconds:.1f} seconds")
    print(f"Tasks processed: {summary.total_tasks}")
    print(f"Total trajectories: {summary.total_trajectories}")
    print(f"  - Passed: {summary.total_passed}")
    print(f"  - Failed: {summary.total_failed}")
    print(f"\nPTA Generation:")
    print(f"  - Successful: {summary.successful_pta_generations}")
    print(f"  - Failed: {summary.failed_pta_generations}")
    print(f"\nMerging:")
    print(f"  - Successful merges: {summary.successful_merges}")
    print(f"  - Failed merges: {summary.failed_merges}")
    
    # Equivalence stats
    if summary.equivalence_stats:
        print(f"\nEquivalence Checking:")
        print(f"  - Total checks: {summary.equivalence_stats.get('total_checks', 0)}")
        print(f"  - Exact matches: {summary.equivalence_stats.get('exact_matches', 0)}")
        print(f"  - Heuristic matches: {summary.equivalence_stats.get('heuristic_matches', 0)}")
        print(f"  - Cache hits: {summary.equivalence_stats.get('cache_hits', 0)}")
        # Show line overlap and scope stats (separate equiv vs not-equiv)
        line_eq = summary.equivalence_stats.get('line_overlap_equiv', 0)
        line_neq = summary.equivalence_stats.get('line_overlap_not_equiv', 0)
        if line_eq or line_neq:
            print(f"  - Line overlap decisions: {line_eq} equiv, {line_neq} not-equiv")
        scope_eq = summary.equivalence_stats.get('scope_equiv', 0)
        scope_neq = summary.equivalence_stats.get('scope_not_equiv', 0)
        if scope_eq or scope_neq:
            print(f"  - Scope match decisions: {scope_eq} equiv, {scope_neq} not-equiv")
        if summary.equivalence_stats.get('semantic_content_matches', 0) > 0:
            print(f"  - Semantic content matches: {summary.equivalence_stats['semantic_content_matches']}")
        if summary.equivalence_stats.get('terminal_matches', 0) > 0:
            print(f"  - Terminal command matches: {summary.equivalence_stats['terminal_matches']}")
        print(f"  - LLM calls: {summary.equivalence_stats.get('llm_calls', 0)}")
        if summary.equivalence_stats.get('llm_skipped', 0) > 0:
            print(f"  - LLM skipped (disabled): {summary.equivalence_stats.get('llm_skipped', 0)}")
    
    if summary.model_stats:
        print(f"\nModel Statistics:")
        for model, stats in sorted(summary.model_stats.items()):
            print(f"  {model}:")
            print(f"    Total: {stats['total']} | Pass: {stats['passed']} | Fail: {stats['failed']}")
    
    if summary.errors:
        print(f"\nErrors ({len(summary.errors)}):")
        for err in summary.errors[:5]:
            print(f"  - {err}")
        if len(summary.errors) > 5:
            print(f"  ... and {len(summary.errors) - 5} more")
    
    print(f"\nOutput: {summary.output_root}")
    print(f"{'='*60}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run full-scale PTA experiment on trajectory data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('data_root',
                       help='Root directory containing task folders with trajectory ZIPs')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory (default: <data_root>/experiment_outputs)')
    parser.add_argument('--tasks', '-t', default=None,
                       help='Comma-separated list of task names to process')
    parser.add_argument('--limit', '-n', type=int, default=None,
                       help='Limit to first N tasks (for testing)')
    parser.add_argument('--no-merge', action='store_true',
                       help='Skip merging step, only generate individual PTAs')
    parser.add_argument('--use-llm', action='store_true',
                       help='Use LLM for state equivalence during merging')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = data_root / "experiment_outputs"
    
    # Parse tasks
    tasks = None
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',')]
    
    # Run experiment
    summary = run_experiment(
        data_root=data_root,
        output_root=output_root,
        tasks=tasks,
        limit=args.limit,
        do_merge=not args.no_merge,
        use_llm=args.use_llm,
        verbose=args.verbose
    )
    
    # Exit with error if there were failures
    if summary.failed_pta_generations > 0 or summary.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
