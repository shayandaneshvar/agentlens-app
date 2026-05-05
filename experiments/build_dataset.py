"""
Build AgentLens-Bench Dataset
==============================
Combines quality assessment + waste detection into a shareable dataset.

For each eligible task (≥5 passing):
  1. Build merged PTA (k=5, seed=42)
  2. Run match + quality_assessment for ALL non-donor trajectories
  3. Extract complete QualityReport fields (quality_score, coherence, coverage,
     f1, stage_coverage, divergence, etc.)
  4. Run custom waste detections + SDK inefficiency extraction
  5. Output:
     - Tier 1: annotations/trajectories.csv + .parquet (flat per-trajectory)
     - Tier 1: annotations/tasks.csv + .parquet (per-task summary)
     - Tier 2: trajectories/{task}/{traj}.json (individual PTA copies)
     - Tier 2: ground_truth/{task}_merged_pta.json (merged PTA copies)
     - Tier 3: analysis/{task}/{traj}_report.json (full QualityReport)

Usage:
  python experiments/build_dataset.py new_research_experiments/experiment_outputs \\
      -o new_research_experiments/dataset
"""

import argparse
import json
import logging
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdk" / "src"))

from swe_trace_sdk import trace as trace_api
from swe_trace_sdk import match
from swe_trace_sdk.models import State, Trace

# ============================================================================
# Configuration
# ============================================================================

MERGE_K = 5
SEED = 42
TASK_TIMEOUT = 300
IDEAL_MIN = 70
LUCKY_MAX = 47

# ============================================================================
# Helpers
# ============================================================================

def parse_pta_filename(filename: str, task_name: str) -> Optional[Tuple[str, bool, str]]:
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


def get_ordered_states(trace_obj: Trace) -> List[State]:
    states = sorted(trace_obj.states.values(), key=lambda s: s.step)
    states = [s for s in states
              if not (hasattr(s, "log_entry") and s.log_entry
                      and getattr(s.log_entry, "kind", None) == "request")]
    states = [s for s in states if getattr(s, "intent_stage", "")]
    return states


def get_stage(s: State) -> str:
    return getattr(s, "intent_stage", "") or ""


def revised_tier(quality_score: int, passed: bool) -> str:
    if not passed:
        if quality_score >= 40:
            return "partial_fail"
        return "off_track"
    if quality_score >= IDEAL_MIN:
        return "ideal"
    elif quality_score < LUCKY_MAX:
        return "lucky"
    return "solid"


# ============================================================================
# Custom Waste Detection (from run_waste_analysis.py)
# ============================================================================

def detect_regression_loops(cand_states: List[State], gt_states: List[State]) -> Dict[str, Any]:
    gt_regression_positions: Set[int] = set()
    if gt_states:
        gt_stages = [get_stage(s) for s in gt_states]
        for i in range(1, len(gt_stages) - 1):
            if (gt_stages[i] == "implementation" and gt_stages[i - 1] == "exploration"):
                for j in range(i + 1, len(gt_stages)):
                    if gt_stages[j] == "exploration":
                        gt_regression_positions.add(j)
                        break
                    elif gt_stages[j] == "implementation":
                        break

    stages = [get_stage(s) for s in cand_states]
    n = len(stages)
    regression_count = 0
    wasted_steps = 0
    i = 0

    while i < n:
        if stages[i] == "implementation":
            j = i + 1
            while j < n and stages[j] != "exploration":
                j += 1
            if j >= n:
                break
            e_start = j
            k = j
            while k < n and stages[k] != "implementation":
                k += 1
            e_span_len = k - e_start

            if e_span_len >= 2 and k < n:
                prop_pos = e_start / max(n, 1)
                gt_prop_pos = int(prop_pos * len(gt_states)) if gt_states else -1
                is_gt_pattern = gt_states and gt_prop_pos in gt_regression_positions

                if not is_gt_pattern:
                    regression_count += 1
                    wasted_steps += e_span_len

                i = k
                continue
        i += 1

    return {"count": regression_count, "wasted_steps": wasted_steps, "has_any": regression_count > 0}


def detect_redundant_steps(cand_states: List[State], gt_states: List[State]) -> Dict[str, Any]:
    gt_multi_read_files: Set[str] = set()
    if gt_states:
        gt_read_files: Dict[str, int] = defaultdict(int)
        for s in gt_states:
            if get_stage(s) == "exploration" and getattr(s, "file_path", ""):
                gt_read_files[s.file_path] += 1
        gt_multi_read_files = {f for f, c in gt_read_files.items() if c >= 2}

    seen_reads: Dict[str, int] = {}
    redundant_count = 0
    wasted_steps = 0

    for i, s in enumerate(cand_states):
        stg = get_stage(s)
        if stg == "implementation":
            seen_reads.clear()
            continue
        if stg == "exploration":
            fp = getattr(s, "file_path", "") or ""
            if not fp:
                continue
            if fp in gt_multi_read_files:
                seen_reads[fp] = i
                continue
            if fp in seen_reads:
                redundant_count += 1
                wasted_steps += 1
            seen_reads[fp] = i

    return {"count": redundant_count, "wasted_steps": wasted_steps, "has_any": redundant_count > 0}


def detect_unnecessary_exploration(cand_states: List[State], gt_states: List[State], merged_pta: Trace) -> Dict[str, Any]:
    pta_files: Set[str] = set()
    for s in merged_pta.states.values():
        fp = getattr(s, "file_path", "") or ""
        if fp:
            pta_files.add(fp)

    gt_post_impl_explore_files: Set[str] = set()
    if gt_states:
        gt_impl_started = False
        for s in gt_states:
            stg = get_stage(s)
            if stg == "implementation":
                gt_impl_started = True
            if gt_impl_started and stg == "exploration":
                fp = getattr(s, "file_path", "") or ""
                if fp:
                    gt_post_impl_explore_files.add(fp)

    impl_started = False
    unnecessary_count = 0
    wasted_steps = 0

    for s in cand_states:
        stg = get_stage(s)
        if stg == "implementation":
            impl_started = True
            continue
        if impl_started and stg == "exploration":
            fp = getattr(s, "file_path", "") or ""
            if not fp:
                continue
            fp_lower = fp.lower()
            if "test" in fp_lower or "spec" in fp_lower:
                continue
            if fp in pta_files:
                continue
            if fp in gt_post_impl_explore_files:
                continue
            unnecessary_count += 1
            wasted_steps += 1

    return {"count": unnecessary_count, "wasted_steps": wasted_steps, "has_any": unnecessary_count > 0}


def extract_blind_retries(report) -> Dict[str, Any]:
    ineff = report.inefficiencies
    if ineff is None:
        return {"count": 0, "wasted_steps": 0, "has_any": False}
    count = ineff.retry_loop_count
    wasted = sum(max(0, rl.count - 2) for rl in ineff.retry_loops)
    return {"count": count, "wasted_steps": wasted, "has_any": count > 0}


def extract_cyclic_patterns(report) -> Dict[str, Any]:
    ineff = report.inefficiencies
    if ineff is None:
        return {"count": 0, "wasted_steps": 0, "has_any": False}
    count = ineff.cyclic_pattern_count
    wasted = sum(cp.pattern_length * (cp.repetitions - 1) for cp in ineff.cyclic_patterns)
    return {"count": count, "wasted_steps": wasted, "has_any": count > 0}


# ============================================================================
# QualityReport → flat dict
# ============================================================================

def report_to_flat(report, match_result, entry: Dict, n_states: int,
                   regression: Dict, retries: Dict, redundant: Dict,
                   unnecessary: Dict, cyclic: Dict) -> Dict[str, Any]:
    """Convert QualityReport + waste detections to flat record."""
    metrics = report.key_metrics or {}
    # Get precise metrics from MatchResult.metrics
    mm = match_result.metrics if hasattr(match_result, "metrics") and match_result.metrics else None
    ineff = report.inefficiencies

    # Stage coverage
    stage_cov = {}
    if report.stage_coverage:
        for stage_key, detail in report.stage_coverage.items():
            pct = getattr(detail, "coverage_percent", None) or getattr(detail, "matched_percent", 0)
            stage_cov[stage_key] = pct

    # Divergence
    div_step = None
    div_fraction = None
    if report.divergence_point:
        div_step = getattr(report.divergence_point, "step", None)
        if div_step is not None and n_states > 0:
            div_fraction = div_step / n_states

    # Token waste from SDK
    wasted_input = ineff.wasted_input_tokens if ineff else 0
    wasted_output = ineff.wasted_output_tokens if ineff else 0
    total_wasted = (regression["wasted_steps"] + retries["wasted_steps"] +
                    redundant["wasted_steps"] + unnecessary["wasted_steps"] +
                    cyclic["wasted_steps"])
    severity = total_wasted / max(n_states, 1)

    # Failure reasons / strengths
    failure_reasons = []
    if report.failure_reasons:
        for fr in report.failure_reasons:
            failure_reasons.append({
                "reason": getattr(fr, "reason", ""),
                "detail": getattr(fr, "detail", ""),
                "severity": getattr(fr, "severity", ""),
            })

    tier = revised_tier(report.quality_score, entry["passed"])

    return {
        # Identity
        "task_id": entry["task"],
        "model": entry["model"],
        "trajectory_id": entry["trajectory_id"],
        "passed": entry["passed"],
        "n_states": n_states,
        # Quality
        "quality_score": report.quality_score,
        "quality_tier": tier,
        "verdict": getattr(report, "verdict", ""),
        "coverage_percent": mm.coverage_percent if mm else metrics.get("coverage_percent", 0),
        "precision_percent": mm.precision_percent if mm else 0,
        "f1_score": mm.f1_score if mm else 0,
        "coherence_score": mm.coherence_score if mm else metrics.get("coherence", 0),
        "temporal_profile_score": mm.temporal_profile_score if mm else 0,
        "workflow_similarity": mm.workflow_similarity if mm else metrics.get("workflow_similarity", 0),
        "stage_completeness": mm.stage_completeness if mm else metrics.get("stage_completeness", 0),
        "bottleneck_coverage": mm.bottleneck_coverage if mm else 0,
        "weighted_score": mm.weighted_score if mm else 0,
        # Per-stage coverage
        "stage_coverage_E": (mm.stage_coverage.get("exploration", 0) if mm else stage_cov.get("exploration", 0)),
        "stage_coverage_I": (mm.stage_coverage.get("implementation", 0) if mm else stage_cov.get("implementation", 0)),
        "stage_coverage_V": (mm.stage_coverage.get("verification", 0) if mm else stage_cov.get("verification", 0)),
        "stage_coverage_O": (mm.stage_coverage.get("orchestration", 0) if mm else stage_cov.get("orchestration", 0)),
        # Inefficiencies
        "regression_loop_count": regression["count"],
        "regression_loop_waste": regression["wasted_steps"],
        "blind_retry_count": retries["count"],
        "blind_retry_waste": retries["wasted_steps"],
        "redundant_step_count": redundant["count"],
        "redundant_step_waste": redundant["wasted_steps"],
        "unnecessary_exploration_count": unnecessary["count"],
        "unnecessary_exploration_waste": unnecessary["wasted_steps"],
        "cyclic_pattern_count": cyclic["count"],
        "cyclic_pattern_waste": cyclic["wasted_steps"],
        "total_wasted_steps": total_wasted,
        "waste_severity": round(severity, 4),
        "wasted_input_tokens": wasted_input,
        "wasted_output_tokens": wasted_output,
        # Divergence
        "divergence_step": div_step,
        "divergence_fraction": round(div_fraction, 4) if div_fraction is not None else None,
        "stage_order_match": getattr(report, "stage_order_match", None),
        # Failure analysis
        "failure_reasons": json.dumps(failure_reasons) if failure_reasons else None,
        "strengths": json.dumps(report.strengths) if report.strengths else None,
    }


def report_to_json(report, match_result, entry: Dict) -> Dict[str, Any]:
    """Serialize full QualityReport for Tier 3 analysis JSON."""
    # Alignment
    alignment = []
    if match_result.alignment:
        for a in match_result.alignment:
            alignment.append({
                "candidate_step": getattr(a, "candidate_step", None),
                "candidate_state_id": getattr(a, "candidate_state_id", ""),
                "ground_truth_state_id": getattr(a, "ground_truth_state_id", ""),
                "matched": getattr(a, "matched", False),
            })

    # Divergence segments
    div_segments = []
    if report.divergence_points:
        for seg in report.divergence_points:
            div_segments.append({
                "start_step": getattr(seg, "start_step", None),
                "end_step": getattr(seg, "end_step", None),
                "missed_gt_states": getattr(seg, "missed_gt_states", []),
                "candidate_activity": getattr(seg, "candidate_activity", ""),
            })

    # Stage comparison
    stage_comparison = {}
    if report.stage_comparison:
        for stage_key, comp in report.stage_comparison.items():
            stage_comparison[stage_key] = {
                "expected": getattr(comp, "expected", 0),
                "matched": getattr(comp, "matched", 0),
                "missing": getattr(comp, "missing", 0),
                "extra": getattr(comp, "extra", 0),
            }

    # Inefficiency details
    ineff_details = None
    if report.inefficiencies:
        ineff = report.inefficiencies
        ineff_details = {
            "retry_loops": [{"start_step": rl.start_step, "end_step": rl.end_step,
                            "tool": rl.tool, "count": rl.count}
                           for rl in (ineff.retry_loops or [])],
            "backtracks": [{"step": b.step, "from_stage": b.from_stage, "to_stage": b.to_stage}
                          for b in (ineff.backtracks or [])],
            "cyclic_patterns": [{"start_step": cp.start_step, "end_step": cp.end_step,
                                "pattern_length": cp.pattern_length, "repetitions": cp.repetitions}
                               for cp in (ineff.cyclic_patterns or [])],
            "total_wasted_steps": ineff.total_wasted_steps,
            "severity_score": ineff.severity_score,
            "wasted_input_tokens": ineff.wasted_input_tokens,
            "wasted_output_tokens": ineff.wasted_output_tokens,
        }

    # Failure reasons
    failure_reasons = []
    if report.failure_reasons:
        for fr in report.failure_reasons:
            failure_reasons.append({
                "reason": getattr(fr, "reason", ""),
                "detail": getattr(fr, "detail", ""),
                "severity": getattr(fr, "severity", ""),
            })

    return {
        "task_id": entry["task"],
        "model": entry["model"],
        "trajectory_id": entry["trajectory_id"],
        "passed": entry["passed"],
        "quality_score": report.quality_score,
        "quality_tier": report.quality_tier,
        "verdict": getattr(report, "verdict", ""),
        "key_metrics": report.key_metrics,
        "stage_order_match": getattr(report, "stage_order_match", None),
        "failure_reasons": failure_reasons,
        "strengths": report.strengths or [],
        "quality_signals": [str(s) for s in (report.quality_signals or [])],
        "alignment": alignment,
        "divergence_segments": div_segments,
        "stage_comparison": stage_comparison,
        "inefficiency_details": ineff_details,
    }


# ============================================================================
# Main Build
# ============================================================================

def build_dataset(experiment_outputs_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = output_dir / "annotations"
    trajectories_dir = output_dir / "trajectories"
    gt_dir = output_dir / "ground_truth"
    analysis_dir = output_dir / "analysis"
    for d in [annotations_dir, trajectories_dir, gt_dir, analysis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Logging
    logger = logging.getLogger("build_dataset")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    fh = logging.FileHandler(output_dir / "build_dataset.log", mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("Building AgentLens-Bench Dataset")
    logger.info(f"  merge_k={MERGE_K}, seed={SEED}, task_timeout={TASK_TIMEOUT}s")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 70)

    start_time = time.time()
    rng = random.Random(SEED)

    # Discover tasks
    task_dirs = sorted([
        d for d in experiment_outputs_dir.iterdir()
        if d.is_dir() and (d / "pta_outputs").is_dir()
    ])
    logger.info(f"Found {len(task_dirs)} tasks")

    all_flat_records: List[Dict] = []
    task_summaries: List[Dict] = []
    skipped = []

    for task_idx, task_dir in enumerate(task_dirs, 1):
        task_name = task_dir.name
        pta_dir = task_dir / "pta_outputs"
        pta_files_list = sorted(pta_dir.glob("*.json"))

        passing_ptas = []
        failing_ptas = []
        for f in pta_files_list:
            parsed = parse_pta_filename(f.name, task_name)
            if parsed is None:
                continue
            model_name, passed, run_id = parsed
            entry = {"path": f, "model": model_name, "passed": passed,
                     "run_id": run_id, "trajectory_id": f.stem, "task": task_name}
            if passed:
                passing_ptas.append(entry)
            else:
                failing_ptas.append(entry)

        if len(passing_ptas) < MERGE_K:
            skipped.append((task_name, len(passing_ptas)))
            logger.info(f"[{task_idx}/{len(task_dirs)}] {task_name}: SKIP ({len(passing_ptas)} passing)")
            continue

        # Select donors
        passing_shuffled = passing_ptas.copy()
        rng.shuffle(passing_shuffled)
        donors = passing_shuffled[:MERGE_K]
        donor_ids = {d["trajectory_id"] for d in donors}

        # Build merged PTA
        try:
            donor_traces = [trace_api.load(str(d["path"]), format="trace") for d in donors]
            merged_pta = trace_api.merge(donor_traces, use_llm=False)
        except Exception as e:
            logger.warning(f"[{task_idx}] {task_name}: merge failed: {e}")
            skipped.append((task_name, f"merge_error"))
            continue

        num_states = len(merged_pta.states)
        num_transitions = len(merged_pta.transitions)
        all_candidates = passing_ptas + failing_ptas
        assess_count = len(all_candidates) - MERGE_K
        logger.info(f"[{task_idx}/{len(task_dirs)}] {task_name}: merged={num_states} states, assessing {assess_count} trajectories")

        # Save merged PTA (Tier 2: ground_truth)
        merged_dst = gt_dir / f"{task_name}_merged_pta.json"
        # Copy from existing if available, else serialize
        existing_merged = pta_dir / f"{task_name}_merged_pta.json"
        if existing_merged.exists():
            shutil.copy2(existing_merged, merged_dst)
        else:
            # Serialize fresh merge
            merged_json = merged_pta.to_dict() if hasattr(merged_pta, "to_dict") else {"states": {}, "transitions": []}
            with open(merged_dst, 'w') as f:
                json.dump(merged_json, f)

        # Get GT best path for custom detection
        from swe_trace_sdk.match import _enumerate_paths
        gt_paths = _enumerate_paths(merged_pta)
        gt_best: List[State] = gt_paths[0] if gt_paths else []

        # Create task trajectory dir (Tier 2)
        task_traj_dir = trajectories_dir / task_name
        task_traj_dir.mkdir(parents=True, exist_ok=True)
        # Create task analysis dir (Tier 3)
        task_analysis_dir = analysis_dir / task_name
        task_analysis_dir.mkdir(parents=True, exist_ok=True)

        task_start = time.time()
        assessed = 0
        errors = 0
        task_passing = 0
        task_failing = 0

        for entry in all_candidates:
            if entry["trajectory_id"] in donor_ids:
                continue
            if time.time() - task_start > TASK_TIMEOUT:
                logger.warning(f"  Task timeout reached")
                break

            try:
                candidate = trace_api.load(str(entry["path"]), format="trace")
                result = match.run(candidate, merged_pta, use_llm=False)
                report = match.quality_assessment(result, candidate, merged_pta, passed=entry["passed"])

                cand_states = get_ordered_states(candidate)
                n_states = len(cand_states)

                # Waste detections
                regression = detect_regression_loops(cand_states, gt_best)
                retries = extract_blind_retries(report)
                redundant = detect_redundant_steps(cand_states, gt_best)
                unnecessary = detect_unnecessary_exploration(cand_states, gt_best, merged_pta)
                cyclic = extract_cyclic_patterns(report)

                # Tier 1: flat record
                flat = report_to_flat(report, result, entry, n_states,
                                     regression, retries, redundant, unnecessary, cyclic)
                all_flat_records.append(flat)

                # Tier 2: copy individual PTA
                traj_dst = task_traj_dir / f"{entry['trajectory_id']}.json"
                shutil.copy2(entry["path"], traj_dst)

                # Tier 3: full analysis report
                report_json = report_to_json(report, result, entry)
                report_json["waste_detections"] = {
                    "regression_loops": regression,
                    "blind_retries": retries,
                    "redundant_steps": redundant,
                    "unnecessary_exploration": unnecessary,
                    "cyclic_patterns": cyclic,
                }
                report_dst = task_analysis_dir / f"{entry['trajectory_id']}_report.json"
                with open(report_dst, 'w', encoding='utf-8') as f:
                    json.dump(report_json, f, indent=2, default=str)

                assessed += 1
                if entry["passed"]:
                    task_passing += 1
                else:
                    task_failing += 1

            except Exception as e:
                logger.warning(f"  Error on {entry['trajectory_id']}: {e}")
                errors += 1

        task_elapsed = time.time() - task_start
        task_summaries.append({
            "task_id": task_name,
            "n_trajectories": len(all_candidates),
            "n_passing": len(passing_ptas),
            "n_failing": len(failing_ptas),
            "n_donors": MERGE_K,
            "n_assessed": assessed,
            "n_errors": errors,
            "merged_pta_states": num_states,
            "merged_pta_transitions": num_transitions,
            "elapsed_s": round(task_elapsed, 1),
        })
        logger.info(f"  Done: {assessed} assessed, {errors} errors ({task_elapsed:.1f}s)")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal: {len(all_flat_records)} trajectories in {elapsed:.1f}s")

    # ========================================================================
    # Save Tier 1: CSV + Parquet
    # ========================================================================

    logger.info("\nSaving Tier 1 annotations...")

    # Trajectories table
    try:
        import pandas as pd
        df = pd.DataFrame(all_flat_records)
        df.to_csv(annotations_dir / "trajectories.csv", index=False)
        df.to_parquet(annotations_dir / "trajectories.parquet", index=False)
        logger.info(f"  trajectories.csv/parquet: {len(df)} rows, {len(df.columns)} columns")
    except ImportError:
        # Fallback: CSV only
        import csv
        if all_flat_records:
            keys = all_flat_records[0].keys()
            with open(annotations_dir / "trajectories.csv", 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_flat_records)
            logger.info(f"  trajectories.csv: {len(all_flat_records)} rows (pandas not available for parquet)")

    # Tasks table
    try:
        import pandas as pd
        df_tasks = pd.DataFrame(task_summaries)
        df_tasks.to_csv(annotations_dir / "tasks.csv", index=False)
        df_tasks.to_parquet(annotations_dir / "tasks.parquet", index=False)
        logger.info(f"  tasks.csv/parquet: {len(df_tasks)} rows")
    except ImportError:
        import csv
        if task_summaries:
            keys = task_summaries[0].keys()
            with open(annotations_dir / "tasks.csv", 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(task_summaries)

    # ========================================================================
    # Curation Table (Top-k analysis)
    # ========================================================================

    logger.info("\nComputing curation table...")
    passing_records = [r for r in all_flat_records if r["passed"]]
    passing_sorted = sorted(passing_records, key=lambda x: -x["quality_score"])

    def curation_stats(records: List[Dict], label: str) -> Dict:
        n = len(records)
        ideal = sum(1 for r in records if r["quality_score"] >= IDEAL_MIN)
        solid = sum(1 for r in records if LUCKY_MAX <= r["quality_score"] < IDEAL_MIN)
        lucky = sum(1 for r in records if r["quality_score"] < LUCKY_MAX)
        mean_coherence = sum(r.get("coherence_score", 0) for r in records) / max(n, 1)
        mean_score = sum(r["quality_score"] for r in records) / max(n, 1)
        return {
            "strategy": label, "k": n,
            "ideal_pct": round(ideal / max(n, 1) * 100, 1),
            "solid_pct": round(solid / max(n, 1) * 100, 1),
            "lucky_pct": round(lucky / max(n, 1) * 100, 1),
            "mean_coherence": round(mean_coherence, 3),
            "mean_score": round(mean_score, 1),
            "min_score": records[-1]["quality_score"] if records else 0,
        }

    curation_all = curation_stats(passing_sorted, "Random (all passing)")
    curation_50 = curation_stats(passing_sorted[:50], "Top-50 by score")
    curation_25 = curation_stats(passing_sorted[:25], "Top-25 by score")

    curation_table = [curation_all, curation_50, curation_25]

    logger.info("\n" + "=" * 70)
    logger.info("CURATION TABLE: Quality-guided vs Random Selection")
    logger.info("=" * 70)
    logger.info(f"{'Strategy':<25} {'k':>6} {'Ideal%':>8} {'Solid%':>8} {'Lucky%':>8} {'Coherence':>10} {'MeanQS':>8}")
    logger.info("-" * 80)
    for row in curation_table:
        logger.info(f"{row['strategy']:<25} {row['k']:>6} {row['ideal_pct']:>7.1f}% {row['solid_pct']:>7.1f}% {row['lucky_pct']:>7.1f}% {row['mean_coherence']:>10.3f} {row['mean_score']:>8.1f}")

    # Save curation table
    with open(annotations_dir / "curation_table.json", 'w') as f:
        json.dump(curation_table, f, indent=2)

    # ========================================================================
    # Summary
    # ========================================================================

    summary = {
        "dataset_name": "AgentLens-Bench",
        "version": "1.0",
        "config": {"merge_k": MERGE_K, "seed": SEED, "task_timeout": TASK_TIMEOUT,
                   "ideal_threshold": IDEAL_MIN, "lucky_threshold": LUCKY_MAX},
        "stats": {
            "tasks": len(task_summaries),
            "tasks_skipped": len(skipped),
            "trajectories": len(all_flat_records),
            "passing": sum(1 for r in all_flat_records if r["passed"]),
            "failing": sum(1 for r in all_flat_records if not r["passed"]),
            "models": len(set(r["model"] for r in all_flat_records)),
        },
        "tier_distribution": {
            "ideal": sum(1 for r in all_flat_records if r["quality_tier"] == "ideal"),
            "solid": sum(1 for r in all_flat_records if r["quality_tier"] == "solid"),
            "lucky": sum(1 for r in all_flat_records if r["quality_tier"] == "lucky"),
            "partial_fail": sum(1 for r in all_flat_records if r["quality_tier"] == "partial_fail"),
            "off_track": sum(1 for r in all_flat_records if r["quality_tier"] == "off_track"),
        },
        "curation_table": curation_table,
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(output_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info("DATASET BUILD COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Tasks: {summary['stats']['tasks']}")
    logger.info(f"  Trajectories: {summary['stats']['trajectories']} ({summary['stats']['passing']} pass, {summary['stats']['failing']} fail)")
    logger.info(f"  Models: {summary['stats']['models']}")
    logger.info(f"  Tier 1: {annotations_dir}")
    logger.info(f"  Tier 2: {trajectories_dir} + {gt_dir}")
    logger.info(f"  Tier 3: {analysis_dir}")
    logger.info(f"  Elapsed: {elapsed:.1f}s")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build AgentLens-Bench Dataset")
    parser.add_argument("experiment_outputs", type=Path,
                        help="Path to experiment_outputs directory")
    parser.add_argument("--output-dir", "-o", type=Path,
                        default=Path("new_research_experiments/dataset"))
    args = parser.parse_args()

    if not args.experiment_outputs.is_dir():
        print(f"ERROR: {args.experiment_outputs} is not a directory")
        sys.exit(1)

    build_dataset(args.experiment_outputs, args.output_dir)
