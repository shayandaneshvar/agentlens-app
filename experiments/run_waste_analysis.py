"""
Waste / Inefficiency Analysis Experiment
=========================================
For each eligible task (≥5 passing trajectories):
  1. Build a fresh k=5 merged PTA (same donors as quality analysis, same seed)
  2. Run match + quality_assessment on all non-donor trajectories
  3. Extract inefficiency counts per category (SDK detection + custom detection)
  4. Aggregate into Pass/Fail prevalence + waste table and Lucky/Ideal comparison

Detection categories:
  1. Regression loops: E→I→E pattern (≥2 E states before returning to I)
  2. Blind retries: 3+ consecutive identical (tool, file, stage) — SDK retry_loops
  3. Redundant steps: re-read same file with no edit between
  4. Unnecessary exploration: post-impl E on files NOT in merged PTA
  5. Cyclic patterns: multi-step repeated subsequences — SDK cyclic_patterns

All detections are GT-aware: patterns present in the merged PTA are NOT flagged.
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
    """Get ordered, filtered states from a trace (same filter as SDK uses)."""
    states = sorted(trace_obj.states.values(), key=lambda s: s.step)
    states = [s for s in states
              if not (hasattr(s, "log_entry") and s.log_entry
                      and getattr(s.log_entry, "kind", None) == "request")]
    states = [s for s in states if getattr(s, "intent_stage", "")]
    return states


def get_stage(s: State) -> str:
    return getattr(s, "intent_stage", "") or ""


def revised_tier(quality_score: int) -> str:
    if quality_score >= IDEAL_MIN:
        return "ideal"
    elif quality_score < LUCKY_MAX:
        return "lucky"
    return "solid"


# ============================================================================
# Custom Detection: Regression Loops (E→I→E pattern)
# ============================================================================

def detect_regression_loops(
    cand_states: List[State],
    gt_states: List[State],
) -> Dict[str, Any]:
    """Detect E→I→E→I patterns: agent implements, then backtracks to explore,
    then implements again. Only flagged if GT doesn't have same pattern.
    
    Returns dict with 'count' (number of regression loops) and 'wasted_steps'
    (total E-labeled states in the regression spans).
    """
    # Build GT regression set for exclusion
    gt_regression_positions: Set[int] = set()
    if gt_states:
        gt_stages = [get_stage(s) for s in gt_states]
        for i in range(1, len(gt_stages) - 1):
            if (gt_stages[i] == "implementation" and
                gt_stages[i - 1] == "exploration"):
                # Check if there's exploration after this implementation
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
    visited_spans: List[Tuple[int, int]] = []
    
    while i < n:
        # Look for: ...I...E(≥2 states)...I pattern
        if stages[i] == "implementation":
            # Found an I. Look ahead for E→...→I
            j = i + 1
            while j < n and stages[j] != "exploration":
                j += 1
            if j >= n:
                break
            # j is the start of E span after I
            e_start = j
            # Count E states until next I
            k = j
            while k < n and stages[k] != "implementation":
                k += 1
            e_span_len = k - e_start
            
            if e_span_len >= 2 and k < n:
                # We have E→I→E(≥2)→I — check GT exclusion
                # Use proportional position check
                prop_pos = e_start / max(n, 1)
                gt_prop_pos = int(prop_pos * len(gt_states)) if gt_states else -1
                
                is_gt_pattern = False
                if gt_states and gt_prop_pos in gt_regression_positions:
                    is_gt_pattern = True
                
                if not is_gt_pattern:
                    regression_count += 1
                    # Wasted = the E states in the regression span
                    wasted_steps += e_span_len
                    visited_spans.append((e_start, k))
                
                i = k  # Continue from the second I
                continue
        i += 1
    
    return {
        "count": regression_count,
        "wasted_steps": wasted_steps,
        "has_any": regression_count > 0,
    }


# ============================================================================
# Custom Detection: Redundant Steps (re-read with no edit between)
# ============================================================================

def detect_redundant_steps(
    cand_states: List[State],
    gt_states: List[State],
) -> Dict[str, Any]:
    """Detect re-reads of the same file/line-range with no implementation
    state between the two reads. GT files that are read multiple times
    are excluded.
    
    Returns dict with 'count' and 'wasted_steps'.
    """
    # Build GT multi-read set: files read more than once in GT
    gt_multi_read_files: Set[str] = set()
    if gt_states:
        gt_read_files: Dict[str, int] = defaultdict(int)
        for s in gt_states:
            stg = get_stage(s)
            if stg == "exploration" and getattr(s, "file_path", ""):
                gt_read_files[s.file_path] += 1
        gt_multi_read_files = {f for f, c in gt_read_files.items() if c >= 2}

    # Track what we've seen (file_path → last step index where read)
    seen_reads: Dict[str, int] = {}  # file_path -> index of last read
    redundant_count = 0
    wasted_steps = 0
    last_impl_idx = -1

    for i, s in enumerate(cand_states):
        stg = get_stage(s)
        
        if stg == "implementation":
            # Reset seen reads — edits invalidate previous reads
            last_impl_idx = i
            seen_reads.clear()
            continue
        
        if stg == "exploration":
            fp = getattr(s, "file_path", "") or ""
            if not fp:
                continue
            
            # Skip if GT also multi-reads this file
            if fp in gt_multi_read_files:
                seen_reads[fp] = i
                continue
            
            if fp in seen_reads:
                # Same file read again with no impl between
                redundant_count += 1
                wasted_steps += 1
            
            seen_reads[fp] = i

    return {
        "count": redundant_count,
        "wasted_steps": wasted_steps,
        "has_any": redundant_count > 0,
    }


# ============================================================================
# Custom Detection: Unnecessary Exploration (post-impl E on non-PTA files)
# ============================================================================

def detect_unnecessary_exploration(
    cand_states: List[State],
    gt_states: List[State],
    merged_pta: Trace,
) -> Dict[str, Any]:
    """Detect exploration states after first implementation that target files
    NOT present in the merged PTA and NOT test files.
    
    Returns dict with 'count' and 'wasted_steps'.
    """
    # Build PTA file set
    pta_files: Set[str] = set()
    for s in merged_pta.states.values():
        fp = getattr(s, "file_path", "") or ""
        if fp:
            pta_files.add(fp)
    
    # Also build GT exploration-after-impl set for exclusion
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
            
            # Skip test files
            fp_lower = fp.lower()
            if "test" in fp_lower or "spec" in fp_lower:
                continue
            
            # Skip if file is in the PTA (legitimate reference)
            if fp in pta_files:
                continue
            
            # Skip if GT also explores this file post-impl
            if fp in gt_post_impl_explore_files:
                continue
            
            unnecessary_count += 1
            wasted_steps += 1

    return {
        "count": unnecessary_count,
        "wasted_steps": wasted_steps,
        "has_any": unnecessary_count > 0,
    }


# ============================================================================
# SDK-based Detection: Blind Retries (from InefficiencyReport.retry_loops)
# ============================================================================

def extract_blind_retries(report) -> Dict[str, Any]:
    """Extract blind retry stats from SDK's InefficiencyReport."""
    ineff = report.inefficiencies
    if ineff is None:
        return {"count": 0, "wasted_steps": 0, "has_any": False}
    
    count = ineff.retry_loop_count
    # Wasted = count - 2 per loop (first 2 are legitimate attempt + retry)
    wasted = sum(max(0, rl.count - 2) for rl in ineff.retry_loops)
    return {"count": count, "wasted_steps": wasted, "has_any": count > 0}


# ============================================================================
# SDK-based Detection: Cyclic Patterns (from InefficiencyReport.cyclic_patterns)
# ============================================================================

def extract_cyclic_patterns(report) -> Dict[str, Any]:
    """Extract cyclic pattern stats from SDK's InefficiencyReport."""
    ineff = report.inefficiencies
    if ineff is None:
        return {"count": 0, "wasted_steps": 0, "has_any": False}
    
    count = ineff.cyclic_pattern_count
    # Wasted = excess repetitions × pattern_length (beyond first occurrence)
    wasted = sum(cp.pattern_length * (cp.repetitions - 1)
                 for cp in ineff.cyclic_patterns)
    return {"count": count, "wasted_steps": wasted, "has_any": count > 0}


# ============================================================================
# Main Analysis
# ============================================================================

def run_waste_analysis(
    experiment_outputs_dir: Path,
    output_dir: Path,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("waste_analysis")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    fh = logging.FileHandler(output_dir / "waste_analysis.log", mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("Waste / Inefficiency Analysis")
    logger.info(f"  merge_k={MERGE_K}, seed={SEED}, task_timeout={TASK_TIMEOUT}s")
    logger.info(f"  Tiers: Ideal>={IDEAL_MIN}, Lucky<{LUCKY_MAX}")
    logger.info("=" * 70)

    # Load pre-computed quality scores for tier assignment
    qa_results_file = Path("new_research_experiments/quality_analysis_outputs/quality_analysis_results.json")
    score_lookup: Dict[str, int] = {}
    if qa_results_file.exists():
        with open(qa_results_file) as f:
            qa_data = json.load(f)
        for r in qa_data.get("trajectory_results", []):
            score_lookup[r["trajectory_id"]] = r["quality_score"]
        logger.info(f"Loaded {len(score_lookup)} pre-computed quality scores")

    start_time = time.time()
    rng = random.Random(SEED)

    # Discover tasks
    task_dirs = sorted([
        d for d in experiment_outputs_dir.iterdir()
        if d.is_dir() and (d / "pta_outputs").is_dir()
    ])
    logger.info(f"Found {len(task_dirs)} tasks")

    # Per-trajectory results
    all_records: List[Dict[str, Any]] = []

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
            entry = {"path": f, "model": model_name, "passed": passed,
                     "run_id": run_id, "trajectory_id": f.stem}
            if passed:
                passing_ptas.append(entry)
            else:
                failing_ptas.append(entry)

        if len(passing_ptas) < MERGE_K:
            logger.info(f"[{task_idx}/{len(task_dirs)}] {task_name}: SKIP ({len(passing_ptas)} passing)")
            continue

        # Select donors (same seed → same donors as quality analysis)
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
            continue

        num_states = len(merged_pta.states)
        logger.info(f"[{task_idx}/{len(task_dirs)}] {task_name}: merged={num_states} states, assessing {len(passing_ptas) + len(failing_ptas) - MERGE_K} trajectories")

        # Get GT best path for custom detection
        from swe_trace_sdk.match import _enumerate_paths
        gt_paths = _enumerate_paths(merged_pta)
        gt_best: List[State] = gt_paths[0] if gt_paths else []

        task_start = time.time()
        assessed = 0

        for entry in passing_ptas + failing_ptas:
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
                
                # --- Detections ---
                regression = detect_regression_loops(cand_states, gt_best)
                retries = extract_blind_retries(report)
                redundant = detect_redundant_steps(cand_states, gt_best)
                unnecessary = detect_unnecessary_exploration(cand_states, gt_best, merged_pta)
                cyclic = extract_cyclic_patterns(report)

                # Quality score: use pre-computed if available, else from report
                tid = entry["trajectory_id"]
                qs = score_lookup.get(tid, report.quality_score)
                tier = revised_tier(qs) if entry["passed"] else "fail"

                record = {
                    "task": task_name,
                    "model": entry["model"],
                    "trajectory_id": tid,
                    "passed": entry["passed"],
                    "quality_score": qs,
                    "tier": tier,
                    "n_states": len(cand_states),
                    "regression_loops": regression,
                    "blind_retries": retries,
                    "redundant_steps": redundant,
                    "unnecessary_exploration": unnecessary,
                    "cyclic_patterns": cyclic,
                }
                all_records.append(record)
                assessed += 1

            except Exception as e:
                logger.warning(f"  Error on {entry['trajectory_id']}: {e}")

        logger.info(f"  Done: {assessed} assessed ({time.time()-task_start:.1f}s)")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal: {len(all_records)} trajectories in {elapsed:.1f}s")

    # ========================================================================
    # Aggregation
    # ========================================================================

    categories = ["regression_loops", "blind_retries", "redundant_steps",
                  "unnecessary_exploration", "cyclic_patterns"]
    category_labels = ["Regression loops", "Blind retries", "Redundant steps",
                       "Unnecessary expl.", "Cyclic patterns"]

    passing_records = [r for r in all_records if r["passed"]]
    failing_records = [r for r in all_records if not r["passed"]]
    ideal_records = [r for r in passing_records if r["tier"] == "ideal"]
    lucky_records = [r for r in passing_records if r["tier"] == "lucky"]

    def compute_stats(records: List[Dict], cat: str) -> Dict[str, float]:
        n = len(records)
        if n == 0:
            return {"prevalence": 0.0, "mean_waste": 0.0, "n_with": 0}
        with_any = [r for r in records if r[cat]["has_any"]]
        n_with = len(with_any)
        prevalence = n_with / n * 100
        mean_waste = (sum(r[cat]["wasted_steps"] for r in with_any) / n_with) if n_with > 0 else 0.0
        return {"prevalence": prevalence, "mean_waste": mean_waste, "n_with": n_with}

    # --- Table 1: Pass vs Fail ---
    logger.info("\n" + "=" * 70)
    logger.info("TABLE 1: WASTE PREVALENCE & IMPACT (PASS vs FAIL)")
    logger.info("=" * 70)
    hdr = "%-20s %8s %8s %8s %8s %8s" % ("Category", "Prev.P%", "Prev.F%", "F/P", "Waste P", "Waste F")
    logger.info(hdr)
    logger.info("-" * 70)

    table1_data = []
    for cat, label in zip(categories, category_labels):
        ps = compute_stats(passing_records, cat)
        fs = compute_stats(failing_records, cat)
        fp_ratio = fs["prevalence"] / ps["prevalence"] if ps["prevalence"] > 0 else float('inf')
        row = {
            "category": label,
            "prev_pass": ps["prevalence"],
            "prev_fail": fs["prevalence"],
            "fp_ratio": fp_ratio,
            "waste_pass": ps["mean_waste"],
            "waste_fail": fs["mean_waste"],
        }
        table1_data.append(row)
        logger.info("%-20s %7.1f%% %7.1f%% %8.2f %7.1f %7.1f" % (
            label, ps["prevalence"], fs["prevalence"], fp_ratio,
            ps["mean_waste"], fs["mean_waste"]))

    # --- Table 2: Lucky vs Ideal ---
    logger.info("\n" + "=" * 70)
    logger.info("TABLE 2: WASTE BY QUALITY TIER (LUCKY vs IDEAL)")
    logger.info("=" * 70)
    logger.info(f"Ideal (score>={IDEAL_MIN}): n={len(ideal_records)}")
    logger.info(f"Lucky (score<{LUCKY_MAX}): n={len(lucky_records)}")
    hdr2 = "%-20s %9s %9s %9s %9s %8s" % ("Category", "Prev.Ideal", "Prev.Lucky", "L/I", "Waste.I", "Waste.L")
    logger.info(hdr2)
    logger.info("-" * 75)

    table2_data = []
    for cat, label in zip(categories, category_labels):
        ids = compute_stats(ideal_records, cat)
        lks = compute_stats(lucky_records, cat)
        li_ratio = lks["prevalence"] / ids["prevalence"] if ids["prevalence"] > 0 else float('inf')
        row = {
            "category": label,
            "prev_ideal": ids["prevalence"],
            "prev_lucky": lks["prevalence"],
            "li_ratio": li_ratio,
            "waste_ideal": ids["mean_waste"],
            "waste_lucky": lks["mean_waste"],
        }
        table2_data.append(row)
        logger.info("%-20s %8.1f%% %9.1f%% %9.2f %8.1f %8.1f" % (
            label, ids["prevalence"], lks["prevalence"], li_ratio,
            ids["mean_waste"], lks["mean_waste"]))

    # Identify strongest discriminator
    cyclic_li = table2_data[4]["li_ratio"]
    max_li = max(table2_data, key=lambda r: r["li_ratio"] if r["li_ratio"] != float('inf') else 0)
    logger.info(f"\nStrongest Lucky/Ideal discriminator: {max_li['category']} (L/I ratio = {max_li['li_ratio']:.2f})")

    # --- Summary ---
    logger.info(f"\n--- SUMMARY ---")
    logger.info(f"Total assessed: {len(all_records)} ({len(passing_records)} pass, {len(failing_records)} fail)")
    logger.info(f"Ideal: {len(ideal_records)}, Lucky: {len(lucky_records)}")
    logger.info(f"F/P ratios all > 1? {all(r['fp_ratio'] > 1.0 for r in table1_data)}")
    logger.info(f"Elapsed: {elapsed:.1f}s")

    # Save results
    output_data = {
        "config": {"merge_k": MERGE_K, "seed": SEED, "task_timeout": TASK_TIMEOUT,
                   "ideal_min": IDEAL_MIN, "lucky_max": LUCKY_MAX},
        "summary": {
            "total_assessed": len(all_records),
            "passing": len(passing_records),
            "failing": len(failing_records),
            "ideal": len(ideal_records),
            "lucky": len(lucky_records),
            "elapsed_seconds": elapsed,
        },
        "table1_pass_vs_fail": table1_data,
        "table2_lucky_vs_ideal": table2_data,
        "trajectory_records": all_records,
    }

    results_file = output_dir / "waste_analysis_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_file}")

    return output_data


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waste/Inefficiency Analysis")
    parser.add_argument("experiment_dir", type=Path,
                        help="Path to experiment_outputs directory")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path("new_research_experiments/waste_analysis_outputs"))
    args = parser.parse_args()

    run_waste_analysis(args.experiment_dir, args.output_dir)
