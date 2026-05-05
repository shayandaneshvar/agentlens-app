"""Orchestrates SDK calls for the webapp."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from swe_trace_sdk import trace, match
from swe_trace_sdk.io import (
    find_trajectory_file,
    find_openhands_trajectory_file,
    find_atif_trajectory_files,
)
from swe_trace_sdk.intent import (
    label_trace_intents,
    label_intent_stages,
    compute_fingerprint,
    compute_coherence_score,
)
from swe_trace_sdk.models import Trace

from .store import store, TraceRecord, _extract_task as _store_extract_task

logger = logging.getLogger(__name__)


# ── Upload & parse ────────────────────────────────────────────────────────

def _detect_pass_fail_from_zip(zip_dir: Path) -> Optional[bool]:
    """Try to read eval.json inside extracted zip to determine pass/fail."""
    for candidate in [
        zip_dir / "output" / "eval.json",
        zip_dir / "eval.json",
    ]:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                resolved = data.get("resolved", data.get("passed"))
                if isinstance(resolved, bool):
                    return resolved
            except (json.JSONDecodeError, KeyError):
                pass
    return None


def _detect_pass_fail_from_filename(name: str) -> Optional[bool]:
    """Parse evaluation platform naming convention: *-pass-* or *-fail-*."""
    lower = name.lower()
    if "-pass-" in lower or lower.endswith("-pass.zip"):
        return True
    if "-fail-" in lower or lower.endswith("-fail.zip"):
        return False
    return None


def _sanitize_filename(filename: str) -> str:
    """Strip directory components to prevent path traversal."""
    return Path(filename).name


def process_upload(file_bytes: bytes, filename: str) -> List[TraceRecord]:
    """Process an uploaded file (zip or raw JSON) and store the trace(s).

    Returns a list of :class:`TraceRecord` — usually one element,
    but ATIF session ZIPs may contain multiple agent trajectories.
    """
    filename = _sanitize_filename(filename)
    if filename.lower().endswith(".zip"):
        return _process_zip(file_bytes, filename)
    return _process_json(file_bytes, filename)


def _process_zip(file_bytes: bytes, filename: str) -> List[TraceRecord]:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / filename
        zip_path.write_bytes(file_bytes)

        extract_dir = Path(tmp) / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # The zip may contain a top-level folder
        children = list(extract_dir.iterdir())
        root = children[0] if len(children) == 1 and children[0].is_dir() else extract_dir

        # ── Try ATIF first (session ZIP with atif/ directory) ─────────
        atif_paths = find_atif_trajectory_files(str(root))
        if atif_paths:
            records: List[TraceRecord] = []
            base_label = re.sub(r"\.zip$", "", filename, flags=re.IGNORECASE)
            passed = _detect_pass_fail_from_filename(filename)
            for traj_path in atif_paths:
                t = trace.load(str(traj_path), format="atif")
                agent_name = t.metadata.get("agent_name", "unknown")
                label = f"{base_label} ({agent_name})"
                rec = store.add(t, label=label, fmt="atif", passed=passed)
                records.append(rec)
            if records:
                return records

        # ── Try evaluation platform, then OpenHands ───────────────────────────────
        traj_path = find_trajectory_file(str(root))
        fmt = "chatlog"
        if traj_path is None:
            traj_path = find_openhands_trajectory_file(str(root))
            fmt = "openhands"

        if traj_path is None:
            raise ValueError(f"No trajectory file found in {filename}")

        t = trace.load(str(traj_path), format=fmt)

        # Auto-detect pass/fail
        passed = _detect_pass_fail_from_zip(root)
        if passed is None:
            passed = _detect_pass_fail_from_filename(filename)

        label = re.sub(r"\.zip$", "", filename, flags=re.IGNORECASE)
        return [store.add(t, label=label, fmt=fmt, passed=passed)]


def _process_json(file_bytes: bytes, filename: str) -> List[TraceRecord]:
    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / filename
        json_path.write_bytes(file_bytes)

        # Peek at the JSON to detect format
        data = json.loads(file_bytes.decode("utf-8"))
        if isinstance(data, dict):
            # ── ATIF format: dict with schema_version starting with 'ATIF'
            schema = data.get("schema_version", "")
            if isinstance(schema, str) and schema.startswith("ATIF"):
                fmt = "atif"
            elif "initial_state" in data:
                fmt = "trace"  # pre-saved SDK trace
            else:
                fmt = "openhands"
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # evaluation platform chat-export-logs.json is an array of log entries
            if data[0].get("kind") in ("toolCall", "request"):
                fmt = "chatlog"
            else:
                fmt = "openhands"
        else:
            fmt = "chatlog"

        t = trace.load(str(json_path), format=fmt)
        passed = _detect_pass_fail_from_filename(filename)
        label = re.sub(r"\.(json|zip)$", "", filename, flags=re.IGNORECASE)
        if fmt == "atif":
            agent_name = t.metadata.get("agent_name", "unknown")
            label = f"{label} ({agent_name})"
        return [store.add(t, label=label, fmt=fmt, passed=passed)]


# ── Tier 1: Single trajectory profile ────────────────────────────────────

def get_profile(trace_id: str) -> Dict[str, Any]:
    """Return Tier 1 profile data for a single trace."""
    t = store.get_trace(trace_id)
    if t is None:
        raise KeyError(trace_id)
    rec = store.get_record(trace_id)
    _ensure_labeled(t)
    meta = t.metadata or {}

    states = sorted(t.states.values(), key=lambda s: s.step)

    # Stage distribution
    stage_counts: Dict[str, int] = {}
    for s in states:
        stg = s.intent_stage or "unknown"
        stage_counts[stg] = stage_counts.get(stg, 0) + 1
    total_states = len(states)
    stage_pcts = {k: round(v / total_states * 100, 1) if total_states else 0 for k, v in stage_counts.items()}

    # Tool distribution
    tool_counts: Dict[str, int] = {}
    for s in states:
        if s.tool_used:
            tool_counts[s.tool_used] = tool_counts.get(s.tool_used, 0) + 1

    # Files touched
    all_files: set[str] = set()
    for s in states:
        all_files.update(s.files_touched)

    # Coherence
    coherence = compute_coherence_score(states) if states else 0.0

    # Workflow fingerprint
    fp_str, fp_detail = compute_fingerprint(states) if states else ("", [])

    # Stage sequence for timeline (all states, not just tool-bearing ones)
    stage_sequence = [s.intent_stage or "unknown" for s in states]
    tool_sequence = [s.tool_used or "" for s in states if s.tool_used]

    # Coherence label
    if coherence >= 0.7:
        coherence_label = "Clean forward progression"
    elif coherence >= 0.4:
        coherence_label = "Some backtracking detected"
    else:
        coherence_label = "Heavy thrashing"

    # Operation type distribution
    op_counts: Dict[str, int] = {}
    for s in states:
        op = getattr(s, "operation_type", None) or "other"
        op_counts[op] = op_counts.get(op, 0) + 1

    # "Completed" = the last state in the trace has is_terminal metadata.
    # For ATIF format the generator always marks the last state as terminal,
    # so the flag is meaningless — return None (unknown).
    completed: Optional[bool] = None
    if rec and rec.format != "atif" and states:
        last_state = states[-1]
        completed = bool(getattr(last_state, 'metadata', {}).get('is_terminal', False))

    # ── Standalone behavioral indicators ──
    # Exploration-to-implementation ratio
    expl_count = stage_counts.get("exploration", 0)
    impl_count = stage_counts.get("implementation", 0)
    exploration_ratio = round(expl_count / impl_count, 2) if impl_count > 0 else (
        float(expl_count) if expl_count > 0 else 0.0
    )

    # Files modified vs read-only
    modified_files: set[str] = set()
    read_files: set[str] = set()
    edit_tools = {"create_file", "replace_string_in_file", "multi_replace_string_in_file", "apply_patch"}
    for s in states:
        tool = s.tool_used or ""
        fp = getattr(s, "file_path", "") or ""
        if not fp:
            continue
        if tool in edit_tools:
            modified_files.add(fp)
        elif tool in {"read_file", "list_dir"}:
            read_files.add(fp)
    files_read_only = len(read_files - modified_files)

    # ── Human Experience metrics (ATIF only) ──────────────────────────────
    wall_time_ms = meta.get("wall_time_ms") or None
    permission_wait_ms = meta.get("permission_wait_ms") or None
    active_time_ms_val = meta.get("active_time_ms") or None
    human_input_count = meta.get("human_input_count")
    compaction_count_val = meta.get("compaction_count")
    human_input_positions = meta.get("human_input_positions") or None

    # Per-step latencies and cumulative token counts from state metadata
    step_latencies: list[int] | None = None
    step_token_cumulative: list[int] | None = None
    _latencies: list[int] = []
    _cum_tokens: list[int] = []
    _running_tokens = 0
    for s in states:
        smeta = getattr(s, "metadata", {}) or {}
        lat = smeta.get("latency_ms", 0)
        _latencies.append(int(lat) if lat else 0)
        _running_tokens += int(smeta.get("prompt_tokens", 0) or 0)
        _cum_tokens.append(_running_tokens)

    # Only expose if we have meaningful data (at least one non-zero entry)
    if any(v > 0 for v in _latencies):
        step_latencies = _latencies
    if _running_tokens > 0:
        step_token_cumulative = _cum_tokens

    # Time decomposition: agent_work / llm_thinking / human_wait
    time_decomposition: dict[str, int] | None = None
    if wall_time_ms and wall_time_ms > 0:
        llm_thinking_ms = sum(_latencies)
        human_wait_ms = permission_wait_ms or 0
        agent_work_ms = max(0, wall_time_ms - llm_thinking_ms - human_wait_ms)
        time_decomposition = {
            "agent_work_ms": agent_work_ms,
            "llm_thinking_ms": llm_thinking_ms,
            "human_wait_ms": human_wait_ms,
        }

    # Composite Human Experience Score (0–100)
    hx_score: float | None = None
    hx_breakdown: dict[str, float] | None = None
    if human_input_count is not None and total_states > 0:
        # Autonomy: fewer human interventions → higher (0–1)
        autonomy = max(0.0, 1.0 - (human_input_count / total_states)) if total_states > 0 else 1.0

        # Low friction: less permission wait relative to total → higher (0–1)
        if wall_time_ms and wall_time_ms > 0 and permission_wait_ms is not None:
            low_friction = max(0.0, 1.0 - (permission_wait_ms / wall_time_ms))
        else:
            low_friction = 1.0  # No friction data = assume no friction

        # Responsiveness: sigmoid on avg latency — 5s ideal, 30s+ bad (0–1)
        if step_latencies:
            non_zero = [v for v in step_latencies if v > 0]
            avg_lat = sum(non_zero) / len(non_zero) if non_zero else 0
            # Sigmoid: score = 1 / (1 + e^((avg_lat - 15000) / 5000))
            responsiveness = 1.0 / (1.0 + math.exp((avg_lat - 15000) / 5000))
        else:
            responsiveness = 0.5  # No data = neutral

        # Context stability: penalise compactions (0–1)
        cc = compaction_count_val or 0
        stability = max(0.0, 1.0 - cc * 0.2) if cc >= 0 else 1.0

        # Weighted composite
        raw = (autonomy * 0.30 + low_friction * 0.30
               + responsiveness * 0.25 + stability * 0.15)
        hx_score = round(raw * 100, 1)
        hx_breakdown = {
            "autonomy": round(autonomy * 100, 1),
            "low_friction": round(low_friction * 100, 1),
            "responsiveness": round(responsiveness * 100, 1),
            "stability": round(stability * 100, 1),
        }

    return {
        "trace_id": trace_id,
        "state_count": total_states,
        "file_count": len(all_files),
        "tool_count": len(tool_counts),
        "coherence": round(coherence, 4),
        "coherence_label": coherence_label,
        "stage_distribution": stage_counts,
        "stage_percentages": stage_pcts,
        "tool_distribution": tool_counts,
        "files_touched": sorted(all_files),
        "fingerprint": fp_str,
        "fingerprint_detail": fp_detail,
        "operation_types": op_counts,
        "completed": completed,
        "stage_sequence": stage_sequence,
        "tool_sequence": tool_sequence,
        "exploration_ratio": exploration_ratio,
        "files_modified": len(modified_files),
        "files_read_only": files_read_only,
        "model": rec.model if rec else "",
        "agent": (t.metadata or {}).get("agent_name", ""),
        "task": rec.task if rec else "",
        "benchmark": rec.benchmark if rec else "",
        # Behavioural counters (None = not available for this format)
        "human_input_count": human_input_count,
        "subagent_count": meta.get("subagent_count"),
        "active_time_ms": active_time_ms_val,
        "compaction_count": compaction_count_val,
        # Human Experience metrics (None = not available for this format)
        "wall_time_ms": wall_time_ms,
        "permission_wait_ms": permission_wait_ms,
        "human_experience_score": hx_score,
        "hx_breakdown": hx_breakdown,
        "time_decomposition": time_decomposition,
        "step_latencies": step_latencies,
        "step_token_cumulative": step_token_cumulative,
        "human_input_positions": human_input_positions,
    }


# ── Merge ─────────────────────────────────────────────────────────────────

def build_ground_truth(trace_ids: List[str]) -> Dict[str, Any]:
    """Merge specified passing traces into ground truth."""
    traces = []
    for tid in trace_ids:
        t = store.get_trace(tid)
        if t is None:
            raise KeyError(tid)
        traces.append(t)

    gt = trace.merge(traces)
    rec = store.add(gt, label="Ground Truth", fmt="merged", passed=None)
    store.set_ground_truth(rec.trace_id)

    return {
        "gt_id": rec.trace_id,
        "source_count": len(trace_ids),
        "state_count": rec.state_count,
    }


def export_ground_truth() -> Dict[str, Any]:
    """Export the current ground truth as a JSON-serializable dict."""
    gt = store.get_ground_truth()
    if gt is None:
        raise ValueError("No ground truth built yet")
    return gt.to_dict()


def import_ground_truth(gt_data: Dict[str, Any]) -> Dict[str, Any]:
    """Import a previously exported merged GT JSON."""
    from swe_trace_sdk.models import Trace as TraceModel
    gt = TraceModel.from_dict(gt_data)
    _ensure_labeled(gt)
    rec = store.add(gt, label="Imported Ground Truth", fmt="merged", passed=None)
    store.set_ground_truth(rec.trace_id)
    logger.info(
        "import_ground_truth: NEW GT id=%s, %d states",
        rec.trace_id, rec.state_count,
    )
    return {
        "gt_id": rec.trace_id,
        "source_count": gt_data.get("metadata", {}).get("source_count", 0),
        "state_count": rec.state_count,
    }


def assess_with_imported_gt(
    trace_id: str,
    gt_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Assess a trajectory against an imported merged GT JSON."""
    from swe_trace_sdk.models import Trace as TraceModel

    candidate = store.get_trace(trace_id)
    if candidate is None:
        raise KeyError(trace_id)
    _ensure_labeled(candidate)

    gt = TraceModel.from_dict(gt_data)
    _ensure_labeled(gt)

    # Store GT so it can be viewed/exported later
    gt_rec = store.add(gt, label="Imported Ground Truth", fmt="merged", passed=None)
    store.set_ground_truth(gt_rec.trace_id)

    result = match.run(candidate, gt)
    cand_rec = store.get_record(trace_id)
    report = match.quality_assessment(result, candidate, gt, passed=cand_rec.passed if cand_rec else None)

    required_tools = match.extract_required_tools(gt)
    process_cov, missing_tools = match.check_process_coverage(candidate, required_tools)
    required_files = match.extract_required_files(gt)
    file_cov, missing_files = match.check_file_coverage(candidate, required_files)

    comparison = _build_comparison_data(result, candidate, gt)

    return {
        "trace_id": trace_id,
        "gt_source_count": gt_data.get("metadata", {}).get("source_count", 0),
        "gt_state_count": len(gt.states),
        "match_metrics": result.metrics.to_dict(),
        "quality_report": report.to_dict(),
        "process_coverage": round(process_cov, 4),
        "missing_tools": missing_tools,
        "file_coverage": round(file_cov, 4),
        "missing_files": missing_files,
        "comparison": comparison,
    }


# ── Tier 2: Quality assessment ────────────────────────────────────────────

def _ensure_labeled(t: Trace) -> None:
    """Label intent stages on trace states if not already done."""
    states = sorted(t.states.values(), key=lambda s: s.step)
    if states and not any(s.intent_stage for s in states):
        label_intent_stages(states)


def _state_to_summary(state) -> Dict[str, Any]:
    """Convert a State to a compact dict for the comparison view."""
    return {
        "state_id": state.state_id,
        "step": state.step,
        "tool": getattr(state, "tool_used", "") or "",
        "file_path": getattr(state, "file_path", "") or "",
        "intent_stage": getattr(state, "intent_stage", "") or "",
        "resulting_state": getattr(state, "resulting_state", "") or "",
        "content_description": getattr(state, "content_description", "") or "",
    }


def _build_comparison_data(
    result,
    candidate: Trace,
    gt: Trace,
) -> Dict[str, Any]:
    """Build the dual-lane comparison data for GT vs candidate visualization.

    Returns a dict with:
    - gt_path: ordered list of GT state summaries (the best-matching path)
    - candidate_path: ordered list of candidate state summaries
    - alignment: list of {gt_index, candidate_index} pairs for matched states
    - gt_matched_indexes: set of GT indexes that were matched
    - candidate_matched_indexes: set of candidate indexes that were matched
    """
    from swe_trace_sdk.match import _enumerate_paths

    gt_paths = _enumerate_paths(gt)
    if not gt_paths:
        return {"gt_path": [], "candidate_path": [], "alignment": [],
                "gt_matched_indexes": [], "candidate_matched_indexes": []}

    best_idx = result.metrics.best_path_index
    if best_idx >= len(gt_paths):
        best_idx = 0
    gt_seq = gt_paths[best_idx]

    # Candidate states ordered by step
    cand_ordered = sorted(candidate.states.values(), key=lambda s: s.step)
    # Filter actionable (same as match.run does)
    cand_ordered = [s for s in cand_ordered
                    if not (hasattr(s, "log_entry") and s.log_entry and s.log_entry.kind == "request")]

    gt_path_data = [_state_to_summary(s) for s in gt_seq]
    cand_path_data = [_state_to_summary(s) for s in cand_ordered]

    # Build alignment pairs from the match result
    # result.alignment maps candidate steps to GT states
    alignment_pairs = []

    # Build candidate_id → cand_index map for resolving alignment
    cand_id_to_idx = {s.state_id: i for i, s in enumerate(cand_ordered)}

    for align_entry in result.alignment:
        if align_entry.matched and align_entry.ground_truth_state_id:
            # Find the GT index for this GT state
            gt_state_id = align_entry.ground_truth_state_id
            gt_index = None
            for gi, gs in enumerate(gt_seq):
                if gs.state_id == gt_state_id:
                    gt_index = gi
                    break
            cand_index = cand_id_to_idx.get(align_entry.candidate_state_id)
            if gt_index is not None and cand_index is not None:
                alignment_pairs.append({
                    "gt_index": gt_index,
                    "candidate_index": cand_index,
                })

    # Derive matched indexes from alignment pairs (same index space as gt_path/cand_path)
    gt_matched = {p["gt_index"] for p in alignment_pairs}
    cand_matched = {p["candidate_index"] for p in alignment_pairs}

    return {
        "gt_path": gt_path_data,
        "candidate_path": cand_path_data,
        "alignment": alignment_pairs,
        "gt_matched_indexes": sorted(gt_matched),
        "candidate_matched_indexes": sorted(cand_matched),
        "terminal_state_match": result.metrics.terminal_state_match,
    }


def assess_trajectory(trace_id: str) -> Dict[str, Any]:
    """Assess a single trajectory against the current ground truth."""
    candidate = store.get_trace(trace_id)
    if candidate is None:
        raise KeyError(trace_id)
    _ensure_labeled(candidate)

    gt = store.get_ground_truth()
    if gt is None:
        raise ValueError("No ground truth built yet")

    result = match.run(candidate, gt)
    rec = store.get_record(trace_id)
    report = match.quality_assessment(result, candidate, gt, passed=rec.passed if rec else None)

    return {
        "trace_id": trace_id,
        "match_metrics": result.metrics.to_dict(),
        "quality_report": report.to_dict(),
    }


def assess_batch() -> Dict[str, Any]:
    """Assess all non-GT trajectories and return cohort ranking."""
    gt = store.get_ground_truth()
    if gt is None:
        raise ValueError("No ground truth built yet")

    gt_id = store.ground_truth_id
    records = store.list_records()
    candidate_records = [r for r in records if r.trace_id != gt_id]

    results_for_ranking: list[tuple[str, Any]] = []
    candidates: list[Trace] = []
    per_trajectory: list[Dict[str, Any]] = []

    for rec in candidate_records:
        cand = store.get_trace(rec.trace_id)
        if cand is None:
            continue
        _ensure_labeled(cand)
        mr = match.run(cand, gt)
        report = match.quality_assessment(mr, cand, gt, passed=rec.passed)
        results_for_ranking.append((rec.label, mr))
        candidates.append(cand)
        per_trajectory.append({
            "trace_id": rec.trace_id,
            "label": rec.label,
            "quality_report": report.to_dict(),
        })

    ranking = match.rank_in_cohort(results_for_ranking, candidates=candidates, ground_truth=gt)

    return {
        "ranking": ranking.to_dict(),
        "trajectories": per_trajectory,
    }


# ── Visualization data ────────────────────────────────────────────────────

def get_visualization(trace_id: str) -> Dict[str, Any]:
    """Return full trace data for DAG rendering."""
    t = store.get_trace(trace_id)
    if t is None:
        raise KeyError(trace_id)
    return t.to_dict()


# ── Assess with uploaded GT files ─────────────────────────────────────────

def _load_trace_from_bytes(file_bytes: bytes, filename: str) -> Trace:
    """Load a Trace from raw file bytes (zip or json) without storing it."""
    filename = _sanitize_filename(filename)
    if filename.lower().endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / filename
            zip_path.write_bytes(file_bytes)
            extract_dir = Path(tmp) / "extracted"
            extract_dir.mkdir()
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            children = list(extract_dir.iterdir())
            root = children[0] if len(children) == 1 and children[0].is_dir() else extract_dir

            # Try ATIF first
            atif_paths = find_atif_trajectory_files(str(root))
            if atif_paths:
                # Load the first ATIF trajectory found in this zip
                return trace.load(str(atif_paths[0]), format="atif")

            traj_path = find_trajectory_file(str(root))
            fmt = "chatlog"
            if traj_path is None:
                traj_path = find_openhands_trajectory_file(str(root))
                fmt = "openhands"
            if traj_path is None:
                raise ValueError(f"No trajectory file found in {filename}")
            return trace.load(str(traj_path), format=fmt)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / filename
            json_path.write_bytes(file_bytes)
            data = json.loads(file_bytes.decode("utf-8"))
            if isinstance(data, dict):
                schema = data.get("schema_version", "")
                if isinstance(schema, str) and schema.startswith("ATIF"):
                    return trace.load(str(json_path), format="atif")
                if "initial_state" in data:
                    fmt = "trace"
                else:
                    fmt = "openhands"
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                fmt = "chatlog" if data[0].get("kind") in ("toolCall", "request") else "openhands"
            else:
                fmt = "chatlog"
            return trace.load(str(json_path), format=fmt)


def assess_with_uploaded_gt(
    trace_id: str,
    gt_files: List[Tuple[bytes, str]],
) -> Dict[str, Any]:
    """Load GT trajectories from uploaded files, merge, match, and assess.

    The GT files are NOT stored in the trace list — they are used only
    to build the ground truth for this assessment.
    """
    candidate = store.get_trace(trace_id)
    if candidate is None:
        raise KeyError(trace_id)
    _ensure_labeled(candidate)

    # Load all GT traces from raw bytes
    gt_traces: List[Trace] = []
    for fb, fn in gt_files:
        gt_traces.append(_load_trace_from_bytes(fb, fn))

    if len(gt_traces) < 2:
        raise ValueError("Need at least 2 passing trajectories to build ground truth")

    # Merge into ground truth
    gt = trace.merge(gt_traces)

    # Store for later export/reuse
    gt_rec = store.add(gt, label="Ground Truth", fmt="merged", passed=None)
    store.set_ground_truth(gt_rec.trace_id)
    logger.info(
        "assess_with_uploaded_gt: built NEW GT id=%s from %d files, %d states",
        gt_rec.trace_id, len(gt_traces), len(gt.states),
    )

    # Match
    result = match.run(candidate, gt)
    cand_rec = store.get_record(trace_id)
    report = match.quality_assessment(result, candidate, gt, passed=cand_rec.passed if cand_rec else None)

    # Extra coverage signals
    required_tools = match.extract_required_tools(gt)
    process_cov, missing_tools = match.check_process_coverage(candidate, required_tools)
    required_files = match.extract_required_files(gt)
    file_cov, missing_files = match.check_file_coverage(candidate, required_files)

    metrics = result.metrics.to_dict()
    quality = report.to_dict()

    # Build comparison data (GT path + candidate path + alignment)
    comparison = _build_comparison_data(result, candidate, gt)

    return {
        "trace_id": trace_id,
        "gt_source_count": len(gt_traces),
        "gt_state_count": len(gt.states),
        "match_metrics": metrics,
        "quality_report": quality,
        "process_coverage": round(process_cov, 4),
        "missing_tools": missing_tools,
        "file_coverage": round(file_cov, 4),
        "missing_files": missing_files,
        "comparison": comparison,
    }


# ── LLM-Based Behavioral Assessment ──────────────────────────────────────

_LLM_SYSTEM_PROMPT = """\
You are an expert evaluator of AI coding agent trajectories.

You will receive TWO inputs:
1. **PTA Matching Results** — deterministic metrics from matching this trajectory \
against a merged ground-truth Prefix Tree Acceptor (PTA) built from multiple \
passing trajectories.
2. **Trajectory Structure** — the full sequence of actions the agent took, with \
tool calls, file paths, intent stages, and step numbers.

Your job is to provide a holistic behavioral assessment that goes BEYOND what the \
deterministic metrics can capture. Analyze the agent's strategy, decision-making, \
and problem-solving approach.

## Scoring Dimensions

Rate each dimension as "strong", "adequate", or "weak". Write your reasoning \
FIRST — cite specific step numbers and evidence — THEN assign the rating.

1. **Strategy**: Did the agent form a coherent plan? Did it read code before \
editing? Was the exploration purposeful?
2. **Efficiency**: Were steps wasted? Unnecessary re-reads, blind retries, \
redundant explorations?
3. **Verification**: Did the agent verify its changes? Run tests? Check results?
4. **Error Recovery**: When errors occurred, did the agent adapt intelligently \
or repeat the same approach?
5. **Completeness**: Did the agent address the full scope of the task, or only \
a partial solution?

## Output Format

Respond with ONLY valid JSON (no markdown fences, no commentary outside JSON):

{
  "summary": "2-3 sentence overall assessment",
  "dimensions": {
    "strategy":       { "reasoning": "...", "rating": "strong|adequate|weak" },
    "efficiency":     { "reasoning": "...", "rating": "strong|adequate|weak" },
    "verification":   { "reasoning": "...", "rating": "strong|adequate|weak" },
    "error_recovery": { "reasoning": "...", "rating": "strong|adequate|weak" },
    "completeness":   { "reasoning": "...", "rating": "strong|adequate|weak" }
  },
  "key_findings": [
    { "type": "strength|weakness", "observation": "...", "evidence": "step N: ..." }
  ],
  "overall_rating": "strong|adequate|weak",
  "recommendation": "One sentence actionable improvement suggestion"
}
"""


def _condense_trajectory(t: Trace) -> str:
    """Build a compact readable summary of the trajectory structure."""
    states = sorted(t.states.values(), key=lambda s: s.step)
    lines = []
    for s in states:
        tool = s.tool_used or "(none)"
        stage = s.intent_stage or "?"
        fp = s.file_path or ""
        result = (s.resulting_state or "")[:80]
        fp_short = fp.split("/")[-1] if fp else ""
        line = f"  Step {s.step:3d} [{stage:14s}] {tool:30s}"
        if fp_short:
            line += f"  file={fp_short}"
        if result:
            line += f"  -> {result}"
        lines.append(line)
    return "\n".join(lines)


def _condense_gt_structure(gt: Trace) -> str:
    """Build a compact summary of the ground-truth PTA structure."""
    states = sorted(gt.states.values(), key=lambda s: s.step)
    stage_counts: Dict[str, int] = {}
    tool_counts: Dict[str, int] = {}
    for s in states:
        stg = s.intent_stage or "unknown"
        stage_counts[stg] = stage_counts.get(stg, 0) + 1
        if s.tool_used:
            tool_counts[s.tool_used] = tool_counts.get(s.tool_used, 0) + 1

    lines = [
        f"Ground Truth: {len(states)} states, {len(gt.transitions)} transitions",
        f"Stages: {', '.join(f'{k}={v}' for k, v in sorted(stage_counts.items()))}",
        f"Tools: {', '.join(f'{k}={v}' for k, v in sorted(tool_counts.items(), key=lambda x: -x[1])[:10])}",
    ]

    fp, _ = compute_fingerprint(states)
    if fp:
        lines.append(f"Workflow fingerprint: {fp}")

    return "\n".join(lines)


def _format_match_results(
    report_dict: Dict[str, Any],
    metrics_dict: Dict[str, Any],
    process_cov: float,
    file_cov: float,
    missing_tools: List[str],
    missing_files: List[str],
) -> str:
    """Format PTA matching results as readable text for the LLM."""
    lines = [
        "## PTA Matching Results (Deterministic)",
        "",
        f"Quality Score: {report_dict.get('quality_score', '?')}/100",
        f"Verdict: {report_dict.get('verdict', '?')}",
        f"Quality Tier: {report_dict.get('quality_tier', '?')}",
        "",
        "### Metrics",
        f"  Coverage:            {metrics_dict.get('coverage_percent', 0):.1f}%",
        f"  Coherence:           {metrics_dict.get('coherence_score', 0):.3f}",
        f"  Stage Completeness:  {metrics_dict.get('stage_completeness', 0):.3f}",
        f"  Workflow Similarity:  {metrics_dict.get('workflow_similarity', 0):.3f}",
        f"  F1 Score:            {metrics_dict.get('f1_score', 0):.1f}",
        f"  Process Coverage:    {process_cov*100:.1f}%",
        f"  File Coverage:       {file_cov*100:.1f}%",
    ]

    if missing_tools:
        lines.append(f"  Missing Tools:       {', '.join(missing_tools)}")
    if missing_files:
        lines.append(f"  Missing Files:       {', '.join(missing_files[:10])}")

    reasons = report_dict.get("failure_reasons", [])
    if reasons:
        lines.append("")
        lines.append("### Failure Reasons")
        for r in reasons[:5]:
            lines.append(f"  [{r.get('severity', '?')}] {r.get('reason', '?')}")

    signals = report_dict.get("quality_signals", [])
    if signals:
        lines.append("")
        lines.append("### Quality Signals")
        for sig in signals[:6]:
            lines.append(f"  [{sig.get('severity', '?')}] {sig.get('description', '?')}")

    ineff = report_dict.get("inefficiencies", {})
    if ineff:
        lines.append("")
        lines.append("### Inefficiencies Detected")
        lines.append(f"  Inefficiency severity: {ineff.get('severity_score', 0):.1%}")
        lines.append(f"  Retry loops:      {ineff.get('retry_loop_count', 0)}")
        lines.append(f"  Cyclic patterns:  {ineff.get('cyclic_pattern_count', 0)}")
        lines.append(f"  Backtracks:       {ineff.get('backtrack_count', 0)}")
        lines.append(f"  Redundant steps:  {ineff.get('redundant_step_count', 0)}")
        lines.append(f"  Wasted steps:     {ineff.get('total_wasted_steps', 0)}")
        w_in = ineff.get('wasted_input_tokens', 0)
        w_out = ineff.get('wasted_output_tokens', 0)
        if w_in or w_out:
            lines.append(f"  Wasted tokens:    {w_in + w_out:,} ({w_in:,} in / {w_out:,} out)")

    stage_cmp = report_dict.get("stage_comparison", {})
    if stage_cmp:
        lines.append("")
        lines.append("### Per-Stage Comparison")
        for stage, data in stage_cmp.items():
            expected = len(data.get("expected_steps", []))
            matched = len(data.get("matched_steps", []))
            missing = len(data.get("missing_steps", []))
            extra = len(data.get("extra_steps", []))
            lines.append(f"  {stage}: {matched}/{expected} matched, {missing} missing, {extra} extra")

    return "\n".join(lines)


def _get_llm_client():
    """Get an OpenAI-compatible LLM client from environment config.

    Reads AGENTLENS_LLM or DEFAULT_LLM env var in "provider:model[:temp]" format.
    """
    from swe_trace_sdk.llm_providers import get_provider_and_model_from_env, get_client

    result = get_provider_and_model_from_env("AGENTLENS")
    if isinstance(result[0], list):
        provider, model, temp = result[0][0], result[1][0], result[2][0]
    else:
        provider, model, temp = result

    client = get_client(provider)
    return client, model, temp


def llm_assess(trace_id: str) -> Dict[str, Any]:
    """Run LLM-based behavioral assessment on a previously assessed trajectory.

    Requires that a Tier 2 PTA assessment has already been run (so GT exists).
    Sends both the PTA matching results AND the trajectory structure to the LLM.
    """
    candidate = store.get_trace(trace_id)
    if candidate is None:
        raise KeyError(trace_id)
    _ensure_labeled(candidate)

    gt = store.get_ground_truth()
    if gt is None:
        raise ValueError("Run PTA assessment first (need ground truth)")

    result = match.run(candidate, gt)
    rec = store.get_record(trace_id)
    report = match.quality_assessment(
        result, candidate, gt, passed=rec.passed if rec else None,
    )

    required_tools = match.extract_required_tools(gt)
    process_cov, missing_tools = match.check_process_coverage(candidate, required_tools)
    required_files = match.extract_required_files(gt)
    file_cov, missing_files = match.check_file_coverage(candidate, required_files)

    # Build the user prompt with BOTH inputs
    match_text = _format_match_results(
        report.to_dict(), result.metrics.to_dict(),
        process_cov, file_cov, missing_tools, missing_files,
    )
    gt_text = _condense_gt_structure(gt)
    traj_text = _condense_trajectory(candidate)

    cand_states = sorted(candidate.states.values(), key=lambda s: s.step)
    passed_str = "PASSED" if rec and rec.passed else (
        "FAILED" if rec and rec.passed is False else "UNKNOWN"
    )

    user_prompt = (
        f"# Trajectory Assessment\n\n"
        f"## Task Outcome: {passed_str}\n"
        f"## Trajectory: {rec.label if rec else trace_id} ({len(cand_states)} steps)\n\n"
        f"{match_text}\n\n"
        f"## Ground Truth PTA Structure\n{gt_text}\n\n"
        f"## Candidate Trajectory (step-by-step)\n{traj_text}\n\n"
        f"Based on the PTA matching results and trajectory structure above, provide your "
        f"behavioral assessment. The deterministic metrics tell you WHAT happened — your "
        f"job is to explain WHY and assess the quality of the agent's decision-making process."
    )

    try:
        client, model, temp = _get_llm_client()
    except Exception as e:
        raise ValueError(
            f"LLM not configured. Set AGENTLENS_LLM or DEFAULT_LLM env var "
            f"(e.g. 'openai:gpt-4o'). Error: {e}"
        )

    logger.info("Calling LLM (%s) for behavioral assessment of %s", model, trace_id)

    response = client.chat.completions.create(
        model=model,
        temperature=temp,
        messages=[
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2000,
    )

    raw_content = response.choices[0].message.content or ""

    # Parse JSON from response
    json_str = raw_content.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_str = "\n".join(lines)

    start = json_str.find("{")
    end = json_str.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM did not return valid JSON. Raw: {raw_content[:500]}")
    json_str = json_str[start:end + 1]

    try:
        llm_result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM JSON: {e}. Raw: {json_str[:500]}")

    return {
        "trace_id": trace_id,
        "model_used": model,
        "assessment": llm_result,
        "quality_score": report.quality_score,
        "verdict": report.verdict,
    }


# ── LLM-Based Improvement Suggestions ───────────────────────────────────

_LLM_SUGGESTIONS_SYSTEM_PROMPT = """\
You are an expert at diagnosing why AI coding agents fail or waste effort.

You will receive an inefficiency analysis of a coding agent's trajectory, \
including retry loops, cyclic patterns, backtracks, redundant steps, and \
wasted steps — each with actual tool outputs and error messages. You will \
also receive the full condensed trajectory and what the correct ground-truth \
path included.

Your job: produce specific, actionable suggestions for how the agent could \
have avoided each inefficiency. Focus on ROOT CAUSES, not symptoms.

Rules:
- Reference specific step numbers and actual error messages/tool outputs.
- For each retry loop, identify what the agent should have noticed in the \
FIRST failure that would have prevented retries.
- For each cyclic pattern, identify what information was available to break \
the cycle earlier.
- For backtracks, explain what the agent missed that caused regression.
- Prioritize by token/step savings (highest impact first).
- For estimated_savings, use the ACTUAL token counts provided in the data. \
Do NOT guess or hallucinate token numbers. If token data is not available \
for a specific inefficiency, only mention step savings, not tokens.
- Be concrete and specific: "At step 15, the error says 'Cannot find module \
./config' — the agent should have created config.ts before retrying the test" \
NOT "the agent should read errors more carefully".
- Limit to the top 5-7 most impactful suggestions.

Respond with ONLY valid JSON (no markdown fences, no commentary outside JSON):

{
  "suggestions": [
    {
      "priority": "high|medium|low",
      "category": "retry_prevention|exploration_control|verification|error_recovery|tool_usage",
      "title": "5-10 word actionable title",
      "root_cause": "Why the agent made this mistake — what it failed to notice or do",
      "suggestion": "Exactly what should have been done instead, with step references",
      "affected_steps": [15, 16, 17, 18],
      "estimated_savings": "3 steps, ~50K tokens"
    }
  ],
  "improvement_summary": "Single most impactful change in one sentence"
}
"""


def _format_inefficiency_details(
    report_dict: Dict[str, Any],
    candidate: "Trace",
) -> str:
    """Format inefficiency details with surrounding step context for the LLM."""
    ineff = report_dict.get("inefficiencies", {})
    if not ineff:
        return "No inefficiencies detected."

    states = sorted(candidate.states.values(), key=lambda s: s.step)
    # Build step lookup: step_number -> state
    step_map: Dict[int, Any] = {}
    for s in states:
        step_map[s.step] = s

    def _step_line(s: Any) -> str:
        tool = s.tool_used or "(none)"
        stage = s.intent_stage or "?"
        fp = (s.file_path or "").split("/")[-1] if s.file_path else ""
        result = (s.resulting_state or "")[:120]
        meta = getattr(s, "metadata", {}) or {}
        i_tok = int(meta.get("input_tokens", 0) or 0)
        o_tok = int(meta.get("output_tokens", 0) or 0)
        line = f"    Step {s.step} [{stage}] {tool}"
        if fp:
            line += f"  file={fp}"
        if i_tok or o_tok:
            line += f"  ({i_tok + o_tok:,} tok)"
        if result:
            line += f"  -> {result}"
        return line

    def _steps_token_total(step_nums: list) -> int:
        """Sum tokens across a list of step numbers."""
        total = 0
        for sn in step_nums:
            s = step_map.get(sn)
            if s:
                meta = getattr(s, "metadata", {}) or {}
                total += int(meta.get("input_tokens", 0) or 0)
                total += int(meta.get("output_tokens", 0) or 0)
        return total

    def _context_steps(step_num: int, radius: int = 2) -> str:
        lines = []
        for offset in range(-radius, radius + 1):
            s = step_map.get(step_num + offset)
            if s:
                prefix = "  >>>" if s.step == step_num else "     "
                lines.append(prefix + _step_line(s).lstrip())
        return "\n".join(lines)

    lines = [
        f"=== INEFFICIENCY ANALYSIS ===",
        f"Inefficiency Severity: {ineff.get('severity_score', 0):.0%} "
        f"({ineff.get('total_wasted_steps', 0)} of {len(states)} steps wasted)",
    ]

    w_in = ineff.get("wasted_input_tokens", 0)
    w_out = ineff.get("wasted_output_tokens", 0)
    t_in = ineff.get("total_input_tokens", 0)
    t_out = ineff.get("total_output_tokens", 0)
    if w_in or w_out:
        lines.append(
            f"Wasted Tokens: {w_in + w_out:,} of {t_in + t_out:,} total "
            f"({(w_in + w_out) / max(t_in + t_out, 1) * 100:.0f}%)"
        )

    # Retry loops with context
    retry_loops = ineff.get("retry_loops", [])
    if retry_loops:
        lines.append(f"\nRETRY LOOPS ({len(retry_loops)}):")
        for r in retry_loops:
            wasted_steps_in_loop = list(range(r["start_step"] + 2, r["end_step"] + 1))
            loop_tokens = _steps_token_total(wasted_steps_in_loop)
            tok_note = f", {loop_tokens:,} wasted tokens" if loop_tokens else ""
            lines.append(
                f"  Steps {r['start_step']}-{r['end_step']}: "
                f"{r['tool']} on {r.get('file_path', '?')} ({r['count']} consecutive{tok_note})"
            )
            for step_n in range(r["start_step"], r["end_step"] + 1):
                s = step_map.get(step_n)
                if s:
                    lines.append(_step_line(s))

    # Cyclic patterns with context
    cyclic = ineff.get("cyclic_patterns", [])
    if cyclic:
        lines.append(f"\nCYCLIC PATTERNS ({len(cyclic)}):")
        for c in cyclic:
            sig = " -> ".join(c.get("pattern_signature", []))
            # Wasted = excess reps beyond first occurrence
            first_end = c["start_step"] + c["pattern_length"]
            wasted_steps_in_cycle = list(range(first_end, c["end_step"] + 1))
            cycle_tokens = _steps_token_total(wasted_steps_in_cycle)
            tok_note = f", {cycle_tokens:,} wasted tokens" if cycle_tokens else ""
            lines.append(
                f"  Steps {c['start_step']}-{c['end_step']}: "
                f"{c['pattern_length']}-step cycle x{c['repetitions']}: {sig}{tok_note}"
            )
            for step_n in range(c["start_step"], min(c["end_step"] + 1, c["start_step"] + 8)):
                s = step_map.get(step_n)
                if s:
                    lines.append(_step_line(s))
            if c["end_step"] - c["start_step"] > 8:
                lines.append(f"    ... ({c['end_step'] - c['start_step'] - 8} more steps)")

    # Backtracks with context
    backtracks = ineff.get("backtracks", [])
    if backtracks:
        lines.append(f"\nBACKTRACKS ({len(backtracks)}):")
        for b in backtracks:
            lines.append(
                f"  Step {b['step']}: {b['from_stage']} -> {b['to_stage']}"
            )
            lines.append(_context_steps(b["step"]))

    # Per-tool breakdown
    per_tool = ineff.get("per_tool_breakdown", [])
    if per_tool:
        lines.append(f"\nPER-TOOL BREAKDOWN:")
        for t in per_tool:
            parts = []
            if t.get("retries"):
                parts.append(f"{t['retries']} retries")
            if t.get("backtracks"):
                parts.append(f"{t['backtracks']} backtracks")
            if t.get("cycles"):
                parts.append(f"{t['cycles']} cycles")
            if t.get("redundant"):
                parts.append(f"{t['redundant']} redundant")
            if t.get("unnecessary"):
                parts.append(f"{t['unnecessary']} unnecessary")
            lines.append(f"  {t['tool']}: {t['total_wasted']} wasted ({', '.join(parts)})")

    return "\n".join(lines)


def _format_gt_diff(report_dict: Dict[str, Any]) -> str:
    """Format what the GT did differently — missing/extra steps."""
    lines = ["=== WHAT GROUND TRUTH DID DIFFERENTLY ==="]
    stage_cmp = report_dict.get("stage_comparison", {})
    if stage_cmp:
        for stage, data in stage_cmp.items():
            missing = data.get("missing_steps", [])
            extra = data.get("extra_steps", [])
            if missing or extra:
                lines.append(f"\n{stage}:")
                for m in missing[:5]:
                    lines.append(f"  MISSING: {m.get('tool', '?')} on {m.get('file_path', '?')}")
                for e in extra[:5]:
                    lines.append(f"  EXTRA (not in GT): {e.get('tool', '?')} on {e.get('file_path', '?')}")
    return "\n".join(lines)


def llm_suggestions(trace_id: str) -> Dict[str, Any]:
    """Generate LLM-based actionable improvement suggestions.

    Separate from llm_assess — focused solely on diagnosing inefficiencies
    and suggesting concrete fixes. Requires prior Tier 2 PTA assessment.
    """
    candidate = store.get_trace(trace_id)
    if candidate is None:
        raise KeyError(trace_id)
    _ensure_labeled(candidate)

    gt = store.get_ground_truth()
    if gt is None:
        raise ValueError("Run PTA assessment first (need ground truth)")

    result = match.run(candidate, gt)
    rec = store.get_record(trace_id)
    report = match.quality_assessment(
        result, candidate, gt, passed=rec.passed if rec else None,
    )
    report_dict = report.to_dict()

    # Check if there are any inefficiencies to suggest about
    ineff = report_dict.get("inefficiencies", {})
    if ineff.get("total_wasted_steps", 0) == 0:
        return {
            "trace_id": trace_id,
            "model_used": "",
            "suggestions": [],
            "improvement_summary": "No inefficiencies detected — trajectory is clean.",
        }

    ineff_text = _format_inefficiency_details(report_dict, candidate)
    gt_diff_text = _format_gt_diff(report_dict)
    traj_text = _condense_trajectory(candidate)

    cand_states = sorted(candidate.states.values(), key=lambda s: s.step)
    passed_str = "PASSED" if rec and rec.passed else (
        "FAILED" if rec and rec.passed is False else "UNKNOWN"
    )

    # Compute total trajectory tokens for the prompt header
    total_traj_tokens = ineff.get("total_input_tokens", 0) + ineff.get("total_output_tokens", 0)
    token_line = f" | Total tokens: {total_traj_tokens:,}" if total_traj_tokens else ""

    user_prompt = (
        f"# Improvement Analysis\n\n"
        f"## Task Outcome: {passed_str}\n"
        f"## Trajectory: {rec.label if rec else trace_id} ({len(cand_states)} steps{token_line})\n\n"
        f"{ineff_text}\n\n"
        f"{gt_diff_text}\n\n"
        f"## Full Trajectory (condensed)\n{traj_text}\n\n"
        f"Analyze the inefficiencies above. For each one, diagnose the ROOT CAUSE "
        f"and provide a specific, actionable suggestion for what the agent should "
        f"have done differently. Reference actual step numbers and error messages. "
        f"Use the token counts provided above for estimated_savings — do not invent numbers."
    )

    try:
        client, model, temp = _get_llm_client()
    except Exception as e:
        raise ValueError(
            f"LLM not configured. Set AGENTLENS_LLM or DEFAULT_LLM env var "
            f"(e.g. 'openai:gpt-4o'). Error: {e}"
        )

    logger.info("Calling LLM (%s) for improvement suggestions for %s", model, trace_id)

    response = client.chat.completions.create(
        model=model,
        temperature=temp,
        messages=[
            {"role": "system", "content": _LLM_SUGGESTIONS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2000,
    )

    raw_content = response.choices[0].message.content or ""

    # Parse JSON from response
    json_str = raw_content.strip()
    if json_str.startswith("```"):
        json_lines = json_str.split("\n")
        json_lines = [l for l in json_lines if not l.strip().startswith("```")]
        json_str = "\n".join(json_lines)

    start = json_str.find("{")
    end = json_str.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM did not return valid JSON. Raw: {raw_content[:500]}")
    json_str = json_str[start:end + 1]

    try:
        llm_result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM JSON: {e}. Raw: {json_str[:500]}")

    return {
        "trace_id": trace_id,
        "model_used": model,
        "suggestions": llm_result.get("suggestions", []),
        "improvement_summary": llm_result.get("improvement_summary", ""),
    }


# ── Human Experience metric helpers ──────────────────────────────────────

def _compute_hx_metrics(trace_obj: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Compute HX-related fields for a candidate in the compare view.

    Returns a dict of keys to merge into the compare candidate response.
    All values are None when insufficient data is available.
    """
    wall_time_ms = meta.get("wall_time_ms") or None
    permission_wait_ms = meta.get("permission_wait_ms") or None
    active_time_ms_val = meta.get("active_time_ms") or None
    human_input_count = meta.get("human_input_count")
    compaction_count_val = meta.get("compaction_count")

    states = sorted(trace_obj.states.values(), key=lambda s: s.step)
    total_states = len(states)

    # Per-step latencies
    _latencies: list[int] = []
    for s in states:
        smeta = getattr(s, "metadata", {}) or {}
        lat = smeta.get("latency_ms", 0)
        _latencies.append(int(lat) if lat else 0)
    step_latencies = _latencies if any(v > 0 for v in _latencies) else None

    # Time decomposition
    time_decomposition = None
    if wall_time_ms and wall_time_ms > 0:
        llm_ms = sum(_latencies)
        human_ms = permission_wait_ms or 0
        agent_ms = max(0, wall_time_ms - llm_ms - human_ms)
        time_decomposition = {
            "agent_work_ms": agent_ms,
            "llm_thinking_ms": llm_ms,
            "human_wait_ms": human_ms,
        }

    # HX score
    hx_score = None
    hx_breakdown = None
    if human_input_count is not None and total_states > 0:
        autonomy = max(0.0, 1.0 - (human_input_count / total_states)) if total_states > 0 else 1.0
        if wall_time_ms and wall_time_ms > 0 and permission_wait_ms is not None:
            low_friction = max(0.0, 1.0 - (permission_wait_ms / wall_time_ms))
        else:
            low_friction = 1.0
        if step_latencies:
            non_zero = [v for v in step_latencies if v > 0]
            avg_lat = sum(non_zero) / len(non_zero) if non_zero else 0
            responsiveness = 1.0 / (1.0 + math.exp((avg_lat - 15000) / 5000))
        else:
            responsiveness = 0.5
        cc = compaction_count_val or 0
        stability = max(0.0, 1.0 - cc * 0.2)
        raw = autonomy * 0.30 + low_friction * 0.30 + responsiveness * 0.25 + stability * 0.15
        hx_score = round(raw * 100, 1)
        hx_breakdown = {
            "autonomy": round(autonomy * 100, 1),
            "low_friction": round(low_friction * 100, 1),
            "responsiveness": round(responsiveness * 100, 1),
            "stability": round(stability * 100, 1),
        }

    return {
        "wall_time_ms": wall_time_ms,
        "permission_wait_ms": permission_wait_ms,
        "human_experience_score": hx_score,
        "hx_breakdown": hx_breakdown,
        "time_decomposition": time_decomposition,
        "step_latencies": step_latencies,
    }


# ── Multi-Trajectory Comparison ──────────────────────────────────────────


def compare_traces(trace_ids: List[str], *, gt_strategy: str = "best_match") -> Dict[str, Any]:
    """Compare multiple candidate trajectories against the same GT.

    Returns per-candidate metrics, quality reports, stage comparisons,
    and GT structure with per-candidate matched state IDs for overlay rendering.
    """
    if len(trace_ids) < 2:
        raise ValueError("Need at least 2 traces to compare")
    if len(trace_ids) > 5:
        raise ValueError("Maximum 5 traces for comparison")

    # Validate all traces belong to the same task
    tasks: set[str] = set()
    for tid in trace_ids:
        rec = store.get_record(tid)
        if rec is None:
            raise KeyError(f"Trace {tid} not found")
        trace_obj = store.get_trace(tid)
        # Re-extract task to use latest extraction logic
        t = _store_extract_task(trace_obj, rec.label) if trace_obj else rec.task or ""
        t = t.strip().lower()
        if t:
            tasks.add(t)
    if len(tasks) > 1:
        raise ValueError(
            f"Cannot compare trajectories from different tasks: {', '.join(sorted(tasks))}"
        )

    gt = store.get_ground_truth()
    if gt is None:
        raise ValueError("No ground truth built yet — run Tier 2 assessment first")

    logger.info(
        "compare_traces: GT id=%s, states=%d, traces=%s",
        store.ground_truth_id, len(gt.states), trace_ids,
    )

    _ensure_labeled(gt)

    required_tools = match.extract_required_tools(gt)
    required_files = match.extract_required_files(gt)

    # Build GT state_id → state map for overlay data
    gt_states_ordered = sorted(gt.states.values(), key=lambda s: s.step)
    gt_state_ids = [s.state_id for s in gt_states_ordered]

    candidates_data: List[Dict[str, Any]] = []

    for tid in trace_ids:
        candidate = store.get_trace(tid)
        if candidate is None:
            raise KeyError(f"Trace {tid} not found")
        _ensure_labeled(candidate)
        c_meta = candidate.metadata or {}

        rec = store.get_record(tid)
        result = match.run(candidate, gt, gt_strategy=gt_strategy)
        report = match.quality_assessment(
            result, candidate, gt, passed=rec.passed if rec else None,
        )

        process_cov, missing_tools = match.check_process_coverage(candidate, required_tools)
        file_cov, missing_files = match.check_file_coverage(candidate, required_files)

        # Find which GT state_ids this candidate matched (set-based, order-independent)
        matched_gt_state_ids = list(result.matched_gt_state_ids)

        report_dict = report.to_dict()
        metrics_dict = result.metrics.to_dict()

        # Per-stage breakdown
        stage_detail: Dict[str, Any] = {}
        stage_cmp = report_dict.get("stage_comparison", {})
        for stage_name, data in stage_cmp.items():
            stage_detail[stage_name] = {
                "matched": len(data.get("matched_steps", [])),
                "missing": len(data.get("missing_steps", [])),
                "extra": len(data.get("extra_steps", [])),
                "effort_ratio": data.get("effort_ratio", 0),
                "ordering_preserved": data.get("ordering_preserved", True),
            }

        candidates_data.append({
            "trace_id": tid,
            "label": rec.label if rec else tid,
            "passed": rec.passed if rec else None,
            "model": rec.model if rec else "",
            "agent": (candidate.metadata or {}).get("agent_name", ""),
            "state_count": result.metrics.candidate_states,
            "metrics": {
                "quality_score": report.quality_score,
                "verdict": report.verdict,
                "coverage_percent": metrics_dict.get("coverage_percent", 0),
                "coherence_score": metrics_dict.get("coherence_score", 0),
                "stage_completeness": metrics_dict.get("stage_completeness", 0),
                "workflow_similarity": metrics_dict.get("workflow_similarity", 0),
                "f1_score": metrics_dict.get("f1_score", 0),
                "bottleneck_coverage": metrics_dict.get("bottleneck_coverage", 0),
                "bottleneck_stage": min(
                    metrics_dict.get("stage_coverage", {}),
                    key=lambda k: metrics_dict["stage_coverage"][k],
                    default="",
                ) if metrics_dict.get("stage_coverage") else "",
                "process_coverage": round(process_cov, 4),
                "file_coverage": round(file_cov, 4),
            },
            "inefficiencies": {
                "total_wasted_steps": min(
                    report_dict.get("inefficiencies", {}).get("total_wasted_steps", 0),
                    result.metrics.candidate_states,
                ),
                "retry_loop_count": report_dict.get("inefficiencies", {}).get("retry_loop_count", 0),
                "backtrack_count": report_dict.get("inefficiencies", {}).get("backtrack_count", 0),
                "redundant_step_count": report_dict.get("inefficiencies", {}).get("redundant_step_count", 0),
                "cyclic_pattern_count": report_dict.get("inefficiencies", {}).get("cyclic_pattern_count", 0),
                "severity_score": report_dict.get("inefficiencies", {}).get("severity_score", 0.0),
                "wasted_input_tokens": report_dict.get("inefficiencies", {}).get("wasted_input_tokens", 0),
                "wasted_output_tokens": report_dict.get("inefficiencies", {}).get("wasted_output_tokens", 0),
                "total_input_tokens": report_dict.get("inefficiencies", {}).get("total_input_tokens", 0),
                "total_output_tokens": report_dict.get("inefficiencies", {}).get("total_output_tokens", 0),
            },
            "stage_detail": stage_detail,
            "matched_gt_state_ids": matched_gt_state_ids,
            "strengths": report_dict.get("strengths", []),
            "failure_reasons": report_dict.get("failure_reasons", []),
            # Behavioural counters (None = not available for this format)
            "human_input_count": c_meta.get("human_input_count"),
            "subagent_count": c_meta.get("subagent_count"),
            "active_time_ms": c_meta.get("active_time_ms") or None,
            "compaction_count": c_meta.get("compaction_count"),
            # Human Experience metrics
            **_compute_hx_metrics(candidate, c_meta),
        })

    # GT structure for visualization
    gt_data = gt.to_dict()

    return {
        "candidates": candidates_data,
        "gt": gt_data,
        "gt_state_ids": gt_state_ids,
    }


_LLM_COMPARE_SYSTEM_PROMPT = """\
You are an expert evaluator comparing multiple AI coding agent trajectories \
that attempted the same software engineering task.

You will receive:
1. **Individual assessments** — per-trajectory LLM quality assessments with \
5-dimension ratings and key findings.
2. **Key metrics** — deterministic PTA matching scores per trajectory.

Your job is to produce a **comparative analysis** that explains the relative \
strengths and weaknesses across trajectories.

## Output Format

Respond with ONLY valid JSON (no markdown fences):

{
  "comparative_summary": "3-4 sentence comparative overview",
  "dimension_comparison": {
    "strategy": { "ranking": ["label_best", "label_2nd", ...], "analysis": "..." },
    "efficiency": { "ranking": [...], "analysis": "..." },
    "verification": { "ranking": [...], "analysis": "..." },
    "error_recovery": { "ranking": [...], "analysis": "..." },
    "completeness": { "ranking": [...], "analysis": "..." }
  },
  "key_differences": [
    { "aspect": "...", "observation": "...", "labels_compared": ["A", "B"] }
  ],
  "recommendation": "Which trajectory is best and why, in one sentence"
}
"""


def llm_compare(trace_ids: List[str]) -> Dict[str, Any]:
    """Run comparative LLM assessment across multiple trajectories.

    Uses Option C: individual assessments first, then a comparative synthesis.
    """
    # Step 1: Get individual LLM assessments
    individual_results: List[Dict[str, Any]] = []
    for tid in trace_ids:
        try:
            result = llm_assess(tid)
            individual_results.append(result)
        except Exception as e:
            logger.warning("LLM assess failed for %s: %s", tid, e)
            individual_results.append({
                "trace_id": tid,
                "model_used": "N/A",
                "assessment": {"summary": f"Assessment failed: {e}"},
                "quality_score": 0,
                "verdict": "error",
            })

    # Step 2: Build comparative prompt
    sections: List[str] = []
    for res in individual_results:
        rec = store.get_record(res["trace_id"])
        label = rec.label if rec else res["trace_id"]
        passed_str = "PASSED" if rec and rec.passed else (
            "FAILED" if rec and rec.passed is False else "UNKNOWN"
        )
        a = res.get("assessment", {})
        lines = [
            f"## Trajectory: {label}",
            f"Outcome: {passed_str}",
            f"Quality Score: {res.get('quality_score', '?')}/100",
            f"Verdict: {res.get('verdict', '?')}",
        ]
        if "summary" in a:
            lines.append(f"Summary: {a['summary']}")
        dims = a.get("dimensions", {})
        for dim_name, dim_data in dims.items():
            if isinstance(dim_data, dict):
                lines.append(f"  {dim_name}: {dim_data.get('rating', '?')} — {dim_data.get('reasoning', '')[:200]}")
        findings = a.get("key_findings", [])
        for f in findings[:3]:
            if isinstance(f, dict):
                lines.append(f"  [{f.get('type', '?')}] {f.get('observation', '')}")
        sections.append("\n".join(lines))

    user_prompt = (
        "# Comparative Trajectory Assessment\n\n"
        + "\n\n".join(sections)
        + "\n\nBased on the individual assessments above, provide a comparative "
        "analysis. Rank the trajectories per dimension and explain the key "
        "differences in their approaches."
    )

    try:
        client, model, temp = _get_llm_client()
    except Exception as e:
        raise ValueError(
            f"LLM not configured. Set AGENTLENS_LLM or DEFAULT_LLM env var. Error: {e}"
        )

    logger.info("Calling LLM (%s) for comparative assessment of %d traces", model, len(trace_ids))

    response = client.chat.completions.create(
        model=model,
        temperature=temp,
        messages=[
            {"role": "system", "content": _LLM_COMPARE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2000,
    )

    raw_content = response.choices[0].message.content or ""
    json_str = raw_content.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_str = "\n".join(lines)

    start = json_str.find("{")
    end = json_str.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM did not return valid JSON. Raw: {raw_content[:500]}")
    json_str = json_str[start:end + 1]

    comparative = json.loads(json_str)

    return {
        "individual": individual_results,
        "comparative": comparative,
        "model_used": model,
    }
