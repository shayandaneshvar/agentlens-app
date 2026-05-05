"""In-memory store for uploaded traces and analysis results."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swe_trace_sdk.models import Trace


@dataclass
class TraceRecord:
    """Metadata for a stored trace."""

    trace_id: str
    label: str
    format: str  # "chatlog" / "openhands" / "atif" / "trace" / "merged"
    passed: Optional[bool]  # True / False / None (unknown)
    state_count: int
    tool_count: int
    file_count: int
    model: str  # LLM model name
    task: str  # Task / instance id
    benchmark: str  # Benchmark name


@dataclass
class GroundTruthRecord:
    """Metadata for a built ground truth."""

    gt_id: str
    source_ids: List[str]
    trace_count: int


class Store:
    """Thread-safe in-memory session store."""

    def __init__(self) -> None:
        self._traces: Dict[str, Trace] = {}
        self._meta: Dict[str, TraceRecord] = {}
        self._ground_truth_id: Optional[str] = None

    def add(
        self,
        trace_obj: Trace,
        label: str,
        fmt: str,
        passed: Optional[bool],
    ) -> TraceRecord:
        tid = uuid.uuid4().hex[:12]
        tools = {s.tool_used for s in trace_obj.states.values() if s.tool_used}
        files: set[str] = set()
        for s in trace_obj.states.values():
            files.update(s.files_touched)

        # Extract metadata from trace
        model = _extract_model(trace_obj)
        task = _extract_task(trace_obj, label)
        benchmark = _extract_benchmark(trace_obj, label)

        rec = TraceRecord(
            trace_id=tid,
            label=label,
            format=fmt,
            passed=passed,
            state_count=len(trace_obj.states),
            tool_count=len(tools),
            file_count=len(files),
            model=model,
            task=task,
            benchmark=benchmark,
        )
        self._traces[tid] = trace_obj
        self._meta[tid] = rec
        return rec

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self._traces.get(trace_id)

    def get_record(self, trace_id: str) -> Optional[TraceRecord]:
        return self._meta.get(trace_id)

    def list_records(self) -> List[TraceRecord]:
        return list(self._meta.values())

    def delete(self, trace_id: str) -> bool:
        if trace_id in self._traces:
            del self._traces[trace_id]
            del self._meta[trace_id]
            if self._ground_truth_id == trace_id:
                self._ground_truth_id = None
            return True
        return False

    def set_ground_truth(self, trace_id: str) -> None:
        self._ground_truth_id = trace_id

    @property
    def ground_truth_id(self) -> Optional[str]:
        return self._ground_truth_id

    def get_ground_truth(self) -> Optional[Trace]:
        if self._ground_truth_id:
            return self._traces.get(self._ground_truth_id)
        return None

    def passing_ids(self) -> List[str]:
        return [r.trace_id for r in self._meta.values() if r.passed is True]


# ── Metadata extraction helpers ──────────────────────────────────────────

import re as _re


def _extract_model(trace: Trace) -> str:
    """Best-effort extraction of the LLM model name."""
    # 1. Check trace.metadata (OpenHands generator stores it)
    meta = trace.metadata or {}
    if meta.get("model"):
        return str(meta["model"])

    # 2. Scan log entries for model field
    for state in trace.states.values():
        entry = getattr(state, "log_entry", None)
        if entry is None:
            continue
        raw = getattr(entry, "raw_data", None) or {}
        m = raw.get("model", "") or getattr(entry, "model", "")
        if m and m != "unknown":
            return str(m)

    return ""


def _extract_task(trace: Trace, label: str) -> str:
    """Extract the SWE-bench task / instance id from the label.

    Convention: ``<repo>__<repo>-<issue>-logs-<model>-<pass|fail>-<run>``
    We want ``<repo>__<repo>-<issue>`` (e.g. ``astropy__astropy-12907``).

    For ATIF format, uses the scenario name from trace metadata.

    Uses ``__`` as anchor: extracts ``repo-NNNN`` on the right (non-greedy),
    then infers the owner from the left side using the repo name as a hint.
    """
    meta = trace.metadata or {}

    # ATIF: use scenario name as the task identifier
    if meta.get("source_format") == "atif":
        scenario = meta.get("scenario", "")
        if scenario:
            return scenario

    if "__" not in label:
        return ""
    idx = label.index("__")
    # Right: match 'repo-NNNN' (non-greedy to stop at first issue number)
    m = _re.match(r"([a-zA-Z][a-zA-Z0-9_.-]*?)-(\d+)", label[idx + 2:])
    if not m:
        return ""
    repo, issue = m.group(1), m.group(2)
    # Left: determine owner from the last path segment before __
    left = label[:idx]
    segment = _re.split(r"[/\\\s]+", left)[-1] if left else ""
    if not segment:
        return ""
    # In SWE-bench the owner usually ends with the repo name
    if segment.endswith(repo):
        owner = repo
    elif "-" + repo in segment:
        owner = segment[segment.rindex("-" + repo) + 1:]
    else:
        owner = segment
    return f"{owner}__{repo}-{issue}"


def _extract_benchmark(trace: Trace, label: str) -> str:
    """Infer the benchmark name from format + label hints."""
    meta = trace.metadata or {}
    fmt = meta.get("source_format", "")
    if fmt == "atif":
        agent_name = meta.get("agent_name", "unknown")
        return f"CLI SxS ({agent_name.title()})"
    if fmt == "openhands":
        return "SWE-bench (OpenHands)"
    # evaluation platform format
    if meta.get("generator", "") == "swe_trace_sdk":
        return "SWE-bench (evaluation platform)"
    # Fallback: check label
    lower = label.lower()
    if "swe" in lower or "bench" in lower:
        return "SWE-bench"
    return ""


# Singleton store instance
store = Store()
