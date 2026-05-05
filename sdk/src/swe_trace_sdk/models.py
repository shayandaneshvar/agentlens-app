"""Core data models for SWE agent trace analysis.

This module defines the fundamental data structures used throughout the SDK:

- :class:`LogEntry` — a single log entry (tool call or LLM request) from an
  agent trajectory (e.g. ``chat-export-logs.json`` in evaluation platform format).
- :class:`State` — a node in a trace graph representing an observation/action.
- :class:`Transition` — a directed edge between two states.
- :class:`Trace` — the top-level trace graph produced by loading or merging.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """A single log entry from a coding-agent trajectory.

    Can represent either an LLM ``request`` or a ``toolCall``.
    """

    id: str
    kind: str  # "request" | "toolCall"
    raw_data: Dict[str, Any]
    index: int  # position in the original logs array

    # toolCall fields
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    response: Optional[Any] = None

    # request fields
    model: Optional[str] = None
    response_message: Optional[str] = None

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_log(cls, log_data: Dict[str, Any], index: int) -> LogEntry:
        """Create a :class:`LogEntry` from raw JSON data."""
        entry = cls(
            id=log_data.get("id", f"log_{index}"),
            kind=log_data.get("kind", "unknown"),
            raw_data=log_data,
            index=index,
        )

        if entry.kind == "toolCall":
            entry.tool = log_data.get("tool", "")
            args_raw = log_data.get("args", {})
            if isinstance(args_raw, str):
                try:
                    entry.args = json.loads(args_raw)
                except json.JSONDecodeError:
                    entry.args = {"raw": args_raw}
            else:
                entry.args = args_raw
            entry.response = log_data.get("response", [])

        elif entry.kind == "request":
            metadata = log_data.get("metadata", {})
            entry.model = metadata.get("model", "unknown")
            response = log_data.get("response", {})
            if isinstance(response, dict) and "message" in response:
                msg = response["message"]
                if isinstance(msg, list):
                    entry.response_message = " ".join(str(m) for m in msg)
                else:
                    entry.response_message = str(msg)

        return entry

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_signature(self) -> str:
        """Return a short deterministic signature for this entry."""
        if self.kind == "toolCall":
            key_args: Dict[str, Any] = {}
            if self.args:
                for key in ("filePath", "path", "query", "command"):
                    if key in self.args:
                        val = self.args[key]
                        if isinstance(val, str) and len(val) > 100:
                            val = val[:100] + "..."
                        key_args[key] = val
            return f"toolCall:{self.tool}({json.dumps(key_args, sort_keys=True)})"
        return f"request:{self.model}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        d: Dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "index": self.index,
            "signature": self.get_signature(),
        }
        if self.kind == "toolCall":
            d["tool"] = self.tool
            d["args"] = self.args
            if self.response:
                resp_str = str(self.response)
                d["response_preview"] = resp_str[:500] if len(resp_str) > 500 else resp_str
        elif self.kind == "request":
            d["model"] = self.model
            if self.response_message:
                d["response_preview"] = self.response_message[:500]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogEntry:
        """Create a :class:`LogEntry` from a serialised dictionary."""
        entry = cls(
            id=data.get("id", ""),
            kind=data.get("kind", "unknown"),
            raw_data=data.get("raw_data", {}),
            index=data.get("index", 0),
        )
        if entry.kind == "toolCall":
            entry.tool = data.get("tool")
            entry.args = data.get("args")
            entry.response = data.get("response_preview")
        elif entry.kind == "request":
            entry.model = data.get("model")
            entry.response_message = data.get("response_preview")
        return entry


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class State:
    """A node in a trace graph.

    Attributes
    ----------
    state_id : str
        Unique identifier within the trace.
    step : int
        Sequential step number.
    observation : str
        Text description of what was observed (action-based).
    resulting_state : str
        Normalised description of the *effect* (e.g. ``file_created:readme.md``).
    tool_used : str | None
        Tool name if this state resulted from a tool call.
    files_touched : set[str]
        File paths involved.
    log_entry : LogEntry | None
        Original log entry, if available.
    metadata : dict
        Arbitrary extra data (merge counts, branch info, …).
    """

    state_id: str
    step: int

    log_entry: Optional[LogEntry] = None

    observation: str = ""
    resulting_state: str = ""
    files_touched: Set[str] = field(default_factory=set)
    tool_used: Optional[str] = None

    # Content hash for fine-grained matching (distinguishes different edits to same file)
    content_hash: str = ""
    # Content description for semantic comparison (describes what the change does)
    content_description: str = ""

    # Location and scope information for better discrimination
    file_path: str = ""
    line_range: Optional[tuple] = None  # (start_line, end_line), 1-indexed
    relative_position: str = ""  # "early" / "middle" / "late"
    operation_type: str = ""  # "read" / "create" / "modify" / "delete" / "search" / "terminal" / "other"
    edit_type: str = ""  # "add" / "remove" / "replace" / "expand" / "shrink"
    function_name: str = ""
    class_name: str = ""
    scope_path: str = ""  # "ClassName.method_name" or just "function_name"

    # Intent-stage label — the developer's cognitive intent this action
    # belongs to.  Assigned by the intent-labeling algorithm based on
    # tool type, file context (test vs source), and whether
    # implementation has occurred.
    # One of: "exploration" / "implementation" / "verification" / "orchestration" / ""
    intent_stage: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def get_observation_hash(self) -> str:
        """MD5-based short hash of the observation text."""
        return hashlib.md5(self.observation.encode()).hexdigest()[:16]

    def get_resulting_state_hash(self) -> str:
        """MD5-based short hash of the resulting-state text."""
        if not self.resulting_state:
            return ""
        return hashlib.md5(self.resulting_state.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "step": self.step,
            "observation": self.observation[:1000] if len(self.observation) > 1000 else self.observation,
            "observation_hash": self.get_observation_hash(),
            "resulting_state": self.resulting_state,
            "resulting_state_hash": self.get_resulting_state_hash(),
            "content_hash": self.content_hash,
            "content_description": self.content_description,
            "files_touched": sorted(self.files_touched),
            "tool_used": self.tool_used,
            "file_path": self.file_path,
            "line_range": list(self.line_range) if self.line_range else None,
            "relative_position": self.relative_position,
            "operation_type": self.operation_type,
            "edit_type": self.edit_type,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "scope_path": self.scope_path,
            "intent_stage": self.intent_stage,
            "log_entry": self.log_entry.to_dict() if self.log_entry else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> State:
        line_range_data = data.get("line_range")
        line_range = tuple(line_range_data) if line_range_data else None

        state = cls(
            state_id=data["state_id"],
            step=data["step"],
            observation=data.get("observation", ""),
            resulting_state=data.get("resulting_state", ""),
            content_hash=data.get("content_hash", ""),
            content_description=data.get("content_description", ""),
            files_touched=set(data.get("files_touched", [])),
            tool_used=data.get("tool_used"),
            file_path=data.get("file_path", ""),
            line_range=line_range,
            relative_position=data.get("relative_position", ""),
            operation_type=data.get("operation_type", ""),
            edit_type=data.get("edit_type", ""),
            function_name=data.get("function_name", ""),
            class_name=data.get("class_name", ""),
            scope_path=data.get("scope_path", ""),
            intent_stage=data.get("intent_stage", data.get("phase", "")),
            metadata=data.get("metadata", {}),
        )

        log_entry_data = data.get("log_entry")
        if log_entry_data:
            state.log_entry = LogEntry.from_dict(log_entry_data)

        return state


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """A directed edge between two :class:`State` objects."""

    transition_id: str
    from_state: str
    to_state: str
    action_type: str  # e.g. "create_file", "request"
    action_data: Dict[str, Any] = field(default_factory=dict)
    step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "action_type": self.action_type,
            "action_data": self.action_data,
            "step": self.step,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Transition:
        return cls(
            transition_id=data["transition_id"],
            from_state=data["from_state"],
            to_state=data["to_state"],
            action_type=data["action_type"],
            action_data=data.get("action_data", {}),
            step=data.get("step", 0),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Trace (was "PTA" in the prototype)
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    """A trace graph (directed acyclic graph) of coding-agent execution.

    A *Trace* can be either:
    * a **single-run trace** produced by :func:`swe_trace_sdk.trace.load`, or
    * a **merged trace** (ground truth) built by :func:`swe_trace_sdk.trace.merge`.

    Attributes
    ----------
    initial_state : str | None
        ID of the entry-point state (set automatically on first
        :meth:`add_state` call).
    states : dict[str, State]
        All states keyed by ``state_id``.
    transitions : list[Transition]
        All edges.
    metadata : dict
        Arbitrary metadata (source file, merge stats, …).
    branches : dict[str, list[str]]
        Branch-point state → list of alternative target state IDs.
    """

    initial_state: Optional[str] = None
    states: Dict[str, State] = field(default_factory=dict)
    transitions: List[Transition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    branches: Dict[str, List[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_state(self, state: State) -> None:
        """Add a state.  The first state added becomes :attr:`initial_state`."""
        self.states[state.state_id] = state
        if self.initial_state is None:
            self.initial_state = state.state_id

    def add_transition(self, transition: Transition) -> None:
        self.transitions.append(transition)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_outgoing_transitions(self, state_id: str) -> List[Transition]:
        return [t for t in self.transitions if t.from_state == state_id]

    def get_incoming_transitions(self, state_id: str) -> List[Transition]:
        return [t for t in self.transitions if t.to_state == state_id]

    def get_terminal_states(self) -> List[str]:
        """Return IDs of states with no outgoing transitions."""
        with_outgoing = {t.from_state for t in self.transitions}
        return [sid for sid in self.states if sid not in with_outgoing]

    def get_tool_sequence(self) -> List[str]:
        """Tool names in step order (for linear/single-path traces)."""
        tools: List[str] = []
        for t in sorted(self.transitions, key=lambda x: x.step):
            if t.action_type not in ("request", "unknown"):
                tools.append(t.action_type)
        return tools

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_state": self.initial_state,
            "states": {sid: s.to_dict() for sid, s in self.states.items()},
            "transitions": [t.to_dict() for t in self.transitions],
            "metadata": self.metadata,
            "branches": self.branches,
            "statistics": {
                "num_states": len(self.states),
                "num_transitions": len(self.transitions),
                "terminal_states": self.get_terminal_states(),
                "tool_sequence": self.get_tool_sequence(),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Trace:
        tr = cls(
            initial_state=data.get("initial_state"),
            metadata=data.get("metadata", {}),
            branches=data.get("branches", {}),
        )
        for sid, sdata in data.get("states", {}).items():
            tr.states[sid] = State.from_dict(sdata)
        for tdata in data.get("transitions", []):
            tr.transitions.append(Transition.from_dict(tdata))
        return tr

    # ------------------------------------------------------------------
    # File I/O (convenience)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the trace to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> Trace:
        """Read a trace from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Copying
    # ------------------------------------------------------------------

    def copy(self) -> Trace:
        """Return a deep copy."""
        return deepcopy(self)


# ---------------------------------------------------------------------------
# Quality assessment result types
# ---------------------------------------------------------------------------

@dataclass
class FailureReason:
    """A single reason explaining why a trajectory failed."""

    reason: str
    """Short label, e.g. ``"missing_verification"``."""

    detail: str
    """Human-readable explanation."""

    severity: str
    """``"critical"`` / ``"high"`` / ``"medium"`` / ``"low"``."""

    def to_dict(self) -> Dict[str, Any]:
        return {"reason": self.reason, "detail": self.detail, "severity": self.severity}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureReason":
        return cls(reason=data["reason"], detail=data["detail"], severity=data["severity"])


@dataclass
class DivergencePoint:
    """Where the candidate first diverges from ground truth."""

    step: int
    """Candidate step number where divergence occurs."""

    description: str
    """What the candidate did at the divergence point."""

    expected_next: str
    """What the ground truth expected next."""

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "description": self.description, "expected_next": self.expected_next}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DivergencePoint":
        return cls(step=data["step"], description=data["description"], expected_next=data["expected_next"])


@dataclass
class StageCoverageDetail:
    """Coverage detail for a single intent stage."""

    matched: int
    total: int
    percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {"matched": self.matched, "total": self.total, "percent": round(self.percent, 2)}


@dataclass
class DivergenceSegment:
    """A contiguous block of ground-truth states the candidate missed."""

    start_step: int
    """1-indexed start step in the GT path."""

    end_step: int
    """1-indexed end step in the GT path (inclusive)."""

    expected_states: List[Dict[str, str]] = field(default_factory=list)
    """GT states in this gap: [{tool, file_path, intent_stage, resulting_state}]."""

    candidate_activity: List[Dict[str, str]] = field(default_factory=list)
    """What the candidate was doing in this range: [{tool, file_path, intent_stage}]."""

    stage_context: str = ""
    """Dominant intent stage of the missed GT states."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "expected_states": self.expected_states,
            "candidate_activity": self.candidate_activity,
            "stage_context": self.stage_context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DivergenceSegment":
        return cls(**data)


@dataclass
class StageComparison:
    """Detailed comparison of candidate vs GT for a single intent stage."""

    expected_steps: List[Dict[str, str]] = field(default_factory=list)
    """GT states in this stage: [{tool, file_path, resulting_state}]."""

    matched_steps: List[Dict[str, str]] = field(default_factory=list)
    """Candidate states that matched GT in this stage."""

    missing_steps: List[Dict[str, str]] = field(default_factory=list)
    """GT states with no candidate match."""

    extra_steps: List[Dict[str, str]] = field(default_factory=list)
    """Candidate states in this stage with no GT match."""

    ordering_preserved: bool = True
    """Whether matched steps appear in the same relative order as GT."""

    effort_ratio: float = 1.0
    """Candidate states in stage / GT states in stage."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_steps": self.expected_steps,
            "matched_steps": self.matched_steps,
            "missing_steps": self.missing_steps,
            "extra_steps": self.extra_steps,
            "ordering_preserved": self.ordering_preserved,
            "effort_ratio": round(self.effort_ratio, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageComparison":
        return cls(**data)


@dataclass
class RetryLoop:
    """A detected retry loop (≥3 consecutive identical actions)."""
    start_step: int
    end_step: int
    tool: str
    file_path: str
    count: int

    def to_dict(self) -> Dict[str, Any]:
        return {"start_step": self.start_step, "end_step": self.end_step,
                "tool": self.tool, "file_path": self.file_path, "count": self.count}


@dataclass
class Backtrack:
    """A detected stage regression."""
    step: int
    from_stage: str
    to_stage: str

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "from_stage": self.from_stage, "to_stage": self.to_stage}


@dataclass
class RedundantStep:
    """A candidate step that duplicates a nearby matched step."""
    step: int
    tool: str
    file_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "tool": self.tool, "file_path": self.file_path}


@dataclass
class UnnecessaryExploration:
    """Exploration occurring after implementation when GT has none."""
    step: int
    tool: str
    file_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "tool": self.tool, "file_path": self.file_path}


@dataclass
class CyclicPattern:
    """A detected multi-step repeating cycle (e.g. edit→test repeated ≥2 times)."""
    start_step: int
    end_step: int
    pattern_length: int
    repetitions: int
    pattern_signature: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_step": self.start_step, "end_step": self.end_step,
            "pattern_length": self.pattern_length, "repetitions": self.repetitions,
            "pattern_signature": self.pattern_signature,
        }


@dataclass
class ToolInefficiency:
    """Per-tool breakdown of wasted steps."""
    tool: str
    retries: int = 0
    backtracks: int = 0
    cycles: int = 0
    redundant: int = 0
    unnecessary: int = 0
    total_wasted: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool, "retries": self.retries,
            "backtracks": self.backtracks, "cycles": self.cycles,
            "redundant": self.redundant, "unnecessary": self.unnecessary,
            "total_wasted": self.total_wasted,
        }


@dataclass
class InefficiencyReport:
    """Detected inefficiency patterns in a trajectory."""

    retry_loops: List[RetryLoop] = field(default_factory=list)
    backtracks: List[Backtrack] = field(default_factory=list)
    redundant_steps: List[RedundantStep] = field(default_factory=list)
    unnecessary_explorations: List[UnnecessaryExploration] = field(default_factory=list)
    cyclic_patterns: List[CyclicPattern] = field(default_factory=list)

    retry_loop_count: int = 0
    backtrack_count: int = 0
    redundant_step_count: int = 0
    unnecessary_exploration_count: int = 0
    cyclic_pattern_count: int = 0
    total_wasted_steps: int = 0

    severity_score: float = 0.0
    """Fraction of trajectory steps that were wasted (0.0–1.0)."""

    wasted_input_tokens: int = 0
    wasted_output_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    per_tool_breakdown: List[ToolInefficiency] = field(default_factory=list)
    """Per-tool aggregation of wasted steps, sorted by total_wasted desc."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retry_loops": [r.to_dict() for r in self.retry_loops],
            "backtracks": [b.to_dict() for b in self.backtracks],
            "redundant_steps": [r.to_dict() for r in self.redundant_steps],
            "unnecessary_explorations": [u.to_dict() for u in self.unnecessary_explorations],
            "cyclic_patterns": [c.to_dict() for c in self.cyclic_patterns],
            "retry_loop_count": self.retry_loop_count,
            "backtrack_count": self.backtrack_count,
            "redundant_step_count": self.redundant_step_count,
            "unnecessary_exploration_count": self.unnecessary_exploration_count,
            "cyclic_pattern_count": self.cyclic_pattern_count,
            "total_wasted_steps": self.total_wasted_steps,
            "severity_score": round(self.severity_score, 3),
            "wasted_input_tokens": self.wasted_input_tokens,
            "wasted_output_tokens": self.wasted_output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "per_tool_breakdown": [t.to_dict() for t in self.per_tool_breakdown],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InefficiencyReport":
        return cls(
            retry_loops=[RetryLoop(**r) for r in data.get("retry_loops", [])],
            backtracks=[Backtrack(**b) for b in data.get("backtracks", [])],
            redundant_steps=[RedundantStep(**r) for r in data.get("redundant_steps", [])],
            unnecessary_explorations=[UnnecessaryExploration(**u) for u in data.get("unnecessary_explorations", [])],
            cyclic_patterns=[CyclicPattern(**c) for c in data.get("cyclic_patterns", [])],
            retry_loop_count=data.get("retry_loop_count", 0),
            backtrack_count=data.get("backtrack_count", 0),
            redundant_step_count=data.get("redundant_step_count", 0),
            unnecessary_exploration_count=data.get("unnecessary_exploration_count", 0),
            cyclic_pattern_count=data.get("cyclic_pattern_count", 0),
            total_wasted_steps=data.get("total_wasted_steps", 0),
            severity_score=data.get("severity_score", 0.0),
            wasted_input_tokens=data.get("wasted_input_tokens", 0),
            wasted_output_tokens=data.get("wasted_output_tokens", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            per_tool_breakdown=[ToolInefficiency(**t) for t in data.get("per_tool_breakdown", [])],
        )


@dataclass
class QualitySignal:
    """A high-level quality indicator derived from multiple metrics."""

    signal_type: str
    """E.g. ``"inefficient_path_despite_success"``, ``"failure_from_missing_verification"``."""

    description: str
    """Human-readable explanation."""

    severity: str
    """``"info"`` / ``"warning"`` / ``"critical"``."""

    evidence: List[str] = field(default_factory=list)
    """Supporting data points."""

    def to_dict(self) -> Dict[str, Any]:
        return {"signal_type": self.signal_type, "description": self.description,
                "severity": self.severity, "evidence": self.evidence}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualitySignal":
        return cls(**data)


@dataclass
class QualityReport:
    """Result of :func:`~swe_trace_sdk.match.quality_assessment`.

    Answers: Why is it failing? Is a pass lucky or ideal?
    """

    verdict: str
    """``"PASS"`` / ``"LIKELY PASS"`` / ``"UNCERTAIN"`` / ``"LIKELY FAIL"`` / ``"FAIL"``."""

    quality_tier: str
    """``"ideal"`` / ``"solid"`` / ``"lucky"`` / ``"partial_fail"`` / ``"off_track"``."""

    quality_score: int
    """Composite quality score (0–100) for ranking within a cohort."""

    failure_reasons: List[FailureReason] = field(default_factory=list)
    """Why the trajectory failed (empty for passes)."""

    strengths: List[str] = field(default_factory=list)
    """What the agent did well, even in failures."""

    divergence_point: Optional[DivergencePoint] = None
    """Where the candidate first diverged from ground truth."""

    stage_coverage: Dict[str, StageCoverageDetail] = field(default_factory=dict)
    """Per-stage coverage breakdown (E / I / V / O)."""

    key_metrics: Dict[str, float] = field(default_factory=dict)
    """The 4 key numbers: coverage_percent, coherence, stage_completeness, workflow_similarity."""

    divergence_points: List[DivergenceSegment] = field(default_factory=list)
    """All divergence segments where candidate missed GT states."""

    stage_comparison: Dict[str, StageComparison] = field(default_factory=dict)
    """Detailed per-stage comparison (expected/matched/missing/extra steps)."""

    inefficiencies: Optional[InefficiencyReport] = None
    """Detected inefficiency patterns (retry loops, backtracks, redundant steps)."""

    quality_signals: List[QualitySignal] = field(default_factory=list)
    """High-level quality indicators derived from multiple metrics."""

    stage_order_match: bool = True
    """Whether the candidate's stage-entry order matches the GT."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "quality_tier": self.quality_tier,
            "quality_score": self.quality_score,
            "failure_reasons": [fr.to_dict() for fr in self.failure_reasons],
            "strengths": self.strengths,
            "divergence_point": self.divergence_point.to_dict() if self.divergence_point else None,
            "stage_coverage": {k: v.to_dict() for k, v in self.stage_coverage.items()},
            "key_metrics": {k: round(v, 4) for k, v in self.key_metrics.items()},
            "divergence_points": [d.to_dict() for d in self.divergence_points],
            "stage_comparison": {k: v.to_dict() for k, v in self.stage_comparison.items()},
            "inefficiencies": self.inefficiencies.to_dict() if self.inefficiencies else None,
            "quality_signals": [s.to_dict() for s in self.quality_signals],
            "stage_order_match": self.stage_order_match,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityReport":
        ineff_data = data.get("inefficiencies")
        return cls(
            verdict=data["verdict"],
            quality_tier=data["quality_tier"],
            quality_score=data["quality_score"],
            failure_reasons=[FailureReason.from_dict(fr) for fr in data.get("failure_reasons", [])],
            strengths=data.get("strengths", []),
            divergence_point=DivergencePoint.from_dict(data["divergence_point"]) if data.get("divergence_point") else None,
            stage_coverage={k: StageCoverageDetail(**v) for k, v in data.get("stage_coverage", {}).items()},
            key_metrics=data.get("key_metrics", {}),
            divergence_points=[DivergenceSegment.from_dict(d) for d in data.get("divergence_points", [])],
            stage_comparison={k: StageComparison.from_dict(v) for k, v in data.get("stage_comparison", {}).items()},
            inefficiencies=InefficiencyReport.from_dict(ineff_data) if ineff_data else None,
            quality_signals=[QualitySignal.from_dict(s) for s in data.get("quality_signals", [])],
            stage_order_match=data.get("stage_order_match", True),
        )


@dataclass
class CohortEntry:
    """A single trajectory's position within a ranked cohort."""

    label: str
    quality_score: int
    quality_tier: str
    rank: int
    top_failure_reason: str = ""
    """Short reason label (only for failing entries)."""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "label": self.label,
            "quality_score": self.quality_score,
            "quality_tier": self.quality_tier,
            "rank": self.rank,
        }
        if self.top_failure_reason:
            d["top_failure_reason"] = self.top_failure_reason
        return d


@dataclass
class CohortRanking:
    """Result of :func:`~swe_trace_sdk.match.rank_in_cohort`.

    Ranks multiple trajectories within their pass/fail cohorts.
    """

    passing: List[CohortEntry] = field(default_factory=list)
    """Passing trajectories sorted by quality_score (best first)."""

    failing: List[CohortEntry] = field(default_factory=list)
    """Failing trajectories sorted by quality_score (best first)."""

    summary: Dict[str, Any] = field(default_factory=dict)
    """Aggregate stats: ideal_count, lucky_count, …, common_failure_reasons."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passing": [e.to_dict() for e in self.passing],
            "failing": [e.to_dict() for e in self.failing],
            "summary": self.summary,
        }
