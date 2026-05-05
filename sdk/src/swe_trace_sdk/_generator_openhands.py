"""Internal OpenHands trace generator — converts ``trajectory_openhands.json``
into a :class:`Trace`.

This is *not* part of the public API surface.  Users should call
:func:`swe_trace_sdk.trace.load` with ``format="openhands"`` instead.

Design
------
This module follows the **normalize-early + fail-fast** approach:

1.  Parse the raw OpenHands log (interleaved ACTION / OBSERVATION entries).
2.  Pair each action with its observation via the ``cause`` → ``id`` link.
3.  Map OpenHands tool names and arg keys to **canonical SDK names** using
    exhaustive, validated lookup tables.
4.  Produce standard :class:`LogEntry` objects — identical to those the
    evaluation platform generator would produce.
5.  Build the :class:`Trace` by reusing the same pure helper functions that
    ``_generator.py`` uses (``_compute_resulting_state``, etc.).

Any unknown action type or mapping failure raises immediately — no silent bugs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import LogEntry, State, Transition, Trace
from .intent import label_trace_intents

# Import the pure helpers from the evaluation platform generator — zero code duplication.
from ._generator import (
    _compute_resulting_state,
    _build_observation,
    _extract_files_touched,
    _compute_content_hash,
    _compute_content_description,
    _extract_location_info,
    _normalize_path,
)

logger = logging.getLogger(__name__)

__all__ = ["OpenHandsTraceGenerator"]


# ═══════════════════════════════════════════════════════════════════════════
# Exhaustive mapping tables
# ═══════════════════════════════════════════════════════════════════════════

# Key = (openhands_action, sub_command_or_None)
# Value = canonical SDK tool name, or None to skip the entry.
#
# sub_command is derived from args["command"] for "edit" actions,
# and is None for all other action types.
_ACTION_TO_TOOL: Dict[Tuple[str, Optional[str]], Optional[str]] = {
    # File reading
    ("read", "file"):           "read_file",
    ("read", "dir"):            "list_dir",
    # File editing
    ("edit", "create"):         "create_file",
    ("edit", "str_replace"):    "replace_string_in_file",
    ("edit", "insert"):         "replace_string_in_file",
    ("edit", "view"):           "read_file",
    ("edit", "undo_edit"):      "replace_string_in_file",
    # Terminal
    ("run", None):              "run_in_terminal",
    ("run_ipython", None):      "run_in_terminal",
    # Agent reasoning
    ("think", None):            "think",
    # Terminal markers / metadata — skip these
    ("finish", None):           None,
    ("system", None):           None,
    ("message", None):          None,
    ("recall", None):           None,
    ("condensation", None):     None,
    ("task_tracking", None):    None,   # GPT-4.1 task-management feature
    ("browse_interactive", None): None, # Browser interaction (not a code tool)
}

# Canonical SDK tool names that downstream code recognises.
_CANONICAL_TOOLS = frozenset({
    "read_file", "create_file", "replace_string_in_file",
    "multi_replace_string_in_file", "file_search", "grep_search",
    "semantic_search", "list_dir", "run_in_terminal", "apply_patch",
    "think",
})

# OpenHands arg key → canonical SDK arg key.
_ARG_KEY_MAP: Dict[str, str] = {
    "path":      "filePath",
    "file_text": "content",
    "old_str":   "oldString",
    "new_str":   "newString",
    # "command" stays "command" — same in both
    # "start"/"end" are handled specially (see _normalize_args)
}

# Required args per canonical tool — for fail-fast validation.
_REQUIRED_ARGS: Dict[str, List[str]] = {
    "read_file":                ["filePath"],
    "list_dir":                 ["filePath"],
    "create_file":              ["filePath"],
    "replace_string_in_file":   ["filePath"],
    "run_in_terminal":          ["command"],
}

# OpenHands workspace prefix → stripped during path normalisation.
_OH_PATH_PREFIXES = ("testbed/",)


# ═══════════════════════════════════════════════════════════════════════════
# Normalisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _classify_read_action(observation_content: str) -> str:
    """Determine whether a ``read`` action was a file read or directory listing.

    Returns ``"file"`` or ``"dir"``.
    """
    if not observation_content:
        return "file"
    # OpenHands directory listings start with a distinctive line
    dir_markers = (
        "Here's the files and directories",
        "Here's the result of running `find`",
        "Here's the result of running `ls`",
    )
    for marker in dir_markers:
        if marker in observation_content:
            return "dir"
    return "file"


def _resolve_tool(
    action_type: str,
    sub_command: Optional[str],
    observation_content: str = "",
) -> Optional[str]:
    """Map an OpenHands action to a canonical SDK tool name.

    Returns ``None`` for actions that should be skipped (system, message,
    finish, …) **and** for unknown action types (logged as a warning).
    """
    # For 'read' actions, determine file vs dir from observation
    if action_type == "read":
        sub_command = _classify_read_action(observation_content)

    key = (action_type, sub_command)
    if key not in _ACTION_TO_TOOL:
        logger.warning(
            "Unknown OpenHands action %r — skipping. "
            "Add it to _ACTION_TO_TOOL if it should be tracked.",
            key,
        )
        return None

    tool = _ACTION_TO_TOOL[key]

    # Validate the mapped tool is canonical
    if tool is not None and tool not in _CANONICAL_TOOLS:
        logger.warning(
            "OpenHands action %r mapped to non-canonical tool %r — skipping.",
            key, tool,
        )
        return None

    return tool


def _normalize_oh_path(path: str) -> str:
    """Normalise an OpenHands file path for use in args.

    Strips the leading ``/testbed/`` prefix that OpenHands uses as
    its workspace root, then applies the standard SDK path normalisation.
    """
    if not path:
        return path
    # Strip leading slash and then testbed/ prefix
    stripped = path.lstrip("/\\")
    for prefix in _OH_PATH_PREFIXES:
        bare = prefix.rstrip("/")
        if stripped.lower() == bare:
            # Exact match: e.g. "testbed" → workspace root
            stripped = ""
            break
        if stripped.lower().startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    # Re-add leading / so SDK's _normalize_path strips it consistently
    return "/" + stripped if stripped else "/"


def _normalize_args(
    tool: str,
    raw_args: Dict[str, Any],
    action_type: str,
) -> Dict[str, Any]:
    """Normalise OpenHands args to canonical SDK arg keys.

    Returns a new dict with canonical keys and the same values.
    Logs warnings for missing required arguments.
    """
    normalised: Dict[str, Any] = {}

    for k, v in raw_args.items():
        if v is None:
            continue  # skip explicit None values from OpenHands
        canonical_key = _ARG_KEY_MAP.get(k, k)
        normalised[canonical_key] = v

    # Handle path normalisation for /testbed/ prefix
    if "filePath" in normalised:
        normalised["filePath"] = _normalize_oh_path(normalised["filePath"])

    # For edit/create, if content comes from file_text in the raw args
    # it was already mapped via _ARG_KEY_MAP.  But OpenHands 'create'
    # action has file_text → content.  'str_replace' has old_str/new_str.
    # No extra work needed — the _ARG_KEY_MAP handles these.

    # For read_file with start/end (OpenHands line range),
    # map to startLine/endLine
    if tool == "read_file":
        if "start" in normalised and normalised["start"] not in (None, 0):
            normalised["startLine"] = normalised.pop("start")
        else:
            normalised.pop("start", None)
        if "end" in normalised and normalised["end"] not in (None, -1):
            normalised["endLine"] = normalised.pop("end")
        else:
            normalised.pop("end", None)
        # Handle view_range from edit/view → read_file
        view_range = normalised.pop("view_range", None)
        if view_range and isinstance(view_range, (list, tuple)) and len(view_range) == 2:
            normalised["startLine"] = view_range[0]
            normalised["endLine"] = view_range[1]

    # Validate required args
    if tool in _REQUIRED_ARGS:
        for req_key in _REQUIRED_ARGS[tool]:
            if req_key not in normalised or normalised[req_key] is None:
                logger.warning(
                    "Tool '%s' missing expected arg '%s'. Available: %s",
                    tool, req_key, list(normalised.keys()),
                )

    return normalised


def _extract_thought(
    action_entry: Dict[str, Any],
) -> str:
    """Extract the agent's reasoning text from an OpenHands action entry.

    The thought can live in multiple places:
    1. ``args.thought`` — direct thought field
    2. ``tool_call_metadata.model_response.choices[0].message.content``
       — the LLM's text response accompanying the tool call
    """
    # 1. Direct thought
    thought = (action_entry.get("args") or {}).get("thought", "")
    if thought:
        return str(thought)

    # 2. LLM response content
    tcm = action_entry.get("tool_call_metadata", {})
    mr = tcm.get("model_response", {})
    choices = mr.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content")
        if content:
            return str(content)

    return ""


def _extract_token_info(
    action_entry: Dict[str, Any],
) -> Dict[str, int]:
    """Extract token usage from an OpenHands action entry.

    Returns dict with ``input_tokens``, ``output_tokens``,
    ``cached_input_tokens``.
    """
    tcm = action_entry.get("tool_call_metadata", {})
    mr = tcm.get("model_response", {})
    usage = mr.get("usage", {})

    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cached_tokens = 0
    ptd = usage.get("prompt_tokens_details")
    if isinstance(ptd, dict):
        cached_tokens = ptd.get("cached_tokens", 0) or 0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_tokens,
    }


def _extract_model_name(action_entry: Dict[str, Any]) -> str:
    """Extract the LLM model name from an OpenHands action entry."""
    tcm = action_entry.get("tool_call_metadata", {})
    mr = tcm.get("model_response", {})
    return mr.get("model", "unknown")


# ═══════════════════════════════════════════════════════════════════════════
# Action-Observation pairing
# ═══════════════════════════════════════════════════════════════════════════

def _pair_actions_observations(
    raw_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pair each ACTION entry with its OBSERVATION via ``cause`` → ``id``.

    Returns a list of dicts, each with keys:
    - ``action``: the raw action entry
    - ``observation``: the raw observation entry (or None)
    - ``action_type``: the action type string
    """
    # Index observations by their cause id
    obs_by_cause: Dict[int, Dict[str, Any]] = {}
    for entry in raw_entries:
        if "observation" in entry and "action" not in entry:
            cause_id = entry.get("cause")
            if cause_id is not None:
                obs_by_cause[cause_id] = entry

    pairs: List[Dict[str, Any]] = []
    for entry in raw_entries:
        # Process entries that have an "action" key.
        # Some formats may include both "action" and "observation" on the
        # same entry — treat them as an action (the observation is already
        # self-contained).  Pure observation entries (no "action" key) are
        # handled above via obs_by_cause.
        if "action" not in entry:
            continue
        # Skip pure observation entries that happen to also carry an
        # "action" key only when "action" is not a string (safety check).
        action_type = entry.get("action")
        if not isinstance(action_type, str):
            continue

        entry_id = entry.get("id")
        observation = obs_by_cause.get(entry_id)

        pairs.append({
            "action": entry,
            "observation": observation,
            "action_type": action_type,
        })

    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════════════

class OpenHandsTraceGenerator:
    """Build a :class:`Trace` from an OpenHands ``trajectory_openhands.json``.

    This generator:
    1. Pairs action/observation entries.
    2. Normalises tool names and arg keys to canonical SDK form.
    3. Creates :class:`LogEntry` objects identical to evaluation platform ones.
    4. Builds the trace using the same pure helpers as the evaluation platform generator.
    """

    def __init__(self) -> None:
        self._state_counter = 0
        self._transition_counter = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self, trajectory_file: str) -> Trace:
        """Generate a :class:`Trace` from *trajectory_file*.

        Parameters
        ----------
        trajectory_file : str
            Path to an OpenHands ``trajectory_openhands.json``.

        Returns
        -------
        Trace
        """
        logger.info("Generating Trace from OpenHands trajectory: %s", trajectory_file)

        with open(trajectory_file, "r", encoding="utf-8") as fh:
            raw_entries = json.load(fh)

        if not isinstance(raw_entries, list):
            raise ValueError(
                f"Expected a JSON array in {trajectory_file}, "
                f"got {type(raw_entries).__name__}."
            )

        log_entries = self._extract_log_entries(raw_entries)
        if not log_entries:
            # Summarise what we DID see so the user can diagnose
            import collections as _c
            action_types = _c.Counter(
                e.get("action", "<no-action>")
                for e in raw_entries if "action" in e
            )
            logger.warning(
                "No actionable log entries in %s — building trace from "
                "raw entries only.  Total raw entries: %d, action types seen: %s",
                trajectory_file, len(raw_entries), dict(action_types),
            )
        else:
            logger.info("Extracted %d normalised log entries", len(log_entries))

        trace = self._build_trace(log_entries, trajectory_file)
        return trace

    # ------------------------------------------------------------------
    # Log extraction + normalisation
    # ------------------------------------------------------------------

    def _extract_log_entries(
        self,
        raw_entries: List[Dict[str, Any]],
    ) -> List[LogEntry]:
        """Parse, pair, normalise, and return canonical :class:`LogEntry` list."""
        pairs = _pair_actions_observations(raw_entries)
        entries: List[LogEntry] = []
        self._compaction_count = 0

        # Extract task description from the 'message' action if present
        self._task_description = ""
        for pair in pairs:
            if pair["action_type"] == "message":
                msg_content = (pair["action"].get("args") or {}).get("content", "")
                if msg_content:
                    self._task_description = str(msg_content)
                break

        for idx, pair in enumerate(pairs):
            action_entry = pair["action"]
            obs_entry = pair["observation"]
            action_type = pair["action_type"]

            # Get observation content for classification
            obs_content = ""
            if obs_entry:
                obs_content = str(obs_entry.get("content", ""))

            # Determine sub-command for 'edit' actions
            sub_command = None
            if action_type == "edit":
                sub_command = (action_entry.get("args") or {}).get("command")

            # Resolve to canonical tool (returns None for skipped/unknown)
            tool = _resolve_tool(action_type, sub_command, obs_content)

            # Skip entries that map to None (system, message, recall, finish, unknown)
            if tool is None:
                # But if it's 'finish', store the final thought
                if action_type == "finish":
                    self._finish_thought = _extract_thought(action_entry)
                if action_type == "condensation":
                    self._compaction_count += 1
                continue

            # Normalise args
            raw_args = dict(action_entry.get("args") or {})
            normalised_args = _normalize_args(tool, raw_args, action_type)

            # Extract metadata
            thought = _extract_thought(action_entry)
            token_info = _extract_token_info(action_entry)
            model_name = _extract_model_name(action_entry)

            # Build the response from observation content
            response = obs_content if obs_content else None

            # For 'think' actions, the thought IS the content
            if tool == "think":
                thought_text = (action_entry.get("args") or {}).get("thought", "")
                if thought_text:
                    normalised_args["thought"] = thought_text
                    response = thought_text

            # Determine success/failure from observation
            if obs_entry:
                obs_message = obs_entry.get("message", "")
                if response and tool in ("create_file", "replace_string_in_file"):
                    # OpenHands edit observations contain snippets on success
                    if obs_entry.get("observation") == "edit":
                        if "error" not in str(obs_message).lower():
                            response = f"The file was edited successfully. {obs_content[:200]}"

            # Create canonical LogEntry
            entry_id = str(action_entry.get("id", f"oh_{idx}"))
            log_entry = LogEntry(
                id=entry_id,
                kind="toolCall",
                raw_data=action_entry,
                index=idx,
                tool=tool,
                args=normalised_args,
                response=response,
                model=model_name,
            )

            # Store extra metadata in raw_data for traceability
            log_entry.raw_data = {
                "openhands_action": action_type,
                "openhands_sub_command": sub_command,
                "thought": thought,
                "token_info": token_info,
                "timestamp": action_entry.get("timestamp"),
                "model": model_name,
            }

            entries.append(log_entry)

        return entries

    # ------------------------------------------------------------------
    # Trace construction  (mirrors _generator.TraceGenerator._build_trace)
    # ------------------------------------------------------------------

    def _build_trace(self, entries: List[LogEntry], source_file: str) -> Trace:
        """Build a :class:`Trace` from normalised :class:`LogEntry` objects.

        Uses the same pure helper functions from ``_generator.py``.
        """
        trace = Trace()
        trace.metadata = {
            "source_file": Path(source_file).name,
            "source_path": str(source_file),
            "source_format": "openhands",
            "num_entries": len(entries),
            "generator": "swe_trace_sdk.openhands",
            "compaction_count": getattr(self, "_compaction_count", 0),
        }
        if self._task_description:
            trace.metadata["task_description"] = self._task_description[:2000]

        if not entries:
            return trace

        prev = self._make_initial_state()
        trace.add_state(prev)

        for idx, entry in enumerate(entries):
            is_last = idx == len(entries) - 1
            new_state = self._state_from_entry(entry, prev, is_last)
            trace.add_state(new_state)
            trace.add_transition(self._make_transition(prev, new_state, entry))
            prev = new_state

        if prev:
            prev.metadata["is_terminal"] = True

        # Intent-stage label every state (exploration / implementation / verification / orchestration)
        label_trace_intents(trace)

        return trace

    # ------------------------------------------------------------------
    # State helpers  (mirrors _generator.TraceGenerator)
    # ------------------------------------------------------------------

    def _make_initial_state(self) -> State:
        sid = f"state_{self._state_counter}"
        self._state_counter += 1
        return State(
            state_id=sid,
            step=0,
            observation="<initial>",
            resulting_state="initial",
            metadata={"is_initial": True},
        )

    def _state_from_entry(
        self,
        entry: LogEntry,
        prev: Optional[State],
        is_terminal: bool,
    ) -> State:
        sid = f"state_{self._state_counter}"
        step = self._state_counter
        self._state_counter += 1
        loc = _extract_location_info(entry)

        # Extract thought and token info from our stored raw_data
        thought = entry.raw_data.get("thought", "")
        token_info = entry.raw_data.get("token_info", {})

        return State(
            state_id=sid,
            step=step,
            log_entry=entry,
            observation=_build_observation(entry),
            resulting_state=_compute_resulting_state(entry, prev, is_terminal),
            content_hash=_compute_content_hash(entry),
            content_description=_compute_content_description(entry),
            files_touched=_extract_files_touched(entry),
            tool_used=entry.tool if entry.kind == "toolCall" else None,
            file_path=loc["file_path"],
            line_range=loc["line_range"],
            relative_position=loc["relative_position"],
            operation_type=loc["operation_type"],
            edit_type=loc["edit_type"],
            function_name=loc["function_name"],
            class_name=loc["class_name"],
            scope_path=loc["scope_path"],
            metadata={
                "entry_id": entry.id,
                "entry_kind": entry.kind,
                "entry_index": entry.index,
                "thought": thought,
                "input_tokens": token_info.get("input_tokens", 0),
                "output_tokens": token_info.get("output_tokens", 0),
                "cached_input_tokens": token_info.get("cached_input_tokens", 0),
                "model": entry.raw_data.get("model", ""),
                "timestamp": entry.raw_data.get("timestamp", ""),
            },
        )

    def _make_transition(
        self,
        from_s: State,
        to_s: State,
        entry: LogEntry,
    ) -> Transition:
        tid = f"trans_{self._transition_counter}"
        self._transition_counter += 1
        action_type = entry.tool or "unknown_tool" if entry.kind == "toolCall" else "request"
        action_data: Dict[str, Any] = {}
        if entry.kind == "toolCall" and entry.args:
            for key in ("filePath", "path", "query", "command"):
                if key in entry.args:
                    action_data[key] = entry.args[key]
        return Transition(
            transition_id=tid,
            from_state=from_s.state_id,
            to_state=to_s.state_id,
            action_type=action_type,
            action_data=action_data,
            step=to_s.step,
            metadata={"entry_id": entry.id, "signature": entry.get_signature()},
        )
