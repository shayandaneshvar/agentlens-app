"""Internal ATIF trace generator — converts ATIF v1.6 ``trajectory.json``
into a :class:`Trace`.

This is *not* part of the public API surface.  Users should call
:func:`swe_trace_sdk.trace.load` with ``format="atif"`` instead.

Design
------
This module follows the **normalize-early + fail-fast** approach used by the
OpenHands generator:

1.  Parse the ATIF v1.6 JSON (``schema_version``, ``steps``, ``agent``,
    ``final_metrics``, ``extra``).
2.  Filter to agent steps that carry ``tool_calls``.
3.  Map ATIF tool names (Copilot CLI *and* Claude Code) and arg keys to
    **canonical SDK names** using exhaustive, validated lookup tables.
4.  Produce standard :class:`LogEntry` objects — identical to those the
    evaluation platform generator would produce.
5.  Build the :class:`Trace` by reusing the same pure helper functions that
    ``_generator.py`` uses (``_compute_resulting_state``, etc.).
6.  Optionally inline subagent steps from referenced subagent trajectory
    files (located in the same directory).

Any unknown tool that is not explicitly skipped is logged as a warning.

ATIF supports two agents — **Copilot CLI** (tool names like ``glob``,
``view``, ``edit``, ``bash``) and **Claude Code** (PascalCase names like
``Glob``, ``Read``, ``Edit``, ``Bash``).  Both are normalised to the same
canonical SDK tool names.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

__all__ = ["ATIFTraceGenerator"]


# ═══════════════════════════════════════════════════════════════════════════
# Tool-name mapping tables
# ═══════════════════════════════════════════════════════════════════════════

# Map every known ATIF tool → canonical SDK tool name.
# ``None`` means "skip this tool — it's not a code operation".
_TOOL_NAME_MAP: Dict[str, Optional[str]] = {
    # ── Copilot CLI tools ─────────────────────────────────────────
    "view":          "read_file",
    "create":        "create_file",
    "edit":          "replace_string_in_file",
    "glob":          "file_search",
    "grep":          "grep_search",
    "bash":          "run_in_terminal",
    "write_bash":    "run_in_terminal",
    "read_bash":     "run_in_terminal",
    "stop_bash":     None,   # session management, not a code operation
    "list_bash":     None,   # session listing, not a code operation
    "web_fetch":     "fetch_webpage",
    "web_search":    None,   # web search, no file impact
    "task":          None,   # subagent delegation — handled separately
    "report_intent": None,   # UI update, no code impact
    "store_memory":  None,   # agent memory management
    "ask_user":      None,   # user interaction
    "sql":           None,   # agent DB queries
    "read_agent":    None,   # subagent management
    "list_agents":   None,   # subagent management
    "skill":         None,   # skill plugin invocation
    "fetch_copilot_cli_documentation": None,  # documentation fetch
    "show_file":     "read_file",  # show file with optional diff view
    "exit_plan_mode": None,  # planning mode toggle

    # ── Claude Code tools ─────────────────────────────────────────
    "Read":          "read_file",
    "Write":         "create_file",
    "Edit":          "replace_string_in_file",
    "Glob":          "file_search",
    "Grep":          "grep_search",
    "Bash":          "run_in_terminal",
    "Task":          None,   # subagent delegation — handled separately
    "WebFetch":      "fetch_webpage",
    "WebSearch":     None,   # web search, no file impact
    "AskUserQuestion": None,  # user interaction
    "NotebookEdit":  "edit_notebook_file",
    "EnterPlanMode": None,   # planning mode toggle
    "ExitPlanMode":  None,   # planning mode toggle
    "EnterWorktree": None,   # worktree management
    "TaskOutput":    None,   # subagent output retrieval
    "TaskStop":      None,   # subagent stop
    "TaskCreate":    None,   # task tracking
    "TaskGet":       None,   # task tracking
    "TaskUpdate":    None,   # task tracking
    "TaskList":      None,   # task tracking
    "Skill":         None,   # skill plugin invocation

    # ── Cursor / Cursor-CLI tools ─────────────────────────────────
    "Shell":         "run_in_terminal",
    "StrReplace":    "replace_string_in_file",
    "CreatePlan":    None,   # planning/orchestration
    "TodoWrite":     None,   # task tracking, not a code operation
    "ReadLints":     "get_errors",

    # ── Common coding-agent tools (already canonical names) ──────
    "read_file":                  "read_file",
    "create_file":                "create_file",
    "replace_string_in_file":     "replace_string_in_file",
    "multi_replace_string_in_file": "multi_replace_string_in_file",
    "file_search":                "file_search",
    "grep_search":                "grep_search",
    "run_in_terminal":            "run_in_terminal",
    "get_errors":                 "get_errors",
}

# Canonical SDK tool names that the generators produce.
_CANONICAL_TOOLS = frozenset({
    "read_file", "create_file", "replace_string_in_file",
    "multi_replace_string_in_file", "file_search", "grep_search",
    "semantic_search", "list_dir", "run_in_terminal", "apply_patch",
    "fetch_webpage", "edit_notebook_file", "think", "get_errors",
})

# Subagent tool names (these spawn separate trajectory files).
_SUBAGENT_TOOLS = frozenset({"task", "Task", "Agent"})


# ═══════════════════════════════════════════════════════════════════════════
# Argument-key mapping tables
# ═══════════════════════════════════════════════════════════════════════════

# ATIF arg key → canonical SDK arg key.
# Keys not listed here pass through unchanged.
_ARG_KEY_MAP: Dict[str, str] = {
    # Copilot CLI
    "path":        "filePath",
    "old_str":     "oldString",
    "new_str":     "newString",
    # Claude Code
    "file_path":   "filePath",
    "old_string":  "oldString",
    "new_string":  "newString",
    # Cursor-CLI
    "glob_pattern": "query",    # Cursor-CLI Glob tool
    "contents":    "content",   # Cursor-CLI Write tool
    "paths":       "filePaths", # Cursor-CLI ReadLints tool
    # Common
    "content":     "content",
    "command":     "command",
    "pattern":     "query",
}

# Required args per canonical tool — for fail-fast validation.
_REQUIRED_ARGS: Dict[str, List[str]] = {
    "read_file":              ["filePath"],
    "create_file":            ["filePath"],
    "replace_string_in_file": ["filePath"],
    "run_in_terminal":        ["command"],
    "file_search":            ["query"],
    "grep_search":            ["query"],
}

# ATIF workspace path prefixes to strip during normalisation.
# These appear in absolute paths logged by Copilot/Claude agents.
_ATIF_PATH_PREFIXES = (
    "home/semick/repo/evals-research/experiments/sxs-native/sessions/",
    "home/semick/repo/",
    "home/",
    "tmp/",
)


# ═══════════════════════════════════════════════════════════════════════════
# Workspace root detection
# ═══════════════════════════════════════════════════════════════════════════

def _detect_workspace_root(data: Dict[str, Any]) -> Optional[str]:
    """Auto-detect the workspace root directory from file paths in the trajectory.

    Uses two strategies:

    1. **Shell/Bash ``cwd`` arguments** — the most reliable indicator of the
       actual working directory.
    2. **Longest common directory prefix** of source-file paths — fallback
       when ``cwd`` is not available.

    This handles agents (e.g. Cursor, IDE-based agents) whose paths
    do NOT contain ``repo-copilot/`` or ``repo-claude/`` markers.

    Returns the common prefix (forward-slashed, with trailing ``/``) or
    ``None`` if detection fails.
    """
    _PATH_KEYS = ("path", "file_path", "filePath")
    _SHELL_TOOLS = ("Shell", "Bash", "bash", "write_bash")
    _REPO_MARKERS = ("repo-copilot/", "repo-claude/", "repo-clone/", "repo/")

    raw_paths: List[str] = []
    cwd_paths: List[str] = []

    for step in data.get("steps", []):
        for tc in step.get("tool_calls", []):
            fn = tc.get("function_name", "")
            args = tc.get("arguments", {})

            # Collect file paths
            for key in _PATH_KEYS:
                val = args.get(key)
                if val and isinstance(val, str):
                    raw_paths.append(val.replace("\\", "/"))

            # Collect cwd from Shell/Bash commands
            if fn in _SHELL_TOOLS:
                cwd = args.get("cwd")
                if cwd and isinstance(cwd, str):
                    cwd_paths.append(cwd.replace("\\", "/").rstrip("/") + "/")

    if not raw_paths:
        return None

    # If ANY paths contain a repo-<agent>/ marker, the marker-based
    # normalisation already handles them.  Skip workspace-root detection
    # to avoid polluting with unrelated non-marker paths.
    if any(any(m in p for m in _REPO_MARKERS) for p in raw_paths):
        return None

    # Strategy 1: Use cwd from Shell/Bash commands if available.
    if cwd_paths:
        # Pick the shortest cwd (most likely the workspace root, not a subdir).
        shortest_cwd = min(cwd_paths, key=len).lstrip("/")
        if len(shortest_cwd.split("/")) >= 3:  # at least 2+ real components
            logger.debug("Workspace root from cwd: %s", shortest_cwd)
            return shortest_cwd

    # Strategy 2: Longest common directory prefix.
    # Include both source files (use parent dir) and directory-like paths
    # (e.g. Grep targets) for more reliable root detection.
    dir_candidates: List[str] = []
    for p in raw_paths:
        clean = p.lstrip("/")
        if "/" not in clean:
            continue
        # Skip hidden directory paths (.claude/, .cursor/, .git/ etc.)
        components = clean.split("/")
        if any(c.startswith(".") for c in components):
            continue
        last = components[-1]
        if "." in last:
            # File path: use parent directory
            dir_candidates.append("/".join(components[:-1]))
        else:
            # Directory path: use as-is
            dir_candidates.append(clean)

    if len(dir_candidates) < 2:
        return None

    # Find the longest common directory prefix.
    parts_list = [d.split("/") for d in dir_candidates]
    min_len = min(len(parts) for parts in parts_list)

    common_len = 0
    for i in range(min_len):
        ref = parts_list[0][i].lower()
        if all(parts[i].lower() == ref for parts in parts_list):
            common_len = i + 1
        else:
            break

    # Require at least 2 directory components for a meaningful root.
    if common_len < 2:
        return None

    root = "/".join(parts_list[0][:common_len]) + "/"
    logger.debug("Auto-detected workspace root: %s", root)
    return root


# ═══════════════════════════════════════════════════════════════════════════
# Normalisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_tool(function_name: str) -> Optional[str]:
    """Map an ATIF function_name to a canonical SDK tool name.

    Returns ``None`` for tools that should be skipped (orchestration,
    subagent delegation, UI interactions).  Unknown tools that don't
    start with a known prefix are logged as warnings and skipped.
    """
    if function_name in _TOOL_NAME_MAP:
        return _TOOL_NAME_MAP[function_name]

    # Azure MCP tools (azure-*) — skip
    if function_name.startswith("azure-"):
        return None

    # GitHub MCP server tools (github-mcp-server-*) — skip
    if function_name.startswith("github-mcp-server-"):
        return None

    logger.warning(
        "Unknown ATIF tool %r — skipping. "
        "Add it to _TOOL_NAME_MAP if it should be tracked.",
        function_name,
    )
    return None


def _normalize_atif_path(path: str, workspace_root: Optional[str] = None) -> str:
    """Normalise an ATIF file path.

    ATIF trajectories contain absolute paths from the evaluation host,
    e.g. ``/home/semick/repo/evals-research/experiments/sxs-native/
    sessions/20260302-225121-830822/repo-copilot/sdk/monitor/...``.

    This strips the host prefix up to and including the ``repo-copilot/``
    or ``repo-claude/`` or ``repo-clone/`` segment, leaving only the
    workspace-relative path.  For agents like Cursor or IDE-based agents
    that use different workspace roots, the auto-detected
    *workspace_root* parameter is used instead.
    """
    if not path:
        return path

    # Normalise slashes
    normalised = path.replace("\\", "/")

    # Strip everything up to and including repo-<agent>/ segment
    for marker in ("repo-copilot/", "repo-claude/", "repo-clone/", "repo/"):
        idx = normalised.find(marker)
        if idx >= 0:
            return "/" + normalised[idx + len(marker):]

    # Use auto-detected workspace root (handles Cursor, IDE agents, etc.)
    if workspace_root:
        # Strip drive letters and leading slashes for comparison
        def _strip_drive(p: str) -> str:
            if len(p) >= 2 and p[1] == ":":
                return p[2:].lstrip("/")
            return p.lstrip("/")

        path_clean = _strip_drive(normalised).lower()
        root_clean = _strip_drive(workspace_root).lower()
        if path_clean.startswith(root_clean):
            return "/" + _strip_drive(normalised)[len(root_clean):]

    # Fallback: try generic prefix stripping
    stripped = normalised.lstrip("/")
    for prefix in _ATIF_PATH_PREFIXES:
        if stripped.lower().startswith(prefix):
            # Skip past session-id/repo-<agent>/ after the prefix
            rest = stripped[len(prefix):]
            # May have session-id/ and repo-agent/ in the remaining path
            parts = rest.split("/")
            for i, part in enumerate(parts):
                if part.startswith("repo-") or part == "repo":
                    return "/" + "/".join(parts[i + 1:])
            return "/" + rest

    # If it starts with /, return as-is (already clean)
    if normalised.startswith("/"):
        return normalised

    return "/" + normalised


def _normalize_args(
    tool: str,
    raw_args: Dict[str, Any],
    function_name: str,
    workspace_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalise ATIF tool arguments to canonical SDK arg keys.

    Returns a new dict with canonical keys and normalised values.
    """
    normalised: Dict[str, Any] = {}

    for k, v in raw_args.items():
        if v is None:
            continue
        canonical_key = _ARG_KEY_MAP.get(k, k)
        normalised[canonical_key] = v

    # Normalise file paths
    if "filePath" in normalised:
        normalised["filePath"] = _normalize_atif_path(str(normalised["filePath"]), workspace_root)

    # Handle view_range from Copilot's "view" tool → startLine/endLine
    if tool == "read_file":
        view_range = normalised.pop("view_range", None)
        if view_range and isinstance(view_range, (list, tuple)) and len(view_range) == 2:
            normalised["startLine"] = view_range[0]
            normalised["endLine"] = view_range[1]

        # Handle Claude's Read tool offset/limit → startLine/endLine
        offset = normalised.pop("offset", None)
        limit = normalised.pop("limit", None)
        if offset is not None and offset > 0:
            normalised["startLine"] = offset
            if limit is not None and limit > 0:
                normalised["endLine"] = offset + limit - 1

    # Handle Claude Edit's replace_all → keep as metadata
    # (SDK doesn't distinguish; the resulting state will be the same.)

    # For file_search (from glob/Glob), "query" is the glob pattern
    # No further normalisation needed.

    # Validate required args
    if tool in _REQUIRED_ARGS:
        for req_key in _REQUIRED_ARGS[tool]:
            if req_key not in normalised or normalised[req_key] is None:
                logger.warning(
                    "ATIF tool '%s' (from '%s') missing expected arg '%s'. "
                    "Available: %s",
                    tool, function_name, req_key, list(normalised.keys()),
                )

    return normalised


def _extract_observation_content(
    tool_call_id: str,
    observation: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """Extract the content and subagent refs from an ATIF observation.

    Returns (content_str, subagent_refs) where subagent_refs may be None.
    """
    if observation is None:
        return None, None

    results = observation.get("results", [])
    for result in results:
        if result.get("source_call_id") == tool_call_id:
            content = result.get("content")
            # Parse JSON content strings (Copilot wraps results in JSON)
            if content and isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        # Extract meaningful content from wrapper
                        content = parsed.get("detailedContent",
                                             parsed.get("content", content))
                except (json.JSONDecodeError, TypeError):
                    pass  # Use raw string

            subagent_refs = result.get("subagent_trajectory_ref")
            return content, subagent_refs

    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════════════

class ATIFTraceGenerator:
    """Build a :class:`Trace` from an ATIF v1.6 ``trajectory.json``.

    This generator:
    1.  Parses the ATIF JSON and filters to agent steps with tool calls.
    2.  Normalises tool names and arg keys to canonical SDK form.
    3.  Creates :class:`LogEntry` objects identical to evaluation platform ones.
    4.  Optionally inlines subagent steps from referenced trajectory files.
    5.  Builds the trace using the same pure helpers as the evaluation platform generator.
    """

    def __init__(self, *, inline_subagents: bool = True) -> None:
        self._state_counter = 0
        self._transition_counter = 0
        self._inline_subagents = inline_subagents
        self._workspace_root: Optional[str] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self, trajectory_file: str) -> Trace:
        """Generate a :class:`Trace` from an ATIF *trajectory_file*.

        Parameters
        ----------
        trajectory_file : str
            Path to an ATIF ``trajectory.json``.

        Returns
        -------
        Trace
        """
        logger.info("Generating Trace from ATIF trajectory: %s", trajectory_file)

        with open(trajectory_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a JSON object in {trajectory_file}, "
                f"got {type(data).__name__}."
            )

        schema = data.get("schema_version", "")
        if not schema.startswith("ATIF"):
            raise ValueError(
                f"Expected ATIF schema_version in {trajectory_file}, "
                f"got {schema!r}."
            )

        log_entries, skipped_tools, extra_counts = self._extract_log_entries(data, trajectory_file)
        if not log_entries:
            logger.warning(
                "No actionable log entries in %s — building empty trace.",
                trajectory_file,
            )

        if skipped_tools:
            logger.info(
                "Skipped %d tool invocations across %d tool types: %s",
                sum(skipped_tools.values()),
                len(skipped_tools),
                ", ".join(f"{k}({v})" for k, v in sorted(skipped_tools.items())),
            )

        logger.info("Extracted %d normalised log entries", len(log_entries))

        trace = self._build_trace(log_entries, data, trajectory_file, skipped_tools,
                                  extra_counts=extra_counts)
        return trace

    # ------------------------------------------------------------------
    # Log extraction + normalisation
    # ------------------------------------------------------------------

    def _extract_log_entries(
        self,
        data: Dict[str, Any],
        trajectory_file: str,
    ) -> Tuple[List[LogEntry], Dict[str, int], Dict[str, int]]:
        """Parse ATIF steps into canonical :class:`LogEntry` objects.

        Returns
        -------
        entries
            The normalised log entries.
        skipped_tools
            Mapping of ATIF tool names that were skipped → invocation count.
        extra_counts
            Additional counters: human_input_count, subagent_count, compaction_count.
        """
        steps = data.get("steps", [])
        agent_info = data.get("agent", {})
        agent_name = agent_info.get("name", "unknown")
        traj_dir = str(Path(trajectory_file).parent)

        # Auto-detect workspace root for agents that don't use repo-* markers
        self._workspace_root = _detect_workspace_root(data)

        entries: List[LogEntry] = []
        skipped_tools: Dict[str, int] = {}
        entry_idx = 0
        human_input_count = 0
        human_input_positions: List[int] = []
        subagent_count = 0
        compaction_count = 0

        for step_idx, step in enumerate(steps):
            source = step.get("source", "")

            # Skip system, user, and compaction steps
            if source != "agent":
                if source == "user":
                    human_input_count += 1
                    human_input_positions.append(step_idx)
                continue

            # Check for compaction event
            if step.get("extra", {}).get("compaction_event"):
                compaction_count += 1
                continue

            tool_calls = step.get("tool_calls")
            observation = step.get("observation")
            timestamp = step.get("timestamp", "")
            model_name = step.get("model_name", agent_info.get("model_name", ""))
            step_metrics = step.get("metrics", {})

            if not tool_calls:
                # Agent text response without tool calls — treat as request
                message = step.get("message", "")
                if message:
                    log_entry = LogEntry(
                        id=f"atif_{step.get('step_id', entry_idx)}",
                        kind="request",
                        raw_data=step,
                        index=entry_idx,
                        model=model_name,
                        response_message=message,
                    )
                    log_entry.raw_data = {
                        "atif_step_id": step.get("step_id"),
                        "timestamp": timestamp,
                        "model": model_name,
                    }
                    entries.append(log_entry)
                    entry_idx += 1
                continue

            # Process each tool call in the step
            for tc in tool_calls:
                function_name = tc.get("function_name", "")
                tool_call_id = tc.get("tool_call_id", "")
                raw_args = tc.get("arguments", {})

                # Check for subagent delegation
                if function_name in _SUBAGENT_TOOLS:
                    subagent_count += 1
                    if self._inline_subagents:
                        # Extract subagent trajectory ref and inline
                        _, subagent_refs = _extract_observation_content(
                            tool_call_id, observation,
                        )
                        if subagent_refs:
                            subagent_entries = self._load_subagent_entries(
                                subagent_refs, traj_dir, entry_idx,
                            )
                            entries.extend(subagent_entries)
                            entry_idx += len(subagent_entries)
                    continue

                # Resolve to canonical tool
                tool = _resolve_tool(function_name)
                if tool is None:
                    skipped_tools[function_name] = skipped_tools.get(function_name, 0) + 1
                    continue

                # Normalise args
                normalised_args = _normalize_args(tool, raw_args, function_name, self._workspace_root)

                # Extract observation content for this tool call
                content, _ = _extract_observation_content(
                    tool_call_id, observation,
                )

                # Determine success/failure from observation content
                response = content
                if tool in ("create_file", "replace_string_in_file"):
                    if response and (
                        "successfully" in str(response).lower()
                        or "has been" in str(response).lower()
                        or "updated" in str(response).lower()
                    ):
                        response = f"The file was edited successfully. {str(content)[:200]}"

                # Create canonical LogEntry
                log_entry = LogEntry(
                    id=f"atif_{step.get('step_id', entry_idx)}_{tool_call_id[:12]}",
                    kind="toolCall",
                    raw_data=step,
                    index=entry_idx,
                    tool=tool,
                    args=normalised_args,
                    response=response,
                    model=model_name,
                )

                # Store ATIF-specific metadata for traceability
                log_entry.raw_data = {
                    "atif_function_name": function_name,
                    "atif_tool_call_id": tool_call_id,
                    "atif_step_id": step.get("step_id"),
                    "timestamp": timestamp,
                    "model": model_name,
                    "prompt_tokens": step_metrics.get("prompt_tokens", 0),
                    "completion_tokens": step_metrics.get("completion_tokens", 0),
                    "latency_ms": step_metrics.get("extra", {}).get("latency_ms", 0),
                    "agent_name": agent_name,
                }

                entries.append(log_entry)
                entry_idx += 1

        return entries, skipped_tools, {
            "human_input_count": human_input_count,
            "human_input_positions": human_input_positions,
            "subagent_count": subagent_count,
            "compaction_count": compaction_count,
        }

    def _load_subagent_entries(
        self,
        subagent_refs: List[Dict[str, Any]],
        traj_dir: str,
        start_idx: int,
    ) -> List[LogEntry]:
        """Load and inline entries from subagent trajectory files.

        Subagent trajectory files live in the same directory as the main
        trajectory.  They follow the identical ATIF format.
        """
        inlined: List[LogEntry] = []
        idx = start_idx

        for ref in subagent_refs:
            rel_path = ref.get("trajectory_path", "")
            if not rel_path:
                continue

            subagent_file = Path(traj_dir) / rel_path
            if not subagent_file.exists():
                logger.warning(
                    "Subagent trajectory not found: %s", subagent_file,
                )
                continue

            try:
                with open(subagent_file, "r", encoding="utf-8") as fh:
                    sub_data = json.load(fh)

                # Check if subagent failed
                sub_extra = sub_data.get("extra", {})
                if sub_extra.get("failed"):
                    logger.info(
                        "Skipping failed subagent: %s (%s)",
                        rel_path, sub_extra.get("failure_error", "unknown"),
                    )
                    continue

                sub_agent = sub_data.get("agent", {})
                sub_agent_name = sub_agent.get("name", "unknown")

                for step in sub_data.get("steps", []):
                    if step.get("source") != "agent":
                        continue
                    if step.get("extra", {}).get("compaction_event"):
                        continue

                    tool_calls = step.get("tool_calls")
                    if not tool_calls:
                        continue

                    observation = step.get("observation")
                    timestamp = step.get("timestamp", "")
                    model_name = step.get("model_name",
                                          sub_agent.get("model_name", ""))
                    step_metrics = step.get("metrics", {})

                    for tc in tool_calls:
                        fn = tc.get("function_name", "")
                        tc_id = tc.get("tool_call_id", "")
                        raw_args = tc.get("arguments", {})

                        # Don't recurse into sub-subagent delegation
                        if fn in _SUBAGENT_TOOLS:
                            continue

                        tool = _resolve_tool(fn)
                        if tool is None:
                            continue

                        normalised_args = _normalize_args(tool, raw_args, fn, self._workspace_root)
                        content, _ = _extract_observation_content(
                            tc_id, observation,
                        )

                        log_entry = LogEntry(
                            id=f"atif_sub_{rel_path}_{step.get('step_id', idx)}_{tc_id[:12]}",
                            kind="toolCall",
                            raw_data={},
                            index=idx,
                            tool=tool,
                            args=normalised_args,
                            response=content,
                            model=model_name,
                        )
                        log_entry.raw_data = {
                            "atif_function_name": fn,
                            "atif_tool_call_id": tc_id,
                            "atif_step_id": step.get("step_id"),
                            "timestamp": timestamp,
                            "model": model_name,
                            "prompt_tokens": step_metrics.get("prompt_tokens", 0),
                            "completion_tokens": step_metrics.get("completion_tokens", 0),
                            "latency_ms": step_metrics.get("extra", {}).get("latency_ms", 0),
                            "agent_name": sub_agent_name,
                            "subagent_file": rel_path,
                        }
                        inlined.append(log_entry)
                        idx += 1

            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Failed to parse subagent trajectory %s: %s",
                    subagent_file, exc,
                )

        return inlined

    # ------------------------------------------------------------------
    # Trace construction  (mirrors _generator.TraceGenerator._build_trace)
    # ------------------------------------------------------------------

    def _build_trace(
        self,
        entries: List[LogEntry],
        data: Dict[str, Any],
        source_file: str,
        skipped_tools: Optional[Dict[str, int]] = None,
        extra_counts: Optional[Dict[str, int]] = None,
    ) -> Trace:
        """Build a :class:`Trace` from normalised :class:`LogEntry` objects."""
        agent_info = data.get("agent", {})
        extra = data.get("extra", {})
        final_metrics = data.get("final_metrics", {})

        # Resolve model: agent-level field, then first entry, then ""
        model = agent_info.get("model_name") or ""
        if not model:
            for entry in entries:
                m = entry.raw_data.get("model", "")
                if m and m != "unknown":
                    model = m
                    break

        trace = Trace()
        trace.metadata = {
            "source_file": Path(source_file).name,
            "source_path": str(source_file),
            "source_format": "atif",
            "num_entries": len(entries),
            "generator": "swe_trace_sdk.atif",
            # ATIF-specific metadata
            "schema_version": data.get("schema_version", ""),
            "agent_name": agent_info.get("name", "unknown"),
            "agent_version": agent_info.get("version", "unknown"),
            "model": model,
            "scenario": extra.get("scenario", ""),
            "repository": extra.get("repository", ""),
            "commit": extra.get("commit", ""),
            # Timing data
            "wall_time_ms": final_metrics.get("extra", {}).get("wall_time_ms", 0),
            "active_time_ms": final_metrics.get("extra", {}).get("active_time_ms", 0),
            "permission_wait_ms": final_metrics.get("extra", {}).get("permission_wait_ms", 0),
            # Token totals
            "total_prompt_tokens": final_metrics.get("total_prompt_tokens", 0),
            "total_completion_tokens": final_metrics.get("total_completion_tokens", 0),
            "total_cost_usd": final_metrics.get("total_cost_usd", 0),
            # Tools that were skipped (not mapped to canonical SDK tools)
            "skipped_tools": skipped_tools or {},
            # Behavioural counters
            "human_input_count": (extra_counts or {}).get("human_input_count", 0),
            "human_input_positions": (extra_counts or {}).get("human_input_positions", []),
            "subagent_count": (extra_counts or {}).get("subagent_count", 0),
            "compaction_count": (extra_counts or {}).get("compaction_count", 0),
        }

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

        # Intent-stage label every state
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

        # Extract ATIF-specific metadata
        timestamp = entry.raw_data.get("timestamp", "")
        agent_name = entry.raw_data.get("agent_name", "")

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
                "timestamp": timestamp,
                "prompt_tokens": entry.raw_data.get("prompt_tokens", 0),
                "completion_tokens": entry.raw_data.get("completion_tokens", 0),
                "latency_ms": entry.raw_data.get("latency_ms", 0),
                "model": entry.raw_data.get("model", ""),
                "agent_name": agent_name,
                "subagent_file": entry.raw_data.get("subagent_file", ""),
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
