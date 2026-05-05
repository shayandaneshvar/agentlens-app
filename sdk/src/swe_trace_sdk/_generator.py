"""Internal evaluation platform trace generator — converts ``chat-export-logs.json`` into a :class:`Trace`.

This is *not* part of the public API surface.  Users should call
:func:`swe_trace_sdk.trace.load` instead.

This generator is specific to the **evaluation platform** trajectory format.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import LogEntry, State, Transition, Trace
from ._code_analyzer import get_analyzer
from .intent import label_trace_intents
from .tool_registry import registry, CATEGORY_EDIT, CATEGORY_SEARCH, CATEGORY_EXECUTE

logger = logging.getLogger(__name__)

__all__ = ["TraceGenerator"]


class TraceGenerator:
    """Build a :class:`Trace` from a evaluation platform ``chat-export-logs.json`` file.

    Parameters
    ----------
    include_requests : bool
        If *True*, LLM request/response entries are included as states.
        Default is *False* (only tool calls).
    """

    def __init__(self, include_requests: bool = False) -> None:
        self.include_requests = include_requests
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
            Path to a evaluation platform ``chat-export-logs.json``.

        Returns
        -------
        Trace
        """
        logger.info("Generating Trace from: %s", trajectory_file)

        with open(trajectory_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        entries = self._extract_log_entries(data)
        if not entries:
            raise ValueError(
                f"No log entries found in {trajectory_file}. "
                "The file may be empty or corrupted."
            )
        logger.info("Extracted %d relevant log entries", len(entries))

        trace = self._build_trace(entries, trajectory_file)
        return trace

    # ------------------------------------------------------------------
    # Log extraction
    # ------------------------------------------------------------------

    def _extract_log_entries(self, data: Dict[str, Any]) -> List[LogEntry]:
        entries: List[LogEntry] = []
        prompts = data.get("prompts", [])
        if not prompts:
            logger.warning("No prompts found in trajectory")
            return entries

        all_logs: List[Dict[str, Any]] = []
        for prompt in prompts:
            all_logs.extend(prompt.get("logs", []))

        if not all_logs:
            logger.warning("No logs found in any prompt")
            return entries

        for idx, log in enumerate(all_logs):
            kind = log.get("kind", "")
            if idx == 0 or idx == len(all_logs) - 1:
                entries.append(LogEntry.from_log(log, idx))
                continue
            if kind == "toolCall":
                entries.append(LogEntry.from_log(log, idx))
                continue
            if self.include_requests and kind == "request":
                entries.append(LogEntry.from_log(log, idx))

        return entries

    # ------------------------------------------------------------------
    # Trace construction
    # ------------------------------------------------------------------

    def _build_trace(self, entries: List[LogEntry], source_file: str) -> Trace:
        trace = Trace()
        trace.metadata = {
            "source_file": Path(source_file).name,
            "source_path": str(source_file),
            "num_entries": len(entries),
            "generator": "swe_trace_sdk",
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

        # Intent-stage label every state (exploration / implementation / verification / orchestration)
        label_trace_intents(trace)

        return trace

    # ------------------------------------------------------------------
    # State helpers
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
        self._state_counter += 1
        loc = _extract_location_info(entry)
        return State(
            state_id=sid,
            step=self._state_counter,
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
            },
        )

    def _make_transition(self, from_s: State, to_s: State, entry: LogEntry) -> Transition:
        tid = f"trans_{self._transition_counter}"
        self._transition_counter += 1
        action_type = entry.tool or "unknown_tool" if entry.kind == "toolCall" else "request"
        action_data: Dict[str, Any] = {}
        if entry.kind == "toolCall" and entry.args:
            for key in ("filePath", "path", "query"):
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


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    path = path.lstrip("/\\")
    for prefix in ("workspace/", "workspaces/", "home/", "tmp/"):
        if path.lower().startswith(prefix):
            path = path[len(prefix):]
    return path.lower().replace("\\", "/")


# ── Resulting-state handlers ─────────────────────────────────────────
# Each handler takes (tool, args, response) and returns a resulting-state
# string.  Registered by tool name; unknown tools fall through to the
# generic handler.

def _rs_create_file(tool: str, args: Dict[str, Any], response: Any) -> str:
    norm = _normalize_path(args.get("filePath") or args.get("path", ""))
    ok = response and "successfully" in str(response).lower()
    return f"file_created:{norm}" if ok else f"file_create_failed:{norm}"


def _rs_read_file(tool: str, args: Dict[str, Any], response: Any) -> str:
    norm = _normalize_path(args.get("filePath") or args.get("path", ""))
    return f"file_read:{norm}"


def _rs_replace_string(tool: str, args: Dict[str, Any], response: Any) -> str:
    norm = _normalize_path(args.get("filePath") or args.get("path", ""))
    ok = response and "successfully" in str(response).lower()
    return f"file_modified:{norm}" if ok else f"file_modify_failed:{norm}"


def _rs_file_search(tool: str, args: Dict[str, Any], response: Any) -> str:
    query = args.get("query", "").lower()
    resp_str = str(response) if response else ""
    if "No files found" in resp_str or not response:
        return f"file_search:not_found:{query}"
    return f"file_search:found:{query}"


def _rs_grep_search(tool: str, args: Dict[str, Any], response: Any) -> str:
    resp_str = str(response) if response else ""
    if not response or "no matches" in resp_str.lower() or resp_str == "[]":
        return "grep_search:no_matches"
    return "grep_search:found_matches"


def _rs_semantic_search(tool: str, args: Dict[str, Any], response: Any) -> str:
    return "semantic_search:results" if response else "semantic_search:no_results"


def _rs_run_in_terminal(tool: str, args: Dict[str, Any], response: Any) -> str:
    cmd = args.get("command", "")
    base_cmd = cmd.strip().split()[0] if cmd.strip() else "unknown"
    return f"terminal:{base_cmd.lower()}"


def _rs_list_dir(tool: str, args: Dict[str, Any], response: Any) -> str:
    norm = _normalize_path(args.get("path", ""))
    return f"dir_listed:{norm}"


def _rs_apply_patch(tool: str, args: Dict[str, Any], response: Any) -> str:
    patch_input = args.get("input", "")
    file_path = ""
    for line in str(patch_input).split("\n"):
        if line.startswith("*** Update File:") or line.startswith("+++ "):
            file_path = line.split(":", 1)[-1].strip() if ":" in line else line[4:].strip()
            break
    norm = _normalize_path(file_path) if file_path else "unknown"
    return f"file_patched:{norm}"


def _rs_multi_replace(tool: str, args: Dict[str, Any], response: Any) -> str:
    replacements = args.get("replacements", [])
    paths = sorted({
        _normalize_path(r.get("filePath") or r.get("path", ""))
        for r in replacements if isinstance(r, dict)
    })
    return f"files_modified:{','.join(paths)}" if paths else "files_modified:unknown"


def _rs_generic(tool: str, args: Dict[str, Any], response: Any) -> str:
    """Fallback for tools without a dedicated handler."""
    resp_str = str(response) if response else ""
    is_error = any(err in resp_str.lower() for err in [
        "error", "failed", "invalid", "exception",
        "not found", "denied", "timeout", "refused",
    ])
    status = "error" if is_error else "success"
    return f"tool_result:{tool}:{status}"


# Map tool names → resulting-state handler functions.
_RS_HANDLERS: Dict[str, Callable[..., str]] = {
    "create_file": _rs_create_file,
    "read_file": _rs_read_file,
    "replace_string_in_file": _rs_replace_string,
    "file_search": _rs_file_search,
    "grep_search": _rs_grep_search,
    "semantic_search": _rs_semantic_search,
    "run_in_terminal": _rs_run_in_terminal,
    "list_dir": _rs_list_dir,
    "apply_patch": _rs_apply_patch,
    "multi_replace_string_in_file": _rs_multi_replace,
}


def _compute_resulting_state(
    entry: LogEntry,
    prev: Optional[State],
    is_terminal: bool,
) -> str:
    """Normalised resulting-state string."""

    if entry.kind == "toolCall":
        tool = entry.tool or "unknown"
        args = entry.args or {}
        response = entry.response

        handler = _RS_HANDLERS.get(tool)
        if handler is not None:
            return handler(tool, args, response)

        # Fallback: generic handler
        return _rs_generic(tool, args, response)

    if entry.kind == "request":
        if is_terminal and prev:
            prev_rs = prev.resulting_state or ""
            if prev_rs and prev_rs not in ("initial", "llm_response", "llm_response:planning") and not prev_rs.startswith("llm_response:"):
                return f"llm_response:confirmed:{prev_rs}"
            return "llm_response:terminal"
        if prev and prev.resulting_state == "initial":
            return "llm_response:planning"
        return "llm_response:planning"

    return "unknown"


def _build_observation(entry: LogEntry) -> str:
    parts: List[str] = []
    if entry.kind == "toolCall":
        parts.append(f"Tool: {entry.tool}")
        if entry.args:
            for key in ("filePath", "path", "query", "command", "content"):
                if key in entry.args:
                    val = str(entry.args[key])
                    if len(val) > 200:
                        val = val[:200] + "..."
                    parts.append(f"{key}: {val}")
        if entry.response:
            resp_str = str(entry.response)
            if len(resp_str) > 500:
                resp_str = resp_str[:500] + "..."
            parts.append(f"Response: {resp_str}")
    elif entry.kind == "request":
        parts.append(f"LLM Request: {entry.model}")
        if entry.response_message:
            msg = entry.response_message
            if len(msg) > 500:
                msg = msg[:500] + "..."
            parts.append(f"Response: {msg}")
    return "\n".join(parts)


def _extract_files_touched(entry: LogEntry) -> set:
    files: set = set()
    if entry.kind == "toolCall" and entry.args:
        for key in ("filePath", "path"):
            if key in entry.args:
                files.add(entry.args[key])
    return files


# ---------------------------------------------------------------------------
# Content hash / description
# ---------------------------------------------------------------------------

# ── Content-hash handlers ────────────────────────────────────────────

def _ch_create_file(tool: str, args: Dict[str, Any]) -> str:
    content = args.get("content", "")
    return hashlib.md5(content.encode()).hexdigest()[:12] if content else ""


def _ch_replace_string(tool: str, args: Dict[str, Any]) -> str:
    combined = f"{args.get('oldString', '')}||{args.get('newString', '')}"
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def _ch_multi_replace(tool: str, args: Dict[str, Any]) -> str:
    replacements = args.get("replacements", [])
    if replacements:
        sorted_repls = sorted(replacements, key=lambda r: r.get("filePath", ""))
        parts = [f"{r.get('filePath', '')}:{r.get('oldString', '')}||{r.get('newString', '')}" for r in sorted_repls]
        return hashlib.md5("|||".join(parts).encode()).hexdigest()[:12]
    return ""


def _ch_apply_patch(tool: str, args: Dict[str, Any]) -> str:
    patch_input = args.get("input", "") or args.get("patch", "")
    return hashlib.md5(str(patch_input).encode()).hexdigest()[:12] if patch_input else ""


def _ch_terminal(tool: str, args: Dict[str, Any]) -> str:
    command = args.get("command", "")
    return hashlib.md5(command.encode()).hexdigest()[:12] if command else ""


_CH_HANDLERS: Dict[str, Callable[..., str]] = {
    "create_file": _ch_create_file,
    "replace_string_in_file": _ch_replace_string,
    "multi_replace_string_in_file": _ch_multi_replace,
    "apply_patch": _ch_apply_patch,
    "run_in_terminal": _ch_terminal,
}


def _compute_content_hash(entry: LogEntry) -> str:
    """Hash to distinguish different edits to the same file."""
    if entry.kind != "toolCall":
        return ""
    tool = entry.tool or ""
    args = entry.args or {}
    handler = _CH_HANDLERS.get(tool)
    if handler is not None:
        return handler(tool, args)
    return ""


# ── Content-description handlers ─────────────────────────────────────

def _filename(path: str) -> str:
    if not path:
        return "unknown"
    return path.split("/")[-1].split("\\")[-1]


def _cd_create_file(tool: str, args: Dict[str, Any]) -> str:
    path = args.get("filePath") or args.get("path", "")
    content = args.get("content", "")
    return get_analyzer().describe_file_creation(content, path)


def _cd_replace_string(tool: str, args: Dict[str, Any]) -> str:
    path = args.get("filePath") or args.get("path", "")
    fname = _filename(path)
    old_str = args.get("oldString", "")
    new_str = args.get("newString", "")
    comparison = get_analyzer().compare_code(old_str, new_str, path)
    return f"In '{fname}': {comparison['description']}"


def _cd_multi_replace(tool: str, args: Dict[str, Any]) -> str:
    replacements = args.get("replacements", [])
    if not replacements:
        return ""
    descriptions: List[str] = []
    analyzer = get_analyzer()
    for repl in replacements[:5]:
        path = repl.get("filePath") or repl.get("path", "")
        fname = _filename(path)
        comparison = analyzer.compare_code(
            repl.get("oldString", ""), repl.get("newString", ""), path,
        )
        descriptions.append(f"In '{fname}': {comparison['description']}")
    if len(replacements) > 5:
        descriptions.append(f"... and {len(replacements) - 5} more changes")
    return "; ".join(descriptions)


def _parse_patch_hunks(patch_input: str) -> List[Dict[str, Any]]:
    """Parse a V4A / unified-diff patch into per-file old/new code blocks."""
    hunks: List[Dict[str, Any]] = []
    current_file = ""
    old_lines: List[str] = []
    new_lines: List[str] = []

    def _flush() -> None:
        nonlocal old_lines, new_lines, current_file
        if current_file and (old_lines or new_lines):
            hunks.append({
                "file": current_file,
                "old_code": "\n".join(old_lines),
                "new_code": "\n".join(new_lines),
            })
        old_lines, new_lines = [], []

    for line in patch_input.split("\n"):
        if line.startswith("*** Update File:") or line.startswith("+++ "):
            _flush()
            current_file = line.split(":", 1)[-1].strip() if ":" in line else line[4:].strip()
        elif line.startswith("-") and not line.startswith("---"):
            old_lines.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            new_lines.append(line[1:])
        elif not line.startswith("@@") and not line.startswith("***") and not line.startswith("---"):
            # Context line — belongs to both old and new
            text = line[1:] if line.startswith(" ") else line
            old_lines.append(text)
            new_lines.append(text)
    _flush()
    return hunks


def _cd_apply_patch(tool: str, args: Dict[str, Any]) -> str:
    patch_input = args.get("input", "") or args.get("patch", "")
    if not patch_input:
        return ""
    hunks = _parse_patch_hunks(str(patch_input))
    if not hunks:
        return "Applied patch"
    analyzer = get_analyzer()
    descriptions: List[str] = []
    for hunk in hunks[:5]:
        fname = _filename(hunk["file"])
        comparison = analyzer.compare_code(
            hunk["old_code"], hunk["new_code"], hunk["file"],
        )
        descriptions.append(f"In '{fname}': {comparison['description']}")
    if len(hunks) > 5:
        descriptions.append(f"... and {len(hunks) - 5} more files")
    return "; ".join(descriptions) if descriptions else "Applied patch"


def _cd_terminal(tool: str, args: Dict[str, Any]) -> str:
    command = args.get("command", "")
    if not command:
        return ""
    base_cmd = command.strip().split()[0] if command.strip() else ""
    if "python" in base_cmd.lower():
        return f"Ran Python command: {command[:100]}"
    elif "npm" in base_cmd.lower() or "node" in base_cmd.lower():
        return f"Ran Node.js command: {command[:100]}"
    elif "git" in base_cmd.lower():
        return f"Ran Git command: {command[:100]}"
    return f"Ran terminal command: {command[:100]}"


_CD_HANDLERS: Dict[str, Callable[..., str]] = {
    "create_file": _cd_create_file,
    "replace_string_in_file": _cd_replace_string,
    "multi_replace_string_in_file": _cd_multi_replace,
    "apply_patch": _cd_apply_patch,
    "run_in_terminal": _cd_terminal,
}


def _compute_content_description(entry: LogEntry) -> str:
    """Human-readable description of the file change."""
    if entry.kind != "toolCall":
        return ""
    tool = entry.tool or ""
    args = entry.args or {}
    handler = _CD_HANDLERS.get(tool)
    if handler is not None:
        return handler(tool, args)
    return ""


# ---------------------------------------------------------------------------
# Location / scope extraction
# ---------------------------------------------------------------------------

def _determine_operation_type(tool: str) -> str:
    """Look up coarse operation type from the tool registry."""
    return registry.get(tool).operation_type


def _determine_edit_type(old_string: str, new_string: str) -> str:
    if not old_string and new_string:
        return "add"
    if old_string and not new_string:
        return "remove"
    if not old_string and not new_string:
        return ""
    old_len, new_len = len(old_string), len(new_string)
    if new_len > old_len * 1.5:
        return "expand"
    if new_len < old_len * 0.5:
        return "shrink"
    return "replace"


def _extract_line_range_from_read(args: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    start = args.get("startLine")
    end = args.get("endLine")
    if start is not None and end is not None:
        return (int(start), int(end))
    if start is not None:
        return (int(start), int(start))
    return None


def _extract_line_range_from_edit(old_string: str, new_string: str) -> Optional[Tuple[int, int]]:
    if not old_string:
        return None
    old_lines = old_string.count("\n") + 1
    new_lines = new_string.count("\n") + 1 if new_string else 0
    scope = max(old_lines, new_lines)
    return (1, scope)


def _compute_relative_position(line_range: Optional[Tuple[int, int]], file_length: int = 100) -> str:
    if not line_range or file_length <= 0:
        return ""
    start, end = line_range
    mid_line = (start + end) / 2
    ratio = mid_line / file_length
    if ratio <= 0.33:
        return "early"
    if ratio <= 0.66:
        return "middle"
    return "late"


def _extract_scope_from_code(code: str, filename: str = "") -> Dict[str, str]:
    result: Dict[str, str] = {"function_name": "", "class_name": "", "scope_path": ""}
    if not code:
        return result
    analyzer = get_analyzer()
    analysis = analyzer.analyze(code, filename=filename)
    func_names = analysis.get_function_names()
    class_names = analysis.get_class_names()
    if func_names:
        result["function_name"] = sorted(func_names)[0]
    if class_names:
        result["class_name"] = sorted(class_names)[0]
    for func in analysis.functions:
        if func.parent:
            result["scope_path"] = f"{func.parent}.{func.name}"
            result["class_name"] = func.parent
            result["function_name"] = func.name
            break
    if not result["scope_path"]:
        if result["class_name"] and result["function_name"]:
            result["scope_path"] = f"{result['class_name']}.{result['function_name']}"
        elif result["function_name"]:
            result["scope_path"] = result["function_name"]
        elif result["class_name"]:
            result["scope_path"] = result["class_name"]
    return result


# ── Location-info handlers ───────────────────────────────────────────

def _loc_read_file(tool: str, args: Dict[str, Any], info: Dict[str, Any]) -> None:
    info["line_range"] = _extract_line_range_from_read(args)
    end_line = args.get("endLine", 100)
    info["relative_position"] = _compute_relative_position(info["line_range"], end_line * 1.5)


def _loc_replace_string(tool: str, args: Dict[str, Any], info: Dict[str, Any]) -> None:
    file_path = args.get("filePath") or args.get("path", "")
    old_str = args.get("oldString", "")
    new_str = args.get("newString", "")
    info["line_range"] = _extract_line_range_from_edit(old_str, new_str)
    info["edit_type"] = _determine_edit_type(old_str, new_str)
    scope = _extract_scope_from_code(old_str, file_path)
    info["function_name"] = scope["function_name"]
    info["class_name"] = scope["class_name"]
    info["scope_path"] = scope["scope_path"]


def _loc_multi_replace(tool: str, args: Dict[str, Any], info: Dict[str, Any]) -> None:
    replacements = args.get("replacements", [])
    if replacements:
        all_files: set = set()
        total_lines = 0
        all_functions: List[str] = []
        all_classes: List[str] = []
        for repl in replacements:
            rp = repl.get("filePath") or repl.get("path", "")
            if rp:
                all_files.add(_normalize_path(rp))
            old_str = repl.get("oldString", "")
            new_str = repl.get("newString", "")
            if old_str:
                total_lines += old_str.count("\n") + 1
                scope = _extract_scope_from_code(old_str, rp)
                if scope["function_name"]:
                    all_functions.append(scope["function_name"])
                if scope["class_name"]:
                    all_classes.append(scope["class_name"])
        info["file_path"] = ",".join(sorted(all_files)) if all_files else ""
        info["line_range"] = (1, total_lines) if total_lines > 0 else None
        info["edit_type"] = "replace"
        if all_functions:
            info["function_name"] = ",".join(sorted(set(all_functions)))
        if all_classes:
            info["class_name"] = ",".join(sorted(set(all_classes)))
        if all_functions or all_classes:
            info["scope_path"] = info["function_name"] or info["class_name"]


def _loc_create_file(tool: str, args: Dict[str, Any], info: Dict[str, Any]) -> None:
    file_path = args.get("filePath") or args.get("path", "")
    content = args.get("content", "")
    if content:
        lines = content.count("\n") + 1
        info["line_range"] = (1, lines)
        info["edit_type"] = "add"
        scope = _extract_scope_from_code(content, file_path)
        info["function_name"] = scope["function_name"]
        info["class_name"] = scope["class_name"]
        info["scope_path"] = scope["scope_path"]


def _loc_apply_patch(tool: str, args: Dict[str, Any], info: Dict[str, Any]) -> None:
    patch_input = args.get("input", "") or args.get("patch", "")
    if patch_input:
        hunks = _parse_patch_hunks(str(patch_input))
        if hunks:
            # File path from first hunk
            info["file_path"] = _normalize_path(hunks[0]["file"])
            # Aggregate scope from all hunks using tree-sitter
            all_functions: List[str] = []
            all_classes: List[str] = []
            total_lines = 0
            for hunk in hunks:
                old_code = hunk["old_code"]
                total_lines += max(old_code.count("\n") + 1, hunk["new_code"].count("\n") + 1)
                scope = _extract_scope_from_code(old_code, hunk["file"])
                if scope["function_name"]:
                    all_functions.append(scope["function_name"])
                if scope["class_name"]:
                    all_classes.append(scope["class_name"])
                # Also check new code for renamed functions
                scope_new = _extract_scope_from_code(hunk["new_code"], hunk["file"])
                if scope_new["function_name"] and scope_new["function_name"] not in all_functions:
                    all_functions.append(scope_new["function_name"])
                if scope_new["class_name"] and scope_new["class_name"] not in all_classes:
                    all_classes.append(scope_new["class_name"])
            info["line_range"] = (1, total_lines) if total_lines > 0 else None
            info["edit_type"] = "replace"
            if all_functions:
                info["function_name"] = ",".join(sorted(set(all_functions)))
            if all_classes:
                info["class_name"] = ",".join(sorted(set(all_classes)))
            if all_functions or all_classes:
                info["scope_path"] = info["function_name"] or info["class_name"]
        else:
            lines = str(patch_input).count("\n") + 1
            info["line_range"] = (1, lines)
            info["edit_type"] = "replace"
            for line in str(patch_input).split("\n"):
                if line.startswith("*** Update File:") or line.startswith("+++ "):
                    path = line.split(":", 1)[-1].strip() if ":" in line else line[4:].strip()
                    info["file_path"] = _normalize_path(path)
                    break


_LOC_HANDLERS: Dict[str, Callable[..., None]] = {
    "read_file": _loc_read_file,
    "replace_string_in_file": _loc_replace_string,
    "multi_replace_string_in_file": _loc_multi_replace,
    "create_file": _loc_create_file,
    "apply_patch": _loc_apply_patch,
}


def _extract_location_info(entry: LogEntry) -> Dict[str, Any]:
    """Extract location and scope information from a log entry."""
    info: Dict[str, Any] = {
        "file_path": "", "line_range": None, "relative_position": "",
        "operation_type": "", "edit_type": "",
        "function_name": "", "class_name": "", "scope_path": "",
    }
    if entry.kind != "toolCall":
        return info
    tool = entry.tool or ""
    args = entry.args or {}
    file_path = args.get("filePath") or args.get("path", "")
    info["file_path"] = _normalize_path(file_path)
    info["operation_type"] = _determine_operation_type(tool)

    handler = _LOC_HANDLERS.get(tool)
    if handler is not None:
        handler(tool, args, info)
    return info
