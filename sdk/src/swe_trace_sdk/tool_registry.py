"""Central tool registry — single source of truth for tool metadata.

Every tool the SDK knows about is declared **once** here as a
:class:`ToolDescriptor`.  All other modules (generator, phase labeler,
equivalence checker, visualiser) import from this registry instead of
maintaining their own hardcoded tool-name sets and if/elif chains.

Adding a new tool
-----------------
1.  Create a :class:`ToolDescriptor` instance.
2.  Call ``registry.register(descriptor)``.
3.  Done — no other files need changing.

Public API
----------
- :class:`ToolDescriptor`   — per-tool metadata.
- :class:`ToolRegistry`     — queryable collection of descriptors.
- :data:`registry`          — the singleton, pre-populated instance.
- Helper constants re-exported for convenience (``STAGE_*``, ``CATEGORY_*``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

__all__ = [
    "ToolDescriptor",
    "ToolRegistry",
    "registry",
    # intent-stage constants
    "STAGE_EXPLORATION",
    "STAGE_IMPLEMENTATION",
    "STAGE_VERIFICATION",
    "STAGE_ORCHESTRATION",
    # category constants
    "CATEGORY_READ",
    "CATEGORY_EDIT",
    "CATEGORY_SEARCH",
    "CATEGORY_EXECUTE",
    "CATEGORY_VALIDATION",
    "CATEGORY_GENERAL",
]


# ──────────────────────────────────────────────────────────────────────
# Intent-stage & category constants
# ──────────────────────────────────────────────────────────────────────

STAGE_EXPLORATION = "exploration"
STAGE_IMPLEMENTATION = "implementation"
STAGE_VERIFICATION = "verification"
STAGE_ORCHESTRATION = "orchestration"

CATEGORY_READ = "read"
CATEGORY_EDIT = "edit"
CATEGORY_SEARCH = "search"
CATEGORY_EXECUTE = "execute"
CATEGORY_VALIDATION = "validation"
CATEGORY_GENERAL = "general"


# ──────────────────────────────────────────────────────────────────────
# ToolDescriptor
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolDescriptor:
    """Immutable metadata for a single tool.

    Parameters
    ----------
    name : str
        Canonical tool name (e.g. ``"replace_string_in_file"``).
    category : str
        One of the ``CATEGORY_*`` constants (``"read"``, ``"edit"``,
        ``"search"``, ``"execute"``, ``"validation"``, ``"general"``).
    stage_hint : str
        Default intent stage for **fixed-stage** tools.  Context-sensitive
        tools (read_file, edit tools, terminal) use ``""`` — the intent
        labeler decides at runtime.
    equivalence_group : str
        Tools that share an equivalence group are considered
        structurally interchangeable when they operate on overlapping
        files.  E.g. ``"file_edit"`` groups ``replace_string_in_file``,
        ``apply_patch``, ``multi_replace_string_in_file``, etc.
    operation_type : str
        Coarse operation type for the State model (``"read"``,
        ``"create"``, ``"modify"``, ``"delete"``, ``"search"``,
        ``"terminal"``, ``"other"``).
    key_args : tuple[str, ...]
        Argument names relevant for hashing / comparison.
    file_path_args : tuple[str, ...]
        Argument names that contain file paths (for ``files_touched``).
    comparison_strategy : str
        How the equivalence checker should compare two invocations of
        this tool:
        ``"file_path"``  — compare normalised file paths.
        ``"query"``      — compare query string (Jaccard similarity).
        ``"command"``    — compare normalised terminal command.
        ``"identity"``   — exact argument match.
        ``"none"``       — no heuristic (fall through to LLM).
    color : str
        Hex colour for HTML visualisations.
    context_sensitive : bool
        If *True*, the phase label depends on runtime context (whether
        patching has occurred, whether the file is a test file, etc.).
    """

    name: str
    category: str = CATEGORY_GENERAL
    stage_hint: str = STAGE_ORCHESTRATION
    equivalence_group: str = ""
    operation_type: str = "other"
    key_args: Tuple[str, ...] = ()
    file_path_args: Tuple[str, ...] = ()
    comparison_strategy: str = "identity"
    color: str = "#a1a1aa"          # default grey
    context_sensitive: bool = False


# ──────────────────────────────────────────────────────────────────────
# ToolRegistry
# ──────────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Queryable collection of :class:`ToolDescriptor` instances."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDescriptor] = {}
        # A fallback descriptor for tools not explicitly registered.
        self._default = ToolDescriptor(
            name="__default__",
            category=CATEGORY_GENERAL,
            stage_hint=STAGE_ORCHESTRATION,
            operation_type="other",
            comparison_strategy="identity",
            color="#a1a1aa",
        )
        # Pre-compiled MCP prefix pattern
        self._prefix_rules: List[Tuple[str, ToolDescriptor]] = []

    # ── Registration ──────────────────────────────────────────────

    def register(self, descriptor: ToolDescriptor) -> None:
        """Register a tool descriptor (overwrites any previous entry)."""
        self._tools[descriptor.name] = descriptor

    def register_many(self, descriptors: Sequence[ToolDescriptor]) -> None:
        for d in descriptors:
            self.register(d)

    def register_prefix(self, prefix: str, descriptor: ToolDescriptor) -> None:
        """Register a prefix rule (e.g. ``mcp_pylance_`` → localization)."""
        self._prefix_rules.append((prefix, descriptor))

    # ── Lookup ────────────────────────────────────────────────────

    def get(self, tool_name: str) -> ToolDescriptor:
        """Return the descriptor for *tool_name*, or a default."""
        desc = self._tools.get(tool_name)
        if desc is not None:
            return desc
        # Check prefix rules
        for prefix, pdesc in self._prefix_rules:
            if tool_name.startswith(prefix):
                return pdesc
        return self._default

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # ── Bulk queries ──────────────────────────────────────────────

    def by_category(self, category: str) -> FrozenSet[str]:
        """Return all tool names with the given category."""
        return frozenset(
            name for name, d in self._tools.items()
            if d.category == category
        )

    def by_stage(self, stage: str) -> FrozenSet[str]:
        """Return tool names whose *fixed* stage_hint matches."""
        return frozenset(
            name for name, d in self._tools.items()
            if d.stage_hint == stage and not d.context_sensitive
        )

    def by_equivalence_group(self, group: str) -> FrozenSet[str]:
        """Return all tool names in an equivalence group."""
        return frozenset(
            name for name, d in self._tools.items()
            if d.equivalence_group == group
        )

    def context_sensitive_tools(self) -> FrozenSet[str]:
        """Return tools whose phase depends on runtime context."""
        return frozenset(
            name for name, d in self._tools.items()
            if d.context_sensitive
        )

    def get_color(self, tool_name: str) -> str:
        """Return the visualisation color for *tool_name*."""
        return self.get(tool_name).color

    def is_in_equivalence_group(self, tool_a: str, tool_b: str) -> bool:
        """True if *tool_a* and *tool_b* share a non-empty equivalence group."""
        da = self.get(tool_a)
        db = self.get(tool_b)
        return bool(da.equivalence_group and da.equivalence_group == db.equivalence_group)

    def all_names(self) -> FrozenSet[str]:
        return frozenset(self._tools)


# ──────────────────────────────────────────────────────────────────────
# Pre-populated singleton
# ──────────────────────────────────────────────────────────────────────

registry = ToolRegistry()

# ── File-read tools ───────────────────────────────────────────────────

registry.register(ToolDescriptor(
    name="read_file",
    category=CATEGORY_READ,
    stage_hint="",                      # context-sensitive
    equivalence_group="file_read",
    operation_type="read",
    key_args=("filePath", "startLine", "endLine"),
    file_path_args=("filePath", "path"),
    comparison_strategy="file_path",
    color="#60a5fa",
    context_sensitive=True,
))

# ── File-edit tools ───────────────────────────────────────────────────

_EDIT_TOOL_DEFS = [
    ToolDescriptor(
        name="replace_string_in_file",
        category=CATEGORY_EDIT,
        stage_hint="",                  # context-sensitive
        equivalence_group="file_edit",
        operation_type="modify",
        key_args=("filePath", "oldString", "newString"),
        file_path_args=("filePath", "path"),
        comparison_strategy="file_path",
        color="#facc15",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="multi_replace_string_in_file",
        category=CATEGORY_EDIT,
        stage_hint="",
        equivalence_group="file_edit",
        operation_type="modify",
        key_args=("replacements",),
        file_path_args=("filePath", "path"),  # nested inside replacements
        comparison_strategy="file_path",
        color="#facc15",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="apply_patch",
        category=CATEGORY_EDIT,
        stage_hint="",
        equivalence_group="file_edit",
        operation_type="modify",
        key_args=("input", "patch"),
        file_path_args=(),                # extracted from patch content
        comparison_strategy="file_path",
        color="#facc15",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="edit_file",
        category=CATEGORY_EDIT,
        stage_hint="",
        equivalence_group="file_edit",
        operation_type="modify",
        key_args=("filePath", "oldString", "newString"),
        file_path_args=("filePath", "path"),
        comparison_strategy="file_path",
        color="#facc15",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="create_file",
        category=CATEGORY_EDIT,
        stage_hint="",
        equivalence_group="file_edit",
        operation_type="create",
        key_args=("filePath", "content"),
        file_path_args=("filePath", "path"),
        comparison_strategy="file_path",
        color="#4ade80",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="edit_notebook_file",
        category=CATEGORY_EDIT,
        stage_hint="",
        equivalence_group="notebook_edit",
        operation_type="modify",
        key_args=("filePath", "cellId", "editType"),
        file_path_args=("filePath",),
        comparison_strategy="file_path",
        color="#facc15",
        context_sensitive=True,
    ),
]
registry.register_many(_EDIT_TOOL_DEFS)

# ── Delete tools ──────────────────────────────────────────────────────

for _del_name in ("delete_file", "remove_file"):
    registry.register(ToolDescriptor(
        name=_del_name,
        category=CATEGORY_EDIT,
        stage_hint="",
        equivalence_group="file_delete",
        operation_type="delete",
        key_args=("filePath",),
        file_path_args=("filePath", "path"),
        comparison_strategy="file_path",
        color="#f87171",
        context_sensitive=True,
    ))

# ── Search / exploration tools ────────────────────────────────────────

_SEARCH_TOOL_DEFS = [
    ToolDescriptor(
        name="file_search",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("query",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="grep_search",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("query",),
        comparison_strategy="query",
        color="#f472b6",
    ),
    ToolDescriptor(
        name="semantic_search",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("query",),
        comparison_strategy="query",
        color="#fb923c",
    ),
    ToolDescriptor(
        name="list_dir",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="explore",
        operation_type="search",
        key_args=("path",),
        file_path_args=("path",),
        comparison_strategy="file_path",
        color="#2dd4bf",
    ),
    ToolDescriptor(
        name="list_code_usages",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("symbolName",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="get_search_view_results",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        comparison_strategy="identity",
        color="#c084fc",
    ),
]
registry.register_many(_SEARCH_TOOL_DEFS)

# ── Terminal / execution tools ────────────────────────────────────────

_EXEC_TOOL_DEFS = [
    ToolDescriptor(
        name="run_in_terminal",
        category=CATEGORY_EXECUTE,
        stage_hint="",                  # context-sensitive
        equivalence_group="terminal",
        operation_type="terminal",
        key_args=("command",),
        comparison_strategy="command",
        color="#f87171",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="get_terminal_output",
        category=CATEGORY_EXECUTE,
        stage_hint="",
        equivalence_group="terminal",
        operation_type="terminal",
        key_args=("id",),
        comparison_strategy="identity",
        color="#f87171",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="await_terminal",
        category=CATEGORY_EXECUTE,
        stage_hint="",
        equivalence_group="terminal",
        operation_type="terminal",
        key_args=("id",),
        comparison_strategy="identity",
        color="#f87171",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="kill_terminal",
        category=CATEGORY_EXECUTE,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="terminal",
        operation_type="terminal",
        key_args=("id",),
        comparison_strategy="identity",
        color="#f87171",
    ),
    ToolDescriptor(
        name="terminal_last_command",
        category=CATEGORY_EXECUTE,
        stage_hint=STAGE_VERIFICATION,
        equivalence_group="terminal",
        operation_type="terminal",
        comparison_strategy="identity",
        color="#f87171",
    ),
    ToolDescriptor(
        name="terminal_selection",
        category=CATEGORY_EXECUTE,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="terminal",
        operation_type="terminal",
        comparison_strategy="identity",
        color="#f87171",
    ),
    ToolDescriptor(
        name="run_task",
        category=CATEGORY_EXECUTE,
        stage_hint="",
        equivalence_group="terminal",
        operation_type="terminal",
        key_args=("label",),
        comparison_strategy="command",
        color="#f87171",
        context_sensitive=True,
    ),
]
registry.register_many(_EXEC_TOOL_DEFS)

# ── Validation tools ─────────────────────────────────────────────────

_VALIDATION_TOOL_DEFS = [
    ToolDescriptor(
        name="get_errors",
        category=CATEGORY_VALIDATION,
        stage_hint=STAGE_VERIFICATION,
        equivalence_group="diagnostics",
        operation_type="other",
        comparison_strategy="identity",
        color="#f87171",
    ),
    ToolDescriptor(
        name="test_failure",
        category=CATEGORY_VALIDATION,
        stage_hint=STAGE_VERIFICATION,
        equivalence_group="diagnostics",
        operation_type="other",
        comparison_strategy="identity",
        color="#f87171",
    ),
]
registry.register_many(_VALIDATION_TOOL_DEFS)

# ── File creation tools ───────────────────────────────────────────────

registry.register(ToolDescriptor(
    name="create_directory",
    category=CATEGORY_EDIT,
    stage_hint=STAGE_IMPLEMENTATION,
    equivalence_group="file_edit",
    operation_type="create",
    key_args=("dirPath",),
    file_path_args=("dirPath",),
    comparison_strategy="file_path",
    color="#4ade80",
))

# ── VCS / Git tools ──────────────────────────────────────────────────

registry.register(ToolDescriptor(
    name="get_changed_files",
    category=CATEGORY_SEARCH,
    stage_hint=STAGE_VERIFICATION,
    equivalence_group="vcs",
    operation_type="read",
    comparison_strategy="identity",
    color="#c084fc",
))

# ── Notebook tools ────────────────────────────────────────────────────

_NOTEBOOK_TOOL_DEFS = [
    ToolDescriptor(
        name="create_new_jupyter_notebook",
        category=CATEGORY_EDIT,
        stage_hint=STAGE_IMPLEMENTATION,
        equivalence_group="notebook_edit",
        operation_type="create",
        key_args=("query",),
        comparison_strategy="identity",
        color="#4ade80",
    ),
    ToolDescriptor(
        name="copilot_getNotebookSummary",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="notebook_read",
        operation_type="read",
        key_args=("filePath",),
        file_path_args=("filePath",),
        comparison_strategy="file_path",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="run_notebook_cell",
        category=CATEGORY_EXECUTE,
        stage_hint="",
        equivalence_group="notebook_exec",
        operation_type="terminal",
        key_args=("filePath", "cellId"),
        file_path_args=("filePath",),
        comparison_strategy="file_path",
        color="#f87171",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="read_notebook_cell_output",
        category=CATEGORY_READ,
        stage_hint="",
        equivalence_group="notebook_read",
        operation_type="read",
        key_args=("filePath", "cellId"),
        file_path_args=("filePath",),
        comparison_strategy="file_path",
        color="#60a5fa",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="configure_notebook",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="notebook_config",
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="configure_python_notebook",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="notebook_config",
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="configure_non_python_notebook",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="notebook_config",
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="notebook_install_packages",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="env_config",
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="notebook_list_packages",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="env_info",
        operation_type="read",
        comparison_strategy="identity",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="restart_notebook_kernel",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="notebook_config",
        operation_type="other",
        comparison_strategy="identity",
    ),
]
registry.register_many(_NOTEBOOK_TOOL_DEFS)

# ── Python environment tools ─────────────────────────────────────────

_PYTHON_ENV_TOOL_DEFS = [
    ToolDescriptor(
        name="configure_python_environment",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="env_config",
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="get_python_environment_details",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="env_info",
        operation_type="read",
        comparison_strategy="identity",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="get_python_executable_details",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="env_info",
        operation_type="read",
        comparison_strategy="identity",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="install_python_packages",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="env_config",
        operation_type="other",
        comparison_strategy="identity",
    ),
]
registry.register_many(_PYTHON_ENV_TOOL_DEFS)

# ── MCP Pylance tools (individually registered for proper metadata) ──

_MCP_PYLANCE_TOOL_DEFS = [
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceDocString",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="code_analysis",
        operation_type="read",
        key_args=("symbolName",),
        comparison_strategy="query",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceDocuments",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("query",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceFileSyntaxErrors",
        category=CATEGORY_VALIDATION,
        stage_hint=STAGE_VERIFICATION,
        equivalence_group="diagnostics",
        operation_type="other",
        key_args=("filePath",),
        file_path_args=("filePath",),
        comparison_strategy="file_path",
        color="#f87171",
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceImports",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="code_analysis",
        operation_type="search",
        comparison_strategy="identity",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceInstalledTopLevelModules",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="env_info",
        operation_type="search",
        comparison_strategy="identity",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceInvokeRefactoring",
        category=CATEGORY_EDIT,
        stage_hint=STAGE_IMPLEMENTATION,
        equivalence_group="file_edit",
        operation_type="modify",
        key_args=("filePath",),
        file_path_args=("filePath",),
        comparison_strategy="file_path",
        color="#facc15",
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylancePythonEnvironments",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="env_info",
        operation_type="read",
        comparison_strategy="identity",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceRunCodeSnippet",
        category=CATEGORY_EXECUTE,
        stage_hint="",
        equivalence_group="terminal",
        operation_type="terminal",
        key_args=("code",),
        comparison_strategy="command",
        color="#f87171",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceSettings",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="env_info",
        operation_type="read",
        comparison_strategy="identity",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceSyntaxErrors",
        category=CATEGORY_VALIDATION,
        stage_hint=STAGE_VERIFICATION,
        equivalence_group="diagnostics",
        operation_type="other",
        key_args=("code",),
        comparison_strategy="identity",
        color="#f87171",
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceUpdatePythonEnvironment",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="env_config",
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceWorkspaceRoots",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="explore",
        operation_type="search",
        comparison_strategy="identity",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="mcp_pylance_mcp_s_pylanceWorkspaceUserFiles",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="explore",
        operation_type="search",
        comparison_strategy="identity",
        color="#c084fc",
    ),
]
registry.register_many(_MCP_PYLANCE_TOOL_DEFS)

# ── MCP Time tools ───────────────────────────────────────────────────

_MCP_TIME_TOOL_DEFS = [
    ToolDescriptor(
        name="mcp_time_get_current_time",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="mcp_time_convert_time",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="identity",
    ),
]
registry.register_many(_MCP_TIME_TOOL_DEFS)

# ── General / orchestration tools ─────────────────────────────────────

_GENERAL_TOOL_DEFS = [
    ToolDescriptor(
        name="manage_todo_list",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="ask_questions",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="open_simple_browser",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_VERIFICATION,
        operation_type="other",
        key_args=("url",),
        comparison_strategy="identity",
        color="#2dd4bf",
    ),
    ToolDescriptor(
        name="fetch_webpage",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        operation_type="read",
        key_args=("urls",),
        comparison_strategy="query",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="github_repo",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("repo", "query"),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="get_vscode_api",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("query",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="install_extension",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="env_config",
        operation_type="other",
        key_args=("id",),
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="run_vscode_command",
        category=CATEGORY_EXECUTE,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        key_args=("commandId",),
        comparison_strategy="identity",
        color="#f87171",
    ),
    ToolDescriptor(
        name="create_and_run_task",
        category=CATEGORY_EXECUTE,
        stage_hint="",
        equivalence_group="terminal",
        operation_type="terminal",
        key_args=("task",),
        comparison_strategy="command",
        color="#f87171",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="get_task_output",
        category=CATEGORY_EXECUTE,
        stage_hint="",
        equivalence_group="terminal",
        operation_type="terminal",
        comparison_strategy="identity",
        color="#f87171",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="create_new_workspace",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="get_project_setup_info",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="switch_agent",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="runSubagent",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="identity",
    ),
    ToolDescriptor(
        name="renderMermaidDiagram",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_IMPLEMENTATION,
        operation_type="other",
        comparison_strategy="identity",
        color="#2dd4bf",
    ),
    ToolDescriptor(
        name="vscode_searchExtensions_internal",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("keywords",),
        comparison_strategy="query",
        color="#c084fc",
    ),
]
registry.register_many(_GENERAL_TOOL_DEFS)

# ── Prefix rules (fallback for unknown MCP tools) ────────────────────

registry.register_prefix(
    "mcp_pylance_",
    ToolDescriptor(
        name="__mcp_pylance__",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        operation_type="search",
        comparison_strategy="identity",
        color="#c084fc",
    ),
)

registry.register_prefix(
    "mcp_time_",
    ToolDescriptor(
        name="__mcp_time__",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="identity",
    ),
)

# Generic MCP fallback for any MCP tools not explicitly registered
registry.register_prefix(
    "mcp_",
    ToolDescriptor(
        name="__mcp_generic__",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="identity",
    ),
)

# ═══════════════════════════════════════════════════════════════════════
# ATIF / Copilot CLI native tools
# ═══════════════════════════════════════════════════════════════════════
# These are the raw tool names as they appear in ATIF trajectories
# from Copilot CLI.  The ATIF generator normalises them to canonical
# SDK names, but we register them here so that the registry can
# provide proper metadata if they are ever encountered directly.

_COPILOT_CLI_TOOL_DEFS = [
    # ── Copilot CLI core tools ──
    ToolDescriptor(
        name="bash",
        category=CATEGORY_EXECUTE,
        stage_hint="",
        equivalence_group="terminal",
        operation_type="terminal",
        key_args=("command",),
        comparison_strategy="command",
        color="#f87171",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="view",
        category=CATEGORY_READ,
        stage_hint="",
        equivalence_group="file_read",
        operation_type="read",
        key_args=("path", "view_range"),
        file_path_args=("path",),
        comparison_strategy="file_path",
        color="#60a5fa",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="create",
        category=CATEGORY_EDIT,
        stage_hint="",
        equivalence_group="file_edit",
        operation_type="create",
        key_args=("path", "content"),
        file_path_args=("path",),
        comparison_strategy="file_path",
        color="#4ade80",
        context_sensitive=True,
    ),
    # Note: Copilot CLI "edit" ≠ existing SDK "edit_file"; registering
    # it separately so that equivalence-group matching still works.
    ToolDescriptor(
        name="edit",
        category=CATEGORY_EDIT,
        stage_hint="",
        equivalence_group="file_edit",
        operation_type="modify",
        key_args=("path", "old_str", "new_str"),
        file_path_args=("path",),
        comparison_strategy="file_path",
        color="#facc15",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="glob",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("pattern",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="grep",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("pattern",),
        comparison_strategy="query",
        color="#f472b6",
    ),
    ToolDescriptor(
        name="web_fetch",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        operation_type="read",
        key_args=("url",),
        comparison_strategy="query",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="web_search",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        operation_type="search",
        key_args=("query",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="task",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        key_args=("prompt",),
        comparison_strategy="none",
    ),
    # ── Copilot CLI shell management ──
    ToolDescriptor(
        name="write_bash",
        category=CATEGORY_EXECUTE,
        stage_hint="",
        equivalence_group="terminal",
        operation_type="terminal",
        key_args=("input",),
        comparison_strategy="command",
        color="#f87171",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="read_bash",
        category=CATEGORY_EXECUTE,
        stage_hint="",
        equivalence_group="terminal",
        operation_type="terminal",
        comparison_strategy="identity",
        color="#f87171",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="stop_bash",
        category=CATEGORY_EXECUTE,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="terminal",
        operation_type="terminal",
        comparison_strategy="identity",
        color="#f87171",
    ),
    ToolDescriptor(
        name="list_bash",
        category=CATEGORY_EXECUTE,
        stage_hint=STAGE_ORCHESTRATION,
        equivalence_group="terminal",
        operation_type="terminal",
        comparison_strategy="identity",
        color="#f87171",
    ),
    # ── Copilot CLI orchestration tools ──
    ToolDescriptor(
        name="report_intent",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="store_memory",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="ask_user",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="sql",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="read_agent",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="list_agents",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="fetch_copilot_cli_documentation",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        operation_type="read",
        comparison_strategy="identity",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="skill",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="show_file",
        category=CATEGORY_READ,
        stage_hint="",
        equivalence_group="file_read",
        operation_type="read",
        key_args=("path",),
        file_path_args=("path",),
        comparison_strategy="file_path",
        color="#60a5fa",
        context_sensitive=True,
    ),
]
registry.register_many(_COPILOT_CLI_TOOL_DEFS)


# ═══════════════════════════════════════════════════════════════════════
# ATIF / Copilot Azure MCP tools  (55 tools)
# ═══════════════════════════════════════════════════════════════════════
# Azure service integration tools available to Copilot CLI via MCP.
# Each is a hierarchical command router for a specific Azure service.

# Helper: build an Azure MCP descriptor with sensible defaults.
def _azure_mcp(
    name: str,
    *,
    category: str = CATEGORY_GENERAL,
    stage_hint: str = STAGE_ORCHESTRATION,
    operation_type: str = "other",
    equivalence_group: str = "azure_mcp",
    comparison_strategy: str = "identity",
    color: str = "#38bdf8",
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        category=category,
        stage_hint=stage_hint,
        equivalence_group=equivalence_group,
        operation_type=operation_type,
        key_args=("command",),
        comparison_strategy=comparison_strategy,
        color=color,
    )


_AZURE_MCP_TOOL_DEFS = [
    # ── Documentation & best-practices ──
    _azure_mcp("azure-documentation", category=CATEGORY_SEARCH,
               stage_hint=STAGE_EXPLORATION, operation_type="search",
               comparison_strategy="query", color="#c084fc"),
    _azure_mcp("azure-get_azure_bestpractices", category=CATEGORY_READ,
               stage_hint=STAGE_EXPLORATION, operation_type="read", color="#60a5fa"),
    _azure_mcp("azure-azureterraformbestpractices", category=CATEGORY_READ,
               stage_hint=STAGE_EXPLORATION, operation_type="read", color="#60a5fa"),
    _azure_mcp("azure-bicepschema", category=CATEGORY_READ,
               stage_hint=STAGE_EXPLORATION, operation_type="read", color="#60a5fa"),
    _azure_mcp("azure-pricing", category=CATEGORY_READ,
               stage_hint=STAGE_EXPLORATION, operation_type="read", color="#60a5fa"),
    _azure_mcp("azure-marketplace", category=CATEGORY_READ,
               stage_hint=STAGE_EXPLORATION, operation_type="read", color="#60a5fa"),
    # ── Architecture & deployment ──
    _azure_mcp("azure-cloudarchitect"),
    _azure_mcp("azure-deploy", category=CATEGORY_EXECUTE,
               stage_hint=STAGE_IMPLEMENTATION, operation_type="terminal",
               color="#f87171"),
    _azure_mcp("azure-azd", category=CATEGORY_EXECUTE,
               stage_hint=STAGE_IMPLEMENTATION, operation_type="terminal",
               color="#f87171"),
    _azure_mcp("azure-azuremigrate"),
    # ── Compute & containers ──
    _azure_mcp("azure-compute"),
    _azure_mcp("azure-aks"),
    _azure_mcp("azure-acr"),
    _azure_mcp("azure-servicefabric"),
    _azure_mcp("azure-virtualdesktop"),
    _azure_mcp("azure-functionapp"),
    _azure_mcp("azure-appservice"),
    # ── Data & databases ──
    _azure_mcp("azure-cosmos"),
    _azure_mcp("azure-sql"),
    _azure_mcp("azure-mysql"),
    _azure_mcp("azure-postgres"),
    _azure_mcp("azure-redis"),
    _azure_mcp("azure-kusto"),
    _azure_mcp("azure-confidentialledger"),
    # ── Storage & filesystems ──
    _azure_mcp("azure-storage"),
    _azure_mcp("azure-fileshares"),
    _azure_mcp("azure-storagesync"),
    _azure_mcp("azure-managedlustre"),
    # ── Messaging & eventing ──
    _azure_mcp("azure-eventgrid"),
    _azure_mcp("azure-eventhubs"),
    _azure_mcp("azure-servicebus"),
    _azure_mcp("azure-signalr"),
    _azure_mcp("azure-communication"),
    # ── Monitoring & diagnostics ──
    _azure_mcp("azure-monitor", category=CATEGORY_VALIDATION,
               stage_hint=STAGE_VERIFICATION, operation_type="other",
               color="#f87171"),
    _azure_mcp("azure-applicationinsights", category=CATEGORY_VALIDATION,
               stage_hint=STAGE_VERIFICATION, operation_type="other",
               color="#f87171"),
    _azure_mcp("azure-applens", category=CATEGORY_VALIDATION,
               stage_hint=STAGE_VERIFICATION, operation_type="other",
               color="#f87171"),
    _azure_mcp("azure-resourcehealth", category=CATEGORY_VALIDATION,
               stage_hint=STAGE_VERIFICATION, operation_type="other",
               color="#f87171"),
    _azure_mcp("azure-grafana", category=CATEGORY_VALIDATION,
               stage_hint=STAGE_VERIFICATION, operation_type="other",
               color="#f87171"),
    _azure_mcp("azure-datadog", category=CATEGORY_VALIDATION,
               stage_hint=STAGE_VERIFICATION, operation_type="other",
               color="#f87171"),
    _azure_mcp("azure-workbooks", category=CATEGORY_READ,
               stage_hint=STAGE_EXPLORATION, operation_type="read",
               color="#60a5fa"),
    # ── Networking & security ──
    _azure_mcp("azure-keyvault"),
    _azure_mcp("azure-policy"),
    _azure_mcp("azure-role"),
    _azure_mcp("azure-quota", category=CATEGORY_READ,
               stage_hint=STAGE_EXPLORATION, operation_type="read",
               color="#60a5fa"),
    _azure_mcp("azure-loadtesting", category=CATEGORY_VALIDATION,
               stage_hint=STAGE_VERIFICATION, operation_type="other",
               color="#f87171"),
    # ── AI & search ──
    _azure_mcp("azure-search"),
    _azure_mcp("azure-speech"),
    _azure_mcp("azure-foundry"),
    # ── Configuration & management ──
    _azure_mcp("azure-appconfig"),
    _azure_mcp("azure-advisor", category=CATEGORY_READ,
               stage_hint=STAGE_EXPLORATION, operation_type="read",
               color="#60a5fa"),
    # ── Subscription & resource management ──
    _azure_mcp("azure-group_list"),
    _azure_mcp("azure-subscription_list"),
    # ── CLI extensions ──
    _azure_mcp("azure-extension_azqr", category=CATEGORY_VALIDATION,
               stage_hint=STAGE_VERIFICATION, operation_type="other",
               color="#f87171"),
    _azure_mcp("azure-extension_cli_generate", category=CATEGORY_EXECUTE,
               stage_hint=STAGE_IMPLEMENTATION, operation_type="terminal",
               color="#f87171"),
    _azure_mcp("azure-extension_cli_install", category=CATEGORY_GENERAL,
               stage_hint=STAGE_ORCHESTRATION, operation_type="other",
               color="#a1a1aa"),
]
registry.register_many(_AZURE_MCP_TOOL_DEFS)

# Clean up helper so it doesn't pollute the module namespace
del _azure_mcp


# ═══════════════════════════════════════════════════════════════════════
# ATIF / Copilot GitHub MCP Server tools  (18 tools)
# ═══════════════════════════════════════════════════════════════════════
# GitHub API integration tools available to Copilot CLI via MCP server.

_GITHUB_MCP_TOOL_DEFS = [
    # ── Actions ──
    ToolDescriptor(
        name="github-mcp-server-actions_get",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="read",
        comparison_strategy="identity",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="github-mcp-server-actions_list",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="search",
        comparison_strategy="query",
        color="#c084fc",
    ),
    # ── Repository content ──
    ToolDescriptor(
        name="github-mcp-server-get_commit",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="read",
        key_args=("owner", "repo", "sha"),
        comparison_strategy="identity",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="github-mcp-server-get_file_contents",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="read",
        key_args=("owner", "repo", "path"),
        file_path_args=("path",),
        comparison_strategy="file_path",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="github-mcp-server-get_job_logs",
        category=CATEGORY_READ,
        stage_hint=STAGE_VERIFICATION,
        equivalence_group="github_mcp",
        operation_type="read",
        comparison_strategy="identity",
        color="#f87171",
    ),
    # ── Branches & commits ──
    ToolDescriptor(
        name="github-mcp-server-list_branches",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="search",
        key_args=("owner", "repo"),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="github-mcp-server-list_commits",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="search",
        key_args=("owner", "repo"),
        comparison_strategy="query",
        color="#c084fc",
    ),
    # ── Issues ──
    ToolDescriptor(
        name="github-mcp-server-issue_read",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="read",
        key_args=("owner", "repo", "issue_number"),
        comparison_strategy="identity",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="github-mcp-server-list_issues",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="search",
        key_args=("owner", "repo"),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="github-mcp-server-search_issues",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="search",
        key_args=("q",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    # ── Pull requests ──
    ToolDescriptor(
        name="github-mcp-server-list_pull_requests",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="search",
        key_args=("owner", "repo"),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="github-mcp-server-pull_request_read",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="read",
        key_args=("owner", "repo", "pullNumber"),
        comparison_strategy="identity",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="github-mcp-server-search_pull_requests",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="search",
        key_args=("q",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    # ── Code search ──
    ToolDescriptor(
        name="github-mcp-server-search_code",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="search",
        key_args=("q",),
        comparison_strategy="query",
        color="#f472b6",
    ),
    # ── Repository search ──
    ToolDescriptor(
        name="github-mcp-server-search_repositories",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="search",
        key_args=("query",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="github-mcp-server-search_users",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="search",
        key_args=("q",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    # ── Copilot Spaces ──
    ToolDescriptor(
        name="github-mcp-server-get_copilot_space",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="read",
        comparison_strategy="identity",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="github-mcp-server-list_copilot_spaces",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="github_mcp",
        operation_type="search",
        comparison_strategy="identity",
        color="#c084fc",
    ),
]
registry.register_many(_GITHUB_MCP_TOOL_DEFS)


# ═══════════════════════════════════════════════════════════════════════
# ATIF / Claude Code native tools
# ═══════════════════════════════════════════════════════════════════════
# PascalCase tool names from Claude Code ATIF trajectories.

_CLAUDE_CODE_TOOL_DEFS = [
    ToolDescriptor(
        name="Bash",
        category=CATEGORY_EXECUTE,
        stage_hint="",
        equivalence_group="terminal",
        operation_type="terminal",
        key_args=("command",),
        comparison_strategy="command",
        color="#f87171",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="Read",
        category=CATEGORY_READ,
        stage_hint="",
        equivalence_group="file_read",
        operation_type="read",
        key_args=("file_path", "offset", "limit"),
        file_path_args=("file_path",),
        comparison_strategy="file_path",
        color="#60a5fa",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="Write",
        category=CATEGORY_EDIT,
        stage_hint="",
        equivalence_group="file_edit",
        operation_type="create",
        key_args=("file_path", "content"),
        file_path_args=("file_path",),
        comparison_strategy="file_path",
        color="#4ade80",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="Edit",
        category=CATEGORY_EDIT,
        stage_hint="",
        equivalence_group="file_edit",
        operation_type="modify",
        key_args=("file_path", "old_string", "new_string"),
        file_path_args=("file_path",),
        comparison_strategy="file_path",
        color="#facc15",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="Glob",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("pattern",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="Grep",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        equivalence_group="search",
        operation_type="search",
        key_args=("pattern",),
        comparison_strategy="query",
        color="#f472b6",
    ),
    ToolDescriptor(
        name="Task",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        key_args=("prompt",),
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="WebFetch",
        category=CATEGORY_READ,
        stage_hint=STAGE_EXPLORATION,
        operation_type="read",
        key_args=("url",),
        comparison_strategy="query",
        color="#60a5fa",
    ),
    ToolDescriptor(
        name="WebSearch",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        operation_type="search",
        key_args=("query",),
        comparison_strategy="query",
        color="#c084fc",
    ),
    ToolDescriptor(
        name="AskUserQuestion",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="NotebookEdit",
        category=CATEGORY_EDIT,
        stage_hint="",
        equivalence_group="notebook_edit",
        operation_type="modify",
        key_args=("notebook_path", "cell_id"),
        file_path_args=("notebook_path",),
        comparison_strategy="file_path",
        color="#facc15",
        context_sensitive=True,
    ),
    ToolDescriptor(
        name="EnterPlanMode",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="ExitPlanMode",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="EnterWorktree",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="TaskOutput",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="TaskStop",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="TaskCreate",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="TaskGet",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="TaskUpdate",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="TaskList",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="Skill",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="none",
    ),
    ToolDescriptor(
        name="Agent",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        key_args=("prompt",),
        comparison_strategy="none",
    ),
]
registry.register_many(_CLAUDE_CODE_TOOL_DEFS)

# ── ATIF prefix rules ────────────────────────────────────────────────
# Catch-all for Azure MCP tools (azure-*) and GitHub MCP tools
# (github-mcp-server-*) that appear in ATIF trajectories.

registry.register_prefix(
    "azure-",
    ToolDescriptor(
        name="__azure_mcp__",
        category=CATEGORY_GENERAL,
        stage_hint=STAGE_ORCHESTRATION,
        operation_type="other",
        comparison_strategy="identity",
    ),
)

registry.register_prefix(
    "github-mcp-server-",
    ToolDescriptor(
        name="__github_mcp__",
        category=CATEGORY_SEARCH,
        stage_hint=STAGE_EXPLORATION,
        operation_type="search",
        comparison_strategy="query",
        color="#c084fc",
    ),
)


# ── Special pseudo-entries (for visualization / LLM requests) ─────────

registry.register(ToolDescriptor(
    name="request",
    category=CATEGORY_GENERAL,
    stage_hint=STAGE_ORCHESTRATION,
    operation_type="other",
    comparison_strategy="none",
    color="#94a3b8",
))

registry.register(ToolDescriptor(
    name="think",
    category=CATEGORY_GENERAL,
    stage_hint=STAGE_ORCHESTRATION,
    operation_type="other",
    comparison_strategy="none",
    color="#a78bfa",
))
