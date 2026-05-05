"""Intent-stage labeling and workflow fingerprinting for trace states.

Assigns each :class:`~swe_trace_sdk.models.State` an **intent stage**
that captures the developer's *intent* behind an action, not merely its
tool category.  This reframing emphasises cognitive purpose:

* **exploration** — actions aimed at understanding the codebase
  (searching, reading source files, listing directories, …).
* **implementation** — actions that modify source code to address the task
  (creating / editing source files).
* **verification** — actions that confirm whether the change is correct
  (running tests, checking errors, reading test output, …).
* **orchestration** — bookkeeping actions that coordinate the workflow
  but don't directly contribute to exploration, implementation, or
  verification (todo lists, browser, extensions, …).

The algorithm is **context-aware**: the same tool can belong to different
stages depending on:

1. Whether the file being touched is a *test* file or a *source* file.
2. Whether any implementation action has already occurred in prior steps.
3. The specific terminal command being run (test runner vs. other).

Workflow fingerprinting
-----------------------
After labeling, this module can compute a **workflow fingerprint** — a
compact representation of the stage-transition sequence in a trace.
Transitions between consecutive states are classified as:

* **deepen**    — staying in the same stage (e.g. E→E).
* **pivot**     — moving forward to a later stage (e.g. E→I).
* **backtrack** — returning to an earlier stage (e.g. V→E).
* **confirm**   — transitioning into orchestration from any stage.

The fingerprint enables a ``workflow_similarity`` metric that compares
the behavioural pattern of a candidate against the ground truth.

Public API
----------
- :func:`label_trace_intents`   — label all states in a :class:`Trace` in-place.
- :func:`label_intent_stages`   — label a list of states in step order.
- :func:`get_intent_stage`      — get the stage for a single state given context.
- :func:`compute_fingerprint`   — compute a workflow fingerprint for a list of states.
- :func:`workflow_similarity`   — compare two workflow fingerprints (0.0–1.0).

Constants
---------
- :data:`EXPLORATION`, :data:`IMPLEMENTATION`, :data:`VERIFICATION`, :data:`ORCHESTRATION`
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .tool_registry import (
    registry,
    CATEGORY_EDIT,
    CATEGORY_EXECUTE,
    CATEGORY_READ,
    CATEGORY_SEARCH,
    CATEGORY_VALIDATION,
    CATEGORY_GENERAL,
    STAGE_EXPLORATION,
    STAGE_IMPLEMENTATION,
    STAGE_VERIFICATION,
    STAGE_ORCHESTRATION,
)

if TYPE_CHECKING:
    from .models import State, Trace

logger = logging.getLogger(__name__)

__all__ = [
    "label_trace_intents",
    "label_intent_stages",
    "get_intent_stage",
    "compute_fingerprint",
    "workflow_similarity",
    "compute_coherence_score",
    "compute_temporal_profile_divergence",
    "EXPLORATION",
    "IMPLEMENTATION",
    "VERIFICATION",
    "ORCHESTRATION",
]


# ──────────────────────────────────────────────────────────────────────
# Intent-stage constants (re-exported from tool_registry)
# ──────────────────────────────────────────────────────────────────────

EXPLORATION = STAGE_EXPLORATION
IMPLEMENTATION = STAGE_IMPLEMENTATION
VERIFICATION = STAGE_VERIFICATION
ORCHESTRATION = STAGE_ORCHESTRATION

# Ordinal ranking for transition classification
_STAGE_ORDER: Dict[str, int] = {
    EXPLORATION: 0,
    IMPLEMENTATION: 1,
    VERIFICATION: 2,
    ORCHESTRATION: 3,
}


# ──────────────────────────────────────────────────────────────────────
# Tool → fixed-stage mapping (derived from registry)
# ──────────────────────────────────────────────────────────────────────

def _build_tool_sets():
    """Derive tool-category sets from the registry.

    Called once at module load.  The sets are cached as module globals.
    """
    exploration = registry.by_category(CATEGORY_SEARCH)
    verification = registry.by_category(CATEGORY_VALIDATION)
    orchestration_tools = frozenset(
        name for name in registry.all_names()
        if registry.get(name).stage_hint == STAGE_ORCHESTRATION
        and not registry.get(name).context_sensitive
    )
    edit_tools = registry.by_category(CATEGORY_EDIT)
    read_tools = registry.by_category(CATEGORY_READ)
    terminal_tools = registry.by_category(CATEGORY_EXECUTE)
    return exploration, verification, orchestration_tools, edit_tools, read_tools, terminal_tools


(_EXPLORATION_TOOLS,
 _VERIFICATION_TOOLS,
 _ORCHESTRATION_TOOLS,
 _EDIT_TOOLS,
 _READ_TOOLS,
 _TERMINAL_TOOLS) = _build_tool_sets()


# ──────────────────────────────────────────────────────────────────────
# Test-file heuristics
# ──────────────────────────────────────────────────────────────────────

# Patterns that indicate a file is a test file
_TEST_FILE_PATTERNS = re.compile(
    r"(?:^|[/\\])"          # start of path or directory separator
    r"(?:"
    r"test_[^/\\]*"         # test_something.py
    r"|[^/\\]*_test\."      # something_test.py
    r"|tests?[/\\]"         # tests/ or test/ directory
    r"|spec[/\\]"           # spec/ directory
    r"|__tests__[/\\]"      # __tests__/ (JS convention)
    r"|[^/\\]*\.test\."     # something.test.js
    r"|[^/\\]*\.spec\."     # something.spec.js
    r"|conftest\."          # conftest.py
    r"|pytest"              # pytest config files
    r")",
    re.IGNORECASE,
)

# Terminal commands that indicate test execution
_TEST_CMD_PATTERNS = re.compile(
    r"(?:^|\s|&&|;|/)"
    r"(?:"
    r"pytest"
    r"|python\d?\s+(?:-m\s+)?(?:pytest|unittest)"
    r"|python\d?\s+(?:test_|tests/|run_tests)"
    r"|npm\s+(?:test|run\s+test)"
    r"|npx\s+(?:jest|mocha|vitest)"
    r"|jest"
    r"|mocha"
    r"|cargo\s+test"
    r"|go\s+test"
    r"|dotnet\s+test"
    r"|make\s+test"
    r"|tox"
    r"|nosetests"
    r")",
    re.IGNORECASE,
)


def _is_test_file(file_path: str) -> bool:
    """Heuristic: does *file_path* look like a test file?"""
    if not file_path:
        return False
    return bool(_TEST_FILE_PATTERNS.search(file_path))


_SUMMARY_FILE_PATTERNS = re.compile(
    r"(?:"
    r"(?:^|/)(?:FIX_SUMMARY|NOTES|CHANGELOG|SUMMARY|REPORT)"
    r"|_summary|_report|_notes"
    r")"
    r"\.[a-zA-Z]+$",
    re.IGNORECASE,
)


def _is_summary_file(file_path: str) -> bool:
    """Heuristic: is *file_path* a summary/documentation file (not source code)?

    These files are orchestration artefacts — the agent is writing a report
    or summary, not implementing a fix.
    """
    if not file_path:
        return False
    # Files written to /tmp are never source code
    if file_path.startswith("/tmp/") or file_path.startswith("tmp/"):
        return True
    return bool(_SUMMARY_FILE_PATTERNS.search(file_path))


# Terminal commands that are always exploration regardless of context.
# NOTE: ``cat`` is deliberately *not* included here — agents frequently
# use ``cat > file << 'EOF'`` to write code via terminal.  ``cat`` is
# handled as exploration only when it doesn't redirect output to a file.
_EXPLORATION_CMD_PATTERNS = re.compile(
    r"^\s*(?:cd\s+[^&;]+[&;]+\s*)?"  # optional cd prefix
    r"(?:"
    r"grep|rg|ag|ack"               # search tools
    r"|find\b"                       # file finder
    r"|head\b|tail\b"               # file viewers (cat handled below)
    r"|sed\s+-?n"                    # sed in print mode (viewing)
    r"|less\b|more\b"               # pagers
    r"|wc\b"                         # word count
    r"|ls\b|dir\b"                   # directory listing
    r"|tree\b"                       # tree listing
    r"|git\s+(?:log|show|blame)"     # read-only git commands
    r"|file\b"                       # file type check
    r"|jq\b"                         # JSON query tool (read-only)
    r"|realpath\b"                   # path resolution
    r")",
    re.IGNORECASE,
)

# Exploration sub-pattern for ``cat`` — only when NOT redirecting output.
_CAT_EXPLORATION_RE = re.compile(
    r"^\s*(?:cd\s+[^&;]+[&;]+\s*)?cat\b"
    r"(?!\s*>)",                      # negative lookahead: not followed by >
    re.IGNORECASE,
)

# ``git diff`` and ``git status`` are exploration *before* implementation
# but become verification *after* implementation.
_GIT_REVIEW_CMD_RE = re.compile(
    r"\bgit\s+(?:diff|status)\b",
    re.IGNORECASE,
)

# Terminal commands that are workspace-management / cleanup — orchestration.
_ORCHESTRATION_CMD_RE = re.compile(
    r"^\s*(?:cd\s+[^&;]+[&;]+\s*)*"  # optional cd prefix
    r"(?:"
    r"rm\s+(?:-[rfv]+\s+)*\S+"       # rm / rm -f / rm -rf
    r"|git\s+stash(?:\s+(?:pop|drop|apply))?"  # git stash management
    r"|git\s+checkout\s+--\s*\."      # git checkout -- .
    r"|git\s+restore\b"               # git restore
    r"|git\s+clean\b"                 # git clean
    r"|mkdir\s+-?p?\s"                 # directory scaffolding
    r")\s*$",
    re.IGNORECASE,
)


def _is_orchestration_command(command: str) -> bool:
    """Heuristic: is *command* a cleanup / workspace-management action?

    Returns True for commands like ``rm -f temp_file.py``,
    ``git stash pop``, etc. that are orchestration, not E/I/V.
    """
    if not command:
        return False
    # Strip cd prefix for analysis
    stripped = re.sub(r"^\s*(?:cd\s+[^&;]+[&;]+\s*)+", "", command).strip()
    if not stripped:
        return False
    # cat/echo/tee redirecting to /tmp → orchestration (writing summaries)
    if re.search(
        r"(?:cat|echo|printf|tee)\s+.*>\s*/tmp/",
        stripped, re.IGNORECASE,
    ):
        return True
    return bool(_ORCHESTRATION_CMD_RE.search(command))


def _is_verification_command(command: str) -> bool:
    """Heuristic: is this a verification (not exploration) terminal command?

    Returns True for commands that execute scripts, run applications, or
    check results — actions that verify whether implementation is correct.
    Returns False for exploration commands (grep, cat, find, ls, git log…).
    """
    if not command:
        return False
    # Strip common cd prefix
    stripped = re.sub(r"^\s*cd\s+[^&;]+[&;]+\s*", "", command).strip()
    if not stripped:
        return False
    # Exploration commands stay exploration
    if _EXPLORATION_CMD_PATTERNS.search(command):
        return False

    # Script execution — match interpreter + optional flags + script file.
    # Handles: python -u script.py, python3 -m module, node --check file.js, etc.
    if re.search(r"\bpython\d?\s+(?:-\S+\s+)*\S+\.py\b", stripped, re.IGNORECASE):
        return True
    if re.search(r"\bpython\d?\s+-m\s+\S+", stripped, re.IGNORECASE):
        return True
    if re.search(r"\bnode\s+(?:-\S+\s+)*\S+\.(?:js|ts|mjs|cjs)\b", stripped, re.IGNORECASE):
        return True
    if re.search(r"\b(?:bash|sh|zsh|ksh)\s+(?:-\S+\s+)*\S+\.sh\b", stripped, re.IGNORECASE):
        return True
    if re.search(r"\bruby\s+(?:-\S+\s+)*\S+\.rb\b", stripped, re.IGNORECASE):
        return True
    if re.search(r"\bphp\s+(?:-\S+\s+)*\S+\.php\b", stripped, re.IGNORECASE):
        return True
    if re.search(r"\bperl\s+(?:-\S+\s+)*\S+\.pl\b", stripped, re.IGNORECASE):
        return True
    if re.search(r"\bjava\s+(?:-\S+\s+)*\S+", stripped, re.IGNORECASE):
        return True

    # Build/run/serve commands
    if re.search(
        r"\b(?:make|cmake|cargo\s+run|go\s+run|go\s+build"
        r"|npm\s+(?:start|run|exec)|npx\s+\S+"
        r"|yarn\s+(?:start|run)|pnpm\s+(?:start|run)"
        r"|pip\s+install|dotnet\s+run|dotnet\s+build"
        r"|mvn\s+(?:compile|package|exec)|gradle\s+(?:run|build)"
        r"|docker\s+(?:run|compose|build)"
        r"|uvicorn|gunicorn|flask\s+run|streamlit\s+run"
        r")\b",
        stripped, re.IGNORECASE,
    ):
        return True

    # HTTP verification (curl, wget, httpie)
    if re.search(r"\b(?:curl|wget|http|https)\s", stripped, re.IGNORECASE):
        return True

    # Generic execution of a script (./script)
    if re.search(r"\./\S+", stripped):
        return True

    return False


def _is_test_command(command: str) -> bool:
    """Heuristic: does *command* look like a test-execution command?"""
    if not command:
        return False
    return bool(_TEST_CMD_PATTERNS.search(command))


def _get_command_from_state(state: "State") -> str:
    """Extract the terminal command string from a state."""
    entry = getattr(state, "log_entry", None)
    if entry is None:
        return ""
    args = getattr(entry, "args", None) or {}
    return args.get("command", "")


def _get_file_path_from_state(state: "State") -> str:
    """Get the normalized file path from a state."""
    fp = getattr(state, "file_path", "") or ""
    if fp:
        return fp
    # Fallback: try to extract from log_entry args
    entry = getattr(state, "log_entry", None)
    if entry is None:
        return ""
    args = getattr(entry, "args", None) or {}
    return args.get("filePath", args.get("path", ""))


def _is_modifying_command(command: str) -> bool:
    """Heuristic: does *command* modify files (sed -i, echo >>, tee, patch…)?

    Used to detect implicit implementation via terminal commands so that
    subsequent commands are correctly classified as verification.
    """
    if not command:
        return False
    return bool(re.search(
        r"(?:"
        r"\bsed\s+-i\b"                  # sed in-place edit
        r"|\bpatch\b"                     # applying a patch
        r"|\btee\s"                       # tee writes to file
        r"|>\s*\S"                         # redirect to file (>, >>)
        r"|\bchmod\b"                     # permission change
        r"|\bmv\b"                         # move/rename
        r"|\bcp\b"                         # copy
        r"|\binstall\b"                   # install command
        r"|\bpip\s+install\b"            # pip install
        r"|\bnpm\s+install\b"            # npm install
        r"|\brm\s"                        # file deletion
        r"|\bgit\s+checkout\s+--"         # git revert file
        r"|\bgit\s+restore\b"            # git restore file
        r")",
        command, re.IGNORECASE,
    ))


def _is_code_writing_command(command: str) -> bool:
    """Heuristic: does *command* write code/content to a source file via terminal?

    Detects patterns where agents use the terminal as an editor:
    - ``python -c "...open(...write..."``
    - ``cat > file.py << 'EOF'``
    - ``sed -i 's/old/new/' file.py``
    - ``echo '...' > file.py``

    These should be labeled *implementation* rather than exploration or
    verification, because the agent is creating or modifying source code.
    """
    if not command:
        return False
    # Strip cd prefix for analysis
    stripped = re.sub(r"^\s*(?:cd\s+[^&;]+[&;]+\s*)+", "", command).strip()

    # python -c / python3 -c with file write operations
    if re.search(
        r"python\d?\s+-c\s+[\"'].*(?:open\(|write\(|with\s+open)",
        stripped, re.DOTALL | re.IGNORECASE,
    ):
        return True

    # cat > source_file << heredoc
    if re.search(
        r"cat\s+>\s*\S+\.(?:py|js|ts|jsx|tsx|java|cs|rb|go|rs|cpp|c|h|csproj|json|yaml|yml|xml|toml|cfg|ini|sh)\s*<<",
        stripped, re.IGNORECASE,
    ):
        return True

    # echo/printf to source file
    if re.search(
        r"(?:echo|printf)\s+.*>\s*\S+\.(?:py|js|ts|java|cs|rb|go|rs|cpp|c|h)",
        stripped, re.IGNORECASE,
    ):
        return True

    # sed -i on source file
    if re.search(
        r"sed\s+-i\s+.*\S+\.(?:py|js|ts|java|cs|rb|go|rs|cpp|c|h|csproj|json|yaml|yml|xml|toml)",
        stripped, re.IGNORECASE,
    ):
        return True

    # tee to source file
    if re.search(
        r"tee\s+\S+\.(?:py|js|ts|java|cs|rb|go|rs|cpp|c|h)",
        stripped, re.IGNORECASE,
    ):
        return True

    return False


# ──────────────────────────────────────────────────────────────────────
# Core labeling logic
# ──────────────────────────────────────────────────────────────────────

def get_intent_stage(
    state: "State",
    *,
    implementation_occurred: bool = False,
    implemented_files: Optional[Set[str]] = None,
    is_near_end: bool = False,
) -> str:
    """Determine the intent-stage label for a single *state*.

    Parameters
    ----------
    state : State
        The state to label.
    implementation_occurred : bool
        Whether any edit/create to a **source** file has already happened
        in prior steps.  This disambiguates ``read_file`` (exploration
        vs post-implementation review) and ``run_in_terminal``
        (reproduction script vs verification).
    implemented_files : set[str], optional
        Set of file paths that have been modified so far.  Used to
        decide whether a ``read_file`` of a previously-modified file
        is verification (reviewing the result) vs exploration.
    is_near_end : bool
        Whether this state is in the final ~20% of the trajectory.
        Used to bias ambiguous terminal commands toward verification.

    Returns
    -------
    str
        One of ``"exploration"``, ``"implementation"``,
        ``"verification"``, ``"orchestration"``, or ``""``
        (for initial/unknown states).
    """
    if implemented_files is None:
        implemented_files = set()

    tool = getattr(state, "tool_used", None) or ""
    rs = getattr(state, "resulting_state", "") or ""

    # ── Initial state / LLM planning ──────────────────────────────
    if rs in ("initial", ""):
        return ""
    if rs.startswith("llm_response:"):
        if rs == "llm_response:planning":
            return ORCHESTRATION
        return ORCHESTRATION

    # ── Fixed-stage tools ─────────────────────────────────────────
    if tool in _EXPLORATION_TOOLS:
        return EXPLORATION

    if tool in _VERIFICATION_TOOLS:
        return VERIFICATION

    if tool in _ORCHESTRATION_TOOLS:
        return ORCHESTRATION

    # ── Context-sensitive: file edit/create tools ─────────────────
    if tool in _EDIT_TOOLS:
        fp = _get_file_path_from_state(state)
        if _is_test_file(fp):
            return VERIFICATION if implementation_occurred else EXPLORATION
        if _is_summary_file(fp):
            return ORCHESTRATION
        return IMPLEMENTATION

    # ── Context-sensitive: read_file ──────────────────────────────
    if tool in _READ_TOOLS:
        fp = _get_file_path_from_state(state)
        if _is_test_file(fp):
            return VERIFICATION if implementation_occurred else EXPLORATION
        else:
            if implementation_occurred and fp and fp in implemented_files:
                return VERIFICATION
            return EXPLORATION

    # ── Context-sensitive: terminal commands ──────────────────────
    if tool in _TERMINAL_TOOLS:
        cmd = _get_command_from_state(state)
        if _is_test_command(cmd):
            return VERIFICATION
        # Cleanup / workspace management → orchestration
        if _is_orchestration_command(cmd):
            return ORCHESTRATION
        # Terminal code-writing (python -c with open/write, sed -i on
        # source, cat > file.py << EOF) is always *implementation*
        # regardless of whether other edits already happened.
        if _is_code_writing_command(cmd):
            return IMPLEMENTATION
        if implementation_occurred and _is_verification_command(cmd):
            return VERIFICATION
        # git diff / git status after implementation → verification
        if implementation_occurred and _GIT_REVIEW_CMD_RE.search(cmd):
            return VERIFICATION
        # Explicit exploration commands stay exploration anywhere.
        # cat is only exploration when NOT redirecting output.
        if _EXPLORATION_CMD_PATTERNS.search(cmd):
            return EXPLORATION
        if _CAT_EXPLORATION_RE.search(cmd):
            return EXPLORATION
        # After implementation: non-exploration terminal commands are
        # more likely verification (running the result, checking output)
        # than exploration. Especially near the end of a trajectory.
        if implementation_occurred:
            if is_near_end:
                return VERIFICATION
            # Even mid-trajectory, unrecognized commands after impl
            # are more likely verification than exploration
            if _is_verification_command(cmd):
                return VERIFICATION
        # Near the end of a trajectory, even without explicit impl,
        # non-exploration commands likely serve a verification purpose
        if is_near_end and not _EXPLORATION_CMD_PATTERNS.search(cmd) and not _CAT_EXPLORATION_RE.search(cmd):
            return VERIFICATION
        return EXPLORATION

    # ── MCP / prefix-registered tools ──────────────────────────────
    desc = registry.get(tool)
    if desc.stage_hint and not desc.context_sensitive:
        return desc.stage_hint

    # ── Fallback: unknown or unmapped tools ───────────────────────
    if "error" in rs.lower() or "fail" in rs.lower():
        return VERIFICATION if implementation_occurred else ORCHESTRATION

    return ORCHESTRATION


# ──────────────────────────────────────────────────────────────────────
# Trace-level labeling
# ──────────────────────────────────────────────────────────────────────

def label_intent_stages(states: List["State"]) -> None:
    """Label a list of states **in-place**, in step order.

    Maintains the running context (``implementation_occurred`` and
    ``implemented_files``) across the sequence so that context-sensitive
    tools are assigned the correct intent stage.

    Parameters
    ----------
    states : list[State]
        Must be sorted by ``step``.
    """
    implementation_occurred = False
    implemented_files: Set[str] = set()
    total_states = len(states)

    for idx, state in enumerate(states):
        # Positional context: last 20% of trajectory
        is_near_end = (idx >= total_states * 0.8) if total_states > 0 else False

        stage = get_intent_stage(
            state,
            implementation_occurred=implementation_occurred,
            implemented_files=implemented_files,
            is_near_end=is_near_end,
        )
        state.intent_stage = stage

        # Update running context: track when implementation starts.
        # Count edits to non-test source files as "implementation".
        # Also count terminal commands that modify files (sed -i, echo >>)
        # as triggering the implementation flag.
        tool = getattr(state, "tool_used", None) or ""
        if tool in _EDIT_TOOLS and stage == IMPLEMENTATION:
            implementation_occurred = True
            fp = _get_file_path_from_state(state)
            if fp:
                for p in fp.split(","):
                    p = p.strip()
                    if p:
                        implemented_files.add(p)
        elif tool in _TERMINAL_TOOLS:
            cmd = _get_command_from_state(state)
            # Terminal code-writing commands count as implementation
            if _is_code_writing_command(cmd):
                implementation_occurred = True
                fp_match = re.search(
                    r"['\"]?([^'\"\s>]+\.(?:py|js|ts|java|cs|rb|go|rs|cpp|c|h))",
                    cmd,
                )
                if fp_match:
                    implemented_files.add(fp_match.group(1))
            elif not implementation_occurred and _is_modifying_command(cmd):
                implementation_occurred = True


def label_trace_intents(trace: "Trace") -> None:
    """Label all states in *trace* **in-place**.

    States are processed in step order.  The intent-stage label is
    written to each state's :attr:`~State.intent_stage` attribute.

    Parameters
    ----------
    trace : Trace
        The trace to label.  Modified in-place.
    """
    ordered = sorted(trace.states.values(), key=lambda s: s.step)
    label_intent_stages(ordered)
    logger.info(
        "Intent-stage labeled %d states: %s",
        len(ordered),
        _stage_summary(ordered),
    )


def _stage_summary(states: List["State"]) -> str:
    """Return a short summary string like ``E:5 I:3 V:2 O:1``."""
    counts: Dict[str, int] = {
        EXPLORATION: 0, IMPLEMENTATION: 0,
        VERIFICATION: 0, ORCHESTRATION: 0, "": 0,
    }
    for s in states:
        stage = getattr(s, "intent_stage", "")
        counts[stage] = counts.get(stage, 0) + 1
    parts = []
    for stage, abbrev in [
        (EXPLORATION, "E"), (IMPLEMENTATION, "I"),
        (VERIFICATION, "V"), (ORCHESTRATION, "O"),
    ]:
        if counts.get(stage, 0):
            parts.append(f"{abbrev}:{counts[stage]}")
    return " ".join(parts) if parts else "(none)"


# ──────────────────────────────────────────────────────────────────────
# Workflow fingerprinting
# ──────────────────────────────────────────────────────────────────────

# Abbreviation map for compact fingerprint strings
_STAGE_ABBREV: Dict[str, str] = {
    EXPLORATION: "E",
    IMPLEMENTATION: "I",
    VERIFICATION: "V",
    ORCHESTRATION: "O",
}

# Transition type constants
TRANSITION_DEEPEN = "deepen"
TRANSITION_PIVOT = "pivot"
TRANSITION_BACKTRACK = "backtrack"
TRANSITION_CONFIRM = "confirm"


def _classify_transition(from_stage: str, to_stage: str) -> str:
    """Classify the transition between two intent stages.

    Returns one of: ``"deepen"``, ``"pivot"``, ``"backtrack"``,
    ``"confirm"``.
    """
    if to_stage == ORCHESTRATION and from_stage != ORCHESTRATION:
        return TRANSITION_CONFIRM
    from_ord = _STAGE_ORDER.get(from_stage, -1)
    to_ord = _STAGE_ORDER.get(to_stage, -1)
    if from_ord == to_ord:
        return TRANSITION_DEEPEN
    elif to_ord > from_ord:
        return TRANSITION_PIVOT
    else:
        return TRANSITION_BACKTRACK


def compute_fingerprint(states: List["State"]) -> Tuple[str, List[str]]:
    """Compute a workflow fingerprint from labeled states.

    Parameters
    ----------
    states : list[State]
        States sorted by step, each having ``intent_stage`` set.

    Returns
    -------
    tuple[str, list[str]]
        ``(fingerprint_string, transition_types)``

        The fingerprint string is a compact representation like
        ``"E→E→I→V→O"`` showing the sequence of intent stages.

        The transition_types list contains the classification of each
        consecutive transition (``"deepen"``, ``"pivot"``,
        ``"backtrack"``, ``"confirm"``).
    """
    labeled = [s for s in states if getattr(s, "intent_stage", "")]
    if not labeled:
        return "", []

    # Build stage sequence (abbreviations)
    abbrevs = [_STAGE_ABBREV.get(s.intent_stage, "?") for s in labeled]
    fingerprint = "→".join(abbrevs)

    # Classify transitions between consecutive stages
    transitions: List[str] = []
    for i in range(1, len(labeled)):
        t = _classify_transition(labeled[i - 1].intent_stage, labeled[i].intent_stage)
        transitions.append(t)

    return fingerprint, transitions


def fingerprint_summary(transitions: List[str]) -> Dict[str, int]:
    """Count each transition type in a transition list.

    Returns a dict like ``{"deepen": 5, "pivot": 3, "backtrack": 1, "confirm": 2}``.
    """
    counts = {
        TRANSITION_DEEPEN: 0,
        TRANSITION_PIVOT: 0,
        TRANSITION_BACKTRACK: 0,
        TRANSITION_CONFIRM: 0,
    }
    for t in transitions:
        counts[t] = counts.get(t, 0) + 1
    return counts


def workflow_similarity(fp_a: str, fp_b: str) -> float:
    """Compute similarity between two workflow fingerprints.

    Uses longest-common-subsequence (LCS) ratio to compare the stage
    sequences encoded in two fingerprint strings.

    Parameters
    ----------
    fp_a, fp_b : str
        Fingerprint strings (e.g. ``"E→E→I→V→O"``).

    Returns
    -------
    float
        Similarity ratio in [0.0, 1.0].  1.0 = identical stage
        sequence, 0.0 = completely different.
    """
    # Extract just the stage letters from the fingerprint
    seq_a = [c for c in fp_a if c in ("E", "I", "V", "O")]
    seq_b = [c for c in fp_b if c in ("E", "I", "V", "O")]

    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0

    # LCS via dynamic programming
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    # Ratio: LCS length / max sequence length
    return lcs_len / max(m, n)


# ──────────────────────────────────────────────────────────────────────
# Trajectory coherence score
# ──────────────────────────────────────────────────────────────────────

def _count_blind_retries(states: List["State"]) -> int:
    """Count consecutive state triples with identical (tool, file, stage).

    A "blind retry" is when three or more consecutive states share the
    same tool, file path, and intent stage — the agent is repeating the
    exact same type of action without changing approach.

    Returns the total number of *excess* repetitions (i.e. the count
    beyond the second consecutive identical state in each run).
    """
    if len(states) < 3:
        return 0

    retries = 0
    run_length = 1
    for i in range(1, len(states)):
        prev = states[i - 1]
        curr = states[i]
        prev_sig = (
            getattr(prev, "tool_used", "") or "",
            getattr(prev, "file_path", "") or "",
            getattr(prev, "intent_stage", "") or "",
        )
        curr_sig = (
            getattr(curr, "tool_used", "") or "",
            getattr(curr, "file_path", "") or "",
            getattr(curr, "intent_stage", "") or "",
        )
        if curr_sig == prev_sig and curr_sig != ("", "", ""):
            run_length += 1
            if run_length >= 3:
                retries += 1
        else:
            run_length = 1
    return retries


def compute_coherence_score(states: List["State"]) -> float:
    """Compute a trajectory coherence score from labeled states.

    Measures how *progressive* a trajectory is: does the agent move
    forward through intent stages (pivots) and reach completion
    (confirms), or does it regress (backtracks) and repeat actions
    blindly (retries)?

    The score rewards forward-progress transitions and penalises
    repetitive behaviour:

    .. math::

        \\text{coherence} = \\frac{\\text{pivots} + \\text{confirms}}
            {\\text{pivots} + \\text{confirms} + \\text{backtracks} + \\epsilon}
            \\times (1 - \\text{retry\\_ratio})

    Parameters
    ----------
    states : list[State]
        States sorted by step, each having ``intent_stage`` set.

    Returns
    -------
    float
        Coherence score in [0.0, 1.0].  Higher is better — the agent
        made clean forward progress with minimal regression and blind
        retries.
    """
    _, transitions = compute_fingerprint(states)
    if not transitions:
        # Zero or one labeled state — no transitions to evaluate.
        # Absence of transitions is not evidence of incoherence.
        return 1.0

    summary = fingerprint_summary(transitions)
    pivots = summary.get(TRANSITION_PIVOT, 0)
    confirms = summary.get(TRANSITION_CONFIRM, 0)
    backtracks = summary.get(TRANSITION_BACKTRACK, 0)

    forward = pivots + confirms
    total_directed = forward + backtracks

    # When all transitions are "deepen" (staying in the same stage),
    # there are no pivots, confirms, OR backtracks.  This is perfectly
    # coherent — the agent stayed focused without regressing.  Only
    # backtracks and blind retries should reduce the score.
    if total_directed == 0:
        progress_ratio = 1.0
    else:
        progress_ratio = forward / total_directed

    # Penalise blind retries — consecutive identical (tool, file, stage) triples
    retries = _count_blind_retries(states)
    total_transitions = len(transitions)
    retry_ratio = retries / total_transitions if total_transitions > 0 else 0.0

    return progress_ratio * (1.0 - min(retry_ratio, 1.0))


# ──────────────────────────────────────────────────────────────────────
# Temporal stage profile divergence
# ──────────────────────────────────────────────────────────────────────

_PROFILE_STAGES = ("E", "I", "V", "O")


def _stage_distribution(stages: List[str]) -> Dict[str, float]:
    """Compute a normalised distribution over intent-stage abbreviations.

    Returns a dict mapping each of E, I, V, O to its frequency (0–1).
    Uses Laplace smoothing (alpha=0.01) to avoid zero probabilities.
    """
    alpha = 0.01
    counts = {s: alpha for s in _PROFILE_STAGES}
    for s in stages:
        if s in counts:
            counts[s] += 1.0
    total = sum(counts.values())
    return {s: counts[s] / total for s in _PROFILE_STAGES}


def _jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence between two stage distributions.

    Returns a value in [0, 1] (using log base 2).
    """
    import math
    keys = _PROFILE_STAGES
    # Midpoint distribution
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def _kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        val = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            bk = b.get(k, 0.0)
            if ak > 0 and bk > 0:
                val += ak * math.log2(ak / bk)
        return val

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compute_temporal_profile_divergence(
    candidate_states: List["State"],
    gt_states: List["State"],
    *,
    n_segments: int = 3,
) -> float:
    """Compare the temporal intent-stage profiles of two traces.

    Divides each trace into *n_segments* equal temporal segments and
    computes the intent-stage distribution in each.  The score is based
    on the average Jensen-Shannon divergence (JSD) between corresponding
    segments of the candidate and ground-truth:

    .. math::

        \\text{profile\\_score} = 1 - \\frac{1}{K}
            \\sum_{k=1}^{K} \\text{JSD}(P_{\\text{cand}}^{(k)}
        \\| P_{\\text{GT}}^{(k)})

    Successful trajectories follow a canonical progression — exploration
    front-loaded, implementation in the middle, verification at the end.
    This score detects deviations from that temporal structure, even when
    set-level coverage is identical.

    Parameters
    ----------
    candidate_states : list[State]
        Candidate trace states sorted by step with ``intent_stage`` set.
    gt_states : list[State]
        Ground-truth (merged) trace states sorted by step with
        ``intent_stage`` set.
    n_segments : int
        Number of temporal segments to divide each trace into (default 3
        gives early / middle / late).

    Returns
    -------
    float
        Profile similarity score in [0.0, 1.0].  1.0 means the
        candidate's temporal phase distribution perfectly matches the
        ground truth.
    """
    abbrev = _STAGE_ABBREV

    def _segment_stages(states: List["State"]) -> List[List[str]]:
        labeled = [abbrev.get(getattr(s, "intent_stage", ""), "")
                   for s in states if getattr(s, "intent_stage", "")]
        if not labeled:
            return [[] for _ in range(n_segments)]
        seg_size = max(1, len(labeled) // n_segments)
        segments: List[List[str]] = []
        for k in range(n_segments):
            start = k * seg_size
            end = start + seg_size if k < n_segments - 1 else len(labeled)
            segments.append(labeled[start:end])
        return segments

    cand_segments = _segment_stages(candidate_states)
    gt_segments = _segment_stages(gt_states)

    if not any(cand_segments) or not any(gt_segments):
        return 0.0

    total_jsd = 0.0
    for cand_seg, gt_seg in zip(cand_segments, gt_segments):
        p = _stage_distribution(cand_seg)
        q = _stage_distribution(gt_seg)
        total_jsd += _jsd(p, q)

    avg_jsd = total_jsd / n_segments
    return max(0.0, 1.0 - avg_jsd)
