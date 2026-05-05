"""Pluggable state-equivalence strategies.

This module provides a multi-tier approach for deciding whether two
:class:`~swe_trace_sdk.models.State` objects are semantically equivalent:

1. **Exact match** — same tool + same resulting-state hash.
2. **Heuristic match** — tool-specific rules (same file path, normalised
   command, similar search query, …).
3. **LLM semantic match** — optional LLM-backed comparison for ambiguous
   cases.

The main entry point is :class:`StateEquivalence` which combines all tiers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .models import State

from .tool_registry import registry, CATEGORY_EDIT, CATEGORY_READ, CATEGORY_SEARCH

logger = logging.getLogger(__name__)

__all__ = [
    "EquivalenceResult",
    "StateEquivalence",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EquivalenceResult:
    """Outcome of a single equivalence check."""

    equivalent: bool
    confidence: float  # 0.0 – 1.0
    reasoning: str
    method: str  # "exact_match" | "resulting_state" | "heuristic" | "llm" | …
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "method": self.method,
            "metadata": self.metadata or {},
        }


# ---------------------------------------------------------------------------
# Main equivalence checker
# ---------------------------------------------------------------------------

class StateEquivalence:
    """Multi-tier state-equivalence checker.

    Parameters
    ----------
    use_llm : bool
        Enable the LLM tier for ambiguous comparisons.
    llm_fn : callable, optional
        A function ``(state_a, state_b) -> EquivalenceResult | None``
        used for LLM-backed semantic comparison.  If *None* and
        *use_llm* is *True*, an internal default using the ``llm``
        sub-module will be attempted.
    cache : bool
        Cache results keyed on observation pair hashes.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_fn: Optional[Callable[["State", "State"], Optional[EquivalenceResult]]] = None,
        cache: bool = True,
        use_intent_stage: bool = True,
    ) -> None:
        self.use_llm = use_llm
        self.use_intent_stage = use_intent_stage
        self._llm_fn = llm_fn
        self._cache_enabled = cache
        self._cache: Dict[str, EquivalenceResult] = {}
        self.stats: Dict[str, int] = {
            "total_checks": 0,
            "cache_hits": 0,
            "exact_matches": 0,
            "heuristic_matches": 0,
            "llm_calls": 0,
            "stage_rejections": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        state_a: "State",
        state_b: "State",
        *,
        position: Optional[int] = None,
    ) -> EquivalenceResult:
        """Check whether *state_a* and *state_b* are semantically equivalent.

        Parameters
        ----------
        state_a, state_b : State
            The two states to compare.
        position : int, optional
            Step position in the trace (position 0 ⇒ initial states are
            always equivalent).
        """
        self.stats["total_checks"] += 1

        # Initial states always match
        if position == 0:
            return EquivalenceResult(True, 1.0, "Initial states are always merged", "position_zero")

        # Intent-stage-aware rejection: states with different cognitive
        # intents are NOT equivalent, even if tool and file match.
        # E.g. read_file during exploration ≠ read_file during verification.
        if self.use_intent_stage:
            stage_a = getattr(state_a, "intent_stage", "") or ""
            stage_b = getattr(state_b, "intent_stage", "") or ""
            # Only reject when BOTH states have a non-empty stage and they differ.
            # Empty stage = not labeled (legacy data) → skip this check.
            if stage_a and stage_b and stage_a != stage_b:
                self.stats["stage_rejections"] += 1
                result = EquivalenceResult(
                    False, 0.95,
                    f"Different intent stages: {stage_a} vs {stage_b}",
                    "stage_mismatch",
                )
                if self._cache_enabled:
                    cache_key = self._cache_key(state_a, state_b)
                    self._cache[cache_key] = result
                return result

        # Cache lookup
        cache_key = self._cache_key(state_a, state_b)
        if self._cache_enabled and cache_key in self._cache:
            self.stats["cache_hits"] += 1
            cached = self._cache[cache_key]
            return EquivalenceResult(
                cached.equivalent, cached.confidence,
                f"[cached] {cached.reasoning}", "cached", cached.metadata,
            )

        # Tier 1: exact
        result = self._exact(state_a, state_b)
        if result is not None:
            self.stats["exact_matches"] += 1
            self._cache[cache_key] = result
            return result

        tool_a = getattr(state_a, "tool_used", None)
        desc_a = registry.get(tool_a or "")
        _is_file_op = desc_a.category in (CATEGORY_EDIT, CATEGORY_READ)
        _is_edit_op = desc_a.category == CATEGORY_EDIT

        # For file operations: try scope/line-overlap BEFORE generic heuristic
        if _is_file_op:
            if _is_edit_op:
                scope_result = self._check_scope_match(state_a, state_b)
                if scope_result is not None:
                    self._cache[cache_key] = scope_result
                    return scope_result
            line_result = self._check_line_overlap(state_a, state_b)
            if line_result is not None:
                self._cache[cache_key] = line_result
                return line_result

        # For non-file operations: check resulting_state
        if not _is_file_op:
            rs_a = getattr(state_a, "resulting_state", "") or ""
            rs_b = getattr(state_b, "resulting_state", "") or ""
            if rs_a and rs_b and rs_a == rs_b:
                # If both states have content_hash and they differ, don't match
                # on the coarse resulting_state alone — defer to more precise
                # checks (terminal similarity, heuristic command comparison).
                ch_a = getattr(state_a, "content_hash", "") or ""
                ch_b = getattr(state_b, "content_hash", "") or ""
                if ch_a and ch_b and ch_a != ch_b:
                    pass  # Defer to terminal similarity / heuristic
                else:
                    res = EquivalenceResult(True, 0.95, f"Same resulting state: {rs_a}", "resulting_state")
                    self._cache[cache_key] = res
                    return res

        # Terminal command similarity
        if desc_a.comparison_strategy == "command":
            term_result = self._check_terminal_similarity(state_a, state_b)
            if term_result is not None:
                self._cache[cache_key] = term_result
                return term_result

        # Semantic content comparison (LLM-backed, before heuristic)
        if self.use_llm:
            semantic = self._check_semantic_content(state_a, state_b)
            if semantic is not None:
                self._cache[cache_key] = semantic
                return semantic

        # Tier 2: heuristic
        heuristic = self._heuristic(state_a, state_b)
        if heuristic.confidence >= 0.8:
            self.stats["heuristic_matches"] += 1
            self._cache[cache_key] = heuristic
            return heuristic

        # Tier 3: LLM
        if self.use_llm:
            llm_result = self._llm(state_a, state_b)
            if llm_result is not None:
                self.stats["llm_calls"] += 1
                self._cache[cache_key] = llm_result
                return llm_result

        # Fall back to heuristic
        self._cache[cache_key] = heuristic
        return heuristic

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(a: "State", b: "State") -> str:
        def _key_parts(s: "State") -> str:
            parts = [
                getattr(s, "observation", "") or "",
                getattr(s, "tool_used", "") or "",
                getattr(s, "resulting_state", "") or "",
                getattr(s, "file_path", "") or "",
                getattr(s, "content_hash", "") or "",
                getattr(s, "scope_path", "") or "",
                getattr(s, "intent_stage", "") or "",
            ]
            lr = getattr(s, "line_range", None)
            if lr:
                parts.append(f"{lr[0]}-{lr[1]}")
            combined = "|".join(parts)
            return combined if combined.strip("|") else str(s.to_dict())
        return hashlib.md5(f"{_key_parts(a)}|||{_key_parts(b)}".encode()).hexdigest()

    # ------------------------------------------------------------------
    # Tier 1 — exact
    # ------------------------------------------------------------------

    @staticmethod
    def _exact(a: "State", b: "State") -> Optional[EquivalenceResult]:
        tool_a = getattr(a, "tool_used", None)
        tool_b = getattr(b, "tool_used", None)

        if tool_a != tool_b:
            # When both tools are in the same equivalence group, defer
            # to heuristic for cross-tool matching (e.g. apply_patch
            # vs replace_string_in_file).
            if registry.is_in_equivalence_group(tool_a or "", tool_b or ""):
                return None  # let heuristic decide
            return EquivalenceResult(False, 1.0, f"Different tools: {tool_a} vs {tool_b}", "exact_match")

        # Content hash match
        ch_a = getattr(a, "content_hash", "") or ""
        ch_b = getattr(b, "content_hash", "") or ""
        if ch_a and ch_b:
            if ch_a == ch_b:
                return EquivalenceResult(True, 1.0, f"Same content hash: {ch_a[:8]}...", "content_hash")
            # Different hashes — defer to semantic comparison
            # (don't return here so scope/heuristic/LLM can handle it)

        # Resulting state — only decide if no content hash info to contradict
        rs_a = getattr(a, "resulting_state", "") or ""
        rs_b = getattr(b, "resulting_state", "") or ""
        if rs_a and rs_b and rs_a == rs_b:
            if not ch_a or not ch_b:
                return EquivalenceResult(True, 1.0, f"Same resulting state: {rs_a}", "resulting_state")
            # Same resulting_state but different content_hash → defer

        hash_a = a.get_observation_hash() if hasattr(a, "get_observation_hash") else None
        hash_b = b.get_observation_hash() if hasattr(b, "get_observation_hash") else None
        if hash_a and hash_b and hash_a == hash_b:
            return EquivalenceResult(True, 1.0, "Identical observations", "exact_match")

        return None

    # ------------------------------------------------------------------
    # Tier 2 — heuristic
    # ------------------------------------------------------------------

    def _heuristic(self, a: "State", b: "State") -> EquivalenceResult:
        tool_a = getattr(a, "tool_used", None)
        tool_b = getattr(b, "tool_used", None)

        if tool_a != tool_b:
            # Cross-tool equivalence: tools in the same equivalence group
            # (e.g. apply_patch and replace_string_in_file are both in
            # "file_edit") that operate on overlapping files.
            if registry.is_in_equivalence_group(tool_a or "", tool_b or ""):
                files_a = _extract_modified_files(a)
                files_b = _extract_modified_files(b)
                if files_a and files_b and files_a & files_b:
                    overlap = files_a & files_b
                    # Content-level check: if both states have content_hash
                    # and they differ, the edits are semantically different
                    # even though they touch the same files.
                    ch_a = getattr(a, "content_hash", "") or ""
                    ch_b = getattr(b, "content_hash", "") or ""
                    if ch_a and ch_b and ch_a != ch_b:
                        # Scope check: compare function/class names extracted
                        # by tree-sitter to decide if they edited the same code
                        scope_a = getattr(a, "scope_path", "") or ""
                        scope_b = getattr(b, "scope_path", "") or ""
                        if scope_a and scope_b:
                            # Parse comma-separated scope lists and check overlap
                            set_a = {s.strip() for s in scope_a.split(",")}
                            set_b = {s.strip() for s in scope_b.split(",")}
                            common = set_a & set_b
                            if common:
                                return EquivalenceResult(
                                    True, 0.85,
                                    f"Cross-tool edit: {tool_a} & {tool_b} "
                                    f"on {overlap}, common scope {common}",
                                    "heuristic",
                                )
                            return EquivalenceResult(
                                False, 0.8,
                                f"Cross-tool edit: {tool_a} & {tool_b} "
                                f"on {overlap} but different scopes: {set_a} vs {set_b}",
                                "heuristic",
                            )
                        # No scope info — fall through to description check
                        desc_a = _extract_scope_from_description(a)
                        desc_b = _extract_scope_from_description(b)
                        if desc_a and desc_b:
                            common = desc_a & desc_b
                            if common:
                                return EquivalenceResult(
                                    True, 0.80,
                                    f"Cross-tool edit: {tool_a} & {tool_b} "
                                    f"on {overlap}, common symbols {common}",
                                    "heuristic",
                                )
                            return EquivalenceResult(
                                False, 0.75,
                                f"Cross-tool edit: {tool_a} & {tool_b} "
                                f"on {overlap} but different symbols: {desc_a} vs {desc_b}",
                                "heuristic",
                            )
                        # No scope or description info but different hashes —
                        # conservatively treat as equivalent (file overlap is
                        # the strongest signal available)
                    return EquivalenceResult(
                        True, 0.85,
                        f"Cross-tool edit equivalence: {tool_a} & {tool_b} "
                        f"on overlapping files {overlap}",
                        "heuristic",
                    )
            return EquivalenceResult(False, 0.95, f"Different tools: {tool_a} vs {tool_b}", "heuristic")

        entry_a = getattr(a, "log_entry", None)
        entry_b = getattr(b, "log_entry", None)

        if entry_a is None or entry_b is None:
            # Handle initial states
            is_init_a = hasattr(a, "metadata") and a.metadata.get("is_initial")
            is_init_b = hasattr(b, "metadata") and b.metadata.get("is_initial")
            if is_init_a and is_init_b:
                return EquivalenceResult(True, 1.0, "Both are initial states", "heuristic")
            if is_init_a or is_init_b:
                return EquivalenceResult(False, 0.95, "One is initial state, other is not", "heuristic")
            # Same tool, try file_path comparison
            if tool_a is not None and tool_a == tool_b:
                fp_a = getattr(a, "file_path", "") or ""
                fp_b = getattr(b, "file_path", "") or ""
                if fp_a and fp_b:
                    if fp_a == fp_b:
                        return EquivalenceResult(True, 0.85, f"Same tool '{tool_a}' on same file '{fp_a}'", "heuristic")
                    return EquivalenceResult(False, 0.85, f"Same tool '{tool_a}' on different files", "heuristic")
                return EquivalenceResult(True, 0.75, f"Same tool '{tool_a}' (missing detail)", "heuristic")
            return EquivalenceResult(tool_a == tool_b, 0.5, "Missing log entries for detailed comparison", "heuristic")

        if entry_a.kind == "toolCall" and entry_b.kind == "toolCall":
            return self._compare_tool_calls(entry_a, entry_b)

        if entry_a.kind == "request" and entry_b.kind == "request":
            return self._compare_requests(entry_a, entry_b)

        return EquivalenceResult(False, 0.8, f"Different entry kinds: {entry_a.kind} vs {entry_b.kind}", "heuristic")

    # ------------------------------------------------------------------

    def _compare_tool_calls(self, a, b) -> EquivalenceResult:
        tool = a.tool
        args_a = a.args or {}
        args_b = b.args or {}
        desc = registry.get(tool or "")

        if desc.comparison_strategy == "file_path":
            pa = _normalize_path(args_a.get("filePath") or args_a.get("path", ""))
            pb = _normalize_path(args_b.get("filePath") or args_b.get("path", ""))
            if pa != pb:
                return EquivalenceResult(False, 0.9, f"Different files: {pa} vs {pb}", "heuristic")
            # Same file — for read_file, require line-range overlap when available
            if tool == "read_file":
                sl_a = args_a.get("startLine") or args_a.get("start_line")
                el_a = args_a.get("endLine") or args_a.get("end_line")
                sl_b = args_b.get("startLine") or args_b.get("start_line")
                el_b = args_b.get("endLine") or args_b.get("end_line")
                if sl_a is not None and el_a is not None and sl_b is not None and el_b is not None:
                    try:
                        sl_a, el_a, sl_b, el_b = int(sl_a), int(el_a), int(sl_b), int(el_b)
                        overlap_start = max(sl_a, sl_b)
                        overlap_end = min(el_a, el_b)
                        if overlap_end >= overlap_start:
                            overlap_len = overlap_end - overlap_start + 1
                            union_len = max(el_a, el_b) - min(sl_a, sl_b) + 1
                            ratio = overlap_len / union_len if union_len > 0 else 0
                            if ratio >= 0.3:
                                return EquivalenceResult(True, 0.85 + ratio * 0.1, f"Same file ({pa}) with line overlap {ratio:.0%}", "heuristic")
                            return EquivalenceResult(False, 0.85, f"Same file ({pa}) but low line overlap {ratio:.0%}: [{sl_a}-{el_a}] vs [{sl_b}-{el_b}]", "heuristic")
                        return EquivalenceResult(False, 0.85, f"Same file ({pa}) but no line overlap: [{sl_a}-{el_a}] vs [{sl_b}-{el_b}]", "heuristic")
                    except (ValueError, TypeError):
                        pass
            return EquivalenceResult(True, 0.9, f"Same file operation on {pa}", "heuristic")

        if desc.comparison_strategy == "query":
            qa = args_a.get("query", "")
            qb = args_b.get("query", "")
            if qa == qb:
                return EquivalenceResult(True, 0.9, "Same search query", "heuristic")
            sim = _word_similarity(qa, qb)
            if sim > 0.8:
                return EquivalenceResult(True, sim * 0.9, f"Similar search queries (similarity={sim:.2f})", "heuristic")
            return EquivalenceResult(False, 0.7, "Different search queries", "heuristic")

        if desc.comparison_strategy == "command":
            ca = _normalize_command(args_a.get("command", ""))
            cb = _normalize_command(args_b.get("command", ""))
            if ca == cb:
                return EquivalenceResult(True, 0.9, "Same terminal command", "heuristic")
            return EquivalenceResult(False, 0.8, "Different terminal commands", "heuristic")

        if tool == "list_dir":
            pa = _normalize_path(args_a.get("path", "") or args_a.get("filePath", ""))
            pb = _normalize_path(args_b.get("path", "") or args_b.get("filePath", ""))
            same = pa == pb
            return EquivalenceResult(same, 0.9, f"Directory listing: {'same' if same else 'different'} path", "heuristic")


        if args_a == args_b:
            return EquivalenceResult(True, 0.85, f"Identical arguments for {tool}", "heuristic")

        return EquivalenceResult(False, 0.6, f"Different arguments for {tool}, needs semantic comparison", "heuristic")

    @staticmethod
    def _compare_requests(a, b) -> EquivalenceResult:
        return EquivalenceResult(True, 0.85, "LLM request at same semantic position (model-agnostic)", "heuristic")

    # ------------------------------------------------------------------
    # Scope / line-overlap / terminal / semantic tiers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_scope_match(a: "State", b: "State") -> Optional[EquivalenceResult]:
        """Check equivalence based on function/class scope for edit operations."""
        tool_a = getattr(a, "tool_used", None)
        tool_b = getattr(b, "tool_used", None)
        if tool_a != tool_b:
            return None
        if registry.get(tool_a or "").category != CATEGORY_EDIT:
            return None
        fp_a = getattr(a, "file_path", "") or ""
        fp_b = getattr(b, "file_path", "") or ""
        if not fp_a or not fp_b or fp_a != fp_b:
            return None

        scope_a = getattr(a, "scope_path", "") or ""
        scope_b = getattr(b, "scope_path", "") or ""
        func_a = getattr(a, "function_name", "") or ""
        func_b = getattr(b, "function_name", "") or ""
        class_a = getattr(a, "class_name", "") or ""
        class_b = getattr(b, "class_name", "") or ""

        if scope_a and scope_b:
            if scope_a == scope_b:
                return EquivalenceResult(True, 0.9, f"Same file ({fp_a}) and scope: {scope_a}", "scope_match")
            return EquivalenceResult(False, 0.85, f"Same file ({fp_a}) but different scopes: {scope_a} vs {scope_b}", "scope_match")
        if func_a and func_b:
            if func_a == func_b:
                return EquivalenceResult(True, 0.85, f"Same file ({fp_a}) and function: {func_a}", "scope_match")
            return EquivalenceResult(False, 0.8, f"Same file ({fp_a}) but different functions: {func_a} vs {func_b}", "scope_match")
        if class_a and class_b:
            if class_a == class_b:
                return EquivalenceResult(True, 0.8, f"Same file ({fp_a}) and class: {class_a}", "scope_match")
            return EquivalenceResult(False, 0.75, f"Same file ({fp_a}) but different classes: {class_a} vs {class_b}", "scope_match")

        # Try extracting scope from content_description
        extracted_a = _extract_scope_from_description(a)
        extracted_b = _extract_scope_from_description(b)
        if extracted_a and extracted_b:
            common = extracted_a & extracted_b
            if common:
                return EquivalenceResult(True, 0.75, f"Same file ({fp_a}) with common symbols: {common}", "scope_match_extracted")
            return EquivalenceResult(False, 0.7, f"Same file ({fp_a}) but different symbols: {extracted_a} vs {extracted_b}", "scope_match_extracted")
        return None

    @staticmethod
    def _check_line_overlap(a: "State", b: "State") -> Optional[EquivalenceResult]:
        """Check equivalence based on line-range overlap in the same file."""
        tool_a = getattr(a, "tool_used", None)
        tool_b = getattr(b, "tool_used", None)
        if tool_a != tool_b:
            return None
        desc_a = registry.get(tool_a or "")
        if desc_a.category not in (CATEGORY_EDIT, CATEGORY_READ):
            return None
        fp_a = getattr(a, "file_path", "") or ""
        fp_b = getattr(b, "file_path", "") or ""
        if not fp_a or not fp_b:
            return None
        if fp_a != fp_b:
            return EquivalenceResult(False, 0.9, f"Different files: {fp_a} vs {fp_b}", "line_overlap")
        lr_a = getattr(a, "line_range", None)
        lr_b = getattr(b, "line_range", None)
        if not lr_a or not lr_b:
            # For read operations on the same file without line-range info,
            # treat as equivalent — both are reading the same file.
            if desc_a.category == CATEGORY_READ:
                return EquivalenceResult(
                    True, 0.80,
                    f"Same file read ({fp_a}), no line ranges to compare",
                    "line_overlap",
                )
            return None
        overlap = _compute_line_overlap(lr_a, lr_b)
        if overlap >= 0.3:
            return EquivalenceResult(True, 0.80 + overlap * 0.15, f"Same file ({fp_a}) with {overlap:.0%} line overlap: {lr_a} vs {lr_b}", "line_overlap")
        if overlap > 0:
            return None
        # Non-overlapping reads/edits of the same file → not equivalent.
        return EquivalenceResult(False, 0.8, f"Same file ({fp_a}) but no line overlap: {lr_a} vs {lr_b}", "line_overlap")

    @staticmethod
    def _check_terminal_similarity(a: "State", b: "State") -> Optional[EquivalenceResult]:
        """Check if two terminal commands are semantically equivalent.

        Uses semantic command grouping to match commands that serve the
        same purpose even when they use different tools:

        - **search group**: grep, rg, ag, ack, find + grep → compare search target
        - **view group**: cat, sed -n, head, tail, less → compare viewed file
        - **execute group**: python/node/bash + script → compare script name
        """
        entry_a = getattr(a, "log_entry", None)
        entry_b = getattr(b, "log_entry", None)
        if not entry_a or not entry_b:
            return None
        args_a = getattr(entry_a, "args", {}) or {}
        args_b = getattr(entry_b, "args", {}) or {}
        cmd_a = args_a.get("command", "").strip()
        cmd_b = args_b.get("command", "").strip()
        if not cmd_a or not cmd_b:
            return None

        def _norm_cmd(cmd: str) -> str:
            cmd = re.sub(r"^cd\s+[^\s&;]+\s*[&;]+\s*", "", cmd)
            cmd = re.sub(r"\bpython3?\b", "python", cmd)
            cmd = re.sub(r"\bnpm i\b", "npm install", cmd)
            return " ".join(cmd.split()).lower()

        na, nb = _norm_cmd(cmd_a), _norm_cmd(cmd_b)
        if na == nb:
            return EquivalenceResult(True, 0.9, "Equivalent terminal commands", "terminal_similarity")

        # ── Semantic command grouping ─────────────────────────────
        group_a = _classify_terminal_command(na)
        group_b = _classify_terminal_command(nb)

        if group_a and group_b and group_a[0] == group_b[0]:
            grp = group_a[0]
            target_a, target_b = group_a[1], group_b[1]
            if target_a and target_b:
                if target_a == target_b:
                    return EquivalenceResult(
                        True, 0.85,
                        f"Same {grp} target: {target_a}",
                        "terminal_similarity",
                    )
                # For search commands, check if they search for similar terms
                if grp == "search":
                    sim = _word_similarity(target_a, target_b)
                    if sim >= 0.5:
                        return EquivalenceResult(
                            True, 0.7 + sim * 0.15,
                            f"Similar {grp} targets ({sim:.0%}): {target_a} vs {target_b}",
                            "terminal_similarity",
                        )
                # For view commands, file path is the target — already compared
            else:
                # Same group, but can't extract targets — treat as equivalent
                return EquivalenceResult(
                    True, 0.7,
                    f"Same command group '{grp}'",
                    "terminal_similarity",
                )

        # Fall back to base-command comparison
        base_a = na.split()[0] if na else ""
        base_b = nb.split()[0] if nb else ""
        if base_a and base_a == base_b:
            return EquivalenceResult(True, 0.75, f"Same base command '{base_a}' with different args", "terminal_similarity")
        return None

    def _check_semantic_content(self, a: "State", b: "State") -> Optional[EquivalenceResult]:
        """Compare content_description fields for semantic equivalence (LLM-backed)."""
        desc_a = getattr(a, "content_description", "") or ""
        desc_b = getattr(b, "content_description", "") or ""
        if not desc_a or not desc_b:
            return None
        tool_a = getattr(a, "tool_used", None)
        tool_b = getattr(b, "tool_used", None)
        if tool_a != tool_b:
            return None
        ch_a = getattr(a, "content_hash", "") or ""
        ch_b = getattr(b, "content_hash", "") or ""
        if ch_a and ch_b and ch_a == ch_b:
            return None  # exact match would have handled it
        return self._compare_content_descriptions_llm(a, b, desc_a, desc_b)

    def _compare_content_descriptions_llm(
        self, a: "State", b: "State", desc_a: str, desc_b: str,
    ) -> Optional[EquivalenceResult]:
        """Use LLM to compare content descriptions."""
        tool = getattr(a, "tool_used", "unknown")
        rs_a = getattr(a, "resulting_state", "") or ""
        rs_b = getattr(b, "resulting_state", "") or ""
        prompt_lines = [
            f"Tool used: {tool}", "",
            "=== CHANGE 1 ===", f"Description: {desc_a}",
        ]
        if rs_a:
            prompt_lines.append(f"Result: {rs_a}")
        prompt_lines += ["", "=== CHANGE 2 ===", f"Description: {desc_b}"]
        if rs_b:
            prompt_lines.append(f"Result: {rs_b}")
        prompt_lines += [
            "",
            "Are these changes semantically equivalent (accomplishing the same goal)?",
            'Respond with JSON: {"equivalent": true/false, "confidence": 0.0-1.0, "reasoning": "..."}',
        ]
        prompt = "\n".join(prompt_lines)

        try:
            from .llm import llm_semantic_content_check
            result = llm_semantic_content_check(prompt)
            if result is not None:
                return EquivalenceResult(
                    equivalent=result["equivalent"],
                    confidence=result.get("confidence", 0.8),
                    reasoning=f"[semantic] {result['reasoning']}",
                    method="semantic_content",
                    metadata={"desc_a": desc_a[:100], "desc_b": desc_b[:100]},
                )
        except Exception as exc:
            logger.debug("Semantic content comparison unavailable: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Tier 3 — LLM
    # ------------------------------------------------------------------

    def _llm(self, a: "State", b: "State") -> Optional[EquivalenceResult]:
        if self._llm_fn is not None:
            return self._llm_fn(a, b)
        # Try the built-in LLM helper
        try:
            from .llm import llm_equivalence_check
            return llm_equivalence_check(a, b)
        except Exception as exc:
            logger.debug("LLM equivalence unavailable: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Cross-tool equivalence for edit operations
# ---------------------------------------------------------------------------

# Derive from registry: tools in the "file_edit" equivalence group.
_EDIT_TOOLS = registry.by_equivalence_group("file_edit")
"""Tools that modify source files.  Despite different names they are
functionally equivalent when they operate on the same file(s)."""


# Resulting-state prefixes that encode modified file paths.
# These must match what _generator.py emits.
_FILE_STATE_PREFIXES = (
    "file_modified:",
    "file_patched:",
    "file_created:",
    "files_modified:",
    "file_modify_failed:",
    "file_create_failed:",
)


def _extract_modified_files(state: "State") -> Set[str]:
    """Return the set of normalised file paths a state modifies."""
    files: Set[str] = set()

    # Single-file path attribute
    fp = getattr(state, "file_path", "") or ""
    if fp:
        files.add(_normalize_path(fp))

    # Resulting state may encode files:
    #   file_modified:utils/math_ops.py
    #   files_modified:a.py,b.py,c.py
    #   file_patched:utils/math_ops.py
    #   file_created:index.html
    rs = getattr(state, "resulting_state", "") or ""
    for prefix in _FILE_STATE_PREFIXES:
        if rs.startswith(prefix):
            remainder = rs[len(prefix):]
            for part in remainder.split(","):
                part = part.strip()
                if part:
                    files.add(_normalize_path(part))
            break

    return files


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Semantic terminal command classification
# ---------------------------------------------------------------------------

# Regex patterns for each semantic command group
_SEARCH_CMD_RE = re.compile(
    r"^(?:grep|rg|ag|ack|fgrep|egrep)\b"
    r"|\bgrep\b",  # pipes like "find | grep"
    re.IGNORECASE,
)
_VIEW_CMD_RE = re.compile(
    r"^(?:cat|head|tail|less|more)\b"
    r"|^sed\s+-?n",  # sed in print mode
    re.IGNORECASE,
)
_EXEC_CMD_RE = re.compile(
    r"^(?:python|node|bash|sh|ruby|perl)\s",
    re.IGNORECASE,
)


def _classify_terminal_command(normalized_cmd: str) -> tuple | None:
    """Classify a normalised terminal command into a semantic group.

    Returns ``(group_name, target)`` or *None*.

    Groups:
    - ``("search", search_pattern)`` — grep/rg/ag/find+grep
    - ``("view", file_path)``        — cat/sed -n/head/tail
    - ``("execute", script_name)``   — python/node/bash + script
    """
    cmd = normalized_cmd.strip()
    if not cmd:
        return None

    # Search commands: extract the search pattern/term
    if _SEARCH_CMD_RE.search(cmd):
        # Try to extract the main search pattern (first non-flag argument)
        # e.g. "grep -rn 'foo'" → "foo", "rg 'pattern' src/" → "pattern"
        m = re.search(r"['\"]([^'\"]+)['\"]", cmd)
        if m:
            return ("search", m.group(1).lower())
        # Fallback: first non-flag token after the command
        parts = cmd.split()
        for p in parts[1:]:
            if not p.startswith("-") and not p.startswith("/"):
                return ("search", p.lower())
        return ("search", "")

    # View commands: extract the file being viewed
    if _VIEW_CMD_RE.search(cmd):
        # Extract file path — last argument that looks like a path
        parts = cmd.split()
        for p in reversed(parts):
            if "/" in p or p.endswith((".py", ".js", ".ts", ".c", ".h",
                                       ".java", ".rs", ".go", ".rb",
                                       ".txt", ".md", ".json", ".yaml",
                                       ".yml", ".toml", ".cfg", ".ini")):
                return ("view", p.lower())
        return ("view", "")

    # Execute commands: extract script name
    if _EXEC_CMD_RE.search(cmd):
        m = re.search(r"\b(?:python|node|bash|sh|ruby|perl)\s+(\S+)", cmd)
        if m:
            return ("execute", m.group(1).lower())
        return ("execute", "")

    return None


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    path = path.lstrip("/\\")
    for prefix in ("workspace/", "workspaces/", "home/", "tmp/"):
        if path.lower().startswith(prefix):
            path = path[len(prefix):]
    return path.lower().replace("\\", "/")


def _normalize_command(cmd: str) -> str:
    if not cmd:
        return ""
    return re.sub(r"\s+", " ", cmd.strip()).lower()


def _word_similarity(a: str, b: str) -> float:
    """Jaccard similarity over whitespace-tokenised words."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _compute_line_overlap(range1: tuple, range2: tuple) -> float:
    """Jaccard-like overlap between two line ranges."""
    if not range1 or not range2:
        return 0.0
    s1, e1 = range1
    s2, e2 = range2
    intersection = max(0, min(e1, e2) - max(s1, s2) + 1)
    union = max(e1, e2) - min(s1, s2) + 1
    return intersection / union if union > 0 else 0.0


def _extract_scope_from_description(state: "State") -> Set[str]:
    """Extract function/class names from *content_description*."""
    desc = getattr(state, "content_description", "") or ""
    if not desc:
        return set()
    names: Set[str] = set()

    # 'name' (single-quoted identifiers)
    names.update(re.findall(r"'([a-zA-Z_][a-zA-Z0-9_]*)'", desc))

    # functions: name1, name2
    for match in re.findall(r'functions?:\s*([a-zA-Z_][a-zA-Z0-9_,\s]*)', desc, re.IGNORECASE):
        for n in match.split(","):
            n = n.strip()
            if n and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', n):
                names.add(n)

    # classes: Name1, Name2
    for match in re.findall(r'classes?:\s*([a-zA-Z_][a-zA-Z0-9_,\s]*)', desc, re.IGNORECASE):
        for n in match.split(","):
            n = n.strip()
            if n and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', n):
                names.add(n)

    # added/removed function 'name'
    names.update(re.findall(
        r'(?:added|removed)\s+(?:function|class|method)\s+[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?',
        desc, re.IGNORECASE,
    ))

    # function/method/class NAME
    names.update(re.findall(
        r'(?:function|method|class|def)\s+[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?',
        desc, re.IGNORECASE,
    ))

    # renamed X to Y
    for old, new in re.findall(
        r'renamed\s+(?:\w+\s+)?[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?\s+to\s+[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?',
        desc, re.IGNORECASE,
    ):
        names.add(old)
        names.add(new)

    names = {n for n in names if "." not in n or n.endswith(".py")}
    exclude = {"in", "to", "from", "the", "and", "or", "with", "modified", "added",
               "removed", "renamed", "imports", "code", "file", "created", "python", "javascript"}
    return names - exclude
