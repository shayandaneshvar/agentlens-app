"""High-level trace loading and merging API.

Public surface
--------------
- :func:`load` — load a single run trace from a trajectory file.
- :func:`merge` — merge multiple traces into a ground-truth trace.

Examples
--------
>>> from swe_trace_sdk import trace
>>> t = trace.load("path/to/chat-export-logs.json", format="chatlog")
>>> gt = trace.merge([t1, t2, t3])
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List, Optional

from .models import State, Transition, Trace
from .equivalence import StateEquivalence
from .intent import label_trace_intents as _label_trace

logger = logging.getLogger(__name__)

__all__ = ["load", "merge"]


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def load(filename: str, *, format: str) -> Trace:
    """Load a single-run trace.

    *filename* can be either:

    * a raw trajectory file (e.g. ``chat-export-logs.json`` for
      the evaluation platform format), or
    * a previously-saved :class:`~swe_trace_sdk.models.Trace` JSON
      (use ``format="trace"``).

    Parameters
    ----------
    filename : str
        Path to the trajectory or trace file.
    format : str
        ``"chatlog"`` — parse a raw evaluation platform trajectory.
        ``"trace"`` — load a Trace JSON previously saved by the SDK.

    Returns
    -------
    Trace
    """
    from .io import load_trajectory  # deferred to avoid circular

    return load_trajectory(filename, format=format)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def merge(
    traces: List[Trace],
    *,
    equivalence: str = "default",
    use_llm: bool = False,
) -> Trace:
    """Merge multiple traces into a single ground-truth trace.

    Parameters
    ----------
    traces : list[Trace]
        Two or more traces to merge.
    equivalence : str
        Equivalence strategy name.  Currently only ``"default"`` is
        supported (exact → heuristic → optional LLM).
    use_llm : bool
        Enable LLM-backed semantic equivalence for ambiguous cases.

    Returns
    -------
    Trace
        A merged trace preserving branch information and per-state
        ``trace_count`` metadata.
    """
    if not traces:
        return Trace()
    if len(traces) == 1:
        return traces[0].copy()

    merger = _TraceMerger(use_llm=use_llm)
    return merger.merge(traces)


# ---------------------------------------------------------------------------
# Internal merger
# ---------------------------------------------------------------------------

class _TraceMerger:
    """Incremental trace merger (internal)."""

    def __init__(self, use_llm: bool = False) -> None:
        self._eq = StateEquivalence(use_llm=use_llm)
        self._state_counter = 0
        self._trans_counter = 0
        self.stats = _MergeStats()

    # ------------------------------------------------------------------

    def merge(self, traces: List[Trace]) -> Trace:
        merged = self._init_from(traces[0])
        existing_traces = traces[0].metadata.get("num_traces", 1)
        self.stats.traces_merged = existing_traces

        for i, tr in enumerate(traces[1:], 2):
            logger.info("Merging trace %d/%d …", i, len(traces))
            merged = self._merge_one(merged, tr)
            self.stats.traces_merged += tr.metadata.get("num_traces", 1)

        merged = self._consolidate(merged)
        merged = self._remove_backward_edges(merged)

        merged.metadata["merge_stats"] = self.stats.to_dict()
        merged.metadata["equivalence_stats"] = self._eq.get_stats()

        # Re-label intent stages on the merged trace (context may differ from
        # individual traces due to branch consolidation / reordering).
        _label_trace(merged)

        logger.info("Merge complete: %d states, %d transitions", len(merged.states), len(merged.transitions))
        return merged

    # ------------------------------------------------------------------
    # Bootstrap from first trace
    # ------------------------------------------------------------------

    def _init_from(self, trace: Trace) -> Trace:
        merged = Trace()
        merged.metadata = {
            "source": "merger",
            "num_traces": trace.metadata.get("num_traces", 1),
            "trace_sources": list(trace.metadata.get("trace_sources", [trace.metadata.get("source_file", "unknown")])),
        }

        id_map: Dict[str, str] = {}
        for old_id, state in trace.states.items():
            new_id = self._new_sid()
            id_map[old_id] = new_id
            ns = _copy_state(state, new_id)
            ns.metadata["original_ids"] = [old_id]
            ns.metadata["trace_count"] = 1
            merged.add_state(ns)
            self.stats.states_added += 1

        for t in trace.transitions:
            merged.add_transition(Transition(
                transition_id=self._new_tid(),
                from_state=id_map.get(t.from_state, t.from_state),
                to_state=id_map.get(t.to_state, t.to_state),
                action_type=t.action_type,
                action_data=dict(t.action_data) if t.action_data else {},
                step=t.step,
                metadata=dict(t.metadata) if t.metadata else {},
            ))
            self.stats.transitions_added += 1

        if trace.initial_state and trace.initial_state in id_map:
            merged.initial_state = id_map[trace.initial_state]
        elif merged.states:
            merged.initial_state = next(iter(merged.states))

        # Track the real terminal state (last state of this trace).
        # If the input is itself a previously-merged trace, carry forward
        # its existing real_terminal_ids.
        existing_terminals = set(trace.metadata.get("real_terminal_ids", []))
        if existing_terminals:
            merged.metadata["real_terminal_ids"] = [
                id_map.get(tid, tid) for tid in existing_terminals
                if id_map.get(tid, tid) in merged.states
            ]
        else:
            # Single trace: the terminal is the last state in step order
            ordered = _ordered_states(trace)
            if ordered:
                last_old_id = ordered[-1].state_id
                last_new_id = id_map.get(last_old_id, last_old_id)
                merged.metadata["real_terminal_ids"] = [last_new_id]

        return merged

    # ------------------------------------------------------------------
    # Merge a single additional trace
    # ------------------------------------------------------------------

    def _merge_one(self, merged: Trace, new_trace: Trace) -> Trace:
        source = new_trace.metadata.get("source_file", "unknown")
        merged.metadata.setdefault("trace_sources", []).append(source)
        merged.metadata["num_traces"] = merged.metadata.get("num_traces", 1) + 1

        new_ordered = _ordered_states(new_trace)
        merged_ordered = _ordered_states(merged)

        if not new_ordered:
            return merged

        pos = 0
        branch_point: Optional[str] = None
        id_map: Dict[str, str] = {}

        for idx, ns in enumerate(new_ordered):
            if pos < len(merged_ordered):
                ms = merged_ordered[pos]
                result = self._eq.check(ms, ns, position=idx)
                self.stats.equivalence_checks += 1

                if result.equivalent:
                    _merge_state_meta(ms, ns)
                    id_map[ns.state_id] = ms.state_id
                    self.stats.states_merged += 1
                    pos += 1
                    continue
                else:
                    branch_point = ms.state_id
                    self.stats.branches_created += 1

            new_id = self._new_sid()
            added = _copy_state(ns, new_id)
            added.metadata["original_ids"] = [ns.state_id]
            added.metadata["trace_count"] = 1
            if branch_point:
                added.metadata["branch_from"] = branch_point
                merged.branches.setdefault(branch_point, []).append(new_id)

            merged.add_state(added)
            id_map[ns.state_id] = new_id
            self.stats.states_added += 1

        for t in new_trace.transitions:
            f = id_map.get(t.from_state)
            to = id_map.get(t.to_state)
            if not f or not to:
                continue
            if not _transition_exists(merged, f, to, t.action_type):
                merged.add_transition(Transition(
                    transition_id=self._new_tid(),
                    from_state=f, to_state=to,
                    action_type=t.action_type,
                    action_data=dict(t.action_data) if t.action_data else {},
                    step=t.step,
                    metadata=dict(t.metadata) if t.metadata else {},
                ))
                self.stats.transitions_added += 1

        # Track the real terminal state of this new trace.
        existing_rt = set(merged.metadata.get("real_terminal_ids", []))
        new_ordered = _ordered_states(new_trace)
        if new_ordered:
            last_old_id = new_ordered[-1].state_id
            last_new_id = id_map.get(last_old_id, last_old_id)
            existing_rt.add(last_new_id)
        merged.metadata["real_terminal_ids"] = list(existing_rt)

        return merged

    # ------------------------------------------------------------------
    # Post-processing: consolidate duplicate resulting states
    # ------------------------------------------------------------------

    def _consolidate(self, trace: Trace) -> Trace:
        skip = {"llm_response:terminal", "initial", ""}
        cross_step_prefixes = (
            "file_created:", "file_read:", "file_patched:", "files_modified:",
            "llm_response:confirmed:",
        )

        groups: Dict[tuple, List[str]] = {}
        for sid, s in trace.states.items():
            rs = getattr(s, "resulting_state", "") or ""
            if rs in skip:
                continue
            # Intent-stage-aware grouping: the same file action at different
            # stages (e.g. read_file during exploration vs. verification)
            # must NOT be consolidated — they serve different purposes and
            # merging them creates spurious shortcut transitions in the DAG.
            stage = getattr(s, "intent_stage", "") or ""
            key = (rs, stage) if rs.startswith(cross_step_prefixes) else (rs, s.step)
            groups.setdefault(key, []).append(sid)

        remap: Dict[str, str] = {}
        to_remove: List[str] = []

        for _, sids in groups.items():
            if len(sids) <= 1:
                continue
            sids_sorted = sorted(sids, key=lambda s: trace.states[s].step)
            canon = sids_sorted[0]
            canon_state = trace.states[canon]
            for other_id in sids_sorted[1:]:
                other = trace.states[other_id]
                canon_state.metadata["trace_count"] = canon_state.metadata.get("trace_count", 1) + other.metadata.get("trace_count", 1)
                canon_state.metadata.setdefault("original_ids", []).extend(other.metadata.get("original_ids", [other_id]))
                canon_state.files_touched.update(other.files_touched)
                remap[other_id] = canon
                to_remove.append(other_id)

        if not remap:
            return trace

        seen: set = set()
        new_trans: List[Transition] = []
        for t in trace.transitions:
            fs = remap.get(t.from_state, t.from_state)
            ts = remap.get(t.to_state, t.to_state)
            if fs == ts:
                continue
            fs_step = trace.states.get(fs, State(state_id="", step=0)).step
            ts_step = trace.states.get(ts, State(state_id="", step=999)).step
            if ts_step < fs_step:
                continue
            key = (fs, ts, t.action_type)
            if key in seen:
                continue
            seen.add(key)
            new_trans.append(Transition(
                transition_id=t.transition_id,
                from_state=fs, to_state=ts,
                action_type=t.action_type,
                action_data=t.action_data,
                step=t.step, metadata=t.metadata,
            ))

        for sid in to_remove:
            trace.states.pop(sid, None)
        trace.transitions = new_trans

        if trace.branches:
            updated: Dict[str, List[str]] = {}
            for bp, targets in trace.branches.items():
                nbp = remap.get(bp, bp)
                nt = list(dict.fromkeys(remap.get(x, x) for x in targets))
                if nt:
                    # Merge targets when multiple branch points consolidate
                    # to the same canonical state.
                    existing = updated.get(nbp, [])
                    for t in nt:
                        if t not in existing:
                            existing.append(t)
                    updated[nbp] = existing
            trace.branches = updated

        if trace.initial_state in remap:
            trace.initial_state = remap[trace.initial_state]

        # Propagate real_terminal_ids through the remap so that
        # consolidated states keep the real-terminal designation.
        old_rt = trace.metadata.get("real_terminal_ids", [])
        if old_rt:
            new_rt = list(dict.fromkeys(
                remap.get(tid, tid) for tid in old_rt
                if remap.get(tid, tid) in trace.states
            ))
            trace.metadata["real_terminal_ids"] = new_rt

        return trace

    # ------------------------------------------------------------------
    # Remove backward edges / self-loops
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_backward_edges(trace: Trace) -> Trace:
        valid: List[Transition] = []
        for t in trace.transitions:
            fs = trace.states.get(t.from_state)
            ts = trace.states.get(t.to_state)
            if not fs or not ts:
                continue  # Drop transitions with non-existent endpoints
            if t.from_state == t.to_state:
                continue
            if ts.step < fs.step:
                continue
            valid.append(t)
        trace.transitions = valid
        return trace

    # ------------------------------------------------------------------
    # ID generators
    # ------------------------------------------------------------------

    def _new_sid(self) -> str:
        self._state_counter += 1
        return f"merged_state_{self._state_counter}"

    def _new_tid(self) -> str:
        self._trans_counter += 1
        return f"merged_trans_{self._trans_counter}"


# ---------------------------------------------------------------------------
# Merge stats
# ---------------------------------------------------------------------------

class _MergeStats:
    def __init__(self) -> None:
        self.traces_merged = 0
        self.states_added = 0
        self.states_merged = 0
        self.transitions_added = 0
        self.branches_created = 0
        self.equivalence_checks = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "traces_merged": self.traces_merged,
            "states_added": self.states_added,
            "states_merged": self.states_merged,
            "transitions_added": self.transitions_added,
            "branches_created": self.branches_created,
            "equivalence_checks": self.equivalence_checks,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ordered_states(trace: Trace) -> List[State]:
    return sorted(trace.states.values(), key=lambda s: s.step)


def _copy_state(state: State, new_id: str) -> State:
    return State(
        state_id=new_id,
        step=state.step,
        log_entry=state.log_entry,
        observation=state.observation,
        resulting_state=getattr(state, "resulting_state", ""),
        content_hash=getattr(state, "content_hash", ""),
        content_description=getattr(state, "content_description", ""),
        files_touched=set(state.files_touched),
        tool_used=state.tool_used,
        file_path=getattr(state, "file_path", ""),
        line_range=getattr(state, "line_range", None),
        relative_position=getattr(state, "relative_position", ""),
        operation_type=getattr(state, "operation_type", ""),
        edit_type=getattr(state, "edit_type", ""),
        function_name=getattr(state, "function_name", ""),
        class_name=getattr(state, "class_name", ""),
        scope_path=getattr(state, "scope_path", ""),
        intent_stage=getattr(state, "intent_stage", ""),
        metadata=dict(state.metadata) if state.metadata else {},
    )


def _merge_state_meta(dst: State, src: State) -> None:
    dst.metadata["trace_count"] = dst.metadata.get("trace_count", 1) + 1
    dst.metadata.setdefault("original_ids", []).append(src.state_id)
    if src.files_touched:
        dst.files_touched.update(src.files_touched)


def _transition_exists(trace: Trace, from_id: str, to_id: str, action_type: str) -> bool:
    return any(
        t.from_state == from_id and t.to_state == to_id and t.action_type == action_type
        for t in trace.transitions
    )
