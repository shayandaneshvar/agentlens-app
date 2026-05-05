"""Matching a candidate trace against a ground-truth trace.

Public surface
--------------
- :func:`run` — compare a candidate against ground truth.
- :class:`MatchResult` — result container with metrics, alignment, and
  per-step rationales.
Example
-------
>>> from swe_trace_sdk import match
>>> result = match.run(candidate, ground_truth)
>>> print(result.metrics.coverage_percent)
>>> print(result.metrics.terminal_state_match)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    State,
    Trace,
    Transition,
    QualityReport,
    CohortRanking,
    CohortEntry,
    FailureReason,
    DivergencePoint,
    StageCoverageDetail,
    DivergenceSegment,
    StageComparison,
    InefficiencyReport,
    RetryLoop,
    Backtrack,
    RedundantStep,
    UnnecessaryExploration,
    CyclicPattern,
    ToolInefficiency,
    QualitySignal,
)
from .equivalence import StateEquivalence
from .intent import (
    compute_fingerprint,
    workflow_similarity as _wf_similarity,
    compute_coherence_score,
    compute_temporal_profile_divergence,
    _classify_transition,
    EXPLORATION,
    IMPLEMENTATION,
    VERIFICATION,
    ORCHESTRATION,
)

logger = logging.getLogger(__name__)

__all__ = [
    "run",
    "quality_assessment",
    "rank_in_cohort",
    "extract_required_tools",
    "check_process_coverage",
    "extract_required_files",
    "check_file_coverage",
    "MatchResult",
    "MatchMetrics",
    "StepAlignment",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class StepAlignment:
    """Mapping of a single candidate step to a ground-truth state."""

    candidate_step: int
    candidate_state_id: str
    ground_truth_state_id: Optional[str]
    matched: bool
    rationale: str = ""


@dataclass
class MatchMetrics:
    """Quantitative match metrics."""

    coverage_percent: float = 0.0
    """Percentage of ground-truth states matched by the candidate."""

    terminal_state_match: bool = False
    """Whether the candidate's terminal state matches the ground-truth terminal."""

    perfect_match: bool = False
    """Whether every ground-truth state was found in the candidate (100 % coverage)."""

    matched_count: int = 0
    """Number of ground-truth states matched."""

    total_ground_truth_states: int = 0
    """Total number of states in the (best) ground-truth path."""

    candidate_states: int = 0
    """Total states in the candidate trace."""

    best_path_index: int = 0
    """Index of the ground-truth path that yielded the best coverage."""

    total_paths: int = 0
    """Number of paths enumerated from the ground-truth trace."""

    precision_percent: float = 0.0
    """Percentage of candidate states that matched ground-truth states."""

    f1_score: float = 0.0
    """Harmonic mean of coverage (recall) and precision percentages."""

    # Per-stage coverage (only populated when states have intent-stage labels)
    stage_coverage: Dict[str, float] = field(default_factory=dict)
    """Coverage percentage broken down by intent stage (exploration/implementation/verification/orchestration)."""

    stage_matched: Dict[str, int] = field(default_factory=dict)
    """Number of matched states per intent stage."""

    stage_total: Dict[str, int] = field(default_factory=dict)
    """Total ground-truth states per intent stage."""

    stage_completeness: float = 0.0
    """Fraction of GT stages the candidate covered (0.0–1.0).
    E.g. GT path has E, I, V, O but candidate only matches E, O → 0.5."""

    weighted_score: float = 0.0
    """Stage-weighted coverage score (0.0–100.0).
    Stages are weighted by importance: implementation=3, verification=2,
    exploration=1, orchestration=0.5.  This amplifies the signal from
    missing critical stages like implementation."""

    workflow_similarity: float = 0.0
    """Workflow fingerprint similarity (0.0–1.0).
    Compares the candidate's intent-stage transition pattern against the
    ground truth using longest-common-subsequence ratio."""

    coherence_score: float = 0.0
    """Trajectory coherence score (0.0–1.0).
    Measures forward-progress of the candidate's intent-stage transitions
    vs. backtracks and blind retries.  Higher = cleaner trajectory."""

    temporal_profile_score: float = 0.0
    """Temporal stage-profile similarity (0.0–1.0).
    Compares when intent stages occur (early/middle/late) in the candidate
    vs. the ground truth using Jensen-Shannon divergence."""

    bottleneck_coverage: float = 0.0
    """Minimum per-stage coverage (0.0–100.0).
    The coverage percentage of the weakest intent stage.  A value of 0
    means the candidate completely missed at least one GT stage —
    a strong failure indicator."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage_percent": round(self.coverage_percent, 2),
            "terminal_state_match": self.terminal_state_match,
            "perfect_match": self.perfect_match,
            "matched_count": self.matched_count,
            "total_ground_truth_states": self.total_ground_truth_states,
            "candidate_states": self.candidate_states,
            "best_path_index": self.best_path_index,
            "total_paths": self.total_paths,
            "precision_percent": round(self.precision_percent, 2),
            "f1_score": round(self.f1_score, 2),
            "stage_coverage": {k: round(v, 2) for k, v in self.stage_coverage.items()},
            "stage_matched": self.stage_matched,
            "stage_total": self.stage_total,
            "stage_completeness": round(self.stage_completeness, 4),
            "weighted_score": round(self.weighted_score, 2),
            "workflow_similarity": round(self.workflow_similarity, 4),
            "coherence_score": round(self.coherence_score, 4),
            "temporal_profile_score": round(self.temporal_profile_score, 4),
            "bottleneck_coverage": round(self.bottleneck_coverage, 2),
        }


@dataclass
class MatchResult:
    """Full result of matching a candidate against ground truth."""

    metrics: MatchMetrics = field(default_factory=MatchMetrics)

    alignment: List[StepAlignment] = field(default_factory=list)
    """Per-step alignment (candidate step → ground-truth state)."""

    matched_indexes: List[int] = field(default_factory=list)
    """Indexes of ground-truth states that were matched, in order."""

    matched_gt_state_ids: List[str] = field(default_factory=list)
    """State IDs of ground-truth states that were matched (set-based)."""

    missing_indexes: List[int] = field(default_factory=list)
    """Indexes of ground-truth states that the candidate missed."""

    divergence_index: Optional[int] = None
    """Index of the first ground-truth state the candidate diverges at."""

    equivalence_stats: Dict[str, int] = field(default_factory=dict)
    """Statistics from the equivalence checker (comparisons, cache hits, …)."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "matched_indexes": self.matched_indexes,
            "matched_gt_state_ids": self.matched_gt_state_ids,
            "missing_indexes": self.missing_indexes,
            "divergence_index": self.divergence_index,
            "alignment": [
                {
                    "candidate_step": a.candidate_step,
                    "candidate_state_id": a.candidate_state_id,
                    "ground_truth_state_id": a.ground_truth_state_id,
                    "matched": a.matched,
                    "rationale": a.rationale,
                }
                for a in self.alignment
            ],
            "equivalence_stats": self.equivalence_stats,
        }


# ---------------------------------------------------------------------------
# Intent-stage weights (higher = more important for task success)
# ---------------------------------------------------------------------------

STAGE_WEIGHTS: Dict[str, float] = {
    "implementation": 3.0,
    "verification": 2.0,
    "exploration": 1.0,
    "orchestration": 0.5,
}
"""Importance weights per intent stage for the weighted score."""

_DEFAULT_WEIGHT = 0.5  # for unknown / unlabeled stages


def _gt_stage_count(gt_seq: List[State]) -> int:
    """Count distinct non-empty intent stages in a GT path."""
    return len({getattr(s, "intent_stage", "") for s in gt_seq} - {""})


def _gt_has_verification(gt_seq: List[State]) -> bool:
    """Check if a GT path includes at least one verification state."""
    return any(getattr(s, "intent_stage", "") == "verification" for s in gt_seq)


def _is_better(candidate: "_PathMatch", current_best: "_PathMatch") -> bool:
    """Decide whether *candidate* is a better path match than *current_best*.

    Selection priority (in order):
    1. Prefer GT paths with more distinct intent stages — richer
       reference paths that include verification are always better
       benchmarks than paths without it.  A path with {E, I, V} is
       a more demanding (and informative) reference than {E, I}.
    2. More matched states — how much of the GT the candidate
       actually followed.
    3. Higher coverage % — among paths with equal stage diversity
       and matched counts.
    4. Longer path — exposes more stages and more miss opportunities.
    """
    # 1. Prefer GT paths with more distinct intent stages
    cand_stages = _gt_stage_count(candidate.gt_seq)
    best_stages = _gt_stage_count(current_best.gt_seq)
    if cand_stages != best_stages:
        return cand_stages > best_stages
    # 2. More matched states
    if candidate.matched != current_best.matched:
        return candidate.matched > current_best.matched
    # 3. Higher coverage %
    if abs(candidate.coverage - current_best.coverage) > 1e-9:
        return candidate.coverage > current_best.coverage
    # 4. Longer path
    return candidate.total > current_best.total


# ---------------------------------------------------------------------------
# Canonical path selection
# ---------------------------------------------------------------------------

def _select_canonical_path(gt_paths: List[List[State]]) -> int:
    """Select the canonical GT path index — independent of any candidate.

    Selection priority (in order):
    1. Most distinct intent stages (richer reference).
    2. Longest path (more steps = more demanding benchmark).
    """
    best_idx = 0
    best_stages = -1
    best_len = -1
    for idx, path in enumerate(gt_paths):
        stages = len({getattr(s, "intent_stage", "") for s in path} - {""})
        plen = len(path)
        if stages > best_stages or (stages == best_stages and plen > best_len):
            best_idx = idx
            best_stages = stages
            best_len = plen
    return best_idx


# ---------------------------------------------------------------------------
# Path enumeration
# ---------------------------------------------------------------------------

def _enumerate_paths(trace: Trace, *, max_paths: int = 50) -> List[List[State]]:
    """Enumerate all root-to-terminal paths using DFS with cycle detection.

    If the trace has ``real_terminal_ids`` metadata (set during merge),
    only paths ending at a *real* terminal state are returned.  This
    prevents the combinatorial explosion of artifact short paths caused
    by ``_consolidate`` + ``_remove_backward_edges`` creating fake
    dead-end states.

    For single-run traces (no ``real_terminal_ids``), every state with no
    outgoing transitions is treated as a terminal — the original behaviour.

    Parameters
    ----------
    max_paths : int
        Stop enumeration once this many paths have been found. Prevents
        combinatorial explosion on dense PTAs. Default: 500.
    """
    if not trace.initial_state or trace.initial_state not in trace.states:
        return []

    outgoing: Dict[str, List[Transition]] = {}
    for t in trace.transitions:
        outgoing.setdefault(t.from_state, []).append(t)
    for lst in outgoing.values():
        lst.sort(key=lambda tr: tr.transition_id)

    real_terminals: Optional[Set[str]] = None
    rt_meta = trace.metadata.get("real_terminal_ids")
    if rt_meta is not None:
        real_terminals = set(rt_meta)

    paths: List[List[State]] = []

    def dfs(sid: str, path: List[State], visited: Set[str]) -> None:
        if len(paths) >= max_paths:
            return
        if sid in visited or sid not in trace.states:
            return
        new_path = path + [trace.states[sid]]
        new_visited = visited | {sid}
        outs = outgoing.get(sid, [])

        if real_terminals is not None:
            # Merged PTA: only emit paths that end at a real terminal
            if sid in real_terminals:
                paths.append(new_path)
                if len(paths) >= max_paths:
                    return
            # Continue DFS even if this is a real terminal — there may
            # be outgoing edges to other real terminals.
            for tr in outs:
                dfs(tr.to_state, new_path, new_visited)
                if len(paths) >= max_paths:
                    return
        else:
            # Single-run trace: original behaviour
            if not outs:
                paths.append(new_path)
            else:
                for tr in outs:
                    dfs(tr.to_state, new_path, new_visited)
                    if len(paths) >= max_paths:
                        return

    dfs(trace.initial_state, [], set())
    return paths


# ---------------------------------------------------------------------------
# Subsequence coverage
# ---------------------------------------------------------------------------

def _subsequence_coverage(
    gt_seq: List[State],
    cand_seq: List[State],
    eq: StateEquivalence,
) -> Tuple[int, List[int]]:
    """Return (matched_count, matched_gt_indexes) via greedy forward scan."""
    matched = 0
    matched_idx: List[int] = []
    j = 0

    for i, s_gt in enumerate(gt_seq):
        while j < len(cand_seq):
            if eq.check(s_gt, cand_seq[j], position=i).equivalent:
                matched += 1
                matched_idx.append(i)
                j += 1
                break
            j += 1
        else:
            break

    return matched, matched_idx


def _set_coverage(
    gt_seq: List[State],
    cand_seq: List[State],
    eq: StateEquivalence,
) -> Tuple[int, int, List[int]]:
    """Order-independent set matching.

    Returns (gt_matched, cand_matched, matched_gt_indexes).
    Each ground-truth state matches at most one candidate state and
    vice-versa (greedy 1-to-1 assignment).
    """
    used_cand: Set[int] = set()
    matched_gt_idx: List[int] = []

    for i, s_gt in enumerate(gt_seq):
        for j, s_cand in enumerate(cand_seq):
            if j in used_cand:
                continue
            if eq.check(s_gt, s_cand, position=i).equivalent:
                matched_gt_idx.append(i)
                used_cand.add(j)
                break

    return len(matched_gt_idx), len(used_cand), matched_gt_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    candidate: Trace,
    ground_truth: Trace,
    *,
    matcher: str = "subsequence_coverage",
    use_llm: bool = False,
    gt_strategy: str = "best_match",
) -> MatchResult:
    """Match *candidate* against *ground_truth*.

    Parameters
    ----------
    candidate : Trace
        The trace to evaluate.
    ground_truth : Trace
        The reference (merged) trace.
    matcher : str
        Matching algorithm.  Only ``"subsequence_coverage"`` is
        currently supported.
    use_llm : bool
        Forward to the equivalence checker.
    gt_strategy : str
        How to select the ground-truth path for comparison.

        ``"best_match"`` (default) — pick the GT path that maximises
        the candidate's coverage score (optimistic; current behaviour).

        ``"canonical"`` — pick the longest GT path with the most
        intent stages, independent of the candidate.  This gives a
        fixed reference so all candidates are scored against the same
        denominator, enabling fairer cross-agent comparison.

    Returns
    -------
    MatchResult
    """
    eq = StateEquivalence(use_llm=use_llm)

    gt_paths = _enumerate_paths(ground_truth)
    cand_paths = _enumerate_paths(candidate)

    # Cap the number of ground-truth paths to avoid combinatorial explosion.
    # Keep the longest paths (most informative for coverage matching).
    _MAX_GT_PATHS = 500
    if len(gt_paths) > _MAX_GT_PATHS:
        gt_paths.sort(key=len, reverse=True)
        gt_paths = gt_paths[:_MAX_GT_PATHS]

    if not gt_paths:
        return MatchResult(metrics=MatchMetrics())

    if not cand_paths:
        return MatchResult(metrics=MatchMetrics(
            total_ground_truth_states=len(gt_paths[0]),
            total_paths=len(gt_paths),
        ))

    cand_seq = max(cand_paths, key=len)

    # Filter out LLM request states — they always match each other
    # and dilute the signal.  Keep only tool-call (and initial) states.
    def _is_actionable(s: State) -> bool:
        entry = getattr(s, "log_entry", None)
        if entry is None:
            return True          # initial / missing-detail states kept
        return entry.kind != "request"

    cand_seq = [s for s in cand_seq if _is_actionable(s)]
    if not cand_seq:
        return MatchResult(metrics=MatchMetrics(
            total_ground_truth_states=len(gt_paths[0]),
            total_paths=len(gt_paths),
        ))

    best: Optional[_PathMatch] = None
    best_gt_seq: Optional[List[State]] = None

    # ------------------------------------------------------------------
    # Strategy: canonical — pick GT path independently of candidate
    # ------------------------------------------------------------------
    if gt_strategy == "canonical":
        canonical_idx = _select_canonical_path(gt_paths)
        gt_seq = [s for s in gt_paths[canonical_idx] if _is_actionable(s)]
        if gt_seq:
            set_gt_matched, set_cand_matched, set_matched_idx = _set_coverage(gt_seq, cand_seq, eq)
            total = len(gt_seq)
            coverage = (set_gt_matched / total * 100.0) if total else 0.0
            terminal = eq.check(gt_seq[-1], cand_seq[-1], position=total - 1).equivalent
            perfect = set_gt_matched == total
            best = _PathMatch(
                path_idx=canonical_idx,
                gt_seq=gt_seq,
                matched=set_gt_matched,
                total=total,
                coverage=coverage,
                terminal=terminal,
                perfect=perfect,
                matched_idx=set_matched_idx,
            )
            best_gt_seq = gt_seq
    else:
        # ------------------------------------------------------------------
        # Strategy: best_match — pick GT path that maximises candidate score
        # ------------------------------------------------------------------
        for path_idx, gt_seq_raw in enumerate(gt_paths):
            gt_seq = [s for s in gt_seq_raw if _is_actionable(s)]
            if not gt_seq:
                continue

            # Use set-based (order-independent) matching as the primary method.
            # This prevents 0% coverage when the candidate does the same work
            # in a different order than the GT path.
            set_gt_matched, set_cand_matched, set_matched_idx = _set_coverage(gt_seq, cand_seq, eq)
            total = len(gt_seq)
            coverage = (set_gt_matched / total * 100.0) if total else 0.0
            terminal = eq.check(gt_seq[-1], cand_seq[-1], position=total - 1).equivalent
            perfect = set_gt_matched == total

            pm = _PathMatch(
                path_idx=path_idx,
                gt_seq=gt_seq,
                matched=set_gt_matched,
                total=total,
                coverage=coverage,
                terminal=terminal,
                perfect=perfect,
                matched_idx=set_matched_idx,
            )
            if best is None or _is_better(pm, best):
                best = pm
                best_gt_seq = gt_seq

    if best is None or best_gt_seq is None:
        return MatchResult(metrics=MatchMetrics(
            total_ground_truth_states=max((len(p) for p in gt_paths), default=0),
            total_paths=len(gt_paths),
        ))

    # Also run subsequence matching on the winning path for ordering analysis
    subseq_matched, subseq_matched_idx = _subsequence_coverage(best_gt_seq, cand_seq, eq)

    # Divergence: first GT state missed in set-based matching
    gt_matched_set = set(best.matched_idx)
    missing = [i for i in range(best.total) if i not in gt_matched_set]
    divergence = missing[0] if missing else None

    # ── Per-stage coverage ────────────────────────────────────────
    stage_total: Dict[str, int] = {}
    stage_matched: Dict[str, int] = {}
    matched_set = set(best.matched_idx)

    for i, gt_state in enumerate(best.gt_seq):
        stg = getattr(gt_state, "intent_stage", "") or ""
        if not stg:
            continue
        stage_total[stg] = stage_total.get(stg, 0) + 1
        if i in matched_set:
            stage_matched[stg] = stage_matched.get(stg, 0) + 1

    stage_coverage: Dict[str, float] = {}
    for stg, total in stage_total.items():
        matched_in_stage = stage_matched.get(stg, 0)
        stage_coverage[stg] = (matched_in_stage / total * 100.0) if total else 0.0

    # ── Stage completeness: fraction of GT stages with ≥1 match ───
    gt_stages = {stg for stg in stage_total if stage_total[stg] > 0}
    covered_stages = {stg for stg in gt_stages if stage_matched.get(stg, 0) > 0}
    stage_completeness = (
        len(covered_stages) / len(gt_stages) if gt_stages else 1.0
    )

    # ── Weighted score: importance-weighted coverage ───────────────
    w_num = 0.0
    w_den = 0.0
    for stg, total in stage_total.items():
        w = STAGE_WEIGHTS.get(stg, _DEFAULT_WEIGHT)
        w_num += w * stage_matched.get(stg, 0)
        w_den += w * total
    weighted_score = (w_num / w_den * 100.0) if w_den > 0 else 0.0

    # ── Bottleneck coverage: weakest stage ────────────────────────
    bottleneck_cov = min(stage_coverage.values()) if stage_coverage else 0.0

    # ── Workflow fingerprint similarity ───────────────────────────
    gt_fp, _ = compute_fingerprint(best.gt_seq)
    cand_fp, _ = compute_fingerprint(list(cand_seq))
    wf_sim = _wf_similarity(cand_fp, gt_fp)

    # ── Trajectory coherence (candidate self-score) ───────────────
    coherence = compute_coherence_score(list(cand_seq))

    # ── Temporal stage profile divergence (candidate vs GT) ───────
    temporal_profile = compute_temporal_profile_divergence(
        list(cand_seq), best.gt_seq,
    )

    # Build alignment list
    alignment: List[StepAlignment] = []
    gt_idx_iter = iter(best.matched_idx)
    next_gt = next(gt_idx_iter, None)

    for step_i, cand_state in enumerate(cand_seq):
        if next_gt is not None:
            gt_state = best.gt_seq[next_gt]
            if eq.check(gt_state, cand_state, position=next_gt).equivalent:
                alignment.append(StepAlignment(
                    candidate_step=step_i,
                    candidate_state_id=cand_state.state_id,
                    ground_truth_state_id=gt_state.state_id,
                    matched=True,
                    rationale=f"Matched GT step {next_gt}",
                ))
                next_gt = next(gt_idx_iter, None)
                continue

        alignment.append(StepAlignment(
            candidate_step=step_i,
            candidate_state_id=cand_state.state_id,
            ground_truth_state_id=None,
            matched=False,
            rationale="No match in ground truth",
        ))

    # Compute precision and F1 using the set-based (unordered) matching.
    # Recall = GT states found / total GT states — how much ground-truth work was done.
    # Precision = GT-matched cand states / total cand states — how focused was the trajectory.
    # The primary matching is already set-based, so use best.matched directly.
    set_recall = (best.matched / best.total * 100.0) if best.total else 0.0
    # Count distinct candidate states that participated in matches
    _, set_cand_matched, _ = _set_coverage(best_gt_seq, cand_seq, eq)
    set_precision = (set_cand_matched / len(cand_seq) * 100.0) if cand_seq else 0.0
    f1 = (2 * set_precision * set_recall / (set_precision + set_recall)) if (set_precision + set_recall) > 0 else 0.0

    return MatchResult(
        metrics=MatchMetrics(
            coverage_percent=best.coverage,
            terminal_state_match=best.terminal,
            perfect_match=best.perfect,
            matched_count=best.matched,
            total_ground_truth_states=best.total,
            candidate_states=len(cand_seq),
            best_path_index=best.path_idx,
            total_paths=len(gt_paths),
            precision_percent=set_precision,
            f1_score=f1,
            stage_coverage=stage_coverage,
            stage_matched=stage_matched,
            stage_total=stage_total,
            stage_completeness=stage_completeness,
            weighted_score=weighted_score,
            workflow_similarity=wf_sim,
            coherence_score=coherence,
            temporal_profile_score=temporal_profile,
            bottleneck_coverage=bottleneck_cov,
        ),
        alignment=alignment,
        matched_indexes=best.matched_idx,
        matched_gt_state_ids=[best.gt_seq[i].state_id for i in best.matched_idx],
        missing_indexes=missing,
        divergence_index=divergence,
        equivalence_stats=eq.get_stats(),
    )


# ---------------------------------------------------------------------------
# Process coverage
# ---------------------------------------------------------------------------

def extract_required_tools(ground_truth: Trace) -> List[str]:
    """Extract tools that appear in **every** root-to-terminal path.

    These are the mandatory tools for task completion — tools used by
    every successful path in the merged ground-truth trace.

    Parameters
    ----------
    ground_truth : Trace
        A merged ground-truth trace.

    Returns
    -------
    list[str]
        Required tool names (``"llm"`` states are excluded).
    """
    paths = _enumerate_paths(ground_truth)
    if not paths:
        return []

    path_tools = []
    for path in paths:
        tools: Set[str] = set()
        for state in path:
            tool = state.tool_used
            if tool and tool != "llm":
                base_tool = tool.split("[")[0].strip()
                tools.add(base_tool)
        path_tools.append(tools)

    if not path_tools:
        return []

    # Use majority voting: tools that appear in more than half of paths.
    # Strict intersection collapses to only generic tools on merged PTAs.
    from collections import Counter
    tool_counts: Counter = Counter()
    for tools in path_tools:
        for t in tools:
            tool_counts[t] += 1
    threshold = len(path_tools) / 2.0
    return [t for t, count in tool_counts.items() if count > threshold]


def check_process_coverage(
    candidate: Trace, required_tools: List[str]
) -> Tuple[float, List[str]]:
    """Check whether a candidate trajectory used all required tools.

    Parameters
    ----------
    candidate : Trace
        The candidate trajectory to evaluate.
    required_tools : list[str]
        Required tool names (from :func:`extract_required_tools`).

    Returns
    -------
    tuple[float, list[str]]
        ``(coverage_ratio, list_of_missing_tools)``.
        ``coverage_ratio`` is 1.0 when every required tool is present.
    """
    if not required_tools:
        return 1.0, []

    traj_tools: Set[str] = set()
    for state in candidate.states.values():
        tool = state.tool_used
        if tool and tool != "llm":
            base_tool = tool.split("[")[0].strip()
            traj_tools.add(base_tool)

    missing = [t for t in required_tools if t not in traj_tools]
    coverage = (len(required_tools) - len(missing)) / len(required_tools)
    return coverage, missing


def _split_file_paths(raw: str) -> List[str]:
    """Split a potentially comma-separated file path string into individual paths.

    Tools like ``multi_replace_string_in_file`` store file_path as a
    comma-separated list (e.g. ``"main.py,utils/math_ops.py"``).  This
    helper splits them into individual paths.
    """
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def extract_required_files(ground_truth: Trace) -> Set[str]:
    """Extract normalised file paths that are *edited* in the ground-truth trace.

    Only edit operations on **existing source files** are considered
    (replace_string_in_file, multi_replace_string_in_file, apply_patch,
    edit_file).  Files that are *created* (create_file) are excluded —
    those are typically test/reproduction scripts whose names vary across
    trajectories.  The function returns the source files that a correct
    trajectory *must* modify.
    """
    _EDIT_TOOLS = {
        "replace_string_in_file", "multi_replace_string_in_file",
        "apply_patch", "edit_file",
    }
    files: Set[str] = set()
    for state in ground_truth.states.values():
        tool = state.tool_used
        if tool and tool in _EDIT_TOOLS:
            fp = (state.file_path or "").strip()
            if not fp:
                entry = getattr(state, "log_entry", None)
                if entry and entry.args:
                    fp = entry.args.get("filePath") or entry.args.get("path", "")
            if fp:
                for single_fp in _split_file_paths(fp):
                    normed = _normalize_file_path(single_fp)
                    if normed:
                        files.add(normed)
    return files


def check_file_coverage(
    candidate: Trace, required_files: Set[str]
) -> Tuple[float, List[str]]:
    """Check what fraction of required *files* the candidate edits.

    Parameters
    ----------
    candidate : Trace
        The candidate trajectory.
    required_files : set[str]
        Normalised file paths from :func:`extract_required_files`.

    Returns
    -------
    tuple[float, list[str]]
        ``(coverage_ratio, list_of_missing_files)``.
    """
    if not required_files:
        return 1.0, []

    # On the candidate side, include create_file too — some agents create
    # a new file rather than editing the existing one.
    _EDIT_TOOLS = {
        "replace_string_in_file", "multi_replace_string_in_file",
        "create_file", "apply_patch", "edit_file",
    }
    cand_files: Set[str] = set()
    for state in candidate.states.values():
        tool = state.tool_used
        if tool and tool in _EDIT_TOOLS:
            fp = (state.file_path or "").strip()
            if not fp:
                entry = getattr(state, "log_entry", None)
                if entry and entry.args:
                    fp = entry.args.get("filePath") or entry.args.get("path", "")
            if fp:
                for single_fp in _split_file_paths(fp):
                    normed = _normalize_file_path(single_fp)
                    if normed:
                        cand_files.add(normed)

    missing = [f for f in required_files if f not in cand_files]
    coverage = (len(required_files) - len(missing)) / len(required_files)
    return coverage, missing


def _normalize_file_path(path: str) -> str:
    """Normalise a file path for comparison (strip workspace prefixes, lowercase)."""
    if not path:
        return ""
    path = path.replace("\\", "/").strip()
    # Strip workspace-style prefixes
    for prefix in ("/workspace/", "/workspaces/", "/home/", "/tmp/", "workspace/", "workspaces/"):
        if path.lower().startswith(prefix):
            path = path[len(prefix):]
            break
    # Strip leading slash
    path = path.lstrip("/")
    return path.lower()


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

@dataclass
class _PathMatch:
    path_idx: int
    gt_seq: List[State]
    matched: int
    total: int
    coverage: float
    terminal: bool
    perfect: bool
    matched_idx: List[int]


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------

# Stage-importance ordering for severity calculations
_STAGE_ORDER = ["exploration", "implementation", "verification", "orchestration"]


def _compute_verdict(
    coverage: float,
    weighted_score: float,
    stage_completeness: float,
    terminal_match: bool,
    coherence: float,
    bottleneck_coverage: float = 0.0,
) -> str:
    """Compute a pass/fail verdict from match metrics.

    The verdict factors in bottleneck coverage: if the candidate
    completely missed an entire GT stage (bottleneck = 0), the verdict
    is capped — a trace that skips verification entirely cannot get
    PASS even if other metrics look strong.
    """
    # If any GT stage has 0% coverage, the trace missed something critical
    if bottleneck_coverage == 0.0 and stage_completeness < 1.0:
        if weighted_score >= 60:
            return "LIKELY PASS" if terminal_match else "UNCERTAIN"
        if weighted_score >= 40:
            return "LIKELY FAIL"
        return "FAIL"

    if weighted_score >= 75 and stage_completeness >= 0.75:
        return "PASS"
    if weighted_score >= 60 and stage_completeness >= 0.5:
        return "LIKELY PASS" if terminal_match else "UNCERTAIN"
    if weighted_score < 30 or stage_completeness < 0.25:
        return "FAIL"
    if weighted_score < 50:
        return "LIKELY FAIL"
    return "UNCERTAIN"


def _compute_quality_tier(
    verdict: str,
    coverage: float,
    coherence: float,
    stage_completeness: float,
    terminal_match: bool,
    workflow_similarity: float,
) -> str:
    """Classify a trajectory into a quality tier."""
    is_pass = verdict in ("PASS", "LIKELY PASS")

    if is_pass:
        # Ideal: high coverage + high coherence + strong stage completeness
        if coverage >= 70 and coherence >= 0.6 and stage_completeness >= 0.75:
            return "ideal"
        # Solid: generally good but not top-tier
        if coverage >= 50 and coherence >= 0.4 and stage_completeness >= 0.5:
            return "solid"
        # Lucky: outcome matched but approach was weak
        return "lucky"

    # Failing tiers
    if coverage >= 40 and stage_completeness >= 0.5:
        return "partial_fail"
    return "off_track"


def _compute_quality_score(
    coverage: float,
    coherence: float,
    stage_completeness: float,
    workflow_similarity: float,
    f1: float,
    passed: Optional[bool] = None,
) -> int:
    """Compute a 0–100 composite score for ranking within a cohort.

    Includes a 10% outcome component so that passing trajectories
    naturally score higher than failing ones with identical approach
    quality, while still allowing near-miss failures to outscore
    lucky passes.
    """
    outcome = 100.0 if passed is True else (0.0 if passed is False else 50.0)
    score = (
        0.25 * coverage
        + 0.25 * (coherence * 100.0)
        + 0.18 * (stage_completeness * 100.0)
        + 0.12 * (workflow_similarity * 100.0)
        + 0.10 * f1
        + 0.10 * outcome
    )
    return max(0, min(100, int(round(score))))


def _build_failure_reasons(
    metrics: MatchMetrics,
    divergence_index: Optional[int],
    gt_seq_len: int,
    stage_cov: Dict[str, StageCoverageDetail],
) -> List[FailureReason]:
    """Build a list of specific, actionable failure reasons."""
    reasons: List[FailureReason] = []

    # 1. Early divergence
    if divergence_index is not None and gt_seq_len > 0:
        progress = divergence_index / gt_seq_len
        if progress < 0.4:
            reasons.append(FailureReason(
                reason="early_divergence",
                detail=f"Diverged at step {divergence_index + 1} of {gt_seq_len} ground-truth steps ({progress:.0%} through)",
                severity="critical",
            ))

    # 2. Missing verification
    v_detail = stage_cov.get("verification")
    if v_detail and v_detail.total > 0 and v_detail.percent < 25:
        reasons.append(FailureReason(
            reason="missing_verification",
            detail=f"Verification coverage is {v_detail.percent:.0f}% ({v_detail.matched}/{v_detail.total} steps). Agent may not have tested its changes.",
            severity="high",
        ))

    # 3. Incomplete implementation
    i_detail = stage_cov.get("implementation")
    if i_detail and i_detail.total > 0 and i_detail.percent < 50:
        reasons.append(FailureReason(
            reason="incomplete_implementation",
            detail=f"Only {i_detail.percent:.0f}% of implementation steps covered ({i_detail.matched}/{i_detail.total})",
            severity="high" if i_detail.percent < 25 else "medium",
        ))

    # 4. Low coherence (thrashing)
    if metrics.coherence_score < 0.35:
        reasons.append(FailureReason(
            reason="trajectory_thrashing",
            detail=f"Coherence score is {metrics.coherence_score:.2f} — agent appears to backtrack heavily between stages",
            severity="medium",
        ))

    # 5. Low overall coverage
    if metrics.coverage_percent < 30 and not reasons:
        reasons.append(FailureReason(
            reason="low_coverage",
            detail=f"Only {metrics.coverage_percent:.0f}% of ground-truth steps matched. Fundamentally different approach.",
            severity="high",
        ))

    # Sort by severity
    _SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    reasons.sort(key=lambda r: _SEV_ORDER.get(r.severity, 9))
    return reasons


def _build_strengths(
    stage_cov: Dict[str, StageCoverageDetail],
    metrics: MatchMetrics,
) -> List[str]:
    """Identify what the agent did well."""
    strengths: List[str] = []
    for stage_name in _STAGE_ORDER:
        detail = stage_cov.get(stage_name)
        if detail and detail.total > 0 and detail.percent >= 80:
            strengths.append(f"Strong {stage_name} phase ({detail.percent:.0f}% covered)")
    if metrics.coherence_score >= 0.7:
        strengths.append("Clean forward progression (high coherence)")
    if metrics.terminal_state_match:
        strengths.append("Reached correct terminal state")
    return strengths


def _build_divergence_point(
    result: "MatchResult",
    candidate: Trace,
    ground_truth: Trace,
) -> Optional[DivergencePoint]:
    """Find the divergence point between candidate and ground truth."""
    if result.divergence_index is None:
        return None

    gt_paths = _enumerate_paths(ground_truth)
    if not gt_paths:
        return None

    # Use the best-path index from the match
    best_idx = result.metrics.best_path_index
    if best_idx >= len(gt_paths):
        best_idx = 0
    gt_seq = gt_paths[best_idx]

    div_idx = result.divergence_index
    if div_idx >= len(gt_seq):
        return None

    gt_state = gt_seq[div_idx]
    expected = gt_state.tool_used or "unknown"
    if gt_state.file_path:
        expected += f" on {gt_state.file_path}"

    # Find what candidate actually did around this step
    cand_desc = "(no matching candidate step)"
    if result.alignment:
        # Find the last matched step, next unmatched is the divergence
        for align in result.alignment:
            if not align.matched:
                cand_state = candidate.states.get(align.candidate_state_id)
                if cand_state:
                    tool = cand_state.tool_used or "unknown action"
                    fp = cand_state.file_path
                    cand_desc = f"{tool} on {fp}" if fp else tool
                break

    return DivergencePoint(
        step=div_idx + 1,
        description=cand_desc,
        expected_next=expected,
    )


def _state_summary(state: State) -> Dict[str, str]:
    """Build a compact summary dict for a state."""
    return {
        "tool": getattr(state, "tool_used", "") or "",
        "file_path": getattr(state, "file_path", "") or "",
        "intent_stage": getattr(state, "intent_stage", "") or "",
        "resulting_state": getattr(state, "resulting_state", "") or "",
    }


def _build_divergence_points(
    result: "MatchResult",
    candidate: Trace,
    ground_truth: Trace,
) -> List[DivergenceSegment]:
    """Build all divergence segments where candidate missed GT states."""
    if not result.missing_indexes:
        return []

    gt_paths = _enumerate_paths(ground_truth)
    if not gt_paths:
        return []

    best_idx = result.metrics.best_path_index
    if best_idx >= len(gt_paths):
        best_idx = 0
    gt_seq = gt_paths[best_idx]

    # Build candidate step-indexed list for cross-referencing
    cand_ordered = sorted(candidate.states.values(), key=lambda s: s.step)
    # Filter to actionable states only — same filter as run() uses
    cand_ordered = [s for s in cand_ordered
                    if not (hasattr(s, "log_entry") and s.log_entry
                            and getattr(s.log_entry, "kind", None) == "request")]

    # Group consecutive missing GT indexes into segments
    segments: List[DivergenceSegment] = []
    missing_sorted = sorted(result.missing_indexes)

    seg_start = missing_sorted[0]
    seg_end = seg_start

    for idx in missing_sorted[1:]:
        if idx == seg_end + 1:
            seg_end = idx
        else:
            # Flush current segment
            segments.append(_build_one_segment(
                seg_start, seg_end, gt_seq, cand_ordered, result,
            ))
            seg_start = idx
            seg_end = idx

    # Flush last segment
    segments.append(_build_one_segment(
        seg_start, seg_end, gt_seq, cand_ordered, result,
    ))

    return segments


def _build_one_segment(
    start: int,
    end: int,
    gt_seq: List[State],
    cand_ordered: List[State],
    result: "MatchResult",
) -> DivergenceSegment:
    """Build a single DivergenceSegment from a range of missed GT indexes."""
    expected = []
    stage_counts: Dict[str, int] = {}
    for i in range(start, min(end + 1, len(gt_seq))):
        s = gt_seq[i]
        expected.append(_state_summary(s))
        stg = getattr(s, "intent_stage", "") or ""
        if stg:
            stage_counts[stg] = stage_counts.get(stg, 0) + 1

    # Dominant stage in this gap
    stage_context = max(stage_counts, key=stage_counts.get) if stage_counts else ""

    # What was the candidate doing around these steps?
    # Use alignment to find unmatched candidate steps in this range
    candidate_activity: List[Dict[str, str]] = []
    matched_cand_steps = {a.candidate_step for a in result.alignment if a.matched}
    # Find candidate steps roughly corresponding to this GT range
    # (proportional mapping: GT steps start..end maps to candidate step range)
    gt_len = max(len(gt_seq), 1)
    cand_len = len(cand_ordered)
    approx_start = int(start / gt_len * cand_len)
    approx_end = min(int((end + 1) / gt_len * cand_len) + 1, cand_len)
    for ci in range(max(0, approx_start), min(approx_end, cand_len)):
        if ci not in matched_cand_steps:
            s = cand_ordered[ci]
            candidate_activity.append({
                "tool": getattr(s, "tool_used", "") or "",
                "file_path": getattr(s, "file_path", "") or "",
                "intent_stage": getattr(s, "intent_stage", "") or "",
            })

    return DivergenceSegment(
        start_step=start + 1,
        end_step=end + 1,
        expected_states=expected,
        candidate_activity=candidate_activity,
        stage_context=stage_context,
    )


def _build_stage_comparison(
    result: "MatchResult",
    candidate: Trace,
    ground_truth: Trace,
) -> Tuple[Dict[str, StageComparison], bool]:
    """Build detailed per-stage comparison and compute stage-order match.

    Returns (stage_comparison_dict, stage_order_match_bool).
    """
    gt_paths = _enumerate_paths(ground_truth)
    if not gt_paths:
        return {}, True

    best_idx = result.metrics.best_path_index
    if best_idx >= len(gt_paths):
        best_idx = 0
    gt_seq = gt_paths[best_idx]
    matched_gt_set = set(result.matched_indexes)

    # Candidate states ordered by step
    cand_ordered = sorted(candidate.states.values(), key=lambda s: s.step)
    # Filter to actionable states only — same filter as run() uses
    cand_ordered = [s for s in cand_ordered
                    if not (hasattr(s, "log_entry") and s.log_entry
                            and getattr(s.log_entry, "kind", None) == "request")]
    matched_cand_ids = {
        a.candidate_state_id for a in result.alignment if a.matched
    }

    # Collect per-stage data
    stage_data: Dict[str, Dict[str, list]] = {}
    for stg in _STAGE_ORDER:
        stage_data[stg] = {
            "expected": [], "matched": [], "missing": [], "extra": [],
            "matched_gt_positions": [],
        }

    # GT states per stage
    for i, s in enumerate(gt_seq):
        stg = getattr(s, "intent_stage", "") or ""
        if stg not in stage_data:
            continue
        summary = {"tool": s.tool_used or "", "file_path": s.file_path or "",
                    "resulting_state": getattr(s, "resulting_state", "") or ""}
        stage_data[stg]["expected"].append(summary)
        if i in matched_gt_set:
            stage_data[stg]["matched"].append(summary)
            stage_data[stg]["matched_gt_positions"].append(i)
        else:
            stage_data[stg]["missing"].append(summary)

    # Candidate extra steps per stage (not matched to any GT)
    for s in cand_ordered:
        stg = getattr(s, "intent_stage", "") or ""
        if stg not in stage_data:
            continue
        if s.state_id not in matched_cand_ids:
            stage_data[stg]["extra"].append({
                "tool": s.tool_used or "",
                "file_path": s.file_path or "",
                "resulting_state": getattr(s, "resulting_state", "") or "",
            })

    # Build StageComparison objects
    comparisons: Dict[str, StageComparison] = {}
    for stg in _STAGE_ORDER:
        d = stage_data[stg]
        gt_count = len(d["expected"])
        cand_count_in_stage = sum(
            1 for s in cand_ordered
            if (getattr(s, "intent_stage", "") or "") == stg
        )
        effort = (cand_count_in_stage / gt_count) if gt_count > 0 else 0.0

        # Check ordering: are matched GT positions in ascending order?
        positions = d["matched_gt_positions"]
        ordering_ok = all(positions[i] < positions[i + 1] for i in range(len(positions) - 1))

        comparisons[stg] = StageComparison(
            expected_steps=d["expected"],
            matched_steps=d["matched"],
            missing_steps=d["missing"],
            extra_steps=d["extra"],
            ordering_preserved=ordering_ok,
            effort_ratio=effort,
        )

    # Stage-order match: compare first-occurrence order of stages
    # Ignore orchestration — it can appear between any stages without penalty
    def _first_occurrences(states: List[State]) -> List[str]:
        seen: set = set()
        order: List[str] = []
        for s in states:
            stg = getattr(s, "intent_stage", "") or ""
            if stg and stg != "orchestration" and stg not in seen:
                seen.add(stg)
                order.append(stg)
        return order

    gt_order = _first_occurrences(gt_seq)
    cand_order = _first_occurrences(cand_ordered)
    # Check that the candidate's relative order is a subsequence of GT order
    # (allows candidate to have fewer stages, but no reordering)
    gi = 0
    for stg in cand_order:
        while gi < len(gt_order) and gt_order[gi] != stg:
            gi += 1
        if gi >= len(gt_order):
            break
        gi += 1
    stage_order_match = gi <= len(gt_order) and all(
        stg in gt_order for stg in cand_order
    )

    return comparisons, stage_order_match


def _build_inefficiency_report(
    candidate: Trace,
    result: "MatchResult",
    ground_truth: Trace,
) -> InefficiencyReport:
    """Detect inefficiency patterns in the candidate trajectory.

    All detections are GT-aware: patterns that also appear in the ground truth
    best-path are considered task requirements, not inefficiencies.
    """
    cand_ordered = sorted(candidate.states.values(), key=lambda s: s.step)
    # Filter to actionable states only — same filter as run() uses
    cand_ordered = [s for s in cand_ordered
                    if not (hasattr(s, "log_entry") and s.log_entry
                            and getattr(s.log_entry, "kind", None) == "request")]
    # Also filter to labeled states only — matches the fingerprint computation
    # so that index-based step numbers align with fingerprint block positions.
    cand_ordered = [s for s in cand_ordered if getattr(s, "intent_stage", "")]

    # ── Extract GT best path for comparison ──
    gt_paths = _enumerate_paths(ground_truth)
    gt_seq: List[State] = []
    if gt_paths:
        best_idx = result.metrics.best_path_index
        if best_idx >= len(gt_paths):
            best_idx = 0
        gt_seq = gt_paths[best_idx]

    # ── Helper: find retry-loop signatures in a state sequence ──
    def _find_retry_sigs(states: List[State]) -> set:
        """Return set of (tool, file, stage) tuples that appear ≥3 times
        consecutively — these are expected retry patterns."""
        sigs: set = set()
        if len(states) < 3:
            return sigs
        run_start = 0
        for i in range(1, len(states)):
            prev_sig = (states[i - 1].tool_used or "", states[i - 1].file_path or "",
                        getattr(states[i - 1], "intent_stage", "") or "")
            curr_sig = (states[i].tool_used or "", states[i].file_path or "",
                        getattr(states[i], "intent_stage", "") or "")
            if curr_sig == prev_sig and curr_sig != ("", "", ""):
                continue
            else:
                if i - run_start >= 3:
                    sigs.add((states[run_start].tool_used or "",
                              states[run_start].file_path or "",
                              getattr(states[run_start], "intent_stage", "") or ""))
                run_start = i
        if len(states) - run_start >= 3:
            sigs.add((states[run_start].tool_used or "",
                      states[run_start].file_path or "",
                      getattr(states[run_start], "intent_stage", "") or ""))
        return sigs

    # ── Helper: find backtrack transitions in a state sequence ──
    def _find_backtrack_transitions(states: List[State]) -> set:
        """Return set of (from_stage, to_stage) that are backtracks in the GT."""
        transitions: set = set()
        for i in range(1, len(states)):
            prev_stg = getattr(states[i - 1], "intent_stage", "") or ""
            curr_stg = getattr(states[i], "intent_stage", "") or ""
            if prev_stg and curr_stg and _classify_transition(prev_stg, curr_stg) == "backtrack":
                transitions.add((prev_stg, curr_stg))
        return transitions

    # GT patterns (task requirements, not inefficiencies)
    gt_retry_sigs = _find_retry_sigs(gt_seq)
    gt_backtrack_transitions = _find_backtrack_transitions(gt_seq)

    # 1. Retry loops: only flag if the GT doesn't have the same retry pattern
    retry_loops: List[RetryLoop] = []
    if len(cand_ordered) >= 3:
        run_start = 0
        for i in range(1, len(cand_ordered)):
            prev = cand_ordered[i - 1]
            curr = cand_ordered[i]
            prev_sig = (prev.tool_used or "", prev.file_path or "", getattr(prev, "intent_stage", "") or "")
            curr_sig = (curr.tool_used or "", curr.file_path or "", getattr(curr, "intent_stage", "") or "")
            if curr_sig == prev_sig and curr_sig != ("", "", ""):
                continue
            else:
                run_len = i - run_start
                if run_len >= 3:
                    s = cand_ordered[run_start]
                    sig = (s.tool_used or "", s.file_path or "", getattr(s, "intent_stage", "") or "")
                    if sig not in gt_retry_sigs:
                        retry_loops.append(RetryLoop(
                            start_step=run_start + 1,
                            end_step=i,
                            tool=s.tool_used or "",
                            file_path=s.file_path or "",
                            count=run_len,
                        ))
                run_start = i
        # Check last run
        run_len = len(cand_ordered) - run_start
        if run_len >= 3:
            s = cand_ordered[run_start]
            sig = (s.tool_used or "", s.file_path or "", getattr(s, "intent_stage", "") or "")
            if sig not in gt_retry_sigs:
                retry_loops.append(RetryLoop(
                    start_step=run_start + 1,
                    end_step=len(cand_ordered),
                    tool=s.tool_used or "",
                    file_path=s.file_path or "",
                    count=run_len,
                ))

    # 2. Backtracks: only flag transitions that the GT doesn't also make
    backtracks: List[Backtrack] = []
    for i in range(1, len(cand_ordered)):
        prev_stg = getattr(cand_ordered[i - 1], "intent_stage", "") or ""
        curr_stg = getattr(cand_ordered[i], "intent_stage", "") or ""
        if prev_stg and curr_stg:
            transition_type = _classify_transition(prev_stg, curr_stg)
            if transition_type == "backtrack":
                if (prev_stg, curr_stg) not in gt_backtrack_transitions:
                    backtracks.append(Backtrack(
                        step=i + 1,
                        from_stage=prev_stg,
                        to_stage=curr_stg,
                    ))

    # 3. Redundant steps: unmatched candidate steps with same (tool, file)
    # as a nearby matched step (within ±3 steps)
    matched_cand_steps = {a.candidate_step for a in result.alignment if a.matched}
    matched_sigs: Dict[int, Tuple[str, str]] = {}
    for a in result.alignment:
        if a.matched:
            s = candidate.states.get(a.candidate_state_id)
            if s:
                matched_sigs[a.candidate_step] = (s.tool_used or "", s.file_path or "")

    redundant: List[RedundantStep] = []
    for i, s in enumerate(cand_ordered):
        if i in matched_cand_steps:
            continue
        sig = (s.tool_used or "", s.file_path or "")
        if sig == ("", ""):
            continue
        for offset in range(-3, 4):
            neighbor = i + offset
            if neighbor in matched_sigs and matched_sigs[neighbor] == sig:
                redundant.append(RedundantStep(
                    step=i + 1,
                    tool=s.tool_used or "",
                    file_path=s.file_path or "",
                ))
                break

    # 4. Unnecessary exploration: exploration after implementation started,
    # where the GT has no exploration at the corresponding relative position
    unnecessary: List[UnnecessaryExploration] = []
    impl_started = False

    gt_exploration_after_impl: set = set()
    if gt_seq:
        gt_impl_started = False
        for gi, gs in enumerate(gt_seq):
            stg = getattr(gs, "intent_stage", "") or ""
            if stg == "implementation":
                gt_impl_started = True
            if gt_impl_started and stg == "exploration":
                gt_exploration_after_impl.add(gi)

    for i, s in enumerate(cand_ordered):
        stg = getattr(s, "intent_stage", "") or ""
        if stg == "implementation":
            impl_started = True
        if impl_started and stg == "exploration":
            # Check if GT has exploration at proportional position
            if gt_seq:
                proportional_gt_idx = int(i / max(len(cand_ordered), 1) * len(gt_seq))
                proportional_gt_idx = min(proportional_gt_idx, len(gt_seq) - 1)
                if proportional_gt_idx not in gt_exploration_after_impl:
                    unnecessary.append(UnnecessaryExploration(
                        step=i + 1,
                        tool=s.tool_used or "",
                        file_path=s.file_path or "",
                    ))

    # 5. Cyclic patterns: multi-step repeating sequences (length 2-4, ≥2 reps)
    #    Only flag cycles not present in the GT.
    def _state_sig(s: State) -> str:
        return f"{s.tool_used or ''}|{s.file_path or ''}|{getattr(s, 'intent_stage', '') or ''}"

    def _find_cycles_in_seq(states: List[State]) -> List[tuple]:
        """Find all (start, pattern_len, reps, sig_tuple) cycles in a state list."""
        sigs = [_state_sig(s) for s in states]
        n = len(sigs)
        found: List[tuple] = []
        for plen in range(2, 5):           # pattern lengths 2, 3, 4
            i = 0
            while i <= n - plen * 2:        # need at least 2 repetitions
                pattern = tuple(sigs[i:i + plen])
                reps = 1
                j = i + plen
                while j + plen <= n and tuple(sigs[j:j + plen]) == pattern:
                    reps += 1
                    j += plen
                if reps >= 2:
                    found.append((i, plen, reps, pattern))
                    i = j  # skip past this cycle
                else:
                    i += 1
        return found

    gt_cycle_sigs = {c[3] for c in _find_cycles_in_seq(gt_seq)} if gt_seq else set()

    cyclic_patterns: List[CyclicPattern] = []
    for (start_idx, plen, reps, sig_tuple) in _find_cycles_in_seq(cand_ordered):
        if sig_tuple not in gt_cycle_sigs:
            # Skip if all steps in the pattern are identical — that's a retry
            # loop, already detected above (e.g. (A, A) ×3 = AAAAAA).
            if len(set(sig_tuple)) <= 1:
                continue
            # Build human-readable signature: "tool(file)" per step
            readable_sig = []
            for sig in sig_tuple:
                parts = sig.split('|')
                tool = parts[0] or '?'
                fpath = parts[1].rsplit('/', 1)[-1] if parts[1] else ''
                readable_sig.append(f"{tool}({fpath})" if fpath else tool)
            cyclic_patterns.append(CyclicPattern(
                start_step=start_idx + 1,
                end_step=start_idx + plen * reps,
                pattern_length=plen,
                repetitions=reps,
                pattern_signature=readable_sig,
            ))

    # Deduplicate across categories — a step should only be counted once
    wasted_indices: set = set()

    # Retry loops: excess steps beyond the first 2 in each run
    for r in retry_loops:
        # start_step/end_step are 1-indexed; add the excess indices
        for idx in range(r.start_step - 1 + 2, r.end_step):
            wasted_indices.add(idx)

    # Redundant steps
    for rs in redundant:
        wasted_indices.add(rs.step - 1)  # step is 1-indexed

    # Unnecessary explorations
    for ue in unnecessary:
        wasted_indices.add(ue.step - 1)  # step is 1-indexed

    # Cyclic patterns: excess repetitions beyond the first occurrence
    for cp in cyclic_patterns:
        # The first occurrence of the pattern is not wasted; excess reps are
        first_end = (cp.start_step - 1) + cp.pattern_length
        for idx in range(first_end, cp.end_step):
            wasted_indices.add(idx)

    total_wasted = len(wasted_indices)

    # ── Severity score: fraction of trajectory steps that were wasted ──
    n_steps = len(cand_ordered)
    severity_score = (total_wasted / n_steps) if n_steps > 0 else 0.0

    # ── Token cost of wasted steps ──
    total_in_tok = 0
    total_out_tok = 0
    wasted_in_tok = 0
    wasted_out_tok = 0
    for i, s in enumerate(cand_ordered):
        meta = getattr(s, "metadata", {}) or {}
        i_tok = int(meta.get("input_tokens", 0) or 0)
        o_tok = int(meta.get("output_tokens", 0) or 0)
        total_in_tok += i_tok
        total_out_tok += o_tok
        if i in wasted_indices:
            wasted_in_tok += i_tok
            wasted_out_tok += o_tok

    # ── Per-tool inefficiency breakdown ──
    # Build per-category index sets for priority-based assignment.
    # Each wasted index is assigned to exactly ONE category so that
    # per-tool totals sum to total_wasted_steps with no double-counting.
    retry_idx_set: set = set()
    for r in retry_loops:
        for idx in range(r.start_step - 1 + 2, r.end_step):
            retry_idx_set.add(idx)

    cycle_idx_set: set = set()
    for cp in cyclic_patterns:
        first_end = (cp.start_step - 1) + cp.pattern_length
        for idx in range(first_end, cp.end_step):
            cycle_idx_set.add(idx)

    redundant_idx_set = {rs.step - 1 for rs in redundant}
    unnecessary_idx_set = {ue.step - 1 for ue in unnecessary}

    # Backtracks: count per tool separately (backtracks are transition-level
    # signals, not wasted steps, so they don't contribute to total_wasted).
    backtrack_by_tool: Dict[str, int] = {}
    for b in backtracks:
        if 0 <= b.step - 1 < len(cand_ordered):
            t = cand_ordered[b.step - 1].tool_used or "(unknown)"
            backtrack_by_tool[t] = backtrack_by_tool.get(t, 0) + 1

    tool_stats: Dict[str, Dict[str, int]] = {}
    for idx in wasted_indices:
        tool = cand_ordered[idx].tool_used or "(unknown)"
        if tool not in tool_stats:
            tool_stats[tool] = {"retries": 0, "backtracks": 0, "cycles": 0,
                                "redundant": 0, "unnecessary": 0, "total_wasted": 0}
        # Priority: retry > cycle > redundant > unnecessary
        if idx in retry_idx_set:
            tool_stats[tool]["retries"] += 1
        elif idx in cycle_idx_set:
            tool_stats[tool]["cycles"] += 1
        elif idx in redundant_idx_set:
            tool_stats[tool]["redundant"] += 1
        elif idx in unnecessary_idx_set:
            tool_stats[tool]["unnecessary"] += 1
        tool_stats[tool]["total_wasted"] += 1

    # Merge backtrack counts into tool_stats (without changing total_wasted)
    for t, cnt in backtrack_by_tool.items():
        if t not in tool_stats:
            tool_stats[t] = {"retries": 0, "backtracks": 0, "cycles": 0,
                             "redundant": 0, "unnecessary": 0, "total_wasted": 0}
        tool_stats[t]["backtracks"] = cnt

    per_tool = sorted(
        [ToolInefficiency(tool=t, **counts) for t, counts in tool_stats.items()],
        key=lambda x: x.total_wasted, reverse=True,
    )

    return InefficiencyReport(
        retry_loops=retry_loops,
        backtracks=backtracks,
        redundant_steps=redundant,
        unnecessary_explorations=unnecessary,
        cyclic_patterns=cyclic_patterns,
        retry_loop_count=len(retry_loops),
        backtrack_count=len(backtracks),
        redundant_step_count=len(redundant),
        unnecessary_exploration_count=len(unnecessary),
        cyclic_pattern_count=len(cyclic_patterns),
        total_wasted_steps=total_wasted,
        severity_score=severity_score,
        wasted_input_tokens=wasted_in_tok,
        wasted_output_tokens=wasted_out_tok,
        total_input_tokens=total_in_tok,
        total_output_tokens=total_out_tok,
        per_tool_breakdown=per_tool,
    )


def _build_quality_signals(
    verdict: str,
    metrics: MatchMetrics,
    inefficiencies: InefficiencyReport,
    stage_cov: Dict[str, StageCoverageDetail],
    stage_comparison: Dict[str, StageComparison],
    stage_order_match: bool,
    passed: Optional[bool] = None,
) -> List[QualitySignal]:
    """Derive high-level quality indicator signals."""
    signals: List[QualitySignal] = []
    is_pass = verdict in ("PASS", "LIKELY PASS")
    is_fail = verdict in ("FAIL", "LIKELY FAIL")
    # Use actual task outcome when available, fall back to verdict
    task_passed = passed if passed is not None else is_pass
    task_failed = (not passed) if passed is not None else is_fail

    # 1. Inefficient path despite success
    if task_passed and inefficiencies.total_wasted_steps > 0:
        wasted_ratio = (
            inefficiencies.total_wasted_steps / max(metrics.candidate_states, 1)
        )
        if wasted_ratio > 0.2 or metrics.coherence_score < 0.5:
            evidence = []
            if inefficiencies.retry_loop_count:
                evidence.append(f"{inefficiencies.retry_loop_count} retry loop(s)")
            if inefficiencies.cyclic_pattern_count:
                evidence.append(f"{inefficiencies.cyclic_pattern_count} cyclic pattern(s)")
            if inefficiencies.redundant_step_count:
                evidence.append(f"{inefficiencies.redundant_step_count} redundant step(s)")
            if inefficiencies.unnecessary_exploration_count:
                evidence.append(f"{inefficiencies.unnecessary_exploration_count} unnecessary exploration(s)")
            if metrics.coherence_score < 0.5:
                evidence.append(f"Low coherence: {metrics.coherence_score:.2f}")
            signals.append(QualitySignal(
                signal_type="inefficient_path_despite_success",
                description="Trajectory reached the correct outcome but took an inefficient path with wasted steps.",
                severity="warning",
                evidence=evidence,
            ))

    # 2. Missing verification
    v_detail = stage_cov.get("verification")
    if v_detail and v_detail.total > 0 and v_detail.percent < 25:
        if task_failed:
            signals.append(QualitySignal(
                signal_type="missing_verification",
                description="Failure likely caused by skipping or insufficiently verifying changes.",
                severity="critical",
                evidence=[
                    f"Verification coverage: {v_detail.percent:.0f}% ({v_detail.matched}/{v_detail.total})",
                    f"Missing {v_detail.total - v_detail.matched} verification step(s)",
                ],
            ))
        else:
            signals.append(QualitySignal(
                signal_type="low_verification",
                description="Trajectory succeeded but covered very few of the ground-truth verification steps.",
                severity="warning",
                evidence=[
                    f"Verification coverage: {v_detail.percent:.0f}% ({v_detail.matched}/{v_detail.total})",
                    f"Missing {v_detail.total - v_detail.matched} verification step(s)",
                ],
            ))

    # 3. Stage ordering differs from GT
    if not stage_order_match:
        gt_stages = [stg for stg in _STAGE_ORDER if stg != "orchestration" and stage_cov.get(stg) and stage_cov[stg].total > 0]
        cand_stages_ordered = []
        for stg in _STAGE_ORDER:
            if stg == "orchestration":
                continue
            sc = stage_comparison.get(stg)
            if sc and (sc.matched_steps or sc.extra_steps):
                cand_stages_ordered.append(stg)
        if task_failed:
            signals.append(QualitySignal(
                signal_type="stage_order_differs",
                description="Failure may be related to performing workflow stages in a different order than the ground truth.",
                severity="critical",
                evidence=[
                    f"Expected relative order: {' → '.join(gt_stages)}",
                    f"Actual relative order: {' → '.join(cand_stages_ordered)}",
                ],
            ))
        else:
            signals.append(QualitySignal(
                signal_type="stage_order_differs",
                description="Trajectory succeeded but followed a different stage order than the ground truth.",
                severity="info",
                evidence=[
                    f"GT stage order: {' → '.join(gt_stages)}",
                    f"Candidate stage order: {' → '.join(cand_stages_ordered)}",
                ],
            ))

    # 4. Excessive exploration
    e_comp = stage_comparison.get("exploration")
    if e_comp and e_comp.effort_ratio > 2.0:
        signals.append(QualitySignal(
            signal_type="excessive_exploration",
            description="Agent spent significantly more time exploring than the ground truth requires.",
            severity="warning",
            evidence=[
                f"Exploration effort ratio: {e_comp.effort_ratio:.1f}× (candidate vs GT)",
                f"Extra exploration steps: {len(e_comp.extra_steps)}",
            ],
        ))

    # 5. Missing planning/orchestration
    o_detail = stage_cov.get("orchestration")
    if o_detail and o_detail.total > 0 and o_detail.matched == 0:
        signals.append(QualitySignal(
            signal_type="missing_planning",
            description="Agent skipped the planning/orchestration steps present in the ground truth.",
            severity="info",
            evidence=[
                f"GT has {o_detail.total} orchestration step(s), none matched",
            ],
        ))

    # 6. Clean ideal pass — positive signal for high-quality passes
    if task_passed and metrics.coherence_score >= 0.7 and metrics.coverage_percent >= 70:
        if metrics.stage_completeness >= 0.75 and inefficiencies.total_wasted_steps == 0:
            signals.append(QualitySignal(
                signal_type="clean_ideal_pass",
                description="Trajectory followed the ground-truth workflow closely with no wasted steps — an ideal execution.",
                severity="info",
                evidence=[
                    f"Coverage: {metrics.coverage_percent:.0f}%",
                    f"Coherence: {metrics.coherence_score:.2f}",
                    f"Stage completeness: {metrics.stage_completeness:.0%}",
                ],
            ))

    # 7. Incomplete coverage — trajectory that didn't cover all GT states
    if task_failed and metrics.coverage_percent >= 40 and metrics.stage_completeness >= 0.5:
        evidence = [f"Coverage: {metrics.coverage_percent:.0f}%"]
        for stg in reversed(_STAGE_ORDER):
            d = stage_cov.get(stg)
            if d and d.matched > 0:
                evidence.append(f"Reached {stg} stage ({d.matched}/{d.total} steps)")
                break
        signals.append(QualitySignal(
            signal_type="incomplete_coverage",
            description="Trajectory covered a significant portion of the ground truth but did not match all expected steps.",
            severity="warning",
            evidence=evidence,
        ))
    elif task_passed and metrics.coverage_percent < 70:
        evidence = [f"Coverage: {metrics.coverage_percent:.0f}%"]
        signals.append(QualitySignal(
            signal_type="low_gt_coverage",
            description="Trajectory succeeded but matched less than 70% of the ground-truth steps, suggesting an alternative approach.",
            severity="info",
            evidence=evidence,
        ))

    return signals


def quality_assessment(
    result: "MatchResult",
    candidate: Trace,
    ground_truth: Trace,
    passed: Optional[bool] = None,
) -> QualityReport:
    """Assess the quality of a matched trajectory.

    Answers: *Why is it failing?* (for fails) and *Is this a lucky or ideal
    pass?* (for passes).  Derives everything from the existing
    :class:`MatchMetrics` — deterministic, no LLM needed.

    Parameters
    ----------
    result : MatchResult
        Output of :func:`run`.
    candidate : Trace
        The candidate trace that was matched.
    ground_truth : Trace
        The ground-truth trace it was matched against.

    Returns
    -------
    QualityReport
    """
    m = result.metrics

    # Build per-stage coverage details
    stage_cov: Dict[str, StageCoverageDetail] = {}
    for stg in _STAGE_ORDER:
        total = m.stage_total.get(stg, 0)
        matched = m.stage_matched.get(stg, 0)
        pct = m.stage_coverage.get(stg, 0.0)
        stage_cov[stg] = StageCoverageDetail(matched=matched, total=total, percent=pct)

    verdict = _compute_verdict(
        coverage=m.coverage_percent,
        weighted_score=m.weighted_score,
        stage_completeness=m.stage_completeness,
        terminal_match=m.terminal_state_match,
        coherence=m.coherence_score,
        bottleneck_coverage=m.bottleneck_coverage,
    )

    tier = _compute_quality_tier(
        verdict=verdict,
        coverage=m.coverage_percent,
        coherence=m.coherence_score,
        stage_completeness=m.stage_completeness,
        terminal_match=m.terminal_state_match,
        workflow_similarity=m.workflow_similarity,
    )

    score = _compute_quality_score(
        coverage=m.coverage_percent,
        coherence=m.coherence_score,
        stage_completeness=m.stage_completeness,
        workflow_similarity=m.workflow_similarity,
        f1=m.f1_score,
        passed=passed,
    )

    is_fail = verdict in ("FAIL", "LIKELY FAIL", "UNCERTAIN")
    failure_reasons = (
        _build_failure_reasons(m, result.divergence_index, m.total_ground_truth_states, stage_cov)
        if is_fail
        else []
    )

    strengths = _build_strengths(stage_cov, m)
    divergence = _build_divergence_point(result, candidate, ground_truth)

    # New: all divergence segments
    divergence_points = _build_divergence_points(result, candidate, ground_truth)

    # New: per-stage comparison
    stage_comparison, stage_order_match = _build_stage_comparison(
        result, candidate, ground_truth,
    )

    # New: inefficiency detection
    inefficiencies = _build_inefficiency_report(candidate, result, ground_truth)

    # New: quality signals
    quality_signals = _build_quality_signals(
        verdict, m, inefficiencies, stage_cov,
        stage_comparison, stage_order_match, passed=passed,
    )

    return QualityReport(
        verdict=verdict,
        quality_tier=tier,
        quality_score=score,
        failure_reasons=failure_reasons,
        strengths=strengths,
        divergence_point=divergence,
        stage_coverage=stage_cov,
        key_metrics={
            "coverage_percent": m.coverage_percent,
            "coherence": m.coherence_score,
            "stage_completeness": m.stage_completeness,
            "workflow_similarity": m.workflow_similarity,
        },
        divergence_points=divergence_points,
        stage_comparison=stage_comparison,
        inefficiencies=inefficiencies,
        quality_signals=quality_signals,
        stage_order_match=stage_order_match,
    )


# ---------------------------------------------------------------------------
# Cohort ranking
# ---------------------------------------------------------------------------

def rank_in_cohort(
    results: List[Tuple[str, "MatchResult"]],
    candidates: Optional[List[Trace]] = None,
    ground_truth: Optional[Trace] = None,
) -> CohortRanking:
    """Rank multiple trajectories within their pass/fail cohorts.

    Parameters
    ----------
    results : list[tuple[str, MatchResult]]
        Pairs of ``(label, match_result)``.
    candidates : list[Trace] | None
        Corresponding candidate traces (same order as *results*).  If
        provided alongside *ground_truth*, full quality reports are
        generated including divergence points.
    ground_truth : Trace | None
        The ground-truth trace used for all matches.

    Returns
    -------
    CohortRanking
    """
    passing_entries: List[Tuple[int, CohortEntry]] = []  # (score, entry)
    failing_entries: List[Tuple[int, CohortEntry]] = []

    tier_counts: Dict[str, int] = {}
    failure_reason_counts: Dict[str, int] = {}

    for i, (label, match_result) in enumerate(results):
        cand = candidates[i] if candidates and i < len(candidates) else None
        gt = ground_truth

        if cand and gt:
            report = quality_assessment(match_result, cand, gt)
        else:
            # Lightweight assessment without full divergence analysis
            m = match_result.metrics
            verdict = _compute_verdict(
                m.coverage_percent, m.weighted_score,
                m.stage_completeness, m.terminal_state_match,
                m.coherence_score,
            )
            tier = _compute_quality_tier(
                verdict, m.coverage_percent, m.coherence_score,
                m.stage_completeness, m.terminal_state_match,
                m.workflow_similarity,
            )
            score = _compute_quality_score(
                m.coverage_percent, m.coherence_score,
                m.stage_completeness, m.workflow_similarity,
                m.f1_score,
            )
            report = QualityReport(
                verdict=verdict, quality_tier=tier, quality_score=score,
            )

        tier_counts[report.quality_tier] = tier_counts.get(report.quality_tier, 0) + 1

        is_pass = report.verdict in ("PASS", "LIKELY PASS")
        top_reason = ""
        if report.failure_reasons:
            top_reason = report.failure_reasons[0].reason
            failure_reason_counts[top_reason] = failure_reason_counts.get(top_reason, 0) + 1

        entry = CohortEntry(
            label=label,
            quality_score=report.quality_score,
            quality_tier=report.quality_tier,
            rank=0,  # assigned after sorting
            top_failure_reason=top_reason,
        )

        if is_pass:
            passing_entries.append((report.quality_score, entry))
        else:
            failing_entries.append((report.quality_score, entry))

    # Sort descending by quality score and assign ranks
    passing_entries.sort(key=lambda x: x[0], reverse=True)
    for rank, (_, entry) in enumerate(passing_entries, 1):
        entry.rank = rank

    failing_entries.sort(key=lambda x: x[0], reverse=True)
    for rank, (_, entry) in enumerate(failing_entries, 1):
        entry.rank = rank

    # Build common failure reasons (sorted by frequency)
    common_reasons = sorted(
        [{'reason': r, 'count': c} for r, c in failure_reason_counts.items()],
        key=lambda x: x['count'],
        reverse=True,
    )

    return CohortRanking(
        passing=[e for _, e in passing_entries],
        failing=[e for _, e in failing_entries],
        summary={
            "total": len(results),
            "passing_count": len(passing_entries),
            "failing_count": len(failing_entries),
            "ideal_count": tier_counts.get("ideal", 0),
            "solid_count": tier_counts.get("solid", 0),
            "lucky_count": tier_counts.get("lucky", 0),
            "partial_fail_count": tier_counts.get("partial_fail", 0),
            "off_track_count": tier_counts.get("off_track", 0),
            "common_failure_reasons": common_reasons,
        },
    )
