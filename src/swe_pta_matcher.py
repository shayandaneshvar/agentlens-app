#!/usr/bin/env python3
"""
SWE PTA Matcher - Compare trajectories against a merged PTA (domtree).

This module allows you to compare individual coding agent trajectories against
a "golden" merged PTA (domtree). The core use case is:

    1. Merge a subset of SUCCESSFUL trajectories into a "domtree" (reference PTA)
    2. Compare remaining trajectories (successful or failed) against this domtree
    3. Successful trajectories should have HIGH overlap (they follow the pattern)
    4. Failed trajectories should have LOW overlap (they deviate from the pattern)

The comparison uses subsequence matching with semantic state equivalence
checking (via LLM or other methods) to determine if states are equivalent.
"""

from __future__ import annotations
import json
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

from swe_models import PTA, State, Transition
from swe_state_equivalence import SWEStateEquivalence, EquivalenceResult

# Try to import task validator for diff-based validation
try:
    from swe_task_validator import SWETaskValidator
    TASK_VALIDATOR_AVAILABLE = True
except ImportError:
    TASK_VALIDATOR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# PTA LOADING UTILITIES

def is_pta_file(path: str) -> bool:
    """Check if a JSON file is a PTA (has states, transitions, metadata keys)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return isinstance(data, dict) and all(k in data for k in ('states', 'transitions'))
    except Exception:
        return False

def load_pta(path: str) -> PTA:
    """Load a PTA from JSON file using the existing PTA model."""
    return PTA.load(path)

def find_trajectory_file(path: str) -> str:
    """
    Find the chat-export-logs.json file from a path (file or directory).
    
    If path is a directory, search for chat-export-logs.json in common locations.
    If path is a file, return it directly.
    """
    from pathlib import Path
    p = Path(path)
    
    # If it's already a file, return it
    if p.is_file():
        return str(p)
    
    # If it's a directory, search for the trajectory file
    if p.is_dir():
        candidates = [
            p / "output" / "vsc-output" / "chat-export-logs.json",
            p / "vsc-output" / "chat-export-logs.json",
            p / "chat-export-logs.json",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        
        # Search recursively
        for found in p.rglob("chat-export-logs.json"):
            return str(found)
    
    return path  # Return original, let later code fail with clear error


def generate_pta_from_trace(path: str) -> PTA:
    """
    Generate a PTA from a raw trajectory file or directory.
    
    Uses PTAGenerator to convert chat-export-logs.json format to PTA.
    If path is a directory, searches for chat-export-logs.json inside.
    """
    try:
        from swe_pta_generator import PTAGenerator
        
        # Find the actual trajectory file
        actual_path = find_trajectory_file(path)
        logger.info(f"Generating PTA from: {actual_path}")
        
        gen = PTAGenerator()
        return gen.generate_pta(actual_path)
    except Exception as e:
        logger.error(f"Failed to generate PTA from trace: {e}")
        raise RuntimeError(f"Cannot generate PTA from {path}: {e}")

def load_pta_or_trace(path: str) -> PTA:
    """
    Smart loader: Detect if input is PTA JSON or raw trace, load accordingly.
    
    Args:
        path: Path to either a PTA JSON file or a raw trajectory file
        
    Returns:
        PTA object
    """
    if is_pta_file(path):
        logger.debug(f"Loading as PTA: {path}")
        return load_pta(path)
    else:
        logger.debug(f"Generating PTA from trace: {path}")
        return generate_pta_from_trace(path)


# PATH ENUMERATION

def enumerate_all_paths(pta: PTA) -> List[List[State]]:
    """
    Enumerate all paths from initial_state to terminal states.
    
    For tree-structured PTAs (merged from multiple traces), this returns
    all possible paths. For linear PTAs, returns a single path.
    
    Uses DFS with cycle detection to handle any graph structure.
    
    Args:
        pta: PTA to enumerate paths from
        
    Returns:
        List of paths, where each path is a list of State objects in order
    
    Example:
        PTA structure:
            A -> B -> C
             \-> D -> E
        
        Returns: [[A, B, C], [A, D, E]]
    """
    if not pta.initial_state or pta.initial_state not in pta.states:
        logger.warning("PTA has no valid initial state")
        return []
    
    # Build outgoing transition map for efficient traversal
    outgoing: Dict[str, List[Transition]] = {}
    for t in pta.transitions:
        outgoing.setdefault(t.from_state, []).append(t)
    
    # Sort for deterministic ordering
    for lst in outgoing.values():
        lst.sort(key=lambda tr: tr.transition_id)
    
    all_paths: List[List[State]] = []
    
    def dfs(state_id: str, path: List[State], visited: Set[str]) -> None:
        """Depth-first search to enumerate all paths."""
        # Cycle detection
        if state_id in visited:
            return
        if state_id not in pta.states:
            return
        
        current_state = pta.states[state_id]
        new_path = path + [current_state]
        new_visited = visited | {state_id}
        
        # Get outgoing transitions
        outs = outgoing.get(state_id, [])
        
        if not outs:
            # Terminal state - save the complete path
            all_paths.append(new_path)
        else:
            # Continue DFS to all successors
            for trans in outs:
                dfs(trans.to_state, new_path, new_visited)
    
    # Start DFS from initial state
    dfs(pta.initial_state, [], set())
    
    logger.debug(f"Enumerated {len(all_paths)} paths from PTA")
    return all_paths


# STATE COMPARATOR (with caching)

class StateComparator:
    """
    Wrapper around SWEStateEquivalence with caching and statistics.
    
    This class:
    1. Uses the same equivalence logic as the merger
    2. Caches comparison results to avoid redundant LLM calls
    3. Tracks statistics (comparisons, cache hits)
    4. Optionally outputs debug information
    
    The comparator ensures consistent behavior between merging and matching:
    if two states were considered equivalent during merge, they will be
    considered equivalent during matching.
    """
    
    def __init__(self, use_llm: bool = True, llm_prefix: str = "DEFAULT", debug: bool = False):
        """
        Initialize the comparator.
        
        Args:
            use_llm: Whether to use LLM for semantic comparison
            llm_prefix: Environment variable prefix for LLM config
            debug: If True, print each comparison result
        """
        self._checker = SWEStateEquivalence(
            use_llm=use_llm,
            llm_prefix=llm_prefix
        )
        self._cache: Dict[Tuple[str, str], bool] = {}
        self.stats = {
            "comparisons": 0,
            "cache_hits": 0,
            "llm_calls": 0
        }
        self.debug = debug
    
    def equivalent(self, state_a: State, state_b: State, position: int) -> bool:
        """
        Check if two states are equivalent.
        
        Args:
            state_a: First state (typically from domtree/OLD)
            state_b: Second state (typically from trajectory/NEW)  
            position: Position in the sequence (0 = initial state)
            
        Returns:
            True if states are semantically equivalent
        """
        # Check cache first
        cache_key = (state_a.state_id, state_b.state_id)
        if cache_key in self._cache:
            self.stats["cache_hits"] += 1
            result = self._cache[cache_key]
            if self.debug:
                logger.debug(f"[CACHED] pos={position} {state_a.state_id} vs {state_b.state_id} => {result}")
            return result
        
        # Perform equivalence check
        self.stats["comparisons"] += 1
        eq_result = self._checker.check_equivalence(state_a, state_b, position=position)
        
        # Track LLM usage
        if eq_result.method == "llm":
            self.stats["llm_calls"] += 1
        
        # Cache and return
        result = eq_result.equivalent
        self._cache[cache_key] = result
        
        if self.debug:
            logger.debug(
                f"[{eq_result.method.upper()}] pos={position} "
                f"{state_a.state_id} vs {state_b.state_id} => {result} "
                f"({eq_result.reasoning})"
            )
        
        return result
    
    def get_equivalence_stats(self) -> Dict[str, Any]:
        """Get combined stats from comparator and underlying checker."""
        return {
            **self.stats,
            **self._checker.get_stats()
        }


# SUBSEQUENCE MATCHING ALGORITHMS

def subsequence_coverage(
    old_seq: List[State], 
    new_seq: List[State], 
    comparator: StateComparator
) -> Tuple[int, int, List[int]]:
    """
    Compute how many OLD states appear in NEW as a subsequence.
    
    A subsequence means OLD states appear in the SAME RELATIVE ORDER,
    but NEW may have additional states interleaved.
    
    Algorithm:
        - Walk through OLD states one by one
        - For each OLD state, scan forward in NEW to find a match
        - Once we advance past a NEW state, we can't go back
    
    Args:
        old_seq: Reference states (from domtree)
        new_seq: Target states (from trajectory)
        comparator: StateComparator for equivalence checks
        
    Returns:
        (matched_count, total_old, matched_old_indexes)
        
    Example:
        old_seq = [A, B, C, D]
        new_seq = [A, X, B, Y, C, D]
        
        i=0 (A): scan NEW, find A at j=0 ✓ matched_indexes=[0]
        i=1 (B): scan NEW from j=1, find B at j=2 ✓ matched_indexes=[0,1]
        i=2 (C): scan NEW from j=3, find C at j=4 ✓ matched_indexes=[0,1,2]
        i=3 (D): scan NEW from j=5, find D at j=5 ✓ matched_indexes=[0,1,2,3]
        
        Result: (4, 4, [0,1,2,3]) → 100% coverage
    """
    matched = 0
    matched_indexes: List[int] = []
    j = 0  # Current position in new_seq
    
    for i, s_old in enumerate(old_seq):
        # Scan forward in new_seq to find a match
        while j < len(new_seq):
            if comparator.equivalent(s_old, new_seq[j], position=i):
                matched += 1
                matched_indexes.append(i)
                j += 1
                break
            j += 1
        else:
            # Exhausted new_seq without finding remaining old states
            break
    
    return matched, len(old_seq), matched_indexes


def subsequence_coverage_incremental(
    old_seq: List[State],
    new_seq: List[State],
    comparator: StateComparator,
    start_offset: int = 0
) -> Tuple[int, int, List[int], bool, int]:
    """
    Incremental subsequence matching for streaming trajectories.
    
    Supports non-cumulative buffers where:
    - NEW is a fresh buffer (not cumulative)
    - First state of NEW might duplicate the last matched state (boundary overlap)
    
    Algorithm:
    1. Check if NEW[0] == OLD[offset-1] (duplicate boundary)
    2. If yes, skip NEW[0] and match from NEW[1]
    3. Match remaining NEW states against OLD starting from offset
    
    Args:
        old_seq: Full reference sequence (domtree path)
        new_seq: Current buffer of new states
        comparator: StateComparator for equivalence
        start_offset: Position in old_seq to start matching from
        
    Returns:
        (matched_count, total_old, matched_indexes, all_new_matched, skipped_count)
    """
    matched = 0
    matched_indexes: List[int] = []
    i = start_offset  # Position in OLD
    j = 0             # Position in NEW
    skipped = 0
    
    # Check for duplicate boundary state
    if start_offset > 0 and len(new_seq) > 0 and start_offset <= len(old_seq):
        # Does NEW[0] match the last state we already matched?
        if comparator.equivalent(old_seq[start_offset - 1], new_seq[0], position=start_offset - 1):
            j = 1  # Skip the duplicate
            skipped = 1
            logger.debug(f"Skipped duplicate boundary state at NEW[0]")
    
    # Match remaining NEW states
    while i < len(old_seq) and j < len(new_seq):
        if comparator.equivalent(old_seq[i], new_seq[j], position=i):
            matched += 1
            matched_indexes.append(i)
            i += 1
            j += 1
        else:
            # NEW state doesn't match - skip it (allows extra states in trajectory)
            j += 1
    
    # Did we match all NEW states (excluding skipped)?
    all_matched = (j == len(new_seq))
    
    return matched, len(old_seq), matched_indexes, all_matched, skipped


#  INCREMENTAL COMPARISON

def compare_incremental(
    old_path: str,
    new_path: str,
    offset: int = 0,
    use_llm: bool = True,
    llm_prefix: str = "DEFAULT",
    json_mode: bool = False,
    verbose: bool = False
) -> Tuple[Dict[str, Any], int]:
    """
    Incremental comparison for streaming/partial trajectories.
    
    Args:
        old_path: Path to reference domtree PTA
        new_path: Path to partial trajectory PTA
        offset: Number of OLD states already matched in previous calls
        use_llm: Whether to use LLM for equivalence
        llm_prefix: LLM configuration prefix
        json_mode: Return JSON-formatted output
        verbose: Enable debug logging
        
    Returns:
        (result_dict, return_code)
        
        result_dict contains:
            - status: "PASS" or "FAIL"
            - offset: New offset for next call
            - matched_old_states: States matched in this call
            - total_old_states: Total states in best path
            - coverage_percent: Coverage from this offset
            - overall_coverage_percent: Total coverage so far
    """
    # Validate inputs
    if not Path(old_path).exists():
        return {"error": f"Domtree file not found: {old_path}"}, 2
    if not Path(new_path).exists():
        return {"error": f"Trajectory file not found: {new_path}"}, 2
    if offset < 0:
        return {"error": f"Invalid offset: {offset} (must be >= 0)"}, 2
    
    # Load PTAs
    old_pta = load_pta_or_trace(old_path)
    new_pta = load_pta_or_trace(new_path)
    
    # Enumerate paths
    old_paths = enumerate_all_paths(old_pta)
    new_paths = enumerate_all_paths(new_pta)
    
    if not old_paths:
        return {"error": "Domtree has no valid paths"}, 3
    if not new_paths:
        return {"error": "Trajectory has no valid paths"}, 3
    
    # For NEW, use the longest path (typically linear anyway)
    if len(new_paths) > 1:
        logger.warning(f"Trajectory has {len(new_paths)} paths; using longest")
    new_seq = max(new_paths, key=len)
    
    # Set up comparator
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    comparator = StateComparator(use_llm=use_llm, llm_prefix=llm_prefix, debug=verbose)
    
    # Try all OLD paths and find best match
    best_result = None
    best_coverage = -1.0
    all_path_results = []
    
    for path_idx, old_seq in enumerate(old_paths):
        # Skip paths shorter than current offset
        if offset >= len(old_seq):
            continue
        
        logger.info(f"Matching against path {path_idx + 1}/{len(old_paths)} (length={len(old_seq)})")
        
        matched, total_old, matched_idx, all_matched, skipped = subsequence_coverage_incremental(
            old_seq, new_seq, comparator, start_offset=offset
        )
        
        # Compute new offset (position after last match)
        new_offset = matched_idx[-1] + 1 if matched_idx else offset
        
        # Coverage from current offset
        remaining = total_old - offset
        coverage_pct = (matched / remaining * 100.0) if remaining > 0 else 0.0
        
        # Overall coverage from start
        overall_pct = (new_offset / total_old * 100.0) if total_old > 0 else 0.0
        
        path_result = {
            'path_index': path_idx,
            'old_states': total_old,
            'matched_in_call': matched,
            'matched_indexes': matched_idx if (verbose or json_mode) else None,
            'new_offset': new_offset,
            'coverage_percent': round(coverage_pct, 2),
            'overall_coverage_percent': round(overall_pct, 2),
            'all_new_matched': all_matched,
            'skipped_duplicate': skipped,
        }
        all_path_results.append(path_result)
        
        if coverage_pct > best_coverage:
            best_coverage = coverage_pct
            best_result = path_result
    
    if best_result is None:
        return {"error": f"No valid paths (all shorter than offset={offset})"}, 3
    
    # PASS = all trajectory states matched in order
    status = "PASS" if best_result['all_new_matched'] else "FAIL"
    
    result = {
        'domtree': old_path,
        'trajectory': new_path,
        'status': status,
        'offset': best_result['new_offset'],
        'input_offset': offset,
        'matched_old_states': best_result['matched_in_call'],
        'total_old_states': best_result['old_states'],
        'new_states': len(new_seq),
        'coverage_percent': best_result['coverage_percent'],
        'overall_coverage_percent': best_result['overall_coverage_percent'],
        'all_new_matched': best_result['all_new_matched'],
        'best_path_index': best_result['path_index'],
        'old_total_paths': len(old_paths),
        'equivalence_comparisons': comparator.stats["comparisons"],
        'cache_hits': comparator.stats["cache_hits"],
        'llm_calls': comparator.stats["llm_calls"],
    }
    
    if verbose or json_mode:
        result['matched_indexes'] = best_result.get('matched_indexes')
        result['all_path_results'] = [
            {k: v for k, v in pr.items() if k != 'matched_indexes'}
            for pr in all_path_results
        ]
    
    return result, 0


# FULL COMPARISON

def compare(
    old_path: str,
    new_path: str,
    use_llm: bool = True,
    llm_prefix: str = "DEFAULT",
    json_mode: bool = False,
    verbose: bool = False,
    offset: int = -1
) -> int:
    """
    Compare a trajectory against a domtree (merged PTA).
    
    Args:
        old_path: Path to domtree (merged PTA)
        new_path: Path to trajectory (single run PTA or raw trace)
        use_llm: Use LLM for semantic equivalence
        llm_prefix: Environment prefix for LLM config
        json_mode: Output JSON format
        verbose: Enable debug output
        offset: If >= 0, use incremental mode
        
    Returns:
        Exit code (0 = success)
    """
    # Dispatch to incremental mode if offset specified
    if offset >= 0:
        result, rc = compare_incremental(
            old_path, new_path, offset=offset,
            use_llm=use_llm, llm_prefix=llm_prefix,
            json_mode=json_mode, verbose=verbose
        )
        
        if rc != 0:
            if 'error' in result:
                print(f"Error: {result['error']}", file=sys.stderr)
            return rc
        
        if json_mode:
            print(json.dumps({k: v for k, v in result.items() if v is not None}, indent=2))
        else:
            print(f"STATUS: {result['status']}")
            print(f"OFFSET: {result['offset']}")
            print(f"INPUT_OFFSET: {result['input_offset']}")
            print(f"MATCHED_OLD_STATES: {result['matched_old_states']}")
            print(f"TOTAL_OLD_STATES: {result['total_old_states']}")
            print(f"NEW_STATES: {result['new_states']}")
            print(f"COVERAGE_PERCENT: {result['coverage_percent']}")
            print(f"OVERALL_COVERAGE_PERCENT: {result['overall_coverage_percent']}")
            print(f"ALL_NEW_MATCHED: {result['all_new_matched']}")
            print(f"BEST_PATH_INDEX: {result['best_path_index']}")
            print(f"OLD_TOTAL_PATHS: {result['old_total_paths']}")
            print(f"EQUIVALENCE_CHECKS: {result['equivalence_comparisons']}")
            print(f"CACHE_HITS: {result['cache_hits']}")
            print(f"LLM_CALLS: {result['llm_calls']}")
        
        return rc
    
    # === FULL COMPARISON MODE ===
    
    # Validate inputs
    if not Path(old_path).exists():
        print(f"Error: Domtree file not found: {old_path}", file=sys.stderr)
        return 2
    if not Path(new_path).exists():
        print(f"Error: Trajectory file not found: {new_path}", file=sys.stderr)
        return 2
    
    # Load PTAs
    logger.info(f"Loading domtree: {old_path}")
    old_pta = load_pta_or_trace(old_path)
    
    logger.info(f"Loading trajectory: {new_path}")
    new_pta = load_pta_or_trace(new_path)
    
    # Enumerate paths
    old_paths = enumerate_all_paths(old_pta)
    new_paths = enumerate_all_paths(new_pta)
    
    if not old_paths:
        print("Error: Domtree has no valid paths", file=sys.stderr)
        return 3
    if not new_paths:
        print("Error: Trajectory has no valid paths", file=sys.stderr)
        return 3
    
    logger.info(f"Domtree has {len(old_paths)} path(s), trajectory has {len(new_paths)} path(s)")
    
    # For NEW, use longest path
    if len(new_paths) > 1:
        logger.warning(f"Trajectory has multiple paths; using longest")
    new_seq = max(new_paths, key=len)
    
    # Set up comparator
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    comparator = StateComparator(use_llm=use_llm, llm_prefix=llm_prefix, debug=verbose)
    
    # Filter out trivially short paths (less than 5 states) to avoid false positives
    # Short paths like [initial, planning, read_file, LLM] match almost anything
    MIN_PATH_LENGTH = 5
    valid_old_paths = [p for p in old_paths if len(p) >= MIN_PATH_LENGTH]
    if not valid_old_paths:
        # Fall back to all paths if none meet minimum length
        logger.warning(f"No paths with >= {MIN_PATH_LENGTH} states, using all paths")
        valid_old_paths = old_paths
    elif len(valid_old_paths) < len(old_paths):
        logger.info(f"Filtered to {len(valid_old_paths)} paths (>= {MIN_PATH_LENGTH} states)")
    
    # Try all valid OLD paths
    best_result = None
    best_coverage = -1.0
    best_old_seq = None
    all_path_results = []
    
    for path_idx, old_seq in enumerate(valid_old_paths):
        logger.info(f"Matching against path {path_idx + 1}/{len(valid_old_paths)} ({len(old_seq)} states)")
        
        matched, total_old, matched_idx = subsequence_coverage(old_seq, new_seq, comparator)
        coverage_pct = (matched / total_old * 100.0) if total_old else 0.0
        
        # Check terminal state match
        terminal_match = comparator.equivalent(old_seq[-1], new_seq[-1], position=total_old - 1)
        
        # Perfect match = all OLD states found
        perfect = (matched == total_old)
        
        path_result = {
            'path_index': path_idx,
            'old_states': total_old,
            'matched_old_states': matched,
            'coverage_percent': round(coverage_pct, 2),
            'terminal_state_match': terminal_match,
            'perfect_subsequence_match': perfect,
            'matched_indexes': matched_idx if (verbose or json_mode) else None,
        }
        all_path_results.append(path_result)
        
        if coverage_pct > best_coverage:
            best_coverage = coverage_pct
            best_result = path_result
            best_old_seq = old_seq
    
    # Build final result
    result = {
        'domtree': old_path,
        'trajectory': new_path,
        'old_total_paths': len(old_paths),
        'new_states': len(new_seq),
        'best_path_index': best_result['path_index'],
        'old_states': best_result['old_states'],
        'matched_old_states': best_result['matched_old_states'],
        'coverage_percent': best_result['coverage_percent'],
        'terminal_state_match': best_result['terminal_state_match'],
        'perfect_subsequence_match': best_result['perfect_subsequence_match'],
        'equivalence_comparisons': comparator.stats["comparisons"],
        'cache_hits': comparator.stats["cache_hits"],
        'llm_calls': comparator.stats["llm_calls"],
        'matched_indexes': best_result.get('matched_indexes') if (verbose or json_mode) else None,
    }
    
    if verbose or json_mode:
        result['all_path_results'] = [
            {k: v for k, v in pr.items() if k != 'matched_indexes'}
            for pr in all_path_results
        ]
    
    # Output
    if json_mode:
        print(json.dumps({k: v for k, v in result.items() if v is not None}, indent=2))
    else:
        print(f"OLD_TOTAL_PATHS: {result['old_total_paths']}")
        print(f"BEST_PATH_INDEX: {result['best_path_index']}")
        print(f"OLD_STATES: {result['old_states']}")
        print(f"NEW_STATES: {result['new_states']}")
        print(f"MATCHED_OLD_STATES: {result['matched_old_states']}")
        print(f"COVERAGE_PERCENT: {result['coverage_percent']}")
        print(f"TERMINAL_STATE_MATCH: {result['terminal_state_match']}")
        print(f"PERFECT_SUBSEQUENCE_MATCH: {result['perfect_subsequence_match']}")
        print(f"EQUIVALENCE_CHECKS: {result['equivalence_comparisons']}")
        print(f"CACHE_HITS: {result['cache_hits']}")
        print(f"LLM_CALLS: {result['llm_calls']}")
        
        if verbose:
            matched_idx = best_result.get('matched_indexes', [])
            missing = [i for i in range(len(best_old_seq)) if i not in matched_idx]
            
            print(f"\nMATCHED_INDEXES: {matched_idx}")
            print(f"MISSING_OLD_INDEXES: {missing}")
            
            if missing:
                print("\nMISSING STATES DETAIL:")
                for i in missing[:20]:
                    st = best_old_seq[i]
                    tool = st.tool_used or "N/A"
                    obs_preview = st.observation[:80].replace('\n', ' ') if st.observation else ""
                    print(f"  [{i}] {st.state_id}: tool={tool}, obs=\"{obs_preview}...\"")
                if len(missing) > 20:
                    print(f"  ... and {len(missing) - 20} more")
            
            print("\nALL PATH SUMMARY:")
            for pr in all_path_results:
                marker = " *BEST*" if pr['path_index'] == result['best_path_index'] else ""
                print(
                    f"  Path {pr['path_index']}: {pr['old_states']} states, "
                    f"{pr['coverage_percent']}% coverage, "
                    f"terminal={pr['terminal_state_match']}{marker}"
                )
    
    return 0


# BATCH COMPARISON

def batch_compare(
    domtree_path: str,
    trajectory_paths: List[str],
    use_llm: bool = True,
    llm_prefix: str = "DEFAULT",
    verbose: bool = False,
    validate_task: bool = True
) -> List[Dict[str, Any]]:
    """
    Compare multiple trajectories against a domtree.
    
    Useful for analyzing a set of runs (successful and failed) against
    a merged reference PTA.
    
    Args:
        domtree_path: Path to the merged PTA (reference)
        trajectory_paths: List of paths to trajectory files
        use_llm: Use LLM for equivalence
        llm_prefix: LLM config prefix
        verbose: Debug output
        
    Returns:
        List of result dictionaries, one per trajectory
        
    Example:
        results = batch_compare(
            "merged.json", 
            ["pass1.json", "pass2.json", "fail1.json", "fail2.json"]
        )
        
        # Analyze results
        for r in sorted(results, key=lambda x: x['coverage_percent'], reverse=True):
            print(f"{r['trajectory']}: {r['coverage_percent']}% - terminal={r['terminal_state_match']}")
    """
    # Load domtree once
    logger.info(f"Loading domtree: {domtree_path}")
    domtree = load_pta_or_trace(domtree_path)
    domtree_paths = enumerate_all_paths(domtree)
    
    if not domtree_paths:
        raise ValueError("Domtree has no valid paths")
    
    # Extract required tools from domtree for process validation
    required_tools = _extract_required_tools(domtree)
    if required_tools:
        logger.info(f"Required tools from domtree: {required_tools}")
    
    # Filter out trivially short paths to avoid false positives
    MIN_PATH_LENGTH = 3  # Lowered from 5 to include shorter valid paths
    valid_domtree_paths = [p for p in domtree_paths if len(p) >= MIN_PATH_LENGTH]
    if not valid_domtree_paths:
        logger.warning(f"No paths with >= {MIN_PATH_LENGTH} states, using all paths")
        valid_domtree_paths = domtree_paths
    elif len(valid_domtree_paths) < len(domtree_paths):
        logger.info(f"Filtered to {len(valid_domtree_paths)} paths (>= {MIN_PATH_LENGTH} states)")
    
    results = []
    
    for traj_path in trajectory_paths:
        logger.info(f"Comparing: {traj_path}")
        
        try:
            traj = load_pta_or_trace(traj_path)
            traj_paths = enumerate_all_paths(traj)
            
            if not traj_paths:
                results.append({
                    'trajectory': traj_path,
                    'error': 'No valid paths in trajectory',
                    'coverage_percent': 0.0,
                    'terminal_state_match': False,
                })
                continue
            
            # Use longest path from trajectory
            traj_seq = max(traj_paths, key=len)
            
            # Comparator per trajectory (fresh cache)
            comparator = StateComparator(use_llm=use_llm, llm_prefix=llm_prefix, debug=verbose)
            
            # Find best matching path
            best_result = None
            best_coverage = -1.0
            
            for path_idx, dom_seq in enumerate(valid_domtree_paths):
                matched, total, matched_idx = subsequence_coverage(dom_seq, traj_seq, comparator)
                coverage = (matched / total * 100.0) if total else 0.0
                terminal = comparator.equivalent(dom_seq[-1], traj_seq[-1], position=total - 1)
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_result = {
                        'path_index': path_idx,
                        'matched': matched,
                        'total': total,
                        'coverage': coverage,
                        'terminal': terminal,
                        'matched_idx': matched_idx,
                    }
            
            result_dict = {
                'trajectory': traj_path,
                'best_path_index': best_result['path_index'],
                'old_states': best_result['total'],
                'new_states': len(traj_seq),
                'matched_old_states': best_result['matched'],
                'coverage_percent': round(best_result['coverage'], 2),
                'terminal_state_match': best_result['terminal'],
                'equivalence_comparisons': comparator.stats['comparisons'],
                'llm_calls': comparator.stats['llm_calls'],
            }
            
            # Add process validation (check for required tools)
            if required_tools:
                process_cov, missing = _check_process_coverage(traj, required_tools)
                result_dict['process_coverage'] = round(process_cov, 2)
                result_dict['missing_tools'] = missing
            
            # Add task validation if available
            if validate_task and TASK_VALIDATOR_AVAILABLE:
                try:
                    # Infer trajectory directory from PTA path
                    traj_dir = _infer_trajectory_dir(traj_path)
                    if traj_dir:
                        validator = SWETaskValidator(llm_prefix=llm_prefix)
                        validation = validator.validate_trajectory(traj_dir)
                        result_dict['task_valid'] = validation.is_valid
                        result_dict['task_score'] = round(validation.score * 100, 1)
                        result_dict['task_issues'] = validation.issues[:2] if validation.issues else []
                except Exception as e:
                    logger.debug(f"Task validation failed for {traj_path}: {e}")
            
            results.append(result_dict)
            
        except Exception as e:
            logger.error(f"Error processing {traj_path}: {e}")
            results.append({
                'trajectory': traj_path,
                'error': str(e),
                'coverage_percent': 0.0,
                'terminal_state_match': False,
            })
    
    return results


def _infer_trajectory_dir(pta_path: str) -> Optional[str]:
    """
    Infer the trajectory directory from a PTA file path or trajectory directory.
    
    E.g., trajectories/python_refactor/_outputs/python_refactor-logs-gpt-4.1-fail_pta.json
    -> trajectories/python_refactor/python_refactor-logs-gpt-4.1-fail
    
    Or if pta_path is already a directory, return it.
    """
    import os
    
    p = Path(pta_path)
    
    # If it's already a directory, return it
    if p.is_dir():
        return str(p)
    
    pta_name = p.stem  # e.g., "python_refactor-logs-gpt-4.1-fail_pta"
    if pta_name.endswith('_pta'):
        traj_name = pta_name[:-4]  # Remove "_pta" suffix
    else:
        return None
    
    # Try to find the trajectory directory
    pta_dir = p.parent  # e.g., trajectories/python_refactor/_outputs
    parent_dir = pta_dir.parent  # e.g., trajectories/python_refactor
    
    traj_dir = parent_dir / traj_name
    if traj_dir.exists() and traj_dir.is_dir():
        return str(traj_dir)
    
    return None


def _extract_required_tools(domtree: PTA) -> List[str]:
    """
    Extract the set of required tools from the domtree paths.
    
    These are tools that appear in ALL successful paths, indicating
    they are mandatory steps for task completion.
    
    Returns:
        List of required tool names (excluding 'llm' states)
    """
    paths = enumerate_all_paths(domtree)
    if not paths:
        return []
    
    # Get tool sets for each path (excluding llm/None)
    path_tools = []
    for path in paths:
        tools = set()
        for state in path:
            tool = state.tool_used
            if tool and tool != 'llm':
                # Normalize tool names (remove parameters)
                base_tool = tool.split('[')[0].strip()
                tools.add(base_tool)
        path_tools.append(tools)
    
    # Find intersection - tools present in ALL paths
    if path_tools:
        required = path_tools[0]
        for tools in path_tools[1:]:
            required = required.intersection(tools)
        return list(required)
    
    return []


def _check_process_coverage(traj: PTA, required_tools: List[str]) -> Tuple[float, List[str]]:
    """
    Check if trajectory includes all required tools/steps.
    
    Args:
        traj: Trajectory PTA
        required_tools: List of required tool names from domtree
        
    Returns:
        Tuple of (coverage_ratio, list of missing tools)
    """
    if not required_tools:
        return 1.0, []
    
    # Get tools used in trajectory
    traj_tools = set()
    for state in traj.states.values():
        tool = state.tool_used
        if tool and tool != 'llm':
            base_tool = tool.split('[')[0].strip()
            traj_tools.add(base_tool)
    
    # Check which required tools are missing
    missing = [t for t in required_tools if t not in traj_tools]
    coverage = (len(required_tools) - len(missing)) / len(required_tools)
    
    return coverage, missing


def _compute_verdict(result: Dict[str, Any]) -> str:
    """
    Compute final verdict based on structural coverage, terminal match, task validity,
    and process coverage.
    
    Verdict logic (coverage-weighted):
    - PASS: High coverage (>=80%) + process coverage 100%
    - LIKELY PASS: Coverage >=60% + Terminal match + process coverage 100%
    - UNCERTAIN: Medium coverage (40-60%) with some issues
    - LIKELY FAIL: Low coverage (<50%) OR process coverage <100%
    - FAIL: Task INVALID with low score (<30%)
    """
    coverage = result.get('coverage_percent', 0)
    terminal = result.get('terminal_state_match', False)
    task_valid = result.get('task_valid')  # None if not available
    task_score = result.get('task_score', 0)
    process_coverage = result.get('process_coverage', 1.0)
    missing_tools = result.get('missing_tools', [])
    
    # Hard failure: Task validation explicitly INVALID with low score
    if task_valid == False and task_score <= 30:
        return "FAIL"
    
    # Hard failure: Very low coverage + missing multiple critical tools
    if coverage < 40 and len(missing_tools) >= 2:
        return "FAIL"
    
    # Missing critical tools is a strong signal of failure
    if process_coverage < 1.0:
        # Even with high coverage, missing required tools is problematic
        if len(missing_tools) >= 2:
            return "LIKELY FAIL"
        elif coverage < 60:
            return "LIKELY FAIL"
        else:
            # High coverage but missing one tool - uncertain
            return "UNCERTAIN"
    
    # Process coverage is 100% - all required tools present
    if coverage >= 80:
        return "PASS"
    
    if coverage >= 60:
        return "LIKELY PASS" if terminal else "UNCERTAIN"
    
    # Medium coverage (40-60%) with all tools
    if coverage >= 40:
        if task_valid == True and terminal:
            return "LIKELY PASS"
        elif task_valid == False:
            return "LIKELY FAIL"
        else:
            return "UNCERTAIN"
    
    # Low coverage (<40%) but all tools present - might be alternative approach
    if task_valid == True and task_score >= 90:
        return "UNCERTAIN"
    
    return "LIKELY FAIL"


def print_batch_summary(results: List[Dict[str, Any]]) -> None:
    """Print a formatted summary of batch comparison results."""
    print("\n" + "=" * 130)
    print("BATCH COMPARISON SUMMARY")
    print("=" * 130)
    
    # Sort by coverage (descending)
    sorted_results = sorted(results, key=lambda x: x.get('coverage_percent', 0), reverse=True)
    
    # Check if task validation and process validation are available
    has_task_validation = any('task_valid' in r for r in results)
    has_process_validation = any('process_coverage' in r for r in results)
    
    # Calculate statistics
    coverages = [r['coverage_percent'] for r in results if 'error' not in r]
    terminal_matches = sum(1 for r in results if r.get('terminal_state_match', False))
    task_valid_count = sum(1 for r in results if r.get('task_valid', False))
    process_valid_count = sum(1 for r in results if r.get('process_coverage', 0) == 1.0)
    
    print(f"\nTotal trajectories: {len(results)}")
    print(f"Successful comparisons: {len(coverages)}")
    print(f"Terminal matches: {terminal_matches}/{len(results)}")
    if has_task_validation:
        print(f"Task validation passed: {task_valid_count}/{len(results)}")
    if has_process_validation:
        print(f"Process validation passed: {process_valid_count}/{len(results)}")
    
    if coverages:
        print(f"\nCoverage statistics:")
        print(f"  Mean: {sum(coverages) / len(coverages):.1f}%")
        print(f"  Min:  {min(coverages):.1f}%")
        print(f"  Max:  {max(coverages):.1f}%")
    
    print(f"\nDetailed results:")
    
    if has_task_validation or has_process_validation:
        print("-" * 130)
        header = f"{'Trajectory':<40} {'Coverage':>9} {'Terminal':>8} {'Process':>8}"
        if has_task_validation:
            header += f" {'TaskValid':>10} {'TaskScore':>9}"
        header += f" {'Verdict':>12}"
        print(header)
        print("-" * 130)
        
        for r in sorted_results:
            traj = Path(r['trajectory']).name
            if len(traj) > 38:
                traj = "..." + traj[-35:]
            
            if 'error' in r:
                print(f"{traj:<40} {'ERROR':>9} {'-':>8} {'-':>8}" + 
                      (f" {'-':>10} {'-':>9}" if has_task_validation else "") + 
                      f" {'-':>12}")
            else:
                terminal = "Y" if r['terminal_state_match'] else "N"
                process = f"{r.get('process_coverage', 1.0)*100:.0f}%" if 'process_coverage' in r else "-"
                
                # Compute final verdict
                verdict = _compute_verdict(r)
                
                row = f"{traj:<40} {r['coverage_percent']:>8.1f}% {terminal:>8} {process:>8}"
                if has_task_validation:
                    task_valid = "VALID" if r.get('task_valid') else "INVALID" if 'task_valid' in r else "-"
                    task_score = f"{r.get('task_score', 0):.0f}%" if 'task_score' in r else "-"
                    row += f" {task_valid:>10} {task_score:>9}"
                row += f" {verdict:>12}"
                print(row)
        
        print("-" * 130)
        
        # Show missing tools for trajectories with process issues
        process_issues = [r for r in sorted_results if r.get('missing_tools')]
        if process_issues:
            print("\nMissing tools (process validation):")
            for r in process_issues:
                traj = Path(r['trajectory']).name
                if len(traj) > 50:
                    traj = "..." + traj[-47:]
                print(f"  {traj}: {', '.join(r['missing_tools'])}")
        
        # Show task validation issues
        invalid_results = [r for r in sorted_results if r.get('task_valid') == False and r.get('task_issues')]
        if invalid_results:
            print("\nTask validation issues:")
            for r in invalid_results:
                traj = Path(r['trajectory']).name
                if len(traj) > 50:
                    traj = "..." + traj[-47:]
                print(f"  {traj}:")
                for issue in r.get('task_issues', [])[:2]:
                    print(f"    - {issue[:70]}")
    else:
        print("-" * 80)
        print(f"{'Trajectory':<50} {'Coverage':>10} {'Terminal':>10}")
        print("-" * 80)
        
        for r in sorted_results:
            traj = Path(r['trajectory']).name
            if len(traj) > 48:
                traj = "..." + traj[-45:]
            
            if 'error' in r:
                print(f"{traj:<50} {'ERROR':>10} {'-':>10}")
            else:
                terminal = "Y" if r['terminal_state_match'] else "N"
                print(f"{traj:<50} {r['coverage_percent']:>9.1f}% {terminal:>10}")
        
        print("-" * 80)


# CLI ENTRY POINT

def main():
    """Command-line interface for PTA matching."""
    parser = argparse.ArgumentParser(
        description="Compare trajectories against a merged PTA (domtree).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full comparison
  python swe_pta_matcher.py merged_pta.json trajectory.json
  
  # JSON output
  python swe_pta_matcher.py merged_pta.json trajectory.json --json
  
  # Verbose debugging
  python swe_pta_matcher.py merged_pta.json trajectory.json --verbose
  
  # Incremental mode (for streaming)
  python swe_pta_matcher.py merged_pta.json partial.json --offset 5
  
  # Batch comparison
  python swe_pta_matcher.py merged_pta.json --batch traj1.json traj2.json traj3.json
  
  # Without LLM (heuristics only)
  python swe_pta_matcher.py merged_pta.json trajectory.json --no-llm
"""
    )
    
    parser.add_argument('domtree', help='Path to merged PTA / domtree (reference)')
    parser.add_argument('trajectory', nargs='?', help='Path to trajectory PTA or raw trace')
    parser.add_argument('--batch', nargs='+', help='Batch mode: compare multiple trajectories')
    parser.add_argument('--offset', type=int, default=None, 
                        help='Starting offset for incremental matching')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose debug output')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM equivalence (use heuristics only)')
    parser.add_argument('--llm-prefix', default='DEFAULT', help='Environment variable prefix for LLM config')
    parser.add_argument('--no-task-validation', action='store_true', help='Disable task validation (diff checking)')
    
    args = parser.parse_args()
    
    use_llm = not args.no_llm
    
    try:
        if args.batch:
            # Batch mode
            results = batch_compare(
                args.domtree,
                args.batch,
                use_llm=use_llm,
                llm_prefix=args.llm_prefix,
                verbose=args.verbose,
                validate_task=not args.no_task_validation
            )
            
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print_batch_summary(results)
            
            return 0
        
        elif args.trajectory:
            # Single comparison
            offset = args.offset if args.offset is not None else -1
            return compare(
                args.domtree,
                args.trajectory,
                use_llm=use_llm,
                llm_prefix=args.llm_prefix,
                json_mode=args.json,
                verbose=args.verbose,
                offset=offset
            )
        
        else:
            parser.error("Either trajectory or --batch must be provided")
            return 1
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 4


if __name__ == '__main__':
    sys.exit(main())
