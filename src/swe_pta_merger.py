#!/usr/bin/env python3
"""
SWE PTA Merger - Merges multiple coding agent PTAs into a unified model.

This module implements the core PTA merging algorithm for SWE/coding agent trajectories:

1. Sequential state comparison using LLM-based equivalence
2. Branch detection when states diverge
3. State consolidation when equivalent states are found
4. Transition deduplication

The merger follows the TUTORIAL.md workflow:
- Start with initial PTA from first trace
- For each additional trace:
  - Walk states in order
  - Compare with existing states at each position
  - Merge if equivalent, branch if not
- Result: a merged PTA that captures common patterns and variations
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

from swe_models import PTA, State, Transition, LogEntry
from swe_state_equivalence import SWEStateEquivalence, EquivalenceResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MergeStats:
    """Statistics about the merge operation."""
    traces_merged: int = 0
    states_added: int = 0
    states_merged: int = 0
    transitions_added: int = 0
    branches_created: int = 0
    equivalence_checks: int = 0
    llm_calls: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "traces_merged": self.traces_merged,
            "states_added": self.states_added,
            "states_merged": self.states_merged,
            "transitions_added": self.transitions_added,
            "branches_created": self.branches_created,
            "equivalence_checks": self.equivalence_checks,
            "llm_calls": self.llm_calls
        }


class SWEPTAMerger:
    """
    Merges multiple SWE coding agent PTAs into a unified model.
    
    The merger maintains:
    - A merged PTA with all states and transitions
    - Branch tracking for divergent execution paths
    - Statistics about the merge process
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        llm_prefix: str = "DEFAULT",
        verbose: bool = False
    ):
        """
        Initialize the merger.
        
        Args:
            use_llm: Whether to use LLM for state equivalence
            llm_prefix: Environment variable prefix for LLM config
            verbose: Enable verbose logging
        """
        self.equivalence_checker = SWEStateEquivalence(
            use_llm=use_llm,
            llm_prefix=llm_prefix
        )
        self.verbose = verbose
        self.stats = MergeStats()
        
        # State ID mapping for merged states
        self._state_id_counter = 0
        self._transition_id_counter = 0
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def merge_ptas(self, ptas: List[PTA]) -> PTA:
        """
        Merge multiple PTAs into a single unified PTA.
        
        Args:
            ptas: List of PTAs to merge
            
        Returns:
            Merged PTA
        """
        if not ptas:
            return PTA()
        
        if len(ptas) == 1:
            return deepcopy(ptas[0])
        
        logger.info(f"Merging {len(ptas)} PTAs...")
        
        # Start with first PTA as base
        merged = self._initialize_merged_pta(ptas[0])
        # Preserve existing trace count if first PTA is already a merged PTA
        existing_traces = ptas[0].metadata.get("num_traces", 1)
        self.stats.traces_merged = existing_traces
        
        # Merge each additional PTA
        for i, pta in enumerate(ptas[1:], 2):
            logger.info(f"Merging PTA {i}/{len(ptas)}...")
            merged = self._merge_single_pta(merged, pta)
            # Add the trace count from this PTA (could be >1 if it's a merged PTA)
            new_traces = pta.metadata.get("num_traces", 1)
            self.stats.traces_merged += new_traces
        
        # Post-processing: consolidate states with same resulting_state
        merged = self._consolidate_by_resulting_state(merged)
        
        # Post-processing: remove any backward edges (transitions where to_step < from_step)
        merged = self._remove_backward_edges(merged)
        
        # Add metadata
        merged.metadata["merge_stats"] = self.stats.to_dict()
        merged.metadata["equivalence_stats"] = self.equivalence_checker.get_stats()
        
        logger.info(f"Merge complete: {len(merged.states)} states, {len(merged.transitions)} transitions")
        
        return merged
    
    def merge_trace_into_pta(self, pta: PTA, trace_file: str) -> PTA:
        """
        Merge a trace file into an existing PTA.
        
        Args:
            pta: Existing PTA to merge into
            trace_file: Path to trajectory JSON file
            
        Returns:
            Updated PTA
        """
        from swe_pta_generator import PTAGenerator
        
        generator = PTAGenerator()
        new_pta = generator.generate_pta(trace_file)
        
        return self._merge_single_pta(pta, new_pta)
    
    def _initialize_merged_pta(self, pta: PTA) -> PTA:
        """Initialize merged PTA from the first PTA."""
        merged = PTA()
        
        # Preserve existing trace count if first PTA is already a merged PTA
        existing_traces = pta.metadata.get("num_traces", 1)
        existing_sources = pta.metadata.get("trace_sources", [pta.metadata.get("source_file", "unknown")])
        
        merged.metadata = {
            "source": "merger",
            "num_traces": existing_traces,
            "trace_sources": list(existing_sources)  # Copy to avoid modifying original
        }
        
        # Copy states with new IDs
        state_id_map = {}  # old_id -> new_id
        
        for old_id, state in pta.states.items():
            new_id = self._new_state_id()
            state_id_map[old_id] = new_id
            
            new_state = self._copy_state(state, new_id)
            new_state.metadata["original_ids"] = [old_id]
            new_state.metadata["trace_count"] = 1
            merged.add_state(new_state)
            self.stats.states_added += 1
        
        # Copy transitions with mapped state IDs
        for trans in pta.transitions:
            new_from = state_id_map.get(trans.from_state, trans.from_state)
            new_to = state_id_map.get(trans.to_state, trans.to_state)
            
            new_trans = Transition(
                transition_id=self._new_transition_id(),
                from_state=new_from,
                to_state=new_to,
                action_type=trans.action_type,
                action_data=trans.action_data.copy() if trans.action_data else {},
                step=trans.step,
                metadata=trans.metadata.copy() if trans.metadata else {}
            )
            merged.add_transition(new_trans)
            self.stats.transitions_added += 1
        
        # Set initial state
        if pta.initial_state and pta.initial_state in state_id_map:
            merged.initial_state = state_id_map[pta.initial_state]
        elif merged.states:
            merged.initial_state = list(merged.states.keys())[0]
        
        return merged
    
    def _merge_single_pta(self, merged: PTA, new_pta: PTA) -> PTA:
        """
        Merge a single new PTA into the merged PTA.
        
        Algorithm:
        1. Get ordered states from new PTA
        2. Walk through states, comparing at each position
        3. If equivalent: merge state metadata, continue
        4. If not equivalent: create branch, add remaining states
        """
        # Track trace source
        source = new_pta.metadata.get("source_file", "unknown")
        merged.metadata.setdefault("trace_sources", []).append(source)
        merged.metadata["num_traces"] = merged.metadata.get("num_traces", 1) + 1
        
        # Get ordered states from new PTA (by step/position)
        new_states_ordered = self._get_ordered_states(new_pta)
        
        if not new_states_ordered:
            logger.warning("New PTA has no states to merge")
            return merged
        
        # Get ordered states from merged PTA for comparison
        merged_states_ordered = self._get_ordered_states(merged)
        
        # Walk through states and merge
        merged_position = 0  # Current position in merged PTA
        branch_point = None  # Where we branched (if any)
        
        state_id_map = {}  # new_pta state_id -> merged state_id
        
        for pos, new_state in enumerate(new_states_ordered):
            logger.debug(f"Processing state at position {pos}: {new_state.state_id}")
            
            # Find candidate state in merged PTA at this position
            if merged_position < len(merged_states_ordered):
                merged_state = merged_states_ordered[merged_position]
                
                # Check equivalence
                result = self.equivalence_checker.check_equivalence(
                    merged_state,
                    new_state,
                    position=pos
                )
                self.stats.equivalence_checks += 1
                
                if result.equivalent:
                    # Merge states
                    logger.debug(f"  Equivalent: merging {new_state.state_id} into {merged_state.state_id}")
                    self._merge_state_metadata(merged_state, new_state)
                    state_id_map[new_state.state_id] = merged_state.state_id
                    self.stats.states_merged += 1
                    merged_position += 1
                    continue
                else:
                    # Branch point detected
                    logger.info(f"  Branch at position {pos}: {result.reasoning}")
                    branch_point = merged_state.state_id
                    self.stats.branches_created += 1
            
            # Add new state (either no more merged states or branched)
            new_id = self._new_state_id()
            added_state = self._copy_state(new_state, new_id)
            added_state.metadata["original_ids"] = [new_state.state_id]
            added_state.metadata["trace_count"] = 1
            
            if branch_point:
                added_state.metadata["branch_from"] = branch_point
                
                # Track branch in merged PTA
                merged.branches.setdefault(branch_point, []).append(new_id)
            
            merged.add_state(added_state)
            state_id_map[new_state.state_id] = new_id
            self.stats.states_added += 1
        
        # Add transitions from new PTA
        for trans in new_pta.transitions:
            # Map state IDs
            from_id = state_id_map.get(trans.from_state)
            to_id = state_id_map.get(trans.to_state)
            
            if not from_id or not to_id:
                logger.debug(f"Skipping transition {trans.transition_id}: unmapped states")
                continue
            
            # Check if equivalent transition already exists
            if not self._transition_exists(merged, from_id, to_id, trans.action_type):
                new_trans = Transition(
                    transition_id=self._new_transition_id(),
                    from_state=from_id,
                    to_state=to_id,
                    action_type=trans.action_type,
                    action_data=trans.action_data.copy() if trans.action_data else {},
                    step=trans.step,
                    metadata=trans.metadata.copy() if trans.metadata else {}
                )
                merged.add_transition(new_trans)
                self.stats.transitions_added += 1
        
        return merged
    
    def _get_ordered_states(self, pta: PTA) -> List[State]:
        """Get states ordered by step number."""
        states = list(pta.states.values())
        return sorted(states, key=lambda s: s.step)
    
    def _copy_state(self, state: State, new_id: str) -> State:
        """Create a copy of a state with a new ID."""
        return State(
            state_id=new_id,
            step=state.step,
            log_entry=state.log_entry,  # Reference, not copy
            observation=state.observation,
            resulting_state=getattr(state, 'resulting_state', ''),  # Copy resulting_state
            content_hash=getattr(state, 'content_hash', ''),  # Copy content_hash for fine-grained matching
            content_description=getattr(state, 'content_description', ''),  # Copy for semantic comparison
            files_touched=set(state.files_touched),
            tool_used=state.tool_used,
            # NEW: Copy location and scope fields
            file_path=getattr(state, 'file_path', ''),
            line_range=getattr(state, 'line_range', None),
            relative_position=getattr(state, 'relative_position', ''),
            operation_type=getattr(state, 'operation_type', ''),
            edit_type=getattr(state, 'edit_type', ''),
            function_name=getattr(state, 'function_name', ''),
            class_name=getattr(state, 'class_name', ''),
            scope_path=getattr(state, 'scope_path', ''),
            metadata=state.metadata.copy() if state.metadata else {}
        )
    
    def _merge_state_metadata(self, merged_state: State, new_state: State):
        """Merge metadata from new state into merged state."""
        # Update trace count
        merged_state.metadata["trace_count"] = merged_state.metadata.get("trace_count", 1) + 1
        
        # Append original ID
        merged_state.metadata.setdefault("original_ids", []).append(new_state.state_id)
        
        # Merge files touched
        if new_state.files_touched:
            merged_state.files_touched.update(new_state.files_touched)
    
    def _transition_exists(self, pta: PTA, from_id: str, to_id: str, action_type: str) -> bool:
        """Check if an equivalent transition already exists."""
        for trans in pta.transitions:
            if (trans.from_state == from_id and 
                trans.to_state == to_id and 
                trans.action_type == action_type):
                return True
        return False
    
    def _new_state_id(self) -> str:
        """Generate a new unique state ID."""
        self._state_id_counter += 1
        return f"merged_state_{self._state_id_counter}"
    
    def _new_transition_id(self) -> str:
        """Generate a new unique transition ID."""
        self._transition_id_counter += 1
        return f"merged_trans_{self._transition_id_counter}"
    
    def _consolidate_by_resulting_state(self, pta: PTA) -> PTA:
        """
        Post-processing: consolidate states with identical resulting_state.
        
        The idea is to merge states that represent the same outcome (e.g., multiple
        create_file states for the same file) into a single state, even if
        they appeared in different branches during the initial merge.
        
        We only consolidate states at the same "depth" (step) to avoid
        creating cycles or backward edges. States at different positions in the
        trace represent different contexts even if they have the same resulting_state.

        NOTE: We might have to change this logic in the future based on observed behavior.
        
        Args:
            pta: Merged PTA to consolidate
            
        Returns:
            Consolidated PTA with duplicate resulting states merged
        """
        # Group states for consolidation
        # File operations (file_created, file_read, file_patched, files_modified) should be
        # consolidated by resulting_state ALONE since they represent the same outcome
        # regardless of when they occurred in the trace.
        # Other states are grouped by (resulting_state, step) to preserve ordering.
        state_groups: Dict[tuple, List[str]] = {}  # group_key -> [state_ids]
        
        # States that should NOT be consolidated
        # - 'initial': The starting state should remain unique
        # - 'llm_response:terminal': Terminal LLM responses may have different conclusions
        # - Empty string: Invalid states
        #
        # Note: 'llm_response:planning' IS allowed to consolidate since planning states
        # at the same step represent the same semantic phase regardless of model
        skip_consolidation = {'llm_response:terminal', 'initial', ''}
        
        # Prefixes that should be consolidated regardless of step
        # - File operations: represent the same file outcome
        # - LLM confirmations: represent the same confirmed outcome (terminal states should converge)
        consolidate_regardless_of_step = (
            'file_created:', 'file_read:', 'file_patched:', 'files_modified:',
            'llm_response:confirmed:',  # Terminal confirmations should converge
        )
        
        for state_id, state in pta.states.items():
            rs = getattr(state, 'resulting_state', '') or ''
            step = getattr(state, 'step', 0)
            
            # Skip empty resulting_state or states that shouldn't be consolidated
            if rs in skip_consolidation:
                continue
            
            # States with these prefixes: consolidate by resulting_state alone (regardless of step)
            # This allows terminal states with same outcome to merge even at different steps
            # Note: content_hash is preserved in states for matching but NOT used for consolidation
            # This allows semantically equivalent file operations to merge during domtree creation
            if rs.startswith(consolidate_regardless_of_step):
                group_key = (rs,)  # Just the resulting_state, no step
            else:
                # Other states: group by both resulting_state AND step to preserve ordering
                group_key = (rs, step)
            
            if group_key not in state_groups:
                state_groups[group_key] = []
            state_groups[group_key].append(state_id)
        
        # Find groups with multiple states (candidates for consolidation)
        consolidation_map: Dict[str, str] = {}  # old_state_id -> canonical_state_id
        states_to_remove: List[str] = []
        consolidated_count = 0
        
        for group_key, state_ids in state_groups.items():
            if len(state_ids) <= 1:
                continue
            
            # Extract resulting_state from group_key (first element)
            resulting_state = group_key[0]
            step_info = f"step {group_key[1]}" if len(group_key) > 1 else "any step"
            
            # Pick the state with lowest step as canonical (earliest occurrence)
            state_ids_sorted = sorted(state_ids, key=lambda sid: pta.states[sid].step)
            canonical_id = state_ids_sorted[0]
            canonical_state = pta.states[canonical_id]
            
            logger.debug(f"Consolidating {len(state_ids)} states with resulting_state='{resulting_state}' at {step_info}")
            
            # Merge all other states into the canonical one
            for other_id in state_ids_sorted[1:]:
                other_state = pta.states[other_id]
                
                # Merge metadata
                canonical_state.metadata["trace_count"] = (
                    canonical_state.metadata.get("trace_count", 1) +
                    other_state.metadata.get("trace_count", 1)
                )
                
                # Merge original_ids
                canonical_state.metadata.setdefault("original_ids", []).extend(
                    other_state.metadata.get("original_ids", [other_id])
                )
                
                # Merge files_touched
                if other_state.files_touched:
                    canonical_state.files_touched.update(other_state.files_touched)
                
                # Track for remapping
                consolidation_map[other_id] = canonical_id
                states_to_remove.append(other_id)
                consolidated_count += 1
        
        if not consolidation_map:
            logger.debug("No states to consolidate")
            return pta
        
        logger.info(f"Consolidating {consolidated_count} duplicate states")
        
        # Update transitions to point to canonical states
        updated_transitions = []
        seen_transitions = set()  # (from, to, action) to deduplicate
        skipped_self_loops = 0
        skipped_backward_edges = 0
        
        for trans in pta.transitions:
            from_state = consolidation_map.get(trans.from_state, trans.from_state)
            to_state = consolidation_map.get(trans.to_state, trans.to_state)
            
            # Skip self-loops (these are invalid and result from over-consolidation)
            if from_state == to_state:
                skipped_self_loops += 1
                logger.debug(f"Skipping self-loop transition: {from_state} -> {to_state}")
                continue
            
            # Skip backward edges (target step < source step) as they create invalid cycles
            from_step = pta.states.get(from_state, State(state_id="", step=0)).step
            to_step = pta.states.get(to_state, State(state_id="", step=999)).step
            if to_step < from_step:
                skipped_backward_edges += 1
                logger.debug(f"Skipping backward edge: {from_state}(step={from_step}) -> {to_state}(step={to_step})")
                continue
            
            # Deduplicate transitions
            trans_key = (from_state, to_state, trans.action_type)
            if trans_key in seen_transitions:
                continue
            seen_transitions.add(trans_key)
            
            # Create updated transition
            updated_trans = Transition(
                transition_id=trans.transition_id,
                from_state=from_state,
                to_state=to_state,
                action_type=trans.action_type,
                action_data=trans.action_data,
                step=trans.step,
                metadata=trans.metadata
            )
            updated_transitions.append(updated_trans)
        
        if skipped_self_loops > 0:
            logger.info(f"Skipped {skipped_self_loops} self-loop transitions")
        
        if skipped_backward_edges > 0:
            logger.info(f"Skipped {skipped_backward_edges} backward edge transitions")
        
        # Final pass: remove any remaining backward edges (from initial merge, not just consolidation)
        final_transitions = []
        final_backward_skipped = 0
        for trans in updated_transitions:
            from_step = pta.states.get(trans.from_state, State(state_id="", step=0)).step
            to_step = pta.states.get(trans.to_state, State(state_id="", step=999)).step
            if to_step < from_step:
                final_backward_skipped += 1
                logger.debug(f"Final filter: skipping backward edge {trans.from_state}(step={from_step}) -> {trans.to_state}(step={to_step})")
                continue
            final_transitions.append(trans)
        
        if final_backward_skipped > 0:
            logger.info(f"Final filter removed {final_backward_skipped} additional backward edges")
        
        # Remove consolidated states
        for state_id in states_to_remove:
            del pta.states[state_id]
        
        # Update transitions
        pta.transitions = final_transitions
        
        # Update branches map if present
        if pta.branches:
            updated_branches = {}
            for branch_point, targets in pta.branches.items():
                new_branch_point = consolidation_map.get(branch_point, branch_point)
                new_targets = [consolidation_map.get(t, t) for t in targets]
                # Remove duplicates
                new_targets = list(dict.fromkeys(new_targets))
                if new_targets:
                    updated_branches[new_branch_point] = new_targets
            pta.branches = updated_branches
        
        # Update initial_state if needed
        if pta.initial_state in consolidation_map:
            pta.initial_state = consolidation_map[pta.initial_state]
        
        # Add consolidation stats to metadata
        pta.metadata["consolidation"] = {
            "states_consolidated": consolidated_count,
            "unique_resulting_states": len([g for g in state_groups.values() if len(g) > 1])
        }
        
        logger.info(f"After consolidation: {len(pta.states)} states, {len(pta.transitions)} transitions")
        
        return pta
    
    def _remove_backward_edges(self, pta: PTA) -> PTA:
        """
        Remove backward edges (transitions where to_state.step < from_state.step).
        
        These represent illogical time-travel and result from merging traces
        where different runs performed actions in different orders.
        
        Args:
            pta: PTA to clean up
            
        Returns:
            PTA with backward edges removed
        """
        valid_transitions = []
        removed_count = 0
        
        for trans in pta.transitions:
            from_state = pta.states.get(trans.from_state)
            to_state = pta.states.get(trans.to_state)
            
            if not from_state or not to_state:
                # Keep transitions with missing states (shouldn't happen, but be safe)
                valid_transitions.append(trans)
                continue
            
            from_step = from_state.step
            to_step = to_state.step
            
            # Skip backward edges
            if to_step < from_step:
                removed_count += 1
                logger.debug(f"Removing backward edge: {trans.from_state}(step={from_step}) -> {trans.to_state}(step={to_step})")
                continue
            
            # Skip self-loops
            if trans.from_state == trans.to_state:
                removed_count += 1
                logger.debug(f"Removing self-loop: {trans.from_state}")
                continue
            
            valid_transitions.append(trans)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} backward edges/self-loops")
        
        pta.transitions = valid_transitions
        return pta
    
    def save_merged_pta(self, pta: PTA, output_path: str):
        """Save merged PTA to file."""
        pta.save(output_path)
        logger.info(f"Saved merged PTA to: {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get merge statistics."""
        return {
            "merge_stats": self.stats.to_dict(),
            "equivalence_stats": self.equivalence_checker.get_stats()
        }


def merge_pta_files(pta_files: List[str], output_path: str, use_llm: bool = True) -> PTA:
    """
    Convenience function to merge multiple PTA files.
    
    Args:
        pta_files: List of paths to PTA JSON files
        output_path: Path to save merged PTA
        use_llm: Whether to use LLM for equivalence
        
    Returns:
        Merged PTA
    """
    # Load PTAs
    ptas = []
    for path in pta_files:
        pta = PTA.load(path)
        ptas.append(pta)
    
    # Merge
    merger = SWEPTAMerger(use_llm=use_llm)
    merged = merger.merge_ptas(ptas)
    
    # Save
    merged.save(output_path)
    
    return merged


def main():
    """Command-line interface for PTA merging."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merge multiple SWE coding agent PTAs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('pta_files', nargs='+', help='PTA JSON files to merge')
    parser.add_argument('--output', '-o', required=True, help='Output file for merged PTA')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM equivalence checking')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Merge
    merger = SWEPTAMerger(use_llm=not args.no_llm, verbose=args.verbose)
    
    ptas = []
    for path in args.pta_files:
        logger.info(f"Loading PTA: {path}")
        ptas.append(PTA.load(path))
    
    merged = merger.merge_ptas(ptas)
    merger.save_merged_pta(merged, args.output)
    
    # Print stats
    stats = merger.get_stats()
    print("\n" + "="*60)
    print("MERGE STATISTICS")
    print("="*60)
    for key, value in stats["merge_stats"].items():
        print(f"  {key}: {value}")
    print("\nEquivalence Stats:")
    for key, value in stats["equivalence_stats"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
