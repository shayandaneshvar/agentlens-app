#!/usr/bin/env python3
"""
SWE State Equivalence - LLM-based state equivalence for coding agent trajectories.

For SWE/coding trajectories, state equivalence is determined by semantic similarity
of observations (tool calls, responses, context) rather than visual screenshots.

This module provides:
1. LLM-based semantic equivalence checking
2. Fallback heuristic equivalence for when LLM is not available
3. Caching of equivalence decisions to avoid redundant LLM calls
"""

import json
import logging
import hashlib
import time
from typing import Dict, Any, Optional, List, Set, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from swe_models import State

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Track which models don't support certain parameters to avoid repeated 400 errors
_MODEL_PARAM_CACHE = {}


def _safe_llm_call(client, model: str, messages: list, max_tokens: int = 200, temperature: float = 0.1, retries: int = 3, timeout: float = 30.0):
    """
    Make an LLM call with model-appropriate parameters.
    
    Caches which parameters work for each model to avoid repeated 400 errors.
    Includes retry logic for transient network errors.
    """
    global _MODEL_PARAM_CACHE
    
    cache_key = model
    
    def _make_call(kwargs):
        """Make a single API call with retry logic for network errors."""
        for attempt in range(retries):
            try:
                return client.chat.completions.create(**kwargs, timeout=timeout)
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a transient network error worth retrying
                is_transient = any(err in error_str for err in [
                    "timeout", "connection", "network", "reset", "eof", 
                    "ssl", "socket", "temporarily unavailable", "503", "502", "429"
                ])
                if is_transient and attempt < retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                    logger.warning(f"Transient error on attempt {attempt + 1}/{retries}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                raise
        raise last_error  # Should not reach here
    
    # Check if we already know what works for this model
    if cache_key in _MODEL_PARAM_CACHE:
        working_params = _MODEL_PARAM_CACHE[cache_key]
        kwargs = {"model": model, "messages": messages}
        # Apply cached working params with actual values
        if "max_completion_tokens" in working_params:
            kwargs["max_completion_tokens"] = max_tokens
        elif "max_tokens" in working_params:
            kwargs["max_tokens"] = max_tokens
        if "temperature" in working_params:
            kwargs["temperature"] = temperature
        return _make_call(kwargs)
    
    # Try different parameter combinations - ordered by most likely to work first
    # Note: Some models (e.g., gpt-5.2-chat) don't support temperature, try without first
    param_combinations = [
        {"max_completion_tokens": max_tokens},  # Most compatible: no temperature
        {"max_tokens": max_tokens},  # Legacy param, no temperature
        {"temperature": temperature, "max_completion_tokens": max_tokens},  # With temperature
        {"temperature": temperature, "max_tokens": max_tokens},  # With temperature, legacy
        {},  # Fallback: no optional params
    ]
    
    last_error = None
    for params in param_combinations:
        try:
            kwargs = {"model": model, "messages": messages}
            kwargs.update(params)
            response = _make_call(kwargs)
            # Success! Cache the param keys that worked
            _MODEL_PARAM_CACHE[cache_key] = set(params.keys())
            return response
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Only continue if it's a parameter error
            if "400" not in str(e) and "invalid" not in error_str and "unsupported" not in error_str:
                raise
    
    # All combinations failed
    raise last_error


@dataclass
class EquivalenceResult:
    """Result of state equivalence check."""
    equivalent: bool
    confidence: float  # 0.0 to 1.0
    reasoning: str
    method: str  # "llm", "heuristic", "exact_match", "cached"
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "method": self.method,
            "metadata": self.metadata or {}
        }


class SWEStateEquivalence:
    """
    LLM-based state equivalence checker for SWE coding trajectories.
    
    State equivalence is determined by comparing:
    1. Tool used (exact match required)
    2. Key arguments (semantic similarity)
    3. Response/observation content (semantic similarity)
    
    For tool calls:
    - Same tool + similar target (file, query) + similar outcome = equivalent
    
    For LLM requests:
    - Similar intent/response = equivalent (more lenient)
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        llm_prefix: str = "DEFAULT",
        cache_enabled: bool = True
    ):
        """
        Initialize the equivalence checker.
        
        Args:
            use_llm: Whether to use LLM for semantic comparison
            llm_prefix: Environment variable prefix for LLM config
            cache_enabled: Whether to cache equivalence decisions
        """
        self.use_llm = use_llm
        self.llm_prefix = llm_prefix
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, EquivalenceResult] = {}
        self._comparison_records: List[Dict[str, Any]] = []
        
        # Track stats
        self.stats = {
            "total_checks": 0,
            "llm_calls": 0,
            "llm_skipped": 0,  # When LLM would be called but is disabled
            "cache_hits": 0,
            "exact_matches": 0,
            "resulting_state_matches": 0,
            "semantic_content_matches": 0,
            "heuristic_matches": 0
        }
    
    def check_equivalence(
        self,
        state1: 'State',
        state2: 'State',
        position: int = None
    ) -> EquivalenceResult:
        """
        Check if two states are semantically equivalent.
        
        Args:
            state1: First state to compare
            state2: Second state to compare
            position: Position in the trace (for special handling of initial states)
            
        Returns:
            EquivalenceResult with decision and reasoning
        """
        self.stats["total_checks"] += 1
        
        # Initial states are always equivalent (position 0)
        if position == 0:
            return EquivalenceResult(
                equivalent=True,
                confidence=1.0,
                reasoning="Initial states are always merged",
                method="position_zero"
            )
        
        # Generate cache key
        cache_key = self._get_cache_key(state1, state2)
        
        # Check cache
        if self.cache_enabled and cache_key in self._cache:
            self.stats["cache_hits"] += 1
            cached = self._cache[cache_key]
            return EquivalenceResult(
                equivalent=cached.equivalent,
                confidence=cached.confidence,
                reasoning=f"[cached] {cached.reasoning}",
                method="cached",
                metadata=cached.metadata
            )
        
        # Try exact match first (fast path)
        exact_result = self._check_exact_match(state1, state2)
        if exact_result is not None:
            self.stats["exact_matches"] += 1
            self._cache[cache_key] = exact_result
            return exact_result
        
        # For file operations, check location-based equivalence BEFORE resulting_state
        # This provides finer-grained discrimination for edit operations
        tool1 = getattr(state1, 'tool_used', None)
        file_ops = {'read_file', 'replace_string_in_file', 'multi_replace_string_in_file', 
                    'create_file', 'apply_patch', 'edit_file'}
        edit_ops = {'replace_string_in_file', 'multi_replace_string_in_file', 
                    'create_file', 'apply_patch', 'edit_file'}
        
        if tool1 in file_ops:
            # For EDIT operations: try scope match FIRST (more semantically meaningful)
            if tool1 in edit_ops:
                scope_result = self._check_scope_match(state1, state2)
                if scope_result is not None:
                    # Track separately: decisions that found equivalent vs not equivalent
                    if scope_result.equivalent:
                        self.stats["scope_equiv"] = self.stats.get("scope_equiv", 0) + 1
                    else:
                        self.stats["scope_not_equiv"] = self.stats.get("scope_not_equiv", 0) + 1
                    # Log details for debugging
                    fp1 = getattr(state1, 'file_path', '') or ''
                    desc1 = getattr(state1, 'content_description', '') or ''
                    desc2 = getattr(state2, 'content_description', '') or ''
                    logger.debug(f"  [SCOPE_MATCH] {tool1}: file={fp1}, equiv={scope_result.equivalent}, method={scope_result.method}, reason={scope_result.reasoning}")
                    logger.debug(f"    desc1={desc1[:80]}...")
                    logger.debug(f"    desc2={desc2[:80]}...")
                    self._cache[cache_key] = scope_result
                    return scope_result
            
            # Try line range overlap match (same file + overlapping regions)
            line_overlap_result = self._check_line_overlap(state1, state2)
            if line_overlap_result is not None:
                # Track separately: decisions that found equivalent vs not equivalent
                if line_overlap_result.equivalent:
                    self.stats["line_overlap_equiv"] = self.stats.get("line_overlap_equiv", 0) + 1
                else:
                    self.stats["line_overlap_not_equiv"] = self.stats.get("line_overlap_not_equiv", 0) + 1
                # Log details for debugging
                fp1 = getattr(state1, 'file_path', '') or ''
                fp2 = getattr(state2, 'file_path', '') or ''
                lr1 = getattr(state1, 'line_range', None)
                lr2 = getattr(state2, 'line_range', None)
                logger.debug(f"  [LINE_OVERLAP] {tool1}: file1={fp1}, file2={fp2}, lr1={lr1}, lr2={lr2}, equiv={line_overlap_result.equivalent}, reason={line_overlap_result.reasoning}")
                self._cache[cache_key] = line_overlap_result
                return line_overlap_result
        
        # For non-file operations: check resulting_state match
        # (File ops already checked via scope/line_overlap which are more precise)
        if tool1 not in file_ops:
            rs1 = getattr(state1, 'resulting_state', None) or ""
            rs2 = getattr(state2, 'resulting_state', None) or ""
            if rs1 and rs2 and rs1 == rs2:
                result = EquivalenceResult(
                    equivalent=True,
                    confidence=0.95,
                    reasoning=f"Same resulting state: {rs1}",
                    method="resulting_state"
                )
                self._cache[cache_key] = result
                return result
        
        # NEW: Check terminal command similarity
        if tool1 == 'run_in_terminal':
            terminal_result = self._check_terminal_similarity(state1, state2)
            if terminal_result is not None:
                self.stats["terminal_matches"] = self.stats.get("terminal_matches", 0) + 1
                self._cache[cache_key] = terminal_result
                return terminal_result
        
        # Try semantic content comparison (when content descriptions are available)
        # This is useful for cases where content_hash differs but the changes are semantically equivalent
        # Only use LLM-based semantic comparison when use_llm is enabled
        if self.use_llm:
            semantic_result = self._check_semantic_content(state1, state2)
            if semantic_result is not None:
                self.stats["semantic_content_matches"] += 1
                self._cache[cache_key] = semantic_result
                return semantic_result
        
        # Try heuristic match (medium confidence)
        heuristic_result = self._check_heuristic_match(state1, state2)
        if heuristic_result.confidence >= 0.8:  # Lowered threshold from 0.9 to 0.8
            self.stats["heuristic_matches"] += 1
            self._cache[cache_key] = heuristic_result
            return heuristic_result
        
        # Use LLM for semantic comparison only when heuristic is low confidence
        if self.use_llm:
            # Get descriptive names for logging
            # Handle cases where log_entry might be None (e.g., initial state)
            def get_state_type(state):
                if state.tool_used:
                    return state.tool_used
                if hasattr(state, 'log_entry') and state.log_entry is not None:
                    return getattr(state.log_entry, 'kind', None) or "unknown"
                # Check metadata for initial state
                if hasattr(state, 'metadata') and state.metadata.get('is_initial'):
                    return "initial"
                return "unknown"
            
            tool1 = get_state_type(state1)
            tool2 = get_state_type(state2)
            logger.info(f"  [LLM] Calling LLM for equivalence: {tool1} vs {tool2} (heuristic confidence: {heuristic_result.confidence:.2f})")
            llm_result = self._check_llm_equivalence(state1, state2)
            if llm_result is not None:
                self.stats["llm_calls"] += 1
                logger.info(f"  [LLM] Result: equivalent={llm_result.equivalent}, reasoning={llm_result.reasoning[:50]}...")
                self._cache[cache_key] = llm_result
                self._record_comparison(state1, state2, llm_result)
                return llm_result
            else:
                logger.info(f"  [LLM] LLM call failed, falling back to heuristic")
        else:
            # Log when LLM would have been called but is disabled
            if heuristic_result.confidence < 0.8:
                self.stats["llm_skipped"] = self.stats.get("llm_skipped", 0) + 1
        
        # Fall back to heuristic result
        self._cache[cache_key] = heuristic_result
        return heuristic_result
    
    def _get_cache_key(self, state1: 'State', state2: 'State') -> str:
        """Generate a cache key for state pair comparison."""
        # Use observation hashes for caching
        obs1 = state1.observation if hasattr(state1, 'observation') else str(state1.to_dict())
        obs2 = state2.observation if hasattr(state2, 'observation') else str(state2.to_dict())
        
        combined = f"{obs1}|||{obs2}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _check_exact_match(self, state1: 'State', state2: 'State') -> Optional[EquivalenceResult]:
        """
        Check for exact match (fast path).
        
        Returns EquivalenceResult if exact match detected, None otherwise.
        
        Priority:
        1. Check tool match (quick rejection if different tools)
        2. Check content_hash match (fast path for identical content)
        3. Check resulting_state (result-based comparison)
        4. Check observation hash (fallback for identical observations)
        
        Note: If content_hash differs, we return None to allow semantic comparison later.
        """
        # Check tool match first (quick rejection)
        tool1 = state1.tool_used if hasattr(state1, 'tool_used') else None
        tool2 = state2.tool_used if hasattr(state2, 'tool_used') else None
        
        if tool1 != tool2:
            return EquivalenceResult(
                equivalent=False,
                confidence=1.0,
                reasoning=f"Different tools: {tool1} vs {tool2}",
                method="exact_match"
            )
        
        # Check content_hash - if both have content hashes and they MATCH, it's equivalent
        ch1 = getattr(state1, 'content_hash', None) or ""
        ch2 = getattr(state2, 'content_hash', None) or ""
        
        if ch1 and ch2:
            if ch1 == ch2:
                # Exact same content
                return EquivalenceResult(
                    equivalent=True,
                    confidence=1.0,
                    reasoning=f"Same content hash: {ch1[:8]}...",
                    method="content_hash"
                )
            else:
                # Different hashes - don't decide here, let semantic comparison handle it
                # Return None so we fall through to heuristic or LLM comparison
                pass
        
        # Check resulting_state for result-based equivalence
        # This is the preferred comparison as it focuses on what changed, not how
        rs1 = getattr(state1, 'resulting_state', None) or ""
        rs2 = getattr(state2, 'resulting_state', None) or ""
        
        if rs1 and rs2 and rs1 == rs2:
            # Same resulting_state but possibly different content
            # If no content_hash, assume equivalent; if different content_hash, defer to semantic
            if not ch1 or not ch2:
                return EquivalenceResult(
                    equivalent=True,
                    confidence=1.0,
                    reasoning=f"Same resulting state: {rs1}",
                    method="resulting_state"
                )
            # Different content_hash with same resulting_state - defer to semantic comparison
            # by returning None
        
        # Check observation hash for exact duplicate (fallback)
        hash1 = state1.get_observation_hash() if hasattr(state1, 'get_observation_hash') else None
        hash2 = state2.get_observation_hash() if hasattr(state2, 'get_observation_hash') else None
        
        if hash1 and hash2 and hash1 == hash2:
            return EquivalenceResult(
                equivalent=True,
                confidence=1.0,
                reasoning="Identical observations",
                method="exact_match"
            )
        
        return None
    
    def _check_semantic_content(self, state1: 'State', state2: 'State') -> Optional[EquivalenceResult]:
        """
        Check semantic equivalence of file changes using content descriptions.
        
        This method compares the content_description fields of two states to determine
        if they represent semantically equivalent operations even when the exact content
        (content_hash) differs.
        
        Use cases:
        - Different variable/function naming but same logic
        - Whitespace/formatting differences
        - Same refactoring with different implementation details
        
        Returns EquivalenceResult if determination can be made, None otherwise.
        """
        # Get content descriptions
        desc1 = getattr(state1, 'content_description', None) or ""
        desc2 = getattr(state2, 'content_description', None) or ""
        
        # Both need content descriptions for this comparison
        # NOTE: We don't hardcode tools here - if the PTA generator assigned
        # a content_description to both states, they are eligible for semantic
        # comparison. This keeps the logic in sync with _compute_content_description
        # in swe_pta_generator.py and automatically supports new tools.
        if not desc1 or not desc2:
            return None
        
        # Get tools for context
        tool1 = getattr(state1, 'tool_used', None)
        tool2 = getattr(state2, 'tool_used', None)
        
        # Must be same tool for semantic comparison to make sense
        if tool1 != tool2:
            return None
        
        # Get content hashes - we only need semantic comparison if hashes differ
        ch1 = getattr(state1, 'content_hash', None) or ""
        ch2 = getattr(state2, 'content_hash', None) or ""
        
        # If hashes match, exact_match would have handled it
        if ch1 and ch2 and ch1 == ch2:
            return None
        
        # Use LLM to compare content descriptions semantically
        return self._compare_content_descriptions_llm(state1, state2, desc1, desc2)
    
    def _compare_content_descriptions_llm(
        self, 
        state1: 'State', 
        state2: 'State',
        desc1: str, 
        desc2: str
    ) -> Optional[EquivalenceResult]:
        """
        Use LLM to compare content descriptions for semantic equivalence.
        
        Args:
            state1, state2: States being compared
            desc1, desc2: Content descriptions to compare
            
        Returns:
            EquivalenceResult or None if LLM call fails
        """
        try:
            try:
                from llm import get_model_and_client
            except ImportError:
                from src.llm import get_model_and_client
        except ImportError:
            logger.warning("LLM module not available, skipping semantic content check")
            return None
        
        # Get additional context
        tool = getattr(state1, 'tool_used', 'unknown')
        rs1 = getattr(state1, 'resulting_state', '') or ''
        rs2 = getattr(state2, 'resulting_state', '') or ''
        
        # Build comparison prompt
        prompt = self._build_content_comparison_prompt(desc1, desc2, tool, rs1, rs2)
        
        try:
            client, model, temp = get_model_and_client(self.llm_prefix)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert at analyzing coding agent trajectories in software development.
Your job is to determine if two code changes represent semantically equivalent STEPS in completing a task.

CONTEXT: We are comparing trajectories of AI coding agents solving the same task. 
We want to know if two changes represent the SAME STEP in the solution process,
NOT whether the exact code produced is identical.

Two changes are SEMANTICALLY EQUIVALENT (as steps) if:
- They accomplish the same logical step in solving the task
- They modify/create the same type of artifact for the same purpose
- The core functionality being implemented is the same
- Differences are implementation details (naming, exact values, formatting)

Two changes are NOT SEMANTICALLY EQUIVALENT if:
- They create/modify fundamentally different things
- They serve different purposes in solving the task
- One implements a feature the other doesn't touch at all

Examples:
- "Defined function 'add_numbers' that adds two values" vs "Defined function 'perform_addition' that adds two values"
  → EQUIVALENT (same step: creating an addition function, naming is implementation detail)
  
- "Created file 'math_ops.py' with arithmetic functions" vs "Created file 'calculator.py' with arithmetic functions"
  → EQUIVALENT (same step: creating a file for arithmetic operations)
  
- "Added error handling to 'calculate' function" vs "Created new 'calculate' function"
  → NOT EQUIVALENT (different types of steps)

- "Modified config to set port=8080" vs "Modified config to set debug=True"
  → NOT EQUIVALENT (different configuration changes)

Respond with EXACTLY a JSON object:
{"equivalent": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Make LLM call with model-appropriate parameters (avoids 400 errors)
            # Use 500 tokens to avoid truncation (finish_reason=length)
            response = _safe_llm_call(client, model, messages, max_tokens=500, temperature=0.1)
            
            content = response.choices[0].message.content.strip()
            
            # Debug logging: show what was sent and received
            logger.debug(f"[semantic_content] Tool: {tool}")
            logger.debug(f"[semantic_content] Desc1: {desc1[:80]}")
            logger.debug(f"[semantic_content] Desc2: {desc2[:80]}")
            logger.debug(f"[semantic_content] LLM response: {content}")
            
            # Parse response
            result = self._parse_content_comparison_response(content)
            if result:
                logger.debug(f"[semantic_content] Parsed result: equivalent={result['equivalent']}, conf={result.get('confidence', 0.8)}")
                return EquivalenceResult(
                    equivalent=result["equivalent"],
                    confidence=result.get("confidence", 0.8),
                    reasoning=f"[semantic] {result['reasoning']}",
                    method="semantic_content",
                    metadata={
                        "desc1": desc1[:100],
                        "desc2": desc2[:100],
                        "raw_response": content
                    }
                )
            else:
                logger.warning(f"[semantic_content] Failed to parse LLM response: {content}")
            
        except Exception as e:
            logger.warning(f"Semantic content comparison failed: {e}")
        
        return None
    
    def _build_content_comparison_prompt(
        self, 
        desc1: str, 
        desc2: str, 
        tool: str, 
        rs1: str, 
        rs2: str
    ) -> str:
        """Build prompt for comparing content descriptions."""
        lines = []
        lines.append(f"Tool used: {tool}")
        lines.append("")
        lines.append("=== CHANGE 1 ===")
        lines.append(f"Description: {desc1}")
        if rs1:
            lines.append(f"Result: {rs1}")
        lines.append("")
        lines.append("=== CHANGE 2 ===")
        lines.append(f"Description: {desc2}")
        if rs2:
            lines.append(f"Result: {rs2}")
        lines.append("")
        lines.append("Are these changes semantically equivalent (accomplishing the same goal)?")
        lines.append('Respond with JSON: {"equivalent": true/false, "confidence": 0.0-1.0, "reasoning": "..."}')
        
        return "\n".join(lines)
    
    def _parse_content_comparison_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response for content comparison."""
        try:
            import re
            
            # Strip markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                # Remove ```json or ``` at start and ``` at end
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            
            # Fix common LLM issues: newlines inside JSON string values
            # Replace literal newlines with spaces (but preserve structure)
            # This handles cases where LLM puts multi-line reasoning
            content_fixed = re.sub(r'"\s*:\s*"([^"]*)"', 
                                   lambda m: '": "' + m.group(1).replace('\n', ' ').replace('\r', '') + '"', 
                                   content)
            
            # Try to parse the fixed content as JSON first
            try:
                result = json.loads(content_fixed)
                if "equivalent" in result:
                    return {
                        "equivalent": bool(result["equivalent"]),
                        "confidence": float(result.get("confidence", 0.8)),
                        "reasoning": str(result.get("reasoning", "LLM decision"))
                    }
            except json.JSONDecodeError:
                pass
            
            # Also try the original content
            try:
                result = json.loads(content)
                if "equivalent" in result:
                    return {
                        "equivalent": bool(result["equivalent"]),
                        "confidence": float(result.get("confidence", 0.8)),
                        "reasoning": str(result.get("reasoning", "LLM decision"))
                    }
            except json.JSONDecodeError:
                pass
            
            # Fallback: Extract boolean values directly with regex
            equiv_match = re.search(r'"equivalent"\s*:\s*(true|false)', content, re.IGNORECASE)
            if equiv_match:
                is_equiv = equiv_match.group(1).lower() == 'true'
                conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', content)
                confidence = float(conf_match.group(1)) if conf_match else 0.8
                reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:"[^"]*)*[^"]*)"', content, re.DOTALL)
                reasoning = reason_match.group(1) if reason_match else "LLM decision"
                # Clean up reasoning
                reasoning = reasoning.replace('\n', ' ').replace('\r', '').strip()
                return {
                    "equivalent": is_equiv,
                    "confidence": confidence,
                    "reasoning": reasoning[:200]  # Truncate if too long
                }
        except Exception as e:
            logger.debug(f"Failed to parse content comparison response: {e}")
        
        return None

    def _extract_scope_from_description(self, state: 'State') -> Set[str]:
        """
        Extract function/class names from content_description.
        
        Parses descriptions like:
        - "renamed function 'add_numbers' to 'perform_add'"
        - "In 'math_ops.py': renamed function 'add_numbers'"
        - "Created Python file 'app.py' functions: hello_world, main"
        - "added function 'calculate'; removed function 'old_calc'"
        - "with classes: Calculator, Helper"
        
        Returns:
            Set of extracted function/class/method names
        """
        import re
        
        desc = getattr(state, 'content_description', '') or ''
        if not desc:
            return set()
        
        names = set()
        
        # Pattern: 'name' (single-quoted identifiers)
        quoted = re.findall(r"'([a-zA-Z_][a-zA-Z0-9_]*)'", desc)
        names.update(quoted)
        
        # Pattern: functions: name1, name2 (tree-sitter format)
        func_list = re.findall(r'functions?:\s*([a-zA-Z_][a-zA-Z0-9_,\s]*)', desc, re.IGNORECASE)
        for match in func_list:
            # Split by comma and clean
            for name in match.split(','):
                name = name.strip()
                if name and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
                    names.add(name)
        
        # Pattern: classes: Name1, Name2 (tree-sitter format)
        class_list = re.findall(r'classes?:\s*([a-zA-Z_][a-zA-Z0-9_,\s]*)', desc, re.IGNORECASE)
        for match in class_list:
            for name in match.split(','):
                name = name.strip()
                if name and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
                    names.add(name)
        
        # Pattern: added/removed function 'name' (tree-sitter comparison format)
        added_removed = re.findall(r'(?:added|removed)\s+(?:function|class|method)\s+[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?', desc, re.IGNORECASE)
        names.update(added_removed)
        
        # Pattern: function/method/class NAME (general)
        keywords = re.findall(r'(?:function|method|class|def)\s+[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?', desc, re.IGNORECASE)
        names.update(keywords)
        
        # Pattern: renamed X to Y - extract both
        renamed = re.findall(r'renamed\s+(?:\w+\s+)?[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?\s+to\s+[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?', desc, re.IGNORECASE)
        for old, new in renamed:
            names.add(old)
            names.add(new)
        
        # Remove file-like names (containing dots that aren't Python files)
        names = {n for n in names if '.' not in n or n.endswith('.py')}
        
        # Remove common non-scope words and filenames
        exclude = {'in', 'to', 'from', 'the', 'and', 'or', 'with', 'modified', 'added', 'removed', 
                   'renamed', 'imports', 'code', 'file', 'created', 'python', 'javascript'}
        names = names - exclude
        
        return names

    def _compute_line_overlap(self, range1: tuple, range2: tuple) -> float:
        """
        Compute Jaccard-like overlap between two line ranges.
        
        Args:
            range1: (start, end) line numbers
            range2: (start, end) line numbers
            
        Returns:
            Overlap ratio between 0.0 and 1.0
        """
        if not range1 or not range2:
            return 0.0
        
        start1, end1 = range1
        start2, end2 = range2
        
        # Calculate intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start + 1)
        
        # Calculate union
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union = union_end - union_start + 1
        
        return intersection / union if union > 0 else 0.0

    def _check_line_overlap(self, state1: 'State', state2: 'State') -> Optional[EquivalenceResult]:
        """
        Check equivalence based on line range overlap.
        
        Two states are equivalent if:
        - Same tool
        - Same file path
        - Line ranges overlap significantly (≥50%)
        
        This is stronger than just "same file" because it checks that the agent
        is working in the same region of the file.
        
        Returns:
            EquivalenceResult if determination can be made, None otherwise.
        """
        # Get tool - must match
        tool1 = getattr(state1, 'tool_used', None)
        tool2 = getattr(state2, 'tool_used', None)
        
        if tool1 != tool2:
            return None
        
        # Only applies to file operations
        file_ops = {'read_file', 'replace_string_in_file', 'multi_replace_string_in_file', 
                    'create_file', 'apply_patch', 'edit_file'}
        if tool1 not in file_ops:
            return None
        
        # Get file paths - must match
        fp1 = getattr(state1, 'file_path', '') or ''
        fp2 = getattr(state2, 'file_path', '') or ''
        
        if not fp1 or not fp2:
            return None
        
        if fp1 != fp2:
            # Different files - not equivalent via line overlap
            return EquivalenceResult(
                equivalent=False,
                confidence=0.9,
                reasoning=f"Different files: {fp1} vs {fp2}",
                method="line_overlap"
            )
        
        # Get line ranges
        lr1 = getattr(state1, 'line_range', None)
        lr2 = getattr(state2, 'line_range', None)
        
        if not lr1 or not lr2:
            # Can't determine without line ranges - defer to other methods
            return None
        
        # Compute overlap
        overlap = self._compute_line_overlap(lr1, lr2)
        
        if overlap >= 0.5:
            # Significant overlap - likely equivalent
            return EquivalenceResult(
                equivalent=True,
                confidence=0.85 + (overlap * 0.1),  # Higher overlap = higher confidence
                reasoning=f"Same file ({fp1}) with {overlap:.0%} line overlap: {lr1} vs {lr2}",
                method="line_overlap"
            )
        elif overlap > 0:
            # Partial overlap - uncertain, defer to other methods
            return None
        else:
            # No overlap - different regions of same file
            return EquivalenceResult(
                equivalent=False,
                confidence=0.8,
                reasoning=f"Same file ({fp1}) but no line overlap: {lr1} vs {lr2}",
                method="line_overlap"
            )

    def _check_scope_match(self, state1: 'State', state2: 'State') -> Optional[EquivalenceResult]:
        """
        Check equivalence based on scope (function/class) match.
        
        Two states are equivalent if:
        - Same tool
        - Same file path
        - Same function or class being modified
        
        This captures semantic equivalence when agents edit the same function
        but in slightly different ways.
        
        Returns:
            EquivalenceResult if determination can be made, None otherwise.
        """
        # Get tool - must match
        tool1 = getattr(state1, 'tool_used', None)
        tool2 = getattr(state2, 'tool_used', None)
        
        if tool1 != tool2:
            return None
        
        # Only applies to edit operations
        edit_ops = {'replace_string_in_file', 'multi_replace_string_in_file', 
                    'create_file', 'apply_patch', 'edit_file'}
        if tool1 not in edit_ops:
            return None
        
        # Get file paths - must match
        fp1 = getattr(state1, 'file_path', '') or ''
        fp2 = getattr(state2, 'file_path', '') or ''
        
        if not fp1 or not fp2 or fp1 != fp2:
            return None  # Different or unknown files - can't determine
        
        # Get scope information
        scope1 = getattr(state1, 'scope_path', '') or ''
        scope2 = getattr(state2, 'scope_path', '') or ''
        
        func1 = getattr(state1, 'function_name', '') or ''
        func2 = getattr(state2, 'function_name', '') or ''
        
        class1 = getattr(state1, 'class_name', '') or ''
        class2 = getattr(state2, 'class_name', '') or ''
        
        # If both have scope paths, compare them
        if scope1 and scope2:
            if scope1 == scope2:
                return EquivalenceResult(
                    equivalent=True,
                    confidence=0.9,
                    reasoning=f"Same file ({fp1}) and scope: {scope1}",
                    method="scope_match"
                )
            else:
                # Different scopes in same file - not equivalent
                return EquivalenceResult(
                    equivalent=False,
                    confidence=0.85,
                    reasoning=f"Same file ({fp1}) but different scopes: {scope1} vs {scope2}",
                    method="scope_match"
                )
        
        # Try function names
        if func1 and func2:
            if func1 == func2:
                return EquivalenceResult(
                    equivalent=True,
                    confidence=0.85,
                    reasoning=f"Same file ({fp1}) and function: {func1}",
                    method="scope_match"
                )
            else:
                return EquivalenceResult(
                    equivalent=False,
                    confidence=0.8,
                    reasoning=f"Same file ({fp1}) but different functions: {func1} vs {func2}",
                    method="scope_match"
                )
        
        # Try class names
        if class1 and class2:
            if class1 == class2:
                return EquivalenceResult(
                    equivalent=True,
                    confidence=0.8,
                    reasoning=f"Same file ({fp1}) and class: {class1}",
                    method="scope_match"
                )
            else:
                return EquivalenceResult(
                    equivalent=False,
                    confidence=0.75,
                    reasoning=f"Same file ({fp1}) but different classes: {class1} vs {class2}",
                    method="scope_match"
                )
        
        # NEW: Try extracting scope from content_description
        extracted1 = self._extract_scope_from_description(state1)
        extracted2 = self._extract_scope_from_description(state2)
        
        if extracted1 and extracted2:
            # Check for intersection - if they share any function/class names, likely equivalent
            common = extracted1 & extracted2
            if common:
                return EquivalenceResult(
                    equivalent=True,
                    confidence=0.75,
                    reasoning=f"Same file ({fp1}) with common symbols in description: {common}",
                    method="scope_match_extracted"
                )
            else:
                # No common symbols - different scopes
                return EquivalenceResult(
                    equivalent=False,
                    confidence=0.7,
                    reasoning=f"Same file ({fp1}) but different symbols: {extracted1} vs {extracted2}",
                    method="scope_match_extracted"
                )
        
        # No scope info available - can't determine
        return None

    def _check_terminal_similarity(self, state1: 'State', state2: 'State') -> Optional[EquivalenceResult]:
        """
        Check if two terminal commands are semantically equivalent.
        
        Examples of equivalent commands:
        - "python main.py" vs "python3 main.py"
        - "npm install" vs "npm i"
        - "cd /workspace && python main.py" vs "python main.py"
        
        Returns:
            EquivalenceResult if determination can be made, None otherwise.
        """
        entry1 = getattr(state1, 'log_entry', None)
        entry2 = getattr(state2, 'log_entry', None)
        
        if not entry1 or not entry2:
            return None
        
        args1 = getattr(entry1, 'args', {}) or {}
        args2 = getattr(entry2, 'args', {}) or {}
        
        cmd1 = args1.get('command', '').strip()
        cmd2 = args2.get('command', '').strip()
        
        if not cmd1 or not cmd2:
            return None
        
        # Normalize commands
        def normalize_command(cmd: str) -> str:
            """Normalize a command for comparison."""
            # Remove cd prefix
            import re
            cmd = re.sub(r'^cd\s+[^\s&;]+\s*[&;]+\s*', '', cmd)
            # Normalize python/python3
            cmd = re.sub(r'\bpython3?\b', 'python', cmd)
            # Normalize npm aliases
            cmd = re.sub(r'\bnpm i\b', 'npm install', cmd)
            # Remove extra whitespace
            cmd = ' '.join(cmd.split())
            return cmd.lower()
        
        norm1 = normalize_command(cmd1)
        norm2 = normalize_command(cmd2)
        
        if norm1 == norm2:
            return EquivalenceResult(
                equivalent=True,
                confidence=0.9,
                reasoning=f"Equivalent terminal commands: '{cmd1[:50]}' ≈ '{cmd2[:50]}'",
                method="terminal_similarity"
            )
        
        # Check if same base command with different args
        base1 = norm1.split()[0] if norm1 else ''
        base2 = norm2.split()[0] if norm2 else ''
        
        if base1 == base2:
            # Same base command - might be equivalent with different args
            return EquivalenceResult(
                equivalent=True,
                confidence=0.75,
                reasoning=f"Same base command '{base1}' with different args",
                method="terminal_similarity"
            )
        
        return None

    def _check_heuristic_match(self, state1: 'State', state2: 'State') -> EquivalenceResult:
        """
        Check for heuristic match based on tool and key arguments.
        
        Uses pattern matching for common equivalence cases.
        """
        tool1 = getattr(state1, 'tool_used', None)
        tool2 = getattr(state2, 'tool_used', None)
        
        # Different tools = not equivalent
        if tool1 != tool2:
            return EquivalenceResult(
                equivalent=False,
                confidence=0.95,
                reasoning=f"Different tools: {tool1} vs {tool2}",
                method="heuristic"
            )
        
        # Get log entries for detailed comparison
        entry1 = state1.log_entry if hasattr(state1, 'log_entry') else None
        entry2 = state2.log_entry if hasattr(state2, 'log_entry') else None
        
        if entry1 is None or entry2 is None:
            # Check if one or both are initial states (no log entry is expected)
            is_initial1 = hasattr(state1, 'metadata') and state1.metadata.get('is_initial')
            is_initial2 = hasattr(state2, 'metadata') and state2.metadata.get('is_initial')
            
            if is_initial1 and is_initial2:
                # Both are initial states - definitely equivalent
                return EquivalenceResult(
                    equivalent=True,
                    confidence=1.0,
                    reasoning="Both are initial states",
                    method="heuristic"
                )
            elif is_initial1 or is_initial2:
                # One is initial, one is not - definitely not equivalent
                return EquivalenceResult(
                    equivalent=False,
                    confidence=0.95,
                    reasoning="One is initial state, other is not",
                    method="heuristic"
                )
            
            # If both have the same tool_used, we can make a reasonable comparison
            # even without full log entries - same tool with same files = likely equivalent
            if tool1 is not None and tool1 == tool2:
                # Try to compare using available state attributes
                file1 = getattr(state1, 'file_path', '') or ''
                file2 = getattr(state2, 'file_path', '') or ''
                
                if file1 and file2:
                    if file1 == file2:
                        return EquivalenceResult(
                            equivalent=True,
                            confidence=0.85,
                            reasoning=f"Same tool '{tool1}' on same file '{file1}'",
                            method="heuristic"
                        )
                    else:
                        return EquivalenceResult(
                            equivalent=False,
                            confidence=0.85,
                            reasoning=f"Same tool '{tool1}' on different files: '{file1}' vs '{file2}'",
                            method="heuristic"
                        )
                
                # Same tool but no file info - moderate confidence it's equivalent
                return EquivalenceResult(
                    equivalent=True,
                    confidence=0.75,
                    reasoning=f"Same tool '{tool1}' (missing detailed comparison data)",
                    method="heuristic"
                )
            
            # Neither is marked as initial but missing log entry - unexpected edge case
            # Use low confidence to trigger LLM verification
            return EquivalenceResult(
                equivalent=tool1 == tool2,
                confidence=0.5,
                reasoning="Missing log entries for non-initial states - uncertain comparison",
                method="heuristic"
            )
        
        # For tool calls, compare key arguments
        if entry1.kind == "toolCall" and entry2.kind == "toolCall":
            return self._compare_tool_calls(entry1, entry2)
        
        # For requests, compare model and general intent
        if entry1.kind == "request" and entry2.kind == "request":
            return self._compare_requests(entry1, entry2)
        
        # Mixed kinds
        return EquivalenceResult(
            equivalent=False,
            confidence=0.8,
            reasoning=f"Different entry kinds: {entry1.kind} vs {entry2.kind}",
            method="heuristic"
        )
    
    def _compare_tool_calls(self, entry1, entry2) -> EquivalenceResult:
        """Compare two tool call entries."""
        # Same tool is prerequisite (already checked in caller)
        tool = entry1.tool
        
        args1 = entry1.args or {}
        args2 = entry2.args or {}
        
        # Tool-specific comparison rules
        if tool in ["read_file", "create_file", "replace_string_in_file"]:
            # File operations: compare file paths
            path1 = args1.get("filePath") or args1.get("path", "")
            path2 = args2.get("filePath") or args2.get("path", "")
            
            # Normalize paths (remove leading slashes, workspace prefixes)
            path1_norm = self._normalize_path(path1)
            path2_norm = self._normalize_path(path2)
            
            if path1_norm == path2_norm:
                return EquivalenceResult(
                    equivalent=True,
                    confidence=0.9,
                    reasoning=f"Same file operation on {path1_norm}",
                    method="heuristic"
                )
            else:
                return EquivalenceResult(
                    equivalent=False,
                    confidence=0.9,
                    reasoning=f"Different files: {path1_norm} vs {path2_norm}",
                    method="heuristic"
                )
        
        elif tool in ["grep_search", "semantic_search", "file_search"]:
            # Search operations: compare queries
            query1 = args1.get("query", "")
            query2 = args2.get("query", "")
            
            # Exact query match
            if query1 == query2:
                return EquivalenceResult(
                    equivalent=True,
                    confidence=0.9,
                    reasoning="Same search query",
                    method="heuristic"
                )
            
            # Similar query (word overlap)
            similarity = self._word_similarity(query1, query2)
            if similarity > 0.8:
                return EquivalenceResult(
                    equivalent=True,
                    confidence=similarity * 0.9,
                    reasoning=f"Similar search queries (similarity={similarity:.2f})",
                    method="heuristic"
                )
            
            return EquivalenceResult(
                equivalent=False,
                confidence=0.7,
                reasoning="Different search queries",
                method="heuristic"
            )
        
        elif tool in ["run_in_terminal"]:
            # Terminal commands: compare commands
            cmd1 = args1.get("command", "")
            cmd2 = args2.get("command", "")
            
            # Normalize commands (strip whitespace, normalize paths)
            cmd1_norm = self._normalize_command(cmd1)
            cmd2_norm = self._normalize_command(cmd2)
            
            if cmd1_norm == cmd2_norm:
                return EquivalenceResult(
                    equivalent=True,
                    confidence=0.9,
                    reasoning="Same terminal command",
                    method="heuristic"
                )
            
            return EquivalenceResult(
                equivalent=False,
                confidence=0.8,
                reasoning="Different terminal commands",
                method="heuristic"
            )
        
        elif tool in ["list_dir"]:
            # Directory listing: compare paths
            path1 = args1.get("path", "")
            path2 = args2.get("path", "")
            
            path1_norm = self._normalize_path(path1)
            path2_norm = self._normalize_path(path2)
            
            return EquivalenceResult(
                equivalent=path1_norm == path2_norm,
                confidence=0.9,
                reasoning=f"Directory listing: {'same' if path1_norm == path2_norm else 'different'} path",
                method="heuristic"
            )
        
        # Default: compare all args
        if args1 == args2:
            return EquivalenceResult(
                equivalent=True,
                confidence=0.85,
                reasoning=f"Identical arguments for {tool}",
                method="heuristic"
            )
        
        # Different args, need LLM or conservative non-equivalence
        return EquivalenceResult(
            equivalent=False,
            confidence=0.6,
            reasoning=f"Different arguments for {tool}, needs semantic comparison",
            method="heuristic"
        )
    
    def _compare_requests(self, entry1, entry2) -> EquivalenceResult:
        """Compare two LLM request entries.
        
        Model names are treated as implementation details. So different models
        producing the same semantic result (e.g., planning) are considered
        equivalent for task-level analysis.
        
        We compare the resulting_state to determine equivalence - if both
        lead to the same abstract result (e.g., "llm_response:planning"),
        they are considered equivalent.
        """
        model1 = entry1.model or "unknown"
        model2 = entry2.model or "unknown"
        
        # We care about the semantic role of the LLM request (planning, confirmation, etc.),
        # not the specific model used. This makes the analysis task-centric.
        if model1 != model2:
            logger.debug(f"Different models ({model1} vs {model2}) treated as equivalent.")
        
        # Check response previews for similarity if available
        resp1 = getattr(entry1, 'response_preview', '') or ''
        resp2 = getattr(entry2, 'response_preview', '') or ''
        
        # If both have response previews, check for keyword overlap
        if resp1 and resp2:
            # Common keywords that indicate similar intent
            key_actions = ['create', 'file', 'search', 'read', 'write', 'edit', 'run', 'server', 'browser', 'terminal']
            
            words1 = set(resp1.lower().split())
            words2 = set(resp2.lower().split())
            
            # Check if both mention similar action keywords
            actions1 = words1 & set(key_actions)
            actions2 = words2 & set(key_actions)
            
            if actions1 and actions2:
                overlap = len(actions1 & actions2) / max(len(actions1 | actions2), 1)
                if overlap >= 0.5:
                    return EquivalenceResult(
                        equivalent=True,
                        confidence=0.85,
                        reasoning=f"Similar LLM intent (action keywords: {actions1 & actions2})",
                        method="heuristic"
                    )
        
        # LLM requests are generally considered equivalent if same position
        # (they represent the agent thinking/planning step)
        # Use confidence 0.85 to skip LLM call - all LLM request states at same
        # position in trajectory are semantically equivalent (just planning/thinking)
        return EquivalenceResult(
            equivalent=True,
            confidence=0.85,
            reasoning="LLM request at same semantic position (model-agnostic)",
            method="heuristic"
        )
    
    def _normalize_path(self, path: str) -> str:
        """Normalize a file path for comparison."""
        if not path:
            return ""
        
        # Remove leading slashes and workspace prefixes
        path = path.lstrip("/\\")
        
        # Remove common prefixes
        prefixes = ["workspace/", "workspaces/", "home/", "tmp/"]
        for prefix in prefixes:
            if path.lower().startswith(prefix):
                path = path[len(prefix):]
        
        # Normalize separators
        path = path.replace("\\", "/")
        
        return path.lower()
    
    def _normalize_command(self, cmd: str) -> str:
        """Normalize a terminal command for comparison."""
        if not cmd:
            return ""
        
        # Strip whitespace
        cmd = cmd.strip()
        
        # Normalize multiple spaces
        import re
        cmd = re.sub(r'\s+', ' ', cmd)
        
        return cmd.lower()
    
    def _word_similarity(self, text1: str, text2: str) -> float:
        """Calculate word-level Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _check_llm_equivalence(self, state1: 'State', state2: 'State') -> Optional[EquivalenceResult]:
        """
        Use LLM to determine semantic equivalence.
        
        Returns EquivalenceResult or None if LLM call fails.
        """
        try:
            try:
                from llm import get_model_and_client
            except ImportError:
                from src.llm import get_model_and_client
        except ImportError:
            logger.warning("LLM module not available, skipping LLM equivalence check")
            return None
        
        # Build comparison prompt
        prompt = self._build_equivalence_prompt(state1, state2)
        
        try:
            client, model, temp = get_model_and_client(self.llm_prefix)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert at analyzing software engineering tasks. 
Your job is to determine if two states in a coding agent's execution represent the same semantic step.

Consider:
1. The tool/action being performed
2. The target (file, query, command)
3. The intent (what the agent is trying to accomplish)
4. The outcome/observation

Two states are EQUIVALENT if:
- They perform the same type of action
- On the same target (or semantically equivalent targets)
- With the same intent
- Leading to the same type of outcome

Two states are NOT EQUIVALENT if:
- They perform different actions
- On different targets
- With different intents
- Leading to different outcomes

Respond with EXACTLY a JSON object with two keys:
- "equivalent": boolean (true or false)
- "reasoning": string (1-2 sentences explaining why)"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Make LLM call with model-appropriate parameters (avoids 400 errors)
            # Use 500 tokens to avoid truncation (finish_reason=length)
            response = _safe_llm_call(client, model, messages, max_tokens=500, temperature=0.1)
            
            # Check for content filtering or empty response
            if not response.choices:
                logger.warning(f"LLM returned no choices")
                return None
            
            choice = response.choices[0]
            if choice.finish_reason == "content_filter":
                logger.warning(f"LLM response blocked by content filter")
                return None
            
            if choice.message.content is None:
                logger.warning(f"LLM returned None content. Finish reason: {choice.finish_reason}")
                return None
                
            content = choice.message.content.strip()
            
            if not content:
                logger.warning(f"LLM returned empty string. Finish reason: {choice.finish_reason}")
                return None
            
            # Debug logging: show what was sent and received
            logger.debug(f"[llm_equivalence] Prompt: {prompt[:200]}...")
            logger.debug(f"[llm_equivalence] LLM response: {content}")
            
            # Parse response
            result = self._parse_llm_response(content)
            if result:
                logger.debug(f"[llm_equivalence] Parsed result: equivalent={result['equivalent']}")
                return EquivalenceResult(
                    equivalent=result["equivalent"],
                    confidence=0.85,
                    reasoning=result["reasoning"],
                    method="llm",
                    metadata={"raw_response": content}
                )
            else:
                # Log parse failure with response snippet for debugging
                content_preview = content[:200] if len(content) > 200 else content
                logger.warning(f"LLM response parse failed. Response: {content_preview}")
            
        except Exception as e:
            logger.warning(f"LLM equivalence check failed: {e}")
        
        return None
    
    def _build_equivalence_prompt(self, state1: 'State', state2: 'State') -> str:
        """Build the comparison prompt for LLM."""
        lines = []
        lines.append("Compare these two states from a coding agent execution:")
        lines.append("")
        lines.append("=== STATE 1 ===")
        lines.append(self._state_to_summary(state1))
        lines.append("")
        lines.append("=== STATE 2 ===")
        lines.append(self._state_to_summary(state2))
        lines.append("")
        lines.append("Are these states semantically equivalent?")
        lines.append("Respond with JSON: {\"equivalent\": true/false, \"reasoning\": \"...\"}")
        
        return "\n".join(lines)
    
    def _state_to_summary(self, state: 'State') -> str:
        """Convert state to a summary string for LLM."""
        lines = []
        
        if hasattr(state, 'tool_used') and state.tool_used:
            lines.append(f"Tool: {state.tool_used}")
        
        if hasattr(state, 'log_entry') and state.log_entry:
            entry = state.log_entry
            if hasattr(entry, 'args') and entry.args:
                # Show key args only
                key_args = {}
                for key in ["filePath", "path", "query", "command"]:
                    if key in entry.args:
                        val = str(entry.args[key])
                        if len(val) > 100:
                            val = val[:100] + "..."
                        key_args[key] = val
                if key_args:
                    lines.append(f"Arguments: {json.dumps(key_args)}")
            
            if hasattr(entry, 'response') and entry.response:
                resp = str(entry.response)
                if len(resp) > 200:
                    resp = resp[:200] + "..."
                lines.append(f"Response: {resp}")
        
        if hasattr(state, 'observation') and state.observation:
            obs = state.observation
            if len(obs) > 300:
                obs = obs[:300] + "..."
            lines.append(f"Observation: {obs}")
        
        if hasattr(state, 'files_touched') and state.files_touched:
            lines.append(f"Files touched: {list(state.files_touched)[:5]}")
        
        return "\n".join(lines) if lines else "(no details available)"
    
    def _call_llm_with_retry(self, client, model: str, messages: list, max_tokens: int = 200):
        """Call LLM with proper parameter handling for different models.
        
        Some models like gpt-5.2-chat only support temperature=1 and require
        max_completion_tokens instead of max_tokens. This method handles
        these variations gracefully.
        """
        # Determine temperature based on model
        # gpt-5.2-chat only supports temperature=1, others can use 0 for determinism
        if "5.2" in model or "5-2" in model:
            temperature = 1  # Only allowed value for gpt-5.2-chat
        else:
            temperature = 0  # Zero temperature for deterministic results
        
        # First try: Use max_completion_tokens (newer API) with appropriate temperature
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens
            )
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle temperature not supported
            if "temperature" in error_str:
                try:
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_completion_tokens=max_tokens
                    )
                except Exception:
                    pass
            
            # If max_completion_tokens isn't supported, try max_tokens
            if "max_completion_tokens" in error_str or isinstance(e, TypeError):
                try:
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except Exception:
                    # Last resort: no temperature, legacy max_tokens
                    try:
                        return client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens
                        )
                    except Exception:
                        pass
            
            # Re-raise the original error if we couldn't handle it
            raise
    
    def _parse_llm_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response JSON."""
        try:
            import re
            
            # Strip markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            
            # Method 1: Try to parse the entire content as JSON
            try:
                result = json.loads(content)
                if "equivalent" in result and "reasoning" in result:
                    return {
                        "equivalent": bool(result["equivalent"]),
                        "reasoning": str(result["reasoning"])
                    }
            except json.JSONDecodeError:
                pass
            
            # Method 2: Find JSON by matching balanced braces
            # Find the first { and then find its matching }
            start = content.find('{')
            if start != -1:
                brace_count = 0
                for i, char in enumerate(content[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = content[start:i+1]
                            try:
                                result = json.loads(json_str)
                                if "equivalent" in result and "reasoning" in result:
                                    return {
                                        "equivalent": bool(result["equivalent"]),
                                        "reasoning": str(result["reasoning"])
                                    }
                            except json.JSONDecodeError:
                                pass
                            break
            
            # Method 3: Extract key-value pairs with regex as fallback
            equiv_match = re.search(r'"equivalent"\s*:\s*(true|false)', content, re.IGNORECASE)
            reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content)
            
            if equiv_match and reason_match:
                return {
                    "equivalent": equiv_match.group(1).lower() == "true",
                    "reasoning": reason_match.group(1)
                }
                
        except Exception as e:
            logger.debug(f"Failed to parse LLM response: {e}")
        
        return None
    
    def _record_comparison(self, state1: 'State', state2: 'State', result: EquivalenceResult):
        """Record a comparison for debugging/analysis."""
        record = {
            "state1_id": state1.state_id if hasattr(state1, 'state_id') else "unknown",
            "state2_id": state2.state_id if hasattr(state2, 'state_id') else "unknown",
            "tool1": getattr(state1, 'tool_used', None),
            "tool2": getattr(state2, 'tool_used', None),
            "result": result.to_dict()
        }
        self._comparison_records.append(record)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about equivalence checks."""
        return self.stats.copy()
    
    def get_comparison_records(self) -> List[Dict[str, Any]]:
        """Get all comparison records for analysis."""
        return self._comparison_records.copy()
