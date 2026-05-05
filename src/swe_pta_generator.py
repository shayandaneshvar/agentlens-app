"""
SWE PTA Generator - Generates PTAs from coding agent trajectory files.

Usage:
    python swe_pta_generator.py <trajectory_json_file> [--output <output_file>]
    
The trajectory file should be a chat-export-logs.json from coding agent runs.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from swe_models import LogEntry, State, Transition, PTA
from code_analyzer import get_analyzer

# Import tree-sitter based code analyzer for better content descriptions
from code_analyzer import compare_code, describe_file_creation, analyze_code

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS FOR LOCATION AND SCOPE EXTRACTION
# ============================================================================

def normalize_file_path(path: str) -> str:
    """
    Normalize file path for consistent comparison.
    - Lowercase
    - Forward slashes
    - Remove common prefixes (workspace/, home/, etc.)
    """
    if not path:
        return ""
    path = path.lstrip("/\\")
    # Remove common prefixes
    for prefix in ["workspace/", "workspaces/", "home/", "tmp/", "c:/", "d:/"]:
        if path.lower().startswith(prefix):
            path = path[len(prefix):]
    return path.lower().replace("\\", "/")


def extract_line_range_from_read(args: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """
    Extract line range from read_file arguments.
    
    Args have startLine and endLine (1-indexed).
    """
    start = args.get("startLine")
    end = args.get("endLine")
    
    if start is not None and end is not None:
        return (int(start), int(end))
    elif start is not None:
        return (int(start), int(start))
    return None


def extract_line_range_from_edit(old_string: str, new_string: str) -> Optional[Tuple[int, int]]:
    """
    Estimate line range from edit operation.
    
    Since we don't know the actual line numbers without file content,
    we return the number of lines affected as a proxy.
    Returns (1, num_lines) to indicate relative scope.
    """
    if not old_string:
        return None
    
    old_lines = old_string.count('\n') + 1
    new_lines = new_string.count('\n') + 1 if new_string else 0
    
    # Return the max of old/new as the scope of the edit
    scope = max(old_lines, new_lines)
    return (1, scope)


def compute_relative_position(line_range: Optional[Tuple[int, int]], file_length: int = 100) -> str:
    """
    Compute relative position in file.
    
    Args:
        line_range: (start, end) line numbers
        file_length: Estimated file length (default 100 if unknown)
    
    Returns:
        "early" (top 33%), "middle", or "late" (bottom 33%)
    """
    if not line_range or file_length <= 0:
        return ""
    
    start, end = line_range
    mid_line = (start + end) / 2
    position_ratio = mid_line / file_length
    
    if position_ratio <= 0.33:
        return "early"
    elif position_ratio <= 0.66:
        return "middle"
    else:
        return "late"


def determine_operation_type(tool: str) -> str:
    """
    Determine operation type from tool name.
    """
    tool = tool.lower() if tool else ""
    
    if tool in ("read_file",):
        return "read"
    elif tool in ("create_file",):
        return "create"
    elif tool in ("replace_string_in_file", "multi_replace_string_in_file", "apply_patch", "edit_file"):
        return "modify"
    elif tool in ("delete_file", "remove_file"):
        return "delete"
    elif tool in ("file_search", "grep_search", "semantic_search", "list_dir"):
        return "search"
    elif tool in ("run_in_terminal", "get_terminal_output"):
        return "terminal"
    else:
        return "other"


def determine_edit_type(old_string: str, new_string: str) -> str:
    """
    Determine edit type based on old and new strings.
    
    Returns:
        "add" - pure addition (no old content)
        "remove" - pure deletion (no new content)
        "replace" - similar size change
        "expand" - new content is significantly larger
        "shrink" - new content is significantly smaller
    """
    if not old_string and new_string:
        return "add"
    elif old_string and not new_string:
        return "remove"
    elif not old_string and not new_string:
        return ""
    
    old_len = len(old_string)
    new_len = len(new_string)
    
    if new_len > old_len * 1.5:
        return "expand"
    elif new_len < old_len * 0.5:
        return "shrink"
    else:
        return "replace"


def extract_scope_from_code(code: str, filename: str = "") -> Dict[str, str]:
    """
    Extract function and class names from a code snippet.
    
    Uses tree-sitter via code_analyzer for robust multi-language parsing.
    
    Args:
        code: Source code string
        filename: Optional filename for language detection
    
    Returns:
        Dict with 'function_name', 'class_name', 'scope_path'
    """
    result = {
        'function_name': '',
        'class_name': '',
        'scope_path': ''
    }
    
    if not code:
        return result
    
    # Use the code analyzer for tree-sitter based parsing
    analyzer = get_analyzer()
    analysis = analyzer.analyze(code, filename=filename)
    
    # Get the first function and class names
    func_names = analysis.get_function_names()
    class_names = analysis.get_class_names()
    
    if func_names:
        result['function_name'] = sorted(func_names)[0]
    if class_names:
        result['class_name'] = sorted(class_names)[0]
    
    # Build scope path using the analysis results
    # Check for methods (functions with parent class)
    for func in analysis.functions:
        if func.parent:
            # This is a method - use Class.method format
            result['scope_path'] = f"{func.parent}.{func.name}"
            result['class_name'] = func.parent
            result['function_name'] = func.name
            break
    
    # If no method found, use simple scope path
    if not result['scope_path']:
        if result['class_name'] and result['function_name']:
            result['scope_path'] = f"{result['class_name']}.{result['function_name']}"
        elif result['function_name']:
            result['scope_path'] = result['function_name']
        elif result['class_name']:
            result['scope_path'] = result['class_name']
    
    return result


def extract_location_info(entry: 'LogEntry') -> Dict[str, Any]:
    """
    Extract all location and scope information from a log entry.
    
    Returns dict with:
        - file_path: normalized file path
        - line_range: (start, end) or None
        - relative_position: early/middle/late
        - operation_type: read/create/modify/delete/search/terminal/other
        - edit_type: add/remove/replace/expand/shrink (for edits only)
        - function_name: extracted function name
        - class_name: extracted class name  
        - scope_path: combined scope path
    """
    result = {
        'file_path': '',
        'line_range': None,
        'relative_position': '',
        'operation_type': '',
        'edit_type': '',
        'function_name': '',
        'class_name': '',
        'scope_path': ''
    }
    
    if entry.kind != "toolCall":
        return result
    
    tool = entry.tool or ""
    args = entry.args or {}
    
    # Get file path
    file_path = args.get("filePath") or args.get("path", "")
    result['file_path'] = normalize_file_path(file_path)
    
    # Get operation type
    result['operation_type'] = determine_operation_type(tool)
    
    # Tool-specific extraction
    if tool == "read_file":
        result['line_range'] = extract_line_range_from_read(args)
        # Estimate file length from endLine if available
        end_line = args.get("endLine", 100)
        result['relative_position'] = compute_relative_position(result['line_range'], end_line * 1.5)
        
    elif tool == "replace_string_in_file":
        old_str = args.get("oldString", "")
        new_str = args.get("newString", "")
        
        result['line_range'] = extract_line_range_from_edit(old_str, new_str)
        result['edit_type'] = determine_edit_type(old_str, new_str)
        
        # Extract scope from the old string (context around the edit)
        scope_info = extract_scope_from_code(old_str, file_path)
        result['function_name'] = scope_info['function_name']
        result['class_name'] = scope_info['class_name']
        result['scope_path'] = scope_info['scope_path']
        
    elif tool == "multi_replace_string_in_file":
        replacements = args.get("replacements", [])
        if replacements:
            # Aggregate info from all replacements
            all_files = set()
            total_lines = 0
            all_functions = []
            all_classes = []
            
            for repl in replacements:
                path = repl.get("filePath") or repl.get("path", "")
                if path:
                    all_files.add(normalize_file_path(path))
                
                old_str = repl.get("oldString", "")
                new_str = repl.get("newString", "")
                
                if old_str:
                    total_lines += old_str.count('\n') + 1
                    scope_info = extract_scope_from_code(old_str, path)
                    if scope_info['function_name']:
                        all_functions.append(scope_info['function_name'])
                    if scope_info['class_name']:
                        all_classes.append(scope_info['class_name'])
            
            # Set aggregated values
            result['file_path'] = ','.join(sorted(all_files)) if all_files else ''
            result['line_range'] = (1, total_lines) if total_lines > 0 else None
            result['edit_type'] = "replace"  # Multi-replace is always a replace operation
            
            if all_functions:
                result['function_name'] = ','.join(sorted(set(all_functions)))
            if all_classes:
                result['class_name'] = ','.join(sorted(set(all_classes)))
            if all_functions or all_classes:
                result['scope_path'] = result['function_name'] or result['class_name']
                
    elif tool == "create_file":
        content = args.get("content", "")
        if content:
            lines = content.count('\n') + 1
            result['line_range'] = (1, lines)
            result['edit_type'] = "add"
            
            # Extract scope from created content
            scope_info = extract_scope_from_code(content, file_path)
            result['function_name'] = scope_info['function_name']
            result['class_name'] = scope_info['class_name']
            result['scope_path'] = scope_info['scope_path']
            
    elif tool == "apply_patch":
        patch_input = args.get("input", "") or args.get("patch", "")
        if patch_input:
            # Count lines in patch
            lines = str(patch_input).count('\n') + 1
            result['line_range'] = (1, lines)
            result['edit_type'] = "replace"
            
            # Try to extract file path from patch
            for line in str(patch_input).split("\n"):
                if line.startswith("*** Update File:") or line.startswith("+++ "):
                    path = line.split(":", 1)[-1].strip() if ":" in line else line[4:].strip()
                    result['file_path'] = normalize_file_path(path)
                    break
    
    return result


# ============================================================================
# PTA GENERATOR CLASS
# ============================================================================

class PTAGenerator:
    """
    Generates a PTA from a coding agent trajectory.
    
    The PTA generation process:
    1. Parse the chat-export-logs.json file
    2. Extract relevant log entries (first, toolCalls, last)
    3. Create states from observations (tool responses)
    4. Create transitions from actions (tool calls)
    """
    
    def __init__(self, include_requests: bool = False):
        """
        Initialize the generator.
        
        Args:
            include_requests: Whether to include LLM request entries as states/transitions.
                             Default False - only tool calls are included.
        """
        self.include_requests = include_requests
        self.state_counter = 0
        self.transition_counter = 0
    
    def generate_pta(self, trajectory_file: str) -> PTA:
        """
        Generate a PTA from a trajectory file.
        
        Args:
            trajectory_file: Path to chat-export-logs.json
            
        Returns:
            PTA object
        """
        logger.info(f"Generating PTA from: {trajectory_file}")
        
        # Load trajectory
        with open(trajectory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract log entries
        log_entries = self._extract_log_entries(data)
        logger.info(f"Extracted {len(log_entries)} relevant log entries")
        
        # Build PTA
        pta = self._build_pta(log_entries, trajectory_file)
        
        return pta
    
    def _extract_log_entries(self, data: Dict[str, Any]) -> List[LogEntry]:
        """
        Extract relevant log entries from trajectory data.
        
        Strategy: Get first entry, all toolCall entries, and last entry.
        """
        entries = []
        
        prompts = data.get("prompts", [])
        if not prompts:
            logger.warning("No prompts found in trajectory")
            return entries
        
        # Concatenate logs from all prompts
        all_logs = []
        for prompt in prompts:
            logs = prompt.get("logs", [])
            all_logs.extend(logs)
        
        if not all_logs:
            logger.warning("No logs found in any prompt")
            return entries
        
        # Extract entries based on our criteria
        for idx, log in enumerate(all_logs):
            kind = log.get("kind", "")
            
            # Always include first entry
            if idx == 0:
                entries.append(LogEntry.from_log(log, idx))
                continue
            
            # Always include last entry
            if idx == len(all_logs) - 1:
                entries.append(LogEntry.from_log(log, idx))
                continue
            
            # Include all toolCall entries
            if kind == "toolCall":
                entries.append(LogEntry.from_log(log, idx))
                continue
            
            # Optionally include request entries
            if self.include_requests and kind == "request":
                entries.append(LogEntry.from_log(log, idx))
        
        return entries
    
    def _build_pta(self, log_entries: List[LogEntry], source_file: str) -> PTA:
        """
        Build a PTA from log entries.
        """
        pta = PTA()
        pta.metadata = {
            "source_file": Path(source_file).name,
            "source_path": str(source_file),
            "num_entries": len(log_entries),
            "generator": "swe_pta_generator"
        }
        
        if not log_entries:
            return pta
        
        # Create initial state
        prev_state = self._create_initial_state()
        pta.add_state(prev_state)
        
        # Process each log entry
        for idx, entry in enumerate(log_entries):
            is_last = (idx == len(log_entries) - 1)
            
            # Create state from entry, passing previous state for context
            new_state = self._create_state_from_entry(entry, prev_state, is_last)
            pta.add_state(new_state)
            
            # Create transition
            transition = self._create_transition(prev_state, new_state, entry)
            pta.add_transition(transition)
            
            prev_state = new_state
        
        # Mark terminal state
        if prev_state:
            prev_state.metadata["is_terminal"] = True
        
        return pta
    
    def _create_initial_state(self) -> State:
        """Create the initial state."""
        state_id = f"state_{self.state_counter}"
        self.state_counter += 1
        
        return State(
            state_id=state_id,
            step=0,
            observation="<initial>",
            resulting_state="initial",
            metadata={"is_initial": True}
        )
    
    def _create_state_from_entry(self, entry: LogEntry, prev_state: State = None, is_terminal: bool = False) -> State:
        """Create a state from a log entry."""
        state_id = f"state_{self.state_counter}"
        self.state_counter += 1
        
        # Build observation from entry (action-based, includes details)
        observation = self._build_observation(entry)
        
        # Build resulting state (result-based, abstracts away details)
        # Pass previous state and terminal flag for context-aware LLM responses
        resulting_state = self._compute_resulting_state(entry, prev_state, is_terminal)
        
        # Compute content hash for fine-grained matching
        content_hash = self._compute_content_hash(entry)
        
        # Compute content description for semantic comparison
        content_description = self._compute_content_description(entry)
        
        # Extract files touched
        files_touched = self._extract_files_touched(entry)
        
        # NEW: Extract location and scope information
        location_info = extract_location_info(entry)
        
        state = State(
            state_id=state_id,
            step=self.state_counter,
            log_entry=entry,
            observation=observation,
            resulting_state=resulting_state,
            content_hash=content_hash,
            content_description=content_description,
            files_touched=files_touched,
            tool_used=entry.tool if entry.kind == "toolCall" else None,
            # NEW: Location and scope fields
            file_path=location_info['file_path'],
            line_range=location_info['line_range'],
            relative_position=location_info['relative_position'],
            operation_type=location_info['operation_type'],
            edit_type=location_info['edit_type'],
            function_name=location_info['function_name'],
            class_name=location_info['class_name'],
            scope_path=location_info['scope_path'],
            metadata={
                "entry_id": entry.id,
                "entry_kind": entry.kind,
                "entry_index": entry.index
            }
        )
        
        return state
    
    def _compute_resulting_state(self, entry: LogEntry, prev_state: State = None, is_terminal: bool = False) -> str:
        """
        Compute the resulting state after a tool call.
        
        This abstracts away the implementation details (like exact content)
        and focuses on the semantic result (e.g., "file was created").
        
        For LLM responses, we differentiate:
        - Planning responses (before actions): "llm_response:planning"
        - Confirmation responses (after actions, terminal): "llm_response:confirmed:<prev_result>"
        
        Args:
            entry: The current log entry
            prev_state: The previous state (for context)
            is_terminal: Whether this is the last entry in the trajectory
        
        Returns a normalized string representing the world state after this action.
        """
        if entry.kind == "toolCall":
            tool = entry.tool or "unknown"
            args = entry.args or {}
            response = entry.response
            
            # Normalize file path for consistency
            def normalize_path(path: str) -> str:
                if not path:
                    return ""
                path = path.lstrip("/\\")
                # Remove common prefixes
                for prefix in ["workspace/", "workspaces/", "home/", "tmp/"]:
                    if path.lower().startswith(prefix):
                        path = path[len(prefix):]
                return path.lower().replace("\\", "/")
            
            # File creation: result is "file now exists at path"
            if tool == "create_file":
                path = args.get("filePath") or args.get("path", "")
                norm_path = normalize_path(path)
                # Check if successful
                success = response and "successfully" in str(response).lower()
                if success:
                    return f"file_created:{norm_path}"
                else:
                    return f"file_create_failed:{norm_path}"
            
            # File reading: result is "file was read from path"
            elif tool == "read_file":
                path = args.get("filePath") or args.get("path", "")
                norm_path = normalize_path(path)
                return f"file_read:{norm_path}"
            
            # File modification: result is "file was modified at path"
            elif tool == "replace_string_in_file":
                path = args.get("filePath") or args.get("path", "")
                norm_path = normalize_path(path)
                success = response and "successfully" in str(response).lower()
                if success:
                    return f"file_modified:{norm_path}"
                else:
                    return f"file_modify_failed:{norm_path}"
            
            # File search: result is "found N files" or "not found"
            elif tool == "file_search":
                query = args.get("query", "")
                resp_str = str(response) if response else ""
                if "No files found" in resp_str or not response:
                    return f"file_search:not_found:{query.lower()}"
                else:
                    # Count results if possible
                    return f"file_search:found:{query.lower()}"
            
            # Grep search: result is "found matches" or "no matches"
            elif tool == "grep_search":
                query = args.get("query", "")
                resp_str = str(response) if response else ""
                if not response or "no matches" in resp_str.lower() or resp_str == "[]":
                    return f"grep_search:no_matches"
                else:
                    return f"grep_search:found_matches"
            
            # Semantic search
            elif tool == "semantic_search":
                query = args.get("query", "")
                if response:
                    return f"semantic_search:results"
                else:
                    return f"semantic_search:no_results"
            
            # Terminal command: result is exit code
            elif tool == "run_in_terminal":
                command = args.get("command", "")
                # Extract base command (first word)
                base_cmd = command.strip().split()[0] if command.strip() else "unknown"
                # For now, just indicate command type
                return f"terminal:{base_cmd.lower()}"
            
            # Directory listing
            elif tool == "list_dir":
                path = args.get("path", "")
                norm_path = normalize_path(path)
                return f"dir_listed:{norm_path}"
            
            # Apply patch: extract file path from patch input
            elif tool == "apply_patch":
                patch_input = args.get("input", "")
                # Try to extract file path from patch format
                file_path = ""
                for line in str(patch_input).split("\n"):
                    if line.startswith("*** Update File:") or line.startswith("+++ "):
                        file_path = line.split(":", 1)[-1].strip() if ":" in line else line[4:].strip()
                        break
                if file_path:
                    norm_path = normalize_path(file_path)
                    return f"file_patched:{norm_path}"
                else:
                    return "file_patched:unknown"
            
            # Multi-replace: extract file paths from replacements
            elif tool == "multi_replace_string_in_file":
                replacements = args.get("replacements", [])
                file_paths = []
                for repl in replacements:
                    if isinstance(repl, dict):
                        path = repl.get("filePath") or repl.get("path", "")
                        if path:
                            file_paths.append(normalize_path(path))
                if file_paths:
                    # Sort for consistent ordering
                    file_paths = sorted(set(file_paths))
                    return f"files_modified:{','.join(file_paths)}"
                else:
                    return "files_modified:unknown"
            
            # Default for unknown tools - include success/error status
            else:
                # Check for error indicators in response
                resp_str = str(response) if response else ""
                is_error = any(err in resp_str.lower() for err in [
                    "error", "failed", "invalid", "exception", 
                    "not found", "denied", "timeout", "refused"
                ])
                status = "error" if is_error else "success"
                return f"tool_result:{tool}:{status}"
        
        elif entry.kind == "request":
            # Differentiate between planning and confirmation LLM responses
            # based on context (previous state and terminal status)
            
            if is_terminal and prev_state:
                # This is a confirmation/summary response at the end of trajectory
                # Link it to what action it's confirming
                prev_result = prev_state.resulting_state or ""
                
                # Skip linking to other LLM responses or initial states
                if prev_result and prev_result not in ("initial", "llm_response", "llm_response:planning") and not prev_result.startswith("llm_response:"):
                    return f"llm_response:confirmed:{prev_result}"
                else:
                    return "llm_response:terminal"
            
            elif prev_state and prev_state.resulting_state == "initial":
                # First LLM response after initial state = planning
                return "llm_response:planning"
            
            else:
                # Mid-trajectory LLM response (could be follow-up planning)
                return "llm_response:planning"
        
        return "unknown"
    
    def _compute_content_hash(self, entry: LogEntry) -> str:
        """
        Compute a content hash for file modification operations.
        
        This provides fine-grained differentiation for matching:
        - Two edits to the same file with different content will have different hashes
        - This allows merging based on coarse resulting_state (file_modified:main.py)
          while matching can use content_hash to detect different edits
        
        Returns:
            A short hash string, or empty string if not applicable
        """
        import hashlib
        
        if entry.kind != "toolCall":
            return ""
        
        tool = entry.tool or ""
        args = entry.args or {}
        
        # For file creation: hash the content
        if tool == "create_file":
            content = args.get("content", "")
            if content:
                return hashlib.md5(content.encode()).hexdigest()[:12]
        
        # For single file replacement: hash oldString + newString
        elif tool == "replace_string_in_file":
            old_str = args.get("oldString", "")
            new_str = args.get("newString", "")
            combined = f"{old_str}||{new_str}"
            return hashlib.md5(combined.encode()).hexdigest()[:12]
        
        # For multi-replace: hash all replacements
        elif tool == "multi_replace_string_in_file":
            replacements = args.get("replacements", [])
            if replacements:
                # Sort by filePath for consistent ordering
                sorted_repls = sorted(replacements, key=lambda r: r.get("filePath", ""))
                parts = []
                for repl in sorted_repls:
                    old_str = repl.get("oldString", "")
                    new_str = repl.get("newString", "")
                    file_path = repl.get("filePath", "")
                    parts.append(f"{file_path}:{old_str}||{new_str}")
                combined = "|||".join(parts)
                return hashlib.md5(combined.encode()).hexdigest()[:12]
        
        # For apply_patch: hash the patch content
        elif tool == "apply_patch":
            patch_input = args.get("input", "") or args.get("patch", "")
            if patch_input:
                return hashlib.md5(str(patch_input).encode()).hexdigest()[:12]
        
        # For terminal commands: hash the command (output is too variable)
        elif tool == "run_in_terminal":
            command = args.get("command", "")
            if command:
                return hashlib.md5(command.encode()).hexdigest()[:12]
        
        return ""
    
    def _compute_content_description(self, entry: LogEntry) -> str:
        """
        Compute a human-readable description of what the file change does.
        
        This is used for semantic comparison when content hashes differ.
        The description captures the intent/effect of the change, not the exact content.
        
        Uses tree-sitter based code analyzer for robust multi-language support.
        
        Returns:
            A description string, or empty string if not applicable
        """
        if entry.kind != "toolCall":
            return ""
        
        tool = entry.tool or ""
        args = entry.args or {}
        
        def normalize_path(path: str) -> str:
            """Extract just the filename from a path."""
            if not path:
                return "unknown"
            return path.split("/")[-1].split("\\")[-1]
        
        analyzer = get_analyzer()
        
        # For file creation: use analyzer to describe
        if tool == "create_file":
            path = args.get("filePath") or args.get("path", "")
            content = args.get("content", "")
            return analyzer.describe_file_creation(content, path)
        
        # For single file replacement: use analyzer to compare
        elif tool == "replace_string_in_file":
            path = args.get("filePath") or args.get("path", "")
            filename = normalize_path(path)
            old_str = args.get("oldString", "")
            new_str = args.get("newString", "")
            
            comparison = analyzer.compare_code(old_str, new_str, path)
            return f"In '{filename}': {comparison['description']}"
        
        # For multi-replace: analyze each replacement
        elif tool == "multi_replace_string_in_file":
            replacements = args.get("replacements", [])
            if not replacements:
                return ""
            
            descriptions = []
            for repl in replacements[:5]:
                path = repl.get("filePath") or repl.get("path", "")
                filename = normalize_path(path)
                old_str = repl.get("oldString", "")
                new_str = repl.get("newString", "")
                
                comparison = analyzer.compare_code(old_str, new_str, path)
                descriptions.append(f"In '{filename}': {comparison['description']}")
            
            if len(replacements) > 5:
                descriptions.append(f"... and {len(replacements) - 5} more changes")
            
            return "; ".join(descriptions)
        
        # For apply_patch: describe the patch
        elif tool == "apply_patch":
            patch_input = args.get("input", "") or args.get("patch", "")
            if not patch_input:
                return ""
            
            # Try to extract file path from patch
            for line in str(patch_input).split("\n"):
                if line.startswith("*** Update File:") or line.startswith("+++ "):
                    file_path = line.split(":", 1)[-1].strip() if ":" in line else line[4:].strip()
                    filename = normalize_path(file_path)
                    return f"Applied patch to '{filename}'"
            
            return "Applied patch"
        
        # For terminal commands: describe the command
        elif tool == "run_in_terminal":
            command = args.get("command", "")
            if not command:
                return ""
            
            # Extract base command
            base_cmd = command.strip().split()[0] if command.strip() else ""
            
            if "python" in base_cmd.lower():
                return f"Ran Python command: {command[:100]}"
            elif "npm" in base_cmd.lower() or "node" in base_cmd.lower():
                return f"Ran Node.js command: {command[:100]}"
            elif "git" in base_cmd.lower():
                return f"Ran Git command: {command[:100]}"
            else:
                return f"Ran terminal command: {command[:100]}"
        
        return ""
    
    def _build_observation(self, entry: LogEntry) -> str:
        """
        Build observation text from a log entry.
        This is what will be used for LLM-based equivalence checking.
        """
        parts = []
        
        if entry.kind == "toolCall":
            parts.append(f"Tool: {entry.tool}")
            
            # Add key arguments
            if entry.args:
                for key in ["filePath", "path", "query", "command", "content"]:
                    if key in entry.args:
                        val = str(entry.args[key])
                        if len(val) > 200:
                            val = val[:200] + "..."
                        parts.append(f"{key}: {val}")
            
            # Add response summary
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
    
    def _extract_files_touched(self, entry: LogEntry) -> set:
        """Extract files touched by this entry."""
        files = set()
        
        if entry.kind == "toolCall" and entry.args:
            for key in ["filePath", "path"]:
                if key in entry.args:
                    files.add(entry.args[key])
        
        return files
    
    def _create_transition(self, from_state: State, to_state: State, entry: LogEntry) -> Transition:
        """Create a transition for a log entry."""
        trans_id = f"trans_{self.transition_counter}"
        self.transition_counter += 1
        
        # Determine action type
        if entry.kind == "toolCall":
            action_type = entry.tool or "unknown_tool"
        else:
            action_type = "request"
        
        # Build action data
        action_data = {}
        if entry.kind == "toolCall" and entry.args:
            # Store key args for comparison
            for key in ["filePath", "path", "query"]:
                if key in entry.args:
                    action_data[key] = entry.args[key]
        
        transition = Transition(
            transition_id=trans_id,
            from_state=from_state.state_id,
            to_state=to_state.state_id,
            action_type=action_type,
            action_data=action_data,
            step=to_state.step,
            metadata={
                "entry_id": entry.id,
                "signature": entry.get_signature()
            }
        )
        
        return transition


def generate_pta(trajectory_file: str, output_file: Optional[str] = None, 
                 include_requests: bool = False) -> PTA:
    """
    Convenience function to generate a PTA from a trajectory file.
    
    Args:
        trajectory_file: Path to chat-export-logs.json
        output_file: Optional path to save PTA JSON
        include_requests: Whether to include LLM requests
        
    Returns:
        Generated PTA
    """
    generator = PTAGenerator(include_requests=include_requests)
    pta = generator.generate_pta(trajectory_file)
    
    if output_file:
        pta.save(output_file)
        logger.info(f"Saved PTA to: {output_file}")
    
    return pta


def print_pta_summary(pta: PTA) -> None:
    """Print a summary of the PTA."""
    print("=" * 60)
    print("SWE PTA Summary")
    print("=" * 60)
    print(f"States: {len(pta.states)}")
    print(f"Transitions: {len(pta.transitions)}")
    print(f"Terminal States: {pta.get_terminal_states()}")
    print()
    
    # Tool sequence
    tools = pta.get_tool_sequence()
    print(f"Tool Sequence ({len(tools)} tools):")
    for i, tool in enumerate(tools, 1):
        print(f"  {i}. {tool}")
    print()
    
    # Source info
    if pta.metadata:
        print(f"Source: {pta.metadata.get('source_file', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate PTA from coding agent trajectory',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('trajectory_file', help='Path to chat-export-logs.json')
    parser.add_argument('--output', '-o', help='Output file for PTA JSON')
    parser.add_argument('--include-requests', action='store_true',
                       help='Include LLM request entries (default: tool calls only)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate PTA
    pta = generate_pta(
        args.trajectory_file,
        output_file=args.output,
        include_requests=args.include_requests
    )
    
    # Print summary
    print_pta_summary(pta)
    
    # Auto-save if no output specified
    if not args.output:
        default_output = Path(args.trajectory_file).stem + "_pta.json"
        pta.save(default_output)
        print(f"\nSaved PTA to: {default_output}")


if __name__ == "__main__":
    main()
