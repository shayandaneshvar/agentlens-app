"""
Data models for SWE/Coding Agent trajectory analysis.
Designed for LLM-based state equivalence checking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import hashlib
import json


@dataclass
class LogEntry:
    """
    Represents a single log entry from the trajectory.
    Can be either a 'request' (LLM call) or 'toolCall'.
    """
    id: str
    kind: str  # "request" or "toolCall"
    raw_data: Dict[str, Any]
    index: int  # Position in the original logs array
    
    # For toolCall entries
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    response: Optional[Any] = None
    
    # For request entries
    model: Optional[str] = None
    response_message: Optional[str] = None
    
    @classmethod
    def from_log(cls, log_data: Dict[str, Any], index: int) -> 'LogEntry':
        """Create LogEntry from raw log data."""
        entry = cls(
            id=log_data.get("id", f"log_{index}"),
            kind=log_data.get("kind", "unknown"),
            raw_data=log_data,
            index=index
        )
        
        if entry.kind == "toolCall":
            entry.tool = log_data.get("tool", "")
            # Parse args - can be string or dict
            args_raw = log_data.get("args", {})
            if isinstance(args_raw, str):
                try:
                    entry.args = json.loads(args_raw)
                except json.JSONDecodeError:
                    entry.args = {"raw": args_raw}
            else:
                entry.args = args_raw
            entry.response = log_data.get("response", [])
            
        elif entry.kind == "request":
            metadata = log_data.get("metadata", {})
            entry.model = metadata.get("model", "unknown")
            response = log_data.get("response", {})
            if isinstance(response, dict) and "message" in response:
                msg = response["message"]
                if isinstance(msg, list):
                    entry.response_message = " ".join(str(m) for m in msg)
                else:
                    entry.response_message = str(msg)
        
        return entry
    
    def get_signature(self) -> str:
        """Generate a signature for this entry."""
        if self.kind == "toolCall":
            # For tool calls, signature is tool + key args
            key_args = {}
            if self.args:
                for key in ["filePath", "path", "query", "command"]:
                    if key in self.args:
                        val = self.args[key]
                        if isinstance(val, str) and len(val) > 100:
                            val = val[:100] + "..."
                        key_args[key] = val
            return f"toolCall:{self.tool}({json.dumps(key_args, sort_keys=True)})"
        else:
            return f"request:{self.model}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "id": self.id,
            "kind": self.kind,
            "index": self.index,
            "signature": self.get_signature()
        }
        if self.kind == "toolCall":
            d["tool"] = self.tool
            d["args"] = self.args
            # Truncate response for storage
            if self.response:
                resp_str = str(self.response)
                d["response_preview"] = resp_str[:500] if len(resp_str) > 500 else resp_str
        elif self.kind == "request":
            d["model"] = self.model
            if self.response_message:
                d["response_preview"] = self.response_message[:500]
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Create LogEntry from dictionary (deserialization)."""
        entry = cls(
            id=data.get("id", ""),
            kind=data.get("kind", "unknown"),
            raw_data=data.get("raw_data", {}),
            index=data.get("index", 0)
        )
        
        if entry.kind == "toolCall":
            entry.tool = data.get("tool")
            entry.args = data.get("args")
            # response is not fully serialized, only preview
            entry.response = data.get("response_preview")
        elif entry.kind == "request":
            entry.model = data.get("model")
            entry.response_message = data.get("response_preview")
        
        return entry


@dataclass
class State:
    """
    Represents a state in the PTA.
    A state captures the observation/context at a point in the trajectory.
    
    For LLM-based equivalence, we store rich context that can be compared semantically.
    
    Key distinction:
    - observation: Describes the ACTION taken (tool + args + response)
    - resulting_state: Describes the RESULT/EFFECT (what changed in the world)
    
    For equivalence checking, we prefer resulting_state as it captures
    the semantic outcome rather than implementation details.
    
    Enhanced with location and scope information for better discrimination:
    - line_range: (start, end) lines for file operations
    - function_name, class_name: Enclosing scope for edits
    - edit_type: add/remove/replace/expand/shrink
    """
    state_id: str
    step: int  # Sequential step number
    
    # The log entry that led to this state
    log_entry: Optional[LogEntry] = None
    
    # Context for LLM-based equivalence
    observation: str = ""  # Text representation of what was observed (action-based)
    files_touched: Set[str] = field(default_factory=set)
    tool_used: Optional[str] = None
    
    # Result-based state representation (what changed in the world)
    # e.g., "file_created:spec.md", "file_search:not_found", etc.
    resulting_state: str = ""
    
    # Content hash for fine-grained matching (distinguishes different edits to same file)
    # This is separate from resulting_state to allow coarse merging but fine matching
    content_hash: str = ""
    
    # Content description for semantic comparison (describes what the change does)
    # Used with LLM to compare semantically equivalent changes even if hashes differ
    # e.g., "Renamed function add_numbers to perform_add in math_ops.py"
    content_description: str = ""
    
    # === NEW: Location and scope information for better discrimination ===
    
    # File path (normalized, lowercase, forward slashes)
    file_path: str = ""
    
    # Line range for file operations: (start_line, end_line), 1-indexed
    # For read_file: lines that were read
    # For edits: approximate lines affected
    line_range: Optional[tuple] = None  # Tuple[int, int]
    
    # Relative position in file: "early" (top 33%), "middle", "late" (bottom 33%)
    # Useful for cross-file-size comparison
    relative_position: str = ""
    
    # Operation type: "read", "create", "modify", "delete", "search", "terminal", "other"
    operation_type: str = ""
    
    # Edit type for modifications: "add", "remove", "replace", "expand", "shrink"
    edit_type: str = ""
    
    # Scope information extracted from code context
    function_name: str = ""  # Enclosing function name
    class_name: str = ""     # Enclosing class name
    scope_path: str = ""     # Combined: "ClassName.method_name" or just "function_name"
    
    # Metadata for analysis
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_observation_hash(self) -> str:
        """Hash of observation for quick comparison."""
        return hashlib.md5(self.observation.encode()).hexdigest()[:16]
    
    def get_resulting_state_hash(self) -> str:
        """Hash of resulting state for quick comparison."""
        if not self.resulting_state:
            return ""
        return hashlib.md5(self.resulting_state.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state_id": self.state_id,
            "step": self.step,
            "observation": self.observation[:1000] if len(self.observation) > 1000 else self.observation,
            "observation_hash": self.get_observation_hash(),
            "resulting_state": self.resulting_state,
            "resulting_state_hash": self.get_resulting_state_hash(),
            "content_hash": self.content_hash,
            "content_description": self.content_description,
            "files_touched": list(self.files_touched),
            "tool_used": self.tool_used,
            # New location and scope fields
            "file_path": self.file_path,
            "line_range": list(self.line_range) if self.line_range else None,
            "relative_position": self.relative_position,
            "operation_type": self.operation_type,
            "edit_type": self.edit_type,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "scope_path": self.scope_path,
            # End new fields
            "log_entry": self.log_entry.to_dict() if self.log_entry else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'State':
        """Create State from dictionary."""
        # Handle line_range - convert list back to tuple
        line_range_data = data.get("line_range")
        line_range = tuple(line_range_data) if line_range_data else None
        
        state = cls(
            state_id=data["state_id"],
            step=data["step"],
            observation=data.get("observation", ""),
            resulting_state=data.get("resulting_state", ""),
            content_hash=data.get("content_hash", ""),
            content_description=data.get("content_description", ""),
            files_touched=set(data.get("files_touched", [])),
            tool_used=data.get("tool_used"),
            # New location and scope fields
            file_path=data.get("file_path", ""),
            line_range=line_range,
            relative_position=data.get("relative_position", ""),
            operation_type=data.get("operation_type", ""),
            edit_type=data.get("edit_type", ""),
            function_name=data.get("function_name", ""),
            class_name=data.get("class_name", ""),
            scope_path=data.get("scope_path", ""),
            # End new fields
            metadata=data.get("metadata", {})
        )
        
        # Reconstruct log_entry if present
        log_entry_data = data.get("log_entry")
        if log_entry_data:
            state.log_entry = LogEntry.from_dict(log_entry_data)
        
        return state


@dataclass
class Transition:
    """
    Represents a transition between states.
    In the coding agent context, this is typically a tool call or LLM action.
    """
    transition_id: str
    from_state: str
    to_state: str
    action_type: str  # e.g., "create_file", "read_file", "request"
    action_data: Dict[str, Any] = field(default_factory=dict)
    step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transition_id": self.transition_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "action_type": self.action_type,
            "action_data": self.action_data,
            "step": self.step,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transition':
        """Create Transition from dictionary."""
        return cls(
            transition_id=data["transition_id"],
            from_state=data["from_state"],
            to_state=data["to_state"],
            action_type=data["action_type"],
            action_data=data.get("action_data", {}),
            step=data.get("step", 0),
            metadata=data.get("metadata", {})
        )


@dataclass
class PTA:
    """
    Prefix Tree Acceptor for SWE coding agent trajectories.
    
    A PTA is a directed graph where:
    - States represent observations/context at points in execution
    - Transitions represent actions (tool calls, LLM requests)
    - Multiple traces can be merged to identify common patterns
    """
    initial_state: Optional[str] = None
    states: Dict[str, State] = field(default_factory=dict)
    transitions: List[Transition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    branches: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_state(self, state: State) -> None:
        """Add a state to the PTA."""
        self.states[state.state_id] = state
        if self.initial_state is None:
            self.initial_state = state.state_id
    
    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the PTA."""
        self.transitions.append(transition)
    
    def get_outgoing_transitions(self, state_id: str) -> List[Transition]:
        """Get all transitions from a state."""
        return [t for t in self.transitions if t.from_state == state_id]
    
    def get_incoming_transitions(self, state_id: str) -> List[Transition]:
        """Get all transitions to a state."""
        return [t for t in self.transitions if t.to_state == state_id]
    
    def get_terminal_states(self) -> List[str]:
        """Get states with no outgoing transitions."""
        states_with_outgoing = {t.from_state for t in self.transitions}
        return [sid for sid in self.states if sid not in states_with_outgoing]
    
    def get_tool_sequence(self) -> List[str]:
        """Get the sequence of tools used (for linear PTAs)."""
        tools = []
        for t in sorted(self.transitions, key=lambda x: x.step):
            if t.action_type not in ["request", "unknown"]:
                tools.append(t.action_type)
        return tools
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PTA to dictionary for serialization."""
        return {
            "initial_state": self.initial_state,
            "states": {sid: s.to_dict() for sid, s in self.states.items()},
            "transitions": [t.to_dict() for t in self.transitions],
            "metadata": self.metadata,
            "branches": self.branches,
            "statistics": {
                "num_states": len(self.states),
                "num_transitions": len(self.transitions),
                "terminal_states": self.get_terminal_states(),
                "tool_sequence": self.get_tool_sequence()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PTA':
        """Create PTA from dictionary."""
        pta = cls(
            initial_state=data.get("initial_state"),
            metadata=data.get("metadata", {}),
            branches=data.get("branches", {})
        )
        
        for sid, sdata in data.get("states", {}).items():
            pta.states[sid] = State.from_dict(sdata)
        
        for tdata in data.get("transitions", []):
            pta.transitions.append(Transition.from_dict(tdata))
        
        return pta
    
    def save(self, path: str) -> None:
        """Save PTA to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'PTA':
        """Load PTA from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
