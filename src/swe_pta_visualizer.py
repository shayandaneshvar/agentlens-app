#!/usr/bin/env python3
"""
SWE PTA Visualizer

Creates visualizations of SWE (Software Engineering) coding agent PTAs.
Shows tool calls, file operations, LLM requests, and the flow of agent actions.
"""

import json
import logging
import html
from typing import Dict, List, Any, Set, Optional
from pathlib import Path
import argparse
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization output."""
    max_observation_length: int = 200
    max_code_preview_length: int = 500
    show_full_args: bool = False
    color_scheme: str = "dark"


class SWEPTAVisualizer:
    """Creates visualizations of SWE PTA structure."""
    
    # Color palette for different tool types
    TOOL_COLORS = {
        'create_file': '#4ade80',      # green
        'replace_string_in_file': '#facc15',  # yellow
        'read_file': '#60a5fa',        # blue
        'file_search': '#c084fc',      # purple
        'grep_search': '#f472b6',      # pink
        'semantic_search': '#fb923c',  # orange
        'run_in_terminal': '#f87171',  # red
        'list_dir': '#2dd4bf',         # teal
        'request': '#94a3b8',          # gray (LLM request)
        'default': '#a1a1aa',          # zinc
    }
    
    def __init__(self, pta_file: str, config: Optional[VisualizationConfig] = None):
        """Load PTA from JSON file."""
        logger.info(f"Loading SWE PTA from: {pta_file}")
        
        self.config = config or VisualizationConfig()
        self.pta_path = Path(pta_file)
        
        with open(pta_file, 'r', encoding='utf-8') as f:
            self.pta_data = json.load(f)
        
        # Core PTA components
        self.states: Dict[str, Any] = self.pta_data.get('states', {})
        self.transitions: List[Dict[str, Any]] = self.pta_data.get('transitions', [])
        self.initial_state = self.pta_data.get('initial_state')
        self.metadata = self.pta_data.get('metadata', {})
        self.branches = self.pta_data.get('branches', {})
        self.statistics = self.pta_data.get('statistics', {})
        
        # Build adjacency structures
        self.adjacency: Dict[str, List[str]] = {}
        self.transition_map: Dict[tuple, Dict] = {}
        self.in_degree: Dict[str, int] = {sid: 0 for sid in self.states}
        self.out_degree: Dict[str, int] = {sid: 0 for sid in self.states}
        
        for transition in self.transitions:
            from_state = transition.get('from_state')
            to_state = transition.get('to_state')
            
            if from_state and to_state:
                if from_state not in self.adjacency:
                    self.adjacency[from_state] = []
                self.adjacency[from_state].append(to_state)
                self.transition_map[(from_state, to_state)] = transition
                self.out_degree[from_state] = self.out_degree.get(from_state, 0) + 1
                self.in_degree[to_state] = self.in_degree.get(to_state, 0) + 1
        
        logger.info(f"Loaded PTA with {len(self.states)} states and {len(self.transitions)} transitions")
    
    def _get_tool_color(self, tool: Optional[str]) -> str:
        """Get color for a tool type."""
        if not tool:
            return self.TOOL_COLORS['default']
        return self.TOOL_COLORS.get(tool, self.TOOL_COLORS['default'])
    
    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + '...'
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return html.escape(str(text))
    
    def _format_observation(self, state: Dict[str, Any]) -> str:
        """Format observation for display."""
        obs = state.get('observation', '')
        return self._truncate(obs, self.config.max_observation_length)
    
    def _format_tool_args(self, log_entry: Dict[str, Any]) -> str:
        """Format tool arguments for display."""
        args = log_entry.get('args', {})
        tool = log_entry.get('tool', '')
        
        if tool == 'create_file':
            path = args.get('filePath', 'unknown')
            content = args.get('content', '')
            preview = self._truncate(content, 100).replace('\n', '↵')
            return f"path: {path}\ncontent: {preview}"
        
        elif tool == 'replace_string_in_file':
            path = args.get('filePath', 'unknown')
            old = self._truncate(args.get('oldString', ''), 50).replace('\n', '↵')
            new = self._truncate(args.get('newString', ''), 50).replace('\n', '↵')
            return f"path: {path}\nold: {old}\nnew: {new}"
        
        elif tool == 'read_file':
            path = args.get('filePath', 'unknown')
            return f"path: {path}"
        
        elif tool in ('file_search', 'grep_search', 'semantic_search'):
            query = args.get('query', '')
            return f"query: {query}"
        
        elif tool == 'run_in_terminal':
            cmd = args.get('command', '')
            return f"cmd: {self._truncate(cmd, 100)}"
        
        elif tool == 'list_dir':
            path = args.get('path', '')
            return f"path: {path}"
        
        else:
            # Generic formatting
            parts = []
            for k, v in list(args.items())[:3]:
                v_str = str(v)
                if len(v_str) > 50:
                    v_str = v_str[:47] + '...'
                parts.append(f"{k}: {v_str}")
            return '\n'.join(parts)
    
    def create_statistics_summary(self) -> str:
        """Create a summary of PTA statistics."""
        logger.info("Creating statistics summary...")
        
        lines = []
        lines.append("SWE PTA STATISTICS SUMMARY")
        lines.append("=" * 60)
        
        # Basic counts
        lines.append(f"\n📊 Basic Counts:")
        lines.append(f"  Total States: {len(self.states)}")
        lines.append(f"  Total Transitions: {len(self.transitions)}")
        lines.append(f"  Initial State: {self.initial_state}")
        
        # Terminal states
        terminal = [sid for sid, deg in self.out_degree.items() if deg == 0]
        lines.append(f"  Terminal States: {len(terminal)}")
        
        # Tool type breakdown
        tool_counts: Dict[str, int] = {}
        for transition in self.transitions:
            action_type = transition.get('action_type', 'unknown')
            tool_counts[action_type] = tool_counts.get(action_type, 0) + 1
        
        lines.append(f"\n🔧 Tool/Action Types:")
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.transitions)) * 100 if self.transitions else 0
            lines.append(f"  {tool}: {count} ({percentage:.1f}%)")
        
        # Branching info
        branching_states = [sid for sid, deg in self.out_degree.items() if deg > 1]
        merging_states = [sid for sid, deg in self.in_degree.items() if deg > 1]
        
        lines.append(f"\n🌳 Branching Analysis:")
        lines.append(f"  Branching States (out > 1): {len(branching_states)}")
        lines.append(f"  Merge States (in > 1): {len(merging_states)}")
        
        if branching_states:
            lines.append(f"  Branch Points: {', '.join(branching_states[:5])}")
            if len(branching_states) > 5:
                lines.append(f"    ... and {len(branching_states) - 5} more")
        
        # Files touched
        all_files: Set[str] = set()
        for state in self.states.values():
            files = state.get('files_touched', [])
            all_files.update(files)
        
        lines.append(f"\n📁 Files Touched: {len(all_files)}")
        for f in sorted(all_files)[:10]:
            lines.append(f"  {f}")
        if len(all_files) > 10:
            lines.append(f"  ... and {len(all_files) - 10} more")
        
        # Metadata from merge (if available)
        if 'merge_stats' in self.metadata:
            ms = self.metadata['merge_stats']
            lines.append(f"\n🔀 Merge Statistics:")
            lines.append(f"  Traces Merged: {ms.get('traces_merged', 'N/A')}")
            lines.append(f"  States Added: {ms.get('states_added', 'N/A')}")
            lines.append(f"  States Merged: {ms.get('states_merged', 'N/A')}")
            lines.append(f"  Branches Created: {ms.get('branches_created', 'N/A')}")
        
        return '\n'.join(lines)
    
    def create_text_flow_diagram(self) -> str:
        """Create a text-based flow diagram."""
        logger.info("Creating text flow diagram...")
        
        lines = []
        lines.append("SWE PTA FLOW DIAGRAM")
        lines.append("=" * 60)
        
        # BFS traversal from initial state
        if not self.initial_state:
            lines.append("No initial state found!")
            return '\n'.join(lines)
        
        visited = set()
        queue = [(self.initial_state, 0)]
        
        while queue and len(visited) < 30:
            state_id, depth = queue.pop(0)
            if state_id in visited:
                continue
            visited.add(state_id)
            
            state = self.states.get(state_id, {})
            tool = state.get('tool_used', 'request')
            step = state.get('step', '?')
            obs = self._format_observation(state)
            
            # Indent based on depth
            indent = "  " * depth
            
            # Show state
            marker = "🟢" if state_id == self.initial_state else "⚪"
            if self.out_degree.get(state_id, 0) == 0:
                marker = "🔴"  # Terminal
            
            lines.append(f"{indent}{marker} [{state_id}] Step {step}")
            lines.append(f"{indent}   Tool: {tool or 'initial'}")
            if obs and obs != '<initial>':
                obs_preview = obs[:80].replace('\n', ' ')
                lines.append(f"{indent}   Obs: {obs_preview}")
            
            # Show outgoing transitions
            for next_state in self.adjacency.get(state_id, []):
                trans = self.transition_map.get((state_id, next_state), {})
                action = trans.get('action_type', '?')
                lines.append(f"{indent}   └─({action})─→ {next_state}")
                
                if next_state not in visited:
                    queue.append((next_state, depth + 1))
        
        if len(visited) >= 30:
            lines.append("\n... (truncated at 30 states)")
        
        return '\n'.join(lines)
    
    def generate_html(self, output_file: str) -> None:
        """Generate an interactive HTML visualization."""
        logger.info(f"Generating HTML visualization: {output_file}")
        
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Identify special states
        branching = {sid for sid, deg in self.out_degree.items() if deg > 1}
        merging = {sid for sid, deg in self.in_degree.items() if deg > 1}
        terminal = {sid for sid, deg in self.out_degree.items() if deg == 0}
        
        # Compute layers using BFS
        layers = self._compute_layers()
        
        # Position nodes
        node_w, node_h = 320, 180
        x_gap, y_gap = 400, 220
        pad = 60
        
        positions: Dict[str, Dict[str, float]] = {}
        for layer_idx, layer in enumerate(layers):
            for row_idx, state_id in enumerate(layer):
                positions[state_id] = {
                    'x': pad + layer_idx * x_gap,
                    'y': pad + row_idx * (node_h + 40)
                }
        
        canvas_width = pad + len(layers) * x_gap + node_w + pad
        max_rows = max((len(layer) for layer in layers), default=1)
        canvas_height = pad + max_rows * (node_h + 40) + pad
        
        # Build HTML
        parts: List[str] = []
        parts.append("<!DOCTYPE html>")
        parts.append("<html lang='en'>")
        parts.append("<head>")
        parts.append("<meta charset='UTF-8'>")
        parts.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        parts.append(f"<title>SWE PTA: {self.pta_path.stem}</title>")
        parts.append(self._get_css())
        parts.append("</head>")
        parts.append("<body>")
        
        # Toolbar
        parts.append(self._get_toolbar_html())
        
        # Stats sidebar
        parts.append(self._get_stats_sidebar_html(branching, merging, terminal))
        
        # Main canvas
        parts.append("<div class='scroll-shell'>")
        parts.append(f"<div class='wrapper' style='width:{canvas_width}px;height:{canvas_height}px;'>")
        parts.append("<div class='canvas-container' id='canvas'>")
        
        # SVG for edges
        parts.append(f"<svg id='graph-svg' width='{canvas_width}' height='{canvas_height}'>")
        parts.append("<defs>")
        parts.append("<marker id='arrow' viewBox='0 0 10 10' refX='7' refY='5' markerWidth='6' markerHeight='6' orient='auto'>")
        parts.append("<path d='M0 0 L10 5 L0 10 z' fill='#5fb0ff'/>")
        parts.append("</marker>")
        parts.append("</defs>")
        
        # Draw edges
        for idx, trans in enumerate(self.transitions):
            f = trans.get('from_state')
            t = trans.get('to_state')
            if not f or not t or f not in positions or t not in positions:
                continue
            
            action = trans.get('action_type', '')
            color = self._get_tool_color(action)
            
            fx = positions[f]['x'] + node_w
            fy = positions[f]['y'] + node_h / 2
            tx = positions[t]['x']
            ty = positions[t]['y'] + node_h / 2
            
            # Bezier curve
            c1x = fx + (tx - fx) * 0.4
            c2x = tx - (tx - fx) * 0.4
            d = f"M {fx} {fy} C {c1x} {fy} {c2x} {ty} {tx} {ty}"
            
            parts.append(f"<path id='edge-{idx}' class='edge' data-from='{f}' data-to='{t}' d='{d}' stroke='{color}' stroke-width='2' fill='none' marker-end='url(#arrow)' opacity='0.8'/>")
            
            # Edge label
            mid_x = (fx + tx) / 2
            mid_y = (fy + ty) / 2 - 12
            parts.append(f"<text class='edge-label' x='{mid_x}' y='{mid_y}' text-anchor='middle' fill='{color}'>{self._escape_html(action)}</text>")
        
        parts.append("</svg>")
        
        # Draw state nodes
        for state_id, pos in positions.items():
            state = self.states.get(state_id, {})
            parts.append(self._render_state_card(state_id, state, pos, branching, merging, terminal))
        
        parts.append("</div></div></div>")  # close canvas, wrapper, scroll-shell
        
        # JavaScript for interactivity
        parts.append(self._get_javascript())
        
        parts.append("</body></html>")
        
        # Write file
        out_path.write_text('\n'.join(parts), encoding='utf-8')
        logger.info(f"HTML visualization saved to {output_file}")
    
    def _compute_layers(self) -> List[List[str]]:
        """Compute layers using BFS from initial state."""
        if not self.initial_state or self.initial_state not in self.states:
            return [list(self.states.keys())]
        
        layers: List[List[str]] = []
        visited: Set[str] = set()
        frontier = [self.initial_state]
        
        while frontier:
            next_frontier: List[str] = []
            layer: List[str] = []
            
            for s in frontier:
                if s in visited:
                    continue
                visited.add(s)
                layer.append(s)
                
                for t in self.adjacency.get(s, []):
                    if t not in visited:
                        next_frontier.append(t)
            
            if layer:
                layers.append(layer)
            frontier = next_frontier
        
        # Add any unvisited states
        remaining = [s for s in self.states.keys() if s not in visited]
        if remaining:
            layers.append(remaining)
        
        return layers
    
    def _render_state_card(self, state_id: str, state: Dict[str, Any], pos: Dict[str, float],
                           branching: Set[str], merging: Set[str], terminal: Set[str]) -> str:
        """Render a state card HTML."""
        tool = state.get('tool_used') or 'request'
        step = state.get('step', '?')
        obs = state.get('observation', '')
        log_entry = state.get('log_entry', {})
        metadata = state.get('metadata', {})
        
        # CSS classes
        classes = ['state']
        if state_id in branching:
            classes.append('branch')
        if state_id in merging:
            classes.append('merge')
        if state_id in terminal:
            classes.append('terminal')
        if state_id == self.initial_state:
            classes.append('initial')
        
        color = self._get_tool_color(tool)
        
        # Build card content
        content_parts = []
        
        # Header
        content_parts.append(f"<div class='state-header' style='border-color:{color};'>")
        content_parts.append(f"<span class='state-id'>{self._escape_html(state_id)}</span>")
        content_parts.append(f"<span class='step-badge'>Step {step}</span>")
        content_parts.append("</div>")
        
        # Tool badge
        tool_display = tool if tool else 'initial'
        content_parts.append(f"<div class='tool-badge' style='background:{color};'>{self._escape_html(tool_display)}</div>")
        
        # Observation preview
        obs_preview = self._truncate(obs, 120).replace('\n', ' ')
        content_parts.append(f"<div class='observation'>{self._escape_html(obs_preview)}</div>")
        
        # Tool args (if available)
        if log_entry and log_entry.get('args'):
            args_preview = self._format_tool_args(log_entry)
            args_html = self._escape_html(args_preview).replace('\n', '<br>')
            content_parts.append(f"<div class='args'>{args_html}</div>")
        
        # Metadata badges
        badges = []
        if state_id == self.initial_state:
            badges.append("<span class='badge initial'>INITIAL</span>")
        if state_id in terminal:
            badges.append("<span class='badge terminal'>TERMINAL</span>")
        if state_id in branching:
            badges.append(f"<span class='badge branch'>OUT:{self.out_degree.get(state_id, 0)}</span>")
        if state_id in merging:
            badges.append(f"<span class='badge merge'>IN:{self.in_degree.get(state_id, 0)}</span>")
        if metadata.get('trace_count'):
            badges.append(f"<span class='badge traces'>Traces:{metadata['trace_count']}</span>")
        
        if badges:
            content_parts.append(f"<div class='badges'>{''.join(badges)}</div>")
        
        # Full card
        return (f"<div class='{' '.join(classes)}' data-sid='{state_id}' "
                f"style='left:{pos['x']}px;top:{pos['y']}px;'>"
                f"{''.join(content_parts)}</div>")
    
    def _get_css(self) -> str:
        """Get CSS styles."""
        return """<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    overflow: hidden;
}
.toolbar {
    position: fixed;
    top: 10px;
    left: 10px;
    z-index: 100;
    background: #161b22;
    border: 1px solid #30363d;
    padding: 8px 12px;
    border-radius: 8px;
    display: flex;
    gap: 10px;
    align-items: center;
    font-size: 0.8rem;
}
.toolbar button {
    background: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
    padding: 4px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.75rem;
}
.toolbar button:hover { background: #30363d; }
.stats-sidebar {
    position: fixed;
    top: 60px;
    right: 10px;
    z-index: 100;
    background: #161b22;
    border: 1px solid #30363d;
    padding: 12px;
    border-radius: 8px;
    width: 220px;
    font-size: 0.7rem;
    max-height: calc(100vh - 80px);
    overflow-y: auto;
}
.stats-sidebar h3 { margin-bottom: 8px; color: #58a6ff; font-size: 0.85rem; }
.stats-sidebar .stat-row { display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #21262d; }
.stats-sidebar .stat-label { color: #8b949e; }
.stats-sidebar .stat-value { color: #c9d1d9; font-weight: 600; }
.scroll-shell {
    width: 100vw;
    height: 100vh;
    overflow: auto;
    padding-left: 10px;
    padding-top: 60px;
}
.wrapper {
    position: relative;
    min-width: 100%;
    min-height: 100%;
}
.canvas-container {
    position: relative;
    transform-origin: 0 0;
}
svg {
    position: absolute;
    left: 0;
    top: 0;
    pointer-events: none;
}
.edge { transition: stroke-width 0.2s; }
.edge-label { font-size: 10px; font-family: monospace; }
.state {
    position: absolute;
    width: 320px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px;
    cursor: grab;
    transition: box-shadow 0.2s;
}
.state:hover { box-shadow: 0 0 20px rgba(88, 166, 255, 0.3); }
.state.initial { border-color: #3fb950; }
.state.terminal { border-color: #f85149; }
.state.branch { border-color: #58a6ff; }
.state.merge { border-color: #d29922; }
.state-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 6px;
    margin-bottom: 8px;
    border-bottom: 2px solid #30363d;
}
.state-id { font-weight: 600; font-size: 0.75rem; color: #58a6ff; }
.step-badge {
    background: #21262d;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.65rem;
    color: #8b949e;
}
.tool-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #0d1117;
    margin-bottom: 8px;
}
.observation {
    font-size: 0.68rem;
    color: #8b949e;
    background: #0d1117;
    padding: 6px;
    border-radius: 4px;
    margin-bottom: 6px;
    max-height: 50px;
    overflow: hidden;
    line-height: 1.3;
}
.args {
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.6rem;
    color: #7ee787;
    background: #0d1117;
    padding: 4px 6px;
    border-radius: 4px;
    border-left: 2px solid #3fb950;
    margin-bottom: 6px;
    max-height: 40px;
    overflow: hidden;
}
.badges { display: flex; flex-wrap: wrap; gap: 4px; }
.badge {
    font-size: 0.55rem;
    padding: 2px 5px;
    border-radius: 3px;
    font-weight: 600;
}
.badge.initial { background: #238636; color: #fff; }
.badge.terminal { background: #da3633; color: #fff; }
.badge.branch { background: #1f6feb; color: #fff; }
.badge.merge { background: #9e6a03; color: #fff; }
.badge.traces { background: #30363d; color: #8b949e; }
/* Detail panel */
.detail-panel {
    display: none;
    position: fixed;
    bottom: 10px;
    left: 10px;
    right: 240px;
    z-index: 100;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
    max-height: 40vh;
    overflow-y: auto;
    font-size: 0.75rem;
}
.detail-panel.visible { display: block; }
.detail-panel h4 { color: #58a6ff; margin-bottom: 8px; }
.detail-panel pre {
    background: #0d1117;
    padding: 8px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 0.65rem;
    color: #c9d1d9;
}
.detail-panel .close-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    background: none;
    border: none;
    color: #8b949e;
    cursor: pointer;
    font-size: 1rem;
}
</style>"""
    
    def _get_toolbar_html(self) -> str:
        """Get toolbar HTML."""
        return """<div class='toolbar'>
<span style='font-weight:600;color:#58a6ff;'>SWE PTA Visualizer</span>
<button id='zoomIn'>+ Zoom</button>
<button id='zoomOut'>- Zoom</button>
<span id='zoomLvl'>100%</span>
<button id='resetLayout'>Reset</button>
<button id='toggleStats'>Stats</button>
</div>"""
    
    def _get_stats_sidebar_html(self, branching: Set[str], merging: Set[str], terminal: Set[str]) -> str:
        """Get stats sidebar HTML."""
        # Tool counts
        tool_counts: Dict[str, int] = {}
        for trans in self.transitions:
            t = trans.get('action_type', 'unknown')
            tool_counts[t] = tool_counts.get(t, 0) + 1
        
        tool_rows = ''.join([
            f"<div class='stat-row'><span class='stat-label'>{t}</span><span class='stat-value'>{c}</span></div>"
            for t, c in sorted(tool_counts.items(), key=lambda x: -x[1])[:8]
        ])
        
        return f"""<div class='stats-sidebar' id='statsSidebar'>
<h3>📊 Statistics</h3>
<div class='stat-row'><span class='stat-label'>States</span><span class='stat-value'>{len(self.states)}</span></div>
<div class='stat-row'><span class='stat-label'>Transitions</span><span class='stat-value'>{len(self.transitions)}</span></div>
<div class='stat-row'><span class='stat-label'>Branching</span><span class='stat-value'>{len(branching)}</span></div>
<div class='stat-row'><span class='stat-label'>Merging</span><span class='stat-value'>{len(merging)}</span></div>
<div class='stat-row'><span class='stat-label'>Terminal</span><span class='stat-value'>{len(terminal)}</span></div>
<h3 style='margin-top:12px;'>🔧 Tools</h3>
{tool_rows}
</div>"""
    
    def _get_javascript(self) -> str:
        """Get JavaScript for interactivity."""
        return """<script>
(function() {
    const canvas = document.getElementById('canvas');
    const nodes = [...document.querySelectorAll('.state[data-sid]')];
    const edges = [...document.querySelectorAll('path.edge')];
    const svg = document.getElementById('graph-svg');
    
    // Store initial positions
    const initPos = {};
    nodes.forEach(n => {
        initPos[n.dataset.sid] = {
            left: parseFloat(n.style.left),
            top: parseFloat(n.style.top)
        };
    });
    
    // Zoom
    let zoom = 1;
    function applyZoom() {
        canvas.style.transform = `scale(${zoom})`;
        document.getElementById('zoomLvl').textContent = Math.round(zoom * 100) + '%';
    }
    document.getElementById('zoomIn').onclick = () => { zoom = Math.min(3, zoom + 0.1); applyZoom(); };
    document.getElementById('zoomOut').onclick = () => { zoom = Math.max(0.2, zoom - 0.1); applyZoom(); };
    
    // Reset
    document.getElementById('resetLayout').onclick = () => {
        zoom = 1;
        applyZoom();
        nodes.forEach(n => {
            const p = initPos[n.dataset.sid];
            n.style.left = p.left + 'px';
            n.style.top = p.top + 'px';
        });
        redrawEdges();
    };
    
    // Toggle stats
    document.getElementById('toggleStats').onclick = () => {
        const sb = document.getElementById('statsSidebar');
        sb.style.display = sb.style.display === 'none' ? 'block' : 'none';
    };
    
    // Redraw edges
    function redrawEdges() {
        edges.forEach(e => {
            const fromNode = document.querySelector(`.state[data-sid='${e.dataset.from}']`);
            const toNode = document.querySelector(`.state[data-sid='${e.dataset.to}']`);
            if (!fromNode || !toNode) return;
            
            const fw = fromNode.offsetWidth, fh = fromNode.offsetHeight;
            const tw = toNode.offsetWidth, th = toNode.offsetHeight;
            const fx = parseFloat(fromNode.style.left) + fw;
            const fy = parseFloat(fromNode.style.top) + fh / 2;
            const tx = parseFloat(toNode.style.left);
            const ty = parseFloat(toNode.style.top) + th / 2;
            
            const c1x = fx + (tx - fx) * 0.4;
            const c2x = tx - (tx - fx) * 0.4;
            const d = `M ${fx} ${fy} C ${c1x} ${fy} ${c2x} ${ty} ${tx} ${ty}`;
            e.setAttribute('d', d);
        });
    }
    
    // Drag nodes
    let dragging = null, offsetX = 0, offsetY = 0;
    nodes.forEach(n => {
        n.addEventListener('pointerdown', e => {
            dragging = n;
            n.setPointerCapture(e.pointerId);
            const rect = n.getBoundingClientRect();
            offsetX = e.clientX - rect.left;
            offsetY = e.clientY - rect.top;
            n.style.cursor = 'grabbing';
            n.style.zIndex = 1000;
        });
    });
    
    window.addEventListener('pointermove', e => {
        if (!dragging) return;
        const x = (e.clientX - offsetX) / zoom;
        const y = (e.clientY - offsetY) / zoom;
        dragging.style.left = x + 'px';
        dragging.style.top = y + 'px';
        redrawEdges();
    });
    
    window.addEventListener('pointerup', e => {
        if (dragging) {
            dragging.style.cursor = 'grab';
            dragging.style.zIndex = '';
            dragging.releasePointerCapture(e.pointerId);
            dragging = null;
        }
    });
    
    // Click to show details
    nodes.forEach(n => {
        n.addEventListener('dblclick', () => {
            const sid = n.dataset.sid;
            const state = window.ptaStates ? window.ptaStates[sid] : null;
            if (state) {
                showDetails(sid, state);
            }
        });
    });
    
    function showDetails(sid, state) {
        let panel = document.getElementById('detailPanel');
        if (!panel) {
            panel = document.createElement('div');
            panel.id = 'detailPanel';
            panel.className = 'detail-panel';
            document.body.appendChild(panel);
        }
        panel.innerHTML = `
            <button class='close-btn' onclick="this.parentElement.classList.remove('visible')">×</button>
            <h4>${sid}</h4>
            <pre>${JSON.stringify(state, null, 2)}</pre>
        `;
        panel.classList.add('visible');
    }
    
    applyZoom();
})();
</script>
<script>
window.ptaStates = """ + json.dumps(self.states) + """;
</script>"""
    
    def generate_list_html(self, output_file: str) -> None:
        """Generate a simple list-based HTML visualization (like the original pta_visualizer)."""
        logger.info(f"Generating list HTML visualization: {output_file}")
        
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        parts: List[str] = []
        parts.append("<!DOCTYPE html>")
        parts.append("<html lang='en'><head><meta charset='UTF-8'>")
        parts.append(f"<title>SWE PTA: {self.pta_path.stem}</title>")
        parts.append("""<style>
body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; margin-bottom: 20px; }
.transition {
    display: flex;
    align-items: flex-start;
    gap: 15px;
    padding: 12px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-bottom: 10px;
}
.state-card {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 10px;
    width: 300px;
    flex-shrink: 0;
}
.state-card h3 { color: #58a6ff; font-size: 0.8rem; margin: 0 0 6px; }
.state-card .tool { 
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #0d1117;
    margin-bottom: 6px;
}
.state-card .obs {
    font-size: 0.68rem;
    color: #8b949e;
    background: #0d1117;
    padding: 6px;
    border-radius: 4px;
    max-height: 80px;
    overflow: hidden;
    line-height: 1.3;
}
.action {
    font-family: monospace;
    font-size: 0.75rem;
    background: #0d1117;
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #30363d;
    align-self: center;
    color: #7ee787;
}
.stats { margin-bottom: 20px; font-size: 0.8rem; color: #8b949e; }
</style>""")
        parts.append("</head><body>")
        parts.append(f"<h1>SWE PTA: {self._escape_html(self.pta_path.stem)}</h1>")
        parts.append(f"<div class='stats'>States: {len(self.states)} | Transitions: {len(self.transitions)}</div>")
        
        for trans in self.transitions:
            f = trans.get('from_state')
            t = trans.get('to_state')
            if not f or not t:
                continue
            
            from_state = self.states.get(f, {})
            to_state = self.states.get(t, {})
            action = trans.get('action_type', 'unknown')
            
            def card(sid: str, st: Dict) -> str:
                tool = st.get('tool_used') or 'initial'
                obs = self._truncate(st.get('observation', ''), 150)
                color = self._get_tool_color(tool)
                return f"""<div class='state-card'>
                    <h3>{self._escape_html(sid)}</h3>
                    <div class='tool' style='background:{color};'>{self._escape_html(tool)}</div>
                    <div class='obs'>{self._escape_html(obs)}</div>
                </div>"""
            
            parts.append(f"<div class='transition'>{card(f, from_state)}<div class='action'>→ {self._escape_html(action)} →</div>{card(t, to_state)}</div>")
        
        parts.append("</body></html>")
        
        out_path.write_text('\n'.join(parts), encoding='utf-8')
        logger.info(f"List HTML saved to {output_file}")
    
    def save_text_visualization(self, output_file: str) -> None:
        """Save text visualization to file."""
        logger.info(f"Saving text visualization to: {output_file}")
        
        parts = []
        parts.append(self.create_statistics_summary())
        parts.append("\n" + "=" * 80 + "\n")
        parts.append(self.create_text_flow_diagram())
        
        Path(output_file).write_text('\n'.join(parts), encoding='utf-8')
        logger.info(f"Text visualization saved to {output_file}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Visualize SWE PTA JSON files")
    parser.add_argument("pta_json", help="Path to SWE PTA JSON file")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (defaults to PTA file directory)")
    parser.add_argument("--no-text", action="store_true", help="Skip text visualization")
    parser.add_argument("--html", action="store_true", help="Generate interactive graph HTML")
    parser.add_argument("--html-list", action="store_true", help="Generate list-style HTML")
    parser.add_argument("--all", "-a", action="store_true", help="Generate all visualizations")
    
    args = parser.parse_args()
    
    pta_path = Path(args.pta_json)
    if not pta_path.exists():
        parser.error(f"File not found: {pta_path}")
    
    out_dir = Path(args.output_dir) if args.output_dir else pta_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = SWEPTAVisualizer(str(pta_path))
    
    # Determine what to generate
    if args.all:
        args.html = True
        args.html_list = True
        args.no_text = False
    
    # Default: at least show text
    if not args.html and not args.html_list and not args.no_text:
        pass  # Will generate text
    
    # Text visualization
    if not args.no_text:
        print(visualizer.create_statistics_summary())
        txt_file = out_dir / f"{pta_path.stem}_visualization.txt"
        visualizer.save_text_visualization(str(txt_file))
        print(f"\nText visualization saved to: {txt_file}")
    
    # Graph HTML
    if args.html:
        html_file = out_dir / f"{pta_path.stem}_graph.html"
        visualizer.generate_html(str(html_file))
        print(f"Graph HTML saved to: {html_file}")
    
    # List HTML
    if args.html_list:
        list_file = out_dir / f"{pta_path.stem}_list.html"
        visualizer.generate_list_html(str(list_file))
        print(f"List HTML saved to: {list_file}")


if __name__ == "__main__":
    main()
