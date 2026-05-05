"""Export traces and match results to file.

Public surface
--------------
- :func:`trace` — render a :class:`~swe_trace_sdk.models.Trace` to file.
- :func:`match` — write a :class:`~swe_trace_sdk.match.MatchResult` to file.
- :func:`match_to_json` — serialise a match result to a JSON string.
- :func:`match_to_dict` — return a match result as a plain dictionary.

Trace formats
~~~~~~~~~~~~~
* ``html`` — interactive DAG (graph) view.
* ``html_list`` — linear transition list.
* ``txt`` — text statistics summary + flow diagram.

Match-result formats
~~~~~~~~~~~~~~~~~~~~
* ``html`` — human-readable report with insights.
* ``json`` — machine-readable, stable schema for CI/dashboards.

Examples
--------
>>> from swe_trace_sdk import export
>>> export.trace(t, "graph.html", format="html")
>>> export.match(result, "report.html")
>>> export.match(result, "report.json", format="json")
"""

from __future__ import annotations

import html as html_lib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .models import Trace as TraceModel
from .match import MatchResult, StepAlignment
from .tool_registry import registry

logger = logging.getLogger(__name__)

__all__ = ["trace", "match", "match_to_json", "match_to_dict"]


# ═══════════════════════════════════════════════════════════════════════════
# Trace export
# ═══════════════════════════════════════════════════════════════════════════

def trace(
    tr: TraceModel,
    output_path: str,
    *,
    format: str = "html",
) -> None:
    """Write a visualisation of *tr* to *output_path*.

    Parameters
    ----------
    tr : Trace
        The trace to visualise.
    output_path : str
        Destination file path.
    format : str
        One of ``"html"`` (default), ``"html_list"``, or ``"txt"``.
    """
    viz = _TraceVisualizer(tr)

    if format == "html":
        viz.generate_html(output_path)
    elif format == "html_list":
        viz.generate_list_html(output_path)
    elif format == "txt":
        viz.save_text(output_path)
    else:
        raise ValueError(f"Unsupported visualisation format: {format!r}")


# ═══════════════════════════════════════════════════════════════════════════
# Match-result export
# ═══════════════════════════════════════════════════════════════════════════

def match_to_dict(result: MatchResult) -> Dict[str, Any]:
    """Return a plain dictionary representation of *result*."""
    return result.to_dict()


def match_to_json(result: MatchResult, *, indent: int = 2) -> str:
    """Return a JSON string representation of *result*."""
    return json.dumps(match_to_dict(result), indent=indent, ensure_ascii=False)


def match(
    result: MatchResult,
    output_path: str,
    *,
    format: Optional[str] = None,
) -> None:
    """Write *result* to *output_path*.

    Parameters
    ----------
    result : MatchResult
        The match result to serialise.
    output_path : str
        Destination file path.
    format : str | None
        ``"json"`` or ``"html"``.  Auto-detected from extension if *None*.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if format is None:
        ext = p.suffix.lower()
        if ext == ".json":
            format = "json"
        else:
            format = "html"

    if format == "json":
        p.write_text(match_to_json(result), encoding="utf-8")
    elif format == "html":
        p.write_text(_render_match_html(result), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported report format: {format!r}")

    logger.info("Report saved to %s (format=%s)", output_path, format)


# ═══════════════════════════════════════════════════════════════════════════
# Trace visualiser (internal)
# ═══════════════════════════════════════════════════════════════════════════

# Tool colour lookup (derived from tool_registry)
_DEFAULT_TOOL_COLOR = "#a1a1aa"


class _TraceVisualizer:
    def __init__(self, tr: TraceModel) -> None:
        self.tr = tr
        self.states = tr.states
        self.transitions = tr.transitions
        self.initial_state = tr.initial_state
        self.metadata = tr.metadata
        self.branches = tr.branches

        # Adjacency
        self.adjacency: Dict[str, List[str]] = {}
        self.transition_map: Dict[tuple, Dict[str, Any]] = {}
        self.in_degree: Dict[str, int] = {s: 0 for s in self.states}
        self.out_degree: Dict[str, int] = {s: 0 for s in self.states}
        for t in self.transitions:
            f, to = t.from_state, t.to_state
            self.adjacency.setdefault(f, []).append(to)
            self.transition_map[(f, to)] = t.to_dict()
            self.out_degree[f] = self.out_degree.get(f, 0) + 1
            self.in_degree[to] = self.in_degree.get(to, 0) + 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _color(tool: Optional[str]) -> str:
        return registry.get_color(tool or "")

    @staticmethod
    def _trunc(text: str, n: int = 200) -> str:
        return text if len(text) <= n else text[: n - 3] + "..."

    @staticmethod
    def _esc(text: str) -> str:
        return html_lib.escape(str(text))

    # ------------------------------------------------------------------
    # Text visualisation
    # ------------------------------------------------------------------

    def statistics_text(self) -> str:
        lines: List[str] = ["SWE TRACE STATISTICS SUMMARY", "=" * 60]

        lines.append(f"\nBasic Counts:")
        lines.append(f"  Total States: {len(self.states)}")
        lines.append(f"  Total Transitions: {len(self.transitions)}")
        lines.append(f"  Initial State: {self.initial_state}")

        terminal = [s for s, d in self.out_degree.items() if d == 0]
        lines.append(f"  Terminal States: {len(terminal)}")

        tool_counts: Dict[str, int] = {}
        for t in self.transitions:
            tool_counts[t.action_type] = tool_counts.get(t.action_type, 0) + 1
        lines.append(f"\nTool/Action Types:")
        for tool, cnt in sorted(tool_counts.items(), key=lambda x: -x[1]):
            pct = (cnt / len(self.transitions) * 100) if self.transitions else 0
            lines.append(f"  {tool}: {cnt} ({pct:.1f}%)")

        branching = [s for s, d in self.out_degree.items() if d > 1]
        merging = [s for s, d in self.in_degree.items() if d > 1]
        lines.append(f"\nBranching Analysis:")
        lines.append(f"  Branching States (out > 1): {len(branching)}")
        lines.append(f"  Merge States (in > 1): {len(merging)}")

        all_files: Set[str] = set()
        for s in self.states.values():
            all_files.update(s.files_touched)
        lines.append(f"\nFiles Touched: {len(all_files)}")
        for fp in sorted(all_files)[:10]:
            lines.append(f"  {fp}")
        if len(all_files) > 10:
            lines.append(f"  ... and {len(all_files) - 10} more")

        ms = self.metadata.get("merge_stats")
        if ms:
            lines.append(f"\nMerge Statistics:")
            for k, v in ms.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    def flow_text(self) -> str:
        lines: List[str] = ["SWE TRACE FLOW DIAGRAM", "=" * 60]
        if not self.initial_state:
            lines.append("No initial state.")
            return "\n".join(lines)

        visited: Set[str] = set()
        queue = [(self.initial_state, 0)]
        while queue and len(visited) < 30:
            sid, depth = queue.pop(0)
            if sid in visited:
                continue
            visited.add(sid)
            state = self.states.get(sid)
            if not state:
                continue
            indent = "  " * depth
            marker = ">" if sid == self.initial_state else ("*" if self.out_degree.get(sid, 0) == 0 else "-")
            lines.append(f"{indent}{marker} [{sid}] Step {state.step}")
            lines.append(f"{indent}   Tool: {state.tool_used or 'initial'}")
            obs = self._trunc(state.observation, 80).replace("\n", " ")
            if obs and obs != "<initial>":
                lines.append(f"{indent}   Obs: {obs}")
            for nxt in self.adjacency.get(sid, []):
                td = self.transition_map.get((sid, nxt), {})
                lines.append(f"{indent}   -> ({td.get('action_type','?')}) -> {nxt}")
                if nxt not in visited:
                    queue.append((nxt, depth + 1))

        if len(visited) >= 30:
            lines.append("\n... (truncated at 30 states)")
        return "\n".join(lines)

    def save_text(self, output_file: str) -> None:
        p = Path(output_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.statistics_text() + "\n\n" + self.flow_text(), encoding="utf-8")
        logger.info("Text visualisation saved to %s", output_file)

    # ------------------------------------------------------------------
    # HTML graph
    # ------------------------------------------------------------------

    def generate_html(self, output_file: str) -> None:
        p = Path(output_file)
        p.parent.mkdir(parents=True, exist_ok=True)

        branching = {s for s, d in self.out_degree.items() if d > 1}
        merging = {s for s, d in self.in_degree.items() if d > 1}
        terminal = {s for s, d in self.out_degree.items() if d == 0}

        layers = self._layers()
        node_w, node_h = 320, 180
        x_gap, y_gap = 400, 220
        pad = 60

        positions: Dict[str, Dict[str, float]] = {}
        for li, layer in enumerate(layers):
            for ri, sid in enumerate(layer):
                positions[sid] = {"x": pad + li * x_gap, "y": pad + ri * (node_h + 40)}

        cw = pad + len(layers) * x_gap + node_w + pad
        ch = pad + max((len(l) for l in layers), default=1) * (node_h + 40) + pad

        parts: List[str] = []
        parts.append(f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><title>Trace</title>")
        parts.append(_TRACE_CSS)
        parts.append("</head><body>")
        parts.append(_TRACE_TOOLBAR)
        parts.append(self._stats_sidebar(branching, merging, terminal))
        parts.append(f"<div class='scroll-shell'><div class='wrapper' style='width:{cw}px;height:{ch}px;'>")
        parts.append(f"<div class='canvas-container' id='canvas'>")

        # SVG edges
        parts.append(f"<svg id='graph-svg' width='{cw}' height='{ch}'>")
        parts.append("<defs><marker id='arrow' viewBox='0 0 10 10' refX='7' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0 0 L10 5 L0 10 z' fill='#5fb0ff'/></marker></defs>")
        for idx, t in enumerate(self.transitions):
            f, to = t.from_state, t.to_state
            if f not in positions or to not in positions:
                continue
            color = self._color(t.action_type)
            fx = positions[f]["x"] + node_w
            fy = positions[f]["y"] + node_h / 2
            tx = positions[to]["x"]
            ty = positions[to]["y"] + node_h / 2
            cx1 = fx + (tx - fx) * 0.4
            cx2 = tx - (tx - fx) * 0.4
            d = f"M {fx} {fy} C {cx1} {fy} {cx2} {ty} {tx} {ty}"
            parts.append(f"<path class='edge' data-from='{f}' data-to='{to}' d='{d}' stroke='{color}' stroke-width='2' fill='none' marker-end='url(#arrow)' opacity='0.8'/>")
            mx, my = (fx + tx) / 2, (fy + ty) / 2 - 12
            parts.append(f"<text class='edge-label' x='{mx}' y='{my}' text-anchor='middle' fill='{color}'>{self._esc(t.action_type)}</text>")
        parts.append("</svg>")

        # Nodes
        for sid, pos in positions.items():
            state = self.states.get(sid)
            if not state:
                continue
            parts.append(self._node_card(sid, state, pos, branching, merging, terminal))

        parts.append("</div></div></div>")
        parts.append(_TRACE_JS.replace("__STATES_JSON__", json.dumps({s: st.to_dict() for s, st in self.states.items()})))
        parts.append("</body></html>")

        p.write_text("\n".join(parts), encoding="utf-8")
        logger.info("HTML graph saved to %s", output_file)

    # ------------------------------------------------------------------
    # HTML list
    # ------------------------------------------------------------------

    def generate_list_html(self, output_file: str) -> None:
        p = Path(output_file)
        p.parent.mkdir(parents=True, exist_ok=True)

        parts: List[str] = []
        parts.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
        parts.append(f"<title>Trace list</title>")
        parts.append(_TRACE_LIST_CSS)
        parts.append("</head><body>")
        parts.append(f"<h1>Trace: {len(self.states)} states</h1>")
        parts.append(f"<div class='stats'>States: {len(self.states)} | Transitions: {len(self.transitions)}</div>")

        for t in self.transitions:
            fs = self.states.get(t.from_state)
            ts = self.states.get(t.to_state)
            if not fs or not ts:
                continue

            def _card(sid: str, st) -> str:
                tool = st.tool_used or "initial"
                obs = self._trunc(st.observation, 150)
                c = self._color(tool)
                return (
                    f"<div class='state-card'><h3>{self._esc(sid)}</h3>"
                    f"<div class='tool' style='background:{c};'>{self._esc(tool)}</div>"
                    f"<div class='obs'>{self._esc(obs)}</div></div>"
                )

            parts.append(
                f"<div class='transition'>{_card(t.from_state, fs)}"
                f"<div class='action'>\u2192 {self._esc(t.action_type)} \u2192</div>"
                f"{_card(t.to_state, ts)}</div>"
            )

        parts.append("</body></html>")
        p.write_text("\n".join(parts), encoding="utf-8")
        logger.info("List HTML saved to %s", output_file)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _layers(self) -> List[List[str]]:
        if not self.initial_state or self.initial_state not in self.states:
            return [list(self.states.keys())]

        layers: List[List[str]] = []
        visited: Set[str] = set()
        frontier = [self.initial_state]
        while frontier:
            nxt: List[str] = []
            layer: List[str] = []
            for s in frontier:
                if s in visited:
                    continue
                visited.add(s)
                layer.append(s)
                for t in self.adjacency.get(s, []):
                    if t not in visited:
                        nxt.append(t)
            if layer:
                layers.append(layer)
            frontier = nxt
        remaining = [s for s in self.states if s not in visited]
        if remaining:
            layers.append(remaining)
        return layers

    def _node_card(self, sid: str, state, pos, branching, merging, terminal) -> str:
        tool = state.tool_used or "request"
        step = state.step
        obs = self._trunc(state.observation, 120).replace("\n", " ")
        color = self._color(tool)

        classes = ["state"]
        if sid in branching:
            classes.append("branch")
        if sid in merging:
            classes.append("merge")
        if sid in terminal:
            classes.append("terminal")
        if sid == self.initial_state:
            classes.append("initial")

        badges: List[str] = []
        if sid == self.initial_state:
            badges.append("<span class='badge initial'>INITIAL</span>")
        if sid in terminal:
            badges.append("<span class='badge terminal'>TERMINAL</span>")
        if sid in branching:
            badges.append(f"<span class='badge branch'>OUT:{self.out_degree.get(sid, 0)}</span>")
        if sid in merging:
            badges.append(f"<span class='badge merge'>IN:{self.in_degree.get(sid, 0)}</span>")
        tc = state.metadata.get("trace_count")
        if tc:
            badges.append(f"<span class='badge traces'>Traces:{tc}</span>")

        return (
            f"<div class='{' '.join(classes)}' data-sid='{sid}' "
            f"style='left:{pos['x']}px;top:{pos['y']}px;'>"
            f"<div class='state-header' style='border-color:{color};'>"
            f"<span class='state-id'>{self._esc(sid)}</span>"
            f"<span class='step-badge'>Step {step}</span></div>"
            f"<div class='tool-badge' style='background:{color};'>{self._esc(tool)}</div>"
            f"<div class='observation'>{self._esc(obs)}</div>"
            f"<div class='badges'>{''.join(badges)}</div>"
            f"</div>"
        )

    def _stats_sidebar(self, branching, merging, terminal) -> str:
        tool_counts: Dict[str, int] = {}
        for t in self.transitions:
            tool_counts[t.action_type] = tool_counts.get(t.action_type, 0) + 1
        rows = "".join(
            f"<div class='stat-row'><span class='stat-label'>{t}</span><span class='stat-value'>{c}</span></div>"
            for t, c in sorted(tool_counts.items(), key=lambda x: -x[1])[:8]
        )
        return (
            f"<div class='stats-sidebar' id='statsSidebar'>"
            f"<h3>Statistics</h3>"
            f"<div class='stat-row'><span class='stat-label'>States</span><span class='stat-value'>{len(self.states)}</span></div>"
            f"<div class='stat-row'><span class='stat-label'>Transitions</span><span class='stat-value'>{len(self.transitions)}</span></div>"
            f"<div class='stat-row'><span class='stat-label'>Branching</span><span class='stat-value'>{len(branching)}</span></div>"
            f"<div class='stat-row'><span class='stat-label'>Merging</span><span class='stat-value'>{len(merging)}</span></div>"
            f"<div class='stat-row'><span class='stat-label'>Terminal</span><span class='stat-value'>{len(terminal)}</span></div>"
            f"<h3 style='margin-top:12px;'>Tools</h3>{rows}</div>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Match-result HTML renderer (internal)
# ═══════════════════════════════════════════════════════════════════════════

def _esc(text: str) -> str:
    return html_lib.escape(str(text))


def _pct_bar(value: float, max_val: float = 100.0) -> str:
    pct = min(value / max_val * 100, 100) if max_val else 0
    if pct >= 80:
        c = "#3fb950"
    elif pct >= 50:
        c = "#d29922"
    else:
        c = "#f85149"
    return (
        f"<div class='bar-bg'>"
        f"<div class='bar-fg' style='width:{pct:.1f}%;background:{c};'></div>"
        f"<span class='bar-label'>{value:.1f}%</span>"
        f"</div>"
    )


def _render_match_html(result: MatchResult) -> str:
    m = result.metrics
    parts: List[str] = []

    parts.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
    parts.append("<title>SWE Trace Match Report</title>")
    parts.append(_MATCH_CSS)
    parts.append("</head><body>")

    # Header
    parts.append("<header><h1>SWE Trace Match Report</h1></header>")
    parts.append("<main>")

    # Summary card
    status = "PASS" if m.perfect_match else ("PARTIAL" if m.coverage_percent > 0 else "FAIL")
    status_cls = "pass" if m.perfect_match else ("partial" if m.coverage_percent > 0 else "fail")

    parts.append(f"<section class='summary'>")
    parts.append(f"<div class='status {status_cls}'>{status}</div>")
    parts.append(f"<div class='kpi-grid'>")
    parts.append(f"<div class='kpi'><div class='kpi-label'>Coverage</div>{_pct_bar(m.coverage_percent)}</div>")
    parts.append(f"<div class='kpi'><div class='kpi-label'>Terminal Match</div>"
                 f"<div class='kpi-value {'good' if m.terminal_state_match else 'bad'}'>"
                 f"{'Yes' if m.terminal_state_match else 'No'}</div></div>")
    parts.append(f"<div class='kpi'><div class='kpi-label'>Matched / Total</div>"
                 f"<div class='kpi-value'>{m.matched_count} / {m.total_ground_truth_states}</div></div>")
    parts.append(f"<div class='kpi'><div class='kpi-label'>Candidate States</div>"
                 f"<div class='kpi-value'>{m.candidate_states}</div></div>")
    parts.append(f"<div class='kpi'><div class='kpi-label'>Best Path</div>"
                 f"<div class='kpi-value'>#{m.best_path_index + 1} of {m.total_paths}</div></div>")
    parts.append("</div></section>")

    # Insights section
    parts.append("<section class='insights'><h2>Insights</h2><ul>")

    if result.divergence_index is not None:
        parts.append(
            f"<li class='insight warn'>First divergence at ground-truth step "
            f"<strong>{result.divergence_index}</strong>.</li>"
        )
    else:
        parts.append("<li class='insight ok'>No divergence \u2014 candidate covers all ground-truth steps.</li>")

    if result.missing_indexes:
        missing_str = ", ".join(str(i) for i in result.missing_indexes[:20])
        extra = f" \u2026 (+{len(result.missing_indexes) - 20} more)" if len(result.missing_indexes) > 20 else ""
        parts.append(f"<li class='insight warn'>Missing ground-truth steps: {missing_str}{extra}</li>")

    if m.terminal_state_match:
        parts.append("<li class='insight ok'>Terminal state matches ground truth.</li>")
    else:
        parts.append("<li class='insight warn'>Terminal state does NOT match ground truth.</li>")

    if result.equivalence_stats:
        cache_pct = 0.0
        total = result.equivalence_stats.get("total_checks", 0)
        hits = result.equivalence_stats.get("cache_hits", 0)
        if total:
            cache_pct = hits / total * 100
        parts.append(
            f"<li class='insight info'>Equivalence checks: {total} total, "
            f"{hits} cache hits ({cache_pct:.0f}%), "
            f"{result.equivalence_stats.get('llm_calls', 0)} LLM calls.</li>"
        )

    parts.append("</ul></section>")

    # Step-by-step alignment table
    if result.alignment:
        parts.append("<section class='alignment'><h2>Step Alignment</h2>")
        parts.append("<table><thead><tr>"
                     "<th>#</th><th>Candidate State</th><th>GT State</th>"
                     "<th>Result</th><th>Rationale</th></tr></thead><tbody>")
        for a in result.alignment:
            cls = "matched" if a.matched else "unmatched"
            gt_id = _esc(a.ground_truth_state_id or "\u2014")
            check = '\u2713' if a.matched else '\u2717'
            parts.append(
                f"<tr class='{cls}'>"
                f"<td>{a.candidate_step}</td>"
                f"<td>{_esc(a.candidate_state_id)}</td>"
                f"<td>{gt_id}</td>"
                f"<td>{check}</td>"
                f"<td>{_esc(a.rationale)}</td></tr>"
            )
        parts.append("</tbody></table></section>")

    # Raw JSON
    parts.append("<section class='raw'><h2>Raw JSON</h2>")
    parts.append(f"<pre><code>{_esc(match_to_json(result))}</code></pre>")
    parts.append("</section>")

    parts.append("</main></body></html>")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# CSS / JS constants
# ═══════════════════════════════════════════════════════════════════════════

# --- Trace graph CSS ---

_TRACE_CSS = """<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#c9d1d9;overflow:hidden}
.toolbar{position:fixed;top:10px;left:10px;z-index:100;background:#161b22;border:1px solid #30363d;padding:8px 12px;border-radius:8px;display:flex;gap:10px;align-items:center;font-size:.8rem}
.toolbar button{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:.75rem}
.toolbar button:hover{background:#30363d}
.stats-sidebar{position:fixed;top:60px;right:10px;z-index:100;background:#161b22;border:1px solid #30363d;padding:12px;border-radius:8px;width:220px;font-size:.7rem;max-height:calc(100vh - 80px);overflow-y:auto}
.stats-sidebar h3{margin-bottom:8px;color:#58a6ff;font-size:.85rem}
.stats-sidebar .stat-row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #21262d}
.stats-sidebar .stat-label{color:#8b949e}
.stats-sidebar .stat-value{color:#c9d1d9;font-weight:600}
.scroll-shell{width:100vw;height:100vh;overflow:auto;padding-left:10px;padding-top:60px}
.wrapper{position:relative;min-width:100%;min-height:100%}
.canvas-container{position:relative;transform-origin:0 0}
svg{position:absolute;left:0;top:0;pointer-events:none}
.edge{transition:stroke-width .2s}
.edge-label{font-size:10px;font-family:monospace}
.state{position:absolute;width:320px;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;cursor:grab;transition:box-shadow .2s}
.state:hover{box-shadow:0 0 20px rgba(88,166,255,.3)}
.state.initial{border-color:#3fb950}
.state.terminal{border-color:#f85149}
.state.branch{border-color:#58a6ff}
.state.merge{border-color:#d29922}
.state-header{display:flex;justify-content:space-between;align-items:center;padding-bottom:6px;margin-bottom:8px;border-bottom:2px solid #30363d}
.state-id{font-weight:600;font-size:.75rem;color:#58a6ff}
.step-badge{background:#21262d;padding:2px 6px;border-radius:4px;font-size:.65rem;color:#8b949e}
.tool-badge{display:inline-block;padding:3px 8px;border-radius:4px;font-size:.7rem;font-weight:600;color:#0d1117;margin-bottom:8px}
.observation{font-size:.68rem;color:#8b949e;background:#0d1117;padding:6px;border-radius:4px;margin-bottom:6px;max-height:50px;overflow:hidden;line-height:1.3}
.badges{display:flex;flex-wrap:wrap;gap:4px}
.badge{font-size:.55rem;padding:2px 5px;border-radius:3px;font-weight:600}
.badge.initial{background:#238636;color:#fff}
.badge.terminal{background:#da3633;color:#fff}
.badge.branch{background:#1f6feb;color:#fff}
.badge.merge{background:#9e6a03;color:#fff}
.badge.traces{background:#30363d;color:#8b949e}
</style>"""

_TRACE_TOOLBAR = """<div class='toolbar'>
<span style='font-weight:600;color:#58a6ff;'>SWE Trace Viewer</span>
<button id='zoomIn'>+ Zoom</button>
<button id='zoomOut'>- Zoom</button>
<span id='zoomLvl'>100%</span>
<button id='resetLayout'>Reset</button>
<button id='toggleStats'>Stats</button>
</div>"""

_TRACE_JS = """<script>
(function(){
const canvas=document.getElementById('canvas');
const nodes=[...document.querySelectorAll('.state[data-sid]')];
const edges=[...document.querySelectorAll('path.edge')];
const initPos={};
nodes.forEach(n=>{initPos[n.dataset.sid]={left:parseFloat(n.style.left),top:parseFloat(n.style.top)}});
let zoom=1;
function applyZoom(){canvas.style.transform=`scale(${zoom})`;document.getElementById('zoomLvl').textContent=Math.round(zoom*100)+'%';}
document.getElementById('zoomIn').onclick=()=>{zoom=Math.min(3,zoom+.1);applyZoom()};
document.getElementById('zoomOut').onclick=()=>{zoom=Math.max(.2,zoom-.1);applyZoom()};
document.getElementById('resetLayout').onclick=()=>{zoom=1;applyZoom();nodes.forEach(n=>{const p=initPos[n.dataset.sid];n.style.left=p.left+'px';n.style.top=p.top+'px'});redrawEdges()};
document.getElementById('toggleStats').onclick=()=>{const s=document.getElementById('statsSidebar');s.style.display=s.style.display==='none'?'block':'none'};
function redrawEdges(){edges.forEach(e=>{const f=document.querySelector(`.state[data-sid='${e.dataset.from}']`),t=document.querySelector(`.state[data-sid='${e.dataset.to}']`);if(!f||!t)return;const fw=f.offsetWidth,fh=f.offsetHeight;const fx=parseFloat(f.style.left)+fw,fy=parseFloat(f.style.top)+fh/2,tx=parseFloat(t.style.left),ty=parseFloat(t.style.top)+t.offsetHeight/2;const c1=fx+(tx-fx)*.4,c2=tx-(tx-fx)*.4;e.setAttribute('d',`M ${fx} ${fy} C ${c1} ${fy} ${c2} ${ty} ${tx} ${ty}`);})}
let dragging=null,ox=0,oy=0;
nodes.forEach(n=>{n.addEventListener('pointerdown',e=>{dragging=n;n.setPointerCapture(e.pointerId);const r=n.getBoundingClientRect();ox=e.clientX-r.left;oy=e.clientY-r.top;n.style.cursor='grabbing';n.style.zIndex=1000})});
window.addEventListener('pointermove',e=>{if(!dragging)return;dragging.style.left=(e.clientX-ox)/zoom+'px';dragging.style.top=(e.clientY-oy)/zoom+'px';redrawEdges()});
window.addEventListener('pointerup',e=>{if(dragging){dragging.style.cursor='grab';dragging.style.zIndex='';dragging.releasePointerCapture(e.pointerId);dragging=null}});
applyZoom();
})();
window.ptaStates=__STATES_JSON__;
</script>"""

# --- Trace list CSS ---

_TRACE_LIST_CSS = """<style>
body{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;padding:20px}
h1{color:#58a6ff;margin-bottom:20px}
.transition{display:flex;align-items:flex-start;gap:15px;padding:12px;background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:10px}
.state-card{background:#21262d;border:1px solid #30363d;border-radius:6px;padding:10px;width:300px;flex-shrink:0}
.state-card h3{color:#58a6ff;font-size:.8rem;margin:0 0 6px}
.state-card .tool{display:inline-block;padding:2px 6px;border-radius:4px;font-size:.7rem;font-weight:600;color:#0d1117;margin-bottom:6px}
.state-card .obs{font-size:.68rem;color:#8b949e;background:#0d1117;padding:6px;border-radius:4px;max-height:80px;overflow:hidden;line-height:1.3}
.action{font-family:monospace;font-size:.75rem;background:#0d1117;padding:8px 12px;border-radius:6px;border:1px solid #30363d;align-self:center;color:#7ee787}
.stats{margin-bottom:20px;font-size:.8rem;color:#8b949e}
</style>"""

# --- Match report CSS ---

_MATCH_CSS = """<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:0;line-height:1.6}
header{background:#161b22;border-bottom:1px solid #30363d;padding:20px 32px}
header h1{color:#58a6ff;font-size:1.4rem}
main{max-width:1100px;margin:0 auto;padding:24px 32px}
h2{color:#58a6ff;margin:24px 0 12px;font-size:1.1rem;border-bottom:1px solid #21262d;padding-bottom:6px}
section{margin-bottom:28px}

/* Summary */
.summary{display:flex;align-items:flex-start;gap:24px;flex-wrap:wrap}
.status{font-size:2rem;font-weight:700;padding:16px 32px;border-radius:12px;text-align:center;min-width:140px}
.status.pass{background:#238636;color:#fff}
.status.partial{background:#9e6a03;color:#fff}
.status.fail{background:#da3633;color:#fff}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;flex:1}
.kpi{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}
.kpi-label{font-size:.75rem;color:#8b949e;margin-bottom:4px}
.kpi-value{font-size:1.1rem;font-weight:600}
.kpi-value.good{color:#3fb950}
.kpi-value.bad{color:#f85149}

/* Bar */
.bar-bg{background:#21262d;border-radius:4px;height:22px;position:relative;overflow:hidden}
.bar-fg{height:100%;border-radius:4px;transition:width .4s}
.bar-label{position:absolute;top:0;left:8px;line-height:22px;font-size:.75rem;font-weight:600;color:#fff}

/* Insights */
.insights ul{list-style:none;padding:0}
.insight{padding:8px 12px;border-radius:6px;margin-bottom:6px;font-size:.85rem}
.insight.ok{background:#0d331c;border-left:4px solid #3fb950}
.insight.warn{background:#3d2400;border-left:4px solid #d29922}
.insight.info{background:#0d1d33;border-left:4px solid #58a6ff}

/* Table */
table{width:100%;border-collapse:collapse;font-size:.8rem}
thead th{background:#161b22;color:#8b949e;padding:8px 10px;text-align:left;border-bottom:2px solid #30363d}
tbody td{padding:6px 10px;border-bottom:1px solid #21262d;word-break:break-all}
tr.matched td:nth-child(4){color:#3fb950}
tr.unmatched td:nth-child(4){color:#f85149}
tr.unmatched{background:#1a1000}

/* Raw JSON */
.raw pre{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;overflow-x:auto;font-size:.75rem}
.raw code{color:#c9d1d9}
</style>"""
