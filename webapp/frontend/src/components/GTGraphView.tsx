import { useRef, useState, useMemo } from 'react';

const STAGE_COLORS: Record<string, string> = {
  exploration: '#3b82f6',
  implementation: '#22c55e',
  verification: '#f59e0b',
  orchestration: '#a855f7',
};

function stageColorHex(stage: string): string {
  return STAGE_COLORS[stage?.toLowerCase()] ?? '#94a3b8';
}

interface GTState {
  state_id?: string;
  step?: number;
  tool_used?: string;
  file_path?: string;
  intent_stage?: string;
  resulting_state?: string;
  metadata?: Record<string, unknown>;
}

interface GTTransition {
  from_state: string;
  to_state: string;
  action_type?: string;
  step?: number;
}

interface GTData {
  initial_state?: string;
  states?: Record<string, GTState>;
  transitions?: GTTransition[];
  branches?: Record<string, string[]>;
  statistics?: { num_states?: number; num_transitions?: number };
}

export interface CandidateOverlay {
  label: string;
  color: string;
  matchedStateIds: string[];
}

interface Props {
  data: GTData;
  /** Optional: overlay candidate paths on the GT graph. */
  candidateOverlays?: CandidateOverlay[];
}

const NODE_W = 110;
const NODE_H = 44;
const COL_GAP = 60;   // horizontal gap between layers (columns)
const ROW_GAP = 24;   // vertical gap between nodes in the same layer
const PADDING = 40;

/** Layout nodes left-to-right in columns using BFS from initial_state. */
function layoutGraph(data: GTData) {
  const states = data.states ?? {};
  const transitions = data.transitions ?? [];
  const branches = data.branches ?? {};
  const initial = data.initial_state;

  // Build adjacency
  const children: Record<string, string[]> = {};
  const parents: Record<string, string[]> = {};
  for (const t of transitions) {
    if (!children[t.from_state]) children[t.from_state] = [];
    children[t.from_state].push(t.to_state);
    if (!parents[t.to_state]) parents[t.to_state] = [];
    parents[t.to_state].push(t.from_state);
  }

  // All branch targets (for highlighting)
  const branchTargets = new Set<string>();
  const branchSources = new Set(Object.keys(branches));
  for (const targets of Object.values(branches)) {
    for (const t of targets) branchTargets.add(t);
  }

  // BFS layering
  const layer: Record<string, number> = {};
  const visited = new Set<string>();
  const queue: string[] = [];

  if (initial && states[initial]) {
    queue.push(initial);
    layer[initial] = 0;
    visited.add(initial);
  } else {
    const allIds = Object.keys(states);
    const hasIncoming = new Set(transitions.map((t) => t.to_state));
    for (const id of allIds) {
      if (!hasIncoming.has(id)) {
        queue.push(id);
        layer[id] = 0;
        visited.add(id);
      }
    }
    if (queue.length === 0 && allIds.length > 0) {
      queue.push(allIds[0]);
      layer[allIds[0]] = 0;
      visited.add(allIds[0]);
    }
  }

  while (queue.length > 0) {
    const id = queue.shift()!;
    const nextLayer = (layer[id] ?? 0) + 1;
    for (const child of children[id] ?? []) {
      if (!visited.has(child)) {
        visited.add(child);
        layer[child] = nextLayer;
        queue.push(child);
      }
    }
  }

  // Place orphans
  for (const id of Object.keys(states)) {
    if (!(id in layer)) layer[id] = 0;
  }

  // Group by layer
  const layers: Record<number, string[]> = {};
  for (const [id, l] of Object.entries(layer)) {
    if (!layers[l]) layers[l] = [];
    layers[l].push(id);
  }

  // Sort within each layer by step number
  for (const ids of Object.values(layers)) {
    ids.sort((a, b) => (states[a]?.step ?? 0) - (states[b]?.step ?? 0));
  }

  const maxLayer = Math.max(...Object.keys(layers).map(Number), 0);
  const maxNodesInLayer = Math.max(...Object.values(layers).map((l) => l.length), 1);

  // Compute positions — layers go left-to-right, items stack vertically
  const positions: Record<string, { x: number; y: number }> = {};
  for (let l = 0; l <= maxLayer; l++) {
    const ids = layers[l] ?? [];
    const totalHeight = ids.length * NODE_H + (ids.length - 1) * ROW_GAP;
    const startY = PADDING + (maxNodesInLayer * (NODE_H + ROW_GAP) - ROW_GAP) / 2 - totalHeight / 2;
    for (let i = 0; i < ids.length; i++) {
      positions[ids[i]] = {
        x: PADDING + l * (NODE_W + COL_GAP),
        y: startY + i * (NODE_H + ROW_GAP),
      };
    }
  }

  const svgWidth = (maxLayer + 1) * (NODE_W + COL_GAP) - COL_GAP + PADDING * 2;
  const svgHeight = maxNodesInLayer * (NODE_H + ROW_GAP) - ROW_GAP + PADDING * 2;

  // Terminal states (no outgoing)
  const withOutgoing = new Set(transitions.map((t) => t.from_state));
  const terminals = new Set(Object.keys(states).filter((id) => !withOutgoing.has(id)));

  return { positions, svgWidth, svgHeight, branchSources, branchTargets, terminals, transitions, states };
}

export function GTGraphView({ data, candidateOverlays }: Props) {
  const layout = useMemo(() => layoutGraph(data), [data]);
  const { positions, svgWidth, svgHeight, branchSources, branchTargets, terminals, transitions, states } = layout;
  const [tooltip, setTooltip] = useState<{ x: number; y: number; state: GTState; id: string } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  // Pre-compute which candidates matched each GT state
  const overlayMap = useMemo(() => {
    if (!candidateOverlays || candidateOverlays.length === 0) return null;
    const m: Record<string, Array<{ label: string; color: string }>> = {};
    for (const ov of candidateOverlays) {
      for (const sid of ov.matchedStateIds) {
        if (!m[sid]) m[sid] = [];
        m[sid].push({ label: ov.label, color: ov.color });
      }
    }
    return m;
  }, [candidateOverlays]);

  // Extra height for overlay dots below nodes
  const overlayRowH = overlayMap ? 14 : 0;
  const effectiveSvgHeight = svgHeight + (overlayMap ? 20 : 0);

  return (
    <div>
      <div style={{ display: 'flex', gap: 16, fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, flexWrap: 'wrap' }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: '#3b82f6', display: 'inline-block' }} /> exploration
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: '#22c55e', display: 'inline-block' }} /> implementation
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: '#f59e0b', display: 'inline-block' }} /> verification
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: '#a855f7', display: 'inline-block' }} /> orchestration
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', border: '2px dashed #e879f9', display: 'inline-block' }} /> branch point
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, border: '2px solid #ef4444', display: 'inline-block' }} /> terminal
        </span>
        {candidateOverlays && candidateOverlays.map((ov) => (
          <span key={ov.label} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: ov.color, display: 'inline-block' }} /> {ov.label}
          </span>
        ))}
      </div>

      <div ref={containerRef} style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: 400, border: '1px solid var(--border)', borderRadius: 6, background: 'var(--bg-card)' }}>
        <svg width={svgWidth} height={effectiveSvgHeight} style={{ display: 'block' }}>
          <defs>
            <marker id="gt-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6 Z" fill="var(--text-muted)" />
            </marker>
            <marker id="gt-arrow-branch" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6 Z" fill="#e879f9" />
            </marker>
          </defs>

          {/* Edges */}
          {transitions.map((t, i) => {
            const from = positions[t.from_state];
            const to = positions[t.to_state];
            if (!from || !to) return null;
            const isBranch = branchSources.has(t.from_state) && branchTargets.has(t.to_state);
            const x1 = from.x + NODE_W;          // right edge of source
            const y1 = from.y + NODE_H / 2;      // vertical center
            const x2 = to.x;                     // left edge of target
            const y2 = to.y + NODE_H / 2;
            // Horizontal curved path
            const dx = x2 - x1;
            const dy = y2 - y1;
            const mid = dx / 2;
            const path = Math.abs(dy) < 5
              ? `M${x1},${y1} L${x2},${y2}`
              : `M${x1},${y1} C${x1 + mid},${y1} ${x2 - mid},${y2} ${x2},${y2}`;
            return (
              <path
                key={`edge-${i}`}
                d={path}
                fill="none"
                stroke={isBranch ? '#e879f9' : 'var(--text-muted)'}
                strokeWidth={isBranch ? 2 : 1.2}
                strokeDasharray={isBranch ? '5 3' : undefined}
                opacity={isBranch ? 0.8 : 0.4}
                markerEnd={isBranch ? 'url(#gt-arrow-branch)' : 'url(#gt-arrow)'}
              />
            );
          })}

          {/* Nodes */}
          {Object.entries(positions).map(([id, pos]) => {
            const s = states[id];
            if (!s) return null;
            const stage = s.intent_stage ?? '';
            const fill = stageColorHex(stage);
            const isBranch = branchSources.has(id);
            const isTerminal = terminals.has(id);
            const tool = s.tool_used || '—';
            const toolLabel = tool.length > 14 ? tool.slice(0, 12) + '…' : tool;
            const file = s.file_path ? s.file_path.split('/').pop()?.slice(0, 10) ?? '' : '';
            const hasOverlayMatch = overlayMap ? !!overlayMap[id] : false;
            const dimmed = overlayMap && !hasOverlayMatch;

            return (
              <g
                key={id}
                style={{ cursor: 'pointer' }}
                onMouseEnter={(e) => setTooltip({ x: e.clientX, y: e.clientY, state: s, id })}
                onMouseLeave={() => setTooltip(null)}
              >
                <rect
                  x={pos.x} y={pos.y}
                  width={NODE_W} height={NODE_H}
                  rx={6}
                  fill={fill}
                  opacity={dimmed ? 0.35 : 0.85}
                  stroke={isTerminal ? '#ef4444' : isBranch ? '#e879f9' : 'transparent'}
                  strokeWidth={isTerminal || isBranch ? 2.5 : 0}
                  strokeDasharray={isBranch ? '5 3' : undefined}
                />
                <text
                  x={pos.x + NODE_W / 2} y={pos.y + (file ? 16 : NODE_H / 2)}
                  textAnchor="middle" dominantBaseline="middle"
                  fill="white" fontSize={10} fontWeight={600}
                >
                  {toolLabel}
                </text>
                {file && (
                  <text
                    x={pos.x + NODE_W / 2} y={pos.y + 32}
                    textAnchor="middle" dominantBaseline="middle"
                    fill="rgba(255,255,255,0.7)" fontSize={8}
                  >
                    {file}
                  </text>
                )}
                {/* Step badge */}
                <circle cx={pos.x + NODE_W - 2} cy={pos.y + 2} r={9} fill="var(--surface)" stroke="var(--border)" strokeWidth={1} />
                <text x={pos.x + NODE_W - 2} y={pos.y + 2} textAnchor="middle" dominantBaseline="central" fontSize={8} fill="var(--text-muted)">
                  {s.step ?? ''}
                </text>
              </g>
            );
          })}

          {/* Candidate overlay dots below nodes */}
          {overlayMap && Object.entries(positions).map(([id, pos]) => {
            const matches = overlayMap[id];
            if (!matches || matches.length === 0) return null;
            const dotR = 5;
            const dotGap = 13;
            const totalW = matches.length * dotGap - (dotGap - dotR * 2);
            const startX = pos.x + NODE_W / 2 - totalW / 2 + dotR;
            return matches.map((m, i) => (
              <circle
                key={`ov-${id}-${i}`}
                cx={startX + i * dotGap}
                cy={pos.y + NODE_H + overlayRowH - 4}
                r={dotR}
                fill={m.color}
                stroke="var(--bg-card)"
                strokeWidth={1.5}
              >
                <title>{m.label}</title>
              </circle>
            ));
          })}
        </svg>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div className="comparison-tooltip" style={{ left: tooltip.x + 12, top: tooltip.y - 10 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Step {tooltip.state.step}</div>
          <div><strong>Tool:</strong> {tooltip.state.tool_used || '—'}</div>
          <div><strong>Stage:</strong> {tooltip.state.intent_stage || '—'}</div>
          {tooltip.state.file_path && <div><strong>File:</strong> {tooltip.state.file_path}</div>}
          {tooltip.state.resulting_state && <div><strong>Result:</strong> {tooltip.state.resulting_state}</div>}
          {branchSources.has(tooltip.id) && <div style={{ color: '#e879f9', marginTop: 4 }}>⑂ Branch point</div>}
          {terminals.has(tooltip.id) && <div style={{ color: '#ef4444', marginTop: 4 }}>◼ Terminal state</div>}
          {overlayMap && overlayMap[tooltip.id] && (
            <div style={{ marginTop: 4 }}>
              <strong>Matched by:</strong>{' '}
              {overlayMap[tooltip.id].map((m, i) => (
                <span key={i} style={{ color: m.color, fontWeight: 600 }}>
                  {i > 0 ? ', ' : ''}{m.label}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
