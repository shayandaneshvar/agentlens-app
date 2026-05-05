import { useRef, useState } from 'react';
import type { ComparisonData, ComparisonStateSummary } from '../types';

const STAGE_COLORS: Record<string, string> = {
  exploration: '#3b82f6',
  implementation: '#16a34a',
  verification: '#ea580c',
  orchestration: '#7c3aed',
};

const BLOCK_W = 56;
const BLOCK_H = 44;
const GAP = 5;
const LANE_GAP = 90;
const LANE_Y_GT = 12;
const LANE_Y_CAND = LANE_Y_GT + BLOCK_H + LANE_GAP;
const HEADER_H = 22;

function stageColor(stage: string): string {
  return STAGE_COLORS[stage.toLowerCase()] ?? '#94a3b8';
}

/** Build a compact 2-line label: tool abbrev (line 1) + file hint (line 2). */
function blockLabel(state: ComparisonStateSummary): [string, string] {
  const tool = state.tool
    ? state.tool.replace(/^(run_terminal_command|execute_command)$/i, 'run')
        .replace(/^(create_file|write_file)$/i, 'create')
        .replace(/^(read_file|view_file)$/i, 'read')
        .replace(/^(list_dir|list_directory)$/i, 'ls')
        .replace(/^(search|grep_search|find)$/i, 'search')
        .substring(0, 6)
    : '?';
  const file = state.file_path
    ? state.file_path.split('/').pop()?.substring(0, 7) ?? ''
    : '';
  return [tool, file];
}

interface Props {
  comparison: ComparisonData;
}

export function ComparisonView({ comparison }: Props) {
  const { gt_path, candidate_path, alignment, gt_matched_indexes, candidate_matched_indexes, terminal_state_match } = comparison;
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; state: ComparisonStateSummary; lane: string } | null>(null);

  const gtMatchedSet = new Set(gt_matched_indexes);
  const candMatchedSet = new Set(candidate_matched_indexes);

  const maxLen = Math.max(gt_path.length, candidate_path.length);
  const svgWidth = Math.max(maxLen * (BLOCK_W + GAP) + 160, 400);
  const svgHeight = LANE_Y_CAND + BLOCK_H + 50;

  // Block x position
  const blockX = (index: number) => 100 + index * (BLOCK_W + GAP);

  // Handle tooltip — use fixed viewport positioning so it's never clipped
  const showTooltip = (e: React.MouseEvent, state: ComparisonStateSummary, lane: string) => {
    setTooltip({ x: e.clientX, y: e.clientY, state, lane });
  };
  const hideTooltip = () => setTooltip(null);

  const containerRef = useRef<HTMLDivElement>(null);

  return (
    <div className="comparison-view">
      <div className="comparison-header">
        <h3 style={{ margin: 0, fontSize: 14 }}>Merged PTA vs Candidate Trajectory</h3>
        <div className="comparison-legend">
          <span className="legend-item"><span className="legend-box" style={{ background: '#22c55e' }} /> Matched</span>
          <span className="legend-item"><span className="legend-box" style={{ background: '#ef4444' }} /> Missed (GT)</span>
          <span className="legend-item"><span className="legend-box" style={{ background: '#f59e0b' }} /> Extra (Candidate)</span>
          <span className="legend-item"><span className="legend-box legend-line" /> Match link</span>
        </div>
      </div>

      <div className="comparison-body" ref={containerRef}>
        <svg
          ref={svgRef}
          width={svgWidth}
          height={svgHeight}
          className="comparison-svg"
        >
          {/* Lane labels */}
          <text x={8} y={LANE_Y_GT + BLOCK_H / 2 + HEADER_H} className="lane-label" dominantBaseline="middle">GT Path</text>
          <text x={8} y={LANE_Y_CAND + BLOCK_H / 2 + HEADER_H} className="lane-label" dominantBaseline="middle">Candidate</text>

          {/* Start indicators */}
          {gt_path.length > 0 && (
            <g>
              <text x={blockX(0) + BLOCK_W / 2} y={LANE_Y_GT + HEADER_H - 14} textAnchor="middle" className="state-marker start-marker">START</text>
            </g>
          )}
          {candidate_path.length > 0 && (
            <g>
              <text x={blockX(0) + BLOCK_W / 2} y={LANE_Y_CAND + HEADER_H - 14} textAnchor="middle" className="state-marker start-marker">START</text>
            </g>
          )}

          {/* Terminal indicators */}
          {gt_path.length > 0 && (
            <g>
              <text x={blockX(gt_path.length - 1) + BLOCK_W / 2} y={LANE_Y_GT + HEADER_H + BLOCK_H + 16} textAnchor="middle" className="state-marker terminal-marker">TERMINAL</text>
            </g>
          )}
          {candidate_path.length > 0 && (
            <g>
              <text
                x={blockX(candidate_path.length - 1) + BLOCK_W / 2}
                y={LANE_Y_CAND + HEADER_H + BLOCK_H + 16}
                textAnchor="middle"
                className={`state-marker terminal-marker ${terminal_state_match ? 'terminal-match' : 'terminal-miss'}`}
              >
                {terminal_state_match ? 'TERMINAL ✓' : 'TERMINAL ✗'}
              </text>
            </g>
          )}

          {/* Step numbers header */}
          {gt_path.map((_, i) => (
            <text key={`gt-step-${i}`} x={blockX(i) + BLOCK_W / 2} y={LANE_Y_GT + HEADER_H - 4} textAnchor="middle" className="step-num">
              {i + 1}
            </text>
          ))}
          {candidate_path.map((_, i) => (
            <text key={`c-step-${i}`} x={blockX(i) + BLOCK_W / 2} y={LANE_Y_CAND + HEADER_H - 4} textAnchor="middle" className="step-num">
              {i + 1}
            </text>
          ))}

          {/* Match lines (drawn first, behind blocks) */}
          {alignment.map((pair, i) => {
            const x1 = blockX(pair.gt_index) + BLOCK_W / 2;
            const y1 = LANE_Y_GT + BLOCK_H + HEADER_H;
            const x2 = blockX(pair.candidate_index) + BLOCK_W / 2;
            const y2 = LANE_Y_CAND + HEADER_H;
            return (
              <line
                key={`match-${i}`}
                x1={x1} y1={y1} x2={x2} y2={y2}
                className="match-line"
              />
            );
          })}

          {/* GT path blocks */}
          {gt_path.map((state, i) => {
            const matched = gtMatchedSet.has(i);
            const x = blockX(i);
            const y = LANE_Y_GT + HEADER_H;
            const fill = stageColor(state.intent_stage);
            const opacity = matched ? 1.0 : 0.35;
            const strokeColor = matched ? '#22c55e' : '#ef4444';
            const [toolLabel, fileLabel] = blockLabel(state);
            return (
              <g key={`gt-${i}`}
                onMouseEnter={(e) => showTooltip(e, state, 'GT')}
                onMouseLeave={hideTooltip}
                style={{ cursor: 'pointer' }}
              >
                <rect
                  x={x} y={y} width={BLOCK_W} height={BLOCK_H}
                  rx={4}
                  fill={fill}
                  opacity={opacity}
                  stroke={strokeColor}
                  strokeWidth={matched ? 2 : 2.5}
                  strokeDasharray={matched ? undefined : '6 3'}
                />
                <text x={x + BLOCK_W / 2} y={y + BLOCK_H / 2 - (fileLabel ? 6 : 0)} textAnchor="middle" dominantBaseline="middle" className="block-text">
                  {toolLabel}
                </text>
                {fileLabel && (
                  <text x={x + BLOCK_W / 2} y={y + BLOCK_H / 2 + 10} textAnchor="middle" dominantBaseline="middle" className="block-file-text">
                    {fileLabel}
                  </text>
                )}
              </g>
            );
          })}

          {/* Candidate path blocks */}
          {candidate_path.map((state, i) => {
            const matched = candMatchedSet.has(i);
            const x = blockX(i);
            const y = LANE_Y_CAND + HEADER_H;
            const fill = stageColor(state.intent_stage);
            const strokeColor = matched ? '#22c55e' : '#f59e0b';
            const [toolLabel, fileLabel] = blockLabel(state);
            return (
              <g key={`cand-${i}`}
                onMouseEnter={(e) => showTooltip(e, state, 'Candidate')}
                onMouseLeave={hideTooltip}
                style={{ cursor: 'pointer' }}
              >
                <rect
                  x={x} y={y} width={BLOCK_W} height={BLOCK_H}
                  rx={4}
                  fill={fill}
                  stroke={strokeColor}
                  strokeWidth={2}
                />
                <text x={x + BLOCK_W / 2} y={y + BLOCK_H / 2 - (fileLabel ? 6 : 0)} textAnchor="middle" dominantBaseline="middle" className="block-text">
                  {toolLabel}
                </text>
                {fileLabel && (
                  <text x={x + BLOCK_W / 2} y={y + BLOCK_H / 2 + 10} textAnchor="middle" dominantBaseline="middle" className="block-file-text">
                    {fileLabel}
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {/* Tooltip — fixed to viewport */}
        {tooltip && (
          <div
            className="comparison-tooltip"
            style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}
          >
            <div className="tt-lane">{tooltip.lane}</div>
            <div className="tt-tool">{tooltip.state.tool || '(no tool)'}</div>
            {tooltip.state.file_path && <div className="tt-file">{tooltip.state.file_path}</div>}
            <div className="tt-stage" style={{ color: stageColor(tooltip.state.intent_stage) }}>
              {tooltip.state.intent_stage}
            </div>
            {tooltip.state.content_description && (
              <div className="tt-desc">{tooltip.state.content_description}</div>
            )}
          </div>
        )}
      </div>

      {/* Summary stats */}
      <div className="comparison-stats">
        <span>GT path: <strong>{gt_path.length}</strong> states</span>
        <span>Candidate: <strong>{candidate_path.length}</strong> states</span>
        <span>Matched: <strong>{alignment.length}</strong></span>
        <span>GT missed: <strong>{gt_path.length - gt_matched_indexes.length}</strong></span>
        <span>Candidate extra: <strong>{candidate_path.length - candidate_matched_indexes.length}</strong></span>
        <span style={{ color: terminal_state_match ? '#16a34a' : '#dc2626' }}>
          Terminal: <strong>{terminal_state_match ? 'Matched ✓' : 'Different ✗'}</strong>
        </span>
      </div>
    </div>
  );
}
