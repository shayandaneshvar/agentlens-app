import { useState } from 'react';
import type { ProfileResponse } from '../types';

const STAGE_COLORS: Record<string, string> = {
  exploration: 'var(--stage-exploration)',
  implementation: 'var(--stage-implementation)',
  verification: 'var(--stage-verification)',
  orchestration: 'var(--stage-orchestration)',
};

const STAGE_DESCRIPTIONS: Record<string, string> = {
  exploration: 'Understanding the codebase. Reading files, searching for symbols, investigating project structure and dependencies.',
  implementation: 'Writing or modifying code. Creating files, editing source, applying patches, and making functional changes.',
  verification: 'Testing and validating changes. Running tests, linting, reviewing output, and confirming correctness.',
  orchestration: 'Coordinating workflow. Planning next steps, delegating to sub-agents, managing context, and compacting history.',
};

const HX_LABELS: Record<string, string> = {
  autonomy: 'Autonomy',
  low_friction: 'Low Friction',
  responsiveness: 'Responsiveness',
  stability: 'Stability',
};

function stageColor(stage: string) {
  return STAGE_COLORS[stage.toLowerCase()] ?? 'var(--stage-unknown)';
}

function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function PhaseInfoButton() {
  const [show, setShow] = useState(false);
  return (
    <span
      style={{ position: 'relative', display: 'inline-flex', cursor: 'pointer' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <path d="M12 16v-4M12 8h.01" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      {show && (
        <div style={{
          position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)',
          marginTop: 6, width: 340, padding: '10px 14px',
          background: 'var(--surface, #fff)', border: '1px solid var(--border)',
          borderRadius: 8, boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
          zIndex: 100, fontSize: 12, lineHeight: 1.5, color: 'var(--text)',
        }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 13 }}>Phase Definitions</div>
          {Object.entries(STAGE_DESCRIPTIONS).map(([stage, desc]) => (
            <div key={stage} style={{ marginBottom: 6, display: 'flex', gap: 8, alignItems: 'flex-start' }}>
              <span style={{
                display: 'inline-block', width: 10, height: 10, borderRadius: 2, flexShrink: 0,
                marginTop: 3, background: STAGE_COLORS[stage] ?? '#999',
              }} />
              <span><strong style={{ textTransform: 'capitalize' }}>{stage}:</strong> {desc}</span>
            </div>
          ))}
        </div>
      )}
    </span>
  );
}

interface Props {
  profile: ProfileResponse;
  passed: boolean | null;
  onStartAssessment: () => void;
}

export function Tier1Dashboard({ profile, passed, onStartAssessment }: Props) {
  const p = profile;
  const isEmpty = p.state_count === 0;

  return (
    <div>
      <h2 style={{ fontSize: 16, marginBottom: 16 }}>Behavioral Profile</h2>

      {/* Trajectory metadata */}
      {(p.agent || p.model || p.task || p.benchmark) && (
        <div className="trace-metadata" style={{
          display: 'flex', gap: 16, flexWrap: 'wrap',
          fontSize: 12, color: 'var(--text-muted)',
          marginBottom: 14, padding: '8px 12px',
          background: 'var(--bg-card)', borderRadius: 6,
          border: '1px solid var(--border)',
        }}>
          {p.agent && <span><strong>Agent:</strong> {p.agent}</span>}
          {p.model && <span><strong>Model:</strong> {p.model}</span>}
          {p.task && <span><strong>Task:</strong> {p.task}</span>}
          {p.benchmark && <span><strong>Benchmark:</strong> {p.benchmark}</span>}
        </div>
      )}

      {/* Pass/fail banner + assessment button in same row */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'stretch', marginBottom: 16 }}>
        {passed !== null && (
          <div className={`outcome-banner ${passed ? 'outcome-pass' : 'outcome-fail'}`} style={{ flex: 1, marginBottom: 0 }}>
            <span className="outcome-icon">{passed ? '✓' : '✗'}</span>
            <div>
              <div className="outcome-title">
                {passed ? 'Passed' : 'Failed'}
              </div>
              <div className="outcome-subtitle">
                {passed
                  ? 'This trajectory resolved the task successfully.'
                  : 'This trajectory did not resolve the task.'}
              </div>
            </div>
          </div>
        )}
        <button className="gt-btn" onClick={onStartAssessment} style={{
          fontSize: 12, padding: '8px 12px', width: 90, textAlign: 'center',
          lineHeight: 1.3, alignSelf: 'stretch',
        }}>
          Run Quality Assessment
        </button>
      </div>

      {isEmpty && (
        <div style={{
          padding: '14px 18px',
          marginBottom: 16,
          background: '#fef3cd',
          border: '1px solid #ffc107',
          borderRadius: 6,
          color: '#664d03',
          fontSize: 13,
          lineHeight: 1.5,
        }}>
          <strong>No actionable steps found.</strong> This trajectory may contain only
          system/message entries, or its format could not be fully parsed. The agent
          may have failed before performing any tool calls. You can still run a quality
          assessment — it will show 0% coverage.
        </div>
      )}

      {/* Stat cards */}
      <div className="stat-cards">
        <StatCard label="States" value={p.state_count} />
        <StatCard label="Files" value={p.file_count} />
        <StatCard label="Tools" value={p.tool_count} />
        <StatCard
          label="Coherence"
          value={`${(p.coherence * 100).toFixed(0)}%`}
          sub={p.coherence_label}
        />
        <StatCard
          label="Completed"
          value={p.completed === null ? '-' : p.completed ? 'Yes' : 'No'}
          sub={p.completed === null ? 'Not available' : p.completed ? 'Agent finished' : 'Agent did not finish'}
          className={p.completed === null ? '' : p.completed ? 'completed-yes' : 'completed-no'}
        />
        <StatCard
          label="Explore / Implement"
          value={`${p.exploration_ratio.toFixed(1)}x`}
          sub={`${p.stage_distribution['exploration'] ?? 0} explore · ${p.stage_distribution['implementation'] ?? 0} implement`}
        />
        <StatCard
          label="Files Modified"
          value={p.files_modified}
          sub={`${p.files_read_only} read-only`}
        />
        <StatCard
          label="Human Inputs"
          value={p.human_input_count ?? '—'}
          sub={p.human_input_count != null ? (p.human_input_count === 0 ? 'Fully autonomous' : 'User interactions') : 'Not available'}
        />
        <StatCard
          label="Subagents"
          value={p.subagent_count ?? '—'}
          sub={p.subagent_count != null ? 'Delegated tasks' : 'Not available'}
        />
        <StatCard
          label="Active Time"
          value={p.active_time_ms != null ? `${(p.active_time_ms / 1000).toFixed(0)}s` : '—'}
          sub={p.active_time_ms != null ? 'Working time' : 'Not available'}
        />
        <StatCard
          label="Compactions"
          value={p.compaction_count ?? '—'}
          sub={p.compaction_count != null ? 'Context window resets' : 'Not available'}
        />
        {p.human_experience_score != null && (
          <HxScoreCard score={p.human_experience_score} breakdown={p.hx_breakdown} />
        )}
      </div>

      {/* Human Experience section (ATIF only — hidden when no data) */}
      {(p.time_decomposition || p.step_latencies || p.step_token_cumulative) && (
        <div className="section">
          <div className="section-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            Human Experience
            <HxInfoButton />
          </div>

          <div className="hx-grid">
            {/* Time Decomposition Card */}
            <div className="hx-card">
              <div className="hx-card-header">
                <span>Time Breakdown</span>
                <InfoTip text="Where session time was spent. Agent Work = tool execution. LLM Thinking = model inference latency. Human Wait = time blocked on permission approvals." />
              </div>
              {p.time_decomposition ? (
                <TimeDecompositionBar data={p.time_decomposition} />
              ) : (
                <div className="hx-card-empty">Data unavailable</div>
              )}
            </div>

            {/* Response Latency Card */}
            <div className="hx-card">
              <div className="hx-card-header">
                <span>Response Latency</span>
                <InfoTip text="LLM response time for each agent step. Spikes indicate slow model responses. Rising trend may signal context window pressure. Dashed line = average." />
              </div>
              {p.step_latencies && p.step_latencies.length > 1 ? (
                <Sparkline values={p.step_latencies} color="var(--accent)" unit="ms" />
              ) : (
                <div className="hx-card-empty">Data unavailable</div>
              )}
            </div>

            {/* Token Pressure Card */}
            <div className="hx-card">
              <div className="hx-card-header">
                <span>Token Pressure</span>
                <InfoTip text="Cumulative prompt tokens fed to the model over the session. Steady growth = expanding context. Drops indicate compaction events (context window resets). Higher values may correlate with slower responses." />
              </div>
              {p.step_token_cumulative && p.step_token_cumulative.length > 1 ? (
                <Sparkline values={p.step_token_cumulative} color="var(--purple)" unit="" />
              ) : (
                <div className="hx-card-empty">Data unavailable</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Stage distribution bars */}
      <div className="section">
        <div className="section-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          Stage Distribution
          <PhaseInfoButton />
        </div>
        <div className="stage-bars">
          {Object.entries(p.stage_percentages).map(([stage, pct]) => (
            <div className="stage-row" key={stage}>
              <span className="stage-label">{stage}</span>
              <div className="stage-bar-bg">
                <div
                  className="stage-bar-fill"
                  style={{ width: `${pct}%`, background: stageColor(stage) }}
                />
              </div>
              <span className="stage-pct">{pct.toFixed(0)}%</span>
              <span className="stage-counts">{p.stage_distribution[stage] ?? 0}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Operation types */}
      <div className="section">
        <div className="section-title">Operation Types</div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(p.operation_types)
            .sort(([, a], [, b]) => b - a)
            .map(([op, count]) => (
              <span key={op} className="summary-chip">{op}: {count}</span>
            ))}
        </div>
      </div>

      {/* Stage Timeline */}
      <div className="section">
        <div className="section-title">
          Stage Timeline
          {p.human_input_positions && p.human_input_positions.length > 0 && (
            <span style={{ fontSize: 10, color: 'var(--orange)', marginLeft: 8, fontWeight: 400 }}>
              ▼ = human intervention
            </span>
          )}
        </div>
        {p.stage_sequence.length > 0 ? (
          <>
            <div className="timeline-scroll">
              <div className="timeline" style={{ position: 'relative' }}>
              {p.stage_sequence.map((stage, i) => (
                <div
                  key={i}
                  className="timeline-block"
                  style={{ background: stageColor(stage), position: 'relative' }}
                  title={`Step ${i + 1}: ${stage}${p.human_input_positions?.includes(i) ? ' (human input)' : ''}`}
                >
                  {p.human_input_positions?.includes(i) && (
                    <div className="human-marker" title={`Human input at step ${i + 1}`}>▼</div>
                  )}
                </div>
              ))}
              </div>
            </div>
            {p.fingerprint && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4, wordBreak: 'break-all' }}>
                {p.fingerprint}
              </div>
            )}
          </>
        ) : (
          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>No stage data available</div>
        )}
      </div>

      {/* Tool distribution */}
      <div className="section">
        <div className="section-title">Tools Used ({Object.keys(p.tool_distribution).length} unique)</div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', maxHeight: 140, overflowY: 'auto' }}>
          {Object.entries(p.tool_distribution)
            .sort(([, a], [, b]) => b - a)
            .map(([tool, count]) => (
              <div key={tool} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                <span>{tool}</span>
                <span>{count}</span>
              </div>
            ))}
        </div>
      </div>

      {/* Files touched */}
      <div className="section">
        <div className="section-title">Files Touched ({p.files_touched.length})</div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', maxHeight: 140, overflowY: 'auto' }}>
          {p.files_touched.map((f) => (
            <div key={f} style={{ padding: '1px 0' }}>{f}</div>
          ))}
        </div>
      </div>


    </div>
  );
}

function StatCard({ label, value, sub, className }: { label: string; value: string | number; sub?: string; className?: string }) {
  return (
    <div className={`stat-card${className ? ' ' + className : ''}`}>
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

/* ── Reusable info tooltip ── */
function InfoTip({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  return (
    <span
      style={{ position: 'relative', display: 'inline-flex', cursor: 'help', marginLeft: 2 }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <path d="M12 16v-4M12 8h.01" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      {show && (
        <div style={{
          position: 'absolute', bottom: 'calc(100% + 8px)', left: '50%', transform: 'translateX(-50%)',
          width: 240, padding: '8px 10px',
          background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: 6, boxShadow: '0 4px 14px rgba(0,0,0,0.12)',
          zIndex: 100, fontSize: 11, lineHeight: 1.5, color: 'var(--text)',
          fontWeight: 400, letterSpacing: 0, textTransform: 'none',
          pointerEvents: 'none',
        }}>
          {text}
        </div>
      )}
    </span>
  );
}

/* ── HX section info button ── */
function HxInfoButton() {
  const [show, setShow] = useState(false);
  return (
    <span
      style={{ position: 'relative', display: 'inline-flex', cursor: 'pointer' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <path d="M12 16v-4M12 8h.01" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      {show && (
        <div style={{
          position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)',
          marginTop: 6, width: 320, padding: '10px 14px',
          background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: 8, boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
          zIndex: 100, fontSize: 12, lineHeight: 1.5, color: 'var(--text)',
        }}>
          <div style={{ fontWeight: 600, marginBottom: 6, fontSize: 13 }}>Human Experience Indicators</div>
          <p style={{ margin: '0 0 6px' }}>
            These metrics capture the <strong>user-facing quality</strong> of an agent session — not just whether the task succeeded, but how the experience <em>felt</em>.
          </p>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div><strong>HX Score</strong> — Composite 0–100 rating combining autonomy, friction, responsiveness, and stability.</div>
            <div><strong>Time Breakdown</strong> — Where session wall time was spent: agent work, LLM inference, or waiting for human approval.</div>
            <div><strong>Response Latency</strong> — Per-step LLM response time. Reveals degradation or spikes.</div>
            <div><strong>Token Pressure</strong> — Cumulative prompt tokens showing context window growth and compaction resets.</div>
          </div>
          <p style={{ margin: '6px 0 0', fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic' }}>
            Only available for ATIF trajectories (real user sessions).
          </p>
        </div>
      )}
    </span>
  );
}

/* ── HX Score Card with hover breakdown ── */
function HxScoreCard({ score, breakdown }: { score: number; breakdown: Record<string, number> | null }) {
  const [showTip, setShowTip] = useState(false);
  const tier = score >= 70 ? 'good' : score >= 40 ? 'mid' : 'poor';
  const tierColor = tier === 'good' ? 'var(--green)' : tier === 'mid' ? 'var(--yellow)' : 'var(--red)';
  const tierLabel = tier === 'good' ? 'Good experience' : tier === 'mid' ? 'Some friction' : 'High friction';

  return (
    <div
      className={`stat-card hx-score-card hx-${tier}`}
      style={{ position: 'relative' }}
      onMouseEnter={() => setShowTip(true)}
      onMouseLeave={() => setShowTip(false)}
    >
      <div className="label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        HX Score
        <InfoTip text="Composite Human Experience score (0–100). Weights: Autonomy 30%, Low Friction 30%, Responsiveness 25%, Context Stability 15%. Higher = better user experience." />
      </div>
      <div className="value" style={{ color: tierColor }}>{score.toFixed(0)}</div>
      <div className="sub">{tierLabel}</div>
      {showTip && breakdown && (
        <div style={{
          position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)',
          marginTop: 6, width: 200, padding: '8px 10px',
          background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: 6, boxShadow: '0 4px 14px rgba(0,0,0,0.12)',
          zIndex: 100, fontSize: 11, lineHeight: 1.6,
        }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Score Breakdown</div>
          {Object.entries(breakdown).map(([k, v]) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>{HX_LABELS[k] ?? k}</span>
              <span style={{ fontWeight: 500 }}>{v.toFixed(0)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Time Decomposition Bar (compact) ── */
function TimeDecompositionBar({ data }: { data: Record<string, number> }) {
  const total = data.agent_work_ms + data.llm_thinking_ms + data.human_wait_ms;
  if (total <= 0) return null;

  const segments = [
    { key: 'agent_work_ms', label: 'Agent Work', color: 'var(--green)', ms: data.agent_work_ms },
    { key: 'llm_thinking_ms', label: 'LLM Thinking', color: 'var(--accent)', ms: data.llm_thinking_ms },
    { key: 'human_wait_ms', label: 'Human Wait', color: 'var(--orange)', ms: data.human_wait_ms },
  ].filter(s => s.ms > 0);

  return (
    <div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 6 }}>
        {fmtMs(total)} total
      </div>
      <div style={{
        display: 'flex', height: 18, borderRadius: 3, overflow: 'hidden',
        border: '1px solid var(--border)',
      }}>
        {segments.map(s => (
          <div
            key={s.key}
            title={`${s.label}: ${fmtMs(s.ms)} (${((s.ms / total) * 100).toFixed(0)}%)`}
            style={{
              width: `${(s.ms / total) * 100}%`,
              background: s.color,
              minWidth: s.ms > 0 ? 2 : 0,
            }}
          />
        ))}
      </div>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 6, fontSize: 10, color: 'var(--text-muted)' }}>
        {segments.map(s => (
          <span key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
            <span style={{ display: 'inline-block', width: 7, height: 7, borderRadius: 2, background: s.color, flexShrink: 0 }} />
            {s.label} {fmtMs(s.ms)} ({((s.ms / total) * 100).toFixed(0)}%)
          </span>
        ))}
      </div>
    </div>
  );
}

/* ── Sparkline (compact, proper aspect ratio) ── */
function Sparkline({ values, color, unit }: {
  values: number[];
  color: string;
  unit: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 260, H = 80, PAD = 4;
  const max = Math.max(...values, 1);
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  const stepW = (W - PAD * 2) / Math.max(values.length - 1, 1);

  const toX = (i: number) => PAD + i * stepW;
  const toY = (v: number) => H - PAD - ((v / max) * (H - PAD * 2));

  const points = values.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');
  const avgY = toY(avg);
  const hoverVal = hover !== null ? values[hover] : null;

  return (
    <div style={{ position: 'relative' }}>
      <svg
        width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
        style={{ display: 'block', background: '#f8fafc', borderRadius: 4, border: '1px solid var(--border)', cursor: 'crosshair' }}
        onMouseMove={e => {
          const rect = (e.target as SVGElement).closest('svg')!.getBoundingClientRect();
          const xRatio = (e.clientX - rect.left) / rect.width;
          const idx = Math.min(Math.max(Math.round(xRatio * (values.length - 1)), 0), values.length - 1);
          setHover(idx);
        }}
        onMouseLeave={() => setHover(null)}
      >
        {/* Fill area */}
        <polygon
          fill={color} opacity="0.06"
          points={`${toX(0)},${H - PAD} ${points} ${toX(values.length - 1)},${H - PAD}`}
        />
        {/* Average line */}
        <line x1={PAD} y1={avgY} x2={W - PAD} y2={avgY}
          stroke={color} strokeWidth="0.5" strokeDasharray="3,3" opacity="0.4" />
        {/* Data line */}
        <polyline fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" points={points} />
        {/* Hover dot + vertical guide */}
        {hover !== null && (
          <>
            <line x1={toX(hover)} y1={PAD} x2={toX(hover)} y2={H - PAD}
              stroke={color} strokeWidth="0.5" opacity="0.3" />
            <circle cx={toX(hover)} cy={toY(values[hover])} r="3" fill={color} />
          </>
        )}
      </svg>
      {/* Labels row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3, fontSize: 10, color: 'var(--text-muted)' }}>
        <span style={{ color }}>
          avg: {unit ? `${Math.round(avg).toLocaleString()}${unit}` : Math.round(avg).toLocaleString()}
        </span>
        {hover !== null && hoverVal !== null && (
          <span>
            Step {hover + 1}: <strong style={{ color }}>{unit ? `${Math.round(hoverVal).toLocaleString()}${unit}` : Math.round(hoverVal).toLocaleString()}</strong>
          </span>
        )}
        {hover === null && (
          <span>{values.length} steps</span>
        )}
      </div>
    </div>
  );
}
