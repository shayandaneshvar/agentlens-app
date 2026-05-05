import { useRef, useState, type DragEvent } from 'react';
import * as api from '../api/client';
import type {
  ProfileResponse, AssessWithGTResponse, QualityReport, FailureReason,
  DivergenceSegment, StageComparison, InefficiencyReport, QualitySignal,
  LLMAssessResponse, LLMSuggestionsResponse, ToolInefficiency,
} from '../types';
import { ComparisonView } from './ComparisonView';
import { GTGraphView } from './GTGraphView';

const STAGE_COLORS: Record<string, string> = {
  exploration: 'var(--stage-exploration)',
  implementation: 'var(--stage-implementation)',
  verification: 'var(--stage-verification)',
  orchestration: 'var(--stage-orchestration)',
};

function stageColor(stage: string) {
  return STAGE_COLORS[stage.toLowerCase()] ?? 'var(--stage-unknown)';
}

interface Props {
  traceId: string;
  profile: ProfileResponse;
  passed: boolean | null;
  onBack: () => void;
}

export function Tier2Assessment({ traceId, profile, passed, onBack }: Props) {
  const [gtFiles, setGtFiles] = useState<File[]>([]);
  const [result, setResult] = useState<AssessWithGTResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const importRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [showGtView, setShowGtView] = useState(false);
  const [gtData, setGtData] = useState<Record<string, unknown> | null>(null);
  const [llmResult, setLlmResult] = useState<LLMAssessResponse | null>(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmError, setLlmError] = useState<string | null>(null);
  const [llmExpanded, setLlmExpanded] = useState(true);
  const [suggestionsResult, setSuggestionsResult] = useState<LLMSuggestionsResponse | null>(null);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [suggestionsError, setSuggestionsError] = useState<string | null>(null);

  const addFiles = (files: FileList | File[]) => {
    const arr = Array.from(files).filter((f) => f.name.endsWith('.zip') || f.name.endsWith('.json'));
    setGtFiles((prev) => [...prev, ...arr]);
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(e.target.files);
    e.target.value = '';
  };

  const removeFile = (idx: number) => {
    setGtFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleAssess = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.assessWithGT(traceId, gtFiles);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleImportGT = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.assessWithImportedGT(traceId, file);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleExportGT = async () => {
    try {
      const blob = await api.exportGT();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'merged_gt.json';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    }
  };

  const handleViewGT = async () => {
    if (gtData) {
      setShowGtView(!showGtView);
      return;
    }
    try {
      const blob = await api.exportGT();
      const text = await blob.text();
      const data = JSON.parse(text);
      setGtData(data);
      setShowGtView(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load GT data');
    }
  };

  // Phase 1: Upload GT files or import pre-built GT
  if (!result) {
    return (
      <div>
        <button className="gt-btn secondary" style={{ marginBottom: 16 }} onClick={onBack}>
          Back to Profile
        </button>

        <h2 style={{ fontSize: 16, marginBottom: 8 }}>Quality Assessment</h2>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 16 }}>
          Upload at least 2 <strong>passing</strong> trajectory zip files for the same task,
          or import a previously exported merged GT JSON.
        </p>

        {/* Option A: Upload passing trajectories to merge */}
        <div
          className={`upload-zone${dragOver ? ' drag-over' : ''}`}
          style={{ marginBottom: 12 }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          Drop passing .zip or .json files here or click to upload
          <input
            ref={inputRef}
            type="file"
            accept=".zip,.json"
            multiple
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>

        {gtFiles.length > 0 && (
          <div style={{ marginBottom: 12 }}>
            {gtFiles.map((f, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '4px 0', fontSize: 13,
              }}>
                <span className="trace-dot pass" />
                <span style={{ flex: 1 }}>{f.name}</span>
                <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                  {(f.size / 1024).toFixed(0)} KB
                </span>
                <button
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#999', fontSize: 14 }}
                  onClick={() => removeFile(i)}
                >x</button>
              </div>
            ))}
          </div>
        )}

        {error && (
          <div style={{ color: 'var(--red)', fontSize: 13, marginBottom: 8 }}>{error}</div>
        )}

        <button
          className="gt-btn"
          disabled={gtFiles.length < 2 || loading}
          onClick={handleAssess}
        >
          {loading ? 'Processing...' : `Build GT & Assess (${gtFiles.length} files)`}
        </button>

        {/* Option B: Import pre-built GT */}
        <div style={{
          marginTop: 20, paddingTop: 16,
          borderTop: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', gap: 12,
        }}>
          <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>Or use a pre-built GT:</span>
          <button
            className="gt-btn secondary"
            style={{ fontSize: 12, padding: '5px 12px' }}
            onClick={() => importRef.current?.click()}
            disabled={loading}
          >
            Import merged GT JSON
          </button>
          <input
            ref={importRef}
            type="file"
            accept=".json"
            onChange={handleImportGT}
            style={{ display: 'none' }}
          />
        </div>
      </div>
    );
  }

  // Phase 2: Show results
  const q: QualityReport = result.quality_report as unknown as QualityReport;
  const p = profile;
  const m = result.match_metrics as Record<string, number>;

  // ── Section fragments (rendered in different order based on outcome) ──

  const qualitySignalsSection = q.quality_signals && q.quality_signals.length > 0 ? (
    <div className="section">
      <div className="section-title">Quality Signals</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {q.quality_signals.map((sig: QualitySignal, i: number) => (
          <div key={i} className={`signal-card signal-${sig.severity}`}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span className="signal-icon">
                {sig.severity === 'critical' ? '🔴' : sig.severity === 'warning' ? '🟡' : 'ℹ️'}
              </span>
              <strong style={{ fontSize: 13 }}>{sig.signal_type.replace(/_/g, ' ')}</strong>
              <span className={`sev-pill ${sig.severity}`}>{sig.severity}</span>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>
              {sig.description}
            </div>
            {sig.evidence.length > 0 && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                {sig.evidence.map((e, j) => (
                  <div key={j}>• {e}</div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  ) : null;

  const divergenceSection = (
    <>
      {q.divergence_point && (!q.divergence_points || q.divergence_points.length === 0) && (
        <div className="divergence-callout">
          <strong>Divergence at step {q.divergence_point.step}:</strong>{' '}
          {q.divergence_point.description}
          {q.divergence_point.expected_next && (
            <div style={{ marginTop: 4, fontSize: 12, color: 'var(--text-muted)' }}>
              Expected next: {q.divergence_point.expected_next}
            </div>
          )}
        </div>
      )}
      {q.divergence_points && q.divergence_points.length > 0 && (
        <div className="section">
          <div className="section-title">Divergence Points ({q.divergence_points.length})</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {q.divergence_points.map((seg: DivergenceSegment, i: number) => (
              <DivergenceSegmentCard key={i} segment={seg} index={i} />
            ))}
          </div>
        </div>
      )}
    </>
  );

  const failureReasonsSection = q.failure_reasons.length > 0 ? (
    <div className="section">
      <div className="section-title">Why it&apos;s failing</div>
      <div className="failure-reasons">
        {q.failure_reasons.map((fr: FailureReason, i: number) => (
          <div key={i} className={`failure-card ${fr.severity}`}>
            <div className="sev">{fr.severity}</div>
            <div className="detail">
              <strong>{fr.reason}</strong>: {fr.detail}
            </div>
          </div>
        ))}
      </div>
    </div>
  ) : null;

  const stageComparisonSection = q.stage_comparison && Object.keys(q.stage_comparison).length > 0 ? (
    <div className="section">
      <div className="section-title">
        Stage-Level Comparison
        {q.stage_order_match === false && (
          <span className="order-mismatch-badge">⚠ Stage order mismatch</span>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {(['exploration', 'implementation', 'verification', 'orchestration'] as const).map((stage) => {
          const comp = q.stage_comparison[stage];
          if (!comp || (comp.expected_steps.length === 0 && comp.extra_steps.length === 0)) return null;
          return <StageComparisonCard key={stage} stage={stage} comparison={comp} />;
        })}
      </div>
    </div>
  ) : null;

  const inefficiencySection = q.inefficiencies && q.inefficiencies.total_wasted_steps > 0 ? (
    <div className="section">
      <div className="section-title">Inefficiencies Detected</div>
      {/* Inefficiency Severity bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, padding: '8px 12px', background: '#f9fafb', borderRadius: 8, border: '1px solid #e5e7eb' }}>
        <span style={{ fontWeight: 600, fontSize: 13, minWidth: 150 }}>Inefficiency Severity</span>
        <div style={{ flex: 1, height: 10, background: '#e5e7eb', borderRadius: 5, overflow: 'hidden' }}>
          <div style={{
            width: `${Math.min((q.inefficiencies.severity_score ?? 0) * 100, 100)}%`,
            height: '100%',
            borderRadius: 5,
            background: (q.inefficiencies.severity_score ?? 0) > 0.4 ? '#ef4444'
              : (q.inefficiencies.severity_score ?? 0) > 0.2 ? '#f59e0b' : '#22c55e',
          }} />
        </div>
        <span style={{ fontWeight: 700, fontSize: 14, minWidth: 50, textAlign: 'right',
          color: (q.inefficiencies.severity_score ?? 0) > 0.4 ? '#ef4444'
            : (q.inefficiencies.severity_score ?? 0) > 0.2 ? '#f59e0b' : '#22c55e',
        }}>
          {((q.inefficiencies.severity_score ?? 0) * 100).toFixed(0)}%
        </span>
      </div>
      <div className="stat-cards" style={{ gridTemplateColumns: 'repeat(5, 1fr)', marginBottom: 12 }}>
        <div className="stat-card">
          <div className="label">Retry Loops</div>
          <div className="value">{q.inefficiencies.retry_loop_count}</div>
        </div>
        <div className="stat-card">
          <div className="label">Cyclic Patterns</div>
          <div className="value">{q.inefficiencies.cyclic_pattern_count}</div>
        </div>
        <div className="stat-card">
          <div className="label">Backtracks</div>
          <div className="value">{q.inefficiencies.backtrack_count}</div>
        </div>
        <div className="stat-card">
          <div className="label">Redundant Steps</div>
          <div className="value">{q.inefficiencies.redundant_step_count}</div>
        </div>
        <div className="stat-card">
          <div className="label">Unnecessary Exploration</div>
          <div className="value">{q.inefficiencies.unnecessary_exploration_count}</div>
        </div>
      </div>
      <div className="ineff-summary">
        Total wasted steps: <strong>{Math.min(q.inefficiencies.total_wasted_steps, p.state_count)}</strong>
        {' '}({Math.min(((q.inefficiencies.total_wasted_steps / Math.max(p.state_count, 1)) * 100), 100).toFixed(0)}% of trajectory)
        {(q.inefficiencies.wasted_input_tokens > 0 || q.inefficiencies.wasted_output_tokens > 0) && (
          <span style={{ marginLeft: 16, color: '#6b7280' }}>
            Wasted tokens: <strong>{(q.inefficiencies.wasted_input_tokens + q.inefficiencies.wasted_output_tokens).toLocaleString()}</strong>
            {' '}of {(q.inefficiencies.total_input_tokens + q.inefficiencies.total_output_tokens).toLocaleString()} total
            {' '}({((q.inefficiencies.wasted_input_tokens + q.inefficiencies.wasted_output_tokens) / Math.max(q.inefficiencies.total_input_tokens + q.inefficiencies.total_output_tokens, 1) * 100).toFixed(0)}%)
          </span>
        )}
      </div>
      {/* Per-tool breakdown table */}
      {q.inefficiencies.per_tool_breakdown && q.inefficiencies.per_tool_breakdown.length > 0 && (
        <PerToolBreakdown tools={q.inefficiencies.per_tool_breakdown} />
      )}
      <InefficiencyDetails report={q.inefficiencies} />
    </div>
  ) : null;

  const strengthsSection = q.strengths.length > 0 ? (
    <div className="section">
      <div className="section-title">Strengths</div>
      {q.strengths.map((s: string, i: number) => (
        <div key={i} className="strength-item">+ {s}</div>
      ))}
    </div>
  ) : null;

  const processCoverageSection = (
    <div className="section">
      <div className="section-title">Process &amp; File Coverage</div>
      <div className="stat-cards" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div className="stat-card">
          <div className="label">Process Coverage</div>
          <div className="value">{(result.process_coverage * 100).toFixed(0)}%</div>
          {result.missing_tools.length > 0 && (
            <div className="sub">Missing: {result.missing_tools.join(', ')}</div>
          )}
        </div>
        <div className="stat-card">
          <div className="label">File Coverage</div>
          <div className="value">{(result.file_coverage * 100).toFixed(0)}%</div>
          {result.missing_files.length > 0 && (
            <div className="sub">Missing: {result.missing_files.join(', ')}</div>
          )}
        </div>
      </div>
    </div>
  );

  const stageCoverageSection = q.stage_coverage && Object.keys(q.stage_coverage).length > 0 ? (
    <div className="section">
      <div className="section-title">Per-Stage Coverage vs Ground Truth</div>
      <div className="stage-bars">
        {Object.entries(q.stage_coverage).map(([stage, detail]) => (
          <div className="stage-row" key={stage}>
            <span className="stage-label">{stage}</span>
            <div className="stage-bar-bg">
              <div
                className="stage-bar-fill"
                style={{ width: `${detail.percent}%`, background: stageColor(stage) }}
              />
            </div>
            <span className="stage-pct">{detail.percent.toFixed(0)}%</span>
            <span className="stage-counts">{detail.matched}/{detail.total}</span>
          </div>
        ))}
      </div>
    </div>
  ) : null;

  // Build backtrack/retry/cycle position sets from GT-aware inefficiency data
  const btPositions = new Set<number>(
    q.inefficiencies?.backtracks?.map((b) => b.step) ?? []
  );
  const retryPositions = new Set<number>();
  if (q.inefficiencies?.retry_loops) {
    for (const r of q.inefficiencies.retry_loops) {
      for (let s = r.start_step; s <= r.end_step; s++) retryPositions.add(s);
    }
  }
  const cyclePositions = new Set<number>();
  if (q.inefficiencies?.cyclic_patterns) {
    for (const c of q.inefficiencies.cyclic_patterns) {
      for (let s = c.start_step; s <= c.end_step; s++) cyclePositions.add(s);
    }
  }

  const fingerprintSection = (
    <div className="section">
      <div className="section-title">Workflow Fingerprint</div>
      {p.fingerprint ? (
        <>
          <div className="fingerprint">
            {p.fingerprint.split('→').map((ch, i) => {
              const label =
                ch === 'E' ? 'exploration' :
                ch === 'I' ? 'implementation' :
                ch === 'V' ? 'verification' :
                ch === 'O' ? 'orchestration' : 'unknown';
              const step = i + 1;  // fingerprint index is 0-based, step numbers are 1-based
              const isBt = btPositions.has(step);
              const isRetry = retryPositions.has(step);
              const isCycle = cyclePositions.has(step);
              return (
                <div
                  key={i}
                  className={`fp-block ${label}${isBt ? ' marker-backtrack' : ''}${isRetry ? ' marker-retry' : ''}${isCycle && !isBt && !isRetry ? ' marker-cycle' : ''}`}
                  title={`Step ${step}: ${label}${isBt ? ' (backtrack)' : ''}${isRetry ? ' (retry loop)' : ''}${isCycle ? ' (cyclic pattern)' : ''}`}
                >
                  {ch}
                  {isBt && <span className="marker marker-bt">&#8635;</span>}
                  {isRetry && !isBt && <span className="marker marker-br">&#8226;</span>}
                  {isCycle && !isBt && !isRetry && <span className="marker marker-cy">&#8634;</span>}
                </div>
              );
            })}
          </div>
          {(btPositions.size > 0 || retryPositions.size > 0 || cyclePositions.size > 0) && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
              {btPositions.size > 0 && (
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ color: '#dc2626', fontSize: 13 }}>&#8635;</span>
                  <span>Backtrack ({btPositions.size})</span>
                </span>
              )}
              {retryPositions.size > 0 && (
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ color: '#f59e0b', fontSize: 16, lineHeight: 1 }}>&#8226;</span>
                  <span>Retry loop ({retryPositions.size} steps)</span>
                </span>
              )}
              {cyclePositions.size > 0 && (
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ color: '#8b5cf6', fontSize: 13 }}>&#8634;</span>
                  <span>Cyclic pattern ({cyclePositions.size} steps)</span>
                </span>
              )}
            </div>
          )}
        </>
      ) : (
        <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>No fingerprint available</div>
      )}
    </div>
  );

  // ── LLM Assessment section ──

  const handleLlmAssess = async () => {
    setLlmLoading(true);
    setLlmError(null);
    setSuggestionsLoading(true);
    setSuggestionsError(null);

    // Fire both calls in parallel
    const assessPromise = api.llmAssess(traceId)
      .then((res) => { setLlmResult(res); })
      .catch((e: unknown) => { setLlmError(e instanceof Error ? e.message : 'LLM assessment failed'); })
      .finally(() => { setLlmLoading(false); });

    const suggestPromise = api.llmSuggestions(traceId)
      .then((res) => { setSuggestionsResult(res); })
      .catch((e: unknown) => { setSuggestionsError(e instanceof Error ? e.message : 'LLM suggestions failed'); })
      .finally(() => { setSuggestionsLoading(false); });

    await Promise.allSettled([assessPromise, suggestPromise]);
  };

  const RATING_STYLE: Record<string, { bg: string; color: string; label: string }> = {
    strong:   { bg: '#dcfce7', color: '#166534', label: 'Strong' },
    adequate: { bg: '#fef9c3', color: '#854d0e', label: 'Adequate' },
    weak:     { bg: '#fee2e2', color: '#991b1b', label: 'Weak' },
  };

  const DIMENSION_LABELS: Record<string, string> = {
    strategy: '🎯 Strategy',
    efficiency: '⚡ Efficiency',
    verification: '✅ Verification',
    error_recovery: '🔄 Error Recovery',
    completeness: '📋 Completeness',
  };

  const llmSection = (
    <div className="section">
      <div className="section-title">
        LLM Assessment & Suggestions
        <InfoTip text="Uses an LLM judge to evaluate the trajectory against both PTA matching results and the full trajectory structure. Provides nuanced quality analysis across 5 dimensions and actionable improvement suggestions." />
      </div>

      {!llmResult && !llmLoading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button className="gt-btn" onClick={handleLlmAssess} disabled={llmLoading || suggestionsLoading}>
            Run LLM Assessment & Suggestions
          </button>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Requires LLM configured via AGENTLENS_LLM env var</span>
        </div>
      )}

      {llmLoading && (
        <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: 28, marginBottom: 8, display: 'inline-block', animation: 'llm-spin 1.2s linear infinite' }}>⚙️</div>
          <style>{`@keyframes llm-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
          <div style={{ fontSize: 13, fontWeight: 500 }}>Running LLM assessment & generating suggestions...</div>
          <div style={{ fontSize: 11, marginTop: 4 }}>Analyzing trajectory structure, PTA matching results, and inefficiencies</div>
        </div>
      )}

      {llmError && (
        <div style={{ color: 'var(--red)', fontSize: 13, padding: '8px 12px', background: '#fef2f2', borderRadius: 6 }}>
          {llmError}
          <button onClick={() => setLlmError(null)} style={{ marginLeft: 8, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--red)', fontWeight: 600 }}>×</button>
        </div>
      )}

      {llmResult && (() => {
        const a = llmResult.assessment;
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Controls */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>via {llmResult.model_used}</span>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
                <button className="gt-btn secondary" style={{ fontSize: 11, padding: '3px 10px' }} onClick={() => setLlmExpanded(v => !v)}>
                  {llmExpanded ? 'Hide' : 'Show'}
                </button>
                <button className="gt-btn secondary" style={{ fontSize: 11, padding: '3px 10px' }} onClick={handleLlmAssess} disabled={llmLoading || suggestionsLoading}>
                  Re-run
                </button>
              </div>
            </div>

            {/* Collapsible body */}
            {!llmExpanded ? null : <>
            {/* Summary */}
            <div style={{ fontSize: 13, lineHeight: 1.6, color: 'var(--text)', padding: '10px 14px', background: 'var(--bg-card)', borderRadius: 8, border: '1px solid var(--border)' }}>
              {a.summary}
            </div>

            {/* Dimension cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
              {Object.entries(a.dimensions).map(([key, dim]) => {
                const rs = RATING_STYLE[dim.rating] ?? RATING_STYLE.adequate;
                return (
                  <div key={key} style={{ padding: '10px 14px', borderRadius: 8, background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                      <span style={{ fontSize: 13, fontWeight: 600 }}>{DIMENSION_LABELS[key] ?? key}</span>
                      <span style={{
                        padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                        background: rs.bg, color: rs.color,
                      }}>
                        {rs.label}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                      {dim.reasoning}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Key findings */}
            {a.key_findings.length > 0 && (
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: 'var(--text)' }}>Key Findings</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {a.key_findings.map((f, i) => (
                    <div key={i} style={{
                      fontSize: 12, padding: '8px 12px', borderRadius: 6,
                      background: f.type === 'strength' ? '#f0fdf4' : '#fef2f2',
                      borderLeft: `3px solid ${f.type === 'strength' ? '#22c55e' : '#ef4444'}`,
                    }}>
                      <div style={{ fontWeight: 600, marginBottom: 2 }}>
                        {f.type === 'strength' ? '✓' : '✗'} {f.observation}
                      </div>
                      <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>{f.evidence}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendation */}
            <div style={{ fontSize: 12, lineHeight: 1.5, padding: '10px 14px', borderRadius: 8, background: '#eff6ff', border: '1px solid #bfdbfe' }}>
              <strong>Recommendation:</strong> {a.recommendation}
            </div>

            {/* Actionable Suggestions (from parallel LLM call) */}
            {suggestionsLoading && (
              <div style={{ padding: 16, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                Generating actionable suggestions...
              </div>
            )}
            {suggestionsError && (
              <div style={{ color: 'var(--red)', fontSize: 12, padding: '8px 12px', background: '#fef2f2', borderRadius: 6 }}>
                Suggestions: {suggestionsError}
              </div>
            )}
            {suggestionsResult && suggestionsResult.suggestions.length > 0 && (
              <SuggestionsPanel result={suggestionsResult} />
            )}
            {suggestionsResult && suggestionsResult.suggestions.length === 0 && (
              <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: '10px 14px', background: '#f0fdf4', borderRadius: 8, border: '1px solid #bbf7d0' }}>
                ✓ {suggestionsResult.improvement_summary}
              </div>
            )}
            </> /* end collapsible */}
          </div>
        );
      })()}
    </div>
  );

  // ── Determine outcome and pick section order ──

  const isPassing = passed === true;
  const isFailing = passed === false;

  // Derive bottleneck stage name from per-stage coverage
  const bottleneckStage = (() => {
    if (!q.stage_coverage || Object.keys(q.stage_coverage).length === 0) return '';
    let minStage = '';
    let minPct = Infinity;
    for (const [stage, detail] of Object.entries(q.stage_coverage)) {
      if (detail.percent < minPct) { minPct = detail.percent; minStage = stage; }
    }
    return minStage;
  })();

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <button className="gt-btn secondary" onClick={onBack}>
          Back to Profile
        </button>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button className="gt-btn secondary" style={{ fontSize: 11, padding: '4px 10px' }} onClick={handleViewGT}>
            {showGtView ? 'Hide GT' : 'View GT'}
          </button>
          <button className="gt-btn secondary" style={{ fontSize: 11, padding: '4px 10px' }} onClick={handleExportGT}>
            Export GT
          </button>
        </div>
      </div>

      {error && (
        <div style={{ color: 'var(--red)', fontSize: 13, marginBottom: 8, padding: '8px 12px', background: '#fef2f2', borderRadius: 6 }}>
          {error}
          <button onClick={() => setError(null)} style={{ marginLeft: 8, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--red)', fontWeight: 600 }}>×</button>
        </div>
      )}

      {/* GT viewer */}
      {showGtView && gtData && (
        <div className="section" style={{ marginBottom: 16 }}>
          <div className="section-title">Merged Ground Truth</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>
            {(gtData as { statistics?: { num_states?: number; num_transitions?: number } }).statistics?.num_states ?? '?'} states, {(gtData as { statistics?: { num_states?: number; num_transitions?: number } }).statistics?.num_transitions ?? '?'} transitions
          </div>
          <GTGraphView data={gtData as Parameters<typeof GTGraphView>[0]['data']} />
        </div>
      )}

      {/* Summary chips */}
      <div className="summary-row">
        {p.agent && <span className="summary-chip"><strong>Agent:</strong> {p.agent}</span>}
        {p.model && <span className="summary-chip"><strong>Model:</strong> {p.model}</span>}
        <span className="summary-chip">{p.state_count} states</span>
        <span className="summary-chip">{p.file_count} files</span>
        <span className="summary-chip">{p.tool_count} tools</span>
        <span className="summary-chip">GT: {result.gt_source_count} traces, {result.gt_state_count} states</span>
        {p.human_input_count != null && <span className="summary-chip"><strong>Human Inputs:</strong> {p.human_input_count}</span>}
        {p.subagent_count != null && <span className="summary-chip"><strong>Subagents:</strong> {p.subagent_count}</span>}
        {p.active_time_ms != null && <span className="summary-chip"><strong>Active Time:</strong> {(p.active_time_ms / 1000).toFixed(0)}s</span>}
        {p.compaction_count != null && <span className="summary-chip"><strong>Compactions:</strong> {p.compaction_count}</span>}
      </div>

      {/* Outcome-aware analysis banner */}
      {passed !== null && (
        <div className={`outcome-banner ${passed ? 'outcome-pass' : 'outcome-fail'}`}>
          <span className="outcome-icon">{passed ? '✓' : '✗'}</span>
          <div>
            <div className="outcome-title">
              {passed ? 'Passed — Efficiency Analysis' : 'Failed — Failure Diagnosis'}
            </div>
            <div className="outcome-subtitle">
              {passed
                ? 'This trajectory succeeded. Analysis focuses on how efficiently it achieved the goal.'
                : 'This trajectory failed. Analysis focuses on where and why it went wrong.'}
            </div>
          </div>
        </div>
      )}

      {/* Key metrics */}
      <div className="stat-cards">
        <div className={`stat-card quality-score-card ${passed === true ? 'score-pass' : passed === false ? 'score-fail' : ''}`}>
          <div className="label">Quality Score<InfoTip text="Composite 0–100 score: 25% coverage + 25% coherence + 18% stage completeness + 12% workflow similarity + 10% F1 + 10% outcome (pass=100, fail=0)." /></div>
          <div className="value">{q.quality_score.toFixed(0)}</div>
        </div>
        <MetricCard label="Coverage" value={m.coverage_percent} pct
          tooltip="% of ground-truth states the candidate matched (order-independent). Higher = more GT work reproduced." />
        <MetricCard label="Coherence" value={(m.coherence_score ?? 0) * 100} pct
          tooltip="How clean the workflow is. Rewards forward progress (pivots, confirms) and penalizes backtracks and blind retries. ≥70% = clean, <40% = heavy thrashing." />
        <MetricCard label="Stage Completeness" value={(m.stage_completeness ?? 0) * 100} pct
          tooltip="Fraction of distinct GT stages (exploration, implementation, verification, orchestration) with at least one matched state." />
        <MetricCard label="Workflow Similarity" value={(m.workflow_similarity ?? 0) * 100} pct
          tooltip="How close the candidate's stage ordering is to the GT, measured by longest common subsequence. 100% = identical sequence." />
        <div className="stat-card">
          <div className="label">Bottleneck<InfoTip text="The weakest per-stage coverage — the minimum across all stages. 0% means at least one entire stage was missed." /></div>
          <div className="value">{(m.bottleneck_coverage ?? 0).toFixed(0)}%</div>
          {bottleneckStage && <div className="sub">{bottleneckStage}</div>}
        </div>
      </div>

      {/* GT Comparison view */}
      {result.comparison && (
        <ComparisonView comparison={result.comparison} />
      )}

      {/* ── Sections in outcome-aware order ── */}
      {isPassing ? (
        <>
          {/* Pass: efficiency focus */}
          {qualitySignalsSection}
          {inefficiencySection}
          {stageComparisonSection}
          {processCoverageSection}
          {stageCoverageSection}
          {strengthsSection}
          {fingerprintSection}
          {llmSection}
        </>
      ) : isFailing ? (
        <>
          {/* Fail: diagnosis focus */}
          {qualitySignalsSection}
          {divergenceSection}
          {failureReasonsSection}
          {stageComparisonSection}
          {processCoverageSection}
          {stageCoverageSection}
          {inefficiencySection}
          {strengthsSection}
          {fingerprintSection}
          {llmSection}
        </>
      ) : (
        <>
          {/* Unknown outcome: balanced order */}
          {qualitySignalsSection}
          {processCoverageSection}
          {divergenceSection}
          {failureReasonsSection}
          {stageComparisonSection}
          {inefficiencySection}
          {strengthsSection}
          {stageCoverageSection}
          {fingerprintSection}
          {llmSection}
        </>
      )}
    </div>
  );
}


/* ── Sub-components ─────────────────────────────────────────────────── */

function DivergenceSegmentCard({ segment, index }: { segment: DivergenceSegment; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const span = segment.end_step - segment.start_step + 1;
  return (
    <div className="divergence-card" onClick={() => setExpanded(!expanded)} style={{ cursor: 'pointer' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span className="divergence-num">#{index + 1}</span>
        <span style={{ fontSize: 13, fontWeight: 600 }}>
          Steps {segment.start_step}–{segment.end_step}
        </span>
        <span className="stage-pill" style={{ background: stageColor(segment.stage_context) }}>
          {segment.stage_context || 'mixed'}
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {span} missed GT state{span > 1 ? 's' : ''}
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 11 }}>{expanded ? '▲' : '▼'}</span>
      </div>
      {expanded && (
        <div style={{ marginTop: 8, fontSize: 12 }}>
          <div style={{ marginBottom: 6 }}>
            <strong>Expected (GT):</strong>
            {segment.expected_states.map((s, i) => (
              <div key={i} className="step-detail">
                <span className="step-tool">{s.tool || '—'}</span>
                {s.file_path && <span className="step-file">{s.file_path}</span>}
                <span className="stage-pill sm" style={{ background: stageColor(s.intent_stage) }}>
                  {s.intent_stage}
                </span>
              </div>
            ))}
          </div>
          {segment.candidate_activity.length > 0 && (
            <div>
              <strong>Candidate did instead:</strong>
              {segment.candidate_activity.map((s, i) => (
                <div key={i} className="step-detail">
                  <span className="step-tool">{s.tool || '—'}</span>
                  {s.file_path && <span className="step-file">{s.file_path}</span>}
                  <span className="stage-pill sm" style={{ background: stageColor(s.intent_stage) }}>
                    {s.intent_stage}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StageComparisonCard({ stage, comparison }: { stage: string; comparison: StageComparison }) {
  const [expanded, setExpanded] = useState(false);
  const color = stageColor(stage);
  return (
    <div className="stage-comp-card" style={{ borderLeftColor: color }}>
      <div
        style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}
        onClick={() => setExpanded(!expanded)}
      >
        <span className="stage-pill" style={{ background: color }}>{stage}</span>
        <span className="comp-counts">
          <span className="comp-matched">{comparison.matched_steps.length} matched</span>
          {comparison.missing_steps.length > 0 && (
            <span className="comp-missing">{comparison.missing_steps.length} missing</span>
          )}
          {comparison.extra_steps.length > 0 && (
            <span className="comp-extra">{comparison.extra_steps.length} extra</span>
          )}
        </span>
        {!comparison.ordering_preserved && (
          <span className="order-warn" title="Steps out of order vs GT">⚠ reordered</span>
        )}
        <span className="effort-ratio" title="Candidate effort / GT effort">
          {comparison.effort_ratio.toFixed(1)}×
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 11 }}>{expanded ? '▲' : '▼'}</span>
      </div>
      {expanded && (
        <div style={{ marginTop: 8, fontSize: 12 }}>
          {comparison.matched_steps.length > 0 && (
            <StepList label="Matched" items={comparison.matched_steps} className="matched" />
          )}
          {comparison.missing_steps.length > 0 && (
            <StepList label="Missing from candidate" items={comparison.missing_steps} className="missing" />
          )}
          {comparison.extra_steps.length > 0 && (
            <StepList label="Extra in candidate" items={comparison.extra_steps} className="extra" />
          )}
        </div>
      )}
    </div>
  );
}

const PRIORITY_STYLE: Record<string, { bg: string; border: string; color: string; label: string }> = {
  high:   { bg: '#fef2f2', border: '#fca5a5', color: '#991b1b', label: 'High' },
  medium: { bg: '#fffbeb', border: '#fcd34d', color: '#92400e', label: 'Medium' },
  low:    { bg: '#f0fdf4', border: '#86efac', color: '#166534', label: 'Low' },
};

const CATEGORY_ICONS: Record<string, string> = {
  retry_prevention: '🔄',
  exploration_control: '🔍',
  verification: '✅',
  error_recovery: '🔧',
  tool_usage: '🛠️',
};

function SuggestionsPanel({ result }: { result: LLMSuggestionsResponse }) {
  const [expanded, setExpanded] = useState(true);
  return (
    <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 13, fontWeight: 700 }}>💡 Actionable Suggestions</span>
        {result.model_used && (
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>via {result.model_used}</span>
        )}
        <button
          className="gt-btn secondary"
          style={{ fontSize: 11, padding: '2px 8px', marginLeft: 'auto' }}
          onClick={() => setExpanded(v => !v)}
        >
          {expanded ? 'Hide' : 'Show'}
        </button>
      </div>

      {expanded && (
        <>
          {/* Improvement summary */}
          {result.improvement_summary && (
            <div style={{ fontSize: 12, lineHeight: 1.5, padding: '8px 14px', marginBottom: 10, borderRadius: 8, background: '#faf5ff', border: '1px solid #d8b4fe' }}>
              <strong>Top priority:</strong> {result.improvement_summary}
            </div>
          )}

          {/* Suggestion cards */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {result.suggestions.map((s, i) => {
              const ps = PRIORITY_STYLE[s.priority] ?? PRIORITY_STYLE.medium;
              const catIcon = CATEGORY_ICONS[s.category] ?? '📌';
              return (
                <div key={i} style={{
                  padding: '10px 14px', borderRadius: 8,
                  background: ps.bg, borderLeft: `4px solid ${ps.border}`,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                    <span style={{
                      padding: '1px 6px', borderRadius: 10, fontSize: 10, fontWeight: 700,
                      background: ps.border, color: ps.color,
                    }}>
                      {ps.label}
                    </span>
                    <span style={{ fontSize: 12, fontWeight: 600 }}>{catIcon} {s.title}</span>
                    {s.estimated_savings && (
                      <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--text-muted)', fontWeight: 500 }}>
                        ~{s.estimated_savings}
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: 12, color: '#444', lineHeight: 1.5, marginBottom: 4 }}>
                    <strong>Root cause:</strong> {s.root_cause}
                  </div>
                  <div style={{ fontSize: 12, color: '#222', lineHeight: 1.5 }}>
                    <strong>Fix:</strong> {s.suggestion}
                  </div>
                  {s.affected_steps && s.affected_steps.length > 0 && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                      Affects steps: {s.affected_steps.join(', ')}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function PerToolBreakdown({ tools }: { tools: ToolInefficiency[] }) {
  const [expanded, setExpanded] = useState(false);
  if (!tools.length) return null;
  return (
    <div style={{ marginTop: 8 }}>
      <button
        className="gt-btn secondary"
        style={{ fontSize: 11, padding: '4px 10px' }}
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? 'Hide per-tool breakdown' : 'Per-tool breakdown'}
      </button>
      {expanded && (
        <div style={{ overflowX: 'auto', marginTop: 8 }}>
          <table className="alens-table" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Tool</th>
                <th style={{ textAlign: 'center' }}>Retries</th>
                <th style={{ textAlign: 'center' }}>Backtracks</th>
                <th style={{ textAlign: 'center' }}>Cycles</th>
                <th style={{ textAlign: 'center' }}>Redundant</th>
                <th style={{ textAlign: 'center' }}>Unnecessary</th>
                <th style={{ textAlign: 'center', fontWeight: 700 }}>Total</th>
              </tr>
            </thead>
            <tbody>
              {tools.map((t) => (
                <tr key={t.tool}>
                  <td style={{ fontWeight: 500, fontFamily: 'var(--font-mono, monospace)', fontSize: 11 }}>{t.tool}</td>
                  <td style={{ textAlign: 'center', color: t.retries ? undefined : '#d1d5db' }}>{t.retries || '—'}</td>
                  <td style={{ textAlign: 'center', color: t.backtracks ? undefined : '#d1d5db' }}>{t.backtracks || '—'}</td>
                  <td style={{ textAlign: 'center', color: t.cycles ? undefined : '#d1d5db' }}>{t.cycles || '—'}</td>
                  <td style={{ textAlign: 'center', color: t.redundant ? undefined : '#d1d5db' }}>{t.redundant || '—'}</td>
                  <td style={{ textAlign: 'center', color: t.unnecessary ? undefined : '#d1d5db' }}>{t.unnecessary || '—'}</td>
                  <td style={{ textAlign: 'center', fontWeight: 700 }}>{t.total_wasted}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function StepList({ label, items, className }: {
  label: string;
  items: Array<{ tool: string; file_path: string; resulting_state?: string }>;
  className: string;
}) {
  return (
    <div className={`step-list ${className}`}>
      <div className="step-list-label">{label}</div>
      {items.map((s, i) => (
        <div key={i} className="step-detail">
          <span className="step-tool">{s.tool || '—'}</span>
          {s.file_path && <span className="step-file">{s.file_path}</span>}
        </div>
      ))}
    </div>
  );
}

function InefficiencyDetails({ report }: { report: InefficiencyReport }) {
  const [expanded, setExpanded] = useState(false);
  if (report.total_wasted_steps === 0) return null;
  return (
    <div>
      <button
        className="gt-btn secondary"
        style={{ fontSize: 11, padding: '4px 10px', marginTop: 6 }}
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? 'Hide details' : 'Show details'}
      </button>
      {expanded && (
        <div style={{ marginTop: 8, fontSize: 12 }}>
          {report.retry_loops.length > 0 && (
            <div className="ineff-group">
              <div className="ineff-group-title">🔄 Retry Loops</div>
              {report.retry_loops.map((r, i) => (
                <div key={i} className="ineff-item">
                  Steps {r.start_step}–{r.end_step}: <strong>{r.tool}</strong>
                  {r.file_path && <> on {r.file_path}</>}
                  {' '}({r.count} consecutive)
                </div>
              ))}
            </div>
          )}
          {report.backtracks.length > 0 && (
            <div className="ineff-group">
              <div className="ineff-group-title">↩️ Backtracks</div>
              {report.backtracks.map((b, i) => (
                <div key={i} className="ineff-item">
                  Step {b.step}: {b.from_stage} → {b.to_stage}
                </div>
              ))}
            </div>
          )}
          {report.cyclic_patterns.length > 0 && (
            <div className="ineff-group">
              <div className="ineff-group-title">🔁 Cyclic Patterns</div>
              {report.cyclic_patterns.map((c, i) => (
                <div key={i} className="ineff-item">
                  Steps {c.start_step}–{c.end_step}: {c.pattern_length}-step cycle ×{c.repetitions}
                  <div style={{ fontSize: 11, color: '#888', marginLeft: 12 }}>
                    {c.pattern_signature.join(' → ')}
                  </div>
                </div>
              ))}
            </div>
          )}
          {report.redundant_steps.length > 0 && (
            <div className="ineff-group">
              <div className="ineff-group-title">♻️ Redundant Steps</div>
              {report.redundant_steps.map((r, i) => (
                <div key={i} className="ineff-item">
                  Step {r.step}: {r.tool}{r.file_path && <> on {r.file_path}</>}
                </div>
              ))}
            </div>
          )}
          {report.unnecessary_explorations.length > 0 && (
            <div className="ineff-group">
              <div className="ineff-group-title">🔍 Unnecessary Exploration</div>
              {report.unnecessary_explorations.map((u, i) => (
                <div key={i} className="ineff-item">
                  Step {u.step}: {u.tool}{u.file_path && <> on {u.file_path}</>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function InfoTip({ text }: { text: string }) {
  return (
    <span className="info-tip">
      <svg width="13" height="13" viewBox="0 0 16 16" fill="none" style={{ verticalAlign: '-1px', marginLeft: 4 }}>
        <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
        <text x="8" y="12" textAnchor="middle" fontSize="10" fill="currentColor" fontWeight="600">i</text>
      </svg>
      <span className="info-tip-text">{text}</span>
    </span>
  );
}

function MetricCard({ label, value, pct, tooltip }: { label: string; value: number; pct?: boolean; tooltip?: string }) {
  return (
    <div className="stat-card">
      <div className="label">{label}{tooltip && <InfoTip text={tooltip} />}</div>
      <div className="value">{value?.toFixed(0) ?? '—'}{pct ? '%' : ''}</div>
    </div>
  );
}


