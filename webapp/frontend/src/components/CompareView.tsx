import { useRef, useState, type DragEvent } from 'react';
import * as api from '../api/client';
import type {
  TraceInfo, CompareResponse, CompareCandidate,
  LLMCompareResponse, LLMDimensionComparison,
} from '../types';
import { GTGraphView, type CandidateOverlay } from './GTGraphView';

const CANDIDATE_COLORS = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#a855f7'];

const STAGES = ['exploration', 'implementation', 'verification', 'orchestration'] as const;

type Phase = 'select' | 'needsGT' | 'results';

interface Props {
  traces: TraceInfo[];
  onBack: () => void;
}

export function CompareView({ traces, onBack }: Props) {
  const [phase, setPhase] = useState<Phase>('select');
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [llmResult, setLlmResult] = useState<LLMCompareResponse | null>(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmError, setLlmError] = useState<string | null>(null);
  const [llmExpanded, setLlmExpanded] = useState(true);

  // GT overlay toggle
  const [showOverlay, setShowOverlay] = useState(false);

  // GT path selection strategy
  const [gtStrategy, setGtStrategy] = useState<'best_match' | 'canonical'>('best_match');

  // GT setup state
  const [gtFiles, setGtFiles] = useState<File[]>([]);
  const [gtLoading, setGtLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const importRef = useRef<HTMLInputElement>(null);

  const toggleSelection = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : prev.length < 5 ? [...prev, id] : prev,
    );
  };

  const runCompare = async () => {
    if (selectedIds.length < 2) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setLlmResult(null);
    try {
      const res = await api.compareTraces(selectedIds, gtStrategy);
      setResult(res);
      setPhase('results');
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Comparison failed';
      if (msg.toLowerCase().includes('no ground truth') || msg.toLowerCase().includes('ground truth')) {
        setPhase('needsGT');
        setError(null);
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = () => {
    if (selectedIds.length < 2) return;
    // Clear previous GT files so user starts fresh each time
    setGtFiles([]);
    // Always ask the user to provide GT (upload trajectories or import GT JSON)
    setPhase('needsGT');
  };

  const handleUploadGtAndCompare = async () => {
    if (gtFiles.length < 2) return;
    setGtLoading(true);
    setError(null);
    try {
      // Use first selected comparison candidate to trigger assessWithGT
      // which both builds the GT and runs assessment
      await api.assessWithGT(selectedIds[0], gtFiles);
      // GT now exists — run the comparison
      await runCompare();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to build ground truth');
    } finally {
      setGtLoading(false);
    }
  };

  const handleImportGtAndCompare = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setGtLoading(true);
    setError(null);
    try {
      await api.importGT(file);
      // GT now exists — run the comparison
      await runCompare();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to import ground truth');
    } finally {
      setGtLoading(false);
    }
  };

  const handleLlmCompare = async () => {
    setLlmLoading(true);
    setLlmError(null);
    try {
      const res = await api.llmCompare(selectedIds);
      setLlmResult(res);
    } catch (e) {
      setLlmError(e instanceof Error ? e.message : 'LLM comparison failed');
    } finally {
      setLlmLoading(false);
    }
  };

  // ── Phase: GT Setup (no ground truth exists yet) ──
  if (phase === 'needsGT') {
    return (
      <div className="compare-page">
        <div className="compare-gt-header">
          <button className="gt-btn secondary" onClick={() => setPhase('select')}>← Back to Selection</button>
          <h3 style={{ margin: 0, fontSize: 16 }}>Provide Ground Truth</h3>
        </div>

        <div style={{
          padding: '16px 20px', borderRadius: 8, marginBottom: 20,
          background: '#eff6ff', border: '1px solid #bfdbfe',
        }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4, color: '#1e40af' }}>
            Ground truth needed for comparison
          </div>
          <div style={{ fontSize: 13, color: '#1e3a5f', lineHeight: 1.5 }}>
            Upload trajectory files to merge into a ground truth (PTA), or import a previously exported merged GT JSON.
            The selected trajectories will then be matched against this ground truth.
          </div>
        </div>

        {error && (
          <div style={{ color: 'var(--red)', fontSize: 13, marginBottom: 12, padding: '8px 12px', background: '#fef2f2', borderRadius: 6 }}>
            {error}
            <button onClick={() => setError(null)} style={{ marginLeft: 8, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--red)', fontWeight: 600 }}>×</button>
          </div>
        )}

        {/* Option 1: Upload passing zip/json files to merge */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
            Option 1: Upload Trajectories to Merge
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>
            Upload at least 2 trajectory .zip or .json files to merge into a ground truth PTA.
          </div>
          <div
            style={{
              padding: 20, textAlign: 'center', borderRadius: 8, cursor: 'pointer',
              border: dragOver ? '2px dashed #3b82f6' : '2px dashed var(--border)',
              background: dragOver ? '#eff6ff' : 'var(--bg-card)', marginBottom: 8,
            }}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e: DragEvent) => { e.preventDefault(); setDragOver(false); const arr = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.zip') || f.name.endsWith('.json')); setGtFiles(prev => [...prev, ...arr]); }}
            onClick={() => inputRef.current?.click()}
          >
            <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
              Drop passing .zip or .json files here, or click to browse
            </div>
            <input ref={inputRef} type="file" accept=".zip,.json" multiple onChange={(e) => { if (e.target.files) { const arr = Array.from(e.target.files).filter(f => f.name.endsWith('.zip') || f.name.endsWith('.json')); setGtFiles(prev => [...prev, ...arr]); } e.target.value = ''; }} style={{ display: 'none' }} />
          </div>
          {gtFiles.length > 0 && (
            <div style={{ marginBottom: 8 }}>
              {gtFiles.map((f, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 0', fontSize: 12 }}>
                  <span className="trace-dot pass" />
                  <span style={{ flex: 1 }}>{f.name}</span>
                  <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{(f.size / 1024).toFixed(0)} KB</span>
                  <button onClick={() => setGtFiles(prev => prev.filter((_, j) => j !== i))} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#999', fontSize: 13 }}>×</button>
                </div>
              ))}
            </div>
          )}
          <button
            className="gt-btn"
            disabled={gtFiles.length < 2 || gtLoading}
            onClick={handleUploadGtAndCompare}
          >
            {gtLoading ? 'Building GT & Comparing...' : `Build GT & Compare (${gtFiles.length} files)`}
          </button>
        </div>

        {/* Option 2: Import pre-built GT */}
        <div style={{ paddingTop: 16, borderTop: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>Option 2 — Import a previously exported merged GT:</span>
          <button className="gt-btn secondary" style={{ fontSize: 12, padding: '5px 12px' }} onClick={() => importRef.current?.click()} disabled={gtLoading}>
            Import merged GT JSON
          </button>
          <input ref={importRef} type="file" accept=".json" onChange={handleImportGtAndCompare} style={{ display: 'none' }} />
        </div>
      </div>
    );
  }

  // ── Phase: Selection ──
  if (phase === 'select') {
    return (
      <div className="compare-page">
        <div className="compare-select-header">
          <button className="gt-btn secondary" onClick={onBack}>← Back to Instances</button>
          <h3 style={{ margin: 0, fontSize: 16 }}>Select Trajectories to Compare</h3>
          <span className="compare-select-counter">
            {selectedIds.length}/5 selected (min 2)
          </span>
        </div>

        {error && (
          <div style={{ color: 'var(--red)', fontSize: 13, marginBottom: 8, padding: '8px 12px', background: '#fef2f2', borderRadius: 6 }}>
            {error}
            <button onClick={() => setError(null)} style={{ marginLeft: 8, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--red)', fontWeight: 600 }}>×</button>
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 16 }}>
          {traces.map((t, i) => {
            const isSelected = selectedIds.includes(t.trace_id);
            const colorIdx = isSelected ? selectedIds.indexOf(t.trace_id) : -1;
            return (
              <div
                key={t.trace_id}
                onClick={() => toggleSelection(t.trace_id)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '8px 14px', borderRadius: 8, cursor: 'pointer',
                  border: isSelected ? `2px solid ${CANDIDATE_COLORS[colorIdx]}` : '2px solid var(--border)',
                  background: isSelected ? `${CANDIDATE_COLORS[colorIdx]}10` : 'var(--bg-card)',
                  transition: 'all 0.15s',
                }}
              >
                <div style={{
                  width: 22, height: 22, borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: isSelected ? CANDIDATE_COLORS[colorIdx] : 'var(--surface)',
                  color: isSelected ? '#fff' : 'var(--text-muted)', fontSize: 12, fontWeight: 600,
                }}>
                  {isSelected ? colorIdx + 1 : i + 1}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {t.label}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {t.model || 'unknown model'} · {t.state_count} steps
                    {t.task ? ` · ${t.task}` : ''}
                  </div>
                </div>
                <span className={`resolved-status ${t.passed === true ? 'pass' : t.passed === false ? 'fail' : 'unknown'}`}>
                  {t.passed === true ? 'Resolved' : t.passed === false ? 'Failed' : 'Unknown'}
                </span>
              </div>
            );
          })}
        </div>

        <button
          className="gt-btn"
          disabled={selectedIds.length < 2 || loading}
          onClick={handleCompare}
        >
          {loading ? 'Comparing...' : `Compare ${selectedIds.length} Trajectories`}
        </button>
      </div>
    );
  }

  const handleExportGT = async () => {
    try {
      const blob = await api.exportGT();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'merged_gt.json';
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError('Failed to export merged PTA');
    }
  };

  // ── Phase: Results ──
  if (!result) return null; // shouldn't happen in 'results' phase
  const candidates = result.candidates;

  // Build overlay data for GT graph
  const overlays: CandidateOverlay[] = candidates.map((c, i) => ({
    label: c.label,
    color: CANDIDATE_COLORS[i % CANDIDATE_COLORS.length],
    matchedStateIds: c.matched_gt_state_ids,
  }));

  return (
    <div className="compare-page">
      <div className="compare-toolbar">
        <button className="gt-btn secondary" onClick={onBack}>← Back to Instances</button>
        <button className="gt-btn secondary" onClick={() => { setResult(null); setLlmResult(null); setGtFiles([]); setPhase('select'); }}>
          Change Selection
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 'auto' }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>GT Path:</span>
          <select
            value={gtStrategy}
            onChange={(e) => {
              const val = e.target.value as 'best_match' | 'canonical';
              setGtStrategy(val);
            }}
            style={{
              fontSize: 12, padding: '3px 8px', borderRadius: 4,
              border: '1px solid var(--border)', background: 'var(--bg-card)',
              color: 'var(--text)', cursor: 'pointer',
            }}
            title="best_match: picks the GT path that maximises each candidate's score (optimistic). canonical: picks the longest GT path with the most stages for all candidates (fixed reference)."
          >
            <option value="best_match">Best Match (optimistic)</option>
            <option value="canonical">Canonical (fixed)</option>
          </select>
          <button
            className="gt-btn secondary"
            style={{ fontSize: 11, padding: '4px 10px' }}
            onClick={runCompare}
            disabled={loading}
          >
            {loading ? 'Re-comparing…' : 'Re-compare'}
          </button>
        </div>
        <span className="toolbar-count">
          Comparing {candidates.length} trajectories
        </span>
      </div>

      {/* ── 1. GT Graph with overlaid paths (toggle) ── */}
      <div className="compare-card">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="section-title" style={{ margin: 0 }}>Ground Truth Coverage Overlay</div>
          <button
            className="gt-btn secondary"
            style={{ fontSize: 11, padding: '4px 10px' }}
            onClick={() => setShowOverlay((v) => !v)}
          >
            {showOverlay ? 'Hide GT Coverage' : 'View GT Coverage'}
          </button>
          <button className="gt-btn secondary" style={{ fontSize: 11, padding: '4px 10px' }} onClick={handleExportGT}>
            Export Merged PTA
          </button>
        </div>
        {showOverlay && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>
              Colored dots below nodes show which candidates matched each GT state. Dimmed nodes were missed by all.
            </div>
            <GTGraphView
              data={result.gt as Parameters<typeof GTGraphView>[0]['data']}
              candidateOverlays={overlays}
            />
          </div>
        )}
      </div>

      {/* ── 2. Score comparison table ── */}
      <div className="compare-card">
        <div className="section-title">Metrics Comparison</div>
        <MetricsTable candidates={candidates} />
      </div>

      {/* ── 2b. Per-stage effort ── */}
      <div className="compare-card">
        <div className="section-title">Per-Stage Effort</div>
        <StageTable candidates={candidates} />
      </div>

      {/* ── 3. LLM comparative assessment ── */}
      <div className="compare-card">
        <div className="section-title">LLM Comparative Assessment</div>

        {!llmResult && !llmLoading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button className="gt-btn" onClick={handleLlmCompare} disabled={llmLoading}>
              Run LLM Comparison
            </button>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Runs individual assessments then comparative synthesis
            </span>
          </div>
        )}

        {llmLoading && (
          <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>
            <div style={{ fontSize: 28, marginBottom: 8, display: 'inline-block', animation: 'llm-spin 1.2s linear infinite' }}>⚙️</div>
            <style>{`@keyframes llm-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
            <div style={{ fontSize: 13, fontWeight: 500 }}>Running comparative LLM assessment...</div>
            <div style={{ fontSize: 11, marginTop: 4 }}>Assessing {candidates.length} trajectories individually, then synthesizing comparison</div>
          </div>
        )}

        {llmError && (
          <div style={{ color: 'var(--red)', fontSize: 13, padding: '8px 12px', background: '#fef2f2', borderRadius: 6 }}>
            {llmError}
            <button onClick={() => setLlmError(null)} style={{ marginLeft: 8, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--red)', fontWeight: 600 }}>×</button>
          </div>
        )}

        {llmResult && (
          <LLMComparisonPanel
            result={llmResult}
            candidates={candidates}
            expanded={llmExpanded}
            onToggle={() => setLlmExpanded((v) => !v)}
            onRerun={handleLlmCompare}
            loading={llmLoading}
          />
        )}
      </div>
    </div>
  );
}


/* ── MetricsTable ─────────────────────────────────────────────────── */

type MetricRow = { key: string; label: string; pct?: boolean; extract: (c: CompareCandidate) => number; sub?: string };

const QUALITY_ROWS: MetricRow[] = [
  { key: 'quality_score', label: 'Quality Score', extract: (c) => c.metrics.quality_score },
  { key: 'coverage', label: 'Structural Coverage', pct: true, extract: (c) => c.metrics.coverage_percent },
  { key: 'coherence', label: 'Trajectory Coherence', pct: true, extract: (c) => c.metrics.coherence_score * 100 },
  { key: 'stage_comp', label: 'Stage Completeness', pct: true, extract: (c) => c.metrics.stage_completeness * 100 },
  { key: 'workflow', label: 'Workflow Similarity', pct: true, extract: (c) => c.metrics.workflow_similarity * 100 },
  { key: 'bottleneck', label: 'Bottleneck Coverage', pct: true, extract: (c) => c.metrics.bottleneck_coverage, sub: 'weakest stage per candidate' },
  { key: 'process_cov', label: 'Process Coverage', pct: true, extract: (c) => c.metrics.process_coverage * 100 },
  { key: 'file_cov', label: 'File Coverage', pct: true, extract: (c) => c.metrics.file_coverage * 100 },
  { key: 'steps', label: 'Total Steps', extract: (c) => c.state_count },
];

const SEVERITY_ROWS: MetricRow[] = [
  { key: 'severity', label: 'Inefficiency Severity', pct: true, extract: (c) => c.inefficiencies.severity_score * 100 },
  { key: 'wasted', label: 'Wasted Steps', extract: (c) => c.inefficiencies.total_wasted_steps },
  { key: 'wasted_tokens', label: 'Wasted Tokens', extract: (c) => c.inefficiencies.wasted_input_tokens + c.inefficiencies.wasted_output_tokens },
  { key: 'retries', label: 'Retry Loops', extract: (c) => c.inefficiencies.retry_loop_count },
  { key: 'cycles', label: 'Cyclic Patterns', extract: (c) => c.inefficiencies.cyclic_pattern_count },
  { key: 'backtracks', label: 'Backtracks', extract: (c) => c.inefficiencies.backtrack_count },
];

const HX_METRIC_ROWS: MetricRow[] = [
  { key: 'hx_score', label: 'HX Score', extract: (c) => c.human_experience_score ?? -1 },
  { key: 'human_inputs', label: 'Human Inputs', extract: (c) => c.human_input_count ?? -1 },
  { key: 'active_time', label: 'Active Time', extract: (c) => c.active_time_ms ?? -1 },
  { key: 'wall_time', label: 'Wall Time', extract: (c) => c.wall_time_ms ?? -1 },
  { key: 'permission_wait', label: 'Permission Wait', extract: (c) => c.permission_wait_ms ?? -1 },
  { key: 'subagents', label: 'Subagents', extract: (c) => c.subagent_count ?? -1 },
  { key: 'compactions', label: 'Compactions', extract: (c) => c.compaction_count ?? -1 },
];

const SEGMENT_STYLES: Record<string, { borderColor: string; bg: string; label: string }> = {
  quality: { borderColor: '#3b82f6', bg: '#eff6ff', label: 'Quality & Coverage' },
  severity: { borderColor: '#f97316', bg: '#fff7ed', label: 'Inefficiency & Severity' },
  hx: { borderColor: '#8b5cf6', bg: '#f5f3ff', label: 'Human Experience' },
};

function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function MetricsTable({ candidates }: { candidates: CompareCandidate[] }) {
  const higherBetter = new Set([
    'quality_score', 'coverage', 'coherence', 'stage_comp', 'workflow', 'bottleneck', 'process_cov', 'file_cov', 'hx_score',
  ]);
  const lowerBetter = new Set(['wasted', 'retries', 'cycles', 'backtracks', 'severity', 'wasted_tokens', 'permission_wait', 'compactions', 'human_inputs']);

  // Lead score keys get bold + segment accent color
  const leadScores = new Set(['quality_score', 'severity', 'hx_score']);

  const renderMetricRows = (rows: MetricRow[], segment: keyof typeof SEGMENT_STYLES) => {
    const seg = SEGMENT_STYLES[segment];
    return rows.map((row) => {
      const values = candidates.map((c) => row.extract(c));

      // Skip wasted_tokens row when no candidate has token data
      if (row.key === 'wasted_tokens' && values.every((v) => v === 0)) return null;

      // For HX rows: -1 means null/unavailable
      const hasData = values.some((v) => v >= 0);
      if (['hx_score', 'human_inputs', 'active_time', 'wall_time', 'permission_wait', 'subagents', 'compactions'].includes(row.key) && !hasData) {
        return null;
      }

      const validValues = values.filter((v) => v >= 0);
      const max = validValues.length > 0 ? Math.max(...validValues) : 0;
      const min = validValues.length > 0 ? Math.min(...validValues) : 0;
      const allSame = max === min;

      const isLead = leadScores.has(row.key);

      return (
        <tr key={row.key}>
          <td style={{
            fontWeight: isLead ? 700 : 500,
            background: seg.bg,
            borderLeft: `3px solid ${seg.borderColor}`,
            color: isLead ? seg.borderColor : undefined,
          }}>
            {row.label}
            {row.sub && (
              <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', fontWeight: 400 }}>
                {row.sub}
              </span>
            )}
          </td>
          {candidates.map((c, i) => {
            const v = values[i];

            // Null handling for HX rows
            if (v < 0) {
              return (
                <td key={c.trace_id} style={{ textAlign: 'center', color: 'var(--text-muted)', background: seg.bg }}>—</td>
              );
            }

            // Bold the best value in the row (no red/green bg)
            let isBest = false;
            if (!allSame && v >= 0) {
              if (higherBetter.has(row.key)) {
                isBest = v === max;
              } else if (lowerBetter.has(row.key)) {
                isBest = v === min;
              }
            }

            const bottleneckStage = row.key === 'bottleneck' ? c.metrics.bottleneck_stage : '';

            // Wasted tokens: show "wasted / total (pct%)" format
            const isWastedTokens = row.key === 'wasted_tokens';
            const totalTokens = isWastedTokens ? c.inefficiencies.total_input_tokens + c.inefficiencies.total_output_tokens : 0;
            const wastedTokenPct = isWastedTokens && totalTokens > 0 ? ((v / totalTokens) * 100).toFixed(0) : '0';

            // Time formatting for HX time rows
            const isTimeRow = ['active_time', 'wall_time', 'permission_wait'].includes(row.key);

            return (
              <td key={c.trace_id} style={{
                textAlign: 'center',
                background: seg.bg,
                fontWeight: isLead ? 700 : isBest ? 600 : 400,
                color: isLead ? seg.borderColor : undefined,
                fontSize: isLead ? 13 : undefined,
              }}>
                {isWastedTokens
                  ? <>{v.toLocaleString()} / {totalTokens.toLocaleString()} ({wastedTokenPct}%)</>
                  : isTimeRow
                    ? fmtMs(v)
                    : <>{v.toFixed(row.pct ? 1 : 0)}{row.pct ? '%' : ''}</>
                }
                {bottleneckStage && (
                  <span style={{ display: 'block', fontSize: 10, color: 'var(--text-muted)', fontWeight: 400, textTransform: 'capitalize' }}>
                    {bottleneckStage}
                  </span>
                )}
              </td>
            );
          })}
        </tr>
      );
    });
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="alens-table" style={{ fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ minWidth: 160 }}>Metric</th>
            {candidates.map((c, i) => (
              <th key={c.trace_id} style={{ textAlign: 'center', minWidth: 100 }}>
                <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: CANDIDATE_COLORS[i], marginRight: 6 }} />
                <span style={{ fontSize: 11 }}>{c.label.length > 20 ? c.label.slice(0, 18) + '…' : c.label}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {/* Agent row */}
          <tr>
            <td style={{ fontWeight: 600 }}>Agent</td>
            {candidates.map((c) => (
              <td key={c.trace_id} style={{ textAlign: 'center' }}>
                {c.agent || '—'}
              </td>
            ))}
          </tr>
          {/* Model row */}
          <tr>
            <td style={{ fontWeight: 600 }}>Model</td>
            {candidates.map((c) => (
              <td key={c.trace_id} style={{ textAlign: 'center' }}>
                {c.model || '—'}
              </td>
            ))}
          </tr>
          {/* Outcome row */}
          <tr>
            <td style={{ fontWeight: 600 }}>Outcome</td>
            {candidates.map((c) => (
              <td key={c.trace_id} style={{ textAlign: 'center' }}>
                <span className={`resolved-status ${c.passed === true ? 'pass' : c.passed === false ? 'fail' : 'unknown'}`}>
                  {c.passed === true ? 'PASS' : c.passed === false ? 'FAIL' : '?'}
                </span>
              </td>
            ))}
          </tr>

          {/* ── Quality & Coverage segment ── */}
          {renderMetricRows(QUALITY_ROWS, 'quality')}

          {/* ── Inefficiency & Severity segment ── */}
          {renderMetricRows(SEVERITY_ROWS, 'severity')}

          {/* ── Human Experience segment ── */}
          {renderMetricRows(HX_METRIC_ROWS, 'hx')}
        </tbody>
      </table>
    </div>
  );
}


/* ── StageTable ───────────────────────────────────────────────────── */

function StageTable({ candidates }: { candidates: CompareCandidate[] }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="alens-table" style={{ fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ minWidth: 120 }}>Stage</th>
            {candidates.map((c, i) => (
              <th key={c.trace_id} style={{ textAlign: 'center', minWidth: 80 }}>
                <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: CANDIDATE_COLORS[i], marginRight: 4 }} />
                <span style={{ fontSize: 11 }}>{c.label.length > 15 ? c.label.slice(0, 13) + '…' : c.label}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {STAGES.map((stage) => {
            const values = candidates.map((c) => c.stage_detail[stage]?.effort_ratio ?? 0);
            return (
              <tr key={stage}>
                <td style={{ fontWeight: 600, textTransform: 'capitalize' }}>{stage}</td>
                {candidates.map((c, i) => {
                  const v = values[i];
                  let bg = '';
                  if (v === 0) {
                    // 0× = skipped stage, always red
                    bg = '#fee2e2';
                  } else if (v > 1) {
                    // All >1 values: compare among themselves
                    const overValues = values.filter((x) => x > 1);
                    const overMin = Math.min(...overValues);
                    const overMax = Math.max(...overValues);
                    if (overMin === overMax) {
                      // All >1 values are the same → all red
                      bg = '#fee2e2';
                    } else {
                      bg = v === overMin ? '#dcfce7' : v === overMax ? '#fee2e2' : '';
                    }
                  }
                  // v > 0 && v <= 1 → no color (ideal range)
                  return (
                    <td key={c.trace_id} style={{
                      textAlign: 'center', fontWeight: bg ? 600 : 400, background: bg,
                    }}>
                      {v.toFixed(1)}×
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
        Effort multiplier: 1.0× = ideal (same steps as GT). Higher = more steps spent.
      </div>
    </div>
  );
}


/* ── LLM Comparison Panel ─────────────────────────────────────────── */

const RATING_STYLE: Record<string, { bg: string; color: string; label: string }> = {
  strong:   { bg: '#dcfce7', color: '#166534', label: 'Strong' },
  adequate: { bg: '#fef9c3', color: '#854d0e', label: 'Adequate' },
  weak:     { bg: '#fee2e2', color: '#991b1b', label: 'Weak' },
};

const DIM_LABELS: Record<string, string> = {
  strategy: '🎯 Strategy',
  efficiency: '⚡ Efficiency',
  verification: '✅ Verification',
  error_recovery: '🔄 Error Recovery',
  completeness: '📋 Completeness',
};

function LLMComparisonPanel({
  result, candidates, expanded, onToggle, onRerun, loading,
}: {
  result: LLMCompareResponse;
  candidates: CompareCandidate[];
  expanded: boolean;
  onToggle: () => void;
  onRerun: () => void;
  loading: boolean;
}) {
  const comp = result.comparative;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>via {result.model_used}</span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          <button className="gt-btn secondary" style={{ fontSize: 11, padding: '3px 10px' }} onClick={onToggle}>
            {expanded ? 'Hide' : 'Show'}
          </button>
          <button className="gt-btn secondary" style={{ fontSize: 11, padding: '3px 10px' }} onClick={onRerun} disabled={loading}>
            Re-run
          </button>
        </div>
      </div>

      {expanded && (
        <>
          {/* Comparative summary */}
          <div style={{ fontSize: 13, lineHeight: 1.6, padding: '10px 14px', background: 'var(--bg-card)', borderRadius: 8, border: '1px solid var(--border)' }}>
            {comp.comparative_summary}
          </div>

          {/* Per-dimension rankings */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 10 }}>
            {Object.entries(comp.dimension_comparison ?? {}).map(([key, dim]) => (
              <DimensionRankingCard
                key={key}
                dimensionKey={key}
                dim={dim}
                candidates={candidates}
              />
            ))}
          </div>

          {/* Key differences */}
          {comp.key_differences && comp.key_differences.length > 0 && (
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Key Differences</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {comp.key_differences.map((d, i) => (
                  <div key={i} style={{ fontSize: 12, padding: '8px 12px', borderRadius: 6, background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <div style={{ fontWeight: 600, marginBottom: 2 }}>{d.aspect}</div>
                    <div style={{ color: 'var(--text-muted)' }}>{d.observation}</div>
                    {d.labels_compared.length > 0 && (
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                        Comparing: {d.labels_compared.join(' vs ')}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recommendation */}
          {comp.recommendation && (
            <div style={{ fontSize: 12, lineHeight: 1.5, padding: '10px 14px', borderRadius: 8, background: '#eff6ff', border: '1px solid #bfdbfe' }}>
              <strong>Recommendation:</strong> {comp.recommendation}
            </div>
          )}

          {/* Individual assessments (collapsible) */}
          <IndividualAssessments individual={result.individual} candidates={candidates} />
        </>
      )}
    </div>
  );
}


function DimensionRankingCard({
  dimensionKey, dim, candidates,
}: {
  dimensionKey: string;
  dim: LLMDimensionComparison;
  candidates: CompareCandidate[];
}) {
  return (
    <div style={{ padding: '10px 14px', borderRadius: 8, background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
        {DIM_LABELS[dimensionKey] ?? dimensionKey}
      </div>
      {/* Ranking */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 6, flexWrap: 'wrap' }}>
        {(dim.ranking ?? []).map((label, rank) => {
          const cIdx = candidates.findIndex((c) => c.label === label);
          const color = cIdx >= 0 ? CANDIDATE_COLORS[cIdx] : 'var(--text-muted)';
          return (
            <span key={label} style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              fontSize: 11, padding: '2px 8px', borderRadius: 12,
              border: `1.5px solid ${color}`, color,
              fontWeight: rank === 0 ? 700 : 400,
            }}>
              #{rank + 1} {label.length > 15 ? label.slice(0, 13) + '…' : label}
            </span>
          );
        })}
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        {dim.analysis}
      </div>
    </div>
  );
}


function IndividualAssessments({
  individual, candidates,
}: {
  individual: LLMCompareResponse['individual'];
  candidates: CompareCandidate[];
}) {
  const [open, setOpen] = useState(false);

  return (
    <div>
      <button
        className="gt-btn secondary"
        style={{ fontSize: 11, padding: '4px 10px' }}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? 'Hide' : 'Show'} Individual Assessments
      </button>

      {open && (
        <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 12 }}>
          {individual.map((res, idx) => {
            const a = res.assessment;
            const color = CANDIDATE_COLORS[idx % CANDIDATE_COLORS.length];
            const cand = candidates[idx];
            return (
              <div key={res.trace_id} style={{ padding: '12px 14px', borderRadius: 8, border: `2px solid ${color}`, background: 'var(--bg-card)' }}>
                <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6, color }}>
                  {cand?.label ?? res.trace_id}
                  <span style={{ fontWeight: 400, fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>
                    Score: {res.quality_score} · {res.verdict}
                  </span>
                </div>
                {a?.summary && (
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>{a.summary}</div>
                )}
                {a?.dimensions && (
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {Object.entries(a.dimensions).map(([dKey, dVal]) => {
                      if (!dVal || typeof dVal !== 'object') return null;
                      const rating = (dVal as { rating?: string }).rating ?? 'adequate';
                      const rs = RATING_STYLE[rating] ?? RATING_STYLE.adequate;
                      return (
                        <span key={dKey} style={{
                          fontSize: 11, padding: '2px 8px', borderRadius: 12,
                          background: rs.bg, color: rs.color, fontWeight: 600,
                        }}>
                          {DIM_LABELS[dKey] ?? dKey}: {rs.label}
                        </span>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
