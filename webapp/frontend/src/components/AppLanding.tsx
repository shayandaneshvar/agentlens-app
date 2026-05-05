import { useRef, useState, type DragEvent } from 'react';
import type { TraceInfo } from '../types';

interface Props {
  traces: TraceInfo[];
  onUpload: (files: File[]) => void;
  onViewProfile: (id: string) => void;
  onDelete: (id: string) => void;
  loading: boolean;
}

const TABS = [
  'Run Scores',
  'Run Errors',
  'Instances',
  'Instance Scores',
  'Tool Call Results',
  'Tool Counts',
  'Trajectory Tree',
  'Logs',
];

export function AppLanding({ traces, onUpload, onViewProfile, onDelete, loading }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dirInputRef = useRef<HTMLInputElement>(null);
  const headerFileRef = useRef<HTMLInputElement>(null);
  const headerDirRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [showUploadMenu, setShowUploadMenu] = useState(false);

  const filterFiles = (fileList: FileList | File[]) =>
    Array.from(fileList).filter(f => f.name.endsWith('.zip') || f.name.endsWith('.json'));

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = filterFiles(e.dataTransfer.files);
    if (files.length > 0) onUpload(files);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = filterFiles(e.target.files);
      if (files.length > 0) onUpload(files);
    }
    e.target.value = '';
  };

  const resolvedCount = traces.filter(t => t.passed === true).length;
  const totalCount = traces.length;
  const resolvedPct = totalCount > 0 ? ((resolvedCount / totalCount) * 100).toFixed(2) : '0';

  return (
    <div className="alens">
      {/* ── Tab bar ─────────────────────────────────────────────── */}
      <div className="alens-tabs">
        {TABS.map(tab => (
          <button
            key={tab}
            className={`alens-tab${tab === 'Instances' ? ' active' : ''}`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* ── Content area ────────────────────────────────────────── */}
      <div className="alens-content">
        {/* Filters row */}
        <div className="alens-filters">
          <span className="filter-label">Filters:</span>
          <button className="filter-chip">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M6 12h12M9 18h6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Instance ID
          </button>
          <button className="filter-chip">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M6 12h12M9 18h6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Resolved Status
          </button>
          <button className="filter-chip">» Additional filters</button>
        </div>

        {traces.length === 0 ? (
          /* ── Empty state: upload zone ──────────────────────────── */
          <div className="alens-upload-area">
            <div
              className={`alens-upload-zone${dragOver ? ' drag-over' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="upload-icon-large">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5">
                  <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 14v4a2 2 0 002 2h12a2 2 0 002-2v-4" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3>Upload Trajectories</h3>
              <p>Drop .zip or .json trajectory files here, or click to browse</p>
              <p className="upload-hint">Upload agent trajectories to view instances and run behavioral analysis</p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".zip,.json"
                multiple
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </div>
            <div className="upload-actions">
              <button
                className="gt-btn secondary"
                style={{ fontSize: 12, padding: '6px 14px' }}
                onClick={e => { e.stopPropagation(); dirInputRef.current?.click(); }}
              >
                Upload Folder
              </button>
              <input
                ref={dirInputRef}
                type="file"
                /* @ts-expect-error webkitdirectory is not in the TS types */
                webkitdirectory=""
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </div>
            {loading && <div className="loading">Processing uploads...</div>}
          </div>
        ) : (
          /* ── Instance table ────────────────────────────────────── */
          <>
            {/* Stats header */}
            <div className="alens-stats-header">
              <div className="alens-stats-left">
                <h3 className="alens-instances-title">Instances</h3>
                <span className="instance-count">
                  Showing {totalCount} of {totalCount} instances
                </span>
                <span className="resolved-badge">
                  Resolved: {resolvedCount}/{totalCount} ({resolvedPct}%)
                </span>
              </div>
              <div className="alens-stats-right">
                <label className="alens-toggle">
                  <span className="toggle-label">Include Scores</span>
                  <span className="toggle-info">ⓘ</span>
                </label>
                <div style={{ position: 'relative', display: 'inline-block' }}>
                  <button className="alens-action-btn" onClick={() => setShowUploadMenu(v => !v)}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: '-2px', marginRight: 4 }}>
                      <path d="M12 4v12m0-12l-4 4m4-4l4 4M4 18h16" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    Upload
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginLeft: 4 }}>
                      <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </button>
                  {showUploadMenu && (
                    <div style={{
                      position: 'absolute', top: '100%', right: 0, marginTop: 4,
                      background: 'var(--surface)', border: '1px solid var(--border)',
                      borderRadius: 6, boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                      zIndex: 50, minWidth: 180, overflow: 'hidden',
                    }}>
                      <button
                        style={{ display: 'block', width: '100%', padding: '8px 14px', fontSize: 13, background: 'none', border: 'none', textAlign: 'left', cursor: 'pointer' }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg)')}
                        onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                        onClick={() => { setShowUploadMenu(false); headerFileRef.current?.click(); }}
                      >
                        Upload Files
                      </button>
                      <button
                        style={{ display: 'block', width: '100%', padding: '8px 14px', fontSize: 13, background: 'none', border: 'none', textAlign: 'left', cursor: 'pointer' }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg)')}
                        onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                        onClick={() => { setShowUploadMenu(false); headerDirRef.current?.click(); }}
                      >
                        Upload Folder
                      </button>
                    </div>
                  )}
                </div>
                <input
                  ref={headerFileRef}
                  type="file"
                  accept=".zip,.json"
                  multiple
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
                <input
                  ref={headerDirRef}
                  type="file"
                  /* @ts-expect-error webkitdirectory is not in the TS types */
                  webkitdirectory=""
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
                <button className="alens-action-btn">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: '-2px', marginRight: 4 }}>
                    <rect x="8" y="2" width="12" height="16" rx="2" /><path d="M4 6h2v14h10" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Copy
                </button>
                <button className="alens-action-btn">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: '-2px', marginRight: 4 }}>
                    <path d="M12 16V4m0 12l-4-4m4 4l4-4M4 18h16" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Export
                </button>
                <button className="alens-action-btn" title="Expand">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 4l5 5M20 4l-5 5M4 20l5-5M20 20l-5-5M4 4h5M4 4v5M20 4h-5M20 4v5M4 20h5M4 20v-5M20 20h-5M20 20v-5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>
              </div>
            </div>

            {loading && <div className="loading">Processing...</div>}

            {/* Data table */}
            <div className="alens-table-wrap">
              <table className="alens-table">
                <thead>
                  <tr>
                    <th>Benchmark</th>
                    <th>
                      Task Name
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" style={{ marginLeft: 4, verticalAlign: '-1px' }}>
                        <path d="M7 10l5-5 5 5M7 14l5 5 5-5" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </th>
                    <th>
                      Instance
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" style={{ marginLeft: 4, verticalAlign: '-1px' }}>
                        <path d="M7 10l5-5 5 5M7 14l5 5 5-5" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </th>
                    <th style={{ width: 110, textAlign: 'center' }}></th>
                    <th>Steps</th>
                    <th>Input Tokens</th>
                    <th>Output Tokens</th>
                    <th>Cached Input Tokens</th>
                    <th>Total Tokens</th>
                    <th>Resolved</th>
                    <th>Infra Start</th>
                  </tr>
                </thead>
                <tbody>
                  {traces.map(t => (
                    <tr key={t.trace_id}>
                      <td className="cell-benchmark">{t.benchmark || 'SWE-Bench'}</td>
                      <td className="cell-instance">
                        <span className="instance-link">{t.task || '—'}</span>
                      </td>
                      <td className="cell-instance">
                        <span className="instance-link">{t.label}</span>
                      </td>
                      <td className="cell-actions">
                        {/* Download raw trajectory */}
                        <button className="tbl-icon-btn" title="Download raw trajectory">
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M12 4v12m0 0l-4-4m4 4l4-4M4 18h16" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </button>
                        {/* View raw trajectory log */}
                        <button className="tbl-icon-btn" title="View raw trajectory log">
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </button>
                        {/* ★ Behavioral Profile / Quality Assessment — our addition */}
                        <button
                          className="tbl-icon-btn profile-action"
                          title="View Behavioral Profile / Quality Assessment"
                          onClick={() => onViewProfile(t.trace_id)}
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="12" width="4" height="8" rx="1" />
                            <rect x="10" y="8" width="4" height="12" rx="1" />
                            <rect x="17" y="4" width="4" height="16" rx="1" />
                          </svg>
                        </button>
                        {/* Delete */}
                        <button
                          className="tbl-icon-btn delete-action"
                          title="Delete trajectory"
                          onClick={e => { e.stopPropagation(); onDelete(t.trace_id); }}
                        >
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M6 18L18 6M6 6l12 12" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </button>
                      </td>
                      <td>{t.state_count}</td>
                      <td className="cell-muted">No data</td>
                      <td className="cell-muted">No data</td>
                      <td className="cell-muted">No data</td>
                      <td className="cell-muted">No data</td>
                      <td>
                        <span className={`resolved-status ${t.passed === true ? 'pass' : t.passed === false ? 'fail' : 'unknown'}`}>
                          {t.passed === true ? 'Resolved' : t.passed === false ? 'Failed' : 'Unknown'}
                        </span>
                      </td>
                      <td className="cell-muted">No data</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
