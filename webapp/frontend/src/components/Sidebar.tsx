import { useRef, useState, type DragEvent } from 'react';
import type { TraceInfo } from '../types';

interface Props {
  traces: TraceInfo[];
  selected: string | null;
  onUpload: (files: File[]) => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export function Sidebar({
  traces,
  selected,
  onUpload,
  onSelect,
  onDelete,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dirInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const filterFiles = (fileList: FileList | File[]) => {
    return Array.from(fileList).filter(
      (f) => f.name.endsWith('.zip') || f.name.endsWith('.json'),
    );
  };

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

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>AgentLens</h1>
        <p>Trajectory quality assessment</p>
      </div>

      <div
        className={`upload-zone${dragOver ? ' drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        Drop files here or click to upload
        <input
          ref={fileInputRef}
          type="file"
          accept=".zip,.json"
          multiple
          onChange={handleFileChange}
        />
      </div>
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: 4, marginBottom: 8 }}>
        <button
          className="gt-btn secondary"
          style={{ fontSize: 11, padding: '3px 10px' }}
          onClick={(e) => { e.stopPropagation(); dirInputRef.current?.click(); }}
        >
          Upload folder
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

      <div className="trace-list">
        {traces.map((t) => (
          <div
            key={t.trace_id}
            className={`trace-item${t.trace_id === selected ? ' active' : ''}`}
            onClick={() => onSelect(t.trace_id)}
          >
            <span
              className={`trace-dot ${t.passed === true ? 'pass' : t.passed === false ? 'fail' : 'unknown'}`}
            />
            <span className="trace-label" title={t.label}>
              {t.label}
            </span>
            <span className="trace-states">{t.state_count}s</span>
            <button
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#999', fontSize: 14 }}
              title="Delete"
              onClick={(e) => { e.stopPropagation(); onDelete(t.trace_id); }}
            >
              x
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
