import { useCallback, useEffect, useState } from 'react';
import * as api from './api/client';
import type { TraceInfo, ProfileResponse } from './types';
import { AppNav } from './components/AppNav';
import { AppLanding } from './components/AppLanding';
import { Tier1Dashboard } from './components/Tier1Dashboard';
import { Tier2Assessment } from './components/Tier2Assessment';
import { CompareView } from './components/CompareView';

type View = 'landing' | 'profile' | 'assess' | 'compare';

export default function App() {
  const [traces, setTraces] = useState<TraceInfo[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [view, setView] = useState<View>('landing');
  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const res = await api.listTraces();
    setTraces(res.traces);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const handleUpload = async (files: File[]) => {
    setLoading(true);
    setError(null);
    try {
      if (files.length === 1) {
        await api.uploadFile(files[0]);
      } else {
        await api.uploadBatch(files);
      }
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleViewProfile = async (id: string) => {
    setSelected(id);
    setLoading(true);
    setError(null);
    try {
      const p = await api.getProfile(id);
      setProfile(p);
      setView('profile');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load profile');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await api.deleteTrace(id);
      if (selected === id) {
        setSelected(null);
        setView('landing');
        setProfile(null);
      }
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Delete failed');
    }
  };

  const handleBackToLanding = () => {
    setView('landing');
    setSelected(null);
    setProfile(null);
  };

  return (
    <div className="app">
      {/* Persistent AgentLens navbar across all views */}
      <AppNav />

      {/* Compare button visible when 2+ traces loaded and on landing */}
      {view === 'landing' && traces.length >= 2 && (
        <div style={{ position: 'fixed', bottom: 24, right: 24, zIndex: 900 }}>
          <button
            className="gt-btn"
            style={{ padding: '10px 20px', fontSize: 14, boxShadow: '0 2px 12px rgba(0,0,0,0.15)' }}
            onClick={() => setView('compare')}
          >
            Compare Trajectories
          </button>
        </div>
      )}

      {error && (
        <div className="error-banner" style={{
          position: 'fixed', top: 64, left: '50%', transform: 'translateX(-50%)',
          zIndex: 1000, padding: '8px 16px', background: '#fef2f2', color: '#991b1b',
          borderRadius: 6, fontSize: 13, boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          {error}
          <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#991b1b', fontWeight: 600 }}>×</button>
        </div>
      )}

      {view === 'landing' && (
        <AppLanding
          traces={traces}
          onUpload={handleUpload}
          onViewProfile={handleViewProfile}
          onDelete={handleDelete}
          loading={loading}
        />
      )}

      {view === 'profile' && (
        <div className="detail-view">
          <div className="detail-view-header">
            <button className="back-btn" onClick={handleBackToLanding}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M19 12H5m0 0l7 7m-7-7l7-7" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Back to Instances
            </button>
            {selected && (
              <span className="detail-view-trace">
                {traces.find(t => t.trace_id === selected)?.task || traces.find(t => t.trace_id === selected)?.label || selected}
              </span>
            )}
          </div>
          <div className="detail-view-body">
            {loading && <div className="loading">Processing...</div>}
            {!loading && profile && (
              <Tier1Dashboard
                profile={profile}
                passed={traces.find(t => t.trace_id === selected)?.passed ?? null}
                onStartAssessment={() => setView('assess')}
              />
            )}
          </div>
        </div>
      )}

      {view === 'assess' && profile && selected && (
        <div className="detail-view">
          <div className="detail-view-header">
            <button className="back-btn" onClick={handleBackToLanding}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M19 12H5m0 0l7 7m-7-7l7-7" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Back to Instances
            </button>
            <span className="detail-view-trace">
              {traces.find(t => t.trace_id === selected)?.task || traces.find(t => t.trace_id === selected)?.label || selected}
            </span>
          </div>
          <div className="detail-view-body">
            <Tier2Assessment
              traceId={selected}
              profile={profile}
              passed={traces.find(t => t.trace_id === selected)?.passed ?? null}
              onBack={() => setView('profile')}
            />
          </div>
        </div>
      )}

      {view === 'compare' && (
        <div className="detail-view">
          <div className="detail-view-header">
            <span className="detail-view-trace">Trajectory Comparison</span>
          </div>
          <div className="detail-view-body">
            <CompareView
              traces={traces}
              onBack={handleBackToLanding}
            />
          </div>
        </div>
      )}
    </div>
  );
}
