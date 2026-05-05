import type { BatchAssessResponse, CohortEntry } from '../types';

interface Props {
  batch: BatchAssessResponse;
  onSelect: (id: string) => void;
}

export function CohortView({ batch, onSelect }: Props) {
  const { ranking, trajectories } = batch;

  return (
    <div>
      <h2 style={{ fontSize: 16, marginBottom: 16 }}>Cohort Comparison</h2>

      {/* Summary chips */}
      {ranking.summary && (
        <div className="summary-row">
          {Object.entries(ranking.summary).map(([k, v]) => (
            <span key={k} className="summary-chip">
              {k.replace(/_/g, ' ')}: {String(v)}
            </span>
          ))}
        </div>
      )}

      {/* Passing table */}
      {ranking.passing.length > 0 && (
        <div className="section">
          <div className="section-title">Passing Trajectories</div>
          <RankingTable
            entries={ranking.passing}
            trajectories={trajectories}
            onSelect={onSelect}
          />
        </div>
      )}

      {/* Failing table */}
      {ranking.failing.length > 0 && (
        <div className="section">
          <div className="section-title">Failing Trajectories</div>
          <RankingTable
            entries={ranking.failing}
            trajectories={trajectories}
            onSelect={onSelect}
          />
        </div>
      )}
    </div>
  );
}

function RankingTable({
  entries,
  trajectories,
  onSelect,
}: {
  entries: CohortEntry[];
  trajectories: BatchAssessResponse['trajectories'];
  onSelect: (id: string) => void;
}) {
  return (
    <table className="ranking-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Label</th>
          <th>Score</th>
          <th>Tier</th>
          <th>Top Issue</th>
        </tr>
      </thead>
      <tbody>
        {entries.map((e) => {
          const t = trajectories.find((tr) => tr.label === e.label);
          return (
            <tr
              key={e.label}
              onClick={() => t && onSelect(t.trace_id)}
            >
              <td>{e.rank}</td>
              <td title={e.label} style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {e.label}
              </td>
              <td>
                <span className="score-bar-bg">
                  <span
                    className="score-bar-fill"
                    style={{
                      width: `${e.quality_score}%`,
                      background: e.quality_score >= 70 ? 'var(--green)' : e.quality_score >= 40 ? 'var(--yellow)' : 'var(--red)',
                    }}
                  />
                </span>
                {e.quality_score.toFixed(0)}
              </td>
              <td>
                <span className={`tier-badge ${e.quality_tier}`}>
                  {e.quality_tier.replace('_', ' ')}
                </span>
              </td>
              <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                {e.top_failure_reason ?? '—'}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
