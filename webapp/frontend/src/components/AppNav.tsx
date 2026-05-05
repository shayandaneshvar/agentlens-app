export function AppNav() {
  return (
    <nav className="alens-nav">
      <div className="alens-nav-left">
        <div className="alens-logo">
          <svg width="22" height="22" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="24" r="20" fill="#1565c0"/>
            <text x="24" y="30" textAnchor="middle" fontSize="20" fontWeight="bold" fill="#fff" fontFamily="sans-serif">A</text>
          </svg>
          <div className="alens-brand">
            <span className="alens-title">AgentLens</span>
            <span className="alens-subtitle">Trajectory Analysis</span>
          </div>
        </div>
        <div className="alens-nav-links">
          <span className="alens-link">Leaderboard</span>
          <span className="alens-link">Reports</span>
          <span className="alens-link">Runs</span>
        </div>
      </div>
      <div className="alens-nav-right">
        <span className="alens-link">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: '-2px', marginRight: 4 }}>
            <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Documentation
        </span>
        <span className="alens-link">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: '-2px', marginRight: 4 }}>
            <path d="M18.364 5.636a9 9 0 11-12.728 0M12 9v4m0 4h.01" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Support
        </span>
      </div>
    </nav>
  );
}
