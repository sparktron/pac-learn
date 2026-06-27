import { fmtNum } from '../uiHelpers';

export interface TopBarProps {
  version: string;
  isTraining: boolean;
  episodeCount: number;
  avgScore: number;
  bestScore: number;
  curEpsilon: number;
  /** Reset the env + trainer stats back to a fresh state. */
  onReset: () => void;
  /** Toggle the training loop (start when idle, pause when running). */
  onToggleTraining: () => void;
}

// A5 slice 5: the topbar (brand, status pill, key stats, action buttons) is a
// pure presentational header. All values are derived in App and passed in; the
// two buttons call back out so the env/trainer wiring stays in App.
export function TopBar({
  version,
  isTraining,
  episodeCount,
  avgScore,
  bestScore,
  curEpsilon,
  onReset,
  onToggleTraining,
}: TopBarProps): JSX.Element {
  return (
    <header className="topbar">
      {/* Brand */}
      <div className="topbar-brand">
        <div className="brand-logo" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 18 18">
            <path d="M9 9 L17 5.7 A8 8 0 1 0 17 12.3 Z" fill="#000" />
          </svg>
        </div>
        <div>
          <div className="brand-name">Pac Learn</div>
          <div className="brand-version">v{version}</div>
        </div>
      </div>

      {/* Status Pill */}
      <div className={`status-pill ${isTraining ? 'training' : 'idle'}`}>
        <div className="status-dot" />
        <span className="status-text">
          {isTraining ? `Training · ep ${episodeCount.toLocaleString()}` : 'Idle'}
        </span>
      </div>

      <div className="topbar-spacer" />

      {/* Key Stats */}
      <div className="topbar-stats">
        <div className="topbar-stat">
          <span className="topbar-stat-label">Episodes</span>
          <span className="topbar-stat-value">{episodeCount.toLocaleString()}</span>
        </div>
        <div className="topbar-stat">
          <span className="topbar-stat-label">Avg Score</span>
          <span className="topbar-stat-value">{fmtNum(avgScore, 1)}</span>
        </div>
        <div className="topbar-stat">
          <span className="topbar-stat-label">Best</span>
          <span className="topbar-stat-value accent">{fmtNum(bestScore, 0)}</span>
        </div>
        <div className="topbar-stat">
          <span className="topbar-stat-label">ε</span>
          <span className="topbar-stat-value">{fmtNum(curEpsilon, 3)}</span>
        </div>
      </div>

      {/* Action buttons */}
      <div className="btn-row">
        {/* N8: top Reset also clears trainer stats, so the HUD chip
            doesn't keep showing the old episode count next to a fresh
            env. Previously this button cleared the env but left the
            stats counter ticking — and "Reset Q" cleared both — which
            meant the two buttons silently disagreed about scope. */}
        <button className="btn btn-ghost" onClick={onReset}>
          Reset
        </button>
        <button className="btn btn-outline" onClick={onToggleTraining}>
          {isTraining ? 'Pause' : 'Resume'} <span className="kbd">␣</span>
        </button>
        <button className="btn btn-primary" onClick={onToggleTraining}>
          {isTraining ? '⏸ Pause' : '▶ Training'}
        </button>
      </div>
    </header>
  );
}
