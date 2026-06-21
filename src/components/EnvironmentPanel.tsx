import { type RefObject } from 'react';
import type { PacmanEnvironment } from '../env/environment';

type ViewMode = 'live' | 'heatmap' | 'qvalues';

export interface EnvironmentPanelProps {
  /** Canvas the renderer draws into (owned by App so its effect can target it). */
  canvasRef: RefObject<HTMLCanvasElement>;
  /** Maze body — fullscreen target (N10: a ref, not a brittle querySelector). */
  mazeBodyRef: RefObject<HTMLDivElement>;
  /** Read live for the HUD (stepCount, pelletsLeft, world dims). Mutated in place;
   *  the panel re-renders with App on every tick, so reads stay current. */
  env: PacmanEnvironment;
  viewMode: ViewMode;
  setViewMode: (v: ViewMode) => void;
  episodeCount: number;
  scatterPhase: boolean;
  numGhosts: number;
  maxEpisodeSteps: number;
  pacScore: number;
  ghostsEatenCombo: number;
}

/**
 * Environment column (A5 slice 4b): the view-mode pills, the maze canvas + HUD
 * chips (episode/step, scatter-vs-chase phase, grid size, px/tile), and the
 * bottom stat strip. Pure presentation — the canvas element is owned by App
 * (its renderer effect targets canvasRef); this component only lays it out.
 */
export function EnvironmentPanel({
  canvasRef, mazeBodyRef, env, viewMode, setViewMode,
  episodeCount, scatterPhase, numGhosts, maxEpisodeSteps, pacScore, ghostsEatenCombo,
}: EnvironmentPanelProps): JSX.Element {
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Environment</span>
        <div className="panel-header-spacer" />
        <div className="pill-group" role="group" aria-label="View mode">
          {(['live', 'heatmap', 'qvalues'] as const).map((v) => (
            <button
              key={v}
              className={`pill-btn${viewMode === v ? ' active' : ''}`}
              onClick={() => setViewMode(v)}
              aria-pressed={viewMode === v}
            >
              {v === 'live' ? 'Live' : v === 'heatmap' ? 'Heatmap' : 'Q-Values'}
            </button>
          ))}
        </div>
        <button className="icon-btn" aria-label="Fullscreen" onClick={() => {
          mazeBodyRef.current?.requestFullscreen?.();
        }}>⤢</button>
      </div>

      <div className="maze-body" ref={mazeBodyRef}>
        <div className="maze-vignette" />
        <div className="maze-stage">
          <div className="hud-chip hud-top-left"
            title={numGhosts > 0 ? 'Ghost AI phase — scatter: flee to corners · chase: hunt Pac-Man' : undefined}>
            EP {episodeCount.toLocaleString()} / Step {env.stepCount}
            {numGhosts > 0 && (
              <>
                {' · '}
                <span style={{
                  display: 'inline-block', width: 7, height: 7, borderRadius: '50%',
                  marginRight: 5, verticalAlign: 'middle',
                  background: scatterPhase ? '#38bdf8' : '#ef4444',
                }} />
                {scatterPhase ? 'SCATTER' : 'CHASE'}
              </>
            )}
          </div>
          <div className="hud-top-right">
            <div className="hud-chip">{env.world.width}×{env.world.height}</div>
            <div className="hud-chip">
              {env.world.width > 0 && canvasRef.current
                ? Math.round(canvasRef.current.width / env.world.width)
                : 0} px/tile
            </div>
          </div>
          <canvas ref={canvasRef} className="maze-canvas" />
        </div>
      </div>

      {/* 5-up stat strip */}
      <div className="stat-strip">
        <div className="stat-strip-item">
          <span className="stat-strip-label">Score</span>
          <span className="stat-strip-value accent">{pacScore}</span>
        </div>
        <div className="stat-strip-item">
          <span className="stat-strip-label">Pellets Left</span>
          <span className="stat-strip-value">{env.pelletsLeft}</span>
        </div>
        <div className="stat-strip-item">
          <span className="stat-strip-label">Step</span>
          <span className="stat-strip-value">
            {env.stepCount}<span className="stat-strip-mute">/{maxEpisodeSteps}</span>
          </span>
        </div>
        <div className="stat-strip-item">
          <span className="stat-strip-label">Ghosts Eaten</span>
          <span className="stat-strip-value green">{ghostsEatenCombo}</span>
        </div>
      </div>
    </div>
  );
}
