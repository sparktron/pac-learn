import { useEffect, useMemo, useRef, useState } from 'react';
import { createDefaultEnv, type EnvParams } from './env/environment';
import { DIRECTIONS } from './engine/types';
import { SeededRng } from './engine/prng';
import { CanvasRenderer } from './render/canvasRenderer';
import { QLearningAgent } from './rl/qlearning';
import { TrainingController } from './rl/trainingController';
import { MAZES } from './mazes/mazes';
import type { GhostAIType } from './ghosts/ghostAi';
import { movingAverage, buildSparkPath, computeDelta, fmtNum, safeNum } from './uiHelpers';

const VERSION = '1.2.1';

// epsilonDecay=0.999997 keeps exploration alive until ~400k episodes (old 0.999
// decayed to floor in ~1600 episodes — far too early for the Q-table to converge).
// epsilonMin=0.20 maintains enough randomness to keep discovering new states.
const baseHyper = { alpha: 0.2, gamma: 0.99, epsilon: 0.5, epsilonDecay: 0.999997, epsilonMin: 0.20 };
const ghostAITypes: GhostAIType[] = ['classic', 'heatmap', 'hybrid'];

const trainingSpeedPresets = {
  slow:   { stepsPerFrame: 1,         renderEveryNSteps: 1,    frameIntervalMs: 240, maxFrameMs: 0 },
  normal: { stepsPerFrame: 1,         renderEveryNSteps: 1,    frameIntervalMs: 120, maxFrameMs: 0 },
  fast:   { stepsPerFrame: 20,        renderEveryNSteps: 5,    frameIntervalMs: 0,   maxFrameMs: 0 },
  turbo:  { stepsPerFrame: 1000,      renderEveryNSteps: 50,   frameIntervalMs: 0,   maxFrameMs: 0 },
  max:    { stepsPerFrame: 1_000_000, renderEveryNSteps: 1000, frameIntervalMs: 0,   maxFrameMs: 12 },
} as const;
type TrainingSpeed = keyof typeof trainingSpeedPresets;
const trainingSpeedOptions = Object.keys(trainingSpeedPresets) as TrainingSpeed[];

// N20: 'default' preset must exactly match defaultParams.reward in environment.ts
// so that selecting "default" from the UI gives the same reward config the env
// uses out-of-the-box. Previously winBonus was 200 here but 1000 in the env.
const rewardPresets: Record<string, EnvParams['reward']> = {
  default:             { pelletReward: 5,  powerPelletReward: 20, deathPenalty: -100, stepPenalty: -0.1,  survivalReward: 0,    ghostEatReward: 30,  winBonus: 1000, reversePenalty: -2 },
  'ghost-hunting':     { pelletReward: 2,  powerPelletReward: 30, deathPenalty: -50,  stepPenalty: -0.05, survivalReward: 0.01, ghostEatReward: 80,  winBonus: 100,  reversePenalty: -2 },
  'pellet-collection': { pelletReward: 15, powerPelletReward: 40, deathPenalty: -120, stepPenalty: -0.1,  survivalReward: 0.02, ghostEatReward: 20,  winBonus: 300,  reversePenalty: -2 },
  'survival':          { pelletReward: 3,  powerPelletReward: 20, deathPenalty: -250, stepPenalty: -0.05, survivalReward: 0.2,  ghostEatReward: 50,  winBonus: 100,  reversePenalty: -2 },
};

// ── Small Components ───────────────────────────────────────

type ToggleProps = { checked: boolean; onChange: (v: boolean) => void; label: string; sublabel?: string; id: string };

const Toggle = ({ checked, onChange, label, sublabel, id }: ToggleProps): JSX.Element => (
  <div className="toggle-row">
    <div className="toggle-labels">
      <label className="toggle-label-text" htmlFor={id}>{label}</label>
      {sublabel && <span className="toggle-sublabel">{sublabel}</span>}
    </div>
    <button
      id={id}
      role="switch"
      aria-checked={checked}
      className={`toggle-switch${checked ? ' on' : ''}`}
      onClick={() => onChange(!checked)}
    >
      <span className="toggle-knob" />
    </button>
  </div>
);

type FieldProps = {
  label: string;
  unit?: string;
  htmlFor?: string;
  children: React.ReactNode;
};

const Field = ({ label, unit, htmlFor, children }: FieldProps): JSX.Element => (
  <div className="field">
    <div className="field-label">
      <label htmlFor={htmlFor}>{label}</label>
      {unit && <span className="field-unit">{unit}</span>}
    </div>
    {children}
  </div>
);

type ChartCardProps = {
  title: string;
  color: string;
  value: string;
  values: number[];
  gradId: string;
};

const ChartCard = ({ title, color, value, values, gradId }: ChartCardProps): JSX.Element => {
  const { line, fill } = buildSparkPath(values);
  const delta = computeDelta(values);
  return (
    <div className="chart-card">
      <div className="chart-card-header">
        <div className="chart-bullet" style={{ background: color }} />
        <span className="chart-title">{title}</span>
        <div className="chart-header-spacer" />
        <span className="chart-value">{value}</span>
        {delta.dir !== 'flat' && (
          <span className={`chart-delta ${delta.dir}`}>
            {delta.dir === 'up' ? '▲' : '▼'} {fmtNum(delta.pct, 1)}%
          </span>
        )}
      </div>
      <div className="chart-sparkline">
        <svg viewBox="0 0 400 90" preserveAspectRatio="none">
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor={color} stopOpacity="0.28" />
              <stop offset="100%" stopColor={color} stopOpacity="0.02" />
            </linearGradient>
          </defs>
          <line x1="0" y1="22" x2="400" y2="22" stroke="#1f2330" strokeWidth="1" />
          <line x1="0" y1="45" x2="400" y2="45" stroke="#1f2330" strokeWidth="1" />
          <line x1="0" y1="68" x2="400" y2="68" stroke="#1f2330" strokeWidth="1" />
          {fill && <path d={fill} fill={`url(#${gradId})`} />}
          {line && <path d={line} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />}
        </svg>
      </div>
    </div>
  );
};

// ── Main App ───────────────────────────────────────────────

export default function App(): JSX.Element {
  const env     = useMemo(() => createDefaultEnv(), []);
  const agent   = useMemo(() => new QLearningAgent(baseHyper), []);
  const trainer = useMemo(() => new TrainingController(env, agent), [env, agent]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  // N10: hold a ref to the maze body so fullscreen doesn't depend on a
  // brittle `.maze-body` querySelector that breaks if a second element ever
  // gets the same class (e.g. an A/B compare panel).
  const mazeBodyRef = useRef<HTMLDivElement>(null);
  const [tick, setTick] = useState(0);
  const [seed, setSeed] = useState(42);
  const [viewMode, setViewMode] = useState<'live' | 'heatmap'>('live');
  const [mode, setMode] = useState<'human' | 'ai'>('ai');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingSpeed, setTrainingSpeed] = useState<TrainingSpeed>('normal');
  const [stepsPerFrame, setStepsPerFrame]                     = useState<number>(trainingSpeedPresets.normal.stepsPerFrame);
  const [renderEveryNSteps, setRenderEveryNSteps]             = useState<number>(trainingSpeedPresets.normal.renderEveryNSteps);
  const [trainingFrameIntervalMs, setTrainingFrameIntervalMs] = useState<number>(trainingSpeedPresets.normal.frameIntervalMs);
  const [trainingMaxFrameMs, setTrainingMaxFrameMs]           = useState<number>(trainingSpeedPresets.normal.maxFrameMs);
  // N16: structuredClone ensures the initial params object (and its reward sub-object)
  // has no shared references with env.params or the rewardPresets entries.
  const [params, setParams]             = useState<EnvParams>(() => structuredClone({ ...env.params, reward: rewardPresets['pellet-collection'] }));
  const [rewardPreset, setRewardPreset] = useState<string>('pellet-collection');
  const [ghostAIType, setGhostAIType]   = useState<GhostAIType>('classic');
  const [timeRange, setTimeRange]       = useState<120 | 500 | 0>(120);
  const [activeTab, setActiveTab]       = useState<'environment' | 'tuning' | 'runtime'>(() => {
    const s = localStorage.getItem('pac-learn-tab');
    if (s === 'rewards' || s === 'learning') return 'tuning';
    return (s as 'environment' | 'tuning' | 'runtime') ?? 'environment';
  });

  const lastStatsLengthRef         = useRef(0);
  const stepsPerFrameRef           = useRef(stepsPerFrame);
  const renderEveryNRef            = useRef(renderEveryNSteps);
  const trainingFrameIntervalMsRef = useRef(trainingFrameIntervalMs);
  const trainingMaxFrameMsRef      = useRef(trainingMaxFrameMs);
  const isTrainingRef              = useRef(isTraining);
  const startTrainingRef           = useRef<(reseed?: boolean) => void>();
  const stopTrainingRef            = useRef<() => void>();
  stepsPerFrameRef.current           = stepsPerFrame;
  renderEveryNRef.current            = renderEveryNSteps;
  trainingFrameIntervalMsRef.current = trainingFrameIntervalMs;
  trainingMaxFrameMsRef.current      = trainingMaxFrameMs;
  isTrainingRef.current              = isTraining;

  // Persist active tab
  useEffect(() => { localStorage.setItem('pac-learn-tab', activeTab); }, [activeTab]);

  // Draw canvas. Persist the renderer instance so its frame counter,
  // hash-skip cache, and computed tile size survive across redraws —
  // otherwise the mouth/pulse animations reset every render, the hash-
  // skip early-return is dead code, and `tile` is recomputed from
  // parentElement.clientWidth on every tick (visible snap on layout).
  const rendererRef = useRef<{ canvas: HTMLCanvasElement; renderer: CanvasRenderer } | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (!rendererRef.current || rendererRef.current.canvas !== canvas) {
      rendererRef.current = { canvas, renderer: new CanvasRenderer(ctx) };
    }
    rendererRef.current.renderer.draw(env, viewMode === 'heatmap');
  }, [env, tick, viewMode]);

  // N6: split the params effect.
  //
  //   1) Live-apply effect — env.setParams on every params change, with NO
  //      env.reset. Editing a reward field or speed no longer kills the
  //      in-flight episode and wipes the trainer's progress bar while the
  //      user is still typing the new value.
  //
  //   2) Reset effect — fires only when a *structural* field changes
  //      (mazeId / numGhosts / seed). These genuinely require
  //      a fresh env. Training is paused across the reset so a Q-update
  //      can't bridge the boundary (its obs is pre-reset and its nextObs
  //      is post-reset, which writes garbage Q-values).
  useEffect(() => { env.setParams(params); }, [env, params]);

  // Keep env.heatmapEnabled in sync with the UI overlay so N2's fast-path
  // (skip heatmap decay when nobody consumes it) doesn't freeze the
  // heatmap view at startup zeros when all ghosts are classic.
  useEffect(() => { env.heatmapEnabled = viewMode === 'heatmap'; }, [env, viewMode]);

  const lastSeedRef = useRef(seed);
  const lastStructuralRef = useRef(`${params.mazeId}|${params.numGhosts}`);
  useEffect(() => {
    const structural = `${params.mazeId}|${params.numGhosts}`;
    const seedChanged = lastSeedRef.current !== seed;
    if (lastStructuralRef.current === structural && !seedChanged) return;
    lastStructuralRef.current = structural;
    lastSeedRef.current = seed;
    const wasTraining = isTrainingRef.current;
    if (wasTraining) trainer.stop();
    env.reset(seed);
    setTick((t) => t + 1);
    if (wasTraining) startTrainingRef.current?.(seedChanged);
  }, [env, trainer, params.mazeId, params.numGhosts, seed]);

  // N11: re-apply ghost AI type only when something that affects the ghost
  // roster changes (the user picks a new AI, ghosts get rebuilt by an env
  // reset, or numGhosts changes). The old deps included `tick`, which fired
  // this effect on every render frame for no benefit.
  useEffect(() => {
    env.ghosts.forEach((_, i) => env.setGhostType(i, ghostAIType));
  }, [env, ghostAIType, params.numGhosts, params.mazeId, seed]);

  // AI-watch loop (non-training). Use a per-loop seeded RNG (not Math.random
  // and not trainer's RNG — that would advance the training stream and break
  // reproducibility). Use DIRECTIONS rather than the hard-coded literal so
  // any future reorder doesn't silently scramble actions.
  useEffect(() => {
    if (mode !== 'ai' || isTraining) return;
    trainer.stop();
    const watchRng = new SeededRng(seed ^ 0xA1A1);
    let episodeCounter = 0;
    const id = setInterval(() => {
      const obs = env.observe();
      const action = agent.act(obs, env.getLegalActions().map((d) => DIRECTIONS.indexOf(d)), () => watchRng.next());
      const result = env.step(action);
      if (result.done) {
        // Re-seed each episode so a death doesn't replay the identical run.
        episodeCounter += 1;
        env.reset((seed * 1000 + episodeCounter) >>> 0);
      }
      setTick((t) => t + 1);
    }, 120);
    return () => clearInterval(id);
  }, [mode, isTraining, env, agent, trainer, seed]);

  // Human keyboard
  const humanEpisodeRef = useRef(0);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (mode !== 'human') return;
      const keyMap: Record<string, number> = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 };
      const action = keyMap[e.key];
      if (action === undefined) return;
      const result = env.step(action);
      if (result.done) {
        // D7.3: reseed each episode so a death doesn't replay the identical run
        // (mirrors AI-watch). Previously reset(seed) made every post-death
        // episode a Groundhog-Day repeat.
        humanEpisodeRef.current += 1;
        env.reset((seed * 1000 + humanEpisodeRef.current) >>> 0);
      }
      setTick((t) => t + 1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [env, mode, seed]);

  // Space = toggle training
  useEffect(() => {
    const onSpace = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'BUTTON' || target.tagName === 'SELECT') return;
      e.preventDefault();
      // D7.1: go through the refs, not the first-render closures. Calling
      // startTraining() directly captured the initial seed/params.numGhosts,
      // so pressing Space after changing the seed trained with the stale value.
      if (isTrainingRef.current) stopTrainingRef.current?.();
      else startTrainingRef.current?.();
    };
    window.addEventListener('keydown', onSpace);
    return () => window.removeEventListener('keydown', onSpace);
  }, []);

  const updateTrainingSpeed = (speed: TrainingSpeed): void => {
    const p = trainingSpeedPresets[speed];
    setTrainingSpeed(speed);
    setStepsPerFrame(p.stepsPerFrame);
    setRenderEveryNSteps(p.renderEveryNSteps);
    setTrainingFrameIntervalMs(p.frameIntervalMs);
    setTrainingMaxFrameMs(p.maxFrameMs);
  };

  // reseed=false preserves the trainer's RNG stream — used when auto-resuming
  // training across a param change so a maze switch doesn't silently rewind
  // the seeded action-tie-breaker stream. Manual Start button (and explicit
  // seed changes) still reseed.
  const startTraining = (reseed = true): void => {
    trainer.stop();
    if (reseed) trainer.setSeed(seed);
    // N18: always sync the trainer's episodeSeed to the current seed so that
    // evaluate() restores the env to the right state even before the first
    // episode completes (episodeSeed defaults to 0 which resets to the wrong state).
    trainer.setCurrentSeed(seed);
    // N7: pin the numGhosts the Q-table will be trained against. Idempotent
    // if already pinned (resume / auto-restart across param edits). When the
    // user changes numGhosts mid-training, the input handler below will
    // catch the mismatch and warn.
    agent.setTrainedNumGhosts(params.numGhosts);
    setIsTraining(true);
    lastStatsLengthRef.current = trainer.stats.episodeScores.length;
    trainer.start(
      () => stepsPerFrameRef.current,
      () => renderEveryNRef.current,
      () => {
        if (trainer.stats.episodeScores.length > lastStatsLengthRef.current) {
          lastStatsLengthRef.current = trainer.stats.episodeScores.length;
        }
        setTick((t) => t + 1);
      },
      {
        getFrameIntervalMs: () => trainingFrameIntervalMsRef.current,
        getMaxFrameMs:      () => trainingMaxFrameMsRef.current,
      },
    );
  };

  const stopTraining = (): void => { trainer.stop(); setIsTraining(false); };
  startTrainingRef.current = startTraining;
  stopTrainingRef.current = stopTraining;

  // N7: when the user types a new numGhosts, refuse the change if it would
  // contradict the Q-table's pinned trained-with value. This is what kept
  // a heterogeneous-N policy from getting silently saved with a misleading
  // numGhostsEncoded field after the user adjusted numGhosts mid-session.
  const changeNumGhosts = (next: number): void => {
    const pinned = agent.trainedNumGhosts;
    if (pinned !== null && pinned !== next) {
      // eslint-disable-next-line no-alert
      const ok = window.confirm(
        `Q-table was trained with ${pinned} ghost(s). Changing to ${next} ` +
        `requires resetting the Q-table (otherwise saved policies will be tagged ` +
        `with the wrong ghost count and won't reload cleanly).\n\nReset Q-table and continue?`,
      );
      if (!ok) return;
      trainer.stop(); setIsTraining(false);
      agent.reset();
      agent.hyper.epsilon = baseHyper.epsilon;
      trainer.resetStats();
    }
    setParams((p) => ({ ...p, numGhosts: next }));
  };

  const savePolicy = (): void => {
    // serialize() prefers agent.trainedNumGhosts when set, so the fallback
    // arg only matters for a fresh-never-trained save.
    const blob = new Blob([JSON.stringify(agent.serialize(params.mazeId, params.numGhosts), null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `policy-${Date.now()}.json`;
    a.click();
  };

  const setReward = (key: keyof EnvParams['reward'], rawValue: string): void => {
    setRewardPreset('custom');
    setParams((p) => ({ ...p, reward: { ...p.reward, [key]: safeNum(rawValue, p.reward[key]) } }));
  };

  // Derived chart data
  const scores   = trainer.stats.episodeScores;
  const lengths  = trainer.stats.episodeLengths;
  const epsilons = trainer.stats.epsilons;
  // Memoize so we don't recompute on unrelated state changes (viewMode,
  // mode toggles, etc). Length-keyed because the trainer mutates the
  // existing array in place; React won't notice without an explicit key.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const movAvg   = useMemo(() => movingAverage(scores, 20), [scores.length]);

  const chartSlice = (vals: number[]): number[] =>
    timeRange === 0 ? vals : vals.slice(-timeRange);

  const episodeCount = scores.length;
  const avgScore     = episodeCount > 0 ? scores.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, scores.length) : 0;
  // Spread on a long array (>~125k) throws RangeError, so reduce.
  // N12: `scores` is mutated in place by the trainer so the array reference
  // never changes — only `.length` actually moves. Drop `scores` from deps
  // (it was misleading), keep `scores.length` as the real change signal.
  const bestScore    = useMemo(() => {
    let mx = -Infinity;
    for (const v of scores) if (v > mx) mx = v;
    return episodeCount > 0 && Number.isFinite(mx) ? mx : 0;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scores.length, episodeCount]);
  const curEpsilon   = agent.hyper.epsilon;
  const pacman       = env.getPacmen()[0];

  // ── JSX ───────────────────────────────────────────────────
  return (
    <div className="app-layout">

      {/* ── Top Bar ──────────────────────────────────────── */}
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
            <div className="brand-version">v{VERSION}</div>
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
          <button className="btn btn-ghost" onClick={() => {
            trainer.stop(); setIsTraining(false);
            trainer.resetStats();
            lastStatsLengthRef.current = 0;
            env.reset(seed); trainer.setCurrentSeed(seed); setTick((t) => t + 1); // N18
          }}>
            Reset
          </button>
          <button className="btn btn-outline" onClick={() => (isTraining ? stopTraining() : startTraining())}>
            {isTraining ? 'Pause' : 'Resume'} <span className="kbd">␣</span>
          </button>
          <button className="btn btn-primary" onClick={() => (isTraining ? stopTraining() : startTraining())}>
            {isTraining ? '⏸ Pause' : '▶ Training'}
          </button>
        </div>
      </header>

      {/* ── Main Grid ────────────────────────────────────── */}
      <main className="main-grid">

        {/* ── Col 1: Maze ──────────────────────────────── */}
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Environment</span>
            <div className="panel-header-spacer" />
            <div className="pill-group" role="group" aria-label="View mode">
              {(['live', 'heatmap'] as const).map((v) => (
                <button
                  key={v}
                  className={`pill-btn${viewMode === v ? ' active' : ''}`}
                  onClick={() => setViewMode(v)}
                  aria-pressed={viewMode === v}
                >
                  {v === 'live' ? 'Live' : 'Heatmap'}
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
              <div className="hud-chip hud-top-left">
                EP {episodeCount.toLocaleString()} / Step {env.stepCount}
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
              <span className="stat-strip-value accent">{pacman?.score ?? 0}</span>
            </div>
            <div className="stat-strip-item">
              <span className="stat-strip-label">Pellets Left</span>
              <span className="stat-strip-value">{env.pelletsLeft}</span>
            </div>
            <div className="stat-strip-item">
              <span className="stat-strip-label">Step</span>
              <span className="stat-strip-value">
                {env.stepCount}<span className="stat-strip-mute">/{params.maxEpisodeSteps}</span>
              </span>
            </div>
            <div className="stat-strip-item">
              <span className="stat-strip-label">Ghosts Eaten</span>
              <span className="stat-strip-value green">{pacman?.ghostsEatenCombo ?? 0}</span>
            </div>
          </div>
        </div>

        {/* ── Col 2: Configuration ──────────────────────── */}
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Configuration</span>
            <div className="panel-header-spacer" />
            <div className="preset-chip">
              <span className="preset-chip-label">Preset ·</span>
              <span className="preset-chip-value">{rewardPreset}</span>
            </div>
            <button className="icon-btn" aria-label="Save params" title="Save params" onClick={() => {
              const data = { env: params, hyper: agent.hyper, seed };
              const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
              const a = document.createElement('a');
              a.href = URL.createObjectURL(blob);
              a.download = `params-${Date.now()}.json`;
              a.click();
            }}>↓</button>
          </div>

          {/* Tab bar. No field counts — they were hard-coded and silently
              lied whenever a Field was added or removed. */}
          <div className="tab-bar" role="tablist">
            {([
              ['environment', 'Environment'],
              ['tuning',      'Tuning'],
              ['runtime',     'Runtime'],
            ] as [string, string][]).map(([id, label]) => (
              <button
                key={id}
                role="tab"
                aria-selected={activeTab === id}
                className={`tab-btn${activeTab === id ? ' active' : ''}`}
                onClick={() => setActiveTab(id as typeof activeTab)}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Scrollable config body */}
          <div className="config-body" role="tabpanel">

            {/* ENVIRONMENT TAB */}
            {activeTab === 'environment' && (
              <>
                <div className="config-section">
                  <div className="section-heading">Mode &amp; Maze</div>
                  <div className="field-grid">
                    <Field label="Mode" htmlFor="cfg-mode">
                      <select id="cfg-mode" className="field-select" value={mode}
                        onChange={(e) => setMode(e.target.value as 'human' | 'ai')}>
                        <option value="human">Human</option>
                        <option value="ai">AI controlled</option>
                      </select>
                    </Field>
                    <Field label="Maze" htmlFor="cfg-maze">
                      <select id="cfg-maze" className="field-select" value={params.mazeId}
                        onChange={(e) => setParams((p) => ({ ...p, mazeId: e.target.value }))}>
                        {MAZES.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
                      </select>
                    </Field>
                    <Field label="numGhosts" unit="int" htmlFor="cfg-nghosts">
                      <input id="cfg-nghosts" className="field-input" type="number"
                        value={params.numGhosts} min={1} max={6} step={1}
                        onChange={(e) => changeNumGhosts(safeNum(e.target.value, params.numGhosts))} />
                    </Field>
                    <Field label="Ghost AI" htmlFor="cfg-ghostai">
                      <select id="cfg-ghostai" className="field-select" value={ghostAIType}
                        onChange={(e) => setGhostAIType(e.target.value as GhostAIType)}>
                        {ghostAITypes.map((t) => <option key={t}>{t}</option>)}
                      </select>
                    </Field>
                    <Field label="Capture rules" htmlFor="cfg-capture">
                      <select id="cfg-capture" className="field-select" value={params.captureRules}
                        onChange={(e) => setParams((p) => ({ ...p, captureRules: e.target.value as 'touch' | 'tile' }))}>
                        <option value="touch">touch</option>
                        <option value="tile">tile</option>
                      </select>
                    </Field>
                    <Field label="ghostSpeed" htmlFor="cfg-gspeed">
                      <input id="cfg-gspeed" className="field-input" type="number"
                        value={params.ghostSpeed} min={0.2} max={3} step={0.05}
                        onChange={(e) => setParams((p) => ({ ...p, ghostSpeed: safeNum(e.target.value, p.ghostSpeed) }))} />
                    </Field>
                    <Field label="pacmanSpeed" htmlFor="cfg-pspeed">
                      <input id="cfg-pspeed" className="field-input" type="number"
                        value={params.pacmanSpeed} min={0.2} max={3} step={0.05}
                        onChange={(e) => setParams((p) => ({ ...p, pacmanSpeed: safeNum(e.target.value, p.pacmanSpeed) }))} />
                    </Field>
                  </div>
                </div>

                <div className="config-section">
                  <div className="section-heading">Toggles</div>
                  <Toggle id="tog-pp" label="Enable power pellets" sublabel="grant temporary ghost-eating window"
                    checked={params.enablePowerPellets}
                    onChange={(v) => setParams((p) => ({ ...p, enablePowerPellets: v }))} />
                  <Toggle id="tog-hm" label="Show ghost heatmap" sublabel="visualize danger overlay"
                    checked={viewMode === 'heatmap'}
                    onChange={(v) => setViewMode(v ? 'heatmap' : 'live')} />
                </div>
              </>
            )}

            {/* TUNING TAB (Rewards + Learning combined) */}
            {activeTab === 'tuning' && (
              <>
                {/* ── Rewards ───────────────────────────────── */}
                <div className="config-section">
                  <div className="section-heading">Rewards</div>
                  <div className="field-grid" style={{ marginBottom: 10 }}>
                    <Field label="Preset" htmlFor="cfg-rpreset">
                      <select id="cfg-rpreset" className="field-select" value={rewardPreset}
                        onChange={(e) => {
                          const preset = e.target.value;
                          setRewardPreset(preset);
                          // N16: spread-clone the preset so params.reward is a fresh
                          // object — NOT a direct reference to the rewardPresets entry.
                          // A direct reference would be mutated by any code that writes
                          // params.reward[key], silently corrupting the preset for the
                          // session. 'custom' has no entry → guard prevents a setParams.
                          if (preset in rewardPresets) setParams((p) => ({ ...p, reward: { ...rewardPresets[preset] } }));
                        }}>
                        {rewardPreset === 'custom' && <option value="custom">custom</option>}
                        {Object.keys(rewardPresets).map((p) => <option key={p} value={p}>{p}</option>)}
                      </select>
                    </Field>
                  </div>
                  <div className="field-grid">
                    <Field label="pelletReward" htmlFor="cfg-pr">
                      <input id="cfg-pr" className="field-input" type="number"
                        value={params.reward.pelletReward} step={1}
                        onChange={(e) => setReward('pelletReward', e.target.value)} />
                    </Field>
                    <Field label="powerPelletReward" htmlFor="cfg-ppr">
                      <input id="cfg-ppr" className="field-input" type="number"
                        value={params.reward.powerPelletReward} step={1}
                        onChange={(e) => setReward('powerPelletReward', e.target.value)} />
                    </Field>
                    <Field label="ghostEatReward" htmlFor="cfg-ger">
                      <input id="cfg-ger" className="field-input" type="number"
                        value={params.reward.ghostEatReward} step={1}
                        onChange={(e) => setReward('ghostEatReward', e.target.value)} />
                    </Field>
                    <Field label="winBonus" htmlFor="cfg-wb">
                      <input id="cfg-wb" className="field-input" type="number"
                        value={params.reward.winBonus} step={10}
                        onChange={(e) => setReward('winBonus', e.target.value)} />
                    </Field>
                    <Field label="deathPenalty" htmlFor="cfg-dp">
                      <input id="cfg-dp" className="field-input" type="number"
                        value={params.reward.deathPenalty} step={1}
                        onChange={(e) => setReward('deathPenalty', e.target.value)} />
                    </Field>
                    <Field label="stepPenalty" htmlFor="cfg-sp">
                      <input id="cfg-sp" className="field-input" type="number"
                        value={params.reward.stepPenalty} step={0.01}
                        onChange={(e) => setReward('stepPenalty', e.target.value)} />
                    </Field>
                    <Field label="survivalReward" htmlFor="cfg-sr">
                      <input id="cfg-sr" className="field-input" type="number"
                        value={params.reward.survivalReward} step={0.01}
                        onChange={(e) => setReward('survivalReward', e.target.value)} />
                    </Field>
                    <Field label="reversePenalty" htmlFor="cfg-rp">
                      <input id="cfg-rp" className="field-input" type="number"
                        value={params.reward.reversePenalty} step={0.5}
                        onChange={(e) => setReward('reversePenalty', e.target.value)} />
                    </Field>
                  </div>
                </div>

                {/* ── Learning ──────────────────────────────── */}
                <div className="config-section">
                  <div className="section-heading">Learning</div>
                  <div className="field-grid">
                    <Field label="epsilon" unit="ε" htmlFor="cfg-eps">
                      <input id="cfg-eps" className="field-input" type="number"
                        value={agent.hyper.epsilon} min={0} max={1} step={0.01}
                        onChange={(e) => { agent.hyper.epsilon = safeNum(e.target.value, agent.hyper.epsilon); setTick((t) => t + 1); }} />
                    </Field>
                    <Field label="epsilonDecay" htmlFor="cfg-epsd">
                      <input id="cfg-epsd" className="field-input" type="number"
                        value={agent.hyper.epsilonDecay} min={0.9} max={1} step={0.0001}
                        onChange={(e) => { agent.hyper.epsilonDecay = safeNum(e.target.value, agent.hyper.epsilonDecay); setTick((t) => t + 1); }} />
                    </Field>
                    <Field label="alpha" unit="α" htmlFor="cfg-alpha">
                      <input id="cfg-alpha" className="field-input" type="number"
                        value={agent.hyper.alpha} min={0} max={1} step={0.01}
                        onChange={(e) => { agent.hyper.alpha = safeNum(e.target.value, agent.hyper.alpha); setTick((t) => t + 1); }} />
                    </Field>
                    <Field label="gamma" unit="γ" htmlFor="cfg-gamma">
                      <input id="cfg-gamma" className="field-input" type="number"
                        value={agent.hyper.gamma} min={0} max={1} step={0.01}
                        onChange={(e) => { agent.hyper.gamma = safeNum(e.target.value, agent.hyper.gamma); setTick((t) => t + 1); }} />
                    </Field>
                    <Field label="heatmapLearningRate" htmlFor="cfg-hlr">
                      <input id="cfg-hlr" className="field-input" type="number"
                        value={params.heatmapLearningRate} min={0.001} max={1} step={0.01}
                        onChange={(e) => setParams((p) => ({ ...p, heatmapLearningRate: safeNum(e.target.value, p.heatmapLearningRate) }))} />
                    </Field>
                    <Field label="heatmapDecayRate" htmlFor="cfg-hdr">
                      <input id="cfg-hdr" className="field-input" type="number"
                        value={params.heatmapDecayRate} min={0.9} max={1} step={0.001}
                        onChange={(e) => setParams((p) => ({ ...p, heatmapDecayRate: safeNum(e.target.value, p.heatmapDecayRate) }))} />
                    </Field>
                  </div>
                </div>

                {/* ── How-To Reference ──────────────────────── */}
                <div className="config-section howto-section">
                  <div className="howto-header">
                    <span className="howto-header-title">Variable Reference</span>
                    <span className="howto-header-sub">how each setting shapes agent behavior</span>
                  </div>

                  <div className="howto-group">
                    <div className="howto-group-label">
                      <span className="howto-group-bar" />
                      Rewards
                    </div>
                    <dl className="howto-list">
                      <div className="howto-item">
                        <dt>pelletReward</dt>
                        <dd>Points earned each time Pac-Man eats a regular dot. Raising this relative to other rewards trains the agent to prioritize pellet collection as its main objective. Too high and it may rush into danger; too low and it may ignore pellets entirely.</dd>
                      </div>
                      <div className="howto-item">
                        <dt>powerPelletReward</dt>
                        <dd>Points for eating a large power pellet. Increasing this relative to <code>pelletReward</code> teaches the agent to actively seek out power-ups before cleaning the board. Only relevant when power pellets are enabled.</dd>
                      </div>
                      <div className="howto-item">
                        <dt>ghostEatReward</dt>
                        <dd>Points for eating a frightened ghost while powered up. Very high values shift the entire strategy toward ghost-hunting — the agent will use power pellets aggressively and linger in dangerous zones to chase ghosts.</dd>
                      </div>
                      <div className="howto-item">
                        <dt>winBonus</dt>
                        <dd>One-time reward when every pellet is cleared. A large bonus pushes the agent toward completion rather than survival. Low values mean the agent treats clearing the board as just one of many possible goals.</dd>
                      </div>
                      <div className="howto-item">
                        <dt>deathPenalty</dt>
                        <dd>Negative reward applied when Pac-Man is caught by a ghost. More negative values create a more cautious, ghost-avoidant agent. If set too large, the agent may freeze in safe corners rather than collect pellets.</dd>
                      </div>
                      <div className="howto-item">
                        <dt>stepPenalty</dt>
                        <dd>Small negative reward applied on every step. Discourages aimless wandering by making episodes with fewer wasted moves more valuable. If too large, the agent rushes and takes unnecessary risks just to end the episode quickly.</dd>
                      </div>
                      <div className="howto-item">
                        <dt>survivalReward</dt>
                        <dd>Small positive reward added every step the agent stays alive. Counteracts <code>stepPenalty</code> and rewards longevity. The "survival" preset raises this to 0.2 — roughly 10× the default — making staying alive the dominant objective over collecting pellets.</dd>
                      </div>
                    </dl>
                  </div>

                  <div className="howto-group">
                    <div className="howto-group-label">
                      <span className="howto-group-bar" />
                      Learning
                    </div>
                    <dl className="howto-list">
                      <div className="howto-item">
                        <dt>epsilon <span className="howto-greek">ε</span></dt>
                        <dd>Exploration rate. At 1.0 the agent acts randomly (pure exploration); at 0.0 it always exploits its current best-known Q-values (pure exploitation). Training starts near 1.0 and decays over episodes as the agent builds knowledge.</dd>
                      </div>
                      <div className="howto-item">
                        <dt>epsilonDecay</dt>
                        <dd>Multiplied against <code>epsilon</code> at the end of every episode. Values close to 1.0 (e.g. 0.999) preserve exploration longer. Values further from 1.0 (e.g. 0.99) shift to exploitation faster — useful for simpler mazes where the agent learns quickly.</dd>
                      </div>
                      <div className="howto-item">
                        <dt>alpha <span className="howto-greek">α</span></dt>
                        <dd>Q-table learning rate. Controls how much each new experience updates stored values. High α (near 1.0) means recent experiences overwrite old ones rapidly, useful when the environment is changing. Low α produces slow, stable convergence.</dd>
                      </div>
                      <div className="howto-item">
                        <dt>gamma <span className="howto-greek">γ</span></dt>
                        <dd>Discount factor for future rewards. Values near 1.0 make the agent plan many steps ahead, valuing long-term outcomes. Lower values produce myopic behavior focused only on immediate rewards. At γ=0.95 the effective planning horizon is ~14 steps — too short for 1000-step episodes. 0.99 (horizon ~69 steps) is recommended.</dd>
                      </div>
                      <div className="howto-item">
                        <dt>heatmapLearningRate</dt>
                        <dd>How strongly each ghost visit marks a tile on the danger heatmap. Higher values create a more reactive but noisier map. Only affects agents using Heatmap or Hybrid ghost AI modes — has no effect on Classic.</dd>
                      </div>
                      <div className="howto-item">
                        <dt>heatmapDecayRate</dt>
                        <dd>Multiplier applied to every heatmap cell each step, causing old danger markings to fade. Values close to 1.0 give the map long memory. Lower values make the map forget danger zones quickly, suited to environments where ghosts move unpredictably.</dd>
                      </div>
                    </dl>
                  </div>
                </div>
              </>
            )}

            {/* RUNTIME TAB */}
            {activeTab === 'runtime' && (
              <div className="config-section">
                <div className="speed-row">
                  <div className="speed-row-label">Training speed</div>
                  <div className="segmented" role="group" aria-label="Training speed">
                    {trainingSpeedOptions.map((s) => (
                      <button
                        key={s}
                        className={`segmented-btn${trainingSpeed === s ? ' active' : ''}`}
                        onClick={() => updateTrainingSpeed(s)}
                        aria-pressed={trainingSpeed === s}
                      >
                        {s.charAt(0).toUpperCase() + s.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="field-grid">
                  <Field label="steps/frame" unit="int" htmlFor="cfg-spf">
                    <input id="cfg-spf" className="field-input" type="number"
                      value={stepsPerFrame} min={1} max={5000} step={1}
                      onChange={(e) => setStepsPerFrame((prev) => safeNum(e.target.value, prev))} />
                  </Field>
                  <Field label="renderEveryN" unit="int" htmlFor="cfg-ren">
                    <input id="cfg-ren" className="field-input" type="number"
                      value={renderEveryNSteps} min={1} max={1000} step={1}
                      onChange={(e) => setRenderEveryNSteps((prev) => safeNum(e.target.value, prev))} />
                  </Field>
                  <Field label="maxEpisodeSteps" unit="int" htmlFor="cfg-mes">
                    <input id="cfg-mes" className="field-input" type="number"
                      value={params.maxEpisodeSteps} min={20} max={10000} step={10}
                      onChange={(e) => setParams((p) => ({ ...p, maxEpisodeSteps: safeNum(e.target.value, p.maxEpisodeSteps) }))} />
                  </Field>
                  <Field label="seed" unit="int" htmlFor="cfg-seed">
                    <input id="cfg-seed" className="field-input" type="number"
                      value={seed} min={0} max={999999} step={1}
                      onChange={(e) => setSeed((prev) => safeNum(e.target.value, prev))} />
                  </Field>
                </div>
              </div>
            )}
          </div>

          {/* Sticky footer */}
          <div className="config-footer">
            <button className="footer-btn" onClick={() => {
              trainer.stop(); setIsTraining(false);
              agent.reset(); agent.hyper.epsilon = baseHyper.epsilon;
              trainer.resetStats(); env.reset(seed); trainer.setCurrentSeed(seed); setTick((t) => t + 1); // N18
            }}>Reset Q</button>
            <button className="footer-btn" onClick={savePolicy}>Save policy</button>
            <label className="footer-btn" style={{ cursor: 'pointer' }}>
              Load
              <input hidden type="file" accept="application/json" onChange={async (e) => {
                const file = e.target.files?.[0];
                if (!file) return;
                // D7.2: a malformed or non-policy file must not become an
                // unhandled promise rejection with no feedback. Parse, validate
                // the shape, and surface a clear error.
                try {
                  const parsed = JSON.parse(await file.text());
                  if (!parsed || typeof parsed !== 'object' || !('qTable' in parsed) || !('observationKeyVersion' in parsed)) {
                    throw new Error('Not a Q-learning policy file (missing qTable/observationKeyVersion).');
                  }
                  // Pass numGhosts so load() can detect a mismatch and refuse —
                  // otherwise the policy's observation-key encoding silently
                  // aliases unrelated states.
                  agent.load(parsed, params.numGhosts);
                  // N17: if the loaded policy was trained with a different numGhosts,
                  // load() would have discarded the Q-table (see mismatch guard) but
                  // params.numGhosts still disagrees with what the agent expects.
                  // Sync the UI to loadedNumGhosts so env, trainer, and agent agree.
                  const loadedN = agent.loadedNumGhosts;
                  if (loadedN !== null && loadedN !== params.numGhosts) {
                    setParams((p) => ({ ...p, numGhosts: loadedN }));
                  }
                  setTick((t) => t + 1);
                } catch (err) {
                  // eslint-disable-next-line no-alert
                  window.alert(`Failed to load policy: ${err instanceof Error ? err.message : String(err)}`);
                } finally {
                  e.target.value = ''; // allow re-selecting the same file after a fix
                }
              }} />
            </label>
          </div>
        </div>

        {/* ── Col 3: Telemetry ──────────────────────────── */}
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Telemetry</span>
            <div className="panel-header-spacer" />
            <div className="time-pills" role="group" aria-label="Time range">
              {([120, 500, 0] as const).map((r) => (
                <button
                  key={r}
                  className={`time-pill${timeRange === r ? ' active' : ''}`}
                  onClick={() => setTimeRange(r)}
                  aria-pressed={timeRange === r}
                >
                  {r === 0 ? 'All' : `${r} ep`}
                </button>
              ))}
            </div>
          </div>

          <div className="chart-stack">
            <ChartCard
              title="Episode Score"
              color="#22c55e"
              gradId="grad-score"
              values={chartSlice(scores)}
              value={scores.length > 0 ? fmtNum(scores[scores.length - 1], 0) : '—'}
            />
            <ChartCard
              title="Episode Length"
              color="#a78bfa"
              gradId="grad-length"
              values={chartSlice(lengths)}
              value={lengths.length > 0 ? fmtNum(lengths[lengths.length - 1], 0) : '—'}
            />
            <ChartCard
              title="Score Moving Avg (20 ep)"
              color="#3b82f6"
              gradId="grad-mavg"
              values={chartSlice(movAvg)}
              value={movAvg.length > 0 ? fmtNum(movAvg[movAvg.length - 1], 1) : '—'}
            />
            <ChartCard
              title="ε Exploration"
              color="#f59e0b"
              gradId="grad-eps"
              values={chartSlice(epsilons)}
              value={fmtNum(curEpsilon, 3)}
            />
          </div>
        </div>

      </main>
    </div>
  );
}
