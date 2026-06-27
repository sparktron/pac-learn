import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { type EnvParams } from './env/environment';
import { useGameEnv } from './hooks/useGameEnv';
import { useTrainingLoop } from './hooks/useTrainingLoop';
import { directionToAction, type Action } from './engine/types';
import { SeededRng } from './engine/prng';
import { CanvasRenderer } from './render/canvasRenderer';
import { QLearningAgent, type SerializedPolicy } from './rl/qlearning';
import { LinearQLearningAgent, type SerializedLinearPolicy } from './rl/linearQlearning';
import { TrainingController } from './rl/trainingController';
import type { GhostAIType } from './ghosts/ghostAi';
import { safeNum, safeLocalGet, safeLocalSet } from './uiHelpers';
import { TelemetryPanel } from './components/TelemetryPanel';
import { EnvironmentPanel } from './components/EnvironmentPanel';
import { ConfigurationPanel, type Algorithm } from './components/ConfigurationPanel';
import { TopBar } from './components/TopBar';

const VERSION = '1.2.1';

// epsilonDecay=0.999997 keeps exploration alive until ~400k episodes (old 0.999
// decayed to floor in ~1600 episodes — far too early for the Q-table to converge).
// epsilonMin=0.20 maintains enough randomness to keep discovering new states.
// Defaults mirror the empirically-tuned overnight-bench config (the "winning"
// setup): alpha 0.1 (sweep-03: 2.7× better than 0.2) and the endgame ε floor
// (0.25 when in the late-game pellet buckets ≤1). Keeps GUI training consistent
// with headless bench/sweep runs. See scripts/overnight-bench.ts.
const baseHyper = { alpha: 0.1, gamma: 0.99, epsilon: 0.5, epsilonDecay: 0.999997, epsilonMin: 0.20, endgameEpsilon: 0.25, endgameBucketThreshold: 1 };

// Training-speed presets + the loop live in hooks/useTrainingLoop.ts; reward
// presets in rl/rewardPresets.ts (D5.11). The Toggle/Field controls + the three
// panels now live under components/ (A5 slices 4a–4c).

// ── Main App ───────────────────────────────────────────────

export default function App(): JSX.Element {
  // Slice 1 (A5): env + editable params + live-apply now live in useGameEnv.
  const { env, params, setParams, rewardPreset, setRewardPreset } = useGameEnv();
  // D7.8: the algorithm selector chooses which agent backs training. Switching
  // rebuilds the agent (and trainer) — handled via changeAlgorithm so the old
  // training loop is stopped first.
  const [algorithm, setAlgorithm] = useState<Algorithm>(
    () => (safeLocalGet('pac-learn-algorithm') === 'linear' ? 'linear' : 'tabular'),
  );
  const agent = useMemo(
    () => (algorithm === 'linear' ? new LinearQLearningAgent(baseHyper) : new QLearningAgent(baseHyper)),
    [algorithm],
  );
  const trainer = useMemo(() => new TrainingController(env, agent), [env, agent]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  // N10: hold a ref to the maze body so fullscreen doesn't depend on a
  // brittle `.maze-body` querySelector that breaks if a second element ever
  // gets the same class (e.g. an A/B compare panel).
  const mazeBodyRef = useRef<HTMLDivElement>(null);
  const [tick, setTick] = useState(0);
  // Stable render-bump for the training loop (its structural-reset effect lists
  // requestRender as a dep — a fresh closure each render would churn it).
  const requestRender = useCallback(() => setTick((t) => t + 1), []);
  const [seed, setSeed] = useState(7); // match the bench default seed for GUI/headless parity
  const [viewMode, setViewMode] = useState<'live' | 'heatmap' | 'qvalues'>('live');
  const [mode, setMode] = useState<'human' | 'ai'>('ai');
  const [ghostAIType, setGhostAIType]   = useState<GhostAIType>('classic');
  const [timeRange, setTimeRange]       = useState<120 | 500 | 0>(120);
  const [activeTab, setActiveTab]       = useState<'environment' | 'tuning' | 'runtime'>(() => {
    const s = safeLocalGet('pac-learn-tab'); // D7.10: guarded — never throws on mount
    if (s === 'rewards' || s === 'learning') return 'tuning';
    return (s as 'environment' | 'tuning' | 'runtime') ?? 'environment';
  });

  // Persist active tab + algorithm (D7.10: guarded writes)
  useEffect(() => { safeLocalSet('pac-learn-tab', activeTab); }, [activeTab]);
  useEffect(() => { safeLocalSet('pac-learn-algorithm', algorithm); }, [algorithm]);

  // Draw canvas. Persist the renderer instance so its frame counter,
  // hash-skip cache, and computed tile size survive across redraws —
  // otherwise the mouth/pulse animations reset every render, the hash-
  // skip early-return is dead code, and `tile` is recomputed from
  // parentElement.clientWidth on every tick (visible snap on layout).
  // Q-value overlay: for each open tile, what max-Q the agent assigns to being
  // there in the current game state (ghosts/pellets fixed, pac moved). null for
  // walls and never-visited states. Recomputed on tick while the view is active.
  const qOverlay = useMemo<(number | null)[][] | undefined>(() => {
    if (viewMode !== 'qvalues') return undefined;
    const { width, height, isWall } = env.world;
    const grid: (number | null)[][] = [];
    for (let y = 0; y < height; y += 1) {
      const row: (number | null)[] = [];
      for (let x = 0; x < width; x += 1) {
        row.push(isWall(x, y) ? null : agent.peekMaxQ(env.observeAt({ x, y })));
      }
      grid.push(row);
    }
    return grid;
    // `tick` is an intentional recompute trigger (the Q-table mutates in place
    // each training step); it isn't read in the body, so the rule flags it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode, tick, env, agent]);

  const rendererRef = useRef<{ canvas: HTMLCanvasElement; renderer: CanvasRenderer } | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (!rendererRef.current || rendererRef.current.canvas !== canvas) {
      rendererRef.current = { canvas, renderer: new CanvasRenderer(ctx) };
    }
    rendererRef.current.renderer.draw(env, viewMode === 'heatmap', qOverlay);
  }, [env, tick, viewMode, qOverlay]);

  // N6: live-apply of params now lives in useGameEnv (env.setParams without a
  // reset). The *structural* reset below fires only when a structural field
  // changes (mazeId / numGhosts / seed) — those genuinely require a fresh env.
  // Training is paused across the reset so a Q-update can't bridge the boundary
  // (its obs is pre-reset and its nextObs is post-reset → garbage Q-values).

  // Keep env.heatmapEnabled in sync with the UI overlay so N2's fast-path
  // (skip heatmap decay when nobody consumes it) doesn't freeze the
  // heatmap view at startup zeros when all ghosts are classic.
  useEffect(() => { env.heatmapEnabled = viewMode === 'heatmap'; }, [env, viewMode]);

  // Slice 3 (A5): the training loop (isTraining, speed presets, start/stop, the
  // Space toggle, and the structural-reset effect) lives in useTrainingLoop.
  // Mounted here — after the env's renderer/heatmap effects — so the structural
  // reset still runs before the ghost-AI re-apply below (env.reset rebuilds
  // ghosts as 'classic'; the ghost-AI effect then restores the selected type).
  const {
    isTraining, startTraining, stopTraining, haltAndResetStats,
    trainingSpeed, updateTrainingSpeed,
    stepsPerFrame, setStepsPerFrame,
    renderEveryNSteps, setRenderEveryNSteps,
  } = useTrainingLoop({
    env, agent, trainer, seed,
    numGhosts: params.numGhosts, mazeId: params.mazeId, requestRender,
  });

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
      const action = agent.act(obs, env.getLegalActionIndices(), () => watchRng.next());
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
      // Map arrow keys to actions via directionToAction (the single source of
      // truth) rather than hand-numbering — so a DIRECTIONS reorder can't
      // silently scramble the controls.
      const keyMap: Record<string, Action> = {
        ArrowUp: directionToAction('up'),
        ArrowDown: directionToAction('down'),
        ArrowLeft: directionToAction('left'),
        ArrowRight: directionToAction('right'),
      };
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
      haltAndResetStats();
      agent.reset();
      agent.hyper.epsilon = baseHyper.epsilon;
    }
    setParams((p) => ({ ...p, numGhosts: next }));
  };

  // D7.8: switching algorithm rebuilds agent + trainer (useMemo). Stop the
  // current loop and clear stats FIRST so the old trainer's rAF loop can't keep
  // stepping the shared env against a now-stale agent.
  const changeAlgorithm = (next: Algorithm): void => {
    if (next === algorithm) return;
    haltAndResetStats();
    setAlgorithm(next);
  };

  // D7.9: trigger a JSON download and immediately revoke the object URL so
  // repeated saves don't leak blobs for the session's lifetime.
  const downloadJson = (filename: string, data: unknown): void => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const savePolicy = (): void => {
    // serialize() prefers agent.trainedNumGhosts when set, so the fallback
    // arg only matters for a fresh-never-trained save. Works for both agents.
    downloadJson(`policy-${Date.now()}.json`, agent.serialize(params.mazeId, params.numGhosts));
  };

  const saveParams = (): void => {
    downloadJson(`params-${Date.now()}.json`, { env: params, hyper: agent.hyper, seed, algorithm });
  };

  // Topbar Reset: halt training + stats, reseed env/trainer, repaint (N18).
  // resetQ does the same plus an agent wipe (kept inline to preserve its
  // halt → agent.reset → env.reset order exactly).
  const resetEnv = (): void => {
    haltAndResetStats();
    env.reset(seed); trainer.setCurrentSeed(seed); requestRender(); // N18
  };

  const resetQ = (): void => {
    haltAndResetStats();
    agent.reset(); agent.hyper.epsilon = baseHyper.epsilon;
    env.reset(seed); trainer.setCurrentSeed(seed); requestRender(); // N18
  };

  // Load a policy file. D7.2: a malformed or non-policy file must surface a clear
  // error, not an unhandled rejection. D7.8: the format depends on the active
  // algorithm — detect a mismatch up front rather than silently no-op'ing.
  const loadPolicy = async (file: File): Promise<void> => {
    try {
      const parsed = JSON.parse(await file.text());
      if (!parsed || typeof parsed !== 'object') throw new Error('Not a JSON object.');
      // Pass numGhosts so load() can also refuse a ghost-count mismatch.
      if (agent instanceof LinearQLearningAgent) {
        if (parsed.algorithm !== 'linear-qlearning' || !('weights' in parsed)) {
          throw new Error('Not a linear policy. Switch Algorithm to "Tabular Q" to load a Q-table policy.');
        }
        agent.load(parsed as SerializedLinearPolicy, params.numGhosts);
      } else {
        if (!('qTable' in parsed) || !('observationKeyVersion' in parsed)) {
          throw new Error('Not a Q-table policy. Switch Algorithm to "Linear FA" to load a linear policy.');
        }
        agent.load(parsed as SerializedPolicy, params.numGhosts);
      }
      // N17: a numGhosts-mismatched policy is discarded by load(); sync the UI to
      // loadedNumGhosts so env, trainer, and agent agree.
      const loadedN = agent.loadedNumGhosts;
      if (loadedN !== null && loadedN !== params.numGhosts) {
        setParams((p) => ({ ...p, numGhosts: loadedN }));
      }
      requestRender();
    } catch (err) {
      // eslint-disable-next-line no-alert
      window.alert(`Failed to load policy: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // A2: set ghost i's personality override. 'auto' clears the slot so the env
  // falls back to id % 4 for that ghost.
  const setGhostPersonality = (i: number, value: string): void => {
    setParams((p) => {
      const arr = p.ghostPersonalities.slice();
      while (arr.length <= i) arr.push(undefined); // keep dense up to i
      arr[i] = value === 'auto' ? undefined : Number(value);
      return { ...p, ghostPersonalities: arr };
    });
  };

  const setReward = (key: keyof EnvParams['reward'], rawValue: string): void => {
    setRewardPreset('custom');
    setParams((p) => ({ ...p, reward: { ...p.reward, [key]: safeNum(rawValue, p.reward[key]) } }));
  };

  // Derived chart data. The moving-average + time-window slicing now live in
  // TelemetryPanel (A5 slice 4a); App keeps `scores` for the topbar stats below.
  const scores   = trainer.stats.episodeScores;
  const lengths  = trainer.stats.episodeLengths;
  const epsilons = trainer.stats.epsilons;

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
  // A1: ghost AI alternates scatter (flee to corners) / chase (hunt Pac). Read
  // at render time; re-reads on every `tick` (the only time the phase changes).
  const scatterPhase = env.isScatterPhase();

  // ── JSX ───────────────────────────────────────────────────
  return (
    <div className="app-layout">

      {/* ── Top Bar ──────────────────────────────────────── */}
      <TopBar
        version={VERSION}
        isTraining={isTraining}
        episodeCount={episodeCount}
        avgScore={avgScore}
        bestScore={bestScore}
        curEpsilon={curEpsilon}
        onReset={resetEnv}
        onToggleTraining={() => (isTraining ? stopTraining() : startTraining())}
      />

      {/* ── Main Grid ────────────────────────────────────── */}
      <main className="main-grid">

        {/* ── Col 1: Maze ──────────────────────────────── */}
        <EnvironmentPanel
          canvasRef={canvasRef}
          mazeBodyRef={mazeBodyRef}
          env={env}
          viewMode={viewMode}
          setViewMode={setViewMode}
          episodeCount={episodeCount}
          scatterPhase={scatterPhase}
          numGhosts={params.numGhosts}
          maxEpisodeSteps={params.maxEpisodeSteps}
          pacScore={pacman?.score ?? 0}
          ghostsEatenCombo={pacman?.ghostsEatenCombo ?? 0}
        />

        {/* ── Col 2: Configuration ──────────────────────── */}
        <ConfigurationPanel
          rewardPreset={rewardPreset}
          onSaveParams={saveParams}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          mode={mode}
          setMode={setMode}
          algorithm={algorithm}
          changeAlgorithm={changeAlgorithm}
          params={params}
          setParams={setParams}
          changeNumGhosts={changeNumGhosts}
          ghostAIType={ghostAIType}
          setGhostAIType={setGhostAIType}
          viewMode={viewMode}
          setViewMode={setViewMode}
          setGhostPersonality={setGhostPersonality}
          setRewardPreset={setRewardPreset}
          setReward={setReward}
          agent={agent}
          requestRender={requestRender}
          trainingSpeed={trainingSpeed}
          updateTrainingSpeed={updateTrainingSpeed}
          stepsPerFrame={stepsPerFrame}
          setStepsPerFrame={setStepsPerFrame}
          renderEveryNSteps={renderEveryNSteps}
          setRenderEveryNSteps={setRenderEveryNSteps}
          seed={seed}
          setSeed={setSeed}
          onResetQ={resetQ}
          onSavePolicy={savePolicy}
          onLoadPolicy={loadPolicy}
        />

        {/* ── Col 3: Telemetry ──────────────────────────── */}
        <TelemetryPanel
          scores={scores}
          lengths={lengths}
          epsilons={epsilons}
          curEpsilon={curEpsilon}
          timeRange={timeRange}
          setTimeRange={setTimeRange}
        />

      </main>
    </div>
  );
}
