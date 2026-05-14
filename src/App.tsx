import { useEffect, useMemo, useRef, useState } from 'react';
import { createDefaultEnv, type EnvParams } from './env/environment';
import { CanvasRenderer } from './render/canvasRenderer';
import { QLearningAgent } from './rl/qlearning';
import { TrainingController } from './rl/trainingController';
import { LineChart } from './ui/LineChart';
import { MAZES } from './mazes/mazes';
import type { GhostAIType } from './ghosts/ghostAi';

const baseHyper = { alpha: 0.2, gamma: 0.95, epsilon: 0.5, epsilonDecay: 0.999, epsilonMin: 0.05 };
const ghostTypes: GhostAIType[] = ['classic', 'heatmap', 'hybrid'];
// Slow/normal use frameIntervalMs for visible, human-observable training.
// Normal intentionally matches the AI-watch interval below; slow is about half of that speed.
// Fast/turbo batch more steps per browser frame, and max uses a time budget so the UI can still yield.
const trainingSpeedPresets = {
  slow: { stepsPerFrame: 1, turbo: false, renderEveryNSteps: 1, frameIntervalMs: 240, maxFrameMs: 0 },
  normal: { stepsPerFrame: 1, turbo: false, renderEveryNSteps: 1, frameIntervalMs: 120, maxFrameMs: 0 },
  fast: { stepsPerFrame: 20, turbo: false, renderEveryNSteps: 5, frameIntervalMs: 0, maxFrameMs: 0 },
  turbo: { stepsPerFrame: 100, turbo: true, renderEveryNSteps: 50, frameIntervalMs: 0, maxFrameMs: 0 },
  max: { stepsPerFrame: 1_000_000, turbo: false, renderEveryNSteps: 1000, frameIntervalMs: 0, maxFrameMs: 12 },
} as const;
type TrainingSpeed = keyof typeof trainingSpeedPresets;
const trainingSpeedOptions = Object.keys(trainingSpeedPresets) as TrainingSpeed[];

// Reward presets for different training objectives
const rewardPresetDescriptions: Record<string, string> = {
  default: 'Balanced baseline: collect pellets, avoid death, and finish the maze without over-favoring one tactic.',
  'ghost-hunting': 'Prioritizes eating edible ghosts after power pellets.',
  'pellet-collection': 'Strongly rewards clearing pellets and winning.',
  survival: 'Prioritizes staying alive and heavily penalizes deaths.',
};

const controlHelp = {
  speed: 'Normal runs one training step every 120 ms, matching paused AI-watch playback; Slow is one step every 240 ms. Fast/Turbo batch steps with less frequent rendering. Max uses a 12 ms frame budget for throughput.',
  training: 'steps/frame is how many environment steps training attempts per browser frame. renderEveryNSteps controls how often the canvas refreshes while batched training runs.',
  rewards: 'Rewards are added each step: pellet and power-pellet values encourage collection, deathPenalty discourages captures, stepPenalty/survivalReward shape episode length, ghostEatReward rewards eating edible ghosts, and winBonus rewards clearing the board.',
  hyper: 'epsilon is exploration probability; alpha is how strongly new rewards update Q-values; gamma is how much future reward matters. epsilonDecay lowers epsilon after each episode.',
};

const rewardPresets: Record<string, EnvParams['reward']> = {
  default: { pelletReward: 5, powerPelletReward: 20, deathPenalty: -100, stepPenalty: -0.1, survivalReward: 0.02, ghostEatReward: 30, winBonus: 200 },
  'ghost-hunting': { pelletReward: 1, powerPelletReward: 10, deathPenalty: -50, stepPenalty: -0.05, survivalReward: 0.01, ghostEatReward: 100, winBonus: 50 },
  'pellet-collection': { pelletReward: 50, powerPelletReward: 100, deathPenalty: -200, stepPenalty: -0.2, survivalReward: 0.01, ghostEatReward: 10, winBonus: 500 },
  'survival': { pelletReward: 1, powerPelletReward: 5, deathPenalty: -500, stepPenalty: 0, survivalReward: 1, ghostEatReward: 20, winBonus: 100 },
};

const numberInput = (value: number, onChange: (v: number) => void, min?: number, max?: number, step = 0.1): JSX.Element => (
  <input type="number" value={value} step={step} min={min} max={max} onChange={(e) => onChange(Number(e.target.value))} />
);

const VERSION = '1.2.1';

export default function App(): JSX.Element {
  const env = useMemo(() => createDefaultEnv(), []);
  const agent = useMemo(() => new QLearningAgent(baseHyper), []);
  const trainer = useMemo(() => new TrainingController(env, agent), [env, agent]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tick, setTick] = useState(0);
  const [seed, setSeed] = useState(42);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [stepsPerFrame, setStepsPerFrame] = useState<number>(trainingSpeedPresets.normal.stepsPerFrame);
  const [turbo, setTurbo] = useState<boolean>(trainingSpeedPresets.normal.turbo);
  const [renderEveryNSteps, setRenderEveryNSteps] = useState<number>(trainingSpeedPresets.normal.renderEveryNSteps);
  const [trainingFrameIntervalMs, setTrainingFrameIntervalMs] = useState<number>(trainingSpeedPresets.normal.frameIntervalMs);
  const [trainingMaxFrameMs, setTrainingMaxFrameMs] = useState<number>(trainingSpeedPresets.normal.maxFrameMs);
  const [mode, setMode] = useState<'human' | 'ai'>('ai');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingSpeed, setTrainingSpeed] = useState<TrainingSpeed>('normal');

  // Refs so training-loop lambdas always read the latest slider values (fixes stale-closure bug).
  const turboRef = useRef(turbo);
  turboRef.current = turbo;
  const stepsPerFrameRef = useRef(stepsPerFrame);
  stepsPerFrameRef.current = stepsPerFrame;
  const renderEveryNRef = useRef(renderEveryNSteps);
  renderEveryNRef.current = renderEveryNSteps;
  const trainingFrameIntervalMsRef = useRef(trainingFrameIntervalMs);
  trainingFrameIntervalMsRef.current = trainingFrameIntervalMs;
  const trainingMaxFrameMsRef = useRef(trainingMaxFrameMs);
  trainingMaxFrameMsRef.current = trainingMaxFrameMs;
  const [evalResult, setEvalResult] = useState('');
  const [params, setParams] = useState<EnvParams>(env.params);
  const [rewardPreset, setRewardPreset] = useState<string>('default');
  const [comparisonMode, setComparisonMode] = useState(false);
  const comparisonAgent = useMemo(() => new QLearningAgent(baseHyper), []);
  const comparisonTrainer = useMemo(() => new TrainingController(createDefaultEnv(), comparisonAgent), []);
  const lastStatsLengthRef = useRef(0);
  const [timeScale, setTimeScale] = useState<'recent' | 'full'>('recent');

  // Apply training speed presets
  const updateTrainingSpeed = (speed: TrainingSpeed): void => {
    const preset = trainingSpeedPresets[speed];
    setTrainingSpeed(speed);
    setStepsPerFrame(preset.stepsPerFrame);
    setTurbo(preset.turbo);
    setRenderEveryNSteps(preset.renderEveryNSteps);
    setTrainingFrameIntervalMs(preset.frameIntervalMs);
    setTrainingMaxFrameMs(preset.maxFrameMs);
  };

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    new CanvasRenderer(ctx).draw(env, showHeatmap);
  }, [env, tick, showHeatmap]);

  useEffect(() => {
    env.setParams(params);
    env.reset(seed);
    setTick((t) => t + 1);
  }, [params, seed, env]);

  useEffect(() => {
    // When training is active, the training RAF loop drives env.step — don't also run the AI interval.
    if (mode !== 'ai' || isTraining) return;
    // Stop any running training loop so it doesn't conflict with the AI-watch interval.
    trainer.stop();
    const id = setInterval(() => {
      const obs = env.observe();
      const action = agent.act(obs, env.getLegalActions().map((d) => ['up', 'down', 'left', 'right'].indexOf(d)), Math.random);
      const result = env.step(action);
      if (result.done) {
        env.reset(seed);
      }
      setTick((t) => t + 1);
    }, 120);
    return () => clearInterval(id);
  }, [mode, isTraining, env, agent, trainer, seed]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (mode !== 'human') return;
      const keyMap: Record<string, number> = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 };
      const action = keyMap[e.key];
      if (action === undefined) return;
      const result = env.step(action);
      if (result.done) {
        env.reset(seed);
      }
      setTick((t) => t + 1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [env, mode, seed]);

  const startTraining = (): void => {
    // Stop any existing loop before starting a new one (prevents duplicate RAF loops).
    trainer.stop();
    trainer.setSeed(seed);
    setIsTraining(true);
    lastStatsLengthRef.current = trainer.stats.episodeScores.length;
    trainer.start(
      () => (turboRef.current ? stepsPerFrameRef.current * 10 : stepsPerFrameRef.current),
      () => renderEveryNRef.current,
      () => {
        if (trainer.stats.episodeScores.length > lastStatsLengthRef.current) {
          lastStatsLengthRef.current = trainer.stats.episodeScores.length;
        }
        setTick((t) => t + 1);
      },
      {
        getFrameIntervalMs: () => trainingFrameIntervalMsRef.current,
        getMaxFrameMs: () => trainingMaxFrameMsRef.current,
      },
    );
  };

  const stopTraining = (): void => {
    trainer.stop();
    setIsTraining(false);
  };

  const savePolicy = (): void => {
    const blob = new Blob([JSON.stringify(agent.serialize(params.mazeId), null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `policy-${Date.now()}.json`;
    a.click();
  };

  const saveParams = (): void => {
    const data = { env: params, hyper: agent.hyper, seed };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `params-${Date.now()}.json`;
    a.click();
  };

  const setReward = (key: keyof EnvParams['reward'], value: number): void => {
    setRewardPreset('custom');
    setParams((p) => ({ ...p, reward: { ...p.reward, [key]: value } }));
  };

  const movingAverage = (values: number[], windowSize: number): number[] => values.map((_, i, a) => {
    const start = Math.max(0, i - windowSize + 1);
    const slice = a.slice(start, i + 1);
    return slice.reduce((x, y) => x + y, 0) / slice.length;
  });

  const chartSlice = (values: number[]): number[] => values.slice(timeScale === 'recent' ? -120 : 0);

  const controlButtonStyle = { padding: '4px 8px', fontSize: 12, minWidth: 82 };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12, padding: 12, color: '#e5e7eb', background: '#030712', minHeight: '100vh', fontFamily: 'sans-serif' }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
          <div>
            <h1 style={{ margin: 0 }}>AI Pac-Man Lab</h1>
            <div style={{ fontSize: 11, color: '#6b7280', marginTop: 2 }}>v{VERSION}</div>
          </div>
          {isTraining && (
            <span style={{ background: '#16a34a', color: '#fff', padding: '2px 8px', borderRadius: 4, fontSize: 12, fontWeight: 600 }}>
              ● TRAINING — episode {trainer.stats.episodeScores.length}
            </span>
          )}
        </div>
        {/* canvas is sized by the renderer; display:block removes inline-block gap */}
        <canvas ref={canvasRef} style={{ display: 'block', maxWidth: '100%', imageRendering: 'pixelated', border: '2px solid #1e3a8a' }} />
        <p style={{ margin: '6px 0', fontSize: 13 }}>
          Score: <strong>{env.getPacmen()[0].score}</strong> | Pellets left: <strong>{env.pelletsLeft}</strong> | Step: <strong>{env.stepCount}</strong>
          {mode === 'human' && <span style={{ marginLeft: 12, color: '#9ca3af' }}>(arrow keys to move)</span>}
        </p>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: '95vh', overflow: 'auto' }}>
        <label>Mode <select value={mode} onChange={(e) => setMode(e.target.value as 'human' | 'ai')}><option value="human">Human</option><option value="ai">AI controlled</option></select></label>
        <label>Maze <select value={params.mazeId} onChange={(e) => setParams((p) => ({ ...p, mazeId: e.target.value }))}>{MAZES.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select></label>
        <label>numGhosts {numberInput(params.numGhosts, (v) => setParams((p) => ({ ...p, numGhosts: v })), 1, 6, 1)}</label>
        <label>numPacmen {numberInput(params.numPacmen, (v) => setParams((p) => ({ ...p, numPacmen: v })), 1, 4, 1)}</label>
        {env.ghosts.map((g, i) => <label key={g.id}>ghost {i} AI <select value={g.aiType} onChange={(e) => env.setGhostType(i, e.target.value as GhostAIType)}>{ghostTypes.map((t) => <option key={t}>{t}</option>)}</select></label>)}
        <label>captureRules <select value={params.captureRules} onChange={(e) => setParams((p) => ({ ...p, captureRules: e.target.value as 'touch' | 'tile' }))}><option value="touch">touch</option><option value="tile">tile</option></select></label>
        <label><input type="checkbox" checked={params.cooperativePacmen} onChange={(e) => setParams((p) => ({ ...p, cooperativePacmen: e.target.checked }))} /> cooperative clones</label>
        <label>heatmapDecayRate {numberInput(params.heatmapDecayRate, (v) => setParams((p) => ({ ...p, heatmapDecayRate: v })), 0.9, 1, 0.001)}</label>
        <label>heatmapLearningRate {numberInput(params.heatmapLearningRate, (v) => setParams((p) => ({ ...p, heatmapLearningRate: v })), 0.001, 1, 0.01)}</label>
        <label>maxEpisodeSteps {numberInput(params.maxEpisodeSteps, (v) => setParams((p) => ({ ...p, maxEpisodeSteps: v })), 20, 10000, 10)}</label>
        <label>pelletDensity {numberInput(params.pelletDensity, (v) => setParams((p) => ({ ...p, pelletDensity: v })), 0.1, 1, 0.05)}</label>
        <label>ghostSpeed {numberInput(params.ghostSpeed, (v) => setParams((p) => ({ ...p, ghostSpeed: v })), 0.2, 3, 0.1)}</label>
        <label>pacmanSpeed {numberInput(params.pacmanSpeed, (v) => setParams((p) => ({ ...p, pacmanSpeed: v })), 0.2, 3, 0.1)}</label>
        <label><input type="checkbox" checked={params.enablePowerPellets} onChange={(e) => setParams((p) => ({ ...p, enablePowerPellets: e.target.checked }))} /> enablePowerPellets</label>
        <label>powerPelletDuration {numberInput(params.powerPelletDuration, (v) => setParams((p) => ({ ...p, powerPelletDuration: v })), 1, 200, 1)}</label>
        <label>Reward preset <select value={rewardPreset} onChange={(e) => {
          const preset = e.target.value;
          setRewardPreset(preset);
          if (preset in rewardPresets) {
            setParams((p) => ({ ...p, reward: rewardPresets[preset] }));
          }
        }}>
          {rewardPreset === 'custom' && <option value="custom">custom</option>}
          {Object.keys(rewardPresets).map((p) => <option key={p} value={p}>{p}</option>)}
        </select></label>
        <small style={{ color: '#9ca3af', lineHeight: 1.35 }}>{rewardPresetDescriptions[rewardPreset] ?? 'Custom reward values. Adjust the fields below to shape what the agent learns.'}</small>
        <label>pelletReward {numberInput(params.reward.pelletReward, (v) => setReward('pelletReward', v), -100, 200, 1)}</label>
        <label>powerPelletReward {numberInput(params.reward.powerPelletReward, (v) => setReward('powerPelletReward', v), -100, 500, 1)}</label>
        <label>deathPenalty {numberInput(params.reward.deathPenalty, (v) => setReward('deathPenalty', v), -500, 0, 1)}</label>
        <label>stepPenalty {numberInput(params.reward.stepPenalty, (v) => setReward('stepPenalty', v), -10, 10, 0.1)}</label>
        <label>survivalReward {numberInput(params.reward.survivalReward, (v) => setReward('survivalReward', v), -10, 10, 0.1)}</label>
        <label>ghostEatReward {numberInput(params.reward.ghostEatReward, (v) => setReward('ghostEatReward', v), -100, 200, 1)}</label>
        <label>winBonus {numberInput(params.reward.winBonus, (v) => setReward('winBonus', v), 0, 1000, 10)}</label>
        <label>seed {numberInput(seed, setSeed, 0, 999999, 1)}</label>
        <small style={{ color: '#9ca3af', lineHeight: 1.35 }}>{controlHelp.rewards}</small>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: 12 }}>Training speed:</span>
          {trainingSpeedOptions.map((s) => (
            <button
              key={s}
              onClick={() => updateTrainingSpeed(s)}
              style={{
                padding: '2px 8px',
                fontSize: 11,
                background: trainingSpeed === s ? '#22c55e' : '#374151',
                color: '#fff',
                border: 'none',
                borderRadius: 3,
                cursor: 'pointer',
                textTransform: 'capitalize',
              }}
            >
              {s}
            </button>
          ))}
        </div>
        <small style={{ color: '#9ca3af', lineHeight: 1.35 }}>{controlHelp.speed}</small>
        <label>steps/frame {numberInput(stepsPerFrame, setStepsPerFrame, 1, 5000, 1)}</label>
        <label>renderEveryNSteps {numberInput(renderEveryNSteps, setRenderEveryNSteps, 1, 1000, 1)}</label>
        <label><input type="checkbox" checked={turbo} onChange={(e) => setTurbo(e.target.checked)} /> turbo</label>
        <small style={{ color: '#9ca3af', lineHeight: 1.35 }}>{controlHelp.training}</small>
        <label><input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} /> show ghost heatmap</label>
        <label>epsilon {numberInput(agent.hyper.epsilon, (v) => { agent.hyper.epsilon = v; setTick((t) => t + 1); }, 0, 1, 0.01)}</label>
        <label>alpha {numberInput(agent.hyper.alpha, (v) => { agent.hyper.alpha = v; setTick((t) => t + 1); }, 0, 1, 0.01)}</label>
        <label>gamma {numberInput(agent.hyper.gamma, (v) => { agent.hyper.gamma = v; setTick((t) => t + 1); }, 0, 1, 0.01)}</label>
        <label>epsilonDecay {numberInput(agent.hyper.epsilonDecay, (v) => { agent.hyper.epsilonDecay = v; setTick((t) => t + 1); }, 0.9, 1, 0.0001)}</label>
        <small style={{ color: '#9ca3af', lineHeight: 1.35 }}>{controlHelp.hyper}</small>
        <label><input type="checkbox" checked={comparisonMode} onChange={(e) => setComparisonMode(e.target.checked)} /> A-B comparison mode</label>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(82px, 1fr))', gap: 6 }}>
          <button style={controlButtonStyle} onClick={() => { trainer.stop(); setIsTraining(false); env.reset(seed); setTick((t) => t + 1); }}>Reset</button>
          <button style={controlButtonStyle} onClick={startTraining}>Start training</button>
          <button style={controlButtonStyle} onClick={stopTraining}>Pause</button>
          <button style={controlButtonStyle} onClick={() => { trainer.singleStep(); setTick((t) => t + 1); }}>Single step</button>
          <button style={controlButtonStyle} onClick={() => { trainer.stop(); setIsTraining(false); const r = trainer.evaluate(20); setEvalResult(`avgScore=${r.avgScore.toFixed(1)}, avgLength=${r.avgLength.toFixed(1)}, winRate=${(r.winRate * 100).toFixed(1)}%`); env.reset(seed); setTick((t) => t + 1); }}>Evaluate</button>
          <button style={controlButtonStyle} onClick={() => { agent.reset(); agent.hyper.epsilon = baseHyper.epsilon; setTick((t) => t + 1); }}>Reset Q</button>
          <button style={controlButtonStyle} onClick={savePolicy}>Save policy</button>
          <button style={controlButtonStyle} onClick={saveParams}>Save params</button>
          <label style={{ ...controlButtonStyle, border: '1px solid #374151', cursor: 'pointer', textAlign: 'center' }}>Load policy<input hidden type="file" accept="application/json" onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            agent.load(JSON.parse(await file.text()));
          }} /></label>
          {comparisonMode && (
            <label style={{ ...controlButtonStyle, border: '1px solid #a78bfa', cursor: 'pointer', textAlign: 'center' }}>Load comparison policy<input hidden type="file" accept="application/json" onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              comparisonAgent.load(JSON.parse(await file.text()));
              setTick((t) => t + 1);
            }} /></label>
          )}
        </div>
        <small>{evalResult}</small>
      </div>
      <div style={{ gridColumn: '1 / -1', borderTop: '1px solid #374151', paddingTop: 12, marginTop: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#d1d5db' }}>Training History</div>
          <div style={{ fontSize: 12, color: '#9ca3af' }}>
            {trainer.stats.episodeScores.length} episodes
            {trainer.stats.episodeScores.length > 0 && (
              <span style={{ marginLeft: 12, color: '#60a5fa', fontWeight: 500 }}>
                Avg: {movingAverage(trainer.stats.episodeScores, 20)[trainer.stats.episodeScores.length - 1]?.toFixed(1) ?? '—'}
              </span>
            )}
            {comparisonMode && <span style={{ marginLeft: 8, color: '#a78bfa' }}>/ B: {comparisonTrainer.stats.episodeScores.length}</span>}
          </div>
          {!comparisonMode && (
            <div style={{ display: 'flex', gap: 4 }}>
              <button
                onClick={() => setTimeScale('recent')}
                style={{ padding: '4px 12px', fontSize: 11, background: timeScale === 'recent' ? '#22c55e' : '#374151', color: '#fff', border: 'none', borderRadius: 3, cursor: 'pointer' }}
              >
                Last 120 episodes
              </button>
              <button
                onClick={() => setTimeScale('full')}
                style={{ padding: '4px 12px', fontSize: 11, background: timeScale === 'full' ? '#22c55e' : '#374151', color: '#fff', border: 'none', borderRadius: 3, cursor: 'pointer' }}
              >
                Full history
              </button>
            </div>
          )}
        </div>
        {comparisonMode ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#22c55e', marginBottom: 4 }}>Policy A</div>
              <LineChart values={trainer.stats.episodeScores.slice(-120)} height={160} color="#22c55e" label="Episode Score" xLabel="Episode" yLabel="Score" />
              <LineChart
                values={movingAverage(trainer.stats.episodeScores, 20).slice(-120)}
                height={160} color="#60a5fa" label="Moving Avg Score" xLabel="Episode" yLabel="Score"
              />
            </div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#a78bfa', marginBottom: 4 }}>Policy B</div>
              <LineChart values={comparisonTrainer.stats.episodeScores.slice(-120)} height={160} color="#a78bfa" label="Episode Score" xLabel="Episode" yLabel="Score" />
              <LineChart
                values={movingAverage(comparisonTrainer.stats.episodeScores, 20).slice(-120)}
                height={160} color="#c084fc" label="Moving Avg Score" xLabel="Episode" yLabel="Score"
              />
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16, width: '100%' }}>
            <LineChart
              values={chartSlice(trainer.stats.episodeScores)}
              height={200} color="#22c55e" label="Episode Score" includeZero xLabel="Episode" yLabel="Score"
            />
            <LineChart
              values={chartSlice(trainer.stats.episodeLengths)}
              height={200} color="#a78bfa" label="Episode Length" yMin={0} xLabel="Episode" yLabel="Steps"
            />
            <LineChart
              values={chartSlice(movingAverage(trainer.stats.episodeScores, 20))}
              height={200} color="#60a5fa" label="Moving Average Score (20 episode window)" xLabel="Episode" yLabel="Score"
            />
            <LineChart
              values={chartSlice(trainer.stats.epsilons)}
              height={200} color="#f59e0b" label="Epsilon (Exploration Rate)" yMin={0} yMax={1} xLabel="Episode" yLabel="Epsilon"
            />
          </div>
        )}
      </div>
    </div>
  );
}
