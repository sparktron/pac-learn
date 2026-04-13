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

const numberInput = (value: number, onChange: (v: number) => void, min?: number, max?: number, step = 0.1): JSX.Element => (
  <input type="number" value={value} step={step} min={min} max={max} onChange={(e) => onChange(Number(e.target.value))} />
);

export default function App(): JSX.Element {
  const env = useMemo(() => createDefaultEnv(), []);
  const agent = useMemo(() => new QLearningAgent(baseHyper), []);
  const trainer = useMemo(() => new TrainingController(env, agent), [env, agent]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tick, setTick] = useState(0);
  const [seed, setSeed] = useState(42);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [stepsPerFrame, setStepsPerFrame] = useState(20);
  const [turbo, setTurbo] = useState(false);
  const [renderEveryNSteps, setRenderEveryNSteps] = useState(10);
  const [mode, setMode] = useState<'human' | 'ai'>('human');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingSpeed, setTrainingSpeed] = useState<'slow' | 'normal' | 'fast' | 'turbo'>('normal');

  // Refs so training-loop lambdas always read the latest slider values (fixes stale-closure bug).
  const turboRef = useRef(turbo);
  turboRef.current = turbo;
  const stepsPerFrameRef = useRef(stepsPerFrame);
  stepsPerFrameRef.current = stepsPerFrame;
  const renderEveryNRef = useRef(renderEveryNSteps);
  renderEveryNRef.current = renderEveryNSteps;
  const [evalResult, setEvalResult] = useState('');
  const [params, setParams] = useState<EnvParams>(env.params);

  // Apply training speed presets
  const updateTrainingSpeed = (speed: 'slow' | 'normal' | 'fast' | 'turbo'): void => {
    setTrainingSpeed(speed);
    switch (speed) {
      case 'slow': setStepsPerFrame(10); setTurbo(false); setRenderEveryNSteps(1); break;
      case 'normal': setStepsPerFrame(20); setTurbo(false); setRenderEveryNSteps(10); break;
      case 'fast': setStepsPerFrame(100); setTurbo(false); setRenderEveryNSteps(50); break;
      case 'turbo': setStepsPerFrame(200); setTurbo(true); setRenderEveryNSteps(100); break;
    }
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
    if (mode !== 'ai') return;
    // Stop any running training loop so it doesn't conflict with the AI-watch interval.
    trainer.stop();
    setIsTraining(false);
    const id = setInterval(() => {
      const obs = env.observe();
      const action = agent.act(obs, env.getLegalActions().map((d) => ['up', 'down', 'left', 'right'].indexOf(d)), Math.random);
      env.step(action);
      setTick((t) => t + 1);
    }, 120);
    return () => clearInterval(id);
  }, [mode, env, agent, trainer]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (mode !== 'human') return;
      const keyMap: Record<string, number> = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 };
      const action = keyMap[e.key];
      if (action === undefined) return;
      env.step(action);
      setTick((t) => t + 1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [env, mode]);

  const startTraining = (): void => {
    // Stop any existing loop before starting a new one (prevents duplicate RAF loops).
    trainer.stop();
    trainer.setSeed(seed);
    setIsTraining(true);
    trainer.start(
      () => (turboRef.current ? stepsPerFrameRef.current * 10 : stepsPerFrameRef.current),
      () => renderEveryNRef.current,
      () => setTick((t) => t + 1),
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

  const setReward = (key: keyof EnvParams['reward'], value: number): void => setParams((p) => ({ ...p, reward: { ...p.reward, [key]: value } }));

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12, padding: 12, color: '#e5e7eb', background: '#030712', minHeight: '100vh', fontFamily: 'sans-serif' }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
          <h1 style={{ margin: 0 }}>AI Pac-Man Lab</h1>
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
        <label>cooperative clones <input type="checkbox" checked={params.cooperativePacmen} onChange={(e) => setParams((p) => ({ ...p, cooperativePacmen: e.target.checked }))} /></label>
        <label>heatmapDecayRate {numberInput(params.heatmapDecayRate, (v) => setParams((p) => ({ ...p, heatmapDecayRate: v })), 0.9, 1, 0.001)}</label>
        <label>heatmapLearningRate {numberInput(params.heatmapLearningRate, (v) => setParams((p) => ({ ...p, heatmapLearningRate: v })), 0.001, 1, 0.01)}</label>
        <label>maxEpisodeSteps {numberInput(params.maxEpisodeSteps, (v) => setParams((p) => ({ ...p, maxEpisodeSteps: v })), 20, 10000, 10)}</label>
        <label>pelletDensity {numberInput(params.pelletDensity, (v) => setParams((p) => ({ ...p, pelletDensity: v })), 0.1, 1, 0.05)}</label>
        <label>ghostSpeed {numberInput(params.ghostSpeed, (v) => setParams((p) => ({ ...p, ghostSpeed: v })), 0.2, 3, 0.1)}</label>
        <label>pacmanSpeed {numberInput(params.pacmanSpeed, (v) => setParams((p) => ({ ...p, pacmanSpeed: v })), 0.2, 3, 0.1)}</label>
        <label><input type="checkbox" checked={params.enablePowerPellets} onChange={(e) => setParams((p) => ({ ...p, enablePowerPellets: e.target.checked }))} /> enablePowerPellets</label>
        <label>powerPelletDuration {numberInput(params.powerPelletDuration, (v) => setParams((p) => ({ ...p, powerPelletDuration: v })), 1, 200, 1)}</label>
        <label>pelletReward {numberInput(params.reward.pelletReward, (v) => setReward('pelletReward', v), -100, 200, 1)}</label>
        <label>deathPenalty {numberInput(params.reward.deathPenalty, (v) => setReward('deathPenalty', v), -500, 0, 1)}</label>
        <label>stepPenalty {numberInput(params.reward.stepPenalty, (v) => setReward('stepPenalty', v), -10, 10, 0.1)}</label>
        <label>survivalReward {numberInput(params.reward.survivalReward, (v) => setReward('survivalReward', v), -10, 10, 0.1)}</label>
        <label>ghostEatReward {numberInput(params.reward.ghostEatReward, (v) => setReward('ghostEatReward', v), -100, 200, 1)}</label>
        <label>winBonus {numberInput(params.reward.winBonus, (v) => setReward('winBonus', v), 0, 1000, 10)}</label>
        <label>seed {numberInput(seed, setSeed, 0, 999999, 1)}</label>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: 12 }}>Training speed:</span>
          {(['slow', 'normal', 'fast', 'turbo'] as const).map((s) => (
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
        <label>steps/frame {numberInput(stepsPerFrame, setStepsPerFrame, 1, 5000, 1)}</label>
        <label>renderEveryNSteps {numberInput(renderEveryNSteps, setRenderEveryNSteps, 1, 1000, 1)}</label>
        <label><input type="checkbox" checked={turbo} onChange={(e) => setTurbo(e.target.checked)} /> turbo</label>
        <label><input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} /> show ghost heatmap</label>
        <label>epsilon {numberInput(agent.hyper.epsilon, (v) => { agent.hyper.epsilon = v; setTick((t) => t + 1); }, 0, 1, 0.01)}</label>
        <label>alpha {numberInput(agent.hyper.alpha, (v) => { agent.hyper.alpha = v; setTick((t) => t + 1); }, 0, 1, 0.01)}</label>
        <label>gamma {numberInput(agent.hyper.gamma, (v) => { agent.hyper.gamma = v; setTick((t) => t + 1); }, 0, 1, 0.01)}</label>
        <label>epsilonDecay {numberInput(agent.hyper.epsilonDecay, (v) => { agent.hyper.epsilonDecay = v; setTick((t) => t + 1); }, 0.9, 1, 0.0001)}</label>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <button onClick={() => { env.reset(seed); setTick((t) => t + 1); }}>Reset</button>
          <button onClick={startTraining}>Start training</button>
          <button onClick={stopTraining}>Pause</button>
          <button onClick={() => { trainer.singleStep(); setTick((t) => t + 1); }}>Single step</button>
          <button onClick={() => { const r = trainer.evaluate(20); setEvalResult(`avgScore=${r.avgScore.toFixed(1)}, avgLength=${r.avgLength.toFixed(1)}, winRate=${(r.winRate * 100).toFixed(1)}%`); }}>Evaluate</button>
          <button onClick={() => { agent.reset(); }}>Reset Q</button>
          <button onClick={savePolicy}>Save policy</button>
          <label style={{ border: '1px solid #374151', padding: 4, cursor: 'pointer' }}>Load policy<input hidden type="file" accept="application/json" onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            agent.load(JSON.parse(await file.text()));
          }} /></label>
        </div>
        <small>{evalResult}</small>
        <div>
          <div>Episode score</div><LineChart values={trainer.stats.episodeScores.slice(-120)} color="#22c55e" />
          <div>Episode length</div><LineChart values={trainer.stats.episodeLengths.slice(-120)} color="#a78bfa" />
          <div>Moving avg score</div><LineChart values={trainer.stats.episodeScores.map((_, i, a) => a.slice(Math.max(0, i - 19), i + 1).reduce((x, y) => x + y, 0) / Math.min(20, i + 1)).slice(-120)} color="#60a5fa" />
          <div>Epsilon</div><LineChart values={trainer.stats.epsilons.slice(-120)} color="#f59e0b" />
        </div>
      </div>
    </div>
  );
}
