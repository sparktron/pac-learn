/**
 * Isolated T6 full-grid CNN Double-DQN bench.
 *
 * Uses the same PacmanEnvironment and CnnDqnAgent as the browser path, but does
 * not touch the production linear trainer. A run writes standard panel metrics
 * plus TensorFlow backend, throughput, update count, and tensor memory.
 *
 * Example:
 *   ./node_modules/.bin/vite-node scripts/cnn-bench.ts -- episodes=2000 \
 *     evalEpisodes=50 evalPanels=1000000,2000000,3000000,4000000
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { SeededRng } from '../src/engine/prng';
import { PacmanEnvironment } from '../src/env/environment';
import { percentile } from '../src/rl/benchMetrics';
import { CnnDqnAgent, CNN_DQN_DEFAULTS, encodeCnnState } from '../src/rl/cnnDqn';
import { tf } from '../src/rl/tfRuntime';

const args = new Map<string, string>();
for (const raw of process.argv.slice(2)) {
  if (raw === '--') continue;
  const equals = raw.indexOf('=');
  if (equals <= 0) throw new Error(`arguments must use key=value; received ${raw}`);
  args.set(raw.slice(0, equals), raw.slice(equals + 1));
}
const num = (key: string, fallback: number): number => {
  const value = args.get(key);
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${key}=${value} is not finite`);
  return parsed;
};
const integer = (key: string, fallback: number, min: number): number => {
  const value = num(key, fallback);
  if (!Number.isInteger(value) || value < min) throw new Error(`${key} must be an integer >= ${min}; received ${value}`);
  return value;
};

const episodes = integer('episodes', 2_000, 1);
const evalEpisodes = integer('evalEpisodes', 50, 0); // zero is a runner-throughput smoke.
const seed = integer('seed', 7, 0);
const maxSteps = integer('maxSteps', 1_000, 1);
const trainEvery = integer('trainEvery', 128, 1);
const warmupTransitions = integer('warmupTransitions', CNN_DQN_DEFAULTS.batchSize, 1);
const outDir = resolve(args.get('outDir') ?? `bench-out/${new Date().toISOString().replace(/[:.]/g, '-')}-cnn`);
const evalPanels = (args.get('evalPanels') ?? '1000000,2000000,3000000,4000000')
  .split(',').map((value) => Number(value.trim()));
if (evalPanels.some((value) => !Number.isInteger(value))) throw new Error('evalPanels must be comma-separated integers');

const agent = new CnnDqnAgent({
  ...CNN_DQN_DEFAULTS,
  gamma: num('gamma', CNN_DQN_DEFAULTS.gamma),
  epsilon: num('eps', CNN_DQN_DEFAULTS.epsilon),
  epsilonDecay: num('epsDecay', CNN_DQN_DEFAULTS.epsilonDecay),
  epsilonMin: num('epsMin', CNN_DQN_DEFAULTS.epsilonMin),
  learningRate: num('learningRate', CNN_DQN_DEFAULTS.learningRate),
  replayCapacity: integer('replayCapacity', CNN_DQN_DEFAULTS.replayCapacity, 1),
  batchSize: integer('batchSize', CNN_DQN_DEFAULTS.batchSize, 1),
  targetSyncSteps: integer('targetSyncSteps', CNN_DQN_DEFAULTS.targetSyncSteps, 1),
  seed,
});
if (warmupTransitions < agent.hyper.batchSize) throw new Error('warmupTransitions must be at least batchSize');

const env = new PacmanEnvironment();
env.setParams({ mazeId: args.get('maze') ?? 'pacman-classic', numGhosts: integer('ghosts', 2, 0), maxEpisodeSteps: maxSteps });
const rng = new SeededRng(seed);
const endgameCurriculum = num('endgameCurriculum', 0.90);
if (endgameCurriculum < 0 || endgameCurriculum > 1) throw new Error('endgameCurriculum must be in [0, 1]');

interface PanelResult { panel: number; avgScore: number; avgLength: number; wins: number; winRate: number; minPelletsLeft: number; plP5: number; }

const evaluatePanel = async (panel: number): Promise<PanelResult> => {
  const previousEpsilon = agent.hyper.epsilon;
  agent.hyper.epsilon = 0;
  const evalRng = new SeededRng(0xE0A1);
  let score = 0;
  let length = 0;
  let wins = 0;
  const pellets: number[] = [];
  try {
    for (let i = 0; i < evalEpisodes; i += 1) {
      env.reset(panel + i);
      let done = false;
      while (!done) {
        const action = await agent.act(encodeCnnState(env), env.getLegalActionIndices(), () => evalRng.next());
        const result = env.step(action);
        done = result.done;
        if (done) {
          score += result.info.score;
          length += result.info.step;
          pellets.push(result.info.pelletsLeft);
          if (result.info.pelletsLeft === 0) wins += 1;
        }
      }
    }
  } finally {
    agent.hyper.epsilon = previousEpsilon;
  }
  pellets.sort((a, b) => a - b);
  return {
    panel,
    avgScore: score / evalEpisodes,
    avgLength: length / evalEpisodes,
    wins,
    winRate: wins / evalEpisodes,
    minPelletsLeft: pellets[0] ?? 0,
    plP5: percentile(pellets, 0.05),
  };
};

const main = async (): Promise<void> => {
  mkdirSync(outDir, { recursive: true });
  const backend = await agent.ready();
  const startedAt = performance.now();
  let totalSteps = 0;
  let updates = 0;
  let trainingWins = 0;
  let lastLoss: number | null = null;

  for (let episode = 0; episode < episodes; episode += 1) {
    env.reset(rng.int(1_000_000));
    if (rng.next() < endgameCurriculum) env.clearPelletsTo(0.10 + rng.next() * 0.15, () => rng.next());
    let done = false;
    while (!done) {
      const state = encodeCnnState(env);
      const action = await agent.act(state, env.getLegalActionIndices(), () => rng.next());
      const result = env.step(action);
      const nextLegal = result.done ? [] : env.getLegalActionIndices();
      agent.observe({ state, action, reward: result.reward, nextState: encodeCnnState(env), done: result.done, nextLegalActions: nextLegal });
      totalSteps += 1;
      if (totalSteps % trainEvery === 0 && agent.replay.size >= warmupTransitions) {
        lastLoss = await agent.learn(() => rng.next());
        if (lastLoss !== null) updates += 1;
      }
      done = result.done;
      if (done && result.info.pelletsLeft === 0) trainingWins += 1;
    }
    agent.endEpisode();
  }

  const panels = evalEpisodes > 0 ? await Promise.all(evalPanels.map(evaluatePanel)) : [];
  const elapsedSec = (performance.now() - startedAt) / 1_000;
  const memory = tf.memory();
  writeFileSync(join(outDir, 'evals.csv'), [
    'episode,avgScore,avgLength,winRate,wins,minPelletsLeft,pl_p5,panel',
    ...panels.map((panel) => `${episodes},${panel.avgScore.toFixed(2)},${panel.avgLength.toFixed(2)},${panel.winRate.toFixed(6)},${panel.wins},${panel.minPelletsLeft},${panel.plP5.toFixed(3)},${panel.panel}`),
    '',
  ].join('\n'));
  writeFileSync(join(outDir, 'summary.json'), JSON.stringify({
    config: { episodes, evalEpisodes, evalPanels, seed, maxSteps, trainEvery, warmupTransitions, endgameCurriculum, hyper: agent.hyper },
    runtime: { backend, elapsedSec, environmentStepsPerSec: totalSteps / Math.max(elapsedSec, 0.001), updates, updatesPerSec: updates / Math.max(elapsedSec, 0.001), tensorMemory: memory },
    totalSteps,
    trainingWins,
    trainingWinRate: trainingWins / episodes,
    lastLoss,
    panels,
  }, null, 2));
  console.log(`[cnn-bench] backend=${backend} steps=${totalSteps} updates=${updates} sps=${(totalSteps / Math.max(elapsedSec, 0.001)).toFixed(1)} tensors=${memory.numTensors} outDir=${outDir}`);
  agent.dispose();
};

main().catch((error: unknown) => {
  console.error('[cnn-bench] failed:', error);
  agent.dispose();
  process.exitCode = 1;
});
