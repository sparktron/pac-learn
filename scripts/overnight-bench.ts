/**
 * Overnight headless training bench.
 *
 * Run:
 *   npx vite-node scripts/overnight-bench.ts -- [options]
 *
 * Q-learning options:
 *   alpha=<f>              learning rate (default: 0.2)
 *   gamma=<f>              discount factor (default: 0.99)
 *   eps=<f>                starting epsilon (default: 0.5)
 *   epsDecay=<f>           per-episode epsilon decay (default: 0.99999)
 *   epsMin=<f>             epsilon floor (default: 0.15)
 *
 * Environment options:
 *   maze=<id>              maze id (default: pacman-classic)
 *   ghosts=<n>             numGhosts (default: 2)
 *   maxSteps=<n>           maxEpisodeSteps (default: 400)
 *   ghostSpeed=<f>         ghost speed in tiles/step (default: 0.95)
 *   capture=<touch|tile>   collision detection mode (default: tile)
 *   powerPellets=<bool>    enable power pellets (default: true)
 *   illegalMove=<noop|stay> illegal-move handling (default: stay)
 *   preset=<name>          reward preset: default | ghost-hunting |
 *                          pellet-collection | survival (default: default)
 *
 * Run control:
 *   seed=<n>               RNG seed (default: 7)
 *   loadPolicy=<path>      load a SerializedPolicy JSON before training
 *   outDir=<path>          output directory (default: ./bench-out)
 *   reportEvery=<sec>      progress log interval in seconds (default: 60)
 *   evalEvery=<episodes>   greedy-eval interval in episodes (default: 2000; 0=off)
 *   evalEpisodes=<n>       episodes per eval pass (default: 30)
 *   snapshotEvery=<sec>    policy snapshot interval (default: 600; 0=off)
 *   episodes=<n>           stop after N episodes (default: Infinity)
 *   durationMin=<n>        stop after N minutes (default: Infinity)
 *
 * Output (in outDir):
 *   policy-latest.json     most recent snapshot (rewritten periodically + at exit)
 *   episodes.csv           per-episode: score / length / epsilon / qTableSize
 *   evals.csv              greedy eval rows: episode / avgScore / avgLen / winRate
 *   summary.json           final summary including full config
 */
import { mkdirSync, writeFileSync, appendFileSync, readFileSync, existsSync } from 'node:fs';
import { resolve, join } from 'node:path';

import { PacmanEnvironment } from '../src/env/environment';
import { QLearningAgent, type SerializedPolicy } from '../src/rl/qlearning';
import { TrainingController } from '../src/rl/trainingController';
import { SeededRng } from '../src/engine/prng';
import { DIRECTIONS } from '../src/engine/types';

// ---------- arg parsing ----------
const args = new Map<string, string>();
for (const raw of process.argv.slice(2)) {
  if (raw === '--') continue;
  const eq = raw.indexOf('=');
  if (eq <= 0) continue;
  args.set(raw.slice(0, eq).replace(/^-+/, ''), raw.slice(eq + 1));
}
const arg = (k: string, def: string): string => args.get(k) ?? def;
const num = (k: string, def: number): number => {
  const v = args.get(k);
  return v === undefined ? def : Number(v);
};

// ---------- reward presets ----------
type RewardCfg = { pelletReward: number; powerPelletReward: number; deathPenalty: number; stepPenalty: number; survivalReward: number; ghostEatReward: number; winBonus: number };
const PRESETS: Record<string, RewardCfg> = {
  // 'default' is win-seeking: winBonus dominates, survivalReward 0 to avoid loitering.
  // Per-pellet escalation (in env) ramps pelletReward up to 6× as pellets are cleared.
  'default':           { pelletReward: 5,  powerPelletReward: 20, deathPenalty: -100, stepPenalty: -0.1,  survivalReward: 0,    ghostEatReward: 30,  winBonus: 1000 },
  'ghost-hunting':     { pelletReward: 2,  powerPelletReward: 30, deathPenalty: -50,  stepPenalty: -0.05, survivalReward: 0.01, ghostEatReward: 80,  winBonus: 100 },
  'pellet-collection': { pelletReward: 15, powerPelletReward: 40, deathPenalty: -120, stepPenalty: -0.1,  survivalReward: 0.02, ghostEatReward: 20,  winBonus: 300 },
  'survival':          { pelletReward: 3,  powerPelletReward: 20, deathPenalty: -250, stepPenalty: -0.05, survivalReward: 0.2,  ghostEatReward: 50,  winBonus: 100 },
};

// ---------- arg parsing ----------
const mazeId       = arg('maze', 'pacman-classic');
const numGhosts    = num('ghosts', 2);
const maxSteps     = num('maxSteps', 400);
const ghostSpeed   = num('ghostSpeed', 0.95);
const captureRules = arg('capture', 'tile') as 'tile' | 'touch';
const powerPellets = arg('powerPellets', 'true') !== 'false';
const illegalMove  = arg('illegalMove', 'stay') as 'stay' | 'noop';
const presetName   = arg('preset', 'default');
const preset       = PRESETS[presetName] ?? PRESETS['default'];

const alpha        = num('alpha', 0.2);
const gamma        = num('gamma', 0.99);
const epsilon      = num('eps', 0.5);
// Slower decay keeps exploration alive: 0.99999^300_000 ≈ 0.025, so ε reaches
// the floor around episode 300k (vs episode 4.6k with the previous 0.999 decay).
const epsilonDecay = num('epsDecay', 0.99999);
// Higher floor preserves exploration after decay — analysis showed the agent
// locked into a survival policy with epsMin=0.05 and never found wins.
const epsilonMin   = num('epsMin', 0.15);
const seed         = num('seed', 7);
const loadPath     = args.get('loadPolicy');
const outDir       = resolve(arg('outDir', './bench-out'));
const reportEvery  = num('reportEvery', 60);
const evalEvery    = num('evalEvery', 2000);
const evalEpisodes = num('evalEpisodes', 30);
const snapshotEvery= num('snapshotEvery', 600);
const maxEpisodes  = num('episodes', Number.POSITIVE_INFINITY);
const maxDurationMs= num('durationMin', Number.POSITIVE_INFINITY) * 60_000;

mkdirSync(outDir, { recursive: true });
const episodesCsv = join(outDir, 'episodes.csv');
const evalsCsv    = join(outDir, 'evals.csv');
const policyPath  = join(outDir, 'policy-latest.json');
const summaryPath = join(outDir, 'summary.json');
// Extended schema: pelletsLeft + termReason help diagnose 0% win rate by showing
// whether the agent gets close to finishing or dies/times-out far from the goal.
writeFileSync(episodesCsv, 'episode,score,length,epsilon,qTableSize,stepsPerSec,pelletsLeft,termReason\n');
// Eval schema adds stdScore + wins so we can see variance directly and spot
// single wins immediately (instead of waiting for winRate to round above zero).
writeFileSync(evalsCsv,    'episode,avgScore,stdScore,avgLength,winRate,wins,minPelletsLeft\n');

// ---------- setup ----------
const env = new PacmanEnvironment();
env.setParams({
  mazeId,
  numGhosts,
  maxEpisodeSteps: maxSteps,
  ghostSpeed,
  captureRules,
  enablePowerPellets: powerPellets,
  illegalMoveMode: illegalMove,
  reward: preset,
});
env.reset(seed);

const agent = new QLearningAgent({ alpha, gamma, epsilon, epsilonDecay, epsilonMin });
if (loadPath) {
  const data = JSON.parse(readFileSync(loadPath, 'utf-8')) as SerializedPolicy;
  agent.load(data, numGhosts);
  console.log(`[init] loaded policy from ${loadPath} (${Object.keys(data.qTable).length} states, trained with ${data.numGhostsEncoded ?? 'unknown'} ghosts)`);
}

const trainer = new TrainingController(env, agent);
trainer.setSeed(seed);

// ---------- training loop ----------
const startedAt = Date.now();
let lastReportAt = startedAt;
let lastSnapshotAt = startedAt;
let stepsSinceReport = 0;
let totalSteps = 0;
let episodes = 0;
let totalWins = 0;  // # training episodes that ended in a win (pelletsLeft=0)

const writePolicy = (): void => {
  writeFileSync(policyPath, JSON.stringify(agent.serialize(mazeId, numGhosts), null, 2));
};

const writeSummary = (reason: string): void => {
  const scores = trainer.stats.episodeScores;
  const lens   = trainer.stats.episodeLengths;
  const tail   = (arr: number[], n: number): number[] => arr.slice(-n);
  const mean   = (arr: number[]): number => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
  writeFileSync(summaryPath, JSON.stringify({
    reason,
    config: { preset: presetName, ghosts: numGhosts, maxSteps, ghostSpeed, capture: captureRules, powerPellets, illegalMove, alpha, gamma, eps: epsilon, epsDecay: epsilonDecay, epsMin: epsilonMin, seed },
    elapsedSec: (Date.now() - startedAt) / 1000,
    episodes,
    totalSteps,
    qTableSize: agent.q.size,
    epsilon: agent.hyper.epsilon,
    trainingWins: totalWins,            // # training episodes that hit pelletsLeft=0
    trainingWinRate: episodes > 0 ? totalWins / episodes : 0,
    meanScoreAll: mean(scores),
    meanLenAll: mean(lens),
    meanScoreLast1000: mean(tail(scores, 1000)),
    meanLenLast1000: mean(tail(lens, 1000)),
  }, null, 2));
};

let shuttingDown = false;
const shutdown = (sig: string) => {
  if (shuttingDown) return;
  shuttingDown = true;
  process.stdout.write(`\n[${sig}] flushing policy + summary...\n`);
  writePolicy();
  writeSummary(sig);
  process.stdout.write(`[${sig}] wrote ${policyPath} and ${summaryPath}\n`);
  process.exit(0);
};
process.on('SIGINT',  () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

const fmt = (n: number, p = 2): string => Number.isFinite(n) ? n.toFixed(p) : String(n);
const report = (force = false): void => {
  const now = Date.now();
  if (!force && (now - lastReportAt) / 1000 < reportEvery) return;
  const elapsed = (now - startedAt) / 1000;
  const recentScores = trainer.stats.episodeScores.slice(-200);
  const recentLens   = trainer.stats.episodeLengths.slice(-200);
  const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
  const sps = stepsSinceReport / Math.max(0.001, (now - lastReportAt) / 1000);
  console.log(
    `[t=${fmt(elapsed, 0)}s] ep=${episodes} steps=${totalSteps} ` +
    `sps=${fmt(sps, 0)} ε=${fmt(agent.hyper.epsilon, 4)} ` +
    `qStates=${agent.q.size} ` +
    `avgScore200=${fmt(mean(recentScores))} avgLen200=${fmt(mean(recentLens), 1)}`,
  );
  lastReportAt = now;
  stepsSinceReport = 0;
};

console.log(`[init] preset=${presetName} ghosts=${numGhosts} maxSteps=${maxSteps} ghostSpeed=${ghostSpeed} capture=${captureRules} powerPellets=${powerPellets} illegalMove=${illegalMove}`);
console.log(`[init] α=${alpha} γ=${gamma} ε=${epsilon} decay=${epsilonDecay} epsMin=${epsilonMin}`);
console.log(`[init] outDir=${outDir}`);
console.log(`[init] reportEvery=${reportEvery}s evalEvery=${evalEvery}ep snapshotEvery=${snapshotEvery}s`);
console.log(`[init] training started — press Ctrl-C to stop and save.`);

// We replicate trainingController.singleStep manually so we can drive the loop
// at maximum speed without rAF and without DOM-y options.
const rng = new SeededRng(seed);
let episodeSeed = seed;

// Termination reason for each episode — distinguishes "won" / "died" / "timeout".
// Inferred from final pelletsLeft + step count (0 pellets => won; step==maxSteps => timeout; else died).
const inferTermReason = (pelletsLeft: number, stepCount: number): string => {
  if (pelletsLeft === 0) return 'won';
  if (stepCount >= maxSteps) return 'timeout';
  return 'died';
};

const stepOnce = (): boolean => {
  const obs = env.observe();
  const legal = env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
  const action = agent.act(obs, legal, () => rng.next());
  const res = env.step(action);
  const nextLegal = res.done ? [] : env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
  agent.update(obs, action, res.reward, res.obs, res.done, nextLegal);
  totalSteps += 1;
  stepsSinceReport += 1;
  if (res.done) {
    const score        = res.info.score;
    const length       = res.info.step;
    const pelletsLeft  = res.info.pelletsLeft;
    const termReason   = inferTermReason(pelletsLeft, length);
    if (termReason === 'won') totalWins += 1;
    trainer.stats.episodeScores.push(score);
    trainer.stats.episodeLengths.push(length);
    trainer.stats.epsilons.push(agent.hyper.epsilon);
    agent.endEpisode();
    episodes += 1;
    const sps = stepsSinceReport / Math.max(0.001, (Date.now() - lastReportAt) / 1000);
    appendFileSync(
      episodesCsv,
      `${episodes},${score},${length},${agent.hyper.epsilon.toFixed(6)},${agent.q.size},${sps.toFixed(0)},${pelletsLeft},${termReason}\n`,
    );
    episodeSeed = rng.int(1_000_000);
    env.reset(episodeSeed);
    return true;
  }
  return false;
};

const runEvalPass = (): void => {
  const savedEps = agent.hyper.epsilon;
  agent.hyper.epsilon = 0;
  const scores: number[] = [];
  let lenSum = 0, wins = 0, minPelletsLeft = Infinity;
  for (let i = 0; i < evalEpisodes; i += 1) {
    env.reset(1_000_000 + i);
    let done = false;
    while (!done) {
      const obs = env.observe();
      const legal = env.getLegalActions().map((d) => DIRECTIONS.indexOf(d));
      const a = agent.act(obs, legal, () => rng.next());
      const r = env.step(a);
      done = r.done;
      if (done) {
        scores.push(r.info.score);
        lenSum += r.info.step;
        if (r.info.pelletsLeft === 0) wins += 1;
        if (r.info.pelletsLeft < minPelletsLeft) minPelletsLeft = r.info.pelletsLeft;
      }
    }
  }
  agent.hyper.epsilon = savedEps;
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  // Population std dev of eval scores — direct measurement of run-to-run noise.
  // Compare with score deltas across evals to decide if a change is signal or noise.
  const variance = scores.reduce((a, b) => a + (b - avgScore) ** 2, 0) / scores.length;
  const stdScore = Math.sqrt(variance);
  const avgLen   = lenSum / evalEpisodes;
  const winRate  = wins  / evalEpisodes;
  appendFileSync(
    evalsCsv,
    `${episodes},${avgScore.toFixed(2)},${stdScore.toFixed(2)},${avgLen.toFixed(2)},${winRate.toFixed(3)},${wins},${Number.isFinite(minPelletsLeft) ? minPelletsLeft : -1}\n`,
  );
  console.log(
    `[eval ep=${episodes}] avgScore=${avgScore.toFixed(2)}±${stdScore.toFixed(1)} ` +
    `avgLen=${avgLen.toFixed(2)} wins=${wins}/${evalEpisodes} minPelletsLeft=${Number.isFinite(minPelletsLeft) ? minPelletsLeft : '?'}`,
  );
  // restore RNG-driven episode position
  env.reset(episodeSeed);
};

let lastEvalEpisode = 0;
report(true);

while (episodes < maxEpisodes && (Date.now() - startedAt) < maxDurationMs) {
  // Burst a chunk of steps before checking timers — keeps overhead negligible.
  for (let i = 0; i < 5_000; i += 1) stepOnce();

  report();

  if (evalEvery > 0 && episodes - lastEvalEpisode >= evalEvery) {
    lastEvalEpisode = episodes;
    runEvalPass();
  }

  if (snapshotEvery > 0 && (Date.now() - lastSnapshotAt) / 1000 >= snapshotEvery) {
    lastSnapshotAt = Date.now();
    writePolicy();
    console.log(`[snapshot ep=${episodes}] wrote ${policyPath} (${agent.q.size} states)`);
  }
}

report(true);
writePolicy();
writeSummary('completed');
console.log(`[done] episodes=${episodes} steps=${totalSteps} elapsed=${((Date.now() - startedAt) / 1000).toFixed(0)}s`);
console.log(`[done] policy: ${policyPath}`);
console.log(`[done] summary: ${summaryPath}`);
