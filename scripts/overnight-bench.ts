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
 *   endgameEps=<f>         state-conditional ε floor when in endgame pellet
 *                          buckets (default: 0 = disabled). Suggested: 0.4
 *   endgameBucket=<n>      bucket threshold (≤ this triggers endgameEps).
 *                          0=only-final, 1=late+final (default: 1)
 *
 * Environment options:
 *   maze=<id>              maze id (default: pacman-classic)
 *   ghosts=<n>             numGhosts (default: 2)
 *   maxSteps=<n>           maxEpisodeSteps (default: 800)
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
 *   evalEpisodes=<n>       episodes per eval pass (default: 200)
 *   snapshotEvery=<sec>    policy snapshot interval (default: 600; 0=off)
 *   episodes=<n>           stop after N episodes (default: Infinity)
 *   durationMin=<n>        stop after N minutes (default: Infinity)
 *   endgameCurriculum=<f>  probability of starting an episode in an endgame
 *                          state (0-1, default: 0 = off). When triggered, the
 *                          env clears most pellets so the agent has to learn
 *                          endgame survival directly. Eval episodes always
 *                          start from a full maze.
 *
 * Output (in outDir):
 *   policy-latest.json     most recent snapshot (rewritten periodically + at exit)
 *   episodes.csv           per-episode: score / length / epsilon / qTableSize
 *   evals.csv              greedy eval rows: episode / avgScore / avgLen / winRate
 *   summary.json           final summary including full config
 */
import { mkdirSync, writeFileSync, appendFileSync, readFileSync, existsSync, renameSync } from 'node:fs';
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
  if (eq <= 0) {
    // Flag-only args (e.g. --clean) are not supported by this script; warn
    // loudly instead of silently ignoring so a typo isn't a no-op.
    console.error(`[warn] ignoring unsupported flag-style arg: ${raw} (use key=value)`);
    continue;
  }
  args.set(raw.slice(0, eq).replace(/^-+/, ''), raw.slice(eq + 1));
}
const arg = (k: string, def: string): string => args.get(k) ?? def;
const num = (k: string, def: number): number => {
  const v = args.get(k);
  if (v === undefined) return def;
  const n = Number(v);
  // Reject non-numeric values loudly rather than coercing to NaN. A NaN here
  // poisons every subsequent Q-update silently — training looks like it's
  // running, scores stay flat, no error message ever surfaces.
  if (!Number.isFinite(n) && def !== Number.POSITIVE_INFINITY) {
    console.error(`[abort] CLI arg ${k}=${v} is not a finite number`);
    process.exit(1);
  }
  return Number.isFinite(n) ? n : def;
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
// 800 lets the agent physically have enough steps to win: a maze has ~280
// pellets, the agent collects 1 per tile, and an optimal path is ~290 steps.
// The previous default (400) made winning structurally impossible — the agent
// always died before it had enough time to reach the last cluster. Confirmed
// in the 6-run benchmark: 0 wins across 1.13M episodes, episode length capped
// before the agent could finish the maze.
const maxSteps     = num('maxSteps', 800);
const ghostSpeed   = num('ghostSpeed', 0.95);
const captureRules = arg('capture', 'tile') as 'tile' | 'touch';
const powerPellets = ((): boolean => {
  // Strict boolean parse so powerPellets=0 / powerPellets=off / typos don't
  // silently enable power pellets via the prior "anything-but-false" rule.
  const v = arg('powerPellets', 'true').toLowerCase();
  if (v === 'true'  || v === '1' || v === 'yes' || v === 'on')  return true;
  if (v === 'false' || v === '0' || v === 'no'  || v === 'off') return false;
  console.error(`[abort] powerPellets=${v} unrecognized (use true/false)`);
  process.exit(1);
})();
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
// State-conditional ε floor for endgame states (Priority 3b). 0 = disabled.
const endgameEpsilon         = num('endgameEps', 0);
const endgameBucketThreshold = num('endgameBucket', 1);
const seed         = num('seed', 7);
const loadPath     = args.get('loadPolicy');
const outDir       = resolve(arg('outDir', './bench-out'));
const reportEvery  = num('reportEvery', 60);
const evalEvery    = num('evalEvery', 2000);
// 200 (was 30) — with per-eval stdScore typically 500-900, evalEpisodes=30 gave
// SE of the mean ~100 points, larger than most learning gains we'd want to
// detect. 200 quarters that to ~50 points, enough resolution to distinguish
// signal from noise. Cost: ~30s per eval pass instead of ~5s.
const evalEpisodes = num('evalEpisodes', 200);
const snapshotEvery= num('snapshotEvery', 600);
const maxEpisodes  = num('episodes', Number.POSITIVE_INFINITY);
const maxDurationMs= num('durationMin', Number.POSITIVE_INFINITY) * 60_000;
const endgameCurriculum = num('endgameCurriculum', 0); // P(start in endgame) — 0 = off

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
// pelletsLeft p5/p25/p50/p75/p95 = quintile distribution across eval games —
// reveals whether the agent is *consistently* close to winning, or only
// occasionally (e.g. p50 falling while p5 stays flat means the median game
// improved without the agent ever pushing the best game closer to a win).
writeFileSync(evalsCsv,    'episode,avgScore,stdScore,avgLength,winRate,wins,minPelletsLeft,pl_p5,pl_p25,pl_p50,pl_p75,pl_p95\n');

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

const agent = new QLearningAgent({
  alpha, gamma, epsilon, epsilonDecay, epsilonMin,
  endgameEpsilon, endgameBucketThreshold,
});
if (loadPath) {
  const data = JSON.parse(readFileSync(loadPath, 'utf-8')) as SerializedPolicy;
  agent.load(data, numGhosts);
  // BUG FIX: agent.load() replaces hyper with the saved policy's hyper, so CLI
  // overrides (eps/epsDecay/epsMin/alpha/gamma) were silently discarded. This
  // made run2-resume and run3-explore produce byte-identical evals despite
  // claiming different ε. Reapply CLI hypers here to restore intended semantics.
  agent.hyper = {
    ...agent.hyper,
    alpha, gamma, epsilon, epsilonDecay, epsilonMin,
    endgameEpsilon, endgameBucketThreshold,
  };
  console.log(`[init] loaded policy from ${loadPath} (${Object.keys(data.qTable).length} states, trained with ${data.numGhostsEncoded ?? 'unknown'} ghosts)`);
  console.log(`[init] hyper reapplied from CLI: α=${alpha} γ=${gamma} ε=${epsilon} decay=${epsilonDecay} epsMin=${epsilonMin}`);
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
let episodeStartedAt = startedAt;
let totalWins = 0;  // # training episodes that ended in a win (pelletsLeft=0)

const writePolicy = (): void => {
  // Atomic write: serialize to a tmp file then rename. A SIGKILL during the
  // serial write of a 10MB+ JSON would leave a truncated policy-latest.json
  // that merge-policies.ts silently skips on JSON.parse failure — and you'd
  // never know a worker had been dropped from the merge.
  const tmp = `${policyPath}.tmp`;
  writeFileSync(tmp, JSON.stringify(agent.serialize(mazeId, numGhosts), null, 2));
  renameSync(tmp, policyPath);
};

const writeSummary = (reason: string): void => {
  const scores = trainer.stats.episodeScores;
  const lens   = trainer.stats.episodeLengths;
  const tail   = (arr: number[], n: number): number[] => arr.slice(-n);
  const mean   = (arr: number[]): number => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
  writeFileSync(summaryPath, JSON.stringify({
    reason,
    config: { preset: presetName, ghosts: numGhosts, maxSteps, ghostSpeed, capture: captureRules, powerPellets, illegalMove, alpha, gamma, eps: epsilon, epsDecay: epsilonDecay, epsMin: epsilonMin, seed, endgameCurriculum, endgameEpsilon, endgameBucketThreshold },
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
if (endgameCurriculum > 0) {
  console.log(`[init] endgameCurriculum=${endgameCurriculum} (P of starting episode in 10-25%-pellets endgame state)`);
}
if (endgameEpsilon > 0) {
  console.log(`[init] endgameEps=${endgameEpsilon} when pelletsRemainingBucket ≤ ${endgameBucketThreshold}`);
}
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
    // Per-episode steps-per-second: length / wall-clock elapsed since this
    // episode started. The prior column divided `stepsSinceReport` by
    // elapsed-since-report-time, so every episode in a 60s window got the
    // same monotonically-growing sps until the next report() reset the
    // counter — meaningless per-row data.
    const epElapsedSec = Math.max(0.001, (Date.now() - episodeStartedAt) / 1000);
    const sps = length / epElapsedSec;
    appendFileSync(
      episodesCsv,
      `${episodes},${score},${length},${agent.hyper.epsilon.toFixed(6)},${agent.q.size},${sps.toFixed(0)},${pelletsLeft},${termReason}\n`,
    );
    episodeStartedAt = Date.now();
    episodeSeed = rng.int(1_000_000);
    env.reset(episodeSeed);
    // Endgame curriculum (3a): with the configured probability, fast-forward
    // the env into a late-game state (10-25% pellets remaining) so the agent
    // gets repeated exposure to endgame survival without having to navigate
    // the full maze first. Only applies during training; eval always uses a
    // fresh maze (see runEvalPass).
    if (endgameCurriculum > 0 && rng.next() < endgameCurriculum) {
      const targetFrac = 0.10 + rng.next() * 0.15; // 10-25% remaining
      env.clearPelletsTo(targetFrac, () => rng.next());
    }
    return true;
  }
  return false;
};

/** Linear-interpolated percentile of a sorted-ascending array. */
const percentile = (sortedAsc: number[], p: number): number => {
  if (sortedAsc.length === 0) return NaN;
  if (sortedAsc.length === 1) return sortedAsc[0];
  const idx = (sortedAsc.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedAsc[lo];
  const frac = idx - lo;
  return sortedAsc[lo] * (1 - frac) + sortedAsc[hi] * frac;
};

const runEvalPass = (): void => {
  // Greedy eval: ε=0 globally AND zero out endgameEpsilon, otherwise the
  // state-conditional ε floor would force exploration in late-game states.
  const savedEps         = agent.hyper.epsilon;
  const savedEndgameEps  = agent.hyper.endgameEpsilon;
  agent.hyper.epsilon = 0;
  agent.hyper.endgameEpsilon = 0;
  const scores: number[] = [];
  const pelletsLeftSamples: number[] = [];
  let lenSum = 0, wins = 0;
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
        pelletsLeftSamples.push(r.info.pelletsLeft);
        if (r.info.pelletsLeft === 0) wins += 1;
      }
    }
  }
  agent.hyper.epsilon = savedEps;
  agent.hyper.endgameEpsilon = savedEndgameEps;
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  // Population std dev of eval scores — direct measurement of run-to-run noise.
  // Compare with score deltas across evals to decide if a change is signal or noise.
  const variance = scores.reduce((a, b) => a + (b - avgScore) ** 2, 0) / scores.length;
  const stdScore = Math.sqrt(variance);
  const avgLen   = lenSum / evalEpisodes;
  const winRate  = wins  / evalEpisodes;
  // Quintile distribution of pelletsLeft across eval games. p5 = best game's
  // pelletsLeft (effectively minPelletsLeft); p95 = worst game; p50 = median.
  // Comparing p5 vs p50 distinguishes "occasional lucky run" from "consistent
  // close finishes."
  const pelletsSorted = [...pelletsLeftSamples].sort((a, b) => a - b);
  const p5  = percentile(pelletsSorted, 0.05);
  const p25 = percentile(pelletsSorted, 0.25);
  const p50 = percentile(pelletsSorted, 0.50);
  const p75 = percentile(pelletsSorted, 0.75);
  const p95 = percentile(pelletsSorted, 0.95);
  const minPelletsLeft = pelletsSorted[0] ?? -1;
  appendFileSync(
    evalsCsv,
    `${episodes},${avgScore.toFixed(2)},${stdScore.toFixed(2)},${avgLen.toFixed(2)},${winRate.toFixed(3)},${wins},${minPelletsLeft},${p5.toFixed(1)},${p25.toFixed(1)},${p50.toFixed(1)},${p75.toFixed(1)},${p95.toFixed(1)}\n`,
  );
  console.log(
    `[eval ep=${episodes}] avgScore=${avgScore.toFixed(2)}±${stdScore.toFixed(1)} ` +
    `avgLen=${avgLen.toFixed(2)} wins=${wins}/${evalEpisodes} ` +
    `pelletsLeft p5/p50/p95=${p5.toFixed(0)}/${p50.toFixed(0)}/${p95.toFixed(0)}`,
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
