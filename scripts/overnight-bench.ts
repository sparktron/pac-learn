/**
 * Overnight headless training bench.
 *
 * Run:
 *   npx vite-node scripts/overnight-bench.ts -- [options]
 *
 * Algorithm selection:
 *   algorithm=<tabular|linear> algorithm type (default: tabular)
 *     tabular: discrete Q-table with bucketed state features (~120k states)
 *     linear:  linear approximation with continuous distance features (9 weights)
 *
 * Q-learning options:
 *   alpha=<f>              learning rate (default: 0.1 tabular, 0.02 linear)
 *   gamma=<f>              discount factor (default: 0.99)
 *   eps=<f>                starting epsilon (default: 0.5 tabular, 0.3 linear)
 *   epsDecay=<f>           per-episode epsilon decay (default: 0.999997
 *                          tabular, 0.9995 linear)
 *   epsMin=<f>             epsilon floor (default: 0.20 tabular, 0.05 linear)
 *   endgameEps=<f>         state-conditional ε floor when in endgame pellet
 *                          buckets (default: 0.25 tabular, 0 linear)
 *   endgameBucket=<n>      bucket threshold (≤ this triggers endgameEps).
 *                          0=only-final, 1=late+final (default: 1)
 *   targetSyncSteps=<n>    linear agent only: TD-bootstrap target network
 *                          sync interval, in update() calls (default: 2000
 *                          for linear, 0/ignored for tabular). 0 disables it.
 *   epsMinDecay=<f>        tabular agent only: per-episode decay applied to
 *                          epsilonMin itself, once ε has reached it (default:
 *                          1 = disabled, epsilonMin stays fixed forever).
 *   epsMinFloor=<f>        floor the epsMinDecay above shrinks epsilonMin
 *                          toward (default: epsMin, i.e. no-op unless set
 *                          lower than epsMin).
 *
 * Environment options:
 *   maze=<id>              maze id (default: pacman-classic)
 *   ghosts=<n>             numGhosts (default: 2)
 *   maxSteps=<n>           maxEpisodeSteps (default: 1000, matching the env/GUI)
 *   ghostSpeed=<f>         ghost speed in tiles/step (default: 0.95)
 *   capture=<touch|tile>   collision detection mode (default: tile)
 *   powerPellets=<bool>    enable power pellets (default: true)
 *   illegalMove=<noop|stay> illegal-move handling (default: stay)
 *   preset=<name>          reward preset: default | ghost-hunting |
 *                          pellet-collection | survival (default: default)
 *   stepPenalty=<f>        override reward.stepPenalty
 *   reversePenalty=<f>     override reward.reversePenalty
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
import { mkdirSync, writeFileSync, appendFileSync, readFileSync, renameSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';

import { PacmanEnvironment } from '../src/env/environment';
import { observationKey, observationKeyToString, type Observation } from '../src/env/observation';
import { QLearningAgent, type SerializedPolicy } from '../src/rl/qlearning';
import { LinearQLearningAgent, type SerializedLinearPolicy } from '../src/rl/linearQlearning';
import { TrainingController } from '../src/rl/trainingController';
import { inferTermReason, percentile } from '../src/rl/benchMetrics';
import { REWARD_PRESETS, type RewardConfig } from '../src/rl/rewardPresets';
import { SeededRng } from '../src/engine/prng';
import { DIRECTIONS } from '../src/engine/types';
import { LINEAR_HYPER_DEFAULTS, TABULAR_HYPER_DEFAULTS } from '../src/rl/hyperDefaults';

// ---------- arg parsing ----------
const args = new Map<string, string>();
for (const raw of process.argv.slice(2)) {
  if (raw === '--') continue;
  if (raw === '--diagnostic-log') {
    args.set('diagnosticLog', 'true');
    continue;
  }
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
  // Reject non-numeric values loudly rather than coercing to NaN. A NaN here
  // poisons every subsequent Q-update silently — training looks like it's
  // running, scores stay flat, no error message ever surfaces.
  // Note: we abort on any explicit non-finite input even when the default is
  // Infinity (e.g. durationMin). Otherwise a typo like durationMin=abc would
  // silently fall back to "run forever".
  const n = Number(v);
  if (!Number.isFinite(n)) {
    console.error(`[abort] CLI arg ${k}=${v} is not a finite number`);
    process.exit(1);
  }
  return n;
};

// ---------- reward presets ----------
// EMPIRICALLY VALIDATED (via sweep-01→sweep-02→sweep-03→final-1hr):
// - 'default' preset ONLY viable option (0% wins with pellet-collection over 162M episodes)
// - With endgameCurriculum=0.90 + endgameEps=0.25 + alpha=0.1, achieves 0.676% win rate
//   (sweep-03 discovered alpha=0.1 critical: 2.7× improvement over alpha=0.2)
// - Other presets (ghost-hunting, pellet-collection, survival) do not converge to winning
//
// The 'default' preset's high winBonus (1000 vs others' 100-300) is critical to pushing
// the agent to actually finish mazes rather than settling on safe partial strategies.
// Reward presets are the shared REWARD_PRESETS from src/rl/rewardPresets.ts
// (D5.11) so the bench, App, and presetBench.test can't drift. See that module
// for the empirical notes on why 'default' is the only win-converging preset.

// ---------- arg parsing ----------
const mazeId       = arg('maze', 'pacman-classic');
const numGhosts    = num('ghosts', 2);
// 1000 to match the env/GUI default (defaultParams.maxEpisodeSteps) so headless
// training and the in-app trainer cap episodes identically. A maze has ~280
// pellets at 1 per tile (optimal path ~290 steps); the prior bench defaults (400
// then 800) capped episodes below what the GUI used, so overnight runs didn't
// reflect the in-app environment. Earlier 400 made winning structurally
// impossible (0 wins across 1.13M episodes — agent died before the last cluster).
const maxSteps     = num('maxSteps', 1000);
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
// 'default' preset is empirically better: winBonus=1000 vs pellet-collection's 300.
// pellet-collection got 0% wins over 14M episodes; default got wins within 1 hour.
const presetName   = arg('preset', 'default');
const presetBase   = REWARD_PRESETS[presetName] ?? REWARD_PRESETS['default'];
const preset: RewardConfig = {
  ...presetBase,
  stepPenalty: num('stepPenalty', presetBase.stepPenalty),
  reversePenalty: num('reversePenalty', presetBase.reversePenalty),
};

// Algorithm selection
const algorithmName = arg('algorithm', 'tabular');
if (algorithmName !== 'tabular' && algorithmName !== 'linear') {
  console.error(`[abort] algorithm=${algorithmName} unrecognized (use tabular or linear)`);
  process.exit(1);
}
const algorithm = algorithmName as 'tabular' | 'linear';
const hyperDefaults = algorithm === 'linear' ? LINEAR_HYPER_DEFAULTS : TABULAR_HYPER_DEFAULTS;
console.log(`[setup] using ${algorithm} Q-learning`);

// VALIDATED via sweep-03: alpha=0.1 converges 2.7× better than 0.2 (0.676% vs 0.237%)
// Slower, more careful Q-value updates = better exploration and convergence.
// Linear approximation typically needs smaller alpha (0.01-0.05) to avoid weight divergence.
const alpha        = num('alpha', hyperDefaults.alpha);
const gamma        = num('gamma', hyperDefaults.gamma);
const epsilon      = num('eps', hyperDefaults.epsilon);
// 0.999997 keeps ε above the floor until ~400k episodes (0.999997^400_000 ≈ 0.30
// before hitting the 0.20 floor). Prior default 0.99999 decayed to epsMin at
// ~120k episodes — the Q-table only had 54k states at that point (vs 253k at
// end of run), so the agent locked into a near-greedy policy on an undertrained
// table and plateaued for the remaining 880k episodes.
const epsilonDecay = num('epsDecay', hyperDefaults.epsilonDecay);
// 0.20 floor keeps enough exploration to discover new states throughout a run.
// Prior 0.15 was too low — Q-table was still growing at 1M episodes with
// near-zero exploration, meaning the agent spent most of the run re-exploiting
// states it already knew poorly.
const epsilonMin   = num('epsMin', hyperDefaults.epsilonMin);
// State-conditional ε floor for endgame states. Validated: 0.25 optimal.
// Sweep results: 0.25 >> 0.30 > 0.35 > 0.40. Less randomness in endgame = better.
// Converged value from sweep-01→sweep-02→final-1hr validation.
const endgameEpsilon = num(
  'endgameEps',
  algorithm === 'linear' ? 0 : TABULAR_HYPER_DEFAULTS.endgameEpsilon,
);
const endgameBucketThreshold = num(
  'endgameBucket',
  algorithm === 'linear' ? 1 : TABULAR_HYPER_DEFAULTS.endgameBucketThreshold,
);
// Root cause #3 (2026-07-01 win-rate investigation) / qlearning.ts D10: a
// fixed epsilonMin explores randomly forever. epsilonMinDecay=1 (default)
// keeps that behavior exactly; a tabular-agent-only knob, ignored by linear.
const epsilonMinDecay = num('epsMinDecay', 1);
const epsilonMinFloor = num('epsMinFloor', epsilonMin);
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
const diagnosticLog = ['true', '1', 'yes', 'on'].includes(arg('diagnosticLog', 'false').toLowerCase());
const diagnosticLogPath = resolve(arg('diagnosticLogPath', './notebooklm_diagnostics/failure_simulation_log.txt'));
const maxEpisodes  = num('episodes', diagnosticLog ? 1 : Number.POSITIVE_INFINITY);
const maxDurationMs= num('durationMin', Number.POSITIVE_INFINITY) * 60_000;
// Curriculum: % of episodes starting in endgame (10-25% pellets remaining).
// Validated via sweep-01→sweep-02→final-1hr: 0.90 is optimal (0.3355% win rate).
// 0.80-0.95 all good (0.29-0.33%), but 0.90 peak. Beyond 0.95 drops to 0.15-0.29%.
// Sweet spot: aggressive endgame exposure without overfitting.
const endgameCurriculum = num('endgameCurriculum', 0.90);
// D9: target-network sync interval for the linear agent's TD bootstrap (see
// linearQlearning.ts header). Ignored by the tabular agent. 0 = disabled
// (bootstraps off the live weights, the pre-D9 behavior).
const targetSyncSteps = num('targetSyncSteps', algorithm === 'linear' ? LINEAR_HYPER_DEFAULTS.targetSyncSteps : 0);

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

// Instantiate appropriate agent type
type Agent = QLearningAgent | LinearQLearningAgent;
let agent: Agent;

if (algorithm === 'linear') {
  agent = new LinearQLearningAgent({
    alpha, gamma, epsilon, epsilonDecay, epsilonMin,
    endgameEpsilon, endgameBucketThreshold,
    lambda: 0, // L2 regularization off by default
    targetSyncSteps,
  });
} else {
  agent = new QLearningAgent({
    alpha, gamma, epsilon, epsilonDecay, epsilonMin,
    endgameEpsilon, endgameBucketThreshold,
    epsilonMinDecay, epsilonMinFloor,
  });
}

if (loadPath) {
  const data = JSON.parse(readFileSync(loadPath, 'utf-8')) as SerializedPolicy | SerializedLinearPolicy;

  // Detect which algorithm the saved policy uses
  if ('algorithm' in data) {
    if (data.algorithm === 'linear-qlearning' && algorithm === 'tabular') {
      console.error(`[abort] saved policy is linear-qlearning but algorithm=tabular requested`);
      process.exit(1);
    }
    if (data.algorithm === 'qlearning' && algorithm === 'linear') {
      console.error(`[abort] saved policy is tabular qlearning but algorithm=linear requested`);
      process.exit(1);
    }
  }

  // Narrow on the agent type so each load() gets its matching serialized shape
  // (the union can't be passed positionally without this). Mirrors App.tsx.
  const loaded = agent instanceof LinearQLearningAgent
    ? agent.load(data as SerializedLinearPolicy, numGhosts)
    : agent.load(data as SerializedPolicy, numGhosts);
  if (!loaded) {
    console.error(`[abort] policy ${loadPath} is incompatible with the selected algorithm/environment`);
    process.exit(1);
  }
  // BUG FIX: agent.load() replaces hyper with the saved policy's hyper, so CLI
  // overrides (eps/epsDecay/epsMin/alpha/gamma) were silently discarded. This
  // made run2-resume and run3-explore produce byte-identical evals despite
  // claiming different ε. Reapply CLI hypers here to restore intended semantics.
  agent.hyper = {
    ...agent.hyper,
    alpha, gamma, epsilon, epsilonDecay, epsilonMin,
    endgameEpsilon, endgameBucketThreshold,
    ...(algorithm === 'linear' ? { targetSyncSteps } : { epsilonMinDecay, epsilonMinFloor }),
  };

  if (algorithm === 'linear') {
    console.log(`[init] loaded linear policy from ${loadPath} (trained with ${data.numGhostsEncoded ?? 'unknown'} ghosts)`);
  } else {
    const tabularData = data as SerializedPolicy;
    console.log(`[init] loaded tabular policy from ${loadPath} (${Object.keys(tabularData.qTable).length} states, trained with ${data.numGhostsEncoded ?? 'unknown'} ghosts)`);
  }
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

const getAgentSize = (): number => {
  if (agent instanceof QLearningAgent) {
    return agent.q.size;
  } else {
    // LinearQLearningAgent doesn't have state count; return 0 as placeholder
    return 0;
  }
};

const writeSummary = (reason: string): void => {
  const scores = trainer.stats.episodeScores;
  const lens   = trainer.stats.episodeLengths;
  const tail   = (arr: number[], n: number): number[] => arr.slice(-n);
  const mean   = (arr: number[]): number => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
  writeFileSync(summaryPath, JSON.stringify({
    reason,
    config: { algorithm, preset: presetName, ghosts: numGhosts, maxSteps, ghostSpeed, capture: captureRules, powerPellets, illegalMove, reward: preset, alpha, gamma, eps: epsilon, epsDecay: epsilonDecay, epsMin: epsilonMin, seed, endgameCurriculum, endgameEpsilon, endgameBucketThreshold, ...(algorithm === 'linear' ? { targetSyncSteps } : { epsilonMinDecay, epsilonMinFloor }) },
    elapsedSec: (Date.now() - startedAt) / 1000,
    episodes,
    totalSteps,
    qTableSize: getAgentSize(),
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
    `qStates=${getAgentSize()} ` +
    `avgScore200=${fmt(mean(recentScores))} avgLen200=${fmt(mean(recentLens), 1)}`,
  );
  lastReportAt = now;
  stepsSinceReport = 0;
};

console.log(`[init] preset=${presetName} ghosts=${numGhosts} maxSteps=${maxSteps} ghostSpeed=${ghostSpeed} capture=${captureRules} powerPellets=${powerPellets} illegalMove=${illegalMove}`);
console.log(`[init] rewards stepPenalty=${preset.stepPenalty} reversePenalty=${preset.reversePenalty} winBonus=${preset.winBonus}`);
console.log(`[init] α=${alpha} γ=${gamma} ε=${epsilon} decay=${epsilonDecay} epsMin=${epsilonMin}`);
if (algorithm === 'tabular' && epsilonMinDecay < 1) {
  console.log(`[init] epsilonMinDecay=${epsilonMinDecay} epsilonMinFloor=${epsilonMinFloor} (second-stage floor decay, engages once ε reaches epsMin)`);
}
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

// inferTermReason + percentile now live in src/rl/benchMetrics.ts (D8.4) so they
// are typechecked and unit-tested. maxSteps is passed explicitly at the call site.

const manhattanWrapX = (a: { x: number; y: number }, b: { x: number; y: number }, width: number): number => {
  const rawDx = Math.abs(a.x - b.x);
  const dx = Math.min(rawDx, width - rawDx);
  return dx + Math.abs(a.y - b.y);
};

const nearestPelletDistance = (): number | null => {
  const pac = env.getPacmen()[0]?.pos;
  if (!pac) return null;
  let best = Number.POSITIVE_INFINITY;
  for (let y = 0; y < env.world.height; y += 1) {
    for (let x = 0; x < env.world.width; x += 1) {
      if (!env.world.pellets[y]?.[x] && !env.world.powerPellets[y]?.[x]) continue;
      best = Math.min(best, manhattanWrapX(pac, { x, y }, env.world.width));
    }
  }
  return Number.isFinite(best) ? best : null;
};

const effectiveEpsilon = (obs: Observation): number => {
  const endgameEps = agent.hyper.endgameEpsilon ?? 0;
  const threshold = agent.hyper.endgameBucketThreshold ?? 0;
  if (endgameEps > agent.hyper.epsilon && obs.pelletsRemainingBucket <= threshold) return endgameEps;
  return agent.hyper.epsilon;
};

const actionName = (action: number): string => DIRECTIONS[action] ?? `unknown(${action})`;

const compactStateSummary = (obs: Observation): string => [
  `key=${observationKeyToString(observationKey(obs))}`,
  `wallMask=${obs.wallMask}`,
  `nearestPelletDir=${obs.nearestPelletDir}`,
  `ghostCodes=${obs.ghostCodes.join('/')}`,
  `ghostHeadings=${obs.ghostHeadings.join('/')}`,
  `lastAction=${obs.lastAction}`,
  `pelletBucket=${obs.pelletsRemainingBucket}`,
  `powerBucket=${obs.powerPelletsLeftBucket}`,
].join(';');

let diagnosticLines: string[] = [];
if (diagnosticLog) {
  mkdirSync(dirname(diagnosticLogPath), { recursive: true });
  diagnosticLines = [
    '# Pac-Learn diagnostic failure simulation log',
    `# generatedAt=${new Date().toISOString()}`,
    `# command=${process.argv.join(' ')}`,
    `# config algorithm=${algorithm} maze=${mazeId} ghosts=${numGhosts} seed=${seed} preset=${presetName} maxSteps=${maxSteps}`,
    `# hyper alpha=${alpha} gamma=${gamma} epsilon=${epsilon} epsilonDecay=${epsilonDecay} epsilonMin=${epsilonMin} endgameEpsilon=${endgameEpsilon} endgameBucketThreshold=${endgameBucketThreshold}`,
    'tick\tepisode\tscore\tpacPos\tghostPositions\tnearestGhostDistance\tnearestPelletDistance\tavailableActions\tactionSelected\tactionSource\treward\tdone\tterminationReason\tepsilon\tstateSummary',
  ];
  console.log(`[diagnostic] writing one-episode step log to ${diagnosticLogPath}`);
}

const stepOnce = (): boolean => {
  const obs = env.observe();
  const legal = env.getLegalActionIndices();
  const randomDraws: number[] = [];
  const epsForDecision = effectiveEpsilon(obs);
  const action = agent.act(obs, legal, () => {
    const v = rng.next();
    randomDraws.push(v);
    return v;
  });
  const pacBefore = { ...env.getPacmen()[0].pos };
  const ghostsBefore = env.ghosts.map((g) => ({ ...g.pos }));
  const nearestGhost = ghostsBefore.length
    ? Math.min(...ghostsBefore.map((g) => manhattanWrapX(pacBefore, g, env.world.width)))
    : null;
  const nearestPellet = nearestPelletDistance();
  const actionSource = epsForDecision > 0 && (randomDraws[0] ?? 1) < epsForDecision ? 'random' : 'policy';
  const res = env.step(action);
  const nextLegal = res.done ? [] : env.getLegalActionIndices();
  agent.update(obs, action, res.reward, res.obs, res.done, nextLegal);
  totalSteps += 1;
  stepsSinceReport += 1;
  if (diagnosticLog) {
    diagnosticLines.push([
      totalSteps,
      episodes + 1,
      res.info.score.toFixed(2),
      `${pacBefore.x},${pacBefore.y}`,
      ghostsBefore.map((g) => `${g.x},${g.y}`).join('|'),
      nearestGhost ?? 'none',
      nearestPellet ?? 'none',
      legal.map(actionName).join('|'),
      actionName(action),
      actionSource,
      res.reward.toFixed(2),
      res.done,
      res.done ? inferTermReason(res.info.pelletsLeft, res.info.step, maxSteps) : '',
      epsForDecision.toFixed(6),
      compactStateSummary(obs),
    ].join('\t'));
  }
  if (res.done) {
    const score        = res.info.score;
    const length       = res.info.step;
    const pelletsLeft  = res.info.pelletsLeft;
    const termReason   = inferTermReason(pelletsLeft, length, maxSteps);
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
      `${episodes},${score},${length},${agent.hyper.epsilon.toFixed(6)},${getAgentSize()},${sps.toFixed(0)},${pelletsLeft},${termReason}\n`,
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
      for (const ghost of env.ghosts) ghost.releaseDelay = 0;
    }
    if (diagnosticLog) {
      writeFileSync(diagnosticLogPath, `${diagnosticLines.join('\n')}\n`);
      console.log(`[diagnostic] wrote ${diagnosticLogPath}`);
    }
    return true;
  }
  return false;
};

const runEvalPass = (): void => {
  // Greedy eval: ε=0 globally AND zero out endgameEpsilon, otherwise the
  // state-conditional ε floor would force exploration in late-game states.
  const savedEps         = agent.hyper.epsilon;
  const savedEndgameEps  = agent.hyper.endgameEpsilon;
  agent.hyper.epsilon = 0;
  agent.hyper.endgameEpsilon = 0;
  const evalRng = new SeededRng(0xE0A1);
  const scores: number[] = [];
  const pelletsLeftSamples: number[] = [];
  let lenSum = 0, wins = 0;
  for (let i = 0; i < evalEpisodes; i += 1) {
    env.reset(1_000_000 + i);
    let done = false;
    while (!done) {
      const obs = env.observe();
      const legal = env.getLegalActionIndices();
      const a = agent.act(obs, legal, () => evalRng.next());
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

// N13: reset episodeStartedAt right before the main loop so the very first
// episode's sps column doesn't include process-boot + JIT warm-up time.
episodeStartedAt = Date.now();

while (episodes < maxEpisodes && (Date.now() - startedAt) < maxDurationMs) {
  // Burst a chunk of steps before checking timers — keeps overhead negligible,
  // but still honor short smoke-test limits promptly.
  for (let i = 0; i < 5_000; i += 1) {
    if (episodes >= maxEpisodes || (Date.now() - startedAt) >= maxDurationMs) break;
    stepOnce();
  }

  report();

  if (evalEvery > 0 && episodes - lastEvalEpisode >= evalEvery) {
    lastEvalEpisode = episodes;
    runEvalPass();
  }

  if (snapshotEvery > 0 && (Date.now() - lastSnapshotAt) / 1000 >= snapshotEvery) {
    lastSnapshotAt = Date.now();
    writePolicy();
    const agentInfo = algorithm === 'linear' ? `(weights)` : `(${getAgentSize()} states)`;
    console.log(`[snapshot ep=${episodes}] wrote ${policyPath} ${agentInfo}`);
  }
}

report(true);
writePolicy();
writeSummary('completed');
console.log(`[done] episodes=${episodes} steps=${totalSteps} elapsed=${((Date.now() - startedAt) / 1000).toFixed(0)}s`);
console.log(`[done] policy: ${policyPath}`);
console.log(`[done] summary: ${summaryPath}`);
