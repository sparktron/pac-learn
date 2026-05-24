/**
 * Q-table merger for parallel federated training.
 *
 * Merges Q-values across N policy JSON files. Used by run-parallel.sh after
 * spawning multiple independent training workers — each worker explores a
 * different region of state-space (different seeds) and the merged Q-table
 * is an ensemble that benefits from all of their experience.
 *
 * Merge semantics (per state, per slot):
 *   • If any worker has visitTable[key][a] > 0, the merged Q is the
 *     visit-weighted average over those workers only. Slots no worker has
 *     visited stay at optimisticInit. This prevents the prior catastrophic
 *     mode where an untouched slot's optimisticInit (50) was averaged with
 *     a learned −90, masking the death signal entirely.
 *   • Legacy policies without visitTable fall back to "skip values that
 *     equal optimisticInit". Less precise (a learned value happens to equal
 *     50 is treated as untouched) but the right direction.
 *   • The output visitTable is the sum of input visits per slot — so a
 *     merged policy resumed in another federated run will weight its slots
 *     correctly against fresh workers.
 *
 * Usage:
 *   npx vite-node scripts/merge-policies.ts -- out=<path> <p1.json> <p2.json> ...
 *
 * Output: a SerializedPolicy at `out=`. Metadata (mazeId, numGhostsEncoded,
 * observationKeyVersion) is copied from the first input policy; timestamp
 * is set to merge time. hyper.epsilon is reset to the MAX across inputs so
 * a resumed worker explores rather than running near-greedy from a decayed
 * end-of-training ε.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import type { SerializedPolicy } from '../src/rl/qlearning';

const rawArgs = process.argv.slice(2).filter((a) => a !== '--');
const allowPartial = rawArgs.includes('--allow-partial');
const outArg = rawArgs.find((a) => a.startsWith('out='));
const outPath = outArg ? outArg.slice('out='.length) : './merged-policy.json';
const inputs = rawArgs.filter((a) => !a.startsWith('out=') && !a.startsWith('--'));

if (inputs.length === 0) {
  console.error('usage: merge-policies.ts -- [--allow-partial] out=<path> <policy1.json> <policy2.json> ...');
  process.exit(1);
}

const policies: SerializedPolicy[] = [];
const failed: string[] = [];
for (const path of inputs) {
  try {
    policies.push(JSON.parse(readFileSync(path, 'utf-8')) as SerializedPolicy);
  } catch (err) {
    failed.push(`${path}: ${(err as Error).message}`);
  }
}

if (policies.length === 0) {
  console.error('no policies could be loaded');
  process.exit(1);
}

// Fail loudly when some inputs couldn't be read — silently dropping workers
// from a federated merge can mask catastrophic atomic-write failures and
// produces a much weaker merged policy without warning. The --allow-partial
// flag opts in for cases where partial merge is desired (e.g. fast iteration).
if (failed.length > 0) {
  for (const f of failed) console.error(`[skip] could not read ${f}`);
  if (!allowPartial) {
    console.error(`[abort] ${failed.length}/${inputs.length} inputs failed. Pass --allow-partial to merge anyway.`);
    process.exit(1);
  }
  console.error(`[warn] --allow-partial: continuing with ${policies.length}/${inputs.length} workers.`);
}

// Validate: all policies should have the same observationKeyVersion. Mixed
// versions would silently miscompare keys, so we refuse to merge them.
const version = policies[0].observationKeyVersion;
for (let i = 1; i < policies.length; i += 1) {
  if (policies[i].observationKeyVersion !== version) {
    console.error(`[abort] policy ${inputs[i]} has observationKeyVersion=${policies[i].observationKeyVersion} but expected ${version}`);
    process.exit(1);
  }
}

const init = policies[0].hyper.optimisticInit ?? 50;

// Cap per-slot weight when merging visit-weighted Q-values. Without a cap,
// a worker that visited a state 1M times totally drowns out 31 peers with
// modest visit counts — the federation collapses to that worker's policy on
// hot states. The square-root taper preserves "more visits = more credible"
// while keeping any single worker's contribution within an order of magnitude
// of its peers.
const visitWeight = (v: number): number => (v > 0 ? Math.sqrt(v) : 0);

// Per-slot accumulators: weighted sum of Q × visits, and total visits.
// Slots with zero visits in every input stay at optimisticInit in the merge.
const qSums      = new Map<string, [number, number, number, number]>();
const visitSums  = new Map<string, [number, number, number, number]>();
const rawVisitSums = new Map<string, [number, number, number, number]>();
const stateCount = new Map<string, number>();
let totalSlotsWithVisits = 0;

// N14: visitTable is now always written by serialize(); the legacy
// "skip values that equal optimisticInit" fallback was a footgun (it
// would silently drop legitimately learned values that happened to
// equal init). Refuse legacy policies loud and clear — re-train any
// genuinely-needed ancient policies with the current code.
for (let i = 0; i < policies.length; i += 1) {
  if (!policies[i].visitTable) {
    console.error(`[abort] ${inputs[i]} has no visitTable (legacy format). Re-train with current code or run the legacy merger.`);
    process.exit(1);
  }
}

for (const policy of policies) {
  for (const [key, values] of Object.entries(policy.qTable)) {
    let qSum = qSums.get(key);
    let vSum = visitSums.get(key);
    let rvSum = rawVisitSums.get(key);
    if (!qSum) { qSum = [0, 0, 0, 0]; qSums.set(key, qSum); }
    if (!vSum) { vSum = [0, 0, 0, 0]; visitSums.set(key, vSum); }
    if (!rvSum) { rvSum = [0, 0, 0, 0]; rawVisitSums.set(key, rvSum); }
    stateCount.set(key, (stateCount.get(key) ?? 0) + 1);

    // visitTable is guaranteed present by the legacy-refusal check above.
    const slotVisits = policy.visitTable![key];
    for (let i = 0; i < 4; i += 1) {
      const q = values[i];
      if (q === undefined) continue;
      const rawVisits = slotVisits?.[i] ?? 0;
      const w = visitWeight(rawVisits);
      if (w > 0) {
        totalSlotsWithVisits += 1;
        qSum[i] += q * w;
        vSum[i] += w;
        rvSum[i] += rawVisits;
      }
    }
  }
}

const mergedQ: Record<string, number[]> = {};
const mergedVisits: Record<string, number[]> = {};
let sharedStates = 0;
for (const [key, qSum] of qSums.entries()) {
  if ((stateCount.get(key) ?? 0) > 1) sharedStates += 1;
  const vSum = visitSums.get(key)!;
  const rvSum = rawVisitSums.get(key)!;
  mergedQ[key] = qSum.map((s, i) => (vSum[i] > 0 ? s / vSum[i] : init));
  // Persist raw visit counts (not weights) so subsequent merges of the
  // merged policy taper the same way as a first-pass merge.
  mergedVisits[key] = [rvSum[0], rvSum[1], rvSum[2], rvSum[3]];
}

const totalCoverage = Array.from(stateCount.values()).reduce((a, b) => a + b, 0);
const avgWorkersPerState = totalCoverage / stateCount.size;
const sharedFraction = sharedStates / stateCount.size;

// Reset ε to the MAX across inputs. Workers serialize their decayed end-of-
// training ε; if we just copied policies[0].hyper, a resume would start
// nearly-greedy and federated exploration would collapse.
const maxEpsilon = policies.reduce((m, p) => Math.max(m, p.hyper.epsilon), 0);
const mergedHyper = { ...policies[0].hyper, epsilon: maxEpsilon };

const merged: SerializedPolicy = {
  algorithm: policies[0].algorithm,
  mazeId: policies[0].mazeId,
  timestamp: new Date().toISOString(),
  numGhostsEncoded: policies[0].numGhostsEncoded,
  observationKeyVersion: policies[0].observationKeyVersion,
  hyper: mergedHyper,
  qTable: mergedQ,
  visitTable: mergedVisits,
};

writeFileSync(outPath, JSON.stringify(merged, null, 2));

console.log(`[merge] merged ${policies.length} policies → ${outPath}`);
console.log(`[merge]   total states: ${stateCount.size}`);
console.log(`[merge]   shared across ≥2 workers: ${sharedStates} (${(sharedFraction * 100).toFixed(1)}%)`);
console.log(`[merge]   avg workers per state: ${avgWorkersPerState.toFixed(2)} / ${policies.length}`);
console.log(`[merge]   slot-visits used (weighted): ${totalSlotsWithVisits}`);
console.log(`[merge]   reset ε → ${maxEpsilon.toFixed(4)} (max across inputs)`);
console.log(`[merge]   per-worker state counts: ${policies.map((p) => Object.keys(p.qTable).length).join(', ')}`);
