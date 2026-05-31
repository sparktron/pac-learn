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
import { mergePolicies } from '../src/rl/policyMerge';

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

// The merge algorithm (visit-weighted average, √-taper, version/legacy guards)
// lives in src/rl/policyMerge.ts so it's typechecked and unit-tested. This
// script is the thin CLI around it: arg parsing, file IO, and reporting.
let merged: SerializedPolicy;
let stats;
try {
  ({ merged, stats } = mergePolicies(policies));
} catch (err) {
  console.error(`[abort] ${(err as Error).message}`);
  process.exit(1);
}

writeFileSync(outPath, JSON.stringify(merged, null, 2));

console.log(`[merge] merged ${stats.mergedPolicies} policies → ${outPath}`);
console.log(`[merge]   total states: ${stats.totalStates}`);
console.log(`[merge]   shared across ≥2 workers: ${stats.sharedStates} (${(stats.sharedFraction * 100).toFixed(1)}%)`);
console.log(`[merge]   avg workers per state: ${stats.avgWorkersPerState.toFixed(2)} / ${stats.mergedPolicies}`);
console.log(`[merge]   slot-visits used (weighted): ${stats.slotsWithVisitsUsed}`);
console.log(`[merge]   reset ε → ${stats.maxEpsilon.toFixed(4)} (max across inputs)`);
console.log(`[merge]   per-worker state counts: ${policies.map((p) => Object.keys(p.qTable).length).join(', ')}`);
