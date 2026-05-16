/**
 * Q-table merger for parallel federated training.
 *
 * Averages Q-values across N policy JSON files. Used by run-parallel.sh
 * after spawning multiple independent training workers — each worker
 * explores a different region of state-space (different seeds), and the
 * merged Q-table is an ensemble that benefits from all of their experience.
 *
 * Merge semantics:
 *   • For each state-action present in any worker's qTable, the merged
 *     value is the arithmetic mean across workers that have that state.
 *   • States seen by only 1 worker keep that worker's value (no averaging).
 *   • A state is "present" if the worker added it to qTable at some point —
 *     which happens on the first observe() call for that state, not only
 *     after an update. This is acceptable in practice because workers
 *     observe states they actually visit during exploration.
 *
 * Usage:
 *   npx vite-node scripts/merge-policies.ts -- out=<path> <p1.json> <p2.json> ...
 *
 * Output: a SerializedPolicy at `out=` with averaged Q-values. Metadata
 * (mazeId, numGhostsEncoded, observationKeyVersion) is copied from the
 * first input policy; timestamp is set to merge time.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import type { SerializedPolicy } from '../src/rl/qlearning';

const rawArgs = process.argv.slice(2).filter((a) => a !== '--');
const outArg = rawArgs.find((a) => a.startsWith('out='));
const outPath = outArg ? outArg.slice('out='.length) : './merged-policy.json';
const inputs = rawArgs.filter((a) => !a.startsWith('out='));

if (inputs.length === 0) {
  console.error('usage: merge-policies.ts -- out=<path> <policy1.json> <policy2.json> ...');
  process.exit(1);
}

const policies: SerializedPolicy[] = [];
for (const path of inputs) {
  try {
    policies.push(JSON.parse(readFileSync(path, 'utf-8')) as SerializedPolicy);
  } catch (err) {
    console.error(`[skip] could not read ${path}: ${(err as Error).message}`);
  }
}

if (policies.length === 0) {
  console.error('no policies could be loaded');
  process.exit(1);
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

const sums = new Map<string, [number, number, number, number]>();
const counts = new Map<string, number>();

for (const policy of policies) {
  for (const [key, values] of Object.entries(policy.qTable)) {
    let sum = sums.get(key);
    if (!sum) {
      sum = [0, 0, 0, 0];
      sums.set(key, sum);
    }
    for (let i = 0; i < 4; i += 1) sum[i] += values[i] ?? 0;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
}

const mergedQ: Record<string, number[]> = {};
let sharedStates = 0;
for (const [key, sum] of sums.entries()) {
  const c = counts.get(key) ?? 1;
  if (c > 1) sharedStates += 1;
  mergedQ[key] = sum.map((v) => v / c);
}

const totalCoverage = Array.from(counts.values()).reduce((a, b) => a + b, 0);
const avgWorkersPerState = totalCoverage / counts.size;
const sharedFraction = sharedStates / counts.size;

const merged: SerializedPolicy = {
  algorithm: policies[0].algorithm,
  mazeId: policies[0].mazeId,
  timestamp: new Date().toISOString(),
  numGhostsEncoded: policies[0].numGhostsEncoded,
  observationKeyVersion: policies[0].observationKeyVersion,
  hyper: policies[0].hyper,
  qTable: mergedQ,
};

writeFileSync(outPath, JSON.stringify(merged, null, 2));

console.log(`[merge] merged ${policies.length} policies → ${outPath}`);
console.log(`[merge]   total states: ${counts.size}`);
console.log(`[merge]   shared across ≥2 workers: ${sharedStates} (${(sharedFraction * 100).toFixed(1)}%)`);
console.log(`[merge]   avg workers per state: ${avgWorkersPerState.toFixed(2)} / ${policies.length}`);
console.log(`[merge]   per-worker state counts: ${policies.map((p) => Object.keys(p.qTable).length).join(', ')}`);
