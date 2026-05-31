#!/usr/bin/env bash
# Fast smoke test for federated/parallel training.
# Runs two short workers, merges their policy outputs, and validates the merged
# policy is loadable and contains both Q-values and visit counts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_BASE="$REPO_DIR/bench-out/parallel-merge-smoke"

rm -rf "$OUT_BASE"

"$SCRIPT_DIR/run-parallel.sh" \
  -j 2 \
  desc=parallel-merge-smoke \
  outBase="$OUT_BASE" \
  episodes=2 \
  evalEvery=0 \
  snapshotEvery=0 \
  reportEvery=9999

node --input-type=module - "$OUT_BASE" <<'NODE'
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const outBase = process.argv[2];
const mergedPath = join(outBase, 'policy-merged.json');
if (!existsSync(mergedPath)) {
  throw new Error(`missing merged policy: ${mergedPath}`);
}

const merged = JSON.parse(readFileSync(mergedPath, 'utf8'));
const stateCount = Object.keys(merged.qTable ?? {}).length;
const visitCount = Object.keys(merged.visitTable ?? {}).length;
if (merged.algorithm !== 'qlearning') {
  throw new Error(`expected qlearning policy, got ${merged.algorithm}`);
}
if (stateCount === 0) {
  throw new Error('merged policy has no Q-table states');
}
if (visitCount !== stateCount) {
  throw new Error(`visitTable state count ${visitCount} does not match qTable state count ${stateCount}`);
}

for (const id of ['00', '01']) {
  const workerPolicy = join(outBase, `worker-${id}`, 'policy-latest.json');
  if (!existsSync(workerPolicy)) {
    throw new Error(`missing worker policy: ${workerPolicy}`);
  }
}

console.log(`[smoke] merged ${stateCount} states from 2 workers at ${mergedPath}`);
NODE
