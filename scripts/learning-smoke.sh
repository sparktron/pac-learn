#!/usr/bin/env bash
# Deterministic I1 regression smoke for the promoted T7 linear baseline.
#
# Runs the same single-worker configuration twice, checks byte-identical learning
# outputs (apart from elapsed wall time), then asserts conservative quality floors.
# The fixed episode budget keeps this suitable for CI; output is temporary unless
# outDir=<path> is supplied for local inspection.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VITE_NODE="$REPO_DIR/node_modules/.bin/vite-node"

if [[ ! -x "$VITE_NODE" ]]; then
  echo "[abort] vite-node is unavailable; run npm ci first" >&2
  exit 1
fi

OUT_BASE=""
for value in "$@"; do
  case "$value" in
    outDir=*) OUT_BASE="${value#outDir=}" ;;
    *) echo "[abort] unrecognized argument '$value' (expected outDir=<path>)" >&2; exit 1 ;;
  esac
done

if [[ -n "$OUT_BASE" ]]; then
  if [[ -e "$OUT_BASE" ]]; then
    echo "[abort] outDir already exists: $OUT_BASE" >&2
    exit 1
  fi
  mkdir -p "$OUT_BASE"
else
  OUT_BASE="$(mktemp -d "${TMPDIR:-/tmp}/pac-learn-learning-smoke.XXXXXX")"
  trap 'rm -rf "$OUT_BASE"' EXIT
fi

cd "$REPO_DIR"
COMMON_ARGS=(
  algorithm=linear
  ghosts=2
  seed=7
  episodes=2000
  endgameCurriculum=0.90
  stepPenalty=-0.02
  alpha=0.02
  targetSyncSteps=2000
  evalEvery=2000
  evalEpisodes=50
  evalPanels=1000000,2000000,3000000,4000000
  snapshotEvery=0
  reportEvery=0
)

for run in run-a run-b; do
  "$VITE_NODE" scripts/overnight-bench.ts -- "outDir=$OUT_BASE/$run" "${COMMON_ARGS[@]}" > "$OUT_BASE/$run.log"
done

node scripts/assert-learning-smoke.mjs "$OUT_BASE"
