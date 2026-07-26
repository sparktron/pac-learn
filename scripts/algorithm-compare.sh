#!/bin/bash
# Quick algorithm comparison: run both linear and tabular side-by-side for 30 min each
# Usage: ./scripts/algorithm-compare.sh [durationMin=30]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Parse duration arg
DURATION=30
DESC="linear-vs-tabular"
for arg in "$@"; do
  if [[ "$arg" == durationMin=* ]]; then
    DURATION="${arg#durationMin=}"
  fi
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_BASE="$REPO_DIR/bench-out/${TIMESTAMP}-${DESC}"
mkdir -p "$OUT_BASE"

echo "Algorithm Comparison: ${DURATION}min each"
echo "Output: $OUT_BASE"
echo ""

# Validated parameters (from sweep-03 final testing)
SHARED_PARAMS="
  endgameCurriculum=0.90
  stepPenalty=-0.02
"

pids=()
algorithms=("tabular" "linear")

for algo in "${algorithms[@]}"; do
  worker_out="$OUT_BASE/$algo"
  mkdir -p "$worker_out"
  log="$worker_out/bench.log"

  # Production defaults are centralized in src/rl/hyperDefaults.ts and applied
  # by overnight-bench. Only the tabular agent uses the endgame epsilon floor.
  if [[ "$algo" == "linear" ]]; then
    ALGO_PARAMS="endgameEps=0"
  else
    ALGO_PARAMS="endgameEps=0.25"
  fi

  echo "Starting $algo worker (production hyper defaults)..."

  setsid npx vite-node "$SCRIPT_DIR/overnight-bench.ts" -- \
    outDir="$worker_out" \
    seed=7 \
    algorithm=$algo \
    durationMin=$DURATION \
    reportEvery=60 \
    snapshotEvery=0 \
    evalEvery=500 \
    $SHARED_PARAMS \
    $ALGO_PARAMS \
    > "$log" 2>&1 &

  pids+=($!)
  echo "$algo: PID $!"
done

echo ""
echo "Waiting for both workers to complete..."
echo "  tail -F $OUT_BASE/*/bench.log to monitor"
echo ""

# Wait for both to finish
for p in "${pids[@]}"; do
  wait "$p" 2>/dev/null || true
done

# Print summary
echo ""
echo "========================================="
echo "COMPARISON RESULTS"
echo "========================================="
echo ""

for algo in "${algorithms[@]}"; do
  summary="$OUT_BASE/$algo/summary.json"
  if [[ -f "$summary" ]]; then
    echo "$algo:"
    jq '.config.algorithm, .config.alpha, .episodes, .trainingWinRate, .meanScoreLast1000' "$summary" | tr '\n' ' '
    echo ""
  fi
done

echo ""
echo "Full summaries:"
for algo in "${algorithms[@]}"; do
  summary="$OUT_BASE/$algo/summary.json"
  if [[ -f "$summary" ]]; then
    echo "--- $algo ---"
    jq '.' "$summary" | head -20
    echo ""
  fi
done
