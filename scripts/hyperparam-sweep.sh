#!/bin/bash
# Hyperparameter sweep: run N workers with different configs for fixed duration
# Usage: ./scripts/hyperparam-sweep.sh [durationMin=30] [desc=sweep-001] [algorithm=tabular]
#
# Creates: bench-out/YYYYMMDD-HHMMSS-<desc>/
#   config-matrix.txt        which worker tested which params
#   worker-00-params.txt     actual params for worker-00
#   ...
#   results.txt              final summary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Parse args
DURATION=30
DESC="sweep"
ALGORITHM="tabular"
PASS_ARGS=()

for arg in "$@"; do
  if [[ "$arg" == durationMin=* ]]; then
    DURATION="${arg#durationMin=}"
  elif [[ "$arg" == desc=* ]]; then
    DESC="${arg#desc=}"
  elif [[ "$arg" == algorithm=* ]]; then
    ALGORITHM="${arg#algorithm=}"
  else
    PASS_ARGS+=("$arg")
  fi
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_BASE="$REPO_DIR/bench-out/${TIMESTAMP}-${DESC}"
mkdir -p "$OUT_BASE"

echo "Hyperparameter sweep: $DURATION minutes"
echo "Output: $OUT_BASE"
echo ""

# Define hyperparameter matrix (16 combinations for 16 workers)
# Sweep-03: Diagnose low win-rate disconnect
# Problem: Training finds 0.33% wins but eval (greedy, fresh maze) = 0%
# Hypothesis: stepPenalty=-0.1 too harsh, discourages full-maze exploration
# Test: step penalty + curriculum combinations
declare -a MATRIX=(
  "stepPenalty=-0.01 endgameCurriculum=0.90"
  "stepPenalty=-0.02 endgameCurriculum=0.90"
  "stepPenalty=-0.05 endgameCurriculum=0.90"
  "stepPenalty=-0.10 endgameCurriculum=0.90"
  "stepPenalty=-0.01 endgameCurriculum=0.70"
  "stepPenalty=-0.02 endgameCurriculum=0.70"
  "stepPenalty=-0.05 endgameCurriculum=0.70"
  "stepPenalty=-0.10 endgameCurriculum=0.70"
  "stepPenalty=-0.01 endgameCurriculum=0.50"
  "stepPenalty=-0.02 endgameCurriculum=0.50"
  "stepPenalty=-0.05 endgameCurriculum=0.50"
  "stepPenalty=-0.10 endgameCurriculum=0.50"
  "stepPenalty=-0.00 endgameCurriculum=0.90"
  "stepPenalty=-0.15 endgameCurriculum=0.90"
  "stepPenalty=-0.20 endgameCurriculum=0.90"
  "alpha=0.1 stepPenalty=-0.02 endgameCurriculum=0.90"
)

# Log matrix (dynamic - just show params as-is)
{
  echo "Hyperparameter Sweep Configuration"
  echo "Duration: ${DURATION}min"
  echo ""
  echo "Worker | Parameters"
  echo "-------|----------------------------------"
  for i in "${!MATRIX[@]}"; do
    worker_id=$(printf '%02d' "$i")
    params="${MATRIX[$i]}"
    printf "%02d    | %s\n" "$i" "$params"
  done
} | tee "$OUT_BASE/config-matrix.txt"

echo ""
echo "Starting workers..."

pids=()
for i in "${!MATRIX[@]}"; do
  worker_id=$(printf '%02d' "$i")
  worker_out="$OUT_BASE/worker-$worker_id"
  mkdir -p "$worker_out"
  
  params="${MATRIX[$i]}"
  
  # Save params to file
  echo "$params" > "$worker_out/params.txt"
  
  log="$worker_out/bench.log"
  
  # Launch worker with its specific params
  setsid npx vite-node "$SCRIPT_DIR/overnight-bench.ts" -- \
    outDir="$worker_out" \
    seed=$((7 + i * 1000)) \
    durationMin=$DURATION \
    reportEvery=120 \
    snapshotEvery=0 \
    evalEvery=500 \
    algorithm=$ALGORITHM \
    ${params} \
    "${PASS_ARGS[@]}" \
    > "$log" 2>&1 &
  
  pids+=($!)
  echo "worker-$worker_id: $params"
done

echo ""
echo "All $((${#pids[@]})) workers started"
echo "Monitoring for $DURATION minutes..."
echo ""

# Wait for all workers
for p in "${pids[@]}"; do
  wait "$p" 2>/dev/null || true
done

echo ""
echo "Sweep complete. Analyzing results..."
echo ""

# Analysis
{
  echo "========================================="
  echo "SWEEP RESULTS"
  echo "========================================="
  echo ""
  echo "Worker | Parameters                         | Episodes    | Wins   | Win %"
  echo "-------|----------------------------------|-------------|--------|-------"

  best_win_pct=0
  best_worker=""
  best_params=""

  for i in "${!MATRIX[@]}"; do
    worker_id=$(printf '%02d' "$i")
    csv="$OUT_BASE/worker-$worker_id/episodes.csv"

    if [[ -f "$csv" ]]; then
      lines=$(wc -l < "$csv")
      eps=$((lines - 1))
      wins=$(grep "won" "$csv" | wc -l)
      win_pct=$(awk "BEGIN {printf \"%.3f\", $wins*100/$eps}")

      # Get params
      params=$(cat "$OUT_BASE/worker-$worker_id/params.txt" 2>/dev/null || echo "unknown")

      printf "%02d    | %-32s | %11d | %6d | %6.3f%%\n" \
        "$i" "$params" "$eps" "$wins" "$win_pct"

      # Track best
      if (( $(echo "$win_pct > $best_win_pct" | bc -l) )); then
        best_win_pct=$win_pct
        best_worker="worker-$worker_id"
        best_params=$params
      fi
    else
      echo "$worker_id | (not yet complete)"
    fi
  done

  echo ""
  echo "========================================="
  echo "BEST PERFORMER:"
  echo "  Worker: $best_worker"
  echo "  Win rate: $best_win_pct%"
  echo "  Parameters: $best_params"
  echo "========================================="
} | tee "$OUT_BASE/results.txt"

cat "$OUT_BASE/results.txt"

