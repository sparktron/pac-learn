#!/usr/bin/env bash
# Parallel federated bench — N independent workers train simultaneously, then
# their Q-tables are merged into one policy at the end.
#
# Why: a single overnight-bench run is single-threaded at ~70k steps/sec. With
# 32 cores idle, that's a 32× speedup left on the table. Each worker explores
# a different region of state-space (different seeds), so the merged Q-table
# is also broader (ensemble of N learners) — better than a single 32×-longer run.
#
# Usage:
#   ./scripts/run-parallel.sh [-j N] [bench args ...]
#
# Options:
#   -j N         number of parallel workers (default: nproc, max 32)
#   bench args   anything overnight-bench.ts accepts. Common picks:
#                   durationMin=60     run each worker for 60 min
#                   ghosts=3 maxSteps=800
#                   endgameCurriculum=0.2 endgameEps=0.4
#                   loadPolicy=path/to/policy.json
#                The script auto-assigns seeds (worker i gets seed=7+i*1000)
#                and outDir; do not pass those manually.
#
# Examples:
#   # Default: 32 workers, each for 60 minutes — total experience = ~32 hours
#   ./scripts/run-parallel.sh durationMin=60
#
#   # 16 workers, endgame-curriculum on, 30-min each
#   ./scripts/run-parallel.sh -j 16 durationMin=30 endgameCurriculum=0.2
#
#   # Resume from previous merged policy
#   ./scripts/run-parallel.sh durationMin=120 loadPolicy=bench-out/parallel-XXX/policy-merged.json
#
# Output:
#   bench-out/parallel-<timestamp>/worker-NN/  — per-worker policy + CSV + log
#   bench-out/parallel-<timestamp>/policy-merged.json  — averaged Q-table
#   bench-out/parallel-<timestamp>/summary.txt — per-worker stats
#
# Ctrl-C kills all workers; each flushes its policy + summary before exit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default worker count: min(nproc, 32). 32 is a soft cap because we've seen
# diminishing returns past that — disk/log contention starts to bite.
DEFAULT_N=$(nproc 2>/dev/null || echo 8)
if [[ $DEFAULT_N -gt 32 ]]; then DEFAULT_N=32; fi
NUM_WORKERS=$DEFAULT_N

# ---- parse args ----
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j)      NUM_WORKERS="$2"; shift 2;;
    -j=*)    NUM_WORKERS="${1#-j=}"; shift;;
    --parallel) NUM_WORKERS="$2"; shift 2;;
    *)       PASS_ARGS+=("$1"); shift;;
  esac
done

# Validate
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$NUM_WORKERS" -lt 1 ]]; then
  echo "error: -j requires a positive integer, got '$NUM_WORKERS'" >&2
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_BASE="$REPO_DIR/bench-out/parallel-$TIMESTAMP"
mkdir -p "$OUT_BASE"

echo "[setup] $NUM_WORKERS workers → $OUT_BASE"
echo "[setup] pass-through args: ${PASS_ARGS[*]:-(none)}"
echo ""

pids=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
  worker_id=$(printf '%02d' "$i")
  worker_out="$OUT_BASE/worker-$worker_id"
  mkdir -p "$worker_out"
  log="$worker_out/bench.log"
  # Each worker gets a unique seed (so they explore different trajectories) and
  # a quieter reportEvery (we don't want 32 progress lines per second).
  npx vite-node "$SCRIPT_DIR/overnight-bench.ts" -- \
    outDir="$worker_out" \
    seed=$((7 + i * 1000)) \
    reportEvery=120 \
    snapshotEvery=600 \
    "${PASS_ARGS[@]}" \
    > "$log" 2>&1 &
  pids+=($!)
done

echo "[parallel] launched ${#pids[@]} workers"
echo "[parallel] tail worker logs:  tail -F $OUT_BASE/worker-*/bench.log"
echo "[parallel] watch progress:    watch -n5 'tail -n1 $OUT_BASE/worker-*/bench.log | grep -E \"t=|done\"'"
echo ""

# Forward signals: on Ctrl-C, terminate all workers and wait for their cleanup.
cleanup() {
  echo ""
  echo "[abort] sending SIGTERM to all workers..."
  for p in "${pids[@]}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait "${pids[@]}" 2>/dev/null || true
  echo "[abort] workers done — partial results in $OUT_BASE"
  exit 1
}
trap cleanup INT TERM

# Wait for all workers
START_TS=$(date +%s)
FAILED=()
for p in "${pids[@]}"; do
  if ! wait "$p"; then
    FAILED+=("$p")
  fi
done
END_TS=$(date +%s)
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))

echo ""
echo "════════════════════════════════════════════════════════"
echo "  All ${#pids[@]} workers complete — elapsed ${ELAPSED_MIN}m"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  WARN: ${#FAILED[@]} workers exited non-zero (PIDs: ${FAILED[*]})"
  echo "  policies still merged (each worker writes on signal)"
fi
echo "════════════════════════════════════════════════════════"

# ---- merge ----
echo ""
echo "[merge] combining policies from ${#pids[@]} workers..."
npx vite-node "$SCRIPT_DIR/merge-policies.ts" -- \
  out="$OUT_BASE/policy-merged.json" \
  "$OUT_BASE"/worker-*/policy-latest.json

# ---- summary table ----
SUMMARY="$OUT_BASE/summary.txt"
{
  echo "── Parallel run summary ─────────────────────────────"
  echo "Workers: ${#pids[@]}    Wall time: ${ELAPSED_MIN}m    Pass-through: ${PASS_ARGS[*]}"
  echo ""
  printf "%-12s %10s %10s %10s %10s %12s\n" "worker" "episodes" "qStates" "trainWins" "score/1k" "minPellets"
  for s in "$OUT_BASE"/worker-*/summary.json; do
    [[ -f "$s" ]] || continue
    label=$(basename "$(dirname "$s")")
    ep=$(jq -r '.episodes // "?"'          "$s" 2>/dev/null || echo "?")
    qs=$(jq -r '.qTableSize // "?"'        "$s" 2>/dev/null || echo "?")
    wins=$(jq -r '.trainingWins // 0'      "$s" 2>/dev/null || echo "?")
    sc=$(jq -r '.meanScoreLast1000 // "?"' "$s" 2>/dev/null || echo "?")
    # Best minPelletsLeft from the worker's evals.csv (column 7)
    mp=$(tail -n +2 "$(dirname "$s")/evals.csv" 2>/dev/null \
         | awk -F, 'BEGIN{m=999999} {if($7+0<m) m=$7} END{print (m==999999?"-":m)}' \
         || echo "-")
    printf "%-12s %10s %10s %10s %10.1f %12s\n" "$label" "$ep" "$qs" "$wins" "$sc" "$mp"
  done
  echo ""
  echo "Merged policy: $OUT_BASE/policy-merged.json"
} | tee "$SUMMARY"
