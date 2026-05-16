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
#   ./scripts/run-parallel.sh [-j N] [desc=NAME] [--clean] [bench args ...]
#
# Options:
#   -j N         number of parallel workers (default: nproc, max 32)
#   desc=<s>     short label appended to the top-level folder name
#                (default: "parallel"). Use to tag experiments,
#                e.g. desc=ab-3a-32x. Non-alphanumeric chars become '_'.
#   --clean      if target top-level already exists with worker folders,
#                wipe it first. Without --clean, the script aborts on
#                collision — prevents merging across separate executions.
#   bench args   anything overnight-bench.ts accepts. Common picks:
#                  durationMin=60       per-worker duration
#                  ghosts=3 maxSteps=800
#                  endgameCurriculum=0.2 endgameEps=0.4
#                  loadPolicy=path/to/policy.json
#                Script auto-assigns seeds (worker i → seed=7+i*1000)
#                and outDir; do not pass those manually.
#
# Examples:
#   # Default: nproc workers (up to 32), 60-min each, total ~32 worker-hours
#   ./scripts/run-parallel.sh durationMin=60
#
#   # Labeled AB test, 16 workers, curriculum on
#   ./scripts/run-parallel.sh -j 16 desc=ab-3a durationMin=30 endgameCurriculum=0.2
#
#   # Resume from a merged policy
#   ./scripts/run-parallel.sh durationMin=120 desc=resume \
#     loadPolicy=bench-out/<prior>/policy-merged.json
#
# Output structure:
#   bench-out/<YYYYMMDD-HHMMSS>-<desc>/
#     worker-00/                    per-worker policy + CSVs + log
#     worker-01/                    …
#     policy-merged.json            averaged Q-table across all workers
#     summary.txt                   per-worker stats
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
DESC="parallel"
CLEAN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j)         NUM_WORKERS="$2"; shift 2;;
    -j=*)       NUM_WORKERS="${1#-j=}"; shift;;
    --parallel) NUM_WORKERS="$2"; shift 2;;
    desc=*)     DESC="${1#desc=}"; shift;;
    --clean)    CLEAN=1; shift;;
    *)          PASS_ARGS+=("$1"); shift;;
  esac
done

# Validate
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$NUM_WORKERS" -lt 1 ]]; then
  echo "error: -j requires a positive integer, got '$NUM_WORKERS'" >&2
  exit 1
fi

# Sanitize desc: strip anything that isn't alnum/dash/underscore
DESC=$(echo "$DESC" | tr -c 'a-zA-Z0-9_-' '_' | sed 's/_*$//')
if [[ -z "$DESC" ]]; then DESC="parallel"; fi

# Pre-flight: validate loadPolicy= argument BEFORE spawning workers.
# Otherwise 32 workers all crash with ENOENT and only the post-mortem
# tells you why — and the merge step then chokes on the empty glob.
for arg in "${PASS_ARGS[@]}"; do
  if [[ "$arg" == loadPolicy=* ]]; then
    policy_path="${arg#loadPolicy=}"
    if [[ ! -f "$policy_path" ]]; then
      echo "[abort] loadPolicy points to a missing file: '$policy_path'" >&2
      echo "        Verify with:  ls -l $policy_path" >&2
      echo "        Available merged policies:" >&2
      ls -1t "$REPO_DIR"/bench-out/*/policy-merged.json 2>/dev/null | head -10 | sed 's|^|          |' >&2 || echo "          (none found)" >&2
      exit 1
    fi
    echo "[setup] loadPolicy verified: $policy_path"
  fi
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_BASE="$REPO_DIR/bench-out/${TIMESTAMP}-${DESC}"

# Failsafe (Option A: fail-fast on collision; Option B: --clean to wipe)
if [[ -d "$OUT_BASE" ]]; then
  existing=$(find "$OUT_BASE" -maxdepth 1 -mindepth 1 -type d -name 'worker-*' 2>/dev/null | wc -l)
  if [[ $existing -gt 0 ]]; then
    if [[ $CLEAN -eq 1 ]]; then
      echo "[setup] --clean: removing $existing existing worker folder(s) under $OUT_BASE"
      rm -rf "$OUT_BASE"
    else
      echo "[abort] $OUT_BASE already contains $existing worker folder(s)."
      echo "        Pass --clean to wipe, or use a different desc= to start fresh." >&2
      exit 1
    fi
  fi
fi
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
# Collect existing policy files explicitly so an empty glob fails loudly
# instead of being passed as a literal "worker-*/policy-latest.json" pattern.
shopt -s nullglob
POLICY_FILES=("$OUT_BASE"/worker-*/policy-latest.json)
shopt -u nullglob
if [[ ${#POLICY_FILES[@]} -eq 0 ]]; then
  echo "[merge] no policy-latest.json files found — workers likely crashed before snapshot."
  echo "[merge] inspect:  head -20 $OUT_BASE/worker-00/bench.log"
else
  npx vite-node "$SCRIPT_DIR/merge-policies.ts" -- \
    out="$OUT_BASE/policy-merged.json" \
    "${POLICY_FILES[@]}"
fi

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
