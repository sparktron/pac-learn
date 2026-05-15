#!/usr/bin/env bash
# Comprehensive preset sweep — trains all 4 reward presets across permutations
# of env/agent variables designed to surface edge-case bugs.
#
# Usage:
#   ./scripts/run-sweep.sh [outRoot] [-j N | --parallel N]
#
#   outRoot      base directory for outputs (default: ./bench-out/sweep-<timestamp>)
#   -j N         run N training jobs in parallel (default: 1, sequential)
#   --parallel N same as -j N
#
# Each run trains for 20 minutes and logs to <outRoot>/<group>/<run-id>/bench.log.
# In parallel mode stdout shows only start/finish lines; tail the log file for detail.
#
# Ctrl-C sends SIGINT to all running jobs (each flushes its policy + summary),
# then prints results for completed runs and exits.
#
# Final output: <outRoot>/report.tsv — one row per run, importable into any spreadsheet.
#
# Groups (56 runs total):
#   A  baseline        — 4 presets, all defaults                              (4)
#   B  ghost-count     — 4 presets × ghosts ∈ {1, 3, 4}                     (12)
#   C  capture-rules   — 4 presets × capture=touch                            (4)
#   D  power-pellets   — 4 presets × powerPellets=false                       (4)
#   E  ghost-speed     — 4 presets × ghostSpeed ∈ {0.5, 1.5}                 (8)
#   F  exploration     — 4 presets × 3 epsilon regimes                       (12)
#   G  episode-length  — 4 presets × maxSteps ∈ {200, 2000}                  (8)
#   H  illegal-move    — 4 presets × illegalMove=noop                         (4)
#
# Wall-clock time estimates (56 runs × 20 min):
#   -j 8  →  ~2.3 h      -j 16 →  ~1.2 h
#   -j 14 →  ~80 min     -j 28 →  ~40 min

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH="npx vite-node $SCRIPT_DIR/overnight-bench.ts --"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# ── argument parsing ─────────────────────────────────────────────────────────
MAX_JOBS=1
ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--parallel) MAX_JOBS="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) ROOT="$1"; shift ;;
  esac
done
ROOT="${ROOT:-$REPO_DIR/bench-out/sweep-$TIMESTAMP}"

if ! [[ "$MAX_JOBS" =~ ^[0-9]+$ ]] || (( MAX_JOBS < 1 )); then
  echo "Error: -j requires a positive integer" >&2; exit 1
fi

# ── constants ────────────────────────────────────────────────────────────────
PRESETS=(default ghost-hunting pellet-collection survival)
DURATION=20         # minutes per run
MAX_STEPS=1000      # default episode step cap
REPORT_INTERVAL=60  # seconds between heartbeat lines in sequential mode
EVAL_EVERY=1000     # episodes between greedy eval passes
EVAL_EPS=30         # episodes per eval pass
TOTAL=56

mkdir -p "$ROOT"
REPORT_TSV="$ROOT/report.tsv"
printf 'group\trun_id\tpreset\tghosts\tcapture\tpowerPellets\tghostSpeed\tillegalMove\tmaxSteps\teps\tepsDecay\tepisodes\tqStates\tscore_last1k\tlen_last1k\n' \
  > "$REPORT_TSV"

# ── job tracking ─────────────────────────────────────────────────────────────
declare -A JOB_GROUP=()
declare -A JOB_RUN_ID=()
declare -A JOB_OUT=()
DISPATCHED=0
FINISHED=0
FAILED=0
START_TIME=$(date +%s)

# ── helpers ──────────────────────────────────────────────────────────────────

append_report() {
  local group="$1" run_id="$2" out="$3"
  local summary="$out/summary.json"
  if [[ ! -f "$summary" ]]; then
    printf '%s\t%s\t(no summary)\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n' \
      "$group" "$run_id" >> "$REPORT_TSV"
    return
  fi
  local preset ghosts capture pp gs im ms eps decay episodes qstates score len
  preset=$(  jq -r '.config.preset          // "?"' "$summary")
  ghosts=$(  jq -r '.config.ghosts          // "?"' "$summary")
  capture=$( jq -r '.config.capture         // "?"' "$summary")
  pp=$(      jq -r '.config.powerPellets    // "?"' "$summary")
  gs=$(      jq -r '.config.ghostSpeed      // "?"' "$summary")
  im=$(      jq -r '.config.illegalMove     // "?"' "$summary")
  ms=$(      jq -r '.config.maxSteps        // "?"' "$summary")
  eps=$(     jq -r '.config.eps             // "?"' "$summary")
  decay=$(   jq -r '.config.epsDecay        // "?"' "$summary")
  episodes=$(jq -r '.episodes               // "?"' "$summary")
  qstates=$( jq -r '.qTableSize             // "?"' "$summary")
  score=$(   jq -r '.meanScoreLast1000      // "?"' "$summary")
  len=$(     jq -r '.meanLenLast1000        // "?"' "$summary")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$group" "$run_id" "$preset" "$ghosts" "$capture" "$pp" "$gs" "$im" \
    "$ms" "$eps" "$decay" "$episodes" "$qstates" "$score" "$len" >> "$REPORT_TSV"
}

reap_finished() {
  # Poll running PIDs for completion; reap and record each one found done.
  local pid
  for pid in "${!JOB_GROUP[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      set +e; wait "$pid" 2>/dev/null; local rc=$?; set -e
      local rid="${JOB_RUN_ID[$pid]}"
      local grp="${JOB_GROUP[$pid]}"
      local out="${JOB_OUT[$pid]}"
      FINISHED=$((FINISHED + 1))
      [[ $rc -ne 0 ]] && FAILED=$((FAILED + 1))
      append_report "$grp" "$rid" "$out"
      local score; score=$(jq -r '.meanScoreLast1000 // "?"' "$out/summary.json" 2>/dev/null || echo "?")
      local elapsed=$(( ($(date +%s) - START_TIME) / 60 ))
      printf '[done %d/%d @ %dm] %-35s  score=%s\n' "$FINISHED" "$DISPATCHED" "$elapsed" "$rid" "$score"
      unset "JOB_GROUP[$pid]" "JOB_RUN_ID[$pid]" "JOB_OUT[$pid]"
    fi
  done
}

wait_for_slot() {
  # Block until a job slot is free.
  while (( ${#JOB_GROUP[@]} >= MAX_JOBS )); do
    reap_finished
    (( ${#JOB_GROUP[@]} >= MAX_JOBS )) && sleep 0.5
  done
}

wait_all() {
  while (( ${#JOB_GROUP[@]} > 0 )); do
    reap_finished
    (( ${#JOB_GROUP[@]} > 0 )) && sleep 0.5
  done
}

run() {
  local group="$1" run_id="$2"; shift 2
  local out="$ROOT/$group/$run_id"
  mkdir -p "$out"
  local log="$out/bench.log"

  wait_for_slot
  DISPATCHED=$((DISPATCHED + 1))

  if (( MAX_JOBS == 1 )); then
    # Sequential: stream output directly to stdout (and log file).
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────────"
    printf "│  [%d/%d]  %s\n" "$DISPATCHED" "$TOTAL" "$run_id"
    echo "└─────────────────────────────────────────────────────────────────────"
    set +e
    $BENCH outDir="$out" durationMin="$DURATION" \
      reportEvery="$REPORT_INTERVAL" evalEvery="$EVAL_EVERY" \
      evalEpisodes="$EVAL_EPS" snapshotEvery=0 "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    FINISHED=$((FINISHED + 1))
    [[ $rc -ne 0 ]] && FAILED=$((FAILED + 1))
    append_report "$group" "$run_id" "$out"
  else
    # Parallel: background the job, redirect output to log.
    $BENCH outDir="$out" durationMin="$DURATION" \
      reportEvery="$REPORT_INTERVAL" evalEvery="$EVAL_EVERY" \
      evalEpisodes="$EVAL_EPS" snapshotEvery=0 "$@" > "$log" 2>&1 &
    local pid=$!
    JOB_GROUP[$pid]="$group"
    JOB_RUN_ID[$pid]="$run_id"
    JOB_OUT[$pid]="$out"
    local elapsed=$(( ($(date +%s) - START_TIME) / 60 ))
    printf '[start %d/%d @ %dm] %-35s  log → %s\n' \
      "$DISPATCHED" "$TOTAL" "$elapsed" "$run_id" "$log"
  fi
}

print_summary() {
  wait_all
  local elapsed=$(( ($(date +%s) - START_TIME) / 60 ))
  echo ""
  echo "══════════════════════════════════════════════════════════════════════"
  printf "  Sweep complete — %d/%d succeeded, %d failed, %dm elapsed\n" \
    "$((FINISHED - FAILED))" "$FINISHED" "$FAILED" "$elapsed"
  printf "  Parallelism:  -j %d\n" "$MAX_JOBS"
  echo "  Results:      $ROOT"
  echo "  Report TSV:   $REPORT_TSV"
  echo "══════════════════════════════════════════════════════════════════════"
  echo ""
  printf '%-35s  %9s  %7s  %8s\n' "run_id" "score/1k" "len/1k" "qStates"
  echo "───────────────────────────────────────────────────────────────────────"
  local grp rid preset ghosts cap pp gs im ms eps dec ep qs sc ln
  while IFS=$'\t' read -r grp rid preset ghosts cap pp gs im ms eps dec ep qs sc ln; do
    [[ "$grp" == "group" ]] && continue
    printf '%-35s  %9s  %7s  %8s\n' "$rid" "$sc" "$ln" "$qs"
  done < "$REPORT_TSV"
  echo "───────────────────────────────────────────────────────────────────────"
}

cleanup() {
  echo ""
  echo "[interrupt] signalling ${#JOB_GROUP[@]} running job(s) to flush and exit..."
  local pid
  for pid in "${!JOB_GROUP[@]}"; do
    kill -SIGINT "$pid" 2>/dev/null || true
  done
  # Give jobs time to flush policy + summary (bench handles SIGINT gracefully).
  local waited=0
  while (( ${#JOB_GROUP[@]} > 0 && waited < 15 )); do
    reap_finished; sleep 1; waited=$((waited + 1))
  done
  # Force-kill anything still running.
  for pid in "${!JOB_GROUP[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    append_report "${JOB_GROUP[$pid]}" "${JOB_RUN_ID[$pid]}" "${JOB_OUT[$pid]}"
    unset "JOB_GROUP[$pid]" "JOB_RUN_ID[$pid]" "JOB_OUT[$pid]"
  done
  print_summary
  exit 1
}
trap cleanup INT TERM

# ── announce ─────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
printf "  PAC-LEARN SWEEP  —  %d runs × %d min  —  -j %d  (≈ %d min wall-clock)\n" \
  "$TOTAL" "$DURATION" "$MAX_JOBS" "$(( TOTAL * DURATION / MAX_JOBS ))"
echo "  Output: $ROOT"
echo "══════════════════════════════════════════════════════════════════════"

# ── Group A: baselines ────────────────────────────────────────────────────────
echo ""; echo "── GROUP A: baselines (all defaults) ──"
for p in "${PRESETS[@]}"; do
  run "A-baseline" "A-${p}" \
    preset="$p" ghosts=2 maxSteps="$MAX_STEPS" ghostSpeed=0.95 \
    capture=tile powerPellets=true illegalMove=stay \
    eps=0.5 epsDecay=0.999 epsMin=0.05
done

# ── Group B: ghost count ──────────────────────────────────────────────────────
echo ""; echo "── GROUP B: ghost count — 4 presets × {1, 3, 4} ghosts ──"
for p in "${PRESETS[@]}"; do
  for g in 1 3 4; do
    run "B-ghost-count" "B-${p}-g${g}" \
      preset="$p" ghosts="$g" maxSteps="$MAX_STEPS" ghostSpeed=0.95 \
      capture=tile powerPellets=true illegalMove=stay \
      eps=0.5 epsDecay=0.999 epsMin=0.05
  done
done

# ── Group C: capture rules ────────────────────────────────────────────────────
echo ""; echo "── GROUP C: capture=touch — 4 presets ──"
for p in "${PRESETS[@]}"; do
  run "C-capture" "C-${p}-touch" \
    preset="$p" ghosts=2 maxSteps="$MAX_STEPS" ghostSpeed=0.95 \
    capture=touch powerPellets=true illegalMove=stay \
    eps=0.5 epsDecay=0.999 epsMin=0.05
done

# ── Group D: power pellets off ────────────────────────────────────────────────
echo ""; echo "── GROUP D: powerPellets=false — 4 presets ──"
for p in "${PRESETS[@]}"; do
  run "D-no-power" "D-${p}-nopow" \
    preset="$p" ghosts=2 maxSteps="$MAX_STEPS" ghostSpeed=0.95 \
    capture=tile powerPellets=false illegalMove=stay \
    eps=0.5 epsDecay=0.999 epsMin=0.05
done

# ── Group E: ghost speed ──────────────────────────────────────────────────────
echo ""; echo "── GROUP E: ghost speed — 4 presets × {0.5, 1.5} ──"
for p in "${PRESETS[@]}"; do
  for gs in 0.5 1.5; do
    tag="${gs/./p}"
    run "E-ghost-speed" "E-${p}-gs${tag}" \
      preset="$p" ghosts=2 maxSteps="$MAX_STEPS" ghostSpeed="$gs" \
      capture=tile powerPellets=true illegalMove=stay \
      eps=0.5 epsDecay=0.999 epsMin=0.05
  done
done

# ── Group F: exploration regimes ──────────────────────────────────────────────
echo ""; echo "── GROUP F: exploration — 4 presets × 3 regimes ──"
echo "     F1 low+nodecay: stays near-greedy (exposes optimistic-init leftover)"
echo "     F2 high+fast:   heavy explore then snap to greedy (happy path)"
echo "     F3 high+nodecay: near-random forever (sanity floor)"
for p in "${PRESETS[@]}"; do
  run "F-exploration" "F-${p}-low-nodecay" \
    preset="$p" ghosts=2 maxSteps="$MAX_STEPS" ghostSpeed=0.95 \
    capture=tile powerPellets=true illegalMove=stay \
    eps=0.05 epsDecay=1.0 epsMin=0.05

  run "F-exploration" "F-${p}-high-fastdecay" \
    preset="$p" ghosts=2 maxSteps="$MAX_STEPS" ghostSpeed=0.95 \
    capture=tile powerPellets=true illegalMove=stay \
    eps=0.9 epsDecay=0.99 epsMin=0.05

  run "F-exploration" "F-${p}-high-nodecay" \
    preset="$p" ghosts=2 maxSteps="$MAX_STEPS" ghostSpeed=0.95 \
    capture=tile powerPellets=true illegalMove=stay \
    eps=0.9 epsDecay=1.0 epsMin=0.9
done

# ── Group G: episode length ───────────────────────────────────────────────────
echo ""; echo "── GROUP G: episode length — 4 presets × {200, 2000} steps ──"
for p in "${PRESETS[@]}"; do
  for ms in 200 2000; do
    run "G-ep-length" "G-${p}-ms${ms}" \
      preset="$p" ghosts=2 maxSteps="$ms" ghostSpeed=0.95 \
      capture=tile powerPellets=true illegalMove=stay \
      eps=0.5 epsDecay=0.999 epsMin=0.05
  done
done

# ── Group H: illegal move mode ────────────────────────────────────────────────
echo ""; echo "── GROUP H: illegalMove=noop — 4 presets ──"
for p in "${PRESETS[@]}"; do
  run "H-illegal-move" "H-${p}-noop" \
    preset="$p" ghosts=2 maxSteps="$MAX_STEPS" ghostSpeed=0.95 \
    capture=tile powerPellets=true illegalMove=noop \
    eps=0.5 epsDecay=0.999 epsMin=0.05
done

# ── finish ────────────────────────────────────────────────────────────────────
print_summary
