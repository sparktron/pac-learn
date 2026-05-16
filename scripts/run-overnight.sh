#!/usr/bin/env bash
# Overnight test suite — runs all bench configurations in sequence.
# Each run writes its own outDir under bench-out/.
#
# Usage:
#   ./scripts/run-overnight.sh [totalTime=MINUTES] [policy.json]
#
# Options:
#   totalTime=<n>  total duration for all runs in minutes (default: 480)
#                  automatically scales each test's duration proportionally
#   policy.json    path to a pre-trained policy JSON to use for
#                  the resume / mismatch runs. Defaults to the most recent
#                  policy-*.json in bench-out/ if one exists.
#
# Examples:
#   ./scripts/run-overnight.sh                    # 8 hours (480 min)
#   ./scripts/run-overnight.sh totalTime=240      # 4 hours
#   ./scripts/run-overnight.sh totalTime=720      # 12 hours
#   ./scripts/run-overnight.sh totalTime=480 policy.json
#
# Ctrl-C at any time: the current run flushes its policy + summary, then
# the script exits cleanly (remaining runs are skipped).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER="npx vite-node $SCRIPT_DIR/overnight-bench.ts --"

# ---- parse arguments ----
# Separate totalTime argument from policy path
TOTAL_TIME_MIN=480  # default: 8 hours
SEED_POLICY=""

for arg in "$@"; do
  if [[ "$arg" == totalTime=* ]]; then
    TOTAL_TIME_MIN="${arg#totalTime=}"
  elif [[ "$arg" != --* ]]; then
    # Assume it's a policy path
    SEED_POLICY="$arg"
  fi
done

# If no seed policy was passed as argument, try to find the most recent one
if [[ -z "$SEED_POLICY" ]]; then
  LATEST=$(ls -t "$REPO_DIR"/bench-out/run*/policy-latest.json 2>/dev/null | head -1 || true)
  if [[ -n "$LATEST" ]]; then
    SEED_POLICY="$LATEST"
    echo "[setup] using most recent policy: $SEED_POLICY"
  else
    echo "[setup] no seed policy found — resume/mismatch runs will train from scratch"
  fi
fi

echo "[setup] total test duration: ${TOTAL_TIME_MIN} minutes"

# ---- calculate scaled durations ----
# Original baseline: 60+60+60+30+30+240 = 480 min
# Scale proportionally for requested total time
SCALE=$(awk "BEGIN {printf \"%.6f\", $TOTAL_TIME_MIN / 480}")
DUR1=$(awk "BEGIN {printf \"%.0f\", 60 * $SCALE}")
DUR2=$(awk "BEGIN {printf \"%.0f\", 60 * $SCALE}")
DUR3=$(awk "BEGIN {printf \"%.0f\", 60 * $SCALE}")
DUR4=$(awk "BEGIN {printf \"%.0f\", 30 * $SCALE}")
DUR5=$(awk "BEGIN {printf \"%.0f\", 30 * $SCALE}")
DUR6=$(awk "BEGIN {printf \"%.0f\", 240 * $SCALE}")
TOTAL_DUR=$((DUR1 + DUR2 + DUR3 + DUR4 + DUR5 + DUR6))

echo "[setup] scaled durations: run1=${DUR1}m run2=${DUR2}m run3=${DUR3}m run4=${DUR4}m run5=${DUR5}m run6=${DUR6}m (total=$TOTAL_DUR min)"

# ---- helpers ----
TOTAL_RUNS=6
CURRENT_RUN=0
FAILED=()

announce() {
  CURRENT_RUN=$((CURRENT_RUN + 1))
  echo ""
  echo "════════════════════════════════════════════════════════"
  echo "  Run $CURRENT_RUN / $TOTAL_RUNS — $1"
  echo "════════════════════════════════════════════════════════"
}

run() {
  local label="$1"; shift
  local out="$REPO_DIR/bench-out/$label"
  mkdir -p "$out"
  announce "$label"
  # Build the vite-node command; seed policy is appended by caller if needed
  if $RUNNER outDir="$out" "$@"; then
    echo "[done] $label → $out"
  else
    echo "[warn] $label exited non-zero (policy still written if training started)"
    FAILED+=("$label")
  fi
}

trap 'echo ""; echo "[interrupt] exiting — runs remaining skipped"; exit 1' INT TERM

START=$(date +%s)

# ── Run 1: baseline (train from scratch) ────────────────────────────
# Uses bench defaults for ε (eps=0.5 epsDecay=0.99999 epsMin=0.15) — slower
# decay + higher floor keeps exploration alive long enough to find wins.
run "run1-baseline" \
  ghosts=3 alpha=0.2 gamma=0.95 \
  evalEvery=5000 evalEpisodes=200 reportEvery=60 snapshotEvery=600 \
  durationMin=$DUR1

# ── Run 2: resume from seed policy ───────────────────────────────
if [[ -n "$SEED_POLICY" ]]; then
  run "run2-resume" \
    ghosts=3 eps=0.05 \
    loadPolicy="$SEED_POLICY" \
    evalEvery=5000 evalEpisodes=200 reportEvery=60 snapshotEvery=600 \
    durationMin=$DUR2
else
  echo "[skip] run2-resume — no seed policy provided"
  TOTAL_RUNS=$((TOTAL_RUNS - 1))
fi

# ── Run 3: wider exploration floor ───────────────────────────────
if [[ -n "$SEED_POLICY" ]]; then
  run "run3-explore" \
    ghosts=3 eps=0.3 epsMin=0.15 \
    loadPolicy="$SEED_POLICY" \
    evalEvery=5000 evalEpisodes=200 reportEvery=60 snapshotEvery=600 \
    durationMin=$DUR3
else
  echo "[skip] run3-explore — no seed policy provided"
  TOTAL_RUNS=$((TOTAL_RUNS - 1))
fi

# ── Run 4: ghost-count mismatch — 2 ghosts ──────────────────────────
if [[ -n "$SEED_POLICY" ]]; then
  run "run4-2ghosts" \
    ghosts=2 \
    loadPolicy="$SEED_POLICY" \
    evalEvery=2000 evalEpisodes=200 reportEvery=60 snapshotEvery=600 \
    durationMin=$DUR4
else
  echo "[skip] run4-2ghosts — no seed policy provided"
  TOTAL_RUNS=$((TOTAL_RUNS - 1))
fi

# ── Run 5: ghost-count mismatch — 4 ghosts ──────────────────────────
if [[ -n "$SEED_POLICY" ]]; then
  run "run5-4ghosts" \
    ghosts=4 \
    loadPolicy="$SEED_POLICY" \
    evalEvery=2000 evalEpisodes=200 reportEvery=60 snapshotEvery=600 \
    durationMin=$DUR5
else
  echo "[skip] run5-4ghosts — no seed policy provided"
  TOTAL_RUNS=$((TOTAL_RUNS - 1))
fi

# ── Run 6: long run with dense eval logging ───────────────────────
# Uses bench defaults for ε (eps=0.5 epsDecay=0.99999 epsMin=0.15).
run "run6-overnight" \
  ghosts=3 \
  evalEvery=2000 evalEpisodes=200 reportEvery=60 snapshotEvery=600 \
  durationMin=$DUR6

# ---- summary ----
END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))
echo ""
echo "════════════════════════════════════════════════════════"
echo "  All runs complete — total elapsed: ${ELAPSED}m"
echo "  Results: $REPO_DIR/bench-out/"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  Non-zero exits: ${FAILED[*]}"
fi
echo "════════════════════════════════════════════════════════"

# Print a quick comparison table from each run's summary.json
echo ""
echo "── Summary ────────────────────────────────────────────"
printf "%-20s %8s %8s %8s %8s\n" "run" "episodes" "qStates" "score/1k" "len/1k"
for summary in "$REPO_DIR"/bench-out/run*/summary.json; do
  label=$(basename "$(dirname "$summary")")
  ep=$(jq -r '.episodes'              "$summary" 2>/dev/null || echo "?")
  qs=$(jq -r '.qTableSize'            "$summary" 2>/dev/null || echo "?")
  sc=$(jq -r '.meanScoreLast1000'     "$summary" 2>/dev/null || echo "?")
  ln=$(jq -r '.meanLenLast1000'       "$summary" 2>/dev/null || echo "?")
  printf "%-20s %8s %8s %8s %8s\n" "$label" "$ep" "$qs" "$sc" "$ln"
done
echo "───────────────────────────────────────────────────────"
