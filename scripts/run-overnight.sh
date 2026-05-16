#!/usr/bin/env bash
# Overnight test suite — runs all 6 bench configurations under a single
# timestamped top-level folder.
#
# Usage:
#   ./scripts/run-overnight.sh [totalTime=MIN] [desc=NAME] [--clean] [policy.json]
#
# Options:
#   totalTime=<n>  total duration for all runs in minutes (default: 480)
#                  durations of each sub-run scale proportionally.
#   desc=<s>       short label appended to the top-level folder name
#                  (default: "overnight"). Use this to tag experiments,
#                  e.g. desc=ab-3a-fullnight. No spaces; non-alphanumeric
#                  chars are stripped to underscores.
#   --clean        if the target top-level folder already exists, wipe it
#                  before running. Without --clean, the script aborts on
#                  collision (Option A) — prevents accidentally mixing
#                  data from two separate executions in one folder.
#   policy.json    optional explicit seed policy for the resume/mismatch
#                  runs. If omitted, the script auto-picks run1's policy
#                  from the *current* top-level after run1 completes.
#                  Cross-experiment policies must be passed explicitly.
#
# Output structure:
#   bench-out/<YYYYMMDD-HHMMSS>-<desc>/
#     run1-baseline/                policy-latest.json, episodes.csv, evals.csv, summary.json
#     run2-resume/                  …
#     run3-explore/                 …
#     run4-2ghosts/                 …
#     run5-4ghosts/                 …
#     run6-overnight/               …
#
# Examples:
#   ./scripts/run-overnight.sh                                  # 8 hour default
#   ./scripts/run-overnight.sh totalTime=240 desc=quick         # 4 hour, labeled
#   ./scripts/run-overnight.sh --clean desc=overnight           # wipe & redo
#   ./scripts/run-overnight.sh desc=resume bench-out/prior/run1-baseline/policy-latest.json
#
# Ctrl-C at any time: the current sub-run flushes its policy + summary, then
# the script exits cleanly (remaining sub-runs are skipped).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER="npx vite-node $SCRIPT_DIR/overnight-bench.ts --"

# ---- parse arguments ----
TOTAL_TIME_MIN=480
DESC="overnight"
CLEAN=0
EXPLICIT_SEED_POLICY=""

for arg in "$@"; do
  case "$arg" in
    totalTime=*) TOTAL_TIME_MIN="${arg#totalTime=}";;
    desc=*)      DESC="${arg#desc=}";;
    --clean)     CLEAN=1;;
    --*)         echo "[warn] unknown flag: $arg";;
    *)
      # Anything else is treated as an explicit seed policy path
      EXPLICIT_SEED_POLICY="$arg";;
  esac
done

# Sanitize desc: strip anything that isn't alnum/dash/underscore
DESC=$(echo "$DESC" | tr -c 'a-zA-Z0-9_-' '_' | sed 's/_*$//')
if [[ -z "$DESC" ]]; then DESC="overnight"; fi

# Top-level folder = timestamp + desc, all sub-runs live under it.
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TOP_LEVEL="$REPO_DIR/bench-out/${TIMESTAMP}-${DESC}"

# Failsafe (Option A: fail-fast on collision; Option B: --clean to wipe)
if [[ -d "$TOP_LEVEL" ]]; then
  existing=$(find "$TOP_LEVEL" -maxdepth 1 -mindepth 1 -type d -name 'run*' 2>/dev/null | wc -l)
  if [[ $existing -gt 0 ]]; then
    if [[ $CLEAN -eq 1 ]]; then
      echo "[setup] --clean: removing $existing existing run folder(s) under $TOP_LEVEL"
      rm -rf "$TOP_LEVEL"
    else
      echo "[abort] $TOP_LEVEL already contains $existing run folder(s)."
      echo "        Pass --clean to wipe, or use a different desc= to start fresh." >&2
      exit 1
    fi
  fi
fi
mkdir -p "$TOP_LEVEL"
echo "[setup] top-level: $TOP_LEVEL"

# ---- seed policy resolution ----
# Strategy: explicit > current top-level > nothing (skip resume runs).
# Crossing across top-level folders is intentional — see analysis of the
# old auto-detect-across-bench-out bug that silently mixed two executions.
SEED_POLICY=""
if [[ -n "$EXPLICIT_SEED_POLICY" ]]; then
  if [[ ! -f "$EXPLICIT_SEED_POLICY" ]]; then
    echo "[abort] explicit seed policy not found: $EXPLICIT_SEED_POLICY" >&2
    exit 1
  fi
  SEED_POLICY="$EXPLICIT_SEED_POLICY"
  echo "[setup] explicit seed policy: $SEED_POLICY"
else
  echo "[setup] no explicit seed — runs 2-5 will use run1's policy after it completes"
fi

# Re-pick seed from within the current top-level (called after run1 finishes,
# so runs 2-5 inherit run1's policy when no explicit seed was given).
maybe_pick_seed_from_current() {
  if [[ -n "$EXPLICIT_SEED_POLICY" ]]; then return 0; fi
  local latest
  latest=$(ls -t "$TOP_LEVEL"/run*/policy-latest.json 2>/dev/null | head -1 || true)
  if [[ -n "$latest" ]]; then
    SEED_POLICY="$latest"
    echo "[seed] picked from current run: $SEED_POLICY"
  fi
}

echo "[setup] total test duration: ${TOTAL_TIME_MIN} minutes"

# ---- calculate scaled durations ----
# Baseline 60+60+60+30+30+240 = 480 min; scale proportionally to requested total.
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
  local out="$TOP_LEVEL/$label"
  mkdir -p "$out"
  announce "$label"
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

# After run1 produces its policy, runs 2-5 can use it (unless an explicit
# seed was passed at script invocation).
maybe_pick_seed_from_current

# ── Run 2: resume from seed policy ───────────────────────────────
if [[ -n "$SEED_POLICY" ]]; then
  run "run2-resume" \
    ghosts=3 eps=0.05 \
    loadPolicy="$SEED_POLICY" \
    evalEvery=5000 evalEpisodes=200 reportEvery=60 snapshotEvery=600 \
    durationMin=$DUR2
else
  echo "[skip] run2-resume — no seed policy available"
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
  echo "[skip] run3-explore — no seed policy available"
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
  echo "[skip] run4-2ghosts — no seed policy available"
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
  echo "[skip] run5-4ghosts — no seed policy available"
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
echo "  Results: $TOP_LEVEL/"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  Non-zero exits: ${FAILED[*]}"
fi
echo "════════════════════════════════════════════════════════"

# Print a quick comparison table from each sub-run's summary.json
echo ""
echo "── Summary ────────────────────────────────────────────"
printf "%-20s %8s %8s %8s %8s %8s\n" "run" "episodes" "qStates" "score/1k" "len/1k" "trainWin"
for summary in "$TOP_LEVEL"/run*/summary.json; do
  [[ -f "$summary" ]] || continue
  label=$(basename "$(dirname "$summary")")
  ep=$(jq -r '.episodes'              "$summary" 2>/dev/null || echo "?")
  qs=$(jq -r '.qTableSize'            "$summary" 2>/dev/null || echo "?")
  sc=$(jq -r '.meanScoreLast1000'     "$summary" 2>/dev/null || echo "?")
  ln=$(jq -r '.meanLenLast1000'       "$summary" 2>/dev/null || echo "?")
  tw=$(jq -r '.trainingWins // 0'     "$summary" 2>/dev/null || echo "?")
  printf "%-20s %8s %8s %8s %8s %8s\n" "$label" "$ep" "$qs" "$sc" "$ln" "$tw"
done
echo "───────────────────────────────────────────────────────"
