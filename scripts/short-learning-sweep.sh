#!/usr/bin/env bash
# Short learning-quality sweep for the current tabular defaults.
#
# Why these knobs:
#   The 4-worker 420m run showed healthy training wins (~1.2%) and strong state
#   overlap, so the next risk is policy quality/generalization rather than raw
#   throughput. This sweep varies exploration floor, curriculum mix, and the two
#   movement-shaping penalties most likely to affect full-maze greedy play.
#
# Usage:
#   ./scripts/short-learning-sweep.sh [durationMin=20] [desc=short-learning-sweep] [bench args ...]
#
# Output:
#   bench-out/<timestamp>-<desc>/report.tsv
#   bench-out/<timestamp>-<desc>/worker-*/{bench.log,summary.json,evals.csv,policy-latest.json}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

DURATION=20
DESC="short-learning-sweep"
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    durationMin=*) DURATION="${1#durationMin=}"; shift ;;
    desc=*)        DESC="${1#desc=}"; shift ;;
    *)             PASS_ARGS+=("$1"); shift ;;
  esac
done

if ! [[ "$DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk "BEGIN { exit ($DURATION > 0 ? 0 : 1) }"; then
  echo "error: durationMin must be a positive number, got '$DURATION'" >&2
  exit 1
fi

DESC=$(echo "$DESC" | tr -c 'a-zA-Z0-9_-' '_' | sed 's/_*$//')
[[ -n "$DESC" ]] || DESC="short-learning-sweep"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_BASE="$REPO_DIR/bench-out/${TIMESTAMP}-${DESC}"
mkdir -p "$OUT_BASE"

# Keep this intentionally short and interpretable. Baseline is first.
declare -a MATRIX=(
  "baseline|endgameCurriculum=0.90 endgameEps=0.25 epsMin=0.20 stepPenalty=-0.10 reversePenalty=-2"
  "lower-eps-floor|endgameCurriculum=0.90 endgameEps=0.25 epsMin=0.12 stepPenalty=-0.10 reversePenalty=-2"
  "mid-curriculum|endgameCurriculum=0.70 endgameEps=0.25 epsMin=0.20 stepPenalty=-0.10 reversePenalty=-2"
  "lighter-step|endgameCurriculum=0.90 endgameEps=0.25 epsMin=0.20 stepPenalty=-0.02 reversePenalty=-2"
  "lighter-reverse|endgameCurriculum=0.90 endgameEps=0.25 epsMin=0.20 stepPenalty=-0.10 reversePenalty=-0.5"
  "low-eps-light-step|endgameCurriculum=0.90 endgameEps=0.25 epsMin=0.12 stepPenalty=-0.02 reversePenalty=-2"
)

{
  echo "Short learning sweep"
  echo "Duration: ${DURATION} min/run"
  echo "Output: $OUT_BASE"
  echo "Pass-through: ${PASS_ARGS[*]:-(none)}"
  echo ""
  printf '%-3s %-18s %s\n' "id" "name" "params"
  for i in "${!MATRIX[@]}"; do
    IFS='|' read -r name params <<< "${MATRIX[$i]}"
    printf '%-3s %-18s %s\n' "$(printf '%02d' "$i")" "$name" "$params"
  done
} | tee "$OUT_BASE/config-matrix.txt"

pids=()
for i in "${!MATRIX[@]}"; do
  worker_id=$(printf '%02d' "$i")
  IFS='|' read -r name params <<< "${MATRIX[$i]}"
  worker_out="$OUT_BASE/worker-$worker_id-$name"
  mkdir -p "$worker_out"
  printf '%s\n' "$params" > "$worker_out/params.txt"

  # shellcheck disable=SC2206 # params is an intentional key=value word list.
  param_args=($params)

  setsid npx vite-node "$SCRIPT_DIR/overnight-bench.ts" -- \
    outDir="$worker_out" \
    seed=$((7 + i * 1000)) \
    durationMin="$DURATION" \
    reportEvery=60 \
    snapshotEvery=0 \
    evalEvery=500 \
    evalEpisodes=100 \
    "${param_args[@]}" \
    "${PASS_ARGS[@]}" \
    > "$worker_out/bench.log" 2>&1 &
  pids+=("$!")
  echo "[start] worker-$worker_id-$name: $params"
done

echo "[sweep] launched ${#pids[@]} workers; tail logs with: tail -F $OUT_BASE/worker-*/bench.log"

failed=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then
    failed=$((failed + 1))
  fi
done

REPORT="$OUT_BASE/report.tsv"
printf 'worker\tname\tparams\tepisodes\ttrainWins\ttrainWinRate\tscoreLast1k\tqStates\tbestEvalWinRate\tbestEvalWins\tbestEvalScore\tbestMinPellets\tfinalEvalWinRate\tfinalEvalScore\tfinalMinPellets\n' > "$REPORT"

for i in "${!MATRIX[@]}"; do
  worker_id=$(printf '%02d' "$i")
  IFS='|' read -r name params <<< "${MATRIX[$i]}"
  worker_out="$OUT_BASE/worker-$worker_id-$name"
  summary="$worker_out/summary.json"
  evals="$worker_out/evals.csv"

  if [[ ! -f "$summary" ]]; then
    printf 'worker-%s\t%s\t%s\tmissing-summary\n' "$worker_id" "$name" "$params" >> "$REPORT"
    continue
  fi

  episodes=$(jq -r '.episodes // 0' "$summary")
  train_wins=$(jq -r '.trainingWins // 0' "$summary")
  train_win_rate=$(jq -r '.trainingWinRate // 0' "$summary")
  score_last=$(jq -r '.meanScoreLast1000 // 0' "$summary")
  qstates=$(jq -r '.qTableSize // 0' "$summary")

  if [[ -f "$evals" ]] && [[ $(wc -l < "$evals") -gt 1 ]]; then
    metrics=$(awk -F, '
      NR > 1 {
        if (NR == 2 || $5 + 0 > bestWr || ($5 + 0 == bestWr && $2 + 0 > bestScore)) {
          bestWr = $5 + 0; bestWins = $6 + 0; bestScore = $2 + 0; bestMin = $7 + 0;
        }
        finalWr = $5 + 0; finalScore = $2 + 0; finalMin = $7 + 0;
      }
      END { printf "%.6f\t%d\t%.2f\t%d\t%.6f\t%.2f\t%d", bestWr, bestWins, bestScore, bestMin, finalWr, finalScore, finalMin }
    ' "$evals")
  else
    metrics=$'0\t0\t0\t-\t0\t0\t-'
  fi

  printf 'worker-%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$worker_id" "$name" "$params" "$episodes" "$train_wins" "$train_win_rate" "$score_last" "$qstates" "$metrics" \
    >> "$REPORT"
done

{
  echo ""
  echo "SWEEP SUMMARY (sorted by best eval win rate, then eval score):"
  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$REPORT" | head -1
    tail -n +2 "$REPORT" | sort -t $'\t' -k9,9gr -k11,11gr | column -t -s $'\t'
  else
    head -1 "$REPORT"
    tail -n +2 "$REPORT" | sort -t $'\t' -k9,9gr -k11,11gr
  fi
  echo ""
  echo "Report: $REPORT"
} | tee "$OUT_BASE/results.txt"

if [[ "$failed" -gt 0 ]]; then
  echo "[sweep] $failed worker(s) failed" >&2
  exit 1
fi
