#!/usr/bin/env bash
#
# run-soak.sh — long multi-seed soak of the D8/D9 linear agent.
#
# Answers the question the 2026-07-26 five-seed confirmation could not: the
# cross-seed *mean* (27.55%, seed std 0.16pp) is repeatable, but individual
# checkpoints still dipped to 1.5% and every result so far was scored on one
# fixed 200-maze panel. This runs each seed far longer and evaluates every
# checkpoint on several held-out panels, so both the tail and generalization
# are measured.
#
# Historical success bar (met by T7 on 2026-07-29):
#   >= 32% mean evaluation wins
#   >= 25% on the worst held-out panel
#   >= 15% at the checkpoint fifth percentile
# New experiments should compare against the matched T7 fallback baseline:
#   35.17% pooled mean, >=32.06% seed-level worst panel, 30.75% ckpt p5.
#
# Seeds run as INDEPENDENT single-worker processes, not federated: merge-
# policies.ts averages tabular Q-tables and has no linear branch, so there is
# nothing to merge here. One process per seed, all in parallel.
#
# Usage:
#   ./scripts/run-soak.sh [durationMin=480] [seeds=7,1007,2007,3007,4007]
#                         [desc=linear-soak] [key=value ...]
#
# Any additional key=value pairs are forwarded verbatim to overnight-bench.ts,
# so this script never needs editing to test a knob.
#
# Output:
#   bench-out/<timestamp>-<desc>/seed-<n>/{bench.log,episodes.csv,evals.csv,
#                                          policy-latest.json,summary.json}
#   bench-out/<timestamp>-<desc>/summary.tsv
set -euo pipefail

cd "$(dirname "$0")/.."

DURATION_MIN=480
SEEDS="7,1007,2007,3007,4007"
DESC="linear-soak"
# Four disjoint 200-game panels. 1000000 is the historical default, kept first
# so soak numbers stay comparable with every prior run in test_history.md; the
# other three are held out and have never been trained or tuned against.
EVAL_PANELS="1000000,2000000,3000000,4000000"
PASSTHROUGH=()

for a in "$@"; do
  case "$a" in
    durationMin=*) DURATION_MIN="${a#*=}" ;;
    seeds=*)       SEEDS="${a#*=}" ;;
    desc=*)        DESC="${a#*=}" ;;
    evalPanels=*)  EVAL_PANELS="${a#*=}" ;;
    *=*)           PASSTHROUGH+=("$a") ;;
    *)
      echo "[abort] unrecognized argument '$a' (expected key=value)" >&2
      exit 1
      ;;
  esac
done

OUT_BASE="bench-out/$(date +%Y%m%d-%H%M%S)-$DESC"
mkdir -p "$OUT_BASE"

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

echo "[soak] out=$OUT_BASE"
echo "[soak] seeds=$SEEDS durationMin=$DURATION_MIN panels=$EVAL_PANELS"
echo "[soak] extra=${PASSTHROUGH[*]:-none}"

pids=()
for seed in "${SEED_LIST[@]}"; do
  seed_out="$OUT_BASE/seed-$seed"
  mkdir -p "$seed_out"
  # Config mirrors the 2026-07-26 linear-multiseed confirmation so the soak is
  # a duration change, not a config change. evalEvery=2000 (vs 500 there):
  # four panels cost 4x per pass, and over 8h the checkpoint count is ample.
  npx tsx scripts/overnight-bench.ts \
    algorithm=linear \
    ghosts=2 \
    "seed=$seed" \
    "durationMin=$DURATION_MIN" \
    endgameCurriculum=0.90 \
    stepPenalty=-0.02 \
    evalEpisodes=200 \
    "evalPanels=$EVAL_PANELS" \
    evalEvery=2000 \
    "outDir=$seed_out" \
    "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" \
    > "$seed_out/bench.log" 2>&1 &
  pids+=("$!")
  echo "[soak] seed=$seed pid=${pids[-1]} -> $seed_out"
done

report() {
REPORT="$OUT_BASE/summary.tsv"
printf 'seed\tepisodes\ttrainWins\ttrainWinRate\tcheckpoints\tmeanWinRate\tworstPanelMean\tcheckpointP5\tfinalWinRate\n' > "$REPORT"

for seed in "${SEED_LIST[@]}"; do
  seed_out="$OUT_BASE/seed-$seed"
  summary="$seed_out/summary.json"
  evals="$seed_out/evals.csv"
  episodes_csv="$seed_out/episodes.csv"

  if [[ -f "$summary" ]]; then
    episodes=$(jq -r '.episodes // 0' "$summary")
    train_wins=$(jq -r '.trainingWins // 0' "$summary")
    train_rate=$(jq -r '.trainingWinRate // 0' "$summary")
  elif [[ -f "$episodes_csv" ]]; then
    # summary.json is written only by the bench's clean exit / signal handler.
    # The 2026-07-28 soak was SIGKILLed and left none, so every completed
    # checkpoint was still on disk but the aggregate reported nothing. Recover
    # the training-side numbers from episodes.csv (col8 = termReason) instead
    # of discarding a six-hour run over a missing summary.
    read -r episodes train_wins train_rate < <(
      awk -F, 'NR>1{n++; if($8=="won")w++} END{printf "%d %d %.6f", n, w+0, (n?(w+0)/n:0)}' "$episodes_csv"
    )
  else
    printf '%s\tno-data\n' "$seed" >> "$REPORT"
    continue
  fi

  if [[ -f "$evals" ]] && [[ $(wc -l < "$evals") -gt 1 ]]; then
    # col5=winRate, col13=panel. Three aggregates the success bar needs:
    #   meanWinRate     — mean over every panel row
    #   worstPanelMean  — the lowest per-panel mean (the "worst held-out panel")
    #   checkpointP5    — 5th percentile over per-checkpoint means, i.e. the
    #                     tail this soak exists to measure
    metrics=$(awk -F, '
      NR > 1 {
        n++; sum += $5;
        panelSum[$13] += $5; panelN[$13]++;
        epSum[$1] += $5; epN[$1]++;
        final = $5;
      }
      END {
        if (n == 0) { printf "0\t0\t0\t0\t0"; exit }
        worst = -1;
        for (p in panelSum) {
          m = panelSum[p] / panelN[p];
          if (worst < 0 || m < worst) worst = m;
        }
        c = 0;
        for (e in epSum) ckpt[c++] = epSum[e] / epN[e];
        # insertion sort — checkpoint counts here are in the hundreds
        for (i = 1; i < c; i++) {
          v = ckpt[i];
          for (j = i - 1; j >= 0 && ckpt[j] > v; j--) ckpt[j + 1] = ckpt[j];
          ckpt[j + 1] = v;
        }
        idx = int(0.05 * (c - 1) + 0.5);
        printf "%d\t%.4f\t%.4f\t%.4f\t%.4f", c, sum / n, worst, ckpt[idx], final;
      }
    ' "$evals")
  else
    metrics=$'0\t0\t0\t0\t0'
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "$episodes" "$train_wins" "$train_rate" "$metrics" >> "$REPORT"
done

echo
echo "[soak] report: $REPORT"
column -t -s $'\t' "$REPORT"
}

# Interrupting a soak must still produce the aggregate — an operator who stops
# a run early has the same right to its partial data as one who lets it finish.
on_signal() {
  echo >&2
  echo "[soak] interrupted — reporting on partial data" >&2
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  report
  exit 130
}
trap on_signal INT TERM

# Report every failure rather than dying on the first: a soak that loses one
# seed to an OOM should still surface the other four.
failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[soak] seed=${SEED_LIST[$i]} FAILED (see $OUT_BASE/seed-${SEED_LIST[$i]}/bench.log)" >&2
    failed=$((failed + 1))
  fi
done

report

if [[ "$failed" -gt 0 ]]; then
  echo "[soak] $failed seed(s) failed" >&2
  exit 1
fi
