#!/usr/bin/env bash
# T2: fixed-budget reward/discount screen for the active linear/T7 baseline.
#
# Every cell uses the same seed, episode count, four evaluation panels, and
# baseline linear settings. This is a screen only: promote a candidate only
# after the separate five-seed confirmation required by ROADMAP.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VITE_NODE="$REPO_DIR/node_modules/.bin/vite-node"

EPISODES=2000
EVAL_EPISODES=50
SEED=7
MAX_PARALLEL=4
DESC="t2-reward-screen"

for value in "$@"; do
  case "$value" in
    episodes=*) EPISODES="${value#episodes=}" ;;
    evalEpisodes=*) EVAL_EPISODES="${value#evalEpisodes=}" ;;
    seed=*) SEED="${value#seed=}" ;;
    maxParallel=*) MAX_PARALLEL="${value#maxParallel=}" ;;
    desc=*) DESC="${value#desc=}" ;;
    *) echo "[abort] unrecognized argument '$value'" >&2; exit 1 ;;
  esac
done

if [[ ! -x "$VITE_NODE" ]]; then
  echo "[abort] vite-node is unavailable; run npm ci first" >&2
  exit 1
fi
if ! [[ "$EPISODES" =~ ^[1-9][0-9]*$ && "$EVAL_EPISODES" =~ ^[1-9][0-9]*$ && "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "[abort] episodes, evalEpisodes, and maxParallel must be positive integers" >&2
  exit 1
fi

cd "$REPO_DIR"
OUT_BASE="$REPO_DIR/bench-out/$(date +%Y%m%d-%H%M%S)-$DESC"
mkdir -p "$OUT_BASE"

# gamma × winBonus × deathPenalty × late-pellet cap = 3 × 3 × 2 × 2 cells.
GAMMAS=(0.99 0.997 0.999)
WIN_BONUSES=(1000 2500 5000)
DEATH_PENALTIES=(-100 -50)
ESCALATION_MAXES=(6 10)

printf 'cell\tgamma\twinBonus\tdeathPenalty\tpelletEscalationMax\n' > "$OUT_BASE/matrix.tsv"

declare -a PIDS=()
declare -a CELLS=()
failed=0
cell=0

wait_batch() {
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then failed=$((failed + 1)); fi
  done
  PIDS=()
}

for gamma in "${GAMMAS[@]}"; do
  for win_bonus in "${WIN_BONUSES[@]}"; do
    for death_penalty in "${DEATH_PENALTIES[@]}"; do
      for escalation_max in "${ESCALATION_MAXES[@]}"; do
        cell_id=$(printf '%02d' "$cell")
        cell_dir="$OUT_BASE/cell-$cell_id"
        printf '%s\t%s\t%s\t%s\t%s\n' "$cell_id" "$gamma" "$win_bonus" "$death_penalty" "$escalation_max" >> "$OUT_BASE/matrix.tsv"
        "$VITE_NODE" scripts/overnight-bench.ts -- \
          "outDir=$cell_dir" \
          algorithm=linear ghosts=2 "seed=$SEED" "episodes=$EPISODES" \
          endgameCurriculum=0.90 stepPenalty=-0.02 alpha=0.02 targetSyncSteps=2000 \
          "gamma=$gamma" "winBonus=$win_bonus" "deathPenalty=$death_penalty" "pelletEscalationMax=$escalation_max" \
          "evalEvery=$EPISODES" "evalEpisodes=$EVAL_EPISODES" \
          evalPanels=1000000,2000000,3000000,4000000 snapshotEvery=0 reportEvery=0 \
          > "$cell_dir.log" 2>&1 &
        PIDS+=("$!")
        CELLS+=("$cell_id")
        echo "[start] cell=$cell_id gamma=$gamma winBonus=$win_bonus deathPenalty=$death_penalty cap=$escalation_max"
        cell=$((cell + 1))
        if [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; then wait_batch; fi
      done
    done
  done
done
wait_batch

if [[ "$failed" -gt 0 ]]; then
  echo "[abort] $failed T2 cell(s) failed; inspect $OUT_BASE/cell-*.log" >&2
  exit 1
fi

REPORT="$OUT_BASE/report.tsv"
printf 'cell\tgamma\twinBonus\tdeathPenalty\tpelletEscalationMax\tepisodes\ttrainWins\ttrainWinRate\tmeanEvalWinRate\tworstPanelWinRate\tmeanPlP5\tmaxPlP5\n' > "$REPORT"
while IFS=$'\t' read -r cell_id gamma win_bonus death_penalty escalation_max; do
  [[ "$cell_id" == "cell" ]] && continue
  summary="$OUT_BASE/cell-$cell_id/summary.json"
  evals="$OUT_BASE/cell-$cell_id/evals.csv"
  read -r episodes train_wins train_rate < <(node -e "const s=require(process.argv[1]); console.log(s.episodes, s.trainingWins, s.trainingWinRate)" "$summary")
  read -r mean_wr worst_wr mean_p5 max_p5 < <(
    awk -F, 'NR>1 {wr += $5; p5 += $8; if (!n || $5 < worst) worst=$5; if (!n || $8 > max) max=$8; n++} END {printf "%.6f %.6f %.3f %.3f\n", wr/n, worst, p5/n, max}' "$evals"
  )
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cell_id" "$gamma" "$win_bonus" "$death_penalty" "$escalation_max" "$episodes" "$train_wins" "$train_rate" "$mean_wr" "$worst_wr" "$mean_p5" "$max_p5" >> "$REPORT"
done < "$OUT_BASE/matrix.tsv"

{
  echo "T2 reward/discount screen"
  echo "seed=$SEED episodes=$EPISODES evalEpisodes=$EVAL_EPISODES cells=$cell"
  echo ""
  echo "Sorted by mean eval win rate, then worst panel (higher is better):"
  head -n 1 "$REPORT"
  tail -n +2 "$REPORT" | sort -t $'\t' -k9,9gr -k10,10gr
  echo ""
  echo "Artifacts: $OUT_BASE"
} | tee "$OUT_BASE/results.txt"
