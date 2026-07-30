#!/usr/bin/env bash
# T5: compare the baseline tabular key (one region) with a 3×3 Pac-Man region.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VITE_NODE="$REPO_DIR/node_modules/.bin/vite-node"
EPISODES=20000
EVAL_EPISODES=50
SEED=7
MAX_PARALLEL=2
DESC="t5-pac-region-screen"
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
if [[ ! -x "$VITE_NODE" ]]; then echo "[abort] vite-node is unavailable; run npm ci first" >&2; exit 1; fi
if ! [[ "$EPISODES" =~ ^[1-9][0-9]*$ && "$EVAL_EPISODES" =~ ^[1-9][0-9]*$ && "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "[abort] episodes, evalEpisodes, and maxParallel must be positive integers" >&2; exit 1
fi

cd "$REPO_DIR"
OUT_BASE="$REPO_DIR/bench-out/$(date +%Y%m%d-%H%M%S)-$DESC"
mkdir -p "$OUT_BASE"
GRIDS=(1 3)
printf 'cell\tpacRegionGrid\n' > "$OUT_BASE/matrix.tsv"
declare -a PIDS=()
failed=0
wait_batch() { for pid in "${PIDS[@]}"; do if ! wait "$pid"; then failed=$((failed + 1)); fi; done; PIDS=(); }
for index in "${!GRIDS[@]}"; do
  grid="${GRIDS[$index]}"; cell_id=$(printf '%02d' "$index"); cell_dir="$OUT_BASE/cell-$cell_id"
  printf '%s\t%s\n' "$cell_id" "$grid" >> "$OUT_BASE/matrix.tsv"
  "$VITE_NODE" scripts/overnight-bench.ts -- \
    "outDir=$cell_dir" algorithm=tabular ghosts=2 "seed=$SEED" "episodes=$EPISODES" \
    "pacRegionGrid=$grid" endgameCurriculum=0.90 nStep=1 \
    "evalEvery=$EPISODES" "evalEpisodes=$EVAL_EPISODES" \
    evalPanels=1000000,2000000,3000000,4000000 snapshotEvery=0 reportEvery=0 \
    > "$cell_dir.log" 2>&1 &
  PIDS+=("$!"); echo "[start] cell=$cell_id pacRegionGrid=$grid"
  if [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; then wait_batch; fi
done
wait_batch
if [[ "$failed" -gt 0 ]]; then echo "[abort] $failed T5 cell(s) failed; inspect $OUT_BASE/cell-*.log" >&2; exit 1; fi

REPORT="$OUT_BASE/report.tsv"
printf 'cell\tpacRegionGrid\tepisodes\tqTableSize\ttrainWins\ttrainWinRate\tmeanEvalWinRate\tworstPanelWinRate\tmeanPlP5\tmaxPlP5\n' > "$REPORT"
while IFS=$'\t' read -r cell_id grid; do
  [[ "$cell_id" == "cell" ]] && continue
  summary="$OUT_BASE/cell-$cell_id/summary.json"; evals="$OUT_BASE/cell-$cell_id/evals.csv"
  read -r episodes q_size train_wins train_rate < <(node -e "const s=require(process.argv[1]); console.log(s.episodes, s.qTableSize, s.trainingWins, s.trainingWinRate)" "$summary")
  read -r mean_wr worst_wr mean_p5 max_p5 < <(awk -F, 'NR>1 {wr += $5; p5 += $8; if (!n || $5 < worst) worst=$5; if (!n || $8 > max) max=$8; n++} END {printf "%.6f %.6f %.3f %.3f\n", wr/n, worst, p5/n, max}' "$evals")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell_id" "$grid" "$episodes" "$q_size" "$train_wins" "$train_rate" "$mean_wr" "$worst_wr" "$mean_p5" "$max_p5" >> "$REPORT"
done < "$OUT_BASE/matrix.tsv"
{
  echo "T5 Pac-Man region screen"
  echo "seed=$SEED episodes=$EPISODES evalEpisodes=$EVAL_EPISODES cells=${#GRIDS[@]}"
  echo ""; echo "Sorted by mean eval win rate, then worst panel (higher is better):"
  head -n 1 "$REPORT"; tail -n +2 "$REPORT" | sort -t $'\t' -k7,7gr -k8,8gr
  echo ""; echo "Artifacts: $OUT_BASE"
} | tee "$OUT_BASE/results.txt"
