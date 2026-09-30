#!/bin/bash
# Submit the cellina v1.1.2 LOO sweep for one dataset as a Slurm array.
#
#   DATASET=merfish bash scripts/cellina112_loo/submit.sh   # 3 slides x 5 cell types = 15 folds
#   DATASET=crc     bash scripts/cellina112_loo/submit.sh   # 6 slides x 5 cell types = 30 folds
#   DRY=1 ...   print jobs file + sbatch command only;  MAXJOBS=N caps concurrency (default 24)
#   SIDS=... CTS=$'a\nb' narrow the sweep (cell types newline separated: MERFISH names contain spaces)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ "$DATASET" = crc ]; then
    SIDS=${SIDS:-"crc_210 crc_221 crc_231 crc_232 crc_242 crc_120"}
    DEFAULT_CTS=$'Endothelial\nEpithelial\nFibroblast\nMyeloid\nT_cell'
    ADATA_DIR=$DATA_ROOT/datasets/crc/raw_zenodo
    SBATCH_EXTRA=(--job-name=c112-crc --time=06:00:00 --mem=96G)   # crc_210 cellina cf needed 52 GB RSS
else
    SIDS=${SIDS:-"C57BL6J-2.036 C57BL6J-2.039 C57BL6J-2.041"}
    DEFAULT_CTS=$'glutamatergic neuron\noligodendrocyte\nastrocyte\nGABAergic neuron\nendothelial cell'
    ADATA_DIR=$DATA_ROOT/datasets/MERFISH_mouse_brain
    SBATCH_EXTRA=(--job-name=c112-merfish --time=04:00:00 --mem=48G)
fi
CTS=${CTS:-$DEFAULT_CTS}
MAXJOBS=${MAXJOBS:-24}

: > "$C112_JOBS"
n=0
for sid in $SIDS; do
    [ -f "$ADATA_DIR/$sid.h5ad" ] || { echo "adata missing: $ADATA_DIR/$sid.h5ad" >&2; exit 1; }
    while IFS= read -r ct; do
        [ -n "$ct" ] || continue
        printf '%s\t%s\n' "$sid" "$ct" >> "$C112_JOBS"
        n=$((n + 1))
    done <<< "$CTS"
done
[ "$n" -gt 0 ] || { echo "no jobs generated" >&2; exit 1; }

echo "dataset: $DATASET  model: ${MODEL_NAME}_${SEED}  folds: $n  cap: %$MAXJOBS  jobs: $C112_JOBS"
CMD=(sbatch "--array=0-$((n - 1))%${MAXJOBS}" "${SBATCH_EXTRA[@]}"
     "--export=ALL,C112_DIR=$C112_DIR,DATASET=$DATASET,MODEL_NAME=$MODEL_NAME,SEED=$SEED"
     "$C112_DIR/run_fold.sbatch")
if [ -n "${DRY:-}" ]; then
    echo "--- jobs ---"; cat -A "$C112_JOBS" | sed 's/\$$//'
    echo "--- would run ---"; printf '%q ' "${CMD[@]}"; echo
    exit 0
fi
"${CMD[@]}"
