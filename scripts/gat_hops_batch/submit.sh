#!/bin/bash
# Submit the Cellina-GAT hops / batch-size sensitivity sweep as ONE Slurm array job.
#
#   bash scripts/gat_hops_batch/submit.sh                 # 3 arms x 3 slides x 5 cell types = 45 folds
#   DRY=1 bash scripts/gat_hops_batch/submit.sh           # print jobs.txt + the sbatch command, submit nothing
#   MAXJOBS=8 bash scripts/gat_hops_batch/submit.sh       # cap concurrency (default 15)
#   ARMS=h1-b256 CTS=astrocyte bash scripts/gat_hops_batch/submit.sh    # narrow the sweep
#   SIDS=C57BL6J-2.036 bash scripts/gat_hops_batch/submit.sh
#   DATASET=crc ARMS=h1-b256 bash scripts/gat_hops_batch/submit.sh   # arm A on the 6 CRC slides -> results/gat_hops_batch/crc/
#
# A Slurm array is used (rather than one sbatch per fold) because it is the only
# way to get a concurrency cap; the per-fold (ARM, SID, CT) triple is looked up
# by $SLURM_ARRAY_TASK_ID in jobs.txt, which this script (re)writes.  Cell type
# names contain spaces, so jobs.txt is tab separated.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ARMS=${ARMS:-$GHB_ARMS_ALL}
# Slides and holdouts as in scripts/train_parallel.py (CRC_*/MERFISH_* lists).
if [ "$DATASET" = crc ]; then
    SIDS=${SIDS:-"crc_210 crc_221 crc_231 crc_232 crc_242 crc_120"}
    DEFAULT_CTS=$'Endothelial\nEpithelial\nFibroblast\nMyeloid\nT_cell'
    ADATA_DIR=$DATA_ROOT/datasets/crc/raw_zenodo
    SBATCH_EXTRA=(--job-name=gat-hops-crc --time=24:00:00 --mem=128G)
else
    SIDS=${SIDS:-"C57BL6J-2.036 C57BL6J-2.039 C57BL6J-2.041"}
    DEFAULT_CTS=$'glutamatergic neuron\noligodendrocyte\nastrocyte\nGABAergic neuron\nendothelial cell'
    ADATA_DIR=$DATA_ROOT/datasets/MERFISH_mouse_brain
    SBATCH_EXTRA=()
fi
CTS=${CTS:-$DEFAULT_CTS}
MAXJOBS=${MAXJOBS:-15}

: > "$GHB_JOBS"
n=0
for arm in $ARMS; do
    case " $GHB_ARMS_ALL " in *" $arm "*) ;; *) echo "unknown arm '$arm'" >&2; exit 1 ;; esac
    for sid in $SIDS; do
        adata=$ADATA_DIR/$sid.h5ad
        [ -f "$adata" ] || { echo "adata missing: $adata" >&2; exit 1; }
        while IFS= read -r ct; do
            [ -n "$ct" ] || continue
            printf '%s\t%s\t%s\n' "$arm" "$sid" "$ct" >> "$GHB_JOBS"
            n=$((n + 1))
        done <<< "$CTS"
    done
done
[ "$n" -gt 0 ] || { echo "no jobs generated" >&2; exit 1; }

LAST=$((n - 1))
echo "dataset: $DATASET"
echo "jobs.txt: $GHB_JOBS ($n folds)"
echo "arms: $ARMS"
echo "sids: $SIDS"
echo "concurrency cap: %$MAXJOBS"

CMD=(sbatch "--array=0-${LAST}%${MAXJOBS}" "${SBATCH_EXTRA[@]+"${SBATCH_EXTRA[@]}"}"
     "--export=ALL,GHB_DIR=$GHB_DIR,DATASET=$DATASET"
     "$GHB_DIR/run_fold.sbatch")

if [ -n "${DRY:-}" ]; then
    echo "--- jobs.txt ---"
    cat -A "$GHB_JOBS" | sed 's/\$$//'
    echo "--- would run ---"
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi
"${CMD[@]}"
