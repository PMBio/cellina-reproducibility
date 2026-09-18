#!/bin/bash
#SBATCH --job-name=terra-s2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=scripts/terra/slurm/logs/s2_%j.log
# Stage 2 for ONE slide ($SID): for each arm in $ARMS ("frozen" | "lora:N", space-separated), for every
# scored cell type in $CTS (SPACE-separated -- sbatch --export splits on commas):
#   2a  make_terra2k.py  -> {sid}_terra2k.h5ad (CPU, minutes; skipped if present).  Not used by the
#                           decoder track itself, but terra_og_eval.py (stage 3) reads it, so it is
#                           built here where the raw slide is already loaded.
#   2b  inference.py     -> embed, count decoder (LOO per type), neighbour perturbation,
#                           eval_loo.py -> correlations/{sid}_terra-<arm>[-null]-cf_{ct}_CRC.json
# The arms run SEQUENTIALLY on one GPU: they share model-independent caches (tokenizer tmp dir,
# ctrl/pert tokenised sets) that inference.py rebuilds in place, so parallel arms of the same slide
# corrupt each other (seen 2026-09-16: stale NFS handles, bus errors).  Slides are independent and
# run in parallel as separate jobs.  Everything is re-entrant through the caches under
# {sid}/{ct}/terra.  Submitted by submit_stage2.sh.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/terra/slurm/env.sh"
A=$DATA_ROOT/datasets/crc/raw_zenodo/$SID.h5ad
CTS_CSV=${CTS// /,}
CT1=${CTS%% *}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "stage2 sid=$SID arms=$ARMS cts=$CTS_CSV node=$SLURMD_NODENAME $(date)"
T2K=$DATA_ROOT/datasets/crc/raw_zenodo_terra2k
mkdir -p "$T2K"
[ -f "$T2K/${SID}_terra2k.h5ad" ] || \
    $PY scripts/terra/make_terra2k.py --dataset_name crc --adata_path "$A" --out_dir "$T2K"
for ARM in $ARMS; do
  if [ "$ARM" = frozen ]; then V=(--variant frozen); else V=(--variant lora --epoch "${ARM#lora:}"); fi
  echo "== $ARM decoder track $(date)"
  $PY scripts/terra/inference.py --dataset_name crc --adata_path "$A" --holdout_celltype "$CT1" \
      --eval-celltypes "$CTS_CSV" "${V[@]}"
  echo "== $ARM done $(date)"
done
echo "stage2 $SID done $(date)"
