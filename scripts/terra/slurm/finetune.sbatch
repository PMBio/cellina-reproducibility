#!/bin/bash
#SBATCH --job-name=terra-ft
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=scripts/terra/slurm/logs/ft_%A_%a.log
# Stage 1: self-supervised LoRA fine-tune of ONE slide (array index -> $SLIDES), all cells,
# batch 128 x 5 epochs x lr 1e-4, then per-epoch repack + collapse guard -> epoch_selection.json.
# Re-entrant: an existing epoch_selection.json makes it a no-op; a run dir with all 5
# checkpoints is reused (export/embed/guard only).  Submit through submit_finetune.sh.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/terra/slurm/env.sh"
read -ra ARR <<< "$SLIDES"; SID=${ARR[$SLURM_ARRAY_TASK_ID]}
A=$DATA_ROOT/datasets/crc/raw_zenodo/$SID.h5ad
RUN=$DATA_ROOT/datasets/crc/$SID/terra/lora_run/run
EXTRA=()
[ -f "$RUN/checkpoint_epoch_${EPOCHS:-5}.pt" ] && EXTRA+=(--from-run-dir "$RUN")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "ft sid=$SID batch=${BATCH:-128} epochs=${EPOCHS:-5} node=$SLURMD_NODENAME $(date)"
$PY scripts/terra/finetuning_lora.py --dataset_name crc --adata_path "$A" \
    --batch-size "${BATCH:-128}" --epochs "${EPOCHS:-5}" --nproc "$SLURM_CPUS_PER_TASK" \
    ${EXTRA[@]+"${EXTRA[@]}"}
echo "ft $SID done $(date)"
