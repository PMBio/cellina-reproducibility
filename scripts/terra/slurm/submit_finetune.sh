#!/bin/bash
# Submit stage 1 (LoRA fine-tune) for the slides in $SLIDES as one job array, one GPU per task.
# The GPU is an OR over the partition's card types with VRAM >= MIN_VRAM (measured by probe.sbatch).
# Our association forbids multi-partition jobs, so pick the partition per call:
#   scripts/terra/slurm/submit_finetune.sh                                  # gpu-el8, >=48 GB cards, all six
#   MIN_VRAM=80 scripts/terra/slurm/submit_finetune.sh                      # gpu-el8, >=80 GB only
#   PARTITION=gpu-training SLIDES="crc_120 crc_210 crc_242" scripts/terra/slurm/submit_finetune.sh
#       (gpu-training: H100/H200/B200, 14-day limit, at most 3 running jobs for account stegle)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
MIN_VRAM=${MIN_VRAM:-48}; PARTITION=${PARTITION:-gpu-el8}
# Slurm node feature -> VRAM (GiB), per partition (2026-09; `sinfo -p <part> -N -o "%N %f"`).
# A40 nodes (48 GB, 1 GPU each) are mostly down and left out.
# rtx6000 / B200 are Blackwell (sm_120): the env's torch 2.5.1+cu121 has kernels up to sm_90 only.
case $PARTITION in
  gpu-el8)      declare -A VRAM=([L40s]=48 [A100]=80 [H100]=80) ;;
  gpu-training) declare -A VRAM=([H100]=80 [H200]=141) ;;
  *) echo "unknown partition $PARTITION" >&2; exit 1 ;;
esac
CONS=""
for g in "${!VRAM[@]}"; do [ "${VRAM[$g]}" -ge "$MIN_VRAM" ] && CONS="${CONS:+$CONS|}gpu=$g"; done
[ -n "$CONS" ] || { echo "no card in $PARTITION has >= $MIN_VRAM GB" >&2; exit 1; }
read -ra ARR <<< "$SLIDES"; N=${#ARR[@]}
TIME=$([ "$PARTITION" = gpu-training ] && echo 14-00:00:00 || echo 7-00:00:00)
echo "partition: $PARTITION | constraint: $CONS | slides: $SLIDES"
sbatch ${TEST_ONLY:+--test-only} --array=0-$((N-1)) --partition="$PARTITION" --gres=gpu:1 \
       --constraint="$CONS" --time="$TIME" \
       --export=ALL,SLIDES="$SLIDES",BATCH="${BATCH:-128}",EPOCHS="${EPOCHS:-5}" \
       "$REPO/scripts/terra/slurm/finetune.sbatch"
