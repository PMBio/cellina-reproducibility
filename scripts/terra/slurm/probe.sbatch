#!/bin/bash
#SBATCH --job-name=terra-probe
#SBATCH --partition=gpu-el8
#SBATCH --gres=gpu:L40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=scripts/terra/slurm/logs/probe_%j.log
# Measure peak GPU memory and sec/step of the LoRA fine-tune at the target batch size on the
# LARGEST slide (crc_120, 741k cells), 1 epoch capped at $STEPS optimizer steps, tiny guard set.
# Side effect worth having: builds crc_120's tokenizer cache, which the real run reuses.
#   sbatch scripts/terra/slurm/probe.sbatch            # batch 128 on an L40s (48 GB)
#   BATCH=128 STEPS=200 sbatch --gres=gpu:H100:1 scripts/terra/slurm/probe.sbatch
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/terra/slurm/env.sh"   # submit from the repo root
SID=${SID:-crc_120}; BATCH=${BATCH:-128}; STEPS=${STEPS:-100}
WORK=$DATA_ROOT/datasets/crc/$SID/terra_probe_b$BATCH
rm -rf "$WORK"       # a probe is never re-entrant
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "probe sid=$SID batch=$BATCH steps=$STEPS work=$WORK  $(date)"
$PY scripts/terra/finetuning_lora.py --dataset_name crc \
    --adata_path "$DATA_ROOT/datasets/crc/raw_zenodo/$SID.h5ad" \
    --epochs 1 --max-steps "$STEPS" --batch-size "$BATCH" --guard-cells 2000 \
    --nproc "$SLURM_CPUS_PER_TASK" --work-dir "$WORK"
$PY - "$WORK/epoch_selection.json" "$STEPS" <<'PYEOF'
import json, sys
j = json.load(open(sys.argv[1])); steps = int(sys.argv[2])
print(f"PROBE_RESULT peak_gpu_gib={j['peak_gpu_gib']:.2f} wall_s={j['wall_seconds']:.0f} "
      f"sec_per_step={j['wall_seconds']/steps:.2f} batch={j['batch_size']} "
      f"guard_passed={j['epochs']['1']['passed']}")
PYEOF
echo "probe done $(date)"
