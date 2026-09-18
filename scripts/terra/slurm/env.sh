#!/bin/bash
# Shared environment for every TERRA Slurm job.  Source from an sbatch script:
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Everything lives under the repository checkout (data/ is gitignored): raw slides, HF model
# cache, tokenizer caches, LoRA runs, decoder checkpoints, scored h5ads and correlation JSONs.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export REPO
export DATA_ROOT=${DATA_ROOT:-$REPO/data}
export HF_HOME=${HF_HOME:-$DATA_ROOT/hf}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}  # less fragmentation for the 2816-token sequences
export TERRA_MODEL=${TERRA_MODEL:-lotfollahi-lab/TERRA-96M}   # 112M: see ../README.md
export HF_HUB_OFFLINE=1          # TERRA-96M (and 112M) are pre-downloaded on the login node; compute nodes may lack egress
export TQDM_DISABLE=1
export PY=${PY:-/g/stegle/ddimitro/miniforge3/envs/terra/bin/python}
export SLIDES=${SLIDES:-"crc_120 crc_210 crc_221 crc_231 crc_232 crc_242"}
LOG=$REPO/scripts/terra/slurm/logs
export LOG
mkdir -p "$LOG"
# terra hardcodes cuda:0 -> Slurm's CUDA_VISIBLE_DEVICES masking makes that the allocated GPU.
# Never override it here.
cd "$REPO"
