#!/bin/bash
# Shared environment for the full LOO sweep of the plain Cellina (MLP) model under
# cellina v1.1.2 (edge-perturbation counterfactual spatial features aggregated from
# adata.layers['lognorm'], i.e. the same representation as the training spatial_x).
#
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# Train + counterfactuals: scripts/multi_seed/train_loo.py (has --cf_spatial_layer, default
# 'lognorm'; seed 0 -> model dir / output basename '<MODEL_NAME>_0').
# Eval: scripts/eval_loo.py (main), so metrics are the paper's (incl. nb_deviance).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export REPO
export DATA_ROOT=${DATA_ROOT:-$REPO/data}
export PY=${PY:-/g/stegle/ddimitro/miniforge3/envs/cellina112/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TQDM_DISABLE=${TQDM_DISABLE:-1}

export DATASET=${DATASET:-merfish}
case "$DATASET" in merfish|crc) ;; *) echo "env.sh: unknown DATASET '$DATASET'" >&2; return 1 2>/dev/null || exit 1 ;; esac

export C112_DIR=$REPO/scripts/cellina112_loo
export C112_LOG=$C112_DIR/logs
export C112_JOBS=${C112_JOBS:-$C112_DIR/jobs_${DATASET}.txt}
export C112_RESULTS=$REPO/results/cellina112_loo/$DATASET
export MODEL_NAME=${MODEL_NAME:-cellina112}   # trained model / outputs are '<MODEL_NAME>_0' (seed suffix)
export SEED=${SEED:-0}
mkdir -p "$C112_LOG" "$C112_RESULTS"
cd "$REPO"
