#!/bin/bash
# Shared environment for the Cellina-GAT hops / batch-size sensitivity test
# (see GAT_HOPS_BATCH_TEST.md and scripts/gat_hops_batch/RUN.md).
#
# Source from a job script:
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# The arm-specific settings are applied by `arm_env <arm>`, which only exports
# environment variables.  Nothing here edits scripts/configs/cellina_graph_config.py:
# that file reads the GAT_* variables at import time and falls back to its own
# committed defaults when they are unset.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export REPO
export DATA_ROOT=${DATA_ROOT:-$REPO/data}
export PY=${PY:-/g/stegle/ddimitro/miniforge3/envs/cellina111/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TQDM_DISABLE=${TQDM_DISABLE:-1}

# DATASET=merfish (default, the original test) or crc (arm A follow-up on the six CRC slides).
export DATASET=${DATASET:-merfish}
case "$DATASET" in
    merfish) DS_SUFFIX="" ;;
    crc)     DS_SUFFIX="_crc" ;;
    *) echo "env.sh: unknown DATASET '$DATASET' (merfish|crc)" >&2; return 1 2>/dev/null || exit 1 ;;
esac
GHB_DIR=$REPO/scripts/gat_hops_batch
export GHB_DIR
export GHB_LOG=$GHB_DIR/logs
export GHB_JOBS=$GHB_DIR/jobs${DS_SUFFIX}.txt
export GHB_RESULTS=$REPO/results/gat_hops_batch${DS_SUFFIX//_//}
mkdir -p "$GHB_LOG" "$GHB_RESULTS"

# The three arms of the test.  Base config (identical in every arm) is
# n_layers=3, N_NEIGHBORS_GRAPH=50, N_NEIGHBORS_PER_SEED=50, seed 0.
#   h1-b256  ("May config")  num_neighbors=[-1,0,0]   batch_size=256  -> reference arm A
#   h1-b512                  num_neighbors=[-1,0,0]   batch_size=512  -> isolates batch size
#   h3-b256                  num_neighbors=[-1,-1,-1] batch_size=256  -> isolates hop count
# In this package version a negative entry means "all neighbours" (verified:
# [-20] samples the same node set as [-1]); 0 means "no new nodes at this hop".
#   h1lit-b256               num_neighbors=[-1]       batch_size=256  -> literal May setting (length-1 list, package warns); should equal A
#   h1lit-b256-pyg           same as h1lit-b256, run with PY=cellina112 (pyg-lib installed) -> timing comparison
#   h1k20-b256               num_neighbors=[20,0,0]   batch_size=256  -> capped fan-out (20 of the k=50 neighbours), CRC, run with pyg-lib (cellina112)
export GHB_ARMS_ALL="h1-b256 h1-b512 h3-b256 h1lit-b256 h1lit-b256-pyg h1k20-b256"

arm_env() {
    local arm=${1:?usage: arm_env <h1-b256|h1-b512|h3-b256|h1lit-b256>}
    export GAT_N_LAYERS=3
    export GAT_N_NEIGHBORS_GRAPH=50
    export GAT_N_NEIGHBORS_PER_SEED=50
    case "$arm" in
        h1-b256) export GAT_NUM_NEIGHBORS="-1,0,0";   export GAT_BATCH_SIZE=256 ;;
        h1-b512) export GAT_NUM_NEIGHBORS="-1,0,0";   export GAT_BATCH_SIZE=512 ;;
        h3-b256) export GAT_NUM_NEIGHBORS="-1,-1,-1"; export GAT_BATCH_SIZE=256 ;;
        h1lit-b256|h1lit-b256-pyg) export GAT_NUM_NEIGHBORS="-1"; export GAT_BATCH_SIZE=256 ;;
        h1k20-b256) export GAT_NUM_NEIGHBORS="20,0,0"; export GAT_BATCH_SIZE=256 ;;
        *) echo "arm_env: unknown arm '$arm' (expected one of: $GHB_ARMS_ALL)" >&2; return 1 ;;
    esac
    export GHB_ARM=$arm
}

cd "$REPO"
