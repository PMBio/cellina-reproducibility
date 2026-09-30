#!/bin/bash
#SBATCH --partition=gpu-el8
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu=A100|gpu=H100|gpu=L40s|gpu=A40|gpu=3090"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=gat-hops
#SBATCH --output=scripts/gat_hops_batch/logs/%x_%j.log
# One job = one (ARM, SID, CT) fold of the Cellina-GAT hops / batch-size sensitivity test.
#
#   ARM  h1-b256 | h1-b512 | h3-b256   (see env.sh: arm_env)
#   SID  C57BL6J-2.036 | .039 | .041
#   CT   one of the five MERFISH holdout cell types (may contain spaces)
#
# ARM/SID/CT come either from --export (single gate job) or, when unset, from
# line $SLURM_ARRAY_TASK_ID of jobs.txt (tab separated), which submit.sh writes.
#
# Steps: train_loo.py (skipped / inference-only when the model already exists),
# eval_loo.py --use_cf, then the metric JSONs and a timing JSON are copied to
# results/gat_hops_batch/$ARM/.
set -euo pipefail

GHB_DIR=${GHB_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/scripts/gat_hops_batch}
source "$GHB_DIR/env.sh"

# ---- resolve the fold -------------------------------------------------------
if [ -z "${ARM:-}" ] || [ -z "${SID:-}" ] || [ -z "${CT:-}" ]; then
    [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || {
        echo "need ARM/SID/CT via --export, or an array task with $GHB_JOBS" >&2; exit 1; }
    [ -f "$GHB_JOBS" ] || { echo "$GHB_JOBS missing: run submit.sh first" >&2; exit 1; }
    line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GHB_JOBS")
    [ -n "$line" ] || { echo "no line $((SLURM_ARRAY_TASK_ID + 1)) in $GHB_JOBS" >&2; exit 1; }
    IFS=$'\t' read -r ARM SID CT <<< "$line"
fi
[ -n "$ARM" ] && [ -n "$SID" ] && [ -n "$CT" ] || { echo "empty ARM/SID/CT" >&2; exit 1; }

arm_env "$ARM"
MODEL_NAME=cellina-graph-$ARM
if [ "$DATASET" = crc ]; then
    ADATA=$DATA_ROOT/datasets/crc/raw_zenodo/$SID.h5ad
    HOLDOUT_DOMAINS=(CRC)
else
    ADATA=$DATA_ROOT/datasets/MERFISH_mouse_brain/$SID.h5ad
    HOLDOUT_DOMAINS=(Isocortex Fiber_tracts)
fi
MODEL_DIR=$DATA_ROOT/data/ood/trained/$SID/$CT/$MODEL_NAME
OUT_DIR=$GHB_RESULTS/$ARM
CORR_DIR=$DATA_ROOT/datasets/$DATASET/correlations
JOBID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}
TAG="${SID}_${CT// /-}"
mkdir -p "$OUT_DIR"

[ -f "$ADATA" ] || { echo "adata missing: $ADATA" >&2; exit 1; }

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
echo "== gat-hops dataset=$DATASET arm=$ARM sid=$SID ct='$CT' model=$MODEL_NAME"
echo "== node=${SLURMD_NODENAME:-$(hostname)} gpu='$GPU_NAME' job=$JOBID $(date)"
echo "== GAT_N_LAYERS=$GAT_N_LAYERS GAT_NUM_NEIGHBORS=$GAT_NUM_NEIGHBORS GAT_BATCH_SIZE=$GAT_BATCH_SIZE"
echo "== GAT_N_NEIGHBORS_GRAPH=$GAT_N_NEIGHBORS_GRAPH GAT_N_NEIGHBORS_PER_SEED=$GAT_N_NEIGHBORS_PER_SEED GHB_DROP_LAST=${GHB_DROP_LAST:-0}"

# ---- GPU memory sampler -----------------------------------------------------
GPU_SAMPLE=$(mktemp "${TMPDIR:-/tmp}/gat_gpu_XXXXXX.csv")
SMI_ID=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Slurm masks the allocation; -i keeps the sampler off other jobs' GPUs when it does not.
    if nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=memory.used \
         --format=csv,noheader,nounits >/dev/null 2>&1; then
        SMI_ID=(-i "${CUDA_VISIBLE_DEVICES%%,*}")
    fi
fi
SMI_PID=""
start_gpu_sampler() {
    nvidia-smi "${SMI_ID[@]+"${SMI_ID[@]}"}" --query-gpu=memory.used \
        --format=csv,noheader,nounits -l 5 > "$GPU_SAMPLE" 2>/dev/null &
    SMI_PID=$!
}
stop_gpu_sampler() {
    [ -n "$SMI_PID" ] && kill "$SMI_PID" 2>/dev/null || true
    wait "$SMI_PID" 2>/dev/null || true
    SMI_PID=""
}
cleanup() { stop_gpu_sampler; rm -f "$GPU_SAMPLE"; }
trap cleanup EXIT

# ---- train ------------------------------------------------------------------
TIME_FILE=$GHB_LOG/time_${ARM}_${TAG}.txt
TRAIN_MODE=full
TRAIN_ARGS_EXTRA=()
CF_ALL=1
for hd in "${HOLDOUT_DOMAINS[@]}"; do
    [ -f "$(dirname "$ADATA")/$SID/$CT/${MODEL_NAME}_counterfactual_x_${hd}.h5ad" ] || CF_ALL=0
done
if [ -f "$MODEL_DIR/model.pt" ]; then
    if [ "$CF_ALL" = 1 ]; then
        TRAIN_MODE=skipped
    else
        TRAIN_MODE=inference_only
        TRAIN_ARGS_EXTRA=(--inference_only)
    fi
fi

T0=$SECONDS
if [ "$TRAIN_MODE" = skipped ]; then
    echo "== train: $MODEL_DIR/model.pt and counterfactuals exist, skipping $(date)"
else
    echo "== train ($TRAIN_MODE) $(date)"
    start_gpu_sampler
    # GHB_DROP_LAST=1: same train_loo.py, via a wrapper that sets drop_last=True on the
    # training loader (only for folds whose n_train = 1 mod batch_size; see train_loo_drop_last.py).
    TRAIN_SCRIPT=scripts/train_loo.py
    [ "${GHB_DROP_LAST:-0}" = 1 ] && TRAIN_SCRIPT=$GHB_DIR/train_loo_drop_last.py
    /usr/bin/time -v -o "$TIME_FILE" \
        "$PY" "$TRAIN_SCRIPT" \
            --dataset_name "$DATASET" \
            --adata_path "$ADATA" \
            --holdout_celltype "$CT" \
            --model_class cellina_graph \
            --model_name "$MODEL_NAME" \
            "${TRAIN_ARGS_EXTRA[@]+"${TRAIN_ARGS_EXTRA[@]}"}"
    stop_gpu_sampler
    cat "$TIME_FILE"
fi
TRAIN_WALL=$((SECONDS - T0))

PEAK_GPU=$(awk 'NF && $1 ~ /^[0-9]+$/ && $1+0 > m {m=$1+0} END {print (m==""?"":m)}' "$GPU_SAMPLE" 2>/dev/null || true)
MAX_RSS_KB=$(awk -F': *' '/Maximum resident set size/ {print $2}' "$TIME_FILE" 2>/dev/null | tr -dc '0-9' || true)
TIME_WALL=$(awk -F': *' '/Elapsed \(wall clock\) time/ {print $2}' "$TIME_FILE" 2>/dev/null | tail -1 || true)

# ---- eval -------------------------------------------------------------------
echo "== eval $(date)"
"$PY" scripts/eval_loo.py \
    --dataset_name "$DATASET" \
    --adata_path "$ADATA" \
    --holdout_celltype "$CT" \
    --model_class cellina_graph \
    --model_name "$MODEL_NAME" \
    --use_cf

# ---- collect ----------------------------------------------------------------
echo "== collect -> $OUT_DIR"
shopt -s nullglob
n_json=0
for f in "$CORR_DIR/${SID}_${MODEL_NAME}-cf_${CT}_"*.json; do
    cp -f "$f" "$OUT_DIR/"
    echo "   $(basename "$f")"
    n_json=$((n_json + 1))
done
shopt -u nullglob
[ "$n_json" -gt 0 ] || echo "WARNING: no metric JSONs matched ${SID}_${MODEL_NAME}-cf_${CT}_*.json" >&2

TIMING_JSON=$OUT_DIR/timing_${SID}_${CT}.json
export ARM SID CT MODEL_NAME TRAIN_MODE TRAIN_WALL TIME_WALL PEAK_GPU MAX_RSS_KB GPU_NAME JOBID N_JSON=$n_json
"$PY" - "$TIMING_JSON" <<PYEOF
import json, os, sys
def num(x, cast=float):
    x = (x or "").strip()
    try:
        return cast(x)
    except (TypeError, ValueError):
        return None
max_rss_kb = num(os.environ.get("MAX_RSS_KB"), int)
rec = {
    "dataset": os.environ.get("DATASET"),
    "arm": os.environ["ARM"],
    "sid": os.environ["SID"],
    "cell_type": os.environ["CT"],
    "model_name": os.environ["MODEL_NAME"],
    "train_mode": os.environ["TRAIN_MODE"],
    "train_wall_seconds": num(os.environ.get("TRAIN_WALL"), int),
    "train_wall_time_v": os.environ.get("TIME_WALL") or None,
    "peak_gpu_mib": num(os.environ.get("PEAK_GPU"), int),
    "max_rss_kb": max_rss_kb,
    "max_rss_gib": round(max_rss_kb / 1048576.0, 3) if max_rss_kb else None,
    "gpu_name": os.environ.get("GPU_NAME") or None,
    "slurm_job_id": os.environ.get("JOBID") or None,
    "node": os.environ.get("SLURMD_NODENAME") or None,
    "n_metric_json": num(os.environ.get("N_JSON"), int),
    "gat_n_layers": num(os.environ.get("GAT_N_LAYERS"), int),
    "gat_num_neighbors": os.environ.get("GAT_NUM_NEIGHBORS"),
    "gat_batch_size": num(os.environ.get("GAT_BATCH_SIZE"), int),
    "gat_n_neighbors_graph": num(os.environ.get("GAT_N_NEIGHBORS_GRAPH"), int),
    "gat_n_neighbors_per_seed": num(os.environ.get("GAT_N_NEIGHBORS_PER_SEED"), int),
}
with open(sys.argv[1], "w") as fh:
    json.dump(rec, fh, indent=2)
print(json.dumps(rec))
PYEOF

echo "== done arm=$ARM sid=$SID ct='$CT' train_wall=${TRAIN_WALL}s peak_gpu=${PEAK_GPU:-NA}MiB $(date)"
