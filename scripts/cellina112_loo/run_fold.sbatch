#!/bin/bash
#SBATCH --partition=gpu-el8
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu=A100|gpu=H100|gpu=L40s|gpu=A40|gpu=3090"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=c112
#SBATCH --output=scripts/cellina112_loo/logs/%x_%A_%a.log
# One array task = one (SID, CT) fold of the cellina v1.1.2 LOO sweep (seed 0).
#   1. scripts/multi_seed/train_loo.py --model_class cellina --cf_spatial_layer lognorm
#      (skipped when model + counterfactual h5ads already exist; --inference_only when only the model exists)
#   2. scripts/eval_loo.py --use_cf   (main eval; paper metrics)
#   3. metric JSONs + a timing JSON copied to results/cellina112_loo/<dataset>/
set -euo pipefail

C112_DIR=${C112_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/scripts/cellina112_loo}
source "$C112_DIR/env.sh"

if [ -z "${SID:-}" ] || [ -z "${CT:-}" ]; then
    [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "need SID/CT via --export or an array task" >&2; exit 1; }
    line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$C112_JOBS")
    [ -n "$line" ] || { echo "no line $((SLURM_ARRAY_TASK_ID + 1)) in $C112_JOBS" >&2; exit 1; }
    IFS=$'\t' read -r SID CT <<< "$line"
fi

OUT_NAME=${MODEL_NAME}_${SEED}
if [ "$DATASET" = crc ]; then
    ADATA=$DATA_ROOT/datasets/crc/raw_zenodo/$SID.h5ad
    HOLDOUT_DOMAINS=(CRC)
else
    ADATA=$DATA_ROOT/datasets/MERFISH_mouse_brain/$SID.h5ad
    HOLDOUT_DOMAINS=(Isocortex Fiber_tracts)
fi
MODEL_DIR=$DATA_ROOT/data/ood/trained/$SID/$CT/$OUT_NAME
CORR_DIR=$DATA_ROOT/datasets/$DATASET/correlations
JOBID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}
TAG="${SID}_${CT// /-}"
[ -f "$ADATA" ] || { echo "adata missing: $ADATA" >&2; exit 1; }

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
echo "== cellina112 dataset=$DATASET sid=$SID ct='$CT' out=$OUT_NAME node=${SLURMD_NODENAME:-?} gpu='$GPU_NAME' job=$JOBID $(date)"
echo "== PY=$PY  cellina=$("$PY" -c 'import cellina;print(cellina.__version__, cellina.__file__)')"

# ---- train + counterfactuals (multi_seed copy: passes layer='lognorm' to the cf helpers)
TIME_FILE=$C112_LOG/time_${DATASET}_${TAG}.txt
CF_ALL=1
for hd in "${HOLDOUT_DOMAINS[@]}"; do
    [ -f "$(dirname "$ADATA")/$SID/$CT/${OUT_NAME}_counterfactual_x_${hd}.h5ad" ] || CF_ALL=0
done
TRAIN_MODE=full; EXTRA=()
if [ -f "$MODEL_DIR/model.pt" ]; then
    if [ "$CF_ALL" = 1 ]; then TRAIN_MODE=skipped; else TRAIN_MODE=inference_only; EXTRA=(--inference_only); fi
fi
T0=$SECONDS
if [ "$TRAIN_MODE" = skipped ]; then
    echo "== train: model and counterfactuals exist, skipping"
else
    echo "== train ($TRAIN_MODE) $(date)"
    PYTHONPATH="$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/time -v -o "$TIME_FILE" \
        "$PY" scripts/multi_seed/train_loo.py \
            --dataset_name "$DATASET" --adata_path "$ADATA" --holdout_celltype "$CT" \
            --model_class cellina --model_name "$MODEL_NAME" --seed "$SEED" \
            --cf_spatial_layer lognorm "${EXTRA[@]+"${EXTRA[@]}"}"
    grep -E 'Elapsed|Maximum resident' "$TIME_FILE" || true
fi
TRAIN_WALL=$((SECONDS - T0))
MAX_RSS_KB=$(awk -F': *' '/Maximum resident set size/ {print $2}' "$TIME_FILE" 2>/dev/null | tr -dc '0-9' || true)

# ---- eval (main scripts/eval_loo.py; no PYTHONPATH so `train_loo` resolves to scripts/train_loo.py)
echo "== eval $(date)"
"$PY" scripts/eval_loo.py \
    --dataset_name "$DATASET" --adata_path "$ADATA" --holdout_celltype "$CT" \
    --model_class cellina --model_name "$OUT_NAME" --use_cf

# ---- collect
echo "== collect -> $C112_RESULTS"
shopt -s nullglob; n_json=0
for f in "$CORR_DIR/${SID}_${OUT_NAME}-cf_${CT}_"*.json; do
    cp -f "$f" "$C112_RESULTS/"; echo "   $(basename "$f")"; n_json=$((n_json + 1))
done
shopt -u nullglob
[ "$n_json" -gt 0 ] || echo "WARNING: no metric JSONs matched ${SID}_${OUT_NAME}-cf_${CT}_*.json" >&2

export SID CT OUT_NAME TRAIN_MODE TRAIN_WALL MAX_RSS_KB GPU_NAME JOBID N_JSON=$n_json
"$PY" - "$C112_RESULTS/timing_${SID}_${CT}.json" <<PYEOF
import json, os, sys
def num(x, cast=float):
    try: return cast((x or "").strip())
    except (TypeError, ValueError): return None
rss = num(os.environ.get("MAX_RSS_KB"), int)
rec = {"dataset": os.environ["DATASET"], "sid": os.environ["SID"], "cell_type": os.environ["CT"],
       "model_name": os.environ["OUT_NAME"], "train_mode": os.environ["TRAIN_MODE"],
       "train_wall_seconds": num(os.environ.get("TRAIN_WALL"), int),
       "max_rss_gib": round(rss / 1048576.0, 3) if rss else None,
       "gpu_name": os.environ.get("GPU_NAME") or None, "slurm_job_id": os.environ.get("JOBID"),
       "node": os.environ.get("SLURMD_NODENAME"), "n_metric_json": num(os.environ.get("N_JSON"), int)}
json.dump(rec, open(sys.argv[1], "w"), indent=2); print(json.dumps(rec))
PYEOF
echo "== done sid=$SID ct='$CT' train_wall=${TRAIN_WALL}s $(date)"
