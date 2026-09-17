#!/bin/bash
# One GPU, a list of slides: fine-tune -> pick arms -> decoder path -> shift path.
# A failing step aborts that SLIDE only; the worker moves on to the next one.
#   scripts/terra/queue/worker.sh auto crc_120 crc_221   # inside a Slurm job: use the allocated GPU
#   scripts/terra/queue/worker.sh 0 crc_120              # bare machine: pin physical GPU 0
set -u
GPU=$1; shift
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LOG=${LOG:-$REPO/scripts/terra/queue/logs}
PY=${PY:-/g/stegle/ddimitro/miniforge3/envs/terra/bin/python}
export DATA_ROOT=${DATA_ROOT:-$REPO/data}
export HF_HOME=${HF_HOME:-$DATA_ROOT/hf} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TQDM_DISABLE=1
[ "$GPU" != auto ] && export CUDA_VISIBLE_DEVICES=$GPU
cd "$REPO"
mkdir -p "$LOG"

step() {  # step SID NAME CMD...
  local sid=$1 n=$2; shift 2
  local st=$LOG/${sid}_status.txt
  echo "START $n $(date)" >> "$st"
  "$@" > "$LOG/${sid}_${n}.log" 2>&1
  local rc=$?
  echo "END $n rc=$rc $(date)" >> "$st"
  [ $rc -ne 0 ] && echo "ABORT at $n" >> "$st"
  return $rc
}

slide() {
  local sid=$1
  local a=$DATA_ROOT/datasets/crc/raw_zenodo/$sid.h5ad
  # cell types scored for this slide = the cellina-pert rows of the reference CSV
  local cts ct1
  cts=$($PY -c "import sys;sys.path.insert(0,'scripts/terra');import summarize as s
d=s.cellina_df();print(','.join(d[d.sid=='$sid'].holdout_celltype))") || return 1
  [ -n "$cts" ] || { echo "no scored cell types for $sid" >> "$LOG/${sid}_status.txt"; return 1; }
  ct1=${cts%%,*}

  local arms
  if [ "${FROZEN_ONLY:-0}" = 1 ]; then      # no fine-tune, no LoRA arms
    arms=frozen
  else
    # re-entrant: a run dir with all 5 checkpoints is reused (export/embed/guard only, no retraining).
    local run=$DATA_ROOT/datasets/crc/$sid/terra/lora_run/run
    local ftargs=(${FT_ARGS:-})        # extra finetuning_lora.py flags, e.g. FT_ARGS="--batch-size 64"
    [ -f "$run/checkpoint_epoch_5.pt" ] && ftargs+=(--from-run-dir "$run")
    step "$sid" ft $PY scripts/terra/finetuning_lora.py --dataset_name crc --adata_path "$a" \
      ${ftargs[@]+"${ftargs[@]}"} || return 1
    # arms: "frozen" and "lora N" (final epoch + the latest guard-passing one when different)
    arms=$($PY -c "import sys;sys.path.insert(0,'scripts/terra');import summarize as s
print('\n'.join(('frozen' if e is None else f'lora {e}') for _,e,_ in s.arms('$DATA_ROOT/datasets/crc/$sid','$sid')))") || return 1
  fi
  echo "$sid arms: $(echo $arms | tr '\n' ';') | cell types: $cts" >> "$LOG/${sid}_status.txt"

  while read -r variant epoch; do
    local ep=() tag=$variant
    [ -n "${epoch:-}" ] && { ep=(--epoch "$epoch"); tag=${variant}_ep$epoch; }
    step "$sid" "inf_$tag" $PY scripts/terra/inference.py --dataset_name crc --adata_path "$a" \
      --holdout_celltype "$ct1" --eval-celltypes "$cts" --variant "$variant" ${ep[@]+"${ep[@]}"} || return 1
    step "$sid" "shift_$tag" $PY scripts/terra/eval_terra.py --dataset_name crc --adata_path "$a" \
      --holdout_celltype "$ct1" --eval-celltypes "$cts" --variant "$variant" ${ep[@]+"${ep[@]}"} \
      --universe terra2k --cellina-cf cellina-pert || return 1
  done <<< "$arms"
  echo "SLIDE DONE $(date)" >> "$LOG/${sid}_status.txt"
}

for sid in "$@"; do
  : > "$LOG/${sid}_status.txt"
  slide "$sid" || echo "[worker $GPU] $sid FAILED, continuing"
done
step "gpu$GPU" summary $PY scripts/terra/summarize.py
echo "[worker $GPU] ALL DONE $(date)"
