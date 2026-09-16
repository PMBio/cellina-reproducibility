#!/bin/bash
# One GPU, a list of slides: fine-tune -> pick arms -> decoder path -> shift path.
# A failing step aborts that SLIDE only; the worker moves on to the next one.
#   scripts/terra/queue/worker.sh 0 crc_120 crc_221
set -u
GPU=$1; shift
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LOG=$REPO/scripts/terra/queue/logs
PY=${PY:-/data/ddimitrov/software/miniforge3/envs/terra/bin/python}
export DATA_ROOT=${DATA_ROOT:-/data/ddimitrov/data}
export CUDA_VISIBLE_DEVICES=$GPU
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
    # crc_232 was trained before the queue existed and keeps its own path.
    local run=$DATA_ROOT/datasets/crc/$sid/terra/lora_run/run
    [ "$sid" = crc_232 ] && run=$DATA_ROOT/datasets/crc/crc_232/terra_lora5ep/lora_run/run
    local ftargs=(${FT_ARGS:-})        # e.g. FT_ARGS="--batch-size 32" for slides that OOM at 64
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
  # /data is full: the scored h5ads and the tokenizer cache are the two big regenerable artefacts.
  # PRUNE=0 keeps them (needed to re-score, e.g. nb_deviance, without re-running inference).
  local d=$DATA_ROOT/datasets/crc/$sid
  if [ "${PRUNE_TOK:-0}" = 1 ]; then    # regenerable from the raw h5ad, 18-27 GB per slide
    rm -rf $d/terra_tok $d/terra_tok_terra2k
    echo "PRUNED tok $(date) avail=$(df --output=avail -BG /data | tail -1)" >> "$LOG/${sid}_status.txt"
  fi
  if [ "${PRUNE_H5AD:-0}" = 1 ]; then   # only if disk forces it: re-scoring then needs a new inference run
    rm -f $d/*/terra-*_recon_x.h5ad $d/*/terra-*_counterfactual_x_*.h5ad
    echo "PRUNED h5ads $(date) avail=$(df --output=avail -BG /data | tail -1)" >> "$LOG/${sid}_status.txt"
  fi
}

for sid in "$@"; do
  : > "$LOG/${sid}_status.txt"
  slide "$sid" || echo "[worker $GPU] $sid FAILED, continuing"
done
step "gpu$GPU" summary $PY scripts/terra/summarize.py
echo "[worker $GPU] ALL DONE $(date)"
