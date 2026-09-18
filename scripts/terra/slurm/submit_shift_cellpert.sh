#!/bin/bash
# Cell+neighbourhood shift track for one slide (see shift_cellpert.sbatch for what it tests):
#   scripts/terra/slurm/submit_shift_cellpert.sh crc_232
# Same arm and cell-type derivation as submit_stage2.sh, so the results pair 1:1 with the
# existing neighbourhood-only `*-native-terra2k_*` JSONs.  Override either list:
#   ARMS="frozen lora:5" CTS=Fibroblast scripts/terra/slurm/submit_shift_cellpert.sh crc_232
# `-p` submits ONE JOB PER ARM instead of one job running them in sequence, so the arms occupy
# separate GPUs.  They share the model-independent ctrl/pert token caches, which is safe because
# eval_terra._save_atomic writes them through a temp dir and keeps the first copy to land; each
# job just rebuilds its own in memory until one wins.  The cache directory is pinned with
# HOLDOUT_CT so every job reads and writes the same paths a single sequential job would.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PAR=0
[ "${1:-}" = "-p" ] && { PAR=1; shift; }
SID=${1:?usage: submit_shift_cellpert.sh [-p] <sid>}
MIN_VRAM=${MIN_VRAM:-48}; PARTITION=${PARTITION:-gpu-el8}
SEL=$DATA_ROOT/datasets/crc/$SID/terra/epoch_selection.json
[ -f "$SEL" ] || { echo "$SEL missing: stage 1 has not finished for $SID" >&2; exit 1; }
ARMS=${ARMS:-$($PY -c "import sys;sys.path.insert(0,'scripts/terra');import summarize as s
print(' '.join('frozen' if e is None else f'lora:{e}' for _,e,_ in s.arms('$DATA_ROOT/datasets/crc/$SID','$SID')))")}
ALL_CTS=$($PY -c "import sys;sys.path.insert(0,'scripts/terra');import summarize as s
d=s.cellina_df();print(' '.join(d[d.sid=='$SID'].holdout_celltype))")
CTS=${CTS:-$ALL_CTS}
export HOLDOUT_CT=${HOLDOUT_CT:-${ALL_CTS%% *}}   # pin the cache dir to the full run's choice
[ -n "$CTS" ] || { echo "no scored cell types for $SID" >&2; exit 1; }
case $PARTITION in
  gpu-el8)      declare -A VRAM=([L40s]=48 [A100]=80 [H100]=80) ;;
  gpu-training) declare -A VRAM=([H100]=80 [H200]=141) ;;
  *) echo "unknown partition $PARTITION" >&2; exit 1 ;;
esac
CONS=""
for g in "${!VRAM[@]}"; do [ "${VRAM[$g]}" -ge "$MIN_VRAM" ] && CONS="${CONS:+$CONS|}gpu=$g"; done
echo "sid: $SID | arms: $ARMS | cell types: $CTS | cache dir: $HOLDOUT_CT | parallel: $PAR"
echo "partition: $PARTITION | constraint: $CONS"
submit () {   # $1 = the arms this job runs, sequentially
  sbatch ${TEST_ONLY:+--test-only} --partition="$PARTITION" --gres=gpu:1 --constraint="$CONS" \
         --export=ALL,SID="$SID",ARMS="$1",CTS="$CTS",HOLDOUT_CT="$HOLDOUT_CT" \
         "$REPO/scripts/terra/slurm/shift_cellpert.sbatch"
}
if [ "$PAR" = 1 ]; then for ARM in $ARMS; do submit "$ARM"; done; else submit "$ARMS"; fi
