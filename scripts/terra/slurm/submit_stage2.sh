#!/bin/bash
# Stage 2 for one slide whose fine-tune has finished (epoch_selection.json exists):
#   scripts/terra/slurm/submit_stage2.sh crc_232
# Derives the arms (frozen, lora-ep{final}, lora-ep{latest_passing}) from epoch_selection.json via
# summarize.arms() and the scored cell types from the cellina-pert reference CSV (summarize.cellina_df()),
# and submits ONE stage2.sbatch job (arms run sequentially; they share caches) on $PARTITION
# (default gpu-el8, any card >= MIN_VRAM GB).  Cell types are exported space-separated: --export splits on commas.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SID=${1:?usage: submit_stage2.sh <sid>}
MIN_VRAM=${MIN_VRAM:-48}; PARTITION=${PARTITION:-gpu-el8}
SEL=$DATA_ROOT/datasets/crc/$SID/terra/epoch_selection.json
[ -f "$SEL" ] || { echo "$SEL missing: stage 1 has not finished for $SID" >&2; exit 1; }
ARMS=$($PY -c "import sys;sys.path.insert(0,'scripts/terra');import summarize as s
print(' '.join('frozen' if e is None else f'lora:{e}' for _,e,_ in s.arms('$DATA_ROOT/datasets/crc/$SID','$SID')))")
CTS=$($PY -c "import sys;sys.path.insert(0,'scripts/terra');import summarize as s
d=s.cellina_df();print(' '.join(d[d.sid=='$SID'].holdout_celltype))")
[ -n "$CTS" ] || { echo "no scored cell types for $SID" >&2; exit 1; }
case $PARTITION in
  gpu-el8)      declare -A VRAM=([L40s]=48 [A100]=80 [H100]=80) ;;
  gpu-training) declare -A VRAM=([H100]=80 [H200]=141) ;;
  *) echo "unknown partition $PARTITION" >&2; exit 1 ;;
esac
CONS=""
for g in "${!VRAM[@]}"; do [ "${VRAM[$g]}" -ge "$MIN_VRAM" ] && CONS="${CONS:+$CONS|}gpu=$g"; done
echo "sid: $SID | arms: $ARMS | cell types: $CTS | partition: $PARTITION | constraint: $CONS"
sbatch ${TEST_ONLY:+--test-only} --partition="$PARTITION" --gres=gpu:1 --constraint="$CONS" \
       --export=ALL,SID="$SID",ARMS="$ARMS",CTS="$CTS" "$REPO/scripts/terra/slurm/stage2.sbatch"
