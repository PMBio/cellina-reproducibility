#!/bin/bash
# Submit the TERRA-native perturbation evaluation for one slide (TERRA_OG_EVAL_SPEC.md §13):
#   scripts/terra/slurm/submit_terra_og.sh crc_232                 # 3 arms x 2 targets + 2 random jobs
#   ARMS=frozen scripts/terra/slurm/submit_terra_og.sh crc_232     # validation: frozen x 2 targets
#   RANDOM_SEEDS="" ARMS="lora:5 lora:2" scripts/terra/slurm/submit_terra_og.sh crc_232
# One GPU per job.  Arms derive from summarize.arms() (epoch_selection.json), never hardcoded.
# Override: ARMS, TARGETS, RANDOM_SEEDS (default "0 1 2 3 4"; "" skips the random jobs),
# CTS, MIN_VRAM, PARTITION, TEST_ONLY=1, ONLY_RANDOM=1 (submit just the random-control jobs),
# SKIP_DONE=1 (skip jobs whose five per-run JSONs already exist -- resubmit only what is missing).
# Per target the frozen job runs first and the other arms + the random job wait for it
# (--dependency=afterany), so one job builds the shared control/ and pert_{target}/ caches.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SID=${1:?usage: submit_terra_og.sh <sid>}
MIN_VRAM=${MIN_VRAM:-40}; PARTITION=${PARTITION:-gpu-el8}
SEL=$DATA_ROOT/datasets/crc/$SID/terra/epoch_selection.json
[ -f "$SEL" ] || { echo "$SEL missing: stage 1 has not finished for $SID" >&2; exit 1; }
ARMS=${ARMS:-$($PY -c "import sys;sys.path.insert(0,'scripts/terra');import summarize as s
print(' '.join('frozen' if e is None else f'lora:{e}' for _,e,_ in s.arms('$DATA_ROOT/datasets/crc/$SID','$SID')))")}
TARGETS=${TARGETS:-"neighb_only ct_neigh"}
RANDOM_SEEDS=${RANDOM_SEEDS-"0 1 2 3 4"}
case $PARTITION in
  gpu-el8)      declare -A VRAM=([L40s]=48 [A100]=40 [H100]=80) ;;   # gpu25-28 are A100-PCIE-40GB
  gpu-training) declare -A VRAM=([H100]=80 [H200]=141) ;;
  *) echo "unknown partition $PARTITION" >&2; exit 1 ;;
esac
CONS=""
for g in "${!VRAM[@]}"; do [ "${VRAM[$g]}" -ge "$MIN_VRAM" ] && CONS="${CONS:+$CONS|}gpu=$g"; done
echo "sid: $SID | arms: $ARMS | targets: $TARGETS | random seeds: '${RANDOM_SEEDS}' | cts: ${CTS:-all}"
echo "partition: $PARTITION | constraint: $CONS"
PER_RUN=$REPO/results/terra_og/per_run
done_already () {   # $1 arm label (terra-frozen / terra-lora-epN), $2 target, $3 "random" or ""
  [ -n "${SKIP_DONE:-}" ] || return 1
  for ct in Endothelial Epithelial Fibroblast Myeloid T_cell; do
    [ -f "$PER_RUN/${SID}_$1_$2${3:+_random}_$ct.json" ] || return 1
  done
}
submit () {   # $1 arm, $2 target, $3 seeds, $4 dependency job id or ""
  local lab; [ "$1" = frozen ] && lab=terra-frozen || lab=terra-lora-ep${1#lora:}
  if done_already "$lab" "$2" "${3:+random}"; then echo "skip $lab $2 ${3:+random}: per-run JSONs exist" >&2; return; fi
  sbatch --parsable ${TEST_ONLY:+--test-only} --partition="$PARTITION" --gres=gpu:1 --constraint="$CONS" \
         ${4:+--dependency=afterany:$4} \
         --job-name="terra-og-${1//:/}-$2${3:+-rand}" \
         --export=ALL,SID="$SID",ARM="$1",TARGET="$2",SEEDS="$3",CTS="${CTS:-}" \
         "$REPO/scripts/terra/slurm/terra_og.sbatch"
}
for T in $TARGETS; do
  DEP=""
  if [ -z "${ONLY_RANDOM:-}" ] && [[ " $ARMS " == *" frozen "* ]]; then
    DEP=$(submit frozen "$T" "" ""); [ -n "$DEP" ] && echo "Submitted batch job $DEP (frozen $T; builds the caches)"
  fi
  [ -z "${ONLY_RANDOM:-}" ] && for ARM in $ARMS; do
    [ "$ARM" = frozen ] && continue
    J=$(submit "$ARM" "$T" "" "$DEP"); [ -n "$J" ] && echo "Submitted batch job $J ($ARM $T${DEP:+, after $DEP})"
  done
  if [ -n "$RANDOM_SEEDS" ] && [[ " $ARMS " == *" frozen "* ]]; then
    J=$(submit frozen "$T" "$RANDOM_SEEDS" "$DEP"); [ -n "$J" ] && echo "Submitted batch job $J (random $T${DEP:+, after $DEP})"
  fi
done
