#!/bin/bash
#SBATCH --job-name=terra-og
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=scripts/terra/slurm/logs/terra_og_%j.log
# TERRA-native perturbation evaluation (scripts/terra/TERRA_OG_EVAL_SPEC.md), one job per
# slide x arm x perturbation target.  Submitted by submit_terra_og.sh.
#   ARM     frozen | lora:N
#   TARGET  neighb_only | ct_neigh
#   SEEDS   optional, e.g. "0 1 2 3 4" -> the random-gene control (frozen arm only)
#   CTS     optional, space-separated scored cell types (default: all five)
# The perturbed datasets are model-independent and cached under
# $DATA_ROOT/datasets/crc/$SID/terra_og_cache/ with an atomic first-writer-wins rename, so the
# arms of one target can run as parallel jobs.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/terra/slurm/env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "terra_og sid=$SID arm=$ARM target=$TARGET seeds=${SEEDS:-} cts=${CTS:-} node=$SLURMD_NODENAME $(date)"
if [ "$ARM" = frozen ]; then V=(--arm frozen); else V=(--arm lora --epoch "${ARM#lora:}"); fi
EXTRA=()
[ -n "${SEEDS:-}" ] && EXTRA+=(--random-seeds "${SEEDS// /,}")
[ -n "${CTS:-}" ] && EXTRA+=(--cell-types "${CTS// /,}")
$PY scripts/terra/terra_og_eval.py --sid "$SID" "${V[@]}" --target "$TARGET" "${EXTRA[@]}"
echo "terra_og $SID $ARM $TARGET done $(date)"
