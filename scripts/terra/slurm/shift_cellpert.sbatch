#!/bin/bash
#SBATCH --job-name=terra-cellpert
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=scripts/terra/slurm/logs/cellpert_%j.log
# Shift track ONLY, with `--perturb-cell`: the logFC shift is applied to the focal cell's own
# token block as well as to the neighbourhood, i.e. TERRA's own
# `perturbation_target=["cell", "neighborhood"]` instead of our neighbourhood-only default.
#
# Why: the neighbourhood-only shift scores at chance on all six slides (precision@50 ~= the
# random-gene control, pearson vs |logFC| negative in 30/32 combinations).  One candidate
# explanation is that the perturbation only reaches the scored tokens -- the cell's own 256
# positions -- through attention, a second-order effect.  Perturbing the cell directly is the
# control that separates "the model does not propagate neighbourhood signal" from "the readout
# is broken".  Paired with the existing `*-native-terra2k_*` results: same slide, same arms,
# same cell types, same seeds, same 15k control-cell cap.
#
# Writes to correlations/{sid}_terra-<arm>-native-terra2k-cellpert_{ct}_CRC.json and
# {sid}/{ct}/terra/gene_shift_<variant>_CRC_<ct>_terra2k_cellpert.csv, so nothing already on
# disk is touched.  The perturbed token sets get their own `_cellpert` caches for the same
# reason.  Arms listed in one job run sequentially; `submit_shift_cellpert.sh -p` instead puts
# each arm in its own job.  That is safe here because the shared caches are written atomically
# (eval_terra._save_atomic): the first job to finish one wins and the rest keep its copy.
# Submitted by submit_shift_cellpert.sh.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/terra/slurm/env.sh"
A=$DATA_ROOT/datasets/crc/raw_zenodo/$SID.h5ad
CTS_CSV=${CTS// /,}
# HOLDOUT_CT only picks the cache directory ({sid}/$HOLDOUT_CT/terra) -- load_dataset does not
# subset by it -- so pinning it to one value keeps per-cell-type caches and results identical to
# the single sequential job, whichever cell types a given job scores.
CT1=${HOLDOUT_CT:-${CTS%% *}}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cellpert sid=$SID arms=$ARMS cts=$CTS_CSV node=$SLURMD_NODENAME $(date)"
for ARM in $ARMS; do
  if [ "$ARM" = frozen ]; then V=(--variant frozen); else V=(--variant lora --epoch "${ARM#lora:}"); fi
  echo "== $ARM cellpert shift track $(date)"
  $PY scripts/terra/eval_terra.py --dataset_name crc --adata_path "$A" --holdout_celltype "$CT1" \
      --eval-celltypes "$CTS_CSV" "${V[@]}" --universe terra2k --perturb-cell
  echo "== $ARM done $(date)"
done
echo "cellpert $SID done $(date)"
