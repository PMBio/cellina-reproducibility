#!/usr/bin/env bash
# Per-cell-type neighbor-graph sensitivity sweep on slide 210 (within-domain).
#
# Runs run_sensitivity.py for each (cell_type, k, seed), pinned to a GPU via
# CUDA_VISIBLE_DEVICES, at most JOBS_PER_GPU procs per GPU. Idempotent: skips
# any (ct, k, seed) whose result JSON already exists.
#
# Results  -> $OUTROOT/<ct>/k{k}_seed{seed}.json
# Ckpts    -> $CKPTROOT/<ct>/k{k}_seed{seed}/
#
# Config via environment variables (all optional):
#   CTS           cell types             (default: 5 LOO benchmark types)
#   KS            k grid                 (default: "10 100 200 1000 10000")
#   SEEDS         seeds                  (default: "0 1 2 3 4")
#   GPUS          GPU ids                (default: "0 1")
#   JOBS_PER_GPU  concurrent procs/GPU   (default: 3)
#   DATA          h5ad path              (default: repo crc_210.h5ad)
#   CONTROL_DOMAIN / TARGET_DOMAIN       (default: 210_REF / 210_CRC)
#   OUTROOT       results root           (default: <script>/results_210)
#   CKPTROOT      checkpoint root        (default: <script>/runs_210)
#   LOGDIR        per-run stdout/stderr  (default: <script>/logs_210)
#   PYTHON        interpreter            (default: cellina_edge env python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CTS="${CTS:-Fibroblast Endothelial Myeloid T_cell Epithelial}"
KS="${KS:-10 100 200 1000 10000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
GPUS="${GPUS:-0 1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
DATA="${DATA:-/data/ddimitrov/repos/cellina-reproducibility/data/crc_wt_cosmx/crc_210.h5ad}"
CONTROL_DOMAIN="${CONTROL_DOMAIN:-210_REF}"
TARGET_DOMAIN="${TARGET_DOMAIN:-210_CRC}"
OUTROOT="${OUTROOT:-$SCRIPT_DIR/results_210}"
CKPTROOT="${CKPTROOT:-$SCRIPT_DIR/runs_210}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs_210}"
PYTHON="${PYTHON:-/data/ddimitrov/software/miniforge3/envs/cellina_edge/bin/python}"

mkdir -p "$OUTROOT" "$CKPTROOT" "$LOGDIR"

# Flat list of GPU slots (e.g. "0 0 0 1 1 1"); job i -> slots[i % nslots].
SLOTS=()
for g in $GPUS; do
  for _ in $(seq 1 "$JOBS_PER_GPU"); do SLOTS+=("$g"); done
done
NSLOTS=${#SLOTS[@]}

echo "== per-cell-type graph-sensitivity sweep (slide 210, within-domain) =="
echo "  CTS          : $CTS"
echo "  KS           : $KS"
echo "  SEEDS        : $SEEDS"
echo "  GPUS         : $GPUS  (x${JOBS_PER_GPU} jobs each -> $NSLOTS concurrent)"
echo "  DATA         : $DATA"
echo "  DOMAINS      : $CONTROL_DOMAIN -> $TARGET_DOMAIN"
echo "  OUTROOT      : $OUTROOT"
echo "  CKPTROOT     : $CKPTROOT"
echo "  LOGDIR       : $LOGDIR"
echo "  PYTHON       : $PYTHON"
echo

declare -a PIDS=()
for ((s=0; s<NSLOTS; s++)); do PIDS[$s]=""; done

FREE_SLOT=-1
wait_for_slot() {
  while true; do
    for ((s=0; s<NSLOTS; s++)); do
      local pid="${PIDS[$s]}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        FREE_SLOT=$s
        return 0
      fi
    done
    sleep 5
  done
}

njobs=0; nskip=0
for ct in $CTS; do
  outdir="$OUTROOT/$ct"
  ckpt_root="$CKPTROOT/$ct"
  mkdir -p "$outdir" "$ckpt_root"
  for k in $KS; do
    for seed in $SEEDS; do
      tag="k${k}_seed${seed}"
      if [[ -f "$outdir/$tag.json" ]]; then
        echo "[skip] $ct/$tag (result exists)"
        nskip=$((nskip + 1))
        continue
      fi
      wait_for_slot
      gpu="${SLOTS[$FREE_SLOT]}"
      echo "[launch] $ct/$tag on GPU $gpu (slot $FREE_SLOT)"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT_DIR/run_sensitivity.py" \
        --k "$k" --seed "$seed" --holdout-ct "$ct" \
        --data "$DATA" \
        --control-domain "$CONTROL_DOMAIN" --target-domain "$TARGET_DOMAIN" \
        --outdir "$outdir" --ckpt-root "$ckpt_root" \
        >"$LOGDIR/${ct}_${tag}.log" 2>&1 &
      PIDS[$FREE_SLOT]=$!
      njobs=$((njobs + 1))
      sleep 2
    done
  done
done

echo
echo "Launched $njobs job(s), skipped $nskip; waiting for completion..."
wait
echo "== per-cell-type sweep complete =="
echo "Results in $OUTROOT ; logs in $LOGDIR"
