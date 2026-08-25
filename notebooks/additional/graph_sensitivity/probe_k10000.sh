#!/usr/bin/env bash
# Measure peak RSS of a single k=10000 run so we can pick a safe concurrency.
# Kills the run if it exceeds RSS_LIMIT_GB so the kernel OOM killer never gets
# to pick a victim among the healthy k<=1000 jobs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CT="${CT:-Fibroblast}"
SEED="${SEED:-0}"
RSS_LIMIT_GB="${RSS_LIMIT_GB:-180}"
PYTHON=/data/ddimitrov/software/miniforge3/envs/cellina_edge/bin/python

mkdir -p "$SCRIPT_DIR/results_210/$CT" "$SCRIPT_DIR/runs_210/$CT" "$SCRIPT_DIR/logs_210"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCRIPT_DIR/run_sensitivity.py" \
  --k 10000 --seed "$SEED" --holdout-ct "$CT" \
  --data /data/ddimitrov/repos/cellina-reproducibility/data/crc_wt_cosmx/crc_210.h5ad \
  --control-domain 210_REF --target-domain 210_CRC \
  --outdir "$SCRIPT_DIR/results_210/$CT" --ckpt-root "$SCRIPT_DIR/runs_210/$CT" \
  > "$SCRIPT_DIR/logs_210/${CT}_k10000_seed${SEED}.log" 2>&1 &
PID=$!

peak=0
while kill -0 "$PID" 2>/dev/null; do
  rss=$(awk '/VmRSS/{print int($2/1048576)}' /proc/$PID/status 2>/dev/null || echo 0)
  [[ -n "$rss" && "$rss" -gt "$peak" ]] && peak=$rss
  if [[ -n "$rss" && "$rss" -gt "$RSS_LIMIT_GB" ]]; then
    echo "[$(date -Is)] RSS ${rss}GB > ${RSS_LIMIT_GB}GB limit -> killing $PID"
    kill -9 "$PID"; break
  fi
  echo "[$(date -Is)] rss=${rss}GB peak=${peak}GB free=$(free -g | awk '/Mem:/{print $7}')GB"
  sleep 20
done
wait "$PID"; rc=$?
echo "[$(date -Is)] exit=$rc peak_rss=${peak}GB"
