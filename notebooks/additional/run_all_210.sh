#!/usr/bin/env bash
# Detached driver for the slide-210 rebuttal sweeps.
#   0. wait for the PIDs in WAIT_PIDS (in-flight runs) so we never double-launch
#   1. graph sensitivity, k <= 1000  -- 6 concurrent (~40 GB RSS each)
#   2. graph sensitivity, k = 10000  -- serial, RSS-capped (the kNN build alone
#                                       peaks past 267 GB on 617k cells)
#   3. node fraction (inference only, reuses the k=200 checkpoints)
# Launch with:
#   WAIT_PIDS="123 456" setsid nohup bash run_all_210.sh > run_all_210.log 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_PIDS="${WAIT_PIDS:-}"

for pid in $WAIT_PIDS; do
  echo "[$(date -Is)] waiting for in-flight pid $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
echo "[$(date -Is)] in-flight runs drained"

echo "[$(date -Is)] === graph sensitivity, k <= 1000 (6 concurrent) ==="
KS="10 100 200 1000" JOBS_PER_GPU=3 bash "$SCRIPT_DIR/graph_sensitivity/launch_per_ct.sh"
echo "[$(date -Is)] k<=1000 exit=$?"

echo "[$(date -Is)] === graph sensitivity, k = 10000 (serial, RSS-capped) ==="
bash "$SCRIPT_DIR/graph_sensitivity/run_k10000_serial.sh"
echo "[$(date -Is)] k=10000 exit=$?"

echo "[$(date -Is)] === node fraction ==="
bash "$SCRIPT_DIR/node_fraction/launch_per_ct.sh"
echo "[$(date -Is)] node fraction exit=$?"
echo "[$(date -Is)] all done"
