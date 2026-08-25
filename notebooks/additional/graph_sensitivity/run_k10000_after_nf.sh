#!/usr/bin/env bash
# Run the k=10000 row AFTER node_fraction finishes.
#
# Three attempts died to the OOM killer while sharing the box: the k=10000
# within-domain graph build needs >=309GB, node_fraction takes up to ~39GB x 6
# jobs, and unrelated user jobs hold ~95GB. 503GB does not cover that, and a
# free-memory gate at start-up is not enough because the other jobs grow while
# this one climbs. So: wait for node_fraction to finish, then take the box.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NF_DIR="$SCRIPT_DIR/../node_fraction/results_210"

echo "[$(date -Is)] waiting for node_fraction to finish (need 150 results)"
while :; do
  n=$(ls "$NF_DIR"/*/*.json 2>/dev/null | wc -l)
  running=$(ps -eo command= | grep -c "[r]un_node_fraction.py")
  [[ "$n" -ge 150 || "$running" -eq 0 ]] && break
  sleep 300
done
echo "[$(date -Is)] node_fraction done ($(ls "$NF_DIR"/*/*.json 2>/dev/null | wc -l)/150), starting k=10000"

FREE_GB_REQUIRED=390 RSS_LIMIT_GB=460 GPU=1 SEEDS="0 1 2 3 4" \
  bash "$SCRIPT_DIR/run_k10000_serial.sh"
echo "[$(date -Is)] k=10000 grid exit=$?"
