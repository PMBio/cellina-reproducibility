#!/usr/bin/env bash
# Complete the k=10000 row of the graph-sensitivity sweep on slide 210.
#
# ONE run at a time, and only when the machine actually has room. With
# --lean-cf a run peaks at ~226GB (graph construction; the counterfactual is no
# longer the hog -- see test_lean_cf.py). The other tenants here are heavy and
# variable: node_fraction grows with perturb_fraction (up to ~39GB per job, 6
# at a time) plus unrelated user jobs. A fixed schedule OOMs, so instead each
# run waits for FREE_GB_REQUIRED before starting, which sequences it behind
# whatever else is busy. Idempotent: skips any (ct, seed) already done.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FREE_GB_REQUIRED="${FREE_GB_REQUIRED:-320}"
export RSS_LIMIT_GB="${RSS_LIMIT_GB:-430}"
export GPU="${GPU:-1}"
echo "[$(date -Is)] k=10000 grid: gate=${FREE_GB_REQUIRED}GB free, cap=${RSS_LIMIT_GB}GB, GPU=$GPU"
SEEDS="0 1 2 3 4" bash "$SCRIPT_DIR/run_k10000_serial.sh"
echo "[$(date -Is)] k=10000 grid exit=$?"
