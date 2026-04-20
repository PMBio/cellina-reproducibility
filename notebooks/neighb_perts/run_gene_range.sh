#!/usr/bin/env bash
# run_gene_range.sh — submit perturb_gene_range.py over all CRC slides.
#
# Usage:
#   bash run_gene_range.sh [-n MAX_CONCURRENT]
#
# Options:
#   -n N   Max concurrent jobs (default: 2)

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SLIDE_IDS=('120' '210' '221' '231' '232' '242') # 222 is excluded due to missing labels; 110 has patches missing
MAX_CONCURRENT=2
GPU_IDS=(1)            # all jobs run on GPU 1
OUT_DIR="results/gene_range"

while getopts "n:" opt; do
    case $opt in
        n) MAX_CONCURRENT=$OPTARG ;;
        *) echo "Usage: $0 [-n MAX_CONCURRENT]" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

pids=()
slot=0

for sid in "${SLIDE_IDS[@]}"; do
    gpu=${GPU_IDS[$((slot % ${#GPU_IDS[@]}))]}

    # Wait until a concurrency slot is free
    while [ ${#pids[@]} -ge "$MAX_CONCURRENT" ]; do
        new_pids=()
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -ge "$MAX_CONCURRENT" ] && sleep 5
    done

    log_file="${OUT_DIR}/slide_${sid}.log"
    echo "[$(date '+%H:%M:%S')] Launching slide $sid on GPU $gpu (log: $log_file)"
    CUDA_VISIBLE_DEVICES=$gpu python perturb_gene_range.py \
        --slide_id "$sid" \
        --gpu 0 \
        --out_dir "$OUT_DIR" \
        > "$log_file" 2>&1 &
    pids+=($!)
    slot=$((slot + 1))
done

wait
echo "[$(date '+%H:%M:%S')] All slides done."
