#!/usr/bin/env bash
# k=10000 pass for slide 210: STRICTLY SERIAL, one run at a time.
#
# Why not launch_per_ct.sh: on 210 (617k cells) the k=10000 within-domain kNN
# build alone peaks past 267 GB of host RAM -- 5 concurrent runs OOM-killed the
# host, and even 2 would not fit. k<=1000 runs only need 5-10 GB, so they keep
# using the normal launcher.
#
# Each run is watched; if it exceeds RSS_LIMIT_GB it is killed (so the kernel
# OOM killer never picks a healthy job as its victim) and we move on, leaving no
# result JSON -- a later re-run picks it up. Idempotent: skips existing results.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTS="${CTS:-Fibroblast Endothelial Myeloid T_cell Epithelial}"
SEEDS="${SEEDS:-0 1 2 3 4}"
# One run peaked at 386 GB and was still OOM-killed by the kernel because other
# jobs held ~60 GB. The hog is make_counterfactual_adata(precomputed=False):
# it round-trips the whole k-NN graph (~5e9 edges at k=10000) through COO and
# takes full copies. So: run solo, and only cap where the box would die anyway.
RSS_LIMIT_GB="${RSS_LIMIT_GB:-430}"
FREE_GB_REQUIRED="${FREE_GB_REQUIRED:-170}"
GPU="${GPU:-0}"
DATA="${DATA:-/data/ddimitrov/repos/cellina-reproducibility/data/crc_wt_cosmx/crc_210.h5ad}"
OUTROOT="${OUTROOT:-$SCRIPT_DIR/results_210}"
CKPTROOT="${CKPTROOT:-$SCRIPT_DIR/runs_210}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs_210}"
PYTHON="${PYTHON:-/data/ddimitrov/software/miniforge3/envs/cellina_edge/bin/python}"

mkdir -p "$LOGDIR"
echo "== k=10000 pass on GPU $GPU, seeds '$SEEDS' (RSS limit ${RSS_LIMIT_GB}GB) =="

for ct in $CTS; do
  mkdir -p "$OUTROOT/$ct" "$CKPTROOT/$ct"
  for seed in $SEEDS; do
    tag="k10000_seed${seed}"
    if [[ -f "$OUTROOT/$ct/$tag.json" ]]; then
      echo "[skip] $ct/$tag (result exists)"; continue
    fi
    # Wait until the rest of the machine is quiet enough to give this run room.
    while [[ $(free -g | awk '/Mem:/{print $7}') -lt "$FREE_GB_REQUIRED" ]]; do sleep 60; done

    echo "[$(date -Is)] [run] $ct/$tag"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/run_sensitivity.py" \
      --k 10000 --seed "$seed" --holdout-ct "$ct" --data "$DATA" --lean-cf \
      --control-domain 210_REF --target-domain 210_CRC \
      --outdir "$OUTROOT/$ct" --ckpt-root "$CKPTROOT/$ct" \
      > "$LOGDIR/${ct}_${tag}.log" 2>&1 &
    pid=$!

    peak=0
    while kill -0 "$pid" 2>/dev/null; do
      rss=$(awk '/VmRSS/{print int($2/1048576)}' /proc/$pid/status 2>/dev/null || echo 0)
      [[ -n "$rss" && "$rss" -gt "$peak" ]] && peak=$rss
      if [[ -n "$rss" && "$rss" -gt "$RSS_LIMIT_GB" ]]; then
        echo "[$(date -Is)] [kill] $ct/$tag rss=${rss}GB > ${RSS_LIMIT_GB}GB"
        kill -9 "$pid"; break
      fi
      sleep 30
    done
    wait "$pid"; rc=$?
    echo "[$(date -Is)] [done] $ct/$tag exit=$rc peak_rss=${peak}GB"
  done
done
echo "== k=10000 serial pass complete =="
