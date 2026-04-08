#!/usr/bin/env bash
# =============================================================================
# run_perturb_loo.sh — LOO perturbation benchmark: cellina vs spatialprop
# =============================================================================
# Runs cellina_loo.py (GPU 1, cellina conda env) and spatialprop_loo.py
# (GPU 0, spprop conda env) concurrently, then merges their results into a
# single benchmark_{slide_id}.csv.
#
# Usage:
#   bash run_perturb_loo.sh [options]
#
# Options:
#   --slide_id       INT   Slide identifier (default: 242)
#   --groupby        STR   Comma-separated holdout cell types
#                          (default: uses DEFAULT_GROUPBY from perturb_utils.py)
#   --top_n_perturb  INT   Genes used to perturb neighbor expression (default: 100)
#   --top_n          INT   Genes used for metric evaluation (default: 100)
#   --max_epochs     INT   Training epochs for both models (default: 100)
#   --batch_size     INT   Batch size for training/inference (default: 512)
#   --out_dir        STR   Output directory (default: script directory)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
SLIDE_ID=242
GROUPBY=""          # empty → Python scripts use DEFAULT_GROUPBY from perturb_utils.py
TOP_N_PERTURB=100
TOP_N=50
MAX_EPOCHS=100
BATCH_SIZE=512
OUT_DIR="$SCRIPT_DIR/results/perturb_benchmark"

# ── Python interpreters (avoids conda run hanging on backgrounded jobs) ───────
CELLINA_PYTHON=/data/ddimitrov/software/miniforge3/envs/cellina/bin/python
SPPROP_PYTHON=/data/ddimitrov/software/miniforge3/envs/spprop/bin/python

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --slide_id)       SLIDE_ID="$2";       shift 2 ;;
        --groupby)        GROUPBY="$2";        shift 2 ;;
        --top_n_perturb)  TOP_N_PERTURB="$2";  shift 2 ;;
        --top_n)          TOP_N="$2";          shift 2 ;;
        --max_epochs)     MAX_EPOCHS="$2";     shift 2 ;;
        --batch_size)     BATCH_SIZE="$2";     shift 2 ;;
        --out_dir)        OUT_DIR="$2";        shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
mkdir -p "$OUT_DIR"

CELLINA_LOG="$OUT_DIR/cellina_${SLIDE_ID}.log"
SPPROP_LOG="$OUT_DIR/spatialprop_${SLIDE_ID}.log"

echo "============================================================"
echo "  LOO Perturbation Benchmark"
echo "  slide_id:      $SLIDE_ID"
echo "  groupby:       ${GROUPBY:-(default from perturb_utils.py)}"
echo "  top_n_perturb: $TOP_N_PERTURB"
echo "  top_n:         $TOP_N"
echo "  max_epochs:    $MAX_EPOCHS"
echo "  batch_size:    $BATCH_SIZE"
echo "  out_dir:       $OUT_DIR"
echo "============================================================"

# ── Launch jobs concurrently ──────────────────────────────────────────────────

echo ""
echo "Launching cellina_loo.py  (GPU 1 | cellina env) → $CELLINA_LOG"
CUDA_VISIBLE_DEVICES=1 "$CELLINA_PYTHON" "$SCRIPT_DIR/cellina_loo.py" \
        --slide_id       "$SLIDE_ID" \
        ${GROUPBY:+--groupby "$GROUPBY"} \
        --top_n_perturb  "$TOP_N_PERTURB" \
        --top_n          "$TOP_N" \
        --max_epochs     "$MAX_EPOCHS" \
        --batch_size     "$BATCH_SIZE" \
        --out_dir        "$OUT_DIR" \
    > "$CELLINA_LOG" 2>&1 &
CELLINA_PID=$!

echo "Launching spatialprop_loo.py (GPU 0 | spprop env)  → $SPPROP_LOG"
CUDA_VISIBLE_DEVICES=0 "$SPPROP_PYTHON" "$SCRIPT_DIR/spatialprop_loo.py" \
        --slide_id       "$SLIDE_ID" \
        ${GROUPBY:+--groupby "$GROUPBY"} \
        --top_n_perturb  "$TOP_N_PERTURB" \
        --top_n          "$TOP_N" \
        --max_epochs     "$MAX_EPOCHS" \
        --batch_size     "$BATCH_SIZE" \
        --out_dir        "$OUT_DIR" \
    > "$SPPROP_LOG" 2>&1 &
SPPROP_PID=$!

echo ""
echo "Running ... (cellina PID=$CELLINA_PID | spatialprop PID=$SPPROP_PID)"

# ── Wait for both jobs ────────────────────────────────────────────────────────
FAILED=0

wait "$CELLINA_PID" && CELLINA_EXIT=0 || CELLINA_EXIT=$?
wait "$SPPROP_PID"  && SPPROP_EXIT=0  || SPPROP_EXIT=$?

if [[ $CELLINA_EXIT -ne 0 ]]; then
    echo "ERROR: cellina_loo.py failed (exit $CELLINA_EXIT). See $CELLINA_LOG"
    FAILED=1
fi
if [[ $SPPROP_EXIT -ne 0 ]]; then
    echo "ERROR: spatialprop_loo.py failed (exit $SPPROP_EXIT). See $SPPROP_LOG"
    FAILED=1
fi

[[ $FAILED -ne 0 ]] && exit 1

# ── Merge results ─────────────────────────────────────────────────────────────
BENCHMARK_PATH="$OUT_DIR/benchmark_${SLIDE_ID}.csv"

"$SPPROP_PYTHON" - <<PYEOF
import os, sys
import pandas as pd

out_dir  = "$OUT_DIR"
slide_id = "$SLIDE_ID"

files = [
    f"{out_dir}/cellina_results_{slide_id}.csv",
    f"{out_dir}/spatialprop_results_{slide_id}.csv",
]
existing = [f for f in files if os.path.exists(f)]

if not existing:
    print("ERROR: no result files found to merge!", file=sys.stderr)
    sys.exit(1)

merged = pd.concat([pd.read_csv(f) for f in existing], ignore_index=True)
out_path = f"{out_dir}/benchmark_{slide_id}.csv"
merged.to_csv(out_path, index=False)
print(f"Merged {len(merged)} rows from {len(existing)} file(s) → {out_path}")
PYEOF

echo ""
echo "Done. Joint results: $BENCHMARK_PATH"
