#!/usr/bin/env bash
# Full per-slide rebuttal sweep in one driver: graph sensitivity -> node
# fraction -> figures + tables.
#
# Point it at a different slide with SLIDE=<id> (data path and domain labels are
# derived from it), e.g.  SLIDE=210 ./run_232.sh
#
# Order is a hard dependency, not a preference: run_node_fraction.py is
# inference-only and loads the k=200 checkpoint that the graph-sensitivity pass
# trains, so gs must finish first and its checkpoints must be retained.
#
#   gs   -> $GS_DIR/results_$SLIDE/<ct>/k{k}_seed{seed}.json
#           $GS_DIR/runs_$SLIDE/<ct>/k{k}_seed{seed}/          (checkpoints)
#   nf   -> $NF_DIR/results_$SLIDE/<ct>/frac{fraction}_seed{seed}.json
#   figs -> results_$SLIDE/{graph_sensitivity,node_fraction}_per_ct.{svg,png}
#
# Both passes are idempotent: a run whose result JSON exists is skipped, so
# re-running after an interruption resumes rather than redoes.
#
# Config via environment (all optional):
#   SLIDE  CTS  KS  FRACTIONS  SEEDS  GPUS  JOBS_PER_GPU  FREE_GB_REQUIRED
#   DATA  CONTROL_DOMAIN  TARGET_DOMAIN  PYTHON  SKIP_GS  SKIP_NF  SKIP_PLOTS
set -euo pipefail

SLIDE="${SLIDE:-232}"
REPO="/data/ddimitrov/repos/cellina-reproducibility"
ADD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GS_DIR="$ADD/graph_sensitivity"
NF_DIR="$ADD/node_fraction"

CTS="${CTS:-Fibroblast Endothelial Myeloid T_cell Epithelial}"
KS="${KS:-10 100 200 1000 10000}"
# fraction 0.0 is the unperturbed anchor; there is no gs equivalent (an empty
# graph leaves no donor pool, so the edge-perturbation counterfactual is
# undefined at k=0 -- hence no k=0 in KS).
FRACTIONS="${FRACTIONS:-0.0 0.05 0.1 0.25 0.5 0.75 1.0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
NF_K="${NF_K:-200}"                     # checkpoint the nf pass reuses

GPUS="${GPUS:-0 1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
# Don't start a new job unless this much RAM is free. The box is shared; a
# k=10000 run transiently takes tens of GB, so this throttles rather than
# trusting a fixed slot count. (An OOM kill here costs a whole run.)
FREE_GB_REQUIRED="${FREE_GB_REQUIRED:-100}"

DATA="${DATA:-$REPO/data/crc_wt_cosmx/crc_${SLIDE}.h5ad}"
CONTROL_DOMAIN="${CONTROL_DOMAIN:-${SLIDE}_REF}"
TARGET_DOMAIN="${TARGET_DOMAIN:-${SLIDE}_CRC}"
PYTHON="${PYTHON:-/data/ddimitrov/software/miniforge3/envs/cellina_edge/bin/python}"

RESULTS_TAG="${RESULTS_TAG:-}"     # e.g. _rescored -> results_232_rescored
GS_EXTRA="${GS_EXTRA:-}"           # e.g. "--reuse-ckpt --artifacts"
NF_EXTRA="${NF_EXTRA:-}"           # nf is inference-only: no --reuse-ckpt
GS_OUT="$GS_DIR/results_${SLIDE}${RESULTS_TAG}"; GS_CKPT="$GS_DIR/runs_$SLIDE"
GS_LOG="$GS_DIR/logs_${SLIDE}${RESULTS_TAG}"
NF_OUT="$NF_DIR/results_${SLIDE}${RESULTS_TAG}"; NF_LOG="$NF_DIR/logs_${SLIDE}${RESULTS_TAG}"
mkdir -p "$GS_OUT" "$GS_CKPT" "$GS_LOG" "$NF_OUT" "$NF_LOG"

[[ -f "$DATA" ]] || { echo "ERROR: no such data file: $DATA" >&2; exit 1; }

SLOTS=(); for g in $GPUS; do for _ in $(seq 1 "$JOBS_PER_GPU"); do SLOTS+=("$g"); done; done
NSLOTS=${#SLOTS[@]}

ts() { date -Is; }
echo "== slide $SLIDE sweep =="
echo "  data          : $DATA"
echo "  domains       : $CONTROL_DOMAIN -> $TARGET_DOMAIN"
echo "  cell types    : $CTS"
echo "  seeds         : $SEEDS"
echo "  k grid        : $KS"
echo "  fractions     : $FRACTIONS  (nf reuses k=$NF_K checkpoints)"
echo "  results dir   : $(basename "$GS_OUT") / $(basename "$NF_OUT")"
echo "  extra args    : gs=${GS_EXTRA:-(none)} nf=${NF_EXTRA:-(none)}"
echo "  concurrency   : $NSLOTS ($JOBS_PER_GPU per GPU on '$GPUS'), gated at ${FREE_GB_REQUIRED}GB free"
echo

declare -a PIDS=(); for ((s=0; s<NSLOTS; s++)); do PIDS[$s]=""; done
FREE_SLOT=-1
wait_for_slot() {
  while true; do
    for ((s=0; s<NSLOTS; s++)); do
      local pid="${PIDS[$s]}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then FREE_SLOT=$s; return 0; fi
    done
    sleep 5
  done
}
wait_for_ram() {
  while [[ $(free -g | awk '/Mem:/{print $7}') -lt "$FREE_GB_REQUIRED" ]]; do sleep 30; done
}

njobs=0; nskip=0
launch() {  # launch <logfile> <cmd...>
  local log="$1"; shift
  wait_for_slot; wait_for_ram
  local gpu="${SLOTS[$FREE_SLOT]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$@" >"$log" 2>&1 &
  PIDS[$FREE_SLOT]=$!
  njobs=$((njobs + 1))
  sleep 2
}

# ---- pass 1: graph sensitivity (trains; retains checkpoints for pass 2) -----
if [[ -z "${SKIP_GS:-}" ]]; then
  echo "[$(ts)] == pass 1/3: graph sensitivity =="
  for ct in $CTS; do
    mkdir -p "$GS_OUT/$ct" "$GS_CKPT/$ct"
    for k in $KS; do
      for seed in $SEEDS; do
        tag="k${k}_seed${seed}"
        if [[ -f "$GS_OUT/$ct/$tag.json" ]]; then nskip=$((nskip+1)); continue; fi
        echo "[$(ts)] [gs] $ct/$tag"
        launch "$GS_LOG/${ct}_${tag}.log" \
          "$PYTHON" "$GS_DIR/run_sensitivity.py" \
          --k "$k" --seed "$seed" --holdout-ct "$ct" --data "$DATA" --lean-cf \
          --control-domain "$CONTROL_DOMAIN" --target-domain "$TARGET_DOMAIN" \
          --outdir "$GS_OUT/$ct" --ckpt-root "$GS_CKPT/$ct" $GS_EXTRA
      done
    done
  done
  wait
  echo "[$(ts)] gs done: $(ls "$GS_OUT"/*/k*_seed*.json 2>/dev/null | wc -l) result(s)"
fi

# ---- pass 2: node fraction (inference only, reuses k=$NF_K checkpoints) -----
if [[ -z "${SKIP_NF:-}" ]]; then
  echo "[$(ts)] == pass 2/3: node fraction =="
  missing=0
  for ct in $CTS; do for seed in $SEEDS; do
    [[ -d "$GS_CKPT/$ct/k${NF_K}_seed${seed}" ]] || { echo "  MISSING ckpt $ct/k${NF_K}_seed${seed}" >&2; missing=1; }
  done; done
  [[ $missing -eq 0 ]] || { echo "ERROR: pass 1 did not leave the k=$NF_K checkpoints pass 2 needs." >&2; exit 1; }

  for ct in $CTS; do
    mkdir -p "$NF_OUT/$ct"
    for frac in $FRACTIONS; do
      for seed in $SEEDS; do
        tag="frac${frac}_seed${seed}"
        if [[ -f "$NF_OUT/$ct/$tag.json" ]]; then nskip=$((nskip+1)); continue; fi
        echo "[$(ts)] [nf] $ct/$tag"
        launch "$NF_LOG/${ct}_${tag}.log" \
          "$PYTHON" "$NF_DIR/run_node_fraction.py" \
          --fraction "$frac" --seed "$seed" --holdout-ct "$ct" --k "$NF_K" --data "$DATA" \
          --control-domain "$CONTROL_DOMAIN" --target-domain "$TARGET_DOMAIN" \
          --outdir "$NF_OUT/$ct" --ckpt-root "$GS_CKPT/$ct" $NF_EXTRA
      done
    done
  done
  wait
  echo "[$(ts)] nf done: $(ls "$NF_OUT"/*/frac*_seed*.json 2>/dev/null | wc -l) result(s)"
fi

# ---- pass 3: figures + tables ----------------------------------------------
if [[ -z "${SKIP_PLOTS:-}" ]]; then
  echo "[$(ts)] == pass 3/3: figures + tables =="
  "$PYTHON" "$GS_DIR/plot_per_ct.py" --results-root "$GS_OUT"
  "$PYTHON" "$NF_DIR/plot_per_ct.py" --results "$NF_OUT"
fi

echo "[$(ts)] == slide $SLIDE complete: $njobs launched, $nskip skipped =="
