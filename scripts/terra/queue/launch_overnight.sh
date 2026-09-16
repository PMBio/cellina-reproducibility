#!/bin/bash
# Split the 6 slides across GPU 0 and GPU 1 by cell count and start one worker on each.
set -eu
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LOG=$REPO/scripts/terra/queue/logs
PY=${PY:-/data/ddimitrov/software/miniforge3/envs/terra/bin/python}
export DATA_ROOT=${DATA_ROOT:-/data/ddimitrov/data}
SLIDES=${SLIDES:-"crc_120 crc_210 crc_221 crc_231 crc_232 crc_242"}
mkdir -p "$LOG"

# n_obs straight from the h5ad index (h5py, X is never read); crc_232 skips the fine-tune,
# which is most of the work, so it counts at ~1/3 of its cells.
read -r G0 G1 < <($PY - "$SLIDES" <<'PY'
import sys, h5py
from pathlib import Path
root = Path(__import__("os").environ["DATA_ROOT"]) / "datasets/crc/raw_zenodo"
w = []
for sid in sys.argv[1].split():
    with h5py.File(root / f"{sid}.h5ad", "r") as f:
        o = f["obs"]
        n = len(o[o.attrs["_index"]])
    w.append((n / 3 if sid == "crc_232" else n, sid, n))
gpu = {0: [], 1: []}, [0.0, 0.0]
for cost, sid, n in sorted(w, reverse=True):          # greedy longest-first
    g = 0 if gpu[1][0] <= gpu[1][1] else 1
    gpu[0][g].append(sid); gpu[1][g] += cost
    print(f"{sid:9s} n_obs={n:8,d} cost={cost:10,.0f} -> GPU {g}", file=sys.stderr)
print(f"GPU 0 cost {gpu[1][0]:,.0f} | GPU 1 cost {gpu[1][1]:,.0f}", file=sys.stderr)
size = {sid: n for _, sid, n in w}
for g in gpu[0].values():                             # smallest first: early results, 232 leads
    g.sort(key=size.get)
print(",".join(gpu[0][0]), ",".join(gpu[0][1]))
PY
)
echo "GPU 0: ${G0//,/ }"
echo "GPU 1: ${G1//,/ }"
nohup "$REPO/scripts/terra/queue/worker.sh" 0 ${G0//,/ } > "$LOG/worker0.log" 2>&1 &
echo "worker 0 pid $!"
nohup "$REPO/scripts/terra/queue/worker.sh" 1 ${G1//,/ } > "$LOG/worker1.log" 2>&1 &
echo "worker 1 pid $!"
