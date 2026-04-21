#!/bin/bash
# Launch all four ablations in parallel.
# clf / disc / domain_clf run simultaneously on GPU 0 (cellina env).
# graph runs on GPU 1 (cellina_edge env).
#
# Usage:
#   bash submit.sh                           # foreground, exits when all jobs finish
#   nohup bash submit.sh > submit.log 2>&1 & # background
#   tail -f logs/*.log                       # monitor progress

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/logs"

echo "=== Launching all 4 ablations in parallel ==="

# GPU 0: three cellina ablations
conda run -n cellina --no-capture-output \
    python -u "$DIR/run_ablations.py" --ablation clf        > "$DIR/logs/clf.log"        2>&1 &

conda run -n cellina --no-capture-output \
    python -u "$DIR/run_ablations.py" --ablation disc       > "$DIR/logs/disc.log"       2>&1 &

conda run -n cellina --no-capture-output \
    python -u "$DIR/run_ablations.py" --ablation domain_clf > "$DIR/logs/domain_clf.log" 2>&1 &

# GPU 1: graph ablation
conda run -n cellina_edge --no-capture-output \
    python -u "$DIR/run_ablations.py" --ablation graph      > "$DIR/logs/graph.log"      2>&1 &

echo "Jobs launched. Monitor with: tail -f $DIR/logs/*.log"
wait
echo "=== All ablations complete ==="
