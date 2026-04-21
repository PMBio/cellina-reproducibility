#!/bin/bash
# Launch all four ablations in separate tmux windows.
# clf / disc / domain_clf on GPU 0 (cellina env); graph on GPU 1 (cellina_edge env).
#
# Usage:
#   bash submit_tmux.sh
#   tmux attach -t ablations    # attach to watch progress
#   tmux kill-session -t ablations  # clean up when done

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION="ablations"
mkdir -p "$DIR/logs"

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n clf
tmux send-keys -t "$SESSION:clf" \
    "cd '$DIR' && conda run -n cellina --no-capture-output python -u run_ablations.py --ablation clf 2>&1 | tee logs/clf.log" Enter

tmux new-window -t "$SESSION" -n disc
tmux send-keys -t "$SESSION:disc" \
    "cd '$DIR' && conda run -n cellina --no-capture-output python -u run_ablations.py --ablation disc 2>&1 | tee logs/disc.log" Enter

tmux new-window -t "$SESSION" -n domain_clf
tmux send-keys -t "$SESSION:domain_clf" \
    "cd '$DIR' && conda run -n cellina --no-capture-output python -u run_ablations.py --ablation domain_clf 2>&1 | tee logs/domain_clf.log" Enter

tmux new-window -t "$SESSION" -n graph
tmux send-keys -t "$SESSION:graph" \
    "cd '$DIR' && conda run -n cellina_edge --no-capture-output python -u run_ablations.py --ablation graph 2>&1 | tee logs/graph.log" Enter

echo "Session '$SESSION' started with 4 windows (clf, disc, domain_clf, graph)."
echo "Attach with:  tmux attach -t $SESSION"
echo "Switch windows: Ctrl-b n / Ctrl-b p  or  Ctrl-b <window-name>"
