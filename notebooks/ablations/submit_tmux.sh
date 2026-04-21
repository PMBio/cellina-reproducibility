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

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n clf
tmux send-keys -t "$SESSION:clf" \
    "cd '$DIR' && conda run -n cellina python -u run_ablations.py --ablation clf" Enter

tmux new-window -t "$SESSION" -n disc
tmux send-keys -t "$SESSION:disc" \
    "cd '$DIR' && conda run -n cellina python -u run_ablations.py --ablation disc" Enter

tmux new-window -t "$SESSION" -n domain_clf
tmux send-keys -t "$SESSION:domain_clf" \
    "cd '$DIR' && conda run -n cellina python -u run_ablations.py --ablation domain_clf" Enter

tmux new-window -t "$SESSION" -n graph
tmux send-keys -t "$SESSION:graph" \
    "cd '$DIR' && conda run -n cellina_edge python -u run_ablations.py --ablation graph" Enter

echo "Session '$SESSION' started with 4 windows (clf, disc, domain_clf, graph)."
echo "Attach with:  tmux attach -t $SESSION"
echo "Switch windows: Ctrl-b n / Ctrl-b p  or  Ctrl-b <window-name>"
