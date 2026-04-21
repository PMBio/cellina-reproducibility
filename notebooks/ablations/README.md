# Ablations

Lambda-sweep ablation studies for CellinaModel and CellinaGraph on CRC slide 242.

## Files

| File | Description |
|------|-------------|
| `run_ablations.py` | Training + evaluation script. Accepts `--ablation {clf,disc,domain_clf,graph}`. For each (lambda, seed) pair: trains a model, evaluates F1 (celltype + spatial domain) and marginal LL, appends to the results CSV. Resume-safe. |
| `submit_tmux.sh` | **Preferred.** Launches all four ablations in a tmux session (`ablations`) with one window per job. Output is shown interactively and simultaneously written to `logs/*.log`. |
| `submit.sh` | Background alternative. Launches all four ablations in parallel as background jobs, redirecting output directly to `logs/*.log`. |
| `plot_ablations.ipynb` | Loads the four result CSVs and produces two figures (F1 and marginal LL), each with four panels. |
| `results/` | Output CSVs created at runtime (one per ablation type). |
| `logs/` | Log files created at runtime (one per ablation type). |

## Ablation design

All four ablations sweep `lambda ∈ [0, 1e-9, 1e-7, 1e-5, 1e-3, 0.1, 1, 10, 100]` over 5 seeds.
All other lambda parameters are held at **1e-7**.

| Ablation key | Swept parameter | Fixed parameters | Model | Conda env |
|---|---|---|---|---|
| `clf` | `classifier_lambda` | `discriminator_lambda=1e-7` | CellinaModel | `cellina` |
| `disc` | `discriminator_lambda` | `classifier_lambda=1e-7` | CellinaModel | `cellina` |
| `domain_clf` | `domain_classifier_lambda` | `clf=1e-7, disc=1e-7` | CellinaModel | `cellina` |
| `graph` | `link_prediction_weight` | `clf=1e-7, disc=1e-7` | CellinaGraph | `cellina_edge` |

Trained models are saved under `trained/` (relative to this directory).

## Example commands

Run a single ablation:
```bash
conda run -n cellina      python run_ablations.py --ablation clf
conda run -n cellina      python run_ablations.py --ablation disc
conda run -n cellina      python run_ablations.py --ablation domain_clf
conda run -n cellina_edge python run_ablations.py --ablation graph
```

Run all four in parallel via tmux:
```bash
bash submit_tmux.sh
tmux attach -t ablations      # watch live output
# Ctrl-b n / Ctrl-b p to switch windows; Ctrl-b d to detach
tail -f logs/*.log            # monitor from outside tmux
```

Run all four in parallel in the background:
```bash
nohup bash submit.sh > submit.log 2>&1 &
tail -f logs/*.log
```
