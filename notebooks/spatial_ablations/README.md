# Spatial Ablations

LOO (leave-one-out) ablation study comparing spatial loss strategies and their effect on counterfactual prediction quality.

## Files

| File | Description |
|---|---|
| `ablation_loo.py` | Trains CellinaGraph across all combinations of holdout cell type × spatial loss type (`supcon`, `domain_clf`) × `link_prediction_weight`. Appends results to `results/ablation_loo.csv` and writes `results/ablation_loo.pdf`. |
| `ablation_loo_base.py` | Same sweep but for CellinaBase (no GCN), varying `domain_classifier_lambda`. |
| `spatial_ablation_utils.py` | Shared constants, `evaluate_model()`, and `generate_pdf()` used by both training scripts. |
| `plot_perturbation_loo.ipynb` | Loads `results/ablation_loo.csv` and produces publication-quality figures (bar plots with SEM) across metrics. |

## Aim

Determine how much the spatial loss type and its weight affect counterfactual prediction on held-out cell types and targets (CRC / TVA domains).

## Outputs

- `results/ablation_loo.csv` — one row per (holdout cell type × spatial loss type × λ × cell type × target)
- `results/ablation_loo.pdf` / `results/ablation_loo_base.pdf` — quick diagnostic PDF per run
- `results/fig_cf_summary.pdf`, `results/fig_cf_by_target.pdf`, `results/fig_run_metrics.pdf` — final figures from the notebook
