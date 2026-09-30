# Cellina v1.1.2 LOO sweep: merfish

Folds: 30/30 metric JSONs (slide x cell type x holdout domain), seed 0, model `cellina112_0` (scripts/multi_seed/train_loo.py --cf_spatial_layer lognorm, eval scripts/eval_loo.py --use_cf).

| Arm | Pearson | Signed precision | E-distance | RMSE_LFC | NB deviance | n |
|---|---|---|---|---|---|---|
| Cellina 1.1.1 main (paper, edge cf from raw counts) | 0.83 ± 0.15 | 0.47 ± 0.18 | 7.96 ± 1.32 | 6.27 ± 5.01 | 0.04 ± 0.04 | 30 |
| Cellina 1.1.1 main (paper, node perturbation) | 0.82 ± 0.15 | 0.47 ± 0.17 | 9.07 ± 1.83 | 6.31 ± 4.95 | 0.04 ± 0.05 | 30 |
| Cellina 1.1.2 (cf from lognorm), pooled | 0.84 ± 0.16 | 0.50 ± 0.19 | 8.08 ± 1.26 | 6.07 ± 5.17 | 0.03 ± 0.03 | 30 |
|   Isocortex | 0.86 ± 0.18 | 0.60 ± 0.14 | 8.00 ± 1.18 | 4.85 ± 4.12 | 0.04 ± 0.03 | 15 |
|   Fiber_tracts | 0.82 ± 0.15 | 0.40 ± 0.17 | 8.15 ± 1.37 | 7.29 ± 5.93 | 0.03 ± 0.03 | 15 |

## Per slide (pooled over cell types and domains)

| slide | Pearson | Signed precision | E-distance | RMSE_LFC | NB deviance | n |
|---|---|---|---|---|---|---|
| C57BL6J-2.036 | 0.83 ± 0.20 | 0.49 ± 0.17 | 8.44 ± 1.25 | 5.88 ± 5.01 | 0.04 ± 0.04 | 10 |
| C57BL6J-2.039 | 0.85 ± 0.13 | 0.53 ± 0.16 | 7.49 ± 1.15 | 4.88 ± 4.66 | 0.03 ± 0.03 | 10 |
| C57BL6J-2.041 | 0.83 ± 0.17 | 0.48 ± 0.23 | 8.30 ± 1.29 | 7.47 ± 5.95 | 0.03 ± 0.03 | 10 |

## Cost

train+cf wall (s): median 133, max 167; max RSS (GiB): median 4.4, max 4.9; n=15 folds, GPUs: ['NVIDIA A100-PCIE-40GB', 'NVIDIA A40', 'NVIDIA GeForce RTX 3090', 'NVIDIA L40S']
