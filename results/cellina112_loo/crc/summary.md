# Cellina v1.1.2 LOO sweep: crc

Folds: 30/30 metric JSONs (slide x cell type x holdout domain), seed 0, model `cellina112_0` (scripts/multi_seed/train_loo.py --cf_spatial_layer lognorm, eval scripts/eval_loo.py --use_cf).

| Arm | Pearson | Signed precision | E-distance | RMSE_LFC | NB deviance | n |
|---|---|---|---|---|---|---|
| Cellina 1.1.1 main (paper, edge cf from raw counts) | 0.82 ± 0.17 | 0.40 ± 0.19 | 7.50 ± 1.14 | 1.29 ± 0.64 | 0.03 ± 0.04 | 30 |
| Cellina 1.1.2 (cf from lognorm), pooled | 0.87 ± 0.14 | 0.42 ± 0.19 | 7.94 ± 1.47 | 1.17 ± 0.67 | 0.03 ± 0.03 | 30 |

## Per slide (pooled over cell types and domains)

| slide | Pearson | Signed precision | E-distance | RMSE_LFC | NB deviance | n |
|---|---|---|---|---|---|---|
| crc_120 | 0.96 ± 0.02 | 0.44 ± 0.11 | 6.80 ± 0.94 | 0.72 ± 0.51 | 0.02 ± 0.03 | 5 |
| crc_210 | 0.93 ± 0.08 | 0.49 ± 0.21 | 8.35 ± 1.39 | 1.23 ± 0.48 | 0.03 ± 0.03 | 5 |
| crc_221 | 0.83 ± 0.13 | 0.56 ± 0.29 | 8.14 ± 1.59 | 1.02 ± 0.82 | 0.07 ± 0.06 | 5 |
| crc_231 | 0.73 ± 0.24 | 0.33 ± 0.17 | 7.55 ± 1.44 | 1.26 ± 0.34 | 0.00 ± 0.01 | 5 |
| crc_232 | 0.87 ± 0.12 | 0.30 ± 0.16 | 9.12 ± 1.10 | 1.22 ± 0.67 | 0.02 ± 0.01 | 5 |
| crc_242 | 0.87 ± 0.06 | 0.39 ± 0.14 | 7.69 ± 1.77 | 1.55 ± 1.04 | 0.03 ± 0.02 | 5 |

## Cost

train+cf wall (s): median 1934, max 8459; max RSS (GiB): median 45.9, max 85.9; n=30 folds, GPUs: ['NVIDIA A100-PCIE-40GB', 'NVIDIA A40', 'NVIDIA GeForce RTX 3090', 'NVIDIA H100 PCIe', 'NVIDIA L40S']
