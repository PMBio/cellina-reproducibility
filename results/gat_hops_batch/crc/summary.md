# Cellina-GAT hops / batch-size sensitivity test

MERFISH leave-one-cell-type-out, top-50-DEG metrics, pooled mean +- std over
slide x cell type (3 slides x 5 cell types per domain). Seed 0.

## Metrics

| Arm | Config | Domain | Pearson | Signed precision | E-distance | RMSE_LFC |
|---|---|---|---|---|---|---|
| h1-b256 | A: num_neighbors [-1,0,0], batch 256 (May config) | CRC | 0.83 +- 0.15 (n=30) | 0.38 +- 0.21 (n=30) | 9.19 +- 1.92 (n=30) | 1.21 +- 0.71 (n=30) |
| h1-b256 | A: num_neighbors [-1,0,0], batch 256 (May config) | average | 0.83 +- 0.15 (n=30) | 0.38 +- 0.21 (n=30) | 9.19 +- 1.92 (n=30) | 1.21 +- 0.71 (n=30) |
| h1k20-b256 | A-k20: num_neighbors [20,0,0] (capped fan-out), batch 256, pyg-lib (cellina112) | CRC | 0.84 +- 0.13 (n=30) | 0.39 +- 0.20 (n=30) | 9.01 +- 1.63 (n=30) | 1.20 +- 0.63 (n=30) |
| h1k20-b256 | A-k20: num_neighbors [20,0,0] (capped fan-out), batch 256, pyg-lib (cellina112) | average | 0.84 +- 0.13 (n=30) | 0.39 +- 0.20 (n=30) | 9.01 +- 1.63 (n=30) | 1.20 +- 0.63 (n=30) |
| _main 2-layer (reference)_ | from GAT_HOPS_BATCH_TEST.md | average | 0.84 +- 0.13 | 0.36 +- 0.20 | 7.78 +- 1.41 | 1.24 +- 0.69 |

Reference rows are quoted from the spec for orientation. The `main` 2-layer row uses the same current pipeline; the RT6 row was produced under the May pipeline and is **not directly comparable**.

## Completeness

| Arm | metric JSONs | expected | timing JSONs |
|---|---|---|---|
| h1-b256 | 30 | 30 | 30 |
| h1k20-b256 | 30 | 30 | 30 |

## Cost

| Arm | folds | median train wall | peak GPU (MiB) | median peak GPU (MiB) | median max RSS (GiB) | GPU |
|---|---|---|---|---|---|---|

## Per slide (mean over 5 cell types)

| Slide | cells | Pearson | Signed precision | E-distance | RMSE_LFC | train wall (median) | peak RSS |
|---|---|---|---|---|---|---|---|
| crc_120 | 741k | 0.79 | 0.30 | 8.16 | 1.13 | 3.45 h | 60 GiB |
| crc_210 | 617k | 0.95 | 0.53 | 10.02 | 1.09 | 3.09 h | 26 GiB |
| crc_221 | 370k | 0.84 | 0.58 | 9.37 | 0.83 | 1.98 h | 18 GiB |
| crc_231 | 298k | 0.74 | 0.30 | 8.29 | 1.30 | 1.40 h | 15 GiB |
| crc_232 | 131k | 0.78 | 0.28 | 9.20 | 1.40 | 1.66 h | 7 GiB |
| crc_242 | 474k | 0.87 | 0.31 | 10.14 | 1.48 | 2.29 h | 28 GiB |

## Per cell type (mean over 6 slides)

| Cell type | Pearson | Signed precision | E-distance | RMSE_LFC |
|---|---|---|---|---|
| Endothelial | 0.90 | 0.55 | 8.93 | 0.96 |
| Epithelial | 0.71 | 0.22 | 11.09 | 2.34 |
| Fibroblast | 0.76 | 0.26 | 8.69 | 1.18 |
| Myeloid | 0.85 | 0.44 | 8.57 | 0.84 |
| T_cell | 0.91 | 0.46 | 8.70 | 0.71 |

Per-fold values: `results/gat_hops_batch/crc/per_fold.csv`.

## Verdict

Arm A (the May config: 3 layers, one hop over all k=50 neighbours, 50 donors, batch 256) on CRC gives
**0.83 +- 0.15 / 0.38 +- 0.21 / 9.19 +- 1.92 / 1.21 +- 0.71** against the current `main` 2-layer row
**0.84 +- 0.13 / 0.36 +- 0.20 / 7.78 +- 1.41 / 1.24 +- 0.69** (quoted from the spec; no per-slide `main`
GAT JSONs exist locally, so only the pooled row can be compared).

- Pearson, signed precision and RMSE_LFC are the same within a few hundredths, i.e. well inside the
  fold spread. The May config does **not** improve CRC.
- E-distance is worse by ~1.4 (9.19 vs 7.78), about one fold-std. That is the only metric that moves,
  and it moves the wrong way.
- Cost is high: 1.4-3.5 h training per fold (k=50 graph, 3 layers) and up to 60 GiB host RAM on crc_120,
  versus ~4 GB GPU. The 3-layer / k=50 / 50-donor config does not pay for itself on CRC.
- Together with MERFISH (where arm A reproduces RT6 and hops/batch are null), the conclusion is: the May
  config reproduces the rebuttal GAT numbers on MERFISH, is not better on CRC, and neither hops nor batch
  size matter. Keeping the `main` 2-layer config is justified.

No fold failed (no CRC training set is 1 mod 256).

## Provenance

Same pipeline, package and env as the MERFISH test (`results/gat_hops_batch/summary.md`, Provenance):
repro `159fe8d` + uncommitted GAT_* env hooks in `scripts/configs/cellina_graph_config.py`,
cellina 1.1.1 in `cellina111`. SLURM array `62479143`, 30 tasks, cap 15, 24 h / 128 G, 2026-09-29/30.
Launched with `DATASET=crc ARMS=h1-b256 bash scripts/gat_hops_batch/submit.sh`. Data:
`data/datasets/crc/raw_zenodo/crc_{120,210,221,231,232,242}.h5ad`, control REF -> holdout CRC.

## `[20,0,0]` arm (h1k20-b256, pyg-lib, 2026-09-30)

Arm A with the first-hop fan-out capped at 20 of the k=50 spatial neighbours (`num_neighbors=[20,0,0]`,
3 layers, batch 256, `N_NEIGHBORS_GRAPH` 50, `N_NEIGHBORS_PER_SEED` 50), all 30 folds at once (array
`62504920`, cap 30), env `cellina112` (cellina 1.1.2 + pyg-lib 0.4.0; logs show the pyg-lib sampler, no
torch_sparse deprecation warning). No fold failed. Note that with a capped fan-out pyg-lib draws a different
random neighbour subset than torch_sparse would for the same seed, so this arm is not bit-reproducible in
an env without pyg-lib; arm A (`-1` fan-out) is unaffected by that.

Pooled numbers: see the Metrics table above (`h1k20-b256` row: 0.84 / 0.39 / 9.01 / 1.20 vs A 0.83 / 0.38 / 9.19 / 1.21
and `main` 0.84 / 0.36 / 7.78 / 1.24).

### Paired, `[20,0,0]` minus A, same fold

| Metric | A mean | `[20,0,0]` mean | mean diff | sd diff | frac `[20,0,0]` better |
|---|---|---|---|---|---|
| Pearson | 0.826 | 0.844 | +0.018 | 0.088 | 0.47 |
| Signed precision | 0.385 | 0.387 | +0.002 | 0.111 | 0.53 |
| E-distance | 9.195 | 9.007 | -0.188 | 1.068 | 0.57 |
| RMSE_LFC | 1.206 | 1.204 | -0.002 | 0.230 | 0.53 |

### Per fold

| Fold | A Pearson | K Pearson | A prec | K prec | A E-dist | K E-dist | A RMSE_LFC | K RMSE_LFC |
|---|---|---|---|---|---|---|---|---|
| crc_120 / Endothelial | 0.984 | 0.972 | 0.48 | 0.36 | 6.15 | 7.02 | 0.64 | 0.76 |
| crc_120 / Epithelial | 0.594 | 0.952 | 0.10 | 0.34 | 13.88 | 10.82 | 2.91 | 2.39 |
| crc_120 / Fibroblast | 0.594 | 0.537 | 0.12 | 0.16 | 7.71 | 8.05 | 1.12 | 1.16 |
| crc_120 / Myeloid | 0.815 | 0.854 | 0.26 | 0.24 | 6.38 | 6.23 | 0.59 | 0.62 |
| crc_120 / T_cell | 0.939 | 0.935 | 0.52 | 0.46 | 6.67 | 6.22 | 0.39 | 0.47 |
| crc_210 / Endothelial | 0.992 | 0.994 | 0.66 | 0.60 | 10.21 | 9.65 | 1.12 | 0.92 |
| crc_210 / Epithelial | 0.880 | 0.870 | 0.48 | 0.66 | 9.73 | 8.03 | 1.86 | 1.87 |
| crc_210 / Fibroblast | 0.961 | 0.939 | 0.50 | 0.38 | 8.80 | 7.90 | 0.98 | 1.26 |
| crc_210 / Myeloid | 0.942 | 0.735 | 0.50 | 0.20 | 10.21 | 9.05 | 0.90 | 1.81 |
| crc_210 / T_cell | 0.970 | 0.964 | 0.52 | 0.60 | 11.17 | 11.91 | 0.59 | 0.57 |
| crc_221 / Endothelial | 0.683 | 0.706 | 0.90 | 0.90 | 8.63 | 7.74 | 0.39 | 0.40 |
| crc_221 / Epithelial | 0.914 | 0.904 | 0.28 | 0.30 | 10.38 | 11.79 | 1.60 | 1.55 |
| crc_221 / Fibroblast | 0.904 | 0.890 | 0.28 | 0.34 | 9.03 | 9.85 | 1.27 | 1.27 |
| crc_221 / Myeloid | 0.715 | 0.702 | 0.76 | 0.78 | 7.96 | 9.17 | 0.39 | 0.35 |
| crc_221 / T_cell | 0.972 | 0.957 | 0.68 | 0.64 | 10.82 | 8.56 | 0.50 | 0.70 |
| crc_231 / Endothelial | 0.965 | 0.961 | 0.58 | 0.50 | 10.14 | 9.49 | 1.21 | 1.23 |
| crc_231 / Epithelial | 0.564 | 0.683 | 0.20 | 0.36 | 8.21 | 8.72 | 2.17 | 2.18 |
| crc_231 / Fibroblast | 0.627 | 0.582 | 0.26 | 0.16 | 6.65 | 7.61 | 0.97 | 0.96 |
| crc_231 / Myeloid | 0.776 | 0.798 | 0.16 | 0.20 | 9.21 | 9.01 | 1.16 | 1.12 |
| crc_231 / T_cell | 0.771 | 0.761 | 0.32 | 0.36 | 7.25 | 7.63 | 0.99 | 0.98 |
| crc_232 / Endothelial | 0.899 | 0.860 | 0.44 | 0.28 | 9.73 | 9.42 | 0.88 | 0.99 |
| crc_232 / Epithelial | 0.596 | 0.725 | 0.14 | 0.04 | 10.14 | 10.39 | 2.20 | 2.08 |
| crc_232 / Fibroblast | 0.527 | 0.655 | 0.06 | 0.08 | 9.77 | 8.95 | 1.85 | 1.79 |
| crc_232 / Myeloid | 0.930 | 0.905 | 0.46 | 0.36 | 9.43 | 8.66 | 1.00 | 1.13 |
| crc_232 / T_cell | 0.943 | 0.948 | 0.32 | 0.30 | 6.91 | 8.50 | 1.09 | 1.06 |
| crc_242 / Endothelial | 0.885 | 0.906 | 0.26 | 0.42 | 8.71 | 9.88 | 1.50 | 1.14 |
| crc_242 / Epithelial | 0.729 | 0.825 | 0.10 | 0.20 | 14.23 | 13.95 | 3.27 | 2.96 |
| crc_242 / Fibroblast | 0.944 | 0.945 | 0.32 | 0.42 | 10.16 | 8.97 | 0.92 | 0.90 |
| crc_242 / Myeloid | 0.894 | 0.942 | 0.50 | 0.56 | 8.21 | 8.21 | 1.02 | 0.88 |
| crc_242 / T_cell | 0.879 | 0.912 | 0.38 | 0.40 | 9.38 | 8.83 | 0.71 | 0.64 |

### Cost per fold (training wall, peak GPU, max RSS)

| Fold | A train (GPU) | `[20,0,0]` train (GPU) | speed-up | A / K peak GPU MiB | A / K max RSS GiB |
|---|---|---|---|---|---|
| crc_120 / Endothelial | 222 min (RTX 3090) | 80 min (L40S) | 2.8x | 3876 / 1235 | 60.0 / 60.0 |
| crc_120 / Epithelial | 176 min (RTX 3090) | 54 min (L40S) | 3.2x | 4366 / 1267 | 60.0 / 60.0 |
| crc_120 / Fibroblast | 207 min (RTX 3090) | 72 min (L40S) | 2.9x | 4198 / 1257 | 60.0 / 60.0 |
| crc_120 / Myeloid | 233 min (RTX 3090) | 73 min (L40S) | 3.2x | 4056 / 1233 | 60.0 / 60.0 |
| crc_120 / T_cell | 203 min (RTX 3090) | 81 min (L40S) | 2.5x | 3876 / 1227 | 60.0 / 60.0 |
| crc_210 / Endothelial | 186 min (RTX 3090) | 97 min (RTX 3090) | 1.9x | 3856 / 1050 | 26.4 / 26.4 |
| crc_210 / Epithelial | 137 min (RTX 3090) | 78 min (RTX 3090) | 1.8x | 4226 / 1052 | 24.1 / 24.1 |
| crc_210 / Fibroblast | 191 min (RTX 3090) | 80 min (RTX 3090) | 2.4x | 4144 / 1056 | 24.9 / 25.0 |
| crc_210 / Myeloid | 201 min (RTX 3090) | 88 min (RTX 3090) | 2.3x | 3866 / 1052 | 25.6 / 25.7 |
| crc_210 / T_cell | 185 min (RTX 3090) | 69 min (L40S) | 2.7x | 4026 / 1249 | 26.4 / 26.5 |
| crc_221 / Endothelial | 119 min (RTX 3090) | 43 min (L40S) | 2.7x | 4046 / 1247 | 17.6 / 17.7 |
| crc_221 / Epithelial | 121 min (RTX 3090) | 69 min (A100) | 1.8x | 4224 / 1267 | 18.0 / 17.7 |
| crc_221 / Fibroblast | 113 min (L40S) | 73 min (A100) | 1.5x | 4283 / 1243 | 16.4 / 16.5 |
| crc_221 / Myeloid | 129 min (L40S) | 68 min (A100) | 1.9x | 4059 / 1243 | 17.5 / 17.6 |
| crc_221 / T_cell | 119 min (L40S) | 69 min (A100) | 1.7x | 4235 / 1241 | 17.5 / 17.5 |
| crc_231 / Endothelial | 131 min (L40S) | 55 min (A100) | 2.4x | 4213 / 1241 | 14.8 / 14.8 |
| crc_231 / Epithelial | 97 min (L40S) | 42 min (A100) | 2.3x | 4243 / 1263 | 14.4 / 14.3 |
| crc_231 / Fibroblast | 80 min (L40S) | 40 min (A100) | 2.0x | 4163 / 1241 | 13.7 / 13.8 |
| crc_231 / Myeloid | 84 min (L40S) | 52 min (A100) | 1.6x | 4233 / 1241 | 14.7 / 14.7 |
| crc_231 / T_cell | 80 min (L40S) | 56 min (A100) | 1.4x | 4243 / 1245 | 14.6 / 14.6 |
| crc_232 / Endothelial | 100 min (RTX 3090) | 46 min (RTX 3090) | 2.2x | 4052 / 1090 | 6.9 / 6.9 |
| crc_232 / Epithelial | 109 min (RTX 3090) | 45 min (RTX 3090) | 2.4x | 4206 / 1118 | 5.8 / 5.4 |
| crc_232 / Fibroblast | 100 min (RTX 3090) | 46 min (RTX 3090) | 2.2x | 4066 / 1090 | 6.1 / 6.1 |
| crc_232 / Myeloid | 82 min (RTX 3090) | 32 min (RTX 3090) | 2.6x | 3926 / 1090 | 6.9 / 6.9 |
| crc_232 / T_cell | 113 min (RTX 3090) | 45 min (RTX 3090) | 2.5x | 4006 / 1090 | 7.0 / 7.1 |
| crc_242 / Endothelial | 129 min (RTX 3090) | 67 min (RTX 3090) | 1.9x | 3758 / 1032 | 27.4 / 27.5 |
| crc_242 / Epithelial | 137 min (RTX 3090) | 60 min (L40S) | 2.3x | 3906 / 1247 | 28.3 / 28.4 |
| crc_242 / Fibroblast | 136 min (RTX 3090) | 50 min (L40S) | 2.7x | 3786 / 1251 | 26.2 / 26.3 |
| crc_242 / Myeloid | 149 min (RTX 3090) | 47 min (L40S) | 3.2x | 3942 / 1253 | 27.3 / 27.3 |
| crc_242 / T_cell | 143 min (RTX 3090) | 48 min (L40S) | 3.0x | 3946 / 1227 | 27.3 / 27.4 |
| **median** | | | **2.3x** | 4058 / 1241 | 21.1 / 20.9 |

Speed-up range 1.4x-3.2x (arm A ran on 3090/L40S, this arm on 3090/L40S/A100, so part of the
spread is GPU class; the same-GPU-class folds still show >2x). Peak GPU memory drops ~3-4x because each
batch carries at most 20 neighbours per seed instead of all 50.


### Verdict

- **Metrics:** capping the fan-out at 20 changes nothing. Paired vs A: Pearson +0.018 +- 0.088, signed
  precision +0.002 +- 0.111, E-distance -0.19 +- 1.07, RMSE_LFC -0.00 +- 0.23; the sign is a coin flip on
  every metric and the paired sd is the fold-to-fold noise of a single seed (one fold, crc_120/Epithelial,
  swings from 0.59 to 0.95 Pearson by itself). Pooled it sits on top of A and on top of `main` for Pearson
  and precision, and like A it is worse than `main` on E-distance (9.0 vs 7.8).
- **Cost:** training is 2.3x faster at the median (1.4x-3.2x per fold) and needs ~3.3x less GPU memory
  (median 4.1 -> 1.2 GB); host RAM is unchanged (dominated by the dataset, up to 27 GB on crc_242).
- So on CRC the k=50 graph with a 20-neighbour sample is the cheapest way to run this config with no
  metric cost, but it still does not beat the `main` 2-layer configuration on E-distance, and the
  recommendation from the arm A follow-up stands: keep `main`.
