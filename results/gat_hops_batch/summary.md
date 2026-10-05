# Cellina-GAT hops / batch-size sensitivity test

MERFISH leave-one-cell-type-out, top-50-DEG metrics, pooled mean +- std over
slide x cell type (3 slides x 5 cell types per domain). Seed 0.

## Metrics

| Arm | Config | Domain | Pearson | Signed precision | E-distance | RMSE_LFC |
|---|---|---|---|---|---|---|
| h1-b256 | A: num_neighbors [-1,0,0], batch 256 (May config) | Isocortex | 0.86 +- 0.18 (n=15) | 0.63 +- 0.15 (n=15) | 8.57 +- 0.92 (n=15) | 4.62 +- 4.09 (n=15) |
| h1-b256 | A: num_neighbors [-1,0,0], batch 256 (May config) | Fiber_tracts | 0.80 +- 0.15 (n=15) | 0.38 +- 0.19 (n=15) | 8.24 +- 1.86 (n=15) | 7.47 +- 5.98 (n=15) |
| h1-b256 | A: num_neighbors [-1,0,0], batch 256 (May config) | average | 0.83 +- 0.16 (n=30) | 0.50 +- 0.21 (n=30) | 8.41 +- 1.45 (n=30) | 6.04 +- 5.24 (n=30) |
| h1-b512 | B: num_neighbors [-1,0,0], batch 512 | Isocortex | 0.86 +- 0.18 (n=15) | 0.63 +- 0.14 (n=15) | 8.10 +- 1.11 (n=15) | 4.61 +- 4.00 (n=15) |
| h1-b512 | B: num_neighbors [-1,0,0], batch 512 | Fiber_tracts | 0.80 +- 0.15 (n=15) | 0.38 +- 0.17 (n=15) | 8.16 +- 1.61 (n=15) | 7.44 +- 6.01 (n=15) |
| h1-b512 | B: num_neighbors [-1,0,0], batch 512 | average | 0.83 +- 0.16 (n=30) | 0.51 +- 0.20 (n=30) | 8.13 +- 1.36 (n=30) | 6.03 +- 5.22 (n=30) |
| h3-b256 | C: num_neighbors [-1,-1,-1], batch 256 | Isocortex | 0.87 +- 0.17 (n=15) | 0.64 +- 0.13 (n=15) | 8.42 +- 0.91 (n=15) | 4.57 +- 4.04 (n=15) |
| h3-b256 | C: num_neighbors [-1,-1,-1], batch 256 | Fiber_tracts | 0.80 +- 0.15 (n=15) | 0.38 +- 0.16 (n=15) | 8.02 +- 1.44 (n=15) | 7.42 +- 5.94 (n=15) |
| h3-b256 | C: num_neighbors [-1,-1,-1], batch 256 | average | 0.83 +- 0.16 (n=30) | 0.51 +- 0.20 (n=30) | 8.22 +- 1.20 (n=30) | 6.00 +- 5.20 (n=30) |
| h1lit-b256 | A-lit: num_neighbors [-1] (literal May), batch 256 | Isocortex | 0.86 +- 0.18 (n=15) | 0.63 +- 0.16 (n=15) | 8.62 +- 1.08 (n=15) | 4.56 +- 4.08 (n=15) |
| h1lit-b256 | A-lit: num_neighbors [-1] (literal May), batch 256 | Fiber_tracts | 0.80 +- 0.15 (n=15) | 0.37 +- 0.18 (n=15) | 8.32 +- 1.81 (n=15) | 7.47 +- 5.99 (n=15) |
| h1lit-b256 | A-lit: num_neighbors [-1] (literal May), batch 256 | average | 0.83 +- 0.17 (n=30) | 0.50 +- 0.22 (n=30) | 8.47 +- 1.47 (n=30) | 6.01 +- 5.25 (n=30) |
| h1lit-b256-pyg | A-lit + pyg-lib (env cellina112), slide .036 only, timing check | Isocortex | 0.87 +- 0.19 (n=5) | 0.64 +- 0.12 (n=5) | 8.89 +- 0.96 (n=5) | 4.08 +- 3.42 (n=5) |
| h1lit-b256-pyg | A-lit + pyg-lib (env cellina112), slide .036 only, timing check | Fiber_tracts | 0.78 +- 0.21 (n=5) | 0.36 +- 0.18 (n=5) | 8.20 +- 2.04 (n=5) | 7.62 +- 6.24 (n=5) |
| h1lit-b256-pyg | A-lit + pyg-lib (env cellina112), slide .036 only, timing check | average | 0.82 +- 0.20 (n=10) | 0.50 +- 0.21 (n=10) | 8.55 +- 1.55 (n=10) | 5.85 +- 5.10 (n=10) |
| _main 2-layer (reference)_ | from GAT_HOPS_BATCH_TEST.md | average | 0.83 +- 0.16 | 0.49 +- 0.16 | 8.07 +- 1.87 | 6.22 +- 4.87 |
| _rebuttal RT6 (not directly comparable)_ | from GAT_HOPS_BATCH_TEST.md | average | 0.85 +- 0.15 | 0.52 +- 0.14 | 8.69 +- 1.45 | 5.80 +- 4.50 |

Reference rows are quoted from the spec for orientation. The `main` 2-layer row uses the same current pipeline; the RT6 row was produced under the May pipeline and is **not directly comparable**.

## Completeness

| Arm | metric JSONs | expected | timing JSONs |
|---|---|---|---|
| h1-b256 | 30 | 30 | 15 |
| h1-b512 | 30 | 30 | 15 |
| h3-b256 | 30 | 30 | 15 |
| h1lit-b256 | 30 | 30 | 15 |
| h1lit-b256-pyg | 10 | 30 | 5  <- incomplete |

## Cost

| Arm | folds | median train wall | peak GPU (MiB) | median peak GPU (MiB) | median max RSS (GiB) | GPU |
|---|---|---|---|---|---|---|
| h1-b256 | 15 | 0h23m16s | 3866 | 3828 | 2.86 | NVIDIA GeForce RTX 3090 |
| h1-b512 | 15 | 0h27m41s | 6824 | 6730 | 2.79 | NVIDIA GeForce RTX 3090 |
| h3-b256 | 15 | 1h13m53s | 16578 | 15546 | 3.26 | NVIDIA GeForce RTX 3090, NVIDIA L40S |
| h1lit-b256 | 15 | 0h22m57s | 4063 | 3864 | 2.89 | NVIDIA A100-PCIE-40GB, NVIDIA GeForce RTX 3090, NVIDIA L40S |
| h1lit-b256-pyg | 5 | 0h21m54s | 4017 | 3975 | 2.9 | NVIDIA A100-PCIE-40GB, NVIDIA L40S |

## Paired comparison (per fold, same slide x cell type x holdout domain in both arms)

Written from `results/gat_hops_batch/per_fold.csv` by `scripts/gat_hops_batch/paired.py`.
Differences are arm minus A on all 30 (fold x domain) pairs (the `.041/oligodendrocyte` fold was
re-run with `drop_last`, see below; the 28-pair version gave the same picture). "sd diff" is
the spread of the paired difference; "frac > 0" is the fraction of pairs where the arm beats A.

| Comparison | n (fold x domain) | Metric | mean diff | sd diff | frac > 0 |
|---|---|---|---|---|---|
| B - A (batch 512 vs 256) | 30 | Pearson | -0.001 | 0.014 | 0.40 |
| B - A (batch 512 vs 256) | 30 | Signed precision | +0.002 | 0.068 | 0.40 |
| B - A (batch 512 vs 256) | 30 | E-distance | -0.275 | 0.920 | 0.40 |
| B - A (batch 512 vs 256) | 30 | RMSE_LFC | -0.014 | 0.192 | 0.47 |
| C - A (3 hops vs 1) | 30 | Pearson | +0.004 | 0.015 | 0.50 |
| C - A (3 hops vs 1) | 30 | Signed precision | +0.005 | 0.059 | 0.53 |
| C - A (3 hops vs 1) | 30 | E-distance | -0.186 | 0.604 | 0.47 |
| C - A (3 hops vs 1) | 30 | RMSE_LFC | -0.043 | 0.186 | 0.47 |

For scale, the pooled between-fold std of Pearson is ~0.15 and of signed precision ~0.20; every
paired mean difference above is well under a tenth of that.

## Verdict

- **Batch size (A vs B, 256 vs 512):** no effect. Paired Pearson -0.001 +- 0.014, signed precision
  +0.002 +- 0.068, E-distance -0.28 +- 0.92, RMSE_LFC -0.01 +- 0.19. Sign is a coin flip on every metric.
- **Hops (A vs C, 1 vs 3):** no effect. Paired Pearson +0.004 +- 0.015, signed precision +0.005 +- 0.059,
  E-distance -0.19 +- 0.60, RMSE_LFC -0.04 +- 0.19. Three hops costs ~3x wall-clock (23 -> 72 min/fold)
  and ~4x GPU memory (3.9 -> 16 GB) for nothing.
- **Revised with the re-run fold (30/30):** neither factor explains anything, and the earlier reading that
  arm A 'lands on the RT6 numbers' was an artefact of the missing fold. `.041/oligodendrocyte` is the
  hardest fold in the sweep (Isocortex Pearson ~0.47 in every arm, including B which always had it), and
  including it moves A to 0.83 / 0.50 / 8.41 / 6.04 -- the same as the `main` 2-layer row
  (0.83 / 0.49 / 8.07 / 6.22) on Pearson and precision, not RT6 (0.85 / 0.52). So under the current pipeline
  the May config (3 layers, k=50, 50 donors, 1 hop, batch 256) and the `main` config give the same MERFISH
  numbers; the RT6 gap is a pipeline/eval difference (A3 items), not a config difference.
- Seed spread was not measured (seed 0 only, per spec); given that the paired differences are an order of
  magnitude below the fold spread, extra seeds for A/B were not run.
- CRC: arm A was run anyway (`results/gat_hops_batch/crc/summary.md`); same conclusion.

## Re-run fold (`C57BL6J-2.041 / oligodendrocyte`, 2026-09-30)

This fold first failed in every batch-256 arm (A, C, A-lit): n_train = 38657 = 1 mod 256, so the last
training batch has one cell and BatchNorm raises
`ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 128])`.
Arm B (batch 512) ran it fine (38657 mod 512 = 257). It was re-run with `drop_last=True` on the training
loader, injected by `scripts/gat_hops_batch/train_loo_drop_last.py` (wraps `CellinaGCN.train` and then runs
`scripts/train_loo.py` unchanged; selected by `GHB_DROP_LAST=1` in `run_fold.sbatch`), env `cellina111`,
jobs `62505032` (A), `62505033` (C), `62505034` (A-lit). Only the trailing size-1 batch is dropped
(1 of 151 batches per epoch). The fold is the hardest in the sweep and is the reason the 28-fold pooled
numbers were ~0.02 higher than the 30-fold ones:

| Arm | Isocortex Pearson / precision / E-dist / RMSE_LFC | Fiber_tracts Pearson / precision / E-dist / RMSE_LFC |
|---|---|---|
| A `h1-b256` (drop_last) | 0.467 / 0.20 / 9.65 / 15.5 | 0.735 / 0.38 / 7.55 / 4.1 |
| B `h1-b512` (original run) | 0.471 / 0.22 / 7.64 / 15.4 | 0.750 / 0.38 / 8.13 / 4.2 |
| C `h3-b256` (drop_last) | 0.484 / 0.24 / 9.73 / 15.4 | 0.725 / 0.38 / 8.14 / 4.3 |
| A-lit `h1lit-b256` (drop_last) | 0.465 / 0.18 / 9.04 / 15.3 | 0.716 / 0.36 / 8.15 / 4.5 |

`main` runs batch 512 and is not exposed to the size-1 batch.

## Provenance

- `cellina-reproducibility` `main` `159fe8d`, plus uncommitted env-var hooks in
  `scripts/configs/cellina_graph_config.py` (`GAT_N_LAYERS`, `GAT_NUM_NEIGHBORS`, `GAT_BATCH_SIZE`,
  `GAT_N_NEIGHBORS_GRAPH`, `GAT_N_NEIGHBORS_PER_SEED`; defaults unchanged when unset) and the
  `scripts/gat_hops_batch/` launcher. `scripts/train_loo.py` / `scripts/eval_loo.py` untouched.
- `cellina` 1.1.1 (`release/v1.1.1`, merge `c77b614`) in conda env `cellina111`, torch cu121,
  PyG NeighborLoader without pyg-lib.
- Base config for all arms: `n_layers` 3, `N_NEIGHBORS_GRAPH` 50, `N_NEIGHBORS_PER_SEED` 50,
  `n_latent` 64, lr 1e-3, wd 1e-4, 100 epochs, patience 10, `gat`, `induced`, seed 0.
  Arm A `num_neighbors=[-1,0,0]` batch 256; B `[-1,0,0]` batch 512; C `[-1,-1,-1]` batch 256
  (`scripts/gat_hops_batch/env.sh`, `arm_env`).
- SLURM array `62467976` (45 tasks, cap 15, one GPU each, RTX 3090 nodes via
  `--constraint="gpu=A100|gpu=H100|gpu=L40s|gpu=A40|gpu=3090"`), 2026-09-29; the
  `.041/oligodendrocyte` fold from the drop_last re-runs `62505032`/`62505033`/`62505034`, 2026-09-30.
- Data: `data/datasets/MERFISH_mouse_brain/` (cellxgene `93c3bb97-ea05-4ee0-a760-a1508cd04612`,
  split by `brain_section_label` into C57BL6J-2.036 / .039 / .041).

## `[-1]` literal check and pyg-lib timing (2026-09-30)

Arm `h1lit-b256` = arm A with `num_neighbors=[-1]` written literally (length-1 list, exactly the May
setting; the package warns about the length mismatch and passes it through), all 3 slides, array
`62503822`. `.041/oligodendrocyte` failed as in arm A (BatchNorm size-1 batch) and was re-run with drop_last (job
`62505034`). Fold-for-fold vs arm A (28 pairs at the time of writing; the re-run fold agrees, see above):

| lit `[-1]` − A `[-1,0,0]` (n=28) | mean diff | sd | max abs |
|---|---|---|---|
| Pearson | +0.0006 | 0.0055 | 0.018 |
| Signed precision | +0.0007 | 0.026 | 0.10 |
| E-distance | +0.07 | 0.39 | 1.44 |
| RMSE_LFC | −0.03 | 0.15 | 0.55 |

Same node sets, same numbers up to GPU non-determinism (scatter ops): `[-1,0,0]` was a faithful stand-in
and arm A is the May config. Pooled lit: see the Metrics table (row `h1lit-b256`).

Arm `h1lit-b256-pyg` = the same config run in env `cellina112` (cellina 1.1.2 editable + pyg-lib 0.4.0),
slide `.036` only, array `62503908`; logs confirm the pyg-lib sampler was used (no deprecation warning).
Metrics fold-for-fold vs lit: Pearson +0.005 ± 0.017, signed precision +0.02 ± 0.04, i.e. noise.

| `.036` fold | A, no pyg-lib (3090) | lit, no pyg-lib (3090) | lit + pyg-lib (A100 / L40S) |
|---|---|---|---|
| GABAergic neuron | 26.4 min | 26.2 min | 27.6 min |
| astrocyte | 30.8 min | 30.7 min | 21.9 min |
| endothelial cell | 25.8 min | 24.9 min | 18.2 min |
| glutamatergic neuron | 24.8 min | 25.7 min | 19.0 min (L40S) |
| oligodendrocyte | 23.3 min | 23.0 min | 24.6 min |
| median | 25.8 | 25.7 | 21.9 |

pyg-lib gives ~15 % on the median (25.7 -> 21.9 min) with all-neighbour fanouts, consistent with the
earlier micro-benchmark (~1.3x on the sampler alone; training is not sampler-bound with `-1` fanouts).
Caveat: the pyg twin landed on A100/L40S nodes rather than 3090s, so part of that is GPU, not pyg-lib.
The large pyg-lib win (15-17x sampler speed) applies only to capped fanouts, which none of these arms use.
