# Cellina-GAT 1.1.1 (directed CF edges), crc_232 LOO, 3 seeds

Follow-up to `../gat_hops_232`: the grid's best variant (kk_k20_h1 = `num_neighbors=[20,20]`, 1 head,
donor -> seed counterfactual edges) re-run with the released fix in cellina `release/v1.1.1`
(tag `v1.1.1`, commit 3156310) for seeds 0/1/2 x Epithelial/Fibroblast/Myeloid.

## Apples-to-apples with cellina-reproducibility@main
`run_seed.py` imports `train_model` / `run_inference` from `scripts/multi_seed/train_loo.py` and the metric
block of `scripts/multi_seed/eval_loo.py` from a detached worktree of `origin/main`
(`/data/ddimitrov/repos/cellina-reproducibility-main`, 99624df). Seed handling as in the multi-seed pipeline:
`set_seed(seed)`, `split_indices(seed=seed)`, counterfactual `seed=seed`. Deviations:
1. `condition_on_intrinsic` dropped from `MODEL_ARGS` (removed in cellina >= 1.1; pipeline config still has it -> crash).
2. `num_neighbors=[20,20]` explicit (pipeline default `None` -> `[-1,-1]`, i.e. no cap on symmetrized nodes with > 20 neighbours).
3. metrics computed in-process from the saved counterfactual h5ad (`crc_232/<ct>/cellina-graph_<seed>_counterfactual_x_CRC.h5ad`).

## Files
- `run_seed.py --seed S --ct CT --gpu G` (asserts cellina 1.1.1 @ release/v1.1.1), `launch.sh` (9 runs, 2 workers/GPU), `aggregate.py`
- `results/<ct>_seed<S>.json`, `results/models/`, `logs/`, `summary.csv`

## Result (3 seeds, batch 256; 2026-09-05)
| | Epi | Fib | Mye | avg |
|---|---|---|---|---|
| Pearson, mean ± sd (n=3) | 0.640 ± 0.029 | 0.753 ± 0.029 | 0.897 ± 0.009 | 0.764 ± 0.015 |
| PAPER v4 (hop-fixed, bidir CF) | 0.593 | 0.709 | 0.919 | 0.740 |
| old GAT Apr-23 (cellina_graph 0.0.3) | 0.606 | 0.844 | 0.932 | 0.794 |

Precision avg 0.211 ± 0.027 (paper 0.167, old GAT 0.200). Directed CF beats the paper row on Epi/Fib for every
seed, is ~0.02 lower on Mye; does not recover the Apr-23 Fib 0.844. Grid kk_k20_h1 Fib precision 0.22 did not
replicate (0.12 / 0.10 / 0.00). Seed 3 `_bs512` (batch 512): Pearson Epi 0.601 / Fib 0.794 / Mye 0.893 = avg 0.763; precision 0.18 / 0.16 / 0.46 = 0.267. Pearson inside the bs256 seed range (neutral), precision avg slightly higher (n=1 seed).

Perf note: runs are CPU-bound (GPU util ~10-15%, PyG sampling in main process). `OMP_NUM_THREADS=16` halves
threads and load without changing epoch pace; does not raise GPU util. Loader workers measured net-slower earlier.
