# Edge-perturbation counterfactuals from the `lognorm` layer (cellina 1.1.2) + LOO sweeps

## The bug and the fix

In cellina 1.1.1 the edge-perturbation counterfactual helpers (`make_counterfactual_adata`,
`get_counterfactual_expression`, `get_counterfactual_latents`, `precomputed=False`) aggregated the
neighbour spatial features from `adata.X`. In the LOO pipeline `adata.X` holds raw counts at that
point, while the training `spatial_x` was built from log1p(CP10K). The model therefore saw
counterfactual spatial features on a different scale from the one it was trained on.

cellina 1.1.2 (PMBio/cellina#41) adds a `layer` argument to those helpers so the caller can name
the representation to aggregate. With `layer=None` the behaviour is bit-identical to 1.1.1.

## What changed in this repo

* `scripts/multi_seed/train_loo.py`
  * `preprocess_spatial_features` keeps the normalized matrix as `adata.layers['lognorm']`
    before restoring raw counts to `adata.X`.
  * New `--cf_spatial_layer` (default `lognorm`): the `adata.layers` key passed as `layer=` to the
    cellina counterfactual helpers. `--cf_spatial_layer none` aggregates `adata.X` instead, i.e. the
    pre-1.1.2 behaviour. Only used for `--model_class cellina`.
  * New `--model_name_suffix`: appended to the output h5ad basenames only (not the model save dir),
    so `--inference_only` can write a second counterfactual arm from the same trained model.
  * MERFISH `step_size_px` synced to 1 to match `scripts/train_loo.py` (main).
* `scripts/multi_seed/{eval_loo,train_parallel,eval_parallel}.py`: `DATA_ROOT` defaults to `<repo>/data`; `eval_loo.py` `step_size_px` synced as above.
* `scripts/cellina112_loo/`: Slurm sweep (`env.sh`, `submit.sh`, `run_fold.sbatch`, job lists) and
  `summarize.py`. Training + counterfactuals use `scripts/multi_seed/train_loo.py`, evaluation uses the
  paper's `scripts/eval_loo.py --use_cf`.
* `results/cellina112_loo/{merfish,crc}/`: per-fold metric JSONs, timing JSONs, `per_fold.csv`,
  `summary.md` for seed 0 (model name `cellina112_0`). `.gitignore` gained a targeted exception for
  these JSON/CSV files.

## Requirements

* cellina >= 1.1.2 (PMBio/cellina#41 merged, or a `v1.1.2` tag) in the python pointed to by `PY`
  (`env.sh` default: `/g/stegle/ddimitro/miniforge3/envs/cellina112/bin/python`, overridable).
* `DATA_ROOT` (default `<repo>/data`) laid out as for the other LOO scripts.

## Running

```bash
DATASET=merfish bash scripts/cellina112_loo/submit.sh   # 3 slides x 5 cell types (x 2 holdout domains) = 30 folds
DATASET=crc     bash scripts/cellina112_loo/submit.sh   # 6 slides x 5 cell types = 30 folds
python scripts/cellina112_loo/summarize.py              # writes results/cellina112_loo/<dataset>/{summary.md,per_fold.csv}
```

## Results (top-50 DEGs, mean +- sd over 30 folds each, seed 0)

| Dataset | Arm | Pearson | Signed precision | E-distance | RMSE_LFC |
|---|---|---|---|---|---|
| CRC | Cellina 1.1.1 (paper, edge cf from raw counts) | 0.82 +- 0.17 | 0.40 +- 0.19 | 7.50 +- 1.14 | 1.29 +- 0.64 |
| CRC | Cellina 1.1.2 (edge cf from lognorm) | 0.87 +- 0.14 | 0.42 +- 0.19 | 7.94 +- 1.47 | 1.17 +- 0.67 |
| MERFISH | Cellina 1.1.1 (paper, edge cf from raw counts) | 0.83 +- 0.15 | 0.47 +- 0.18 | 7.96 +- 1.32 | 6.27 +- 5.01 |
| MERFISH | Cellina 1.1.2 (edge cf from lognorm) | 0.84 +- 0.16 | 0.50 +- 0.19 | 8.08 +- 1.26 | 6.07 +- 5.17 |

Paper rows: MERFISH `git show 7da94fb:results/loo_summary_merfish_DEG_50_v2.csv`, CRC
`results/terra_handoff/loo_summary_crc_DEG_50_v5.csv`, pooled over all slide x cell type x domain folds.
A replication of the paper pipeline in a pristine cellina 1.1.1 env reproduces the paper edge row per
fold (paired Pearson difference -0.002 +- 0.015), so the 1.1.2 change is due to the fix, not retraining.
`summary.md` for MERFISH also lists the paper's node-perturbation row for reference only:
node-perturbation results are out of scope for this PR and are handled separately.
