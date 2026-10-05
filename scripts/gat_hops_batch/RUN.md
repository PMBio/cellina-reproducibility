# Cellina-GAT hops / batch-size sensitivity test — run book

Note: `GAT_HOPS_BATCH_TEST.md` (the handoff spec referenced below and in `env.sh` / `summarize.py`) is a local working document and is not part of this repository.

Plumbing for `GAT_HOPS_BATCH_TEST.md` (arms A/B/C; arm D skipped). MERFISH only,
seed 0, one full leave-one-cell-type-out sweep per arm.

## Commit hashes

| What | Value |
|---|---|
| `cellina-reproducibility` | `159fe8d` (branch `main`) + the changes listed under "What was added" |
| `cellina` package | `ea2c2722d9d2dd15b9c03f7214db0412e550b975` (`release/v1.1.1`, version 1.1.1) |
| Python env | `/g/stegle/ddimitro/miniforge3/envs/cellina111/bin/python` |

The package commit comes from
`.../cellina111/lib/python3.10/site-packages/cellina-1.1.1.dist-info/direct_url.json`:
`{"url": "https://github.com/PMBio/cellina", "vcs_info": {"commit_id": "ea2c272...", "requested_revision": "release/v1.1.1"}}`.
`pyg-lib` is not installed; PyG falls back to the `torch_sparse` / pure-python
neighbour sampler. That changes speed, not the sampled node sets.

## What was added

- `scripts/configs/cellina_graph_config.py` — **defaults unchanged**. An appended
  block adds env-var hooks that fall back to the committed values:

  | Variable | Effect | Fallback |
  |---|---|---|
  | `GAT_N_LAYERS` | `MODEL_ARGS["n_layers"]` | 2 |
  | `GAT_NUM_NEIGHBORS` | `MODEL_ARGS["num_neighbors"]` (comma-separated ints) | key **not set** -> package default |
  | `GAT_BATCH_SIZE` | `TRAIN_ARGS["batch_size"]` | 512 |
  | `GAT_N_NEIGHBORS_GRAPH` | `N_NEIGHBORS_GRAPH` | 20 |
  | `GAT_N_NEIGHBORS_PER_SEED` | `N_NEIGHBORS_PER_SEED` | 20 |

  With none of them set, the module's `MODEL_ARGS` / `TRAIN_ARGS` / `PLAN_KWARGS` /
  `DO_COUNTERFACTUAL` / `N_NEIGHBORS_*` compare equal to `git show HEAD:...` (verified).
  When any override is set, one line is printed to stderr listing the resolved values.
- `scripts/gat_hops_batch/{env.sh,run_fold.sbatch,submit.sh,summarize.py,RUN.md}`
- `results/gat_hops_batch/` (per-arm metric + timing JSONs land here)

`scripts/train_loo.py` and `scripts/eval_loo.py` are **not** modified. Both import
`MODEL_ARGS` / `TRAIN_ARGS` / `N_NEIGHBORS_GRAPH` / `N_NEIGHBORS_PER_SEED` from
`configs.cellina_graph_config` at module import time, so environment variables set
in the job script take effect. `train_loo.py` builds the model as
`CellinaGCN(adata, **model_args)`, so `num_neighbors` is forwarded when present.
`DATA_ROOT` drives every path and is set to `$REPO/data`.

## Arms

Base config, identical in every arm: `n_layers` 3, `N_NEIGHBORS_GRAPH` 50,
`N_NEIGHBORS_PER_SEED` 50, seed 0, `convolution_type` `gat`, `n_latent` 64,
lr 1e-3, weight decay 1e-4, 100 epochs, patience 10. Run names use hyphens only,
because the aggregation splits filenames on `_`.

| Arm | Run name | `GAT_NUM_NEIGHBORS` | `GAT_BATCH_SIZE` | `GAT_N_LAYERS` | `GAT_N_NEIGHBORS_GRAPH` | `GAT_N_NEIGHBORS_PER_SEED` |
|---|---|---|---|---|---|---|
| A (May config) | `cellina-graph-h1-b256` | `-1,0,0` | 256 | 3 | 50 | 50 |
| B | `cellina-graph-h1-b512` | `-1,0,0` | 512 | 3 | 50 | 50 |
| C | `cellina-graph-h3-b256` | `-1,-1,-1` | 256 | 3 | 50 | 50 |

Arm D (`[50,0,0]`) is skipped: on a k=50 graph it samples the same node set as `[-1,0,0]`.

## Negative `num_neighbors` means "all neighbours"

In this package version `cellina/_cellina_gcn_model.py::_resolve_num_neighbors`
returns `[-20] * n_layers` when `num_neighbors` is `None`. PyG's `NeighborLoader`
treats **any** negative fan-out as "take all neighbours", so `-20` is not "20":
the current `main` default is all-neighbours at every hop, contrary to what the
comment in the package suggests.

Verified directly on a synthetic 200-node graph with out-degree 50
(batch_size 8, first batch, `torch_sparse` fallback sampler):

```
[-1]         n_nodes= 58  n_edges= 400
[-20]        n_nodes= 58  n_edges= 400     <- identical to [-1]
[20]         n_nodes= 57  n_edges= 160     <- actually caps at 20
[-1, 0, 0]   n_nodes= 58  n_edges= 400     <- one hop, all neighbours
[-1, -1, -1] n_nodes=158  n_edges=5400     <- three hops, all neighbours
```

`0` means "no new nodes at this hop", as expected. `num_neighbors` must have
length `n_layers` or the package emits a `UserWarning`; all three arms pass a
length-3 list, and constructing `CellinaGCN` with `num_neighbors=[-1,0,0]`,
`n_layers=3` produces no warning (`model._num_neighbors == [-1, 0, 0]`).

## Running it

```bash
cd /g/stegle/ddimitro/repos/cellina-reproducibility

# 1. dry run: writes jobs.txt (45 folds) and prints the sbatch command
DRY=1 bash scripts/gat_hops_batch/submit.sh

# 2. optional gate job: one fold, no array
sbatch --export=ALL,GHB_DIR=$PWD/scripts/gat_hops_batch,ARM=h1-b256,\
SID=C57BL6J-2.036,CT='astrocyte' scripts/gat_hops_batch/run_fold.sbatch

# 3. the full sweep, 15 concurrent folds
bash scripts/gat_hops_batch/submit.sh
MAXJOBS=8 bash scripts/gat_hops_batch/submit.sh     # different cap
ARMS='h1-b256 h3-b256' bash scripts/gat_hops_batch/submit.sh   # subset

# 4. aggregate (safe to run while jobs are still going; prints completeness)
python scripts/gat_hops_batch/summarize.py
```

3 arms x 3 slides (`C57BL6J-2.036/.039/.041`) x 5 holdout cell types
(`glutamatergic neuron`, `oligodendrocyte`, `astrocyte`, `GABAergic neuron`,
`endothelial cell`) = **45 folds**, each producing 2 metric JSONs (holdout
domains `Isocortex` and `Fiber_tracts`) -> 30 metric JSONs per arm.

`submit.sh` uses a single Slurm array (`--array=0-44%$MAXJOBS`) rather than 45
separate `sbatch --export` calls, because an array is the only way to get a
concurrency cap. Each task reads its `(ARM, SID, CT)` triple from line
`$SLURM_ARRAY_TASK_ID + 1` of `scripts/gat_hops_batch/jobs.txt` (tab separated,
since cell type names contain spaces). Passing `ARM`/`SID`/`CT` via `--export`
overrides that, which is what the single gate job does.

## Paths

| What | Where |
|---|---|
| Input slides | `$DATA_ROOT/datasets/MERFISH_mouse_brain/<sid>.h5ad` |
| Trained model | `$DATA_ROOT/data/ood/trained/<sid>/<ct>/cellina-graph-<arm>/model.pt` |
| Counterfactual h5ad | `$DATA_ROOT/datasets/<sid>/<ct>/cellina-graph-<arm>_counterfactual_x_<domain>.h5ad` |
| Metric JSON (source) | `$DATA_ROOT/datasets/merfish/correlations/<sid>_cellina-graph-<arm>-cf_<ct>_<domain>.json` |
| Metric JSON (collected) | `results/gat_hops_batch/<arm>/<same name>` |
| Timing JSON | `results/gat_hops_batch/<arm>/timing_<sid>_<ct>.json` |
| Job logs | `scripts/gat_hops_batch/logs/gat-hops_<jobid>.log` |
| `/usr/bin/time -v` dump | `scripts/gat_hops_batch/logs/time_<arm>_<sid>_<ct-with-dashes>.txt` |
| Summary | `results/gat_hops_batch/summary.md`, `summary.csv` |

## Re-entrancy

`run_fold.sbatch` checks `.../model.pt`:

- model **and** both counterfactual h5ads present -> `train_loo.py` is skipped
  entirely (`train_mode: "skipped"` in the timing JSON; no wall-clock recorded).
- model present, counterfactuals missing -> `train_loo.py --inference_only`
  (`train_mode: "inference_only"`; the recorded wall-clock is inference, not training).
- otherwise a full run (`train_mode: "full"`).

So a re-submitted array does not retrain, but the timing table only reflects the
run that actually trained. To get a clean cost comparison, run each arm once from
an empty `$DATA_ROOT/data/ood/trained/<sid>/<ct>/cellina-graph-<arm>/`.

## Cost instrumentation

`train_loo.py` is wrapped in `/usr/bin/time -v` (max RSS, its own wall clock) and
a background `nvidia-smi --query-gpu=memory.used -l 5` sampler writes to a temp
file; the maximum sampled value becomes `peak_gpu_mib`. The sampler is pinned to
`$CUDA_VISIBLE_DEVICES` with `nvidia-smi -i` when that works, and is killed (and
`wait`ed on) when training finishes, plus on an EXIT trap. A 5 s sampling period
will miss sub-5 s allocation spikes; treat `peak_gpu_mib` as a floor.

## Metrics reported

From each metric JSON: `pearson`, `direction_match_k` ("Signed precision"),
`edistance_pca_log` ("E-distance"), and `rmse_lfc = sqrt(mse_lfc)`. `summarize.py`
pools mean ± std over slide x cell type, per domain and averaged over both, and
prints the two reference rows from the spec (current `main` 2-layer MERFISH
average 0.83 ± 0.16 / 0.49 ± 0.16 / 8.07 ± 1.87 / 6.22 ± 4.87, and rebuttal RT6
0.85 ± 0.15 / 0.52 ± 0.14 / 8.69 ± 1.45 / 5.80 ± 4.50 — RT6 came from the May
pipeline and is **not directly comparable**).

## Aggregation

```
python scripts/gat_hops_batch/summarize.py   # per-arm table + cost -> results/gat_hops_batch/summary.{md,csv}
python scripts/gat_hops_batch/paired.py > results/gat_hops_batch/paired.md   # per-fold paired diffs -> per_fold.csv
```
Note: summarize.py overwrites summary.md; the "Paired comparison" and later sections were appended by hand afterwards.

## CRC follow-up (arm A only)

Submitted 2026-09-29 as array `62479143` (30 folds = 6 slides x 5 cell types, cap 15, 24h/128G):

```
DATASET=crc ARMS=h1-b256 bash scripts/gat_hops_batch/submit.sh
python scripts/gat_hops_batch/summarize.py --dataset crc   # -> results/gat_hops_batch/crc/summary.{md,csv}
```
Results land in `results/gat_hops_batch/crc/h1-b256/`; reference row is the `main` CRC GAT row from the spec (0.84 / 0.36 / 7.78 / 1.24).

## Literal `[-1]` check (arm h1lit-b256)

Submitted 2026-09-30 as array `62503822` (62503742, .036 only, was cancelled and replaced): arm A but `num_neighbors=[-1]` (length-1 list, exactly the May setting; the package warns about the length mismatch and passes it through), all 3 MERFISH slides, 15 folds run concurrently. Expected to match arm A fold for fold.

```
ARMS=h1lit-b256 MAXJOBS=15 bash scripts/gat_hops_batch/submit.sh
```

Timing twin: array `62503908`, arm `h1lit-b256-pyg` = h1lit-b256 run with `PY=.../envs/cellina112/bin/python` (cellina 1.1.2 editable + pyg-lib 0.4.0), slide .036 only, to measure the pyg-lib speed-up on identical folds.

## CRC `[20,0,0]` arm (h1k20-b256)

Submitted 2026-09-30 as array `62504920`: arm A but `num_neighbors=[20,0,0]` (hop 1 samples 20 of the k=50 kNN neighbours, a capped fan-out), all 30 CRC folds at once (cap 30), run with `PY=.../envs/cellina112/bin/python` (cellina 1.1.2 + pyg-lib 0.4.0, since capped fan-outs are sampler-bound without it). Results in `results/gat_hops_batch/crc/h1k20-b256/`; compare fold-for-fold with `crc/h1-b256`.

```
PY=/g/stegle/ddimitro/miniforge3/envs/cellina112/bin/python DATASET=crc ARMS=h1k20-b256 MAXJOBS=30 bash scripts/gat_hops_batch/submit.sh
```

## Re-run of the excluded fold with drop_last (2026-09-30)

`C57BL6J-2.041 / oligodendrocyte` has n_train = 38657 = 1 mod 256, so the last
training minibatch is a single seed cell and BatchNorm raises. `train_loo.py`
overwrites `datasplitter_kwargs`, so the flag is injected by
`scripts/gat_hops_batch/train_loo_drop_last.py` (wraps `CellinaGCN.train`, then
runs `train_loo.py` unchanged); `run_fold.sbatch` selects it when `GHB_DROP_LAST=1`.
Only the size-1 batch is dropped. Submitted as single jobs (env cellina111, as the
original arms) for the three arms in which the fold failed:

```
for arm in h1-b256 h3-b256 h1lit-b256; do
  sbatch --job-name=gat-hops-dl \
    --export=ALL,GHB_DIR=$PWD/scripts/gat_hops_batch,DATASET=merfish,ARM=$arm,SID=C57BL6J-2.041,CT=oligodendrocyte,GHB_DROP_LAST=1,PY=.../cellina111/bin/python \
    scripts/gat_hops_batch/run_fold.sbatch
done
```

All three re-runs completed (A train 26 min, C 74 min, A-lit 18 min); JSONs in `results/gat_hops_batch/<arm>/`,
`summary.md` / `paired.md` / `per_fold.csv` regenerated with 30/30 folds. The fold is the hardest in the sweep
(Isocortex Pearson ~0.47 in all arms) and moves arm A's pooled MERFISH numbers from 0.85/0.52 (28 folds) to
0.83/0.50 (30 folds), i.e. onto the `main` row, not RT6.

Array `62504920` (`h1k20-b256`, 30/30, no failures, 08:00-09:34) done. Fold-for-fold vs arm A: Pearson +0.018 +- 0.088,
precision +0.002 +- 0.111, E-distance -0.19 +- 1.07, RMSE_LFC -0.00 +- 0.23 (null); training 2.3x faster at the median,
peak GPU 4.1 -> 1.2 GB. Section appended to `results/gat_hops_batch/crc/summary.md`.
