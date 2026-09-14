# TERRA on the LOO benchmark

Three scripts (spec: `PIPELINE_SPEC.md`), all run from the repo root:

- `common.py` — benchmark preprocessing (`train_loo.preprocess_*` + `split_indices`) plus a
  row-identical TERRA-harmonised all-genes twin, tokenisation cache, path layout. Not a CLI.
- `finetuning.py` — supervised block-11 fine-tune on the domain label → `work_dir/ft_bundle/`.
- `inference.py` — embed → count decoder (`terra.training.decode`, gene list = the 2000 benchmark
  HVGs in order) → neighbour-token perturbation → writes the four h5ads `eval_loo.py` reads → runs
  `eval_loo.py`. `--variant ft` uses `work_dir/ft_bundle`, `--variant frozen` the pretrained bundle.
- `eval_terra.py` — TERRA-native gene-embedding-shift scoring (no decoder), reads inference's caches.

Env: `/data/ddimitrov/software/miniforge3/envs/terra/bin/python` (has terra + everything `eval_loo.py`
imports; `cellina` is absent, so `preprocess_spatial_features` prints a warning and skips the spatial
graph — `eval_loo.py` never uses it for scoring). `export DATA_ROOT=/data/ddimitrov/data`.
GPU: `CUDA_VISIBLE_DEVICES=0` only — `embed_dataset` hardcodes `cuda:0`; GPU 1 is someone else's.

One (slide, cell type):

```bash
export DATA_ROOT=/data/ddimitrov/data CUDA_VISIBLE_DEVICES=0
PY=/data/ddimitrov/software/miniforge3/envs/terra/bin/python
A=$DATA_ROOT/datasets/crc/raw_zenodo/crc_232.h5ad
$PY scripts/terra/finetuning.py --dataset_name crc --adata_path $A --holdout_celltype Fibroblast
$PY scripts/terra/inference.py  --dataset_name crc --adata_path $A --holdout_celltype Fibroblast --variant ft
$PY scripts/terra/inference.py  --dataset_name crc --adata_path $A --holdout_celltype Fibroblast --variant frozen
$PY scripts/terra/eval_terra.py --dataset_name crc --adata_path $A --holdout_celltype Fibroblast --variant ft
```

All three CRC slides:

```bash
for s in crc_232 crc_231 crc_221; do
  A=$DATA_ROOT/datasets/crc/raw_zenodo/$s.h5ad
  $PY scripts/terra/finetuning.py --dataset_name crc --adata_path $A --holdout_celltype Fibroblast &&
  $PY scripts/terra/inference.py  --dataset_name crc --adata_path $A --holdout_celltype Fibroblast --variant ft
done
```

Outputs land in `$DATA_ROOT/datasets/crc/{sid}/Fibroblast/` — `terra_recon_x.h5ad`,
`terra_counterfactual_x_CRC.h5ad` and the `terra-null` pair (`terra-frozen*` for the frozen arm),
exactly the files `eval_loo.py --use_cf` loads. Caches, decoder checkpoint/metrics, `logfc_CRC.csv`
and the `ctrl_tok_CRC`/`pert_tok_CRC` tokenised sets sit under `.../Fibroblast/terra/`. Scores go to
`$DATA_ROOT/datasets/crc/correlations/{sid}_{model}-cf_Fibroblast_CRC.json`. `inference.py` calls
`eval_loo.py` itself (skip with `--skip-eval`); to re-score later, add
`{"class": "terra", "name": "terra", "extra_args": "--use_cf"}` (and `"terra-null"`) to `MODELS` in
`scripts/eval_parallel.py`. `--max-cells N` shrinks the input h5ad for smoke tests only.

**Species.** TERRA is human-only: its vocabulary is 21,952 ENSG tokens, zero ENSMUSG. The merfish
branch in `common.py` is a cross-species hack — mouse symbols upper-cased and looked up as human
orthologs, dropping the ~79/1120 that miss — not something TERRA's authors released a model or a
protocol for. It is wired up but not run; treat any merfish number from it as illustrative.
