# TERRA benchmark queue

6 slides (`crc_120 crc_210 crc_221 crc_231 crc_232 crc_242`), holdout domain `CRC`.
The scored `(sid, cell type)` pairs are **derived**, never hardcoded: they are the
`model_name == "cellina-pert"` rows of `origin/results:results/loo_cellina_crc_DEG_50_pert_v2.csv`
(`summarize.cellina_df()`).

## Recipe

**Fine-tune** (`finetuning_lora.py`, once per slide, ALL cells, no labels, no holdout):
TERRA-96M, self-supervised I-JEPA LoRA — r16 / alpha 256 / dropout 0.1 on
`qkv, proj, fc1, fc2`, wd 0.04→0.4, EMA 0.9995→1, smooth_l1, clip 2, bf16,
warmup 0.1 epoch, **lr 1e-4** (the tutorial's 1e-3 collapses the encoder), batch 64,
**5 epochs**, `save_every 1`.  The resolved config lands in
`{slide}/terra/lora_run/ft_config.json`.

Every epoch checkpoint is then repacked (`prepare_finetuned_model`) into
`{slide}/terra/lora_ep{N}/lora_bundle`, used to embed a fixed 20,000-cell subsample
(`--guard-cells`, seed 0) and scored by the collapse guard. Full-slide embeddings of the
selected arms are computed and cached by `inference.py` (`{slide}/terra/lora_ep{N}/emb_lora.npz`).
(crc_232 was guarded before this change on its full-slide embeddings with a 1000-cell guard set.)
Results go to `{slide}/terra/epoch_selection.json`; **nothing aborts on a failed guard**.

Guard criteria, per embedding (`cell_emb`, `spatial_cell_emb`, `neighborhood_emb`), on a
fixed 20,000-cell subsample against the frozen encoder: effective rank > 0.5x frozen,
mean per-dim std > 0.5x frozen, mean cosine-to-frozen > 0.5.
`latest_passing_epoch` = the highest epoch passing all of them (`null` if none).

**Arms per slide** — `terra-frozen` (96M as-is), `terra-lora-ep5` (TERRA's own
final-checkpoint default) and `terra-lora-ep{K}` where K = `latest_passing_epoch`,
only when K is neither 5 nor null.

**Encoder exposure.** The encoder is never held out any more: the LoRA fine-tune is
self-supervised on every cell of the slide, including the scored type's CRC cells.
The *decoder* is still strictly LOO per scored type (`train_loo.split_indices` in that
type's own `work_dir`), which is the point of the benchmark. Both facts are recorded in
the `encoder_exposure` column.

**Decoder path** (`inference.py`): embed → count decoder (`terra.training.decode`, the
2000 benchmark HVGs in order) → neighbour-token perturbation → the four h5ads
`eval_loo.py` reads → `eval_loo.py --use_cf --log_norm_x` (`--log_norm_x` makes
`edistance_pca`/`edistance_pca_log` use X = log1p(CP10K), as in
`notebooks/loo_benchmarks/cellina_node_pert.ipynb`).  Writes
`correlations/{sid}_terra-{frozen|lora-ep{N}}[-null]-cf_{ct}_CRC.json`.

**Shift path** (`eval_terra.py`): TERRA-native gene-embedding W2 shift, no decoder,
universe = terra's own 2000 harmonised HVGs (`--universe terra2k`, favourable to TERRA),
`--cellina-cf cellina-pert` for the reference arm. Writes
`correlations/{sid}_terra-…-native-terra2k_{ct}_CRC.json`.

**Summary** (`summarize.py`): `results/terra_crc_decoder_DEG_50.csv` and
`results/terra_crc_shift_terra2k.csv`.

## Running

```bash
scripts/terra/queue/launch_overnight.sh      # splits the 6 slides over GPU 0/1 and nohups two workers
```

The split is by `n_obs` of the raw h5ad (read with h5py, X untouched), greedy longest-first;
`crc_232` counts at 1/3 because it skips the fine-tune (its checkpoints already exist under
`crc_232/terra_lora5ep/lora_run/run`, reused via `--from-run-dir`).

One slide on one GPU:

```bash
scripts/terra/queue/worker.sh 0 crc_221
```

One step by hand (everything is re-entrant — existing bundles, embeddings, decoder
checkpoints and `epoch_selection.json` are reused, delete the artefact to force a redo):

```bash
export DATA_ROOT=/data/ddimitrov/data CUDA_VISIBLE_DEVICES=0
PY=/data/ddimitrov/software/miniforge3/envs/terra/bin/python
A=$DATA_ROOT/datasets/crc/raw_zenodo/crc_221.h5ad
CT=Endothelial,Epithelial,Fibroblast,Myeloid,T_cell
$PY scripts/terra/finetuning_lora.py --dataset_name crc --adata_path $A
$PY scripts/terra/inference.py  --dataset_name crc --adata_path $A --holdout_celltype Endothelial --eval-celltypes $CT --variant lora --epoch 5
$PY scripts/terra/eval_terra.py --dataset_name crc --adata_path $A --holdout_celltype Endothelial --eval-celltypes $CT --variant lora --epoch 5 --universe terra2k --cellina-cf cellina-pert
$PY scripts/terra/summarize.py
```

Logs: `scripts/terra/queue/logs/{sid}_{step}.log`, with `START`/`END rc=`/`ABORT` lines in
`logs/{sid}_status.txt`. A failed step aborts that slide only; the worker continues with the
next one and still writes the summary at the end.
