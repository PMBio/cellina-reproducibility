# TERRA benchmark queue

`worker.sh` is the shared step chain (fine-tune -> arms -> decoder path -> shift path -> summary)
for one GPU. On the cluster, submit through `../slurm/README.md` instead.

6 slides (`crc_120 crc_210 crc_221 crc_231 crc_232 crc_242`), holdout domain `CRC`.
The scored `(sid, cell type)` pairs are **derived**, never hardcoded: they are the
`model_name == "cellina-pert"` rows of `origin/results:results/loo_cellina_crc_DEG_50_pert_v2.csv`
(`summarize.cellina_df()`).

## Recipe

**Fine-tune** (`finetuning_lora.py`, once per slide, ALL cells, no labels, no holdout):
TERRA-96M (`TERRA_MODEL` switches to 112M, see `../PIPELINE_SPEC.md`), self-supervised I-JEPA LoRA — r16 / alpha 256 / dropout 0.1 on
`qkv, proj, fc1, fc2`, wd 0.04→0.4, EMA 0.9995→1, smooth_l1, clip 2, bf16,
warmup 0.1 epoch, **lr 1e-4** (the tutorial's 1e-3 collapses the encoder), **batch 128**,
**5 epochs**, `save_every 1`.  The resolved config lands in
`{slide}/terra/lora_run/ft_config.json`.

Batch 128 for **all six slides**.  The recipe was validated at batch 64 on 24 GB cards; lr and
epochs are **not** rescaled, so at 128 the run takes half the optimizer steps and the EMA target
encoder moves about half as far — a milder fine-tune, but identical across slides.  Old crc_232
numbers (batch 64, 1000-cell guard) are superseded: crc_232 is retrained from scratch like every
other slide.

Every epoch checkpoint is then repacked (`prepare_finetuned_model`) into
`{slide}/terra/lora_ep{N}/lora_bundle`, used to embed a fixed 20,000-cell subsample
(`--guard-cells`, seed 0) and scored by the collapse guard — the same guard for all six slides.
Full-slide embeddings of the selected arms are computed and cached by `inference.py`
(`{slide}/terra/lora_ep{N}/emb_lora.npz`).  Results go to
`{slide}/terra/epoch_selection.json`; **nothing aborts on a failed guard**.

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
`--cellina-cf cellina-pert` for the reference arm, precision@50 (`--k 50`; the TERRA paper uses 100).
A `(slide, cell type)` whose universe (ii) has fewer than 1000 scorable HVGs (`--min-genes`) is skipped
before any GPU work; its JSON holds only a `skipped` reason and `summarize.py` reports it as such.
`make_terra2k.py` must have been run once for the slide (CPU) first. Writes
`correlations/{sid}_terra-…-native-terra2k_{ct}_CRC.json`.

**Summary** (`summarize.py`): `results/terra_crc_decoder_DEG_50.csv` and
`results/terra_crc_shift_terra2k.csv`.

Embeddings are computed with `batch_size=128` (terra's default; inference only, no protocol
content).

## Running

```bash
scripts/terra/queue/worker.sh auto crc_221 crc_231   # inside a Slurm job: use the allocated GPU
scripts/terra/queue/worker.sh 0 crc_221              # bare machine: pin physical GPU 0
```

The first argument is `auto` or a physical GPU index. One slide per GPU at a time; terra's
fine-tune path has no DDP, so never hand a worker several GPUs.

One step by hand (everything is re-entrant — existing bundles, embeddings, decoder
checkpoints and `epoch_selection.json` are reused, delete the artefact to force a redo):

```bash
export DATA_ROOT=$PWD/data HF_HOME=$PWD/data/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=/g/stegle/ddimitro/miniforge3/envs/terra/bin/python
A=$DATA_ROOT/datasets/crc/raw_zenodo/crc_221.h5ad
CT=Endothelial,Epithelial,Fibroblast,Myeloid,T_cell
$PY scripts/terra/finetuning_lora.py --dataset_name crc --adata_path $A
$PY scripts/terra/inference.py  --dataset_name crc --adata_path $A --holdout_celltype Endothelial --eval-celltypes $CT --variant lora --epoch 5
$PY scripts/terra/eval_terra.py --dataset_name crc --adata_path $A --holdout_celltype Endothelial --eval-celltypes $CT --variant lora --epoch 5 --universe terra2k --cellina-cf cellina-pert
$PY scripts/terra/summarize.py
```

Logs (when `worker.sh` is used): `scripts/terra/queue/logs/{sid}_{step}.log`, with
`START`/`END rc=`/`ABORT` lines in `logs/{sid}_status.txt`. A failed step aborts that slide only; the worker continues with the
next one and still writes the summary at the end.
