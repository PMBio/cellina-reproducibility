# TERRA on the LOO benchmark

Five scripts (spec: `PIPELINE_SPEC.md`; protocol + step chain: `queue/README.md`; running on
the EMBL Slurm cluster: `slurm/README.md`), all run from the repo root:

- `common.py` — benchmark preprocessing (`train_loo.preprocess_*` + `split_indices`) plus a
  row-identical TERRA-harmonised all-genes twin, tokenisation + embedding caches, path layout.
  Not a CLI.
- `finetuning_lora.py` — self-supervised (I-JEPA) LoRA fine-tune of TERRA-96M (`TERRA_MODEL` switches to 112M) on **all** cells of
  one slide, 5 epochs, lr 1e-4, batch 128. Exports every epoch checkpoint to
  `{slide}/terra/lora_ep{N}/lora_bundle`, embeds all cells into `.../lora_ep{N}/emb_lora.npz`,
  runs the collapse guard and writes `{slide}/terra/epoch_selection.json`.
- `inference.py` — embed → count decoder (`terra.training.decode`, gene list = the 2000 benchmark
  HVGs in order) → neighbour-token perturbation → writes the four h5ads `eval_loo.py` reads → runs
  `eval_loo.py`. `--variant frozen` = the pretrained bundle, `--variant lora --epoch N` = the
  epoch-N LoRA bundle.
- `eval_terra.py` — TERRA-native gene-embedding-shift scoring (no decoder), reads inference's caches.
- `summarize.py` — all 6 slides → `results/terra_crc_decoder_DEG_50.csv`,
  `results/terra_crc_shift_terra2k.csv`.
- `make_terra2k.py` — builds the shift path's `{sid}_terra2k.h5ad` (terra's own harmonised HVGs).

Env: `/g/stegle/ddimitro/miniforge3/envs/terra/bin/python` (has terra + everything `eval_loo.py`
imports; `cellina` is absent, so `preprocess_spatial_features` prints a warning and skips the spatial
graph — `eval_loo.py` never uses it for scoring). `DATA_ROOT` defaults to `<repo>/data` (gitignored),
with the raw slides under `data/datasets/crc/raw_zenodo/`, and TERRA-96M (and 112M) pre-downloaded into
`data/hf` (`HF_HOME`, jobs run `HF_HUB_OFFLINE=1`).
One GPU per job; `embed_dataset` hardcodes `cuda:0`, so pick the device with `CUDA_VISIBLE_DEVICES`
(on Slurm, its masking already makes the allocated GPU `cuda:0`). terra's fine-tune path has no DDP —
never request several GPUs.

One (slide, cell type):

```bash
export DATA_ROOT=$PWD/data HF_HOME=$PWD/data/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=/g/stegle/ddimitro/miniforge3/envs/terra/bin/python
A=$DATA_ROOT/datasets/crc/raw_zenodo/crc_232.h5ad
$PY scripts/terra/finetuning_lora.py --dataset_name crc --adata_path $A
$PY scripts/terra/inference.py  --dataset_name crc --adata_path $A --holdout_celltype Fibroblast --variant lora --epoch 5
$PY scripts/terra/inference.py  --dataset_name crc --adata_path $A --holdout_celltype Fibroblast --variant frozen
$PY scripts/terra/eval_terra.py --dataset_name crc --adata_path $A --holdout_celltype Fibroblast --variant lora --epoch 5 \
    --universe terra2k --cellina-cf cellina-pert
```

All six slides: submit the stage-1 job array with `scripts/terra/slurm/submit_finetune.sh`
(`slurm/README.md`), or run the whole step chain for a slide on one GPU with
`scripts/terra/queue/worker.sh auto crc_221`.

Outputs land in `$DATA_ROOT/datasets/crc/{sid}/{ct}/` — `{model}_recon_x.h5ad`,
`{model}_counterfactual_x_CRC.h5ad` and the `{model}-null` pair, exactly the files
`eval_loo.py --use_cf` loads; `{model}` is `terra-frozen` or `terra-lora-ep{N}`. Caches, decoder
checkpoint/metrics, `logfc_CRC.csv` and the `ctrl_tok_CRC`/`pert_tok_CRC` tokenised sets sit under
`.../{ct}/terra/`; the slide-level encoder artefacts under `.../{sid}/terra/`. Scores go to
`$DATA_ROOT/datasets/crc/correlations/{sid}_{model}-cf_{ct}_CRC.json`. `inference.py` calls
`eval_loo.py` itself (skip with `--skip-eval`). `--max-cells N` shrinks the input h5ad for smoke
tests only.

**Encoder exposure.** The LoRA fine-tune is self-supervised on every cell of the slide, so the
encoder is *not* held out; the decoder still is, per scored cell type. `summarize.py` records this
in `encoder_exposure`.

**Species.** TERRA is human-only: its vocabulary is 21,952 ENSG tokens, zero ENSMUSG. The merfish
branch in `common.py` is a cross-species hack — mouse symbols upper-cased and looked up as human
orthologs, dropping the ~79/1120 that miss — not something TERRA's authors released a model or a
protocol for. It is wired up but not run; treat any merfish number from it as illustrative.
