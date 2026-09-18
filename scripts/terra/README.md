# TERRA on the LOO benchmark

TERRA-96M (`lotfollahi-lab/TERRA-96M`, Lotfollahi lab) evaluated on the CRC leave-one-cell-type-out
benchmark, two tracks:

- **decoder track** (Table 1 rows `TERRA` / `TERRA-LoRA`, node-perturbation block): TERRA embeddings
  → count decoder → neighbour-token perturbation → the same `scripts/eval_loo.py` metrics as every
  other method.
- **terra_og track** (supplementary): TERRA's own perturbation readout — `perturb_dataset(foldchange)`
  + per-gene Sinkhorn W2 in embedding space, precision@50 against the cellina-pert logFC. Design and
  decision log: `TERRA_OG_EVAL_SPEC.md`; run log and validation: `results/terra_og/README.md`.

Everything runs from the repo root; all data, caches and model files live under `data/` (gitignored):
raw slides `data/datasets/crc/raw_zenodo/crc_{120,210,221,231,232,242}.h5ad` (copied, not symlinked,
from the untouched Zenodo 15574384 record), HF cache `data/hf` (`HF_HOME`, jobs run `HF_HUB_OFFLINE=1`).

## Environment

`environments/terra.yml` → `/g/stegle/ddimitro/miniforge3/envs/terra/bin/python` (`PY` in
`slurm/env.sh`). `terra-st==0.1.13`, torch 2.5.1+cu121, geomloss, plus everything `eval_loo.py`
imports; `cellina` is absent, so `preprocess_spatial_features` warns and skips the spatial graph
(`eval_loo.py` never uses it for scoring). **One GPU per job**: terra's fine-tune path has no DDP and
`embed_dataset` hardcodes `cuda:0` (Slurm's `CUDA_VISIBLE_DEVICES` masking makes that the allocated
card). `TERRA_MODEL=lotfollahi-lab/TERRA-112M` switches models (needs an 80 GB card at batch 128; its
token dictionary has `spv_cosmx`/`spv_colon`, so it may have seen these slides — 96M is the choice of
record, 2026-09-16).

## Scripts

| file | role |
|---|---|
| `common.py` | benchmark preprocessing (`train_loo.preprocess_*` + `split_indices`), the TERRA-harmonised all-genes twin, tokenizer/embedding caches, path layout; `arms()` / `cellina_df()` derive the arms per slide and the scored cell types for the submitters. `python common.py --universe terra2k --tokenize` builds the terra2k tokenizer cache (stage 2a). |
| `finetuning_lora.py` | stage 1: self-supervised I-JEPA LoRA fine-tune on **all** cells of one slide + collapse guard → `epoch_selection.json` |
| `make_terra2k.py` | stage 2a: `{sid}_terra2k.h5ad` + gene list — terra's own 2000 seurat_v3 HVGs (CPU) |
| `inference.py` | stage 2b: decoder track for one arm, all scored cell types; calls `eval_loo.py` |
| `terra_og_eval.py` | stage 3: terra_og track, one job = slide × arm × target |
| `terra_summary_tables.py` | both tracks → per-fold CSVs in the `loo_summary_crc_DEG_50_v4.csv` schema |
| `terra_handoff.py` | the hand-off folder `results/terra_handoff/` (v5 summary CSV + supp shift CSV + rendered tables) |
| `slurm/` | `env.sh`, `probe.sbatch`, `finetune.sbatch` + `submit_finetune.sh`, `stage2.sbatch` + `submit_stage2.sh`, `terra_og.sbatch` + `submit_terra_og.sh`; logs in `slurm/logs/` (gitignored) |

## Stage 0 — probe

```bash
sbatch scripts/terra/slurm/probe.sbatch        # crc_120 (largest slide), batch 128, 100 steps
```

Prints `PROBE_RESULT` with `peak_gpu_gib` / `sec_per_step`, which set `MIN_VRAM` and the walltime
for stage 1. Measured 2026-09-16 on an L40s: TERRA-96M peak 38.5 GiB, 1.48 s/step, guard passed;
TERRA-112M OOMs at batch 128 on 44 GB. A probe wipes its own `terra_probe_b{BATCH}` dir (not re-entrant).

## Stage 1 — LoRA fine-tune, all six slides

```bash
scripts/terra/slurm/submit_finetune.sh                         # gpu-el8, cards >= 48 GB
MIN_VRAM=80 SLIDES="crc_120 crc_210" scripts/terra/slurm/submit_finetune.sh
PARTITION=gpu-training SLIDES="crc_242" scripts/terra/slurm/submit_finetune.sh
TEST_ONLY=1 scripts/terra/slurm/submit_finetune.sh             # sbatch --test-only
```

One job array over `$SLIDES`, one GPU per task. Recipe (`finetuning_lora.py`): LoRA r16 / alpha 256 /
dropout 0.1 on `qkv, proj, fc1, fc2`, wd 0.04→0.4, EMA 0.9995→1, smooth_l1, clip 2, bf16, warmup 0.1
epoch, **lr 1e-4** (the tutorial's 1e-3 collapses the encoder), **batch 128**, **5 epochs**,
`save_every 1`; resolved config in `{sid}/terra/lora_run/ft_config.json`. Every epoch checkpoint is
repacked (`prepare_finetuned_model`) to `{sid}/terra/lora_ep{N}/lora_bundle` and scored by the
**collapse guard** on a fixed 20,000-cell subsample (seed 0) against the frozen encoder, per
embedding (`cell_emb`, `spatial_cell_emb`, `neighborhood_emb`): effective rank > 0.5× frozen, mean
per-dim std > 0.5× frozen, mean cosine-to-frozen > 0.5. `epoch_selection.json` records
`final_epoch`, `latest_passing_epoch` (null if none) and per-epoch verdicts; **nothing aborts on a
failed guard**. Re-entrant: an existing `epoch_selection.json` is a no-op, a run dir holding all
5 checkpoints is reused via `--from-run-dir`.

**Arms per slide** (`common.arms()`): `terra-frozen`, `terra-lora-ep{final}` and
`terra-lora-ep{latest_passing}` when that differs. The manuscript uses **frozen + final epoch**
(decision 2026-09-18: no intervention on our side; crc_221/crc_231's guard rejected epoch 5, which the
supp note states). The guard arm is kept in the per-fold CSVs as provenance.

**Encoder exposure.** The encoder is not held out: the fine-tune is self-supervised on every cell of
the slide, including the scored type's CRC cells. The *decoder* is strictly LOO per scored type
(`train_loo.split_indices` in that type's own `work_dir`).

## Stage 2 — decoder track, one slide per job

```bash
scripts/terra/slurm/submit_stage2.sh crc_232                 # needs {sid}/terra/epoch_selection.json
MIN_VRAM=80 scripts/terra/slurm/submit_stage2.sh crc_120
TEST_ONLY=1 scripts/terra/slurm/submit_stage2.sh crc_232
```

`stage2.sbatch` (one GPU, 16 CPUs, 200 GB, 3 days): **2a** builds `{sid}_terra2k.h5ad` if missing
(consumed by stage 3, built here because the raw slide is already loaded); **2b** runs `inference.py`
per arm — arms **sequentially** (they share caches that are rebuilt in place; parallel arms of one
slide corrupted each other on 2026-09-16), slides in parallel. The scored cell types are the
`cellina-pert` rows of `origin/results:results/loo_cellina_crc_DEG_50_pert_v2.csv`
(`common.cellina_df()`), exported space-separated (`sbatch --export` splits on commas).

`inference.py`: embed → count decoder (`terra.training.decode`, gene list = the 2000 benchmark HVGs in
order) → neighbour-token perturbation (block 0 = focal cell, blocks 1–10 = neighbours, 256 tokens
each) → the four h5ads `eval_loo.py --use_cf --log_norm_x` reads (`{model}_recon_x.h5ad`,
`{model}_counterfactual_x_CRC.h5ad` and the `{model}-null` pair; `-null` = same forward pass with
the perturbation zeroed, a sanity check that stays out of the main table) →
`data/datasets/crc/correlations/{sid}_{model}-cf_{ct}_CRC.json`, `{model}` ∈ `terra-frozen`,
`terra-lora-ep{N}`. `--skip-eval` skips `eval_loo.py`; `--max-cells N` is for smoke tests only.

## Stage 3 — terra_og track

```bash
scripts/terra/slurm/submit_terra_og.sh crc_232                 # 3 arms x 2 targets + random-gene jobs
ARMS=frozen RANDOM_SEEDS="" scripts/terra/slurm/submit_terra_og.sh crc_232
SKIP_DONE=1 scripts/terra/slurm/submit_terra_og.sh crc_231     # resubmit only missing per-run JSONs
```

Requires stage 2a's `{sid}_terra2k.h5ad` and the tokenizer cache `{sid}/terra_tok_terra2k`
(`python scripts/terra/common.py --dataset_name crc --adata_path $A --holdout_celltype Endothelial
--universe terra2k --tokenize`; the `holdout_celltype` only selects a cache dir). Per target the
frozen job runs first and builds the shared control / perturbed-dataset caches under
`{sid}/terra_og_cache/`; the other arms and the random-gene control depend on it. Output:
`results/terra_og/per_run/{sid}_{arm}_{target}[_random]_{ct}.json` + per-gene CSVs. Every arm
asserts the loaded weights differ from the frozen bundle (A1) because `terra`'s `load_checkpoint`
silently falls back to random init on a bad state dict. Full contract: `TERRA_OG_EVAL_SPEC.md`.

## Summary and hand-off

```bash
python scripts/terra/terra_summary_tables.py   # -> results/terra/terra_{decoder,shift}_folds.csv
python scripts/terra/terra_handoff.py          # -> results/terra_handoff/ (v5 CSV, supp shift CSV, tables)
```

Both read the committed reference tables from `origin/results` via `git show`, never a local copy:
the local `correlations/` holds only TERRA JSONs, so re-running the other methods' summary notebook
cell here would destroy their rows. The decoder rows are appended to `loo_summary_crc_DEG_50_v4.csv`
as `_v5`; the shift rows encode the target as `{cell type}_{nb-only|ct-nb}` so
`notebooks/make_table.ipynb` can reuse its `perturbation` axis. Degenerate universes (chance > 0.2,
crc_221 Endothelial/Myeloid) are flagged and excluded from the aggregate. `results/terra_handoff/README.md`
has the notebook edits, including the `MODEL_ORDER` trap (methods missing from it are silently dropped).

## Attempted approaches (removed 2026-09-18, in git history before that date)

- **Native gene-embedding W2 shift on the cellina-pert reference** (`eval_terra.py`,
  `*-native-terra2k_*` JSONs): our own re-implementation of TERRA's paper readout — shift the
  neighbour tokens by the logFC, score each gene's token-cloud W2 between control and perturbed,
  precision@50 against the cellina-pert logFC on the terra2k universe. Scored at chance on all six
  slides for every arm and needed a 15k control-cell cap to fit 200 GB. Superseded by the terra_og
  track, which uses TERRA's own `perturb_dataset` instead of our token edit.
- **Cell + neighbourhood shift control** (`shift_cellpert.sbatch`, `--perturb-cell`,
  `*-native-terra2k-cellpert_*`): same readout with the focal cell's own block also shifted, to
  separate "no propagation" from "readout detects nothing". Ran on crc_232 only; folded into
  terra_og's `ct_neigh` target, where it shows the perturbation is recovered as *which tokens were
  edited*, not which genes respond (precision collapses to chance once the perturbed genes leave
  the universe).
- **Single-GPU pull queue** (`queue/worker.sh`): fine-tune → arms → decoder → shift → summary on
  one bare GPU, predating the Slurm scripts. Slurm supersedes it.
- **`summarize.py`**: aggregated against the per-method CSV `loo_cellina_crc_DEG_50_pert_v2.csv`;
  Table 1 is built from `loo_summary_crc_DEG_50_v4.csv`, so `terra_summary_tables.py` replaced it.
  Its `arms()` / `cellina_df()` moved to `common.py`.

The on-disk outputs of these (old `correlations/*-native-terra2k*` JSONs, `results/terra_og/tables/`,
`results/terra_og/superseded/`) are kept as-is.

## Cluster notes (EMBL, account `stegle`)

| partition | cards | note |
|---|---|---|
| `gpu-el8` | 3090 (24 GB), A40 / L40s (48), A100 (40 on gpu25–28 / 80), H100 (80); 7-day limit | A40 nodes mostly down; RTX PRO 6000 (Blackwell, sm_120) unusable — torch 2.5.1+cu121 stops at sm_90 |
| `gpu-training` | H100 (80), H200 (141); 14-day limit | at most 3 running jobs; B200 (Blackwell) unusable |

The association forbids multi-partition jobs, so `PARTITION` is chosen per submit call; `MIN_VRAM`
selects card types via an OR `--constraint`. Slides by `n_obs`: crc_120 741,155 · crc_210 617,498 ·
crc_242 474,471 · crc_221 370,074 · crc_231 298,151 · crc_232 130,814. Token embeddings are
`(n, 2816, 384)` fp32 on the host (~4.3 MB/cell); `terra_og_eval.py` embeds in chunks of 2,000 cells.

**Species.** TERRA is human-only (21,952 ENSG tokens). The merfish branch in `common.py` upper-cases
mouse symbols and looks them up as human orthologs (dropping ~79/1120) — wired up, never run, not a
protocol TERRA's authors released; treat any merfish number from it as illustrative.
