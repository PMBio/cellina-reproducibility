# TERRA on Slurm (EMBL cluster)

Everything — raw slides, HF cache, tokenizer caches, LoRA runs, decoder checkpoints, scored
h5ads, correlation JSONs — lives under the repo checkout in `data/` (gitignored).
**Always submit from the repo root**; the sbatch scripts source
`$SLURM_SUBMIT_DIR/scripts/terra/slurm/env.sh`.

`env.sh` sets: `DATA_ROOT=$REPO/data`, `HF_HOME=$DATA_ROOT/hf`, `HF_HUB_OFFLINE=1`
(TERRA-96M and 112M are pre-downloaded on the login node; compute nodes may lack egress),
`PY=/g/stegle/ddimitro/miniforge3/envs/terra/bin/python`,
`SLIDES="crc_120 crc_210 crc_221 crc_231 crc_232 crc_242"`, `LOG=scripts/terra/slurm/logs`.

Raw slides were **copied** (not symlinked) from `/g/stegle/data/crc_wt_cosmx/` — the untouched
Zenodo record 15574384 files — to `data/datasets/crc/raw_zenodo/crc_{120,210,221,231,232,242}.h5ad`.

**One GPU per job.** terra's fine-tune path has no DDP, so never request several GPUs.
terra hardcodes `cuda:0`; Slurm's `CUDA_VISIBLE_DEVICES` masking makes that the allocated GPU.

Partitions for account `stegle`:

| partition | cards | note |
|---|---|---|
| `gpu-el8` | 3090 (24 GB), A40 / L40s (48), A100 / H100 (80) | A40 nodes mostly down; RTX PRO 6000 (Blackwell, sm_120) is unusable: the env's torch 2.5.1+cu121 stops at sm_90 |
| `gpu-training` | H100 (80), H200 (141) | at most 3 running jobs for us; B200 is Blackwell, unusable |

Slides by `n_obs`: crc_120 741,155 · crc_210 617,498 · crc_242 474,471 · crc_221 370,074 ·
crc_231 298,151 · crc_232 130,814.

## Stage 0 — probe

```bash
sbatch scripts/terra/slurm/probe.sbatch        # crc_120, batch 128, 100 steps, L40s (48 GB)
```

crc_120 is the largest slide, so it bounds memory and time for all six. Prints a
`PROBE_RESULT` line with `peak_gpu_gib` and `sec_per_step`; those set `MIN_VRAM` and the
walltime for stage 1. With `--max-steps` the run tokenizes only `STEPS*BATCH` cells into its own
work dir, so nothing is reused by stage 1. Measured 2026-09-16 on an L40s: TERRA-96M peak 38.5 GiB, 1.48 s/step at
batch 128, guard passed; TERRA-112M OOMs at batch 128 on 44 GB (`TERRA_MODEL=... BATCH=64` to retry). Overrides: `SID=`, `BATCH=`, `STEPS=`, and any `sbatch --gres=`.
A probe is never re-entrant — it wipes its own `terra_probe_b{BATCH}` work dir.

## Stage 1 — LoRA fine-tune, all six slides

```bash
scripts/terra/slurm/submit_finetune.sh                     # default: cards with >=48 GB
MIN_VRAM=80 scripts/terra/slurm/submit_finetune.sh         # only >=80 GB cards
SLIDES="crc_120 crc_210" scripts/terra/slurm/submit_finetune.sh
PARTITION=gpu-training SLIDES="crc_120 crc_210 crc_242" scripts/terra/slurm/submit_finetune.sh
TEST_ONLY=1 scripts/terra/slurm/submit_finetune.sh         # sbatch --test-only dry run
```

One job array over `$SLIDES`, one task per slide, `--gres=gpu:1` with an OR constraint over
the partition's card types whose VRAM is `>= MIN_VRAM` (default 48). Our association forbids
multi-partition jobs, so the partition is chosen per call: `gpu-el8` (default; features
`L40s|A100|H100`, 7-day limit) or `gpu-training` (`H100|H200`, 14-day limit, at most
3 running jobs). Split the slide list over two calls to use both.
Each task runs `finetuning_lora.py` at batch 128 / 5 epochs / lr 1e-4 and is re-entrant: an
existing `epoch_selection.json` makes it a no-op, and a run dir holding all 5 checkpoints is
reused via `--from-run-dir` (export/embed/guard only, no retraining).

## Stage 2 — inference + shift path, one slide at a time

```bash
scripts/terra/slurm/submit_stage2.sh crc_232                # after its epoch_selection.json exists
MIN_VRAM=80 PARTITION=gpu-el8 scripts/terra/slurm/submit_stage2.sh crc_120
TEST_ONLY=1 scripts/terra/slurm/submit_stage2.sh crc_232
```

One job per slide (`stage2.sbatch`, one GPU, 16 CPUs, 200 GB, 3-day limit). Inside it the
**arms** — `frozen`, `lora:{final_epoch}` and `lora:{latest_passing_epoch}` when that differs,
derived from `epoch_selection.json` via `summarize.arms()` — run **sequentially**: they share
model-independent caches (tokenizer tmp dir, ctrl/pert tokenised sets, terra2k universe) that
`inference.py` / `eval_terra.py` rebuild in place, and parallel arms of one slide corrupted each
other (stale NFS handles, bus errors, 2026-09-16). Slides are independent and run in parallel.
The scored cell types come from the `cellina-pert` rows of the reference CSV
(`summarize.cellina_df()`), never hardcoded, and are exported **space-separated** because
`sbatch --export` splits on commas (that truncated the list to one type once). For every arm
and all scored types of the slide:

1. **decoder track** — `inference.py`: embed, count decoder (LOO per type), neighbour-token
   perturbation, the four `eval_loo.py` h5ads, `eval_loo.py --use_cf --log_norm_x` →
   `correlations/{sid}_terra-<arm>[-null]-cf_{ct}_CRC.json`;
2. **shift track** — `eval_terra.py --universe terra2k --cellina-cf cellina-pert` →
   `correlations/{sid}_terra-<arm>-native-terra2k_{ct}_CRC.json`. The terra2k universe
   (`make_terra2k.py`, CPU, minutes) is built at the start of the job if missing (a separate
   `htc-el8` job was estimated at a 10 h queue wait, not worth it).

Everything is re-entrant through the caches under `{sid}/{ct}/terra`. The `cellina-pert`
reference column of the shift path is scored only when its h5ads exist under
`datasets/crc/{sid}_terra2k/{ct}/`; until they are regenerated (only
`notebooks/loo_benchmarks/cellina_node_pert.ipynb` exists), `eval_terra.py` prints a warning
and skips that column. Re-running `eval_terra.py` later fills it in from the cached shift results.

Then `summarize.py` (CPU, seconds) writes `results/terra_crc_{decoder_DEG_50,shift_terra2k}.csv`.

## Logs

`scripts/terra/slurm/logs/` (gitignored): `probe_{jobid}.log`, `ft_{arrayid}_{task}.log`, `s2_{jobid}.log`.
`queue/worker.sh` writes to `scripts/terra/queue/logs/` instead — see `../queue/README.md`.

Memory note (2026-09-16): the shift track embeds the control cells with token embeddings
(`(n, 2816, 384)` fp32 on the host, ~4.3 MB per cell). crc_231 Epithelial (24,934 control
cells) blew the 200 GB job. `eval_terra.py` now embeds in chunks of 2,000 cells and caps the
control cells at `--max-ctrl-cells 15000` (seeded random subsample, same for every arm; crc_232
had at most 12,645 and is unaffected). The `SKIP gene shift ... < 1000 scorable HVGs` lines are
by design (small cell types have too few token-covered HVGs for a fair precision@50).

## Cell+neighbourhood shift control (2026-09-17)

`shift_cellpert.sbatch` / `submit_shift_cellpert.sh` re-run the **shift track only** with
`eval_terra.py --perturb-cell`, which applies the logFC shift to the focal cell's own token
block as well as to the neighbourhood — TERRA's own
`perturbation_target=["cell", "neighborhood"]` rather than our neighbourhood-only default.

It is a control, not a replacement. The neighbourhood-only shift scores at chance on all six
slides, and the perturbation only reaches the scored tokens (the cell's own 256 positions)
through attention. Perturbing the cell directly separates "the model does not propagate
neighbourhood signal" from "the readout cannot detect anything at all": if precision@50 stays
at the random-gene control here too, the gene-level W2 readout is the problem, and the
neighbourhood-only restriction is exonerated.

Results pair 1:1 with the existing `*-native-terra2k_*` JSONs (same arms, cell types, seeds and
15k control-cell cap) and land in separate files — `*-native-terra2k-cellpert_*` JSONs,
`gene_shift_*_cellpert.csv`, and `ctrl_tok_*_cellpert` / `pert_tok_*_cellpert` caches — so
nothing already on disk is overwritten. Every JSON now records `perturbation_target`.

First run: `crc_232`, all three arms, job 61990015.
