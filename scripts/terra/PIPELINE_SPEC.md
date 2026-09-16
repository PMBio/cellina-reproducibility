# TERRA LOO pipeline — spec (2026-09-14)

Goal: turn `notebooks/loo_benchmarks/terra/terra_node_pert_finetune.ipynb` + `finetune_terra.py` into three
scripts whose outputs plug into the existing benchmark **unchanged**: preprocessing and splits come from
`scripts/train_loo.py`, scoring comes from `scripts/eval_loo.py`. Notebooks stay as they are. Nothing under
the installed `terra` package is edited. PONYTAIL: copy working code from the notebook, do not redesign it.

Env: `/data/ddimitrov/software/miniforge3/envs/terra/bin/python`, run from repo root
(`eval_loo.py` does `sys.path.append('./scripts')`). GPU: **only GPU 0** (`CUDA_VISIBLE_DEVICES=0`); GPU 1
belongs to someone else. Agents may run smoke tests on small subsets; full runs are launched by the orchestrator.

## Files (all in `scripts/terra/`)

| file | owner | job |
|---|---|---|
| `common.py` | agent A | data loading = benchmark preprocessing + TERRA harmonise/tokenise; path + embedding caches |
| `finetuning_lora.py` | agent A | self-supervised LoRA fine-tune on all cells, per-epoch bundle/embedding export + collapse guard |
| `inference.py` | agent B | embed → count decoder → neighbour perturbation → write `eval_loo.py`-format h5ads → call `eval_loo.py` |
| `eval_terra.py` | agent C | TERRA-native scoring (paper protocol B, gene-embedding W2 shift; no decoder) → JSON next to the others |
| `summarize.py` | agent A | all 6 slides x arms → `results/terra_crc_{decoder_DEG_50,shift_terra2k}.csv` |
| `queue/` | agent A | one worker per GPU: fine-tune → arms → decoder path → shift path → summary (`queue/README.md`) |

## Fixed decisions

- Datasets: `--dataset_name {crc,merfish}` everywhere, same choices/config as `eval_loo.py`
  (`configs/adata_crc_config.py`, `configs/adata_merfish_config.py`). crc: labels `coarse_type`, domains
  `typ_clean`, control `REF`, holdout `CRC`, 2000 HVGs. merfish: labels `cell_type`, domains
  `major_brain_region`, control `Thalamus`, holdouts `Isocortex`,`Fiber_tracts`, 1120 HVGs.
- DATA_ROOT: `os.environ.get("DATA_ROOT", ".")` like eval_loo. We use `DATA_ROOT=/data/ddimitrov/data`
  (the a330d tree is read-only for us). Agent B creates symlinks
  `$DATA_ROOT/datasets/crc/raw_zenodo -> /data/a330d/datasets/crc/raw_zenodo` and
  `$DATA_ROOT/datasets/MERFISH_mouse_brain -> /data/a330d/datasets/MERFISH_mouse_brain`.
- Slides (final): `crc_120`, `crc_210`, `crc_221`, `crc_231`, `crc_232`, `crc_242`; scored cell types
  per slide = the `cellina-pert` rows of `origin/results:results/loo_cellina_crc_DEG_50_pert_v2.csv`.
  Merfish only wired, not run.
- Model: `lotfollahi-lab/TERRA-96M` (HF cache already has it; `terra.download_pretrained`).
- Fine-tune (final): self-supervised I-JEPA LoRA on ALL cells of the slide (no labels, no holdout);
  r16 / alpha 256 / dropout 0.1 on `qkv,proj,fc1,fc2`, lr 1e-4, batch 64, 5 epochs, `save_every 1`.
  Every epoch checkpoint is exported + embedded + collapse-guarded; arms are `frozen`, `lora-ep5`
  and `lora-ep{latest_passing_epoch}` when that is neither 5 nor null. See `queue/README.md`.
- Splits: `train_loo.split_indices(adata_hvg, holdout_ct, labels_key, domains_key, holdout_domains, seed=0)`.
  test = holdout ct in holdout domain(s). Fine-tune trains on `train_idx`, validates on `val_idx` — the same
  split cellina uses. Decoder trains on `train_idx ∪ val_idx` (non-holdout) and reports test on `test_idx`
  (diagnostic ceiling only, as in the notebook).
- Perturbation: `counterfactual_analysis.get_global_perturbation_logfc(adata_hvg?, ...)` — **NO**: compute it
  on `adata_terra` (all genes) exactly as notebook cell 23, top `N_PERT_GENES=200` by |logFC|, additive shift
  on neighbour tokens only (`shift_neighbourhood`, notebook cell 25), map symbol→ensembl→token via the bundle's
  `token_dictionary.pkl`. Control cells = holdout ct in control domain.
- Embeddings: `embed_dataset(..., include_spatial_cell_emb=True, ignore_spc_tokens=True, batch_size=32)`;
  use `spatial_cell_emb` for decoding and as `obsm["latents"]`. Assert `cell_emb` unchanged, `spatial_cell_emb`
  changed after perturbation (notebook check 6).
- Decoder: `python -m terra.training.decode` with the notebook's flags, but
  `--gene-selection list --gene-list-path <file with adata_hvg.var_names>` so it decodes exactly the
  benchmark's HVGs, in that order. Train/test h5ads carry `obsm["spatial_cell_emb"]`, `layers["counts"]`.
- Merfish branch: TERRA is human-only (21,952 ENSG tokens, 0 ENSMUSG). Map `var["gene_name"].str.upper()`
  through the bundle's `ensembl_dictionary.pkl`; drop unmapped (79/1120). Coordinates: `obsm["X_spatial_coords"]
  * 0.109` → µm (verify tissue extent is a few mm; print it). Not run now.

## `common.py` contract (agents B and C code against this; agent A implements it)

```python
PX_TO_UM = {"crc": 0.12028, "merfish": 0.109}
MODEL_REPO = "lotfollahi-lab/TERRA-96M"
N_PERT_GENES = 200

def dataset_args(dataset_name) -> dict
    # ADATA_CRC_ARGS / ADATA_MERFISH_ARGS with the eval_loo defaults filled in:
    # keys n_top_genes, labels_key, domains_key, control_domains, holdout_domains

def layout(dataset_name, adata_path, holdout_ct) -> dict(paths)
    # sid       = Path(adata_path).stem
    # out_dir   = $DATA_ROOT/datasets/{crc|merfish}/{sid}/{holdout_ct}      <- where eval_loo looks (crc: .../datasets/crc/{sid}/{ct}; merfish: .../datasets/{sid}/{ct})
    # work_dir  = out_dir/terra                                             <- bundles, decoder ckpts, splits, perturbation caches
    # tok_cache = $DATA_ROOT/datasets/{crc|merfish}/{sid}/terra_tok         <- per slide, shared across holdouts
    # emb_cache = work_dir/emb_{variant}.npz
    # corr_dir  = $DATA_ROOT/datasets/{crc|merfish}/correlations

def load_dataset(dataset_name, adata_path, holdout_ct, model_dir) -> SimpleNamespace(
    adata_hvg,   # preprocess_crc / preprocess_merfish from train_loo + split_indices (sets obs['is_holdout']);
                 # NOT preprocess_spatial_features. This is the eval_loo object: layers['counts'], n_top_genes vars.
    adata_terra, # same cells as adata_hvg (inner-join on obs_names; assert equal), ALL genes surviving
                 # harmonize_adata(model_dir dicts); obs['cell_id']=obs_names (str, unique); obs has labels_key,
                 # domains_key, is_holdout; obsm['spatial'] in µm; layers['counts'] = raw counts; X CSR.
    train_idx, val_idx, test_idx,   # into adata_hvg rows == adata_terra rows
    args,        # dataset_args(...)
    holdout_ct, sid, paths,
)

def tokenize_cached(adata_terra, model_dir, tok_cache, nproc=16) -> datasets.Dataset
    # tokenize_adata(...) once per slide; load_from_disk if exists; assert len == n_obs

def model_dir() -> str   # download_pretrained(MODEL_REPO) local folder
```

## `finetuning_lora.py` CLI
```
python scripts/terra/finetuning_lora.py --dataset_name crc --adata_path .../crc_232.h5ad \
  [--epochs 5 --lr 1e-4 --batch-size 64 --seed 0 --max-cells N --max-steps N --work-dir D]
  [--from-run-dir DIR]     # skip training, export/embed/guard DIR's checkpoint_epoch_*.pt
```
Writes `{slide}/terra/lora_run/` (terra checkpoints + `ft_config.json` = the resolved config),
`{slide}/terra/lora_ep{N}/lora_bundle` + `.../emb_lora.npz` per epoch, and
`{slide}/terra/epoch_selection.json` (per-epoch guard stats, pass/fail per criterion,
`final_epoch`, `latest_passing_epoch`, loss and steps per epoch). Skip if the selection exists.
Structural checks still assert (preflight strict load, strict repack load, LoRA-only tensors changed
in every block, off-target drift within the pretrained EMA gap); the collapse guard only records.

## `inference.py` CLI
```
python scripts/terra/inference.py --dataset_name crc --adata_path ... --holdout_celltype Fibroblast \
  [--eval-celltypes CT,CT,...] --variant frozen | --variant lora --epoch N [--skip-eval]
```
model_name = `terra-frozen` or `terra-lora-ep{N}`, with no `-ho{ct}` tag: the encoder is never held
out, only the decoder is (its own `split_indices` per scored type, in that type's `work_dir`). Outputs in `out_dir`, exactly the shapes eval_loo reads:
- `{model_name}_recon_x.h5ad`: all holdout-ct cells (every domain), X = decoded counts from the unperturbed
  `spatial_cell_emb`, `obsm["latents"]` = spatial_cell_emb, obs/var copied from `adata_hvg[holdout ct]`
  (use `train_loo.save_recon_adata`).
- `{model_name}_counterfactual_x_{hd}.h5ad` per holdout domain: control cells (holdout ct ∩ control domain),
  X = decoded counts from the **perturbed** embedding, latents = perturbed spatial_cell_emb.
- `{model_name}-null_counterfactual_x_{hd}.h5ad` + `{model_name}-null_recon_x.h5ad` (copy): control cells
  decoded from the **unperturbed** re-embed. This is the reference the notebook showed matters
  (pearson 0.37 raw vs 0.53 after removing the null); eval_loo scores it as its own model row.
Counts are the decoder's NB means, unnormalised (eval_loo's get_lfc normalises; cellina saves raw too).
Then run (unless --skip-eval) for each model_name in {terra, terra-null} (or terra-frozen, terra-frozen-null):
`python scripts/eval_loo.py --dataset_name crc --adata_path ... --holdout_celltype Fibroblast --model_class terra --model_name <m> --use_cf`
→ `corr_dir/{sid}_<m>-cf_Fibroblast_CRC.json`. Check which env eval_loo needs (train_loo imports cellina
configs); if the terra env cannot import them, call the cellina env's python
(`/data/ddimitrov/software/miniforge3/envs/cellina/bin/python`) via subprocess and say so in the README.
**The only edit outside scripts/terra/:** add `"terra"` to `--model_class` choices in `scripts/eval_loo.py`.
Caches: `emb_{variant}.npz` (all cells), `ctrl_tok_{hd}` / `pert_tok_{hd}` tokenised datasets and the logfc
vector (needed by eval_terra.py), decoder ckpt `count_decoder_{variant}.pt` + metrics json.

## `eval_terra.py` CLI
```
python scripts/terra/eval_terra.py --dataset_name crc --adata_path ... --holdout_celltype Fibroblast \
  --variant frozen | --variant lora --epoch N [--universe terra2k --cellina-cf cellina-pert]
```
Notebook §8b / `GENE_SHIFT_SPEC.md` protocol B, copied from `score_gene_shift` (notebook cell 28):
per-gene L2-normalised token-embedding clouds (`return_token_embeddings=True`, slice `[:, :256]`), geomloss
Sinkhorn W2 (p=2, blur 0.01, tensorized) control vs perturbed, ranked against |log2FC| pseudobulk CP10K;
unsigned precision@50, Pearson, Spearman; universes (i) all token-covered genes with ≥ `--min-cells 20` control
cells, (ii) ∩ `adata_hvg.var_names` (the benchmark HVGs — do NOT recompute HVGs), (iii) minus perturbed genes;
3 random-gene control perturbations + chance line. Reads the caches inference.py wrote (ctrl/pert tokenised
sets, logfc); if absent, rebuild them the same way. Output `corr_dir/{sid}_{model_name}-native_Fibroblast_{hd}.json`
with one entry per universe (n_genes, precision, pearson, spearman, random_* means) and a per-gene CSV in work_dir.

## Checks every script leaves behind (`if __name__ == "__main__"` runs them; `--max-cells` for smoke)
finite embeddings; row alignment adata_hvg ↔ adata_terra ↔ tokenised set (cell_id); cell_emb unchanged /
spatial_cell_emb changed under perturbation; decoded matrix shape == (n_cells, n_top_genes) and var order ==
adata_hvg.var_names; eval_loo JSON written and loadable.

## Not doing (say so, don't build)
No CPA/scGen-style model_class branches. No changes to the notebooks or to `finetune_terra.py`.
No retraining of the other methods. The supervised block-11 fine-tune (`finetuning.py`, `--variant ft`)
and the TERRA-112M switch were dropped: the final protocol is the self-supervised LoRA arm only.
The driver is `queue/worker.sh` + `queue/launch_overnight.sh` (two GPUs, one worker each).
