All-slides version of `notebooks/application/{01,02}`: the same Hotspot + PROGENy
microenvironment analysis run on all 6 benchmark slides jointly instead of crc_210 alone, to
separate shared programmes from per-slide effects.

| file | role |
|---|---|
| `01_data_prep.ipynb` | load + join 6 slides, per-slide spatial graph, train, Hotspot modules, project to all CRC cells, counterfactuals |
| `02_pathway_analysis.ipynb` | PROGENy/Hallmark on modules, module -> pathway assignment, dotplot, composition bars, counterfactual UMAP, perturbation validation |
| `joint.py` | shared helpers imported by both notebooks and `sweep.py`. `python joint.py` runs `demo()`, an assert-based self-check |
| `sweep.py` | `HOTSPOT_TOP_K` x `HOTSPOT_MIN_GENES` sweep; `prep` / `sweep` / `all`, both stages resumable |

Outputs go to `output/` (~9 GB) and `../../figures/application_all_slides/`, both gitignored.
`N_JOBS` (default 24) sets Hotspot's worker count; `CRC_RAW_DIR` overrides the raw data path.
`RESUME=1` makes 01 reuse `output/model_final` instead of retraining.

## Configuration

| | value | why |
|---|---|---|
| slides | crc_120, 210, 221, 231, 232, 242 | the benchmark set of `scripts/train_parallel.py`; crc_110 excluded there and here |
| cells / genes | 2,367,748 x 2,800 | 18,877 genes after inner join, then the HVG rule below |
| HVG rule | in the top 3,000 of **>= 3 of 6 slides** | panel size is an outcome of cross-slide reproducibility, not a fixed number. `>= k slides`: 1 -> 7700, 2 -> 4483, **3 -> 2800**, 4 -> 1714, 5 -> 901, 6 -> 402 |
| `batch_key` | `sid` (slide) | 5 patients / 6 slides |
| spatial graph | per slide via `spatial_neighbors(library_key='sid')` | block-diagonal, so `(C@X)[slide_i] == C_i @ X_i`. Verified: 20.0 edges/cell, **0 cross-slide edges** |
| splits | 90/10 train/val, **no test holdout** | nothing here evaluates held-out prediction; a holdout would only remove ~237k cells from training |
| latents | `give_mean=True` | **essential — see Gotchas** |
| Hotspot fit | 33,000 CRC cells per slide = **198,000** | slide-balanced, so no slide dominates module detection |
| `HOTSPOT_TOP_K` | 750 | 2,340 genes pass FDR<0.05, so 750 is a free choice, not a cap |
| `HOTSPOT_MIN_GENES` | 25 | the NF-kB module is 37 genes; a 100-gene floor merges it away |
| projection | all 1,270,989 CRC-region cells | Hotspot's KNN is in *latent* space, not physical, so fitting on a subsample and projecting is valid |

## Result — both programmes reproduce, and both are cross-slide

`max enrich` is the module's largest slide share **divided by that slide's baseline share of
the 1.27M CRC cells** (1.0 = as expected). Baselines are unequal — 120: 44.4%, 210: 26.1%,
221: 10.7%, 231: 5.3%, 232: 6.5%, 242: 7.1% — so raw shares are not interpretable and 1/6 is
*not* the right null for the projected cells.

| module | genes | cells | max enrich | slide | TGFb | NFkB | MAPK | NF-kB genes |
|---|---|---|---|---|---|---|---|---|
| **CRC1** | 283 | 409,514 | **1.28** | 242 | **5.02** (8.8e-6) | ns | -3.36 (5.4e-3) | 2/27 |
| CRC2 | 64 | 182,331 | 1.41 | 120 | ns | ns | ns | 0 |
| CRC3 | 68 | 65,074 | 9.72 | 242 | ns | ns | ns | 0 |
| CRC4 | 33 | 95,020 | 7.51 | 231 | ns | ns | ns | 0 |
| **CRC5** | 37 | 80,328 | **1.76** | 231 | ns | **12.53** (1.1e-31) | **6.43** (1.0e-9) | **25/27** |
| CRC6 | 35 | 89,315 | 1.80 | 120 | ns | ns | ns | 0 |
| CRC7 | 56 | 159,599 | 1.54 | 210 | ns | ns | ns | 0 |
| CRC8 | 64 | 98,961 | 3.55 | 210 | ns | ns | ns | 0 |
| CRC9 | 28 | 90,847 | 3.72 | 221 | ns | ns | 3.42 (4.4e-3) | 0 |

82 of the 750 genes are unassigned. PROGENy activity; `ns` = padj >= 0.05.

* **CRC1 = TGFb / EMT.** Largest module, most slide-balanced, and significantly
  MAPK-**negative** — a real contrast with CRC5.
* **CRC5 = NF-kB.** 25 of 27 canonical NF-kB genes (`NFKBIA CXCL8 CXCL5 CCL4 IL1B PTGS2
  ICAM1 SOD2 BCL2A1 SOCS3 TNFAIP2 MMP1/3/10/12 OSM S100A8 ...`), enrichment 1.76 —
  comparable to CRC1, nothing like the single-slide modules.
* **CRC3, CRC4, CRC8, CRC9 are single-slide effects** (enrichment 3.6-9.7). Report them as
  such. This is the question the single-slide analysis structurally could not answer.
* The `(MAPK)` dotplot label lands on CRC9 even though CRC5 has the larger MAPK signal:
  `module_pathway_assignment` is greedy and one-pathway-per-module, so CRC5 is claimed by
  NF-kB first. A labelling artefact of the rule, not of the data.

Perturbation validation: CRC1/TGFb Pearson 0.132, Spearman 0.066 (n=6,000, 304 perturbation
genes); CRC5/NFkB 0.815 / 0.808 (n=5,174, 773 genes). Do **not** read this as the model
handling NF-kB better — CRC5 gets 2.5x more perturbation genes, and the metric scores the
top-100 control-vs-target DEGs, which for CRC5 *are* the PROGENy NF-kB targets being
manipulated (partly circular), while CRC1's top DEGs are structural collagens outside the
TGFb target set.

Sweep finding (`sweep.py`): **`min_genes` is the only knob that matters** — `m=50` and `m=100`
are identical at every `top_k` (3 modules); `m=25` resolves the small programmes. Its module
counts predate the `give_mean` fix below, so re-run it for an apples-to-apples grid.

## Gotchas

1. **`give_mean=True` on `get_latent_representation`.** cellina defaults to
   `give_mean=False`, i.e. it **samples** the posterior (scvi's own default is True). Hotspot
   builds its KNN on `cellina_spatial`, so a sampled latent makes the modules irreproducible:
   sampled -> 6 modules with the NF-kB genes absorbed into the EMT module; posterior mean ->
   9 modules with NF-kB resolved at padj 1e-31. Also affects the single-slide notebooks.
2. **`joint.spatial_features_lowmem` replaces `compute_spatial_features`.** Upstream
   normalises *after* the matmul: `result.multiply(denom)` on a sparse matrix returns COO, so
   at 2.4M x 2,800 the peak is ~150 GB of transients and the process is OOM-killed.
   Row-scaling C first — `diag(1/d) @ C @ X` instead of `(C @ X) * (1/d)` — is the same
   quantity in one float32 matmul. `joint.demo()` asserts the two agree to 1e-5.
3. **The counterfactual UMAP needs per-slide centring.** Neither
   `get_normalized_expression` nor `get_counterfactual_expression` takes `transform_batch`, so
   every arm is decoded with its own batch covariate. Uncentred, slide explained
   **eta^2 = 0.850** of the embedding and the arm contrast **0.011** — a batch plot.
   Centred: slide 0.006, arm 0.141. Even then it is a diffuse cloud, and two-thirds of its
   cells are the same control fibroblasts decoded twice; the logFC scatter is the more
   trustworthy counterfactual figure.
4. `cellina.CellinaModel` was renamed to `Cellina` upstream before v0.99.1, so the committed
   single-slide notebooks do not import as-is. Here: `from cellina import Cellina as
   CellinaModel`.

## Not done

* The NF-kB / EMT split is tested across **re-runs**, not **retrains**. A few training seeds
  is the real stability test before this becomes a figure.
* No all-slides equivalent of `03_subdomain_counterfactuals.py` or
  `04_fibroblast_IGF2_plots.ipynb`; only 01 and 02 were in scope.
