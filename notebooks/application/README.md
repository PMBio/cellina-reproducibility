Application analysis for the CRC spatial transcriptomics dataset using Cellina.

## Files

| File | Type | Role |
|------|------|------|
| `01_data_prep.ipynb` | Notebook | Data prep, model training, Hotspot module detection, microenvironment labelling, global counterfactuals — saves outputs for downstream steps |
| `02_pathway_analysis.ipynb` | Notebook | Loads outputs from 01; pathway enrichment (PROGENy + Hallmark) on Hotspot modules; pathway-guided neighbourhood perturbations (`make_neighbor_perturbation` / `get_perturbed_expression`) for a single slide |
| `03_subdomain_counterfactuals.py` | Script | Loops over slides, runs subdomain edge-swapping counterfactuals per cell type × microenvironment, saves per-slide correlation CSVs, produces dumbbell plots |
| `04_fibroblast_IGF2_plots.ipynb` | Notebook | Trains a separate OOD model (CRC fibroblasts held out), generates counterfactual IGF2 spatial expression maps |

## Workflow

### Main pipeline (01 → 02 → 03)

Run `01_data_prep.ipynb` first for each slide. It saves:
- `{slide_id}/output/adata_with_microenv.h5ad` — annotated AnnData with latent representations and microenvironment labels
- `{slide_id}/output/hotspot.pkl` — fitted Hotspot object
- `{slide_id}/output/results.pkl` — per-cell-type AnnData dict with counterfactual expressions in `.uns`
- Model checkpoint at `data/cellina-reproducibility/application/{slide_id}/`

`02_pathway_analysis.ipynb` loads those outputs and runs pathway-level analyses for a single slide.

`03_subdomain_counterfactuals.py` processes one or more slides in batch and aggregates results:

```bash
python 03_subdomain_counterfactuals.py --slides crc_210 crc_xxx
```

Saves per-slide CSVs to `../../results/microenvironments_{slide_id}.csv` and figures to `../../figures/application/`.

### Standalone (04)

`04_fibroblast_IGF2_plots.ipynb` is independent of the main pipeline. It trains its own model with an OOD split (fibroblasts in CRC regions held out) and visualises IGF2 counterfactual expression spatially. Figures saved to `../../figures/`.

## Inputs

Raw data: `/data/a330d/datasets/crc/raw_zenodo/{slide_id}.h5ad`

## Dependencies

All files add `../../scripts/` to `sys.path` automatically.
