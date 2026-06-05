# Holdout Domain Experiment

Evaluates how well Cellina and SpatialProp predict gene expression changes when
an entire spatial domain (TVA) is held out from training.  The control domain is
REF.

## Datasets

| Dataset | Slides / samples | Path |
|---------|-----------------|------|
| CRC (colorectal cancer, Zenodo) | `crc_120`, `crc_210`, `crc_221`, `crc_231`, `crc_242` | `/data2/a330d/datasets/crc/raw_zenodo/` |

**Preprocessing**: 2 000 HVGs, normalised to 1×10⁴ counts, log-transformed.
Spatial neighbours: 200.  Holdout split: entire TVA domain (`holdout_full_domain=True`).

Holdout cell types evaluated:

- **CRC**: Endothelial, Epithelial, Fibroblast, Myeloid, T\_cell
- **MERFISH**: glutamatergic neuron, oligodendrocyte, astrocyte, GABAergic neuron, endothelial cell

## Files

| File | Purpose |
|------|---------|
| `cellina_tva.ipynb` | Train & evaluate Cellina (standard + GAT) and produce comparison plots |
| `spatialprop_train.py` | Train SpatialProp GNN models (run before eval) |
| `spatialprop_eval.py` | Evaluate pre-trained SpatialProp models and write results CSV |

## Models

| Model | Key hyperparameters |
|-------|-------------------|
| **Cellina** | `n_latent=64`, `gene_likelihood=nb`, `n_layers=2`, 100 epochs, lr=1e-3 |
| **Cellina-GAT** | Same as above with spatial graph connectivity |
| **SpatialProp** | GNN, `k_hop=2`, `augment_hop=2`, `loss=weightedl1`, 100 epochs, lr=1e-3 |

## How to Run

### 1. Train SpatialProp

```bash
python spatialprop_train.py
```

Trains one GNN per slide.  Saves models to:
```
./output/{slide_id}_ood/{slide_id}_{holdout_ct}_ood_expression_2hop_2augment_expression_none/weightedl1_1en03/model.pth
```

### 2. Evaluate SpatialProp

```bash
python spatialprop_eval.py
```

Writes results to `../../results/ood_spatialprop_{dataset}_DEG_50.csv`.

### 3. Run Cellina notebook

Open `cellina_tva.ipynb` and run all cells.  Writes results to
`../../results/ood_cellina_{dataset}_DEG_50.csv` and produces
`cellina_tva_metrics.svg`.

Set `DATASET_NAME = "crc"` or `"merfish"` in the notebook to switch datasets.
