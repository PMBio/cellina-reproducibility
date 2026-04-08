# Neighbor Perturbation LOO Benchmark

Leave-one-out (LOO) evaluation of neighbor perturbation: each CRC cell type is held out, a model is retrained without it, and predictions under CRC spatial context are compared against ground truth.

## Scripts

### `cellina_loo.py`
Runs the Cellina LOO benchmark for one or more held-out cell types. Trains the model, then evaluates two inference strategies: neighbor perturbation and counterfactual swapping.

Key args:
- `--slide_id` (default: 242)
- `--groupby` — comma-separated holdout cell types
- `--top_n_perturb` (default: 100) — genes used for neighbor perturbation
- `--top_n` (default: 100) — genes used for metric evaluation
- `--out_dir` (default: `results/perturb_benchmark`)
- `--min_cells`, `--batch_size`, `--max_epochs`

### `spatialprop_loo.py`
Same LOO benchmark for the SpatialProp GNN baseline. Accepts identical arguments to `cellina_loo.py`.

### `run_perturb_loo.sh`
Runs `cellina_loo.py` (GPU 1) and `spatialprop_loo.py` (GPU 0) in parallel, then merges their result CSVs. Accepts the same arguments forwarded to both scripts.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `perturb_loo.ipynb` | Interactive walkthrough of the Cellina LOO evaluation for a single held-out cell type |
| `spatialprop_loo.ipynb` | Same walkthrough for the SpatialProp GNN baseline |
| `perturb_loo_plot.ipynb` | Reads the merged benchmark CSV and plots per-cell-type metrics comparing both methods |
| `perturb_gene_range.ipynb` | Sensitivity analysis: sweeps the number of perturbed genes to assess its effect on performance |
