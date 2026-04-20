# Neighbor Perturbation LOO Benchmark

Leave-one-out (LOO) evaluation of neighbor perturbation: each CRC cell type is held out, a model is retrained without it, and predictions under CRC spatial context are compared against ground truth.

## Scripts

### `cellina_loo.py`
Runs the Cellina LOO benchmark for one or more held-out cell types. Trains the model, then evaluates two inference strategies: neighbor perturbation and counterfactual swapping.

Key args:
- `--slide_id` (default: 242)
- `--groupby` — comma-separated holdout cell types
- `--top_n_perturb` (default: 100) — genes used for neighbor perturbation
- `--top_n` (default: 50) — genes used for metric evaluation
- `--out_dir` (default: `results/perturb_benchmark`)
- `--min_cells`, `--batch_size`, `--max_epochs`

### `perturb_gene_range.py`
Sweeps perturbation performance vs. number of perturbed genes for a single slide. Writes results to `results/gene_range/<slide_id>/results.csv`.

Key args:
- `--slide_id` (required)
- `--gpu` (default: 0)
- `--out_dir` (default: `results/gene_range`)
- `--batch_size`, `--min_cells`, `--top_n`

### `run_gene_range.sh`
Submits `perturb_gene_range.py` over all CRC slides in parallel (GPU 1). Accepts `-n MAX_CONCURRENT` (default: 2).

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `perturb_gene_range.ipynb` | Interactive walkthrough of perturbation performance vs. number of perturbed genes for a single slide |
| `perturb_gene_range_analysis.ipynb` | Multi-slide analysis: loads gene-range results across all slides and plots aggregated metrics |
