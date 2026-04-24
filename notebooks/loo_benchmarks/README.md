# LOO Benchmarks

Leave-one-celltype-out (LOO) benchmark suite comparing Cellina, SpatialProp, and MintFlow. Each method retrains on CRC and MERFISH data with one cell type held out, then scores counterfactual predictions against ground truth.

## Files

| File | Description |
|------|-------------|
| `cellina_node_pert.ipynb` | Cellina LOO evaluation with node perturbation (200 genes); computes Pearson/Spearman, precision, direction match, and energy distance across CRC and MERFISH slides |
| `spatialprop/` | SpatialProp training (`spatialprop_train_loo.py`) and evaluation (`spatialprop_eval_loo.py`) |
| `mintflow/` | MintFlow training (`mintflow.ipynb`) and evaluation (`mintflow_eval.ipynb`) |
