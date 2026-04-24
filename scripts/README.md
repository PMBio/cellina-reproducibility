# Scripts

## Training & Evaluation

| Script | Description |
|--------|-------------|
| `train_loo.py` | Train models with one cell type held out; saves model, reconstructions, and counterfactuals to `data/ood/trained/` |
| `eval_loo.py` | Evaluate LOO predictions; computes Pearson/Spearman, precision, direction match, mixing index, RMSE, and energy distance; writes per-holdout JSON |
| `train_parallel.py` | Launch `train_loo.py` for multiple slides/cell types concurrently; logs to `parallel_logs/` |
| `eval_parallel.py` | Run `eval_loo.py` sequentially for all configured slide/holdout/model combinations |
| `train_mintflow.py` | Train the MintFlow baseline on CRC or MERFISH data |

## Benchmarking

| Script | Description |
|--------|-------------|
| `benchmark_pipeline.py` | SCIB benchmark: splits data by sample; trains and evaluates PCA, scVI, SCANVI, scVIVA, Cellina, and Cellina-MMD; writes metrics |
| `run_benchmark_pipeline.py` | Orchestrator that calls `BenchmarkPipelineRunner` across datasets |

## Analysis utilities

| Script | Description |
|--------|-------------|
| `counterfactual_analysis.py` | Counterfactual metrics library: mixing index, Pearson/Spearman LFC correlation, RMSE, energy distance, baseline delta |
| `perturb_utils.py` | CRC data loading, coarse-type label mapping, pseudobulk LFC computation, Pearson/Spearman helpers |
| `plotting.py` | Confusion matrix and ROC curve plotting (sklearn-based) |
| `profiler.py` | `TrainingProfiler` class: samples RAM/VRAM during training; outputs timing and memory CSV |
| `utils.py` | `set_seed`, `evaluate_models` cross-validation helper, `plot_results` lambda-sweep visualizer |
| `_labels_to_coarse.py` | Dictionary mapping fine-grained cell type labels to coarse types |

## `configs/`

One config file per model/dataset combination, used as argument namespaces by training scripts.

| Config | Description |
|--------|-------------|
| `adata_crc_config.py` | CRC CosMx preprocessing arguments |
| `adata_merfish_config.py` | MERFISH mouse brain preprocessing arguments |
| `cellina_config.py` | CellinaModel training hyperparameters |
| `cellina_graph_config.py` | CellinaGraph (+ GCN) hyperparameters |
| `cellina_ablated_config.py` | Ablated Cellina variants |
| `cellina_mmd_config.py` | Cellina with MMD loss |
| `cpa_config.py` | CPA baseline hyperparameters |
| `scgen_config.py` | scGen baseline hyperparameters |
