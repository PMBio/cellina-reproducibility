# cellina-reproducibility

Scripts and notebooks to reproduce results for `cellina`, including counterfactual LOO benchmarking, ablation studies, disentanglement benchmarking, and application analyses on the CRC spatial transcriptomics dataset.

## Repository structure

| Path | Description |
|------|-------------|
| `scripts/` | Core training, evaluation, benchmarking, and utility scripts |
| `notebooks/ablations/` | Lambda-sweep ablation studies for model components |
| `notebooks/application/` | End-to-end CRC case study with pathway and counterfactual analysis |
| `notebooks/disentanglement/` | Latent disentanglement benchmark |
| `notebooks/loo_benchmarks/` | Leave-one-celltype-out benchmark suite |
| `notebooks/neighb_perts/` | Gene-range neighborhood perturbation studies |
| `notebooks/spatial_ablations/` | Spatial loss and link-prediction-weight ablations |
| `environments/` | Conda environment files |

## Setup

Each baseline or model variant has its own conda environment:

```bash
conda env create -f environments/<env>.yml
```

| Environment | Used for |
|-------------|----------|
| `cellina.yml` | Main Cellina training and evaluation |
| `cellina_graph.yml` | CellinaGraph (+ GCN) experiments |
| `cpa_env.yml` | CPA |
| `spatialprop_env.yml` | SpatialProp |
| `mintflow_env.yml` | MintFlow |
| `cellina.yml` | scGEN (via pertpy) |

## Sections
1. Ablation study of each component of `cellina` — classifier, discriminator and edge loss
2. Marginal log likelihood for unseen region/cell type combination
3. Benchmark measuring disentanglement on 3 datasets
4. Counterfactual prediction of cell states in unseen niches