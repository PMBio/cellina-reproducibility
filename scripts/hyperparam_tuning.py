# filepath: /data/a330d/projects/cellina-reproducibility-worktrees/major-loo/scripts/hyperparam_tuning.py
"""
Hyperparameter tuning helper: train multiple models across a grid of classifier/discriminator lambda values
and evaluate them on a specified holdout cell type.

- Loads and preprocesses the input AnnData using the same logic as `train_loo.py`.
- For each lambda value, trains a model (same lambda for classifier & discriminator), runs inference
  (reconstruction + counterfactuals), and evaluates the outputs.
- Saves a CSV summary with correlation, edistance, mixing index, MSE and RMSE.

Usage example:
python scripts/hyperparam_tuning.py \
  --adata_path /data2/a330d/datasets/crc/raw_zenodo/crc_242.h5ad \
  --holdout_celltype Epithelial \
"""

import os
import sys
import argparse
import numpy as np
import scanpy as sc
import cellina

# make sure scripts folder is importable when run from repo root
sys.path.append(os.path.dirname(__file__) or '.')

from train_loo import (
    preprocess_adata,
    split_indices,
    train_model,
    DEFAULT_SEED,
    ADATA_ARGS,
    DEFAULT_LABELS_KEY,
    DEFAULT_DOMAINS_KEY,
    DEFAULT_BATCH_KEY,
)

LAMBDAS = [0., 1e-5, 1e-3, 0.01, 0.1, 1, 5, 50]
MODEL_ROOT = "/data2/a330d/data/ood/trained/tuning"


def main():
    adata_path = "/data2/a330d/datasets/crc/raw_zenodo/crc_232.h5ad"
    holdout_celltype = "Endothelial"
    lambdas = LAMBDAS

    model_args = {
        "n_latent": 64,
        "use_observed_lib_size": True,
        "classifier_lambda": 1.,
        "discriminator_lambda": 1.,
        "gene_likelihood": "nb",
    }
    train_args = {
        "max_epochs": 100,
        "batch_size": 4096,
        "check_val_every_n_epoch": 1,
        "early_stopping": True,
        "early_stopping_patience": 25,
        "early_stopping_monitor": "validation_loss",
        "enable_checkpointing": True,
        "devices": [0],
    }
    plan_kwargs = {
        "lr": 1e-4,
        "normalize_losses": True,
    }

    # seed
    np.random.seed(DEFAULT_SEED)

    # load and preprocess adata
    print('Loading adata', adata_path)
    adata = sc.read(adata_path)

    n_top_genes = ADATA_ARGS.get('n_top_genes', 2000)
    n_neighbors = ADATA_ARGS.get('n_neighbors', 50)
    labels_key = ADATA_ARGS.get('labels_key', DEFAULT_LABELS_KEY)
    domains_key = ADATA_ARGS.get('domains_key', DEFAULT_DOMAINS_KEY)
    batch_key = ADATA_ARGS.get('batch_key', DEFAULT_BATCH_KEY)

    adata = preprocess_adata(adata, n_top_genes=n_top_genes, n_neighbors=n_neighbors)

    # create splits so train_model can use them
    train_idx, val_idx, test_idx = split_indices(adata, holdout_celltype, labels_key=labels_key, domains_key=domains_key, seed=DEFAULT_SEED)
    splits = (train_idx, val_idx, test_idx)

    sid = os.path.splitext(os.path.basename(adata_path))[0]

    for lam in lambdas:
        print('\n=== Lambda:', lam, '===')
        # set both classifier and discriminator to same lambda when keys exist
        model_args['classifier_lambda'] = float(lam)
        model_args['discriminator_lambda'] = float(lam)
        model_name = f"cellina_{lam}"

        save_dir = os.path.join(MODEL_ROOT, sid, holdout_celltype, model_name)
        os.makedirs(save_dir, exist_ok=True)

        # Train model (this will save model to save_dir). Use train_model from train_loo.
        model, extras = train_model(
            adata.copy(),
            'cellina',
            model_args,
            train_args,
            save_dir,
            plan_kwargs=plan_kwargs,
            batch_key=batch_key,
            labels_key=labels_key,
            domains_key=domains_key,
            splits=splits,
        )

if __name__ == '__main__':
    main()