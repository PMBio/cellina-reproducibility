# Adatas preprocessing defaults derived from notebooks/counterfactuals.ipynb

ADATA_ARGS = {
    # number of HVGs to keep
    "n_top_genes": 2000,
    # spatial neighbor radius/knn
    "n_neighbors": 200,
    # minimum counts filtering
    "min_counts_cells": 3,
    "min_counts_genes": 3,
    "labels_key": 'coarse_type',
    "domains_key": 'typ_clean',
    "batch_key": 'sid',
    "control_domains": ['REF'],
    "holdout_domains": ['CRC'],
}
