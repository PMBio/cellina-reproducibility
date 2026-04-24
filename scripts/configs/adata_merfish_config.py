ADATA_ARGS = {
    # number of HVGs to keep
    "n_top_genes": 1120,
    # spatial neighbor radius/knn
    "n_neighbors": 200,
    # minimum counts filtering
    "min_counts_cells": 3,
    "min_counts_genes": 3,
    "labels_key": 'cell_type',
    "domains_key": 'major_brain_region',
    "batch_key": 'brain_section_label',
    "control_domains": ['Thalamus'],
    "holdout_domains": ['Isocortex', 'Fiber_tracts'],
}
