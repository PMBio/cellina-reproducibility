"""Precompute the pfish_analysis.ipynb train/val/test split, by cell_id, for reuse outside the
cellina-graph env (which the terra conda env doesn't have `cellina` installed in).

Mirrors pfish_analysis.ipynb cells 1, 3, 9 exactly: same H5, same pfish_prep calls, same seed,
same train_test_split. Run with the cellina-graph conda env:

    conda run -n cellina-graph python notebooks/in_vivo/make_decoder_split.py
"""
import json
import os
import sys

import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfish_prep as pp

seed = 0
np.random.seed(seed)
H5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pfish_full.h5ad")
bw, mn = pp.DEFAULT_BANDWIDTH, pp.DEFAULT_MAX_NEIGHBOURS

adata = sc.read_h5ad(H5)
adata = pp.filter_cells(adata, max_n_perturb=1)
adata.obs_names_make_unique()

pp.build_graph(adata, bandwidth=bw, max_neighbours=mn)
pp.label_context(adata)
idx_control, idx_target, donor_idx = pp.make_edge_swap_sets(adata, responder="T cells")

# pfish_analysis.ipynb cell 9: hold idx_target out of train/val entirely, split the rest 90/10.
trainval = np.setdiff1d(np.arange(adata.n_obs), idx_target)
train_idx, val_idx = train_test_split(trainval, test_size=0.1, random_state=seed, shuffle=True)
print(f"train={len(train_idx)}  val={len(val_idx)}  held-out target(test)={len(idx_target)}  "
      f"control(unused by this split)={len(idx_control)}")

obs_names = adata.obs_names.astype(str).to_numpy()
out = {
    "train_ids": obs_names[train_idx].tolist(),
    "val_ids": obs_names[val_idx].tolist(),
    "test_ids": obs_names[idx_target].tolist(),          # idx_target: held-out perturbed-niche T cells
    "control_ids": obs_names[idx_control].tolist(),       # far T cells, not used in this split (context only)
    "meta": {"seed": seed, "bandwidth": bw, "max_neighbours": mn, "h5": H5,
             "n_obs_after_filter_cells": int(adata.n_obs)},
}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terra_pfish", "decoder_split.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f)
print("wrote", out_path)
