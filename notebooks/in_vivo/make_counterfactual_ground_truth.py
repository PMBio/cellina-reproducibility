"""Precompute the model-independent half of pfish_analysis.ipynb's counterfactual analysis
(cells 1, 3, 14/15, 17's data-only parts) for reuse in the terra env, which lacks `cellina`
(needed for pfish_prep's spatial graph / spatial_x features).

Everything here comes straight from real data (raw counts, the lognorm layer) -- only the
`insert()` function in cell 17 (which calls `model.get_perturbed_expression`) is Cellina-specific
and is NOT reproduced here; that's TERRA's job in in_vivo.ipynb.

Run with the cellina-graph conda env:
    conda run -n cellina-graph python notebooks/in_vivo/make_counterfactual_ground_truth.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfish_prep as pp

seed = 0
np.random.seed(seed)
_HERE = os.path.dirname(os.path.abspath(__file__))
H5 = os.path.join(_HERE, "pfish_full.h5ad")
bw, mn = pp.DEFAULT_BANDWIDTH, pp.DEFAULT_MAX_NEIGHBOURS

# EFF9: the 9 KOs with a real direct effect on their own cancer cell (ko_inventory.csv selection,
# pfish_analysis.ipynb cell 1/14) -- selected on the cancer-cell cause, never the T-cell outcome.
EFF9 = ["MAP2K2", "IRAK1", "IRF7", "PELI1", "IRF3", "IRF5", "NFKBIA", "LBP", "MAP2K3"]

# --- cells 1/3: same preprocessing as pfish_analysis.ipynb ---
adata = sc.read_h5ad(H5)
adata = pp.filter_cells(adata, max_n_perturb=1)
adata.obs_names_make_unique()
pp.build_graph(adata, bandwidth=bw, max_neighbours=mn)
pp.label_context(adata)
idx_control, idx_target, donor_idx = pp.make_edge_swap_sets(adata, responder="T cells")

obs = adata.obs

# --- cell 17: data-only parts (everything except `insert()`, which needs the cellina model) ---
_counts_layer = adata.layers["counts"]
counts = np.asarray(_counts_layer.todense() if hasattr(_counts_layer, "todense") else _counts_layer, float)
ln = np.asarray(adata.layers["lognorm"])   # adata's own layer: sc.pp.normalize_total(target_sum=None) + log1p
pert = obs["perturbation"].astype(str).to_numpy()
nc = obs["n_counts"].to_numpy()
is_cancer = (obs["celltype2"] == "cancer").to_numpy()
is_T = (obs["celltype2"] == "T cells").to_numpy()
pc = is_cancer & (obs["n_perturb"].to_numpy() > 0)
uT = is_T & (obs["n_perturb"].to_numpy() == 0)
conn = adata.obsp[pp.CONN_ORIG_KEY].tocsr()

rng = np.random.default_rng(seed)
far = idx_control if len(idx_control) <= 6000 else rng.choice(idx_control, 6000, replace=False)


def pn(idx):
    x = np.asarray(counts[idx], float)
    return (x / (x.sum(1, keepdims=True) + 1e-8) * 1e4).mean(0)


def lfc(a, b):
    return np.log(a + 1) - np.log(b + 1)


def cd(idx):
    return np.column_stack([np.log1p(nc[idx])])


def match(a, pool):
    Cc = cd(pool)
    mu, sd = Cc.mean(0), Cc.std(0) + 1e-9
    _, mi = cKDTree((Cc - mu) / sd).query((cd(a) - mu) / sd, k=1)
    return pool[mi]


def _normalize_counts(x, eps=1e-8, scale=1e4):
    return x / (x.sum(axis=1, keepdims=True) + eps) * scale


recon = _normalize_counts(np.asarray(adata.X[far], float))   # adata.X == raw counts at this point

mean_shift = lfc(pn(np.where(uT & (np.asarray(conn.dot(pc.astype(float))).ravel() > 0))[0]), pn(far))

OBS = {}
LN_SIG = {}          # ko -> mean lognorm profile of that KO's cancer cells (the "sig" insert() uses)
DEPTH = {}           # ko -> mean real n_counts of that KO's cancer cells (for scaling the synthetic cell)
N_KC = {}
for ko in EFF9:
    kc = pc & (pert == ko)
    near = np.asarray(conn.dot(kc.astype(float))).ravel() > 0
    noth = np.asarray(conn.dot((pc & (pert != ko)).astype(float))).ravel() > 0
    t = np.where(uT & near & ~noth)[0]
    OBS[ko] = lfc(pn(t), pn(match(t, far)))
    LN_SIG[ko] = ln[kc].mean(0)
    DEPTH[ko] = float(nc[kc].mean())
    N_KC[ko] = int(kc.sum())

shared = np.mean([OBS[k] for k in EFF9], axis=0)

print("far:", len(far), "| per-KO exclusive-near-T + signature computed for:", EFF9)
for ko in EFF9:
    print(f"  {ko}: n_kc={N_KC[ko]} depth={DEPTH[ko]:.1f}")

# --- save everything the terra notebook needs, keyed by cell_id (not positional index) ---
obs_names = adata.obs_names.astype(str).to_numpy()
out = {
    "far_ids": obs_names[far],
    "recon": recon.astype(np.float32),
    "mean_shift": mean_shift.astype(np.float32),
    "shared": shared.astype(np.float32),
    "genes_154": np.array(list(adata.var_names), dtype=object),
    "EFF9": np.array(EFF9, dtype=object),
}
for ko in EFF9:
    out[f"OBS_{ko}"] = OBS[ko].astype(np.float32)
    out[f"LN_SIG_{ko}"] = LN_SIG[ko].astype(np.float32)
out_path = os.path.join(_HERE, "terra_pfish", "counterfactual_ground_truth.npz")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez(out_path, **out)

meta = {"EFF9": EFF9, "depth": DEPTH, "n_kc": N_KC, "n_far": int(len(far)), "seed": seed,
        "bandwidth": bw, "max_neighbours": mn, "h5": H5}
with open(os.path.join(_HERE, "terra_pfish", "counterfactual_ground_truth.meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("wrote", out_path)
