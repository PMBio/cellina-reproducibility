"""Stage 2a: raw slide -> TERRA-harmonised -> terra's own 2000 seurat_v3 HVGs.

Writes `{sid}_terra2k.h5ad` (raw counts in .X, all obs intact) + `{sid}_terra2k_genes.txt`,
the gene universe of the terra_og track (terra_og_eval.py reads both).  CPU only.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import scanpy as sc
import scipy.sparse as sp

import common

p = argparse.ArgumentParser()
p.add_argument("--dataset_name", default="crc")
p.add_argument("--adata_path", required=True)
p.add_argument("--n_top_genes", type=int, default=2000)
p.add_argument("--out_dir", default=None, help="default: next to --adata_path")
p.add_argument("--force", action="store_true")
a = p.parse_args()

sid = Path(a.adata_path).stem
out_dir = Path(a.out_dir) if a.out_dir else Path(a.adata_path).parent
h5ad_out = out_dir / f"{sid}_terra2k.h5ad"
genes_out = out_dir / f"{sid}_terra2k_genes.txt"
if h5ad_out.exists() and not a.force:
    raise SystemExit(f"{h5ad_out} exists; pass --force to overwrite")

raw = sc.read_h5ad(a.adata_path)
raw.obs_names_make_unique()
print(f"[raw] {raw.n_obs:,} cells x {raw.n_vars:,} genes")

adata = common._terra_side(raw, a.dataset_name, common.model_dir())
del raw
print(f"[harmonised] {adata.n_obs:,} cells x {adata.n_vars:,} genes")

from terra.training.decode import compute_hvgs

hvgs = compute_hvgs(adata, n_top_genes=a.n_top_genes, flavor="seurat_v3", counts_layer="counts")
assert len(hvgs) == a.n_top_genes, f"{len(hvgs)} HVGs != {a.n_top_genes}"
assert set(hvgs) <= set(adata.var_names), "HVGs not a subset of the harmonised panel"

out = adata[:, list(hvgs)].copy()
out.X = sp.csr_matrix(out.layers["counts"])   # counts, in HVG order
del out.layers["counts"]
d = out.X.data
assert np.array_equal(d, np.rint(d)), ".X is not integer-valued counts"
print(f"[terra2k] {out.n_obs:,} cells x {out.n_vars:,} genes")

out_dir.mkdir(parents=True, exist_ok=True)
out.write_h5ad(h5ad_out, compression="gzip")
genes_out.write_text("\n".join(map(str, hvgs)) + "\n")
print(f"wrote {h5ad_out} ({h5ad_out.stat().st_size / 1e6:.0f} MB)\nwrote {genes_out}")
