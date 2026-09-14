"""Shared data loading / path layout for the TERRA LOO pipeline (PIPELINE_SPEC.md).

One job: hand every TERRA script the *same* objects the cellina benchmark uses
(`train_loo.preprocess_*` + `train_loo.split_indices`) plus a TERRA-harmonised
all-genes twin with identical rows.

Nothing here is TERRA-specific except `harmonize_adata` / `tokenize_adata`.
"""
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scanpy as sc
import scipy.sparse as sp

_SCRIPTS = str(Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

PX_TO_UM = {"crc": 0.12028, "merfish": 0.109}
MODEL_REPO = "lotfollahi-lab/TERRA-96M"
N_PERT_GENES = 200

DATA_ROOT = os.environ.get("DATA_ROOT", ".")

# eval_loo.py:138-144 defaults, kept here so dataset_args() resolves exactly as it does.
_DEFAULTS = {
    "n_top_genes": 2000,
    "labels_key": "coarse_type",
    "domains_key": "typ",
    "control_domains": ["REF"],
    "holdout_domains": ["CRC"],
}
_KEYS = list(_DEFAULTS)


def dataset_args(dataset_name):
    """ADATA_{CRC,MERFISH}_ARGS with eval_loo's defaults filled in."""
    from configs.adata_crc_config import ADATA_ARGS as CRC
    from configs.adata_merfish_config import ADATA_ARGS as MERFISH

    cfg = {"crc": CRC, "merfish": MERFISH}[dataset_name]
    return {k: cfg.get(k, _DEFAULTS[k]) for k in _KEYS}


def layout(dataset_name, adata_path, holdout_ct):
    sid = Path(adata_path).stem
    # eval_loo.py:27-28: crc lives under datasets/crc/, merfish directly under datasets/.
    base = Path(DATA_ROOT) / ("datasets/crc" if dataset_name == "crc" else "datasets")
    out_dir = base / sid / holdout_ct
    work_dir = out_dir / "terra"
    return {
        "sid": sid,
        "out_dir": out_dir,
        "work_dir": work_dir,
        "tok_cache": base / sid / "terra_tok",
        "emb_cache": lambda variant: work_dir / f"emb_{variant}.npz",
        "corr_dir": Path(DATA_ROOT) / "datasets" / dataset_name / "correlations",
    }


def model_dir():
    from terra import download_pretrained

    return download_pretrained(MODEL_REPO)


def _terra_side(raw, dataset_name, model_dir_path):
    """raw (all genes, already row-subset to the HVG object's cells) -> harmonised TERRA adata."""
    from terra.inference import harmonize_adata

    if dataset_name == "merfish":
        # var index is ENSMUSG; TERRA's vocabulary is human ENSG only, so go through
        # the upper-cased symbol and keep what the bundle's dictionary knows.
        import pickle

        ens = pickle.load(open(f"{model_dir_path}/ensembl_dictionary.pkl", "rb"))
        raw.var_names = raw.var["gene_name"].astype(str).str.upper().to_numpy()
        raw.var_names_make_unique()
        keep = np.array([g in ens for g in raw.var_names])
        print(f"[merfish] {keep.sum()}/{raw.n_vars} gene symbols map into the TERRA dictionary")
        raw = raw[:, keep].copy()
        coords = raw.obsm["X_spatial_coords"]
    else:
        coords = raw.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(dtype=np.float64)

    raw.obsm["spatial"] = np.asarray(coords, dtype=np.float64) * PX_TO_UM[dataset_name]
    ext = np.ptp(raw.obsm["spatial"], axis=0)
    print(f"[terra] tissue extent: {ext[0]:.0f} x {ext[1]:.0f} um")

    if sp.issparse(raw.X):
        raw.X = raw.X.tocsr()      # the tokenizer iterates rows
    raw.layers["counts"] = raw.X.copy()
    raw = harmonize_adata(
        raw,
        gene_mapping_dict_file_path=f"{model_dir_path}/ensembl_dictionary.pkl",
        gene_occurrence_count_file_path=f"{model_dir_path}/gene_count_dictionary.pkl",
    )
    if sp.issparse(raw.X):
        raw.X = raw.X.tocsr()
    raw.layers["counts"] = raw.X.copy()     # re-sync after harmonize's gene/cell filtering
    return raw


def load_dataset(dataset_name, adata_path, holdout_ct, model_dir):
    """-> SimpleNamespace(adata_hvg, adata_terra, train_idx, val_idx, test_idx, args, ...)."""
    from train_loo import preprocess_crc, preprocess_merfish, split_indices

    args = dataset_args(dataset_name)
    paths = layout(dataset_name, adata_path, holdout_ct)

    raw = sc.read_h5ad(adata_path)
    raw.obs_names_make_unique()                 # so raw and the preprocessed copy share names
    prep = preprocess_crc if dataset_name == "crc" else preprocess_merfish
    adata_hvg = prep(raw.copy(), n_top_genes=args["n_top_genes"],
                     labels_key=args["labels_key"], domains_key=args["domains_key"])
    print(f"[hvg] {adata_hvg.n_obs:,} cells x {adata_hvg.n_vars:,} genes")

    if dataset_name == "merfish":
        # preprocess_merfish scores/filters on .raw counts; mirror that for the TERRA side.
        raw.X = raw.raw.X.copy()
    adata_terra = _terra_side(raw[adata_hvg.obs_names].copy(), dataset_name, model_dir)
    del raw

    # harmonize_adata can drop cells (min_genes_per_cell); keep the two objects row-identical.
    if adata_terra.n_obs != adata_hvg.n_obs:
        print(f"[align] harmonize dropped {adata_hvg.n_obs - adata_terra.n_obs} cells; subsetting adata_hvg")
        adata_hvg = adata_hvg[adata_terra.obs_names].copy()
    adata_terra = adata_terra[adata_hvg.obs_names].copy()
    assert (adata_hvg.obs_names == adata_terra.obs_names).all(), "row alignment broken"

    train_idx, val_idx, test_idx = split_indices(
        adata_hvg, holdout_ct, labels_key=args["labels_key"],
        domains_key=args["domains_key"], holdout_domains=args["holdout_domains"], seed=0)

    for k in (args["labels_key"], args["domains_key"], "is_holdout"):
        adata_terra.obs[k] = adata_hvg.obs[k].to_numpy()
    adata_terra.obs["cell_id"] = adata_terra.obs_names.astype(str)
    assert adata_terra.obs["cell_id"].is_unique

    print(f"[split] n_obs={adata_hvg.n_obs} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    return SimpleNamespace(adata_hvg=adata_hvg, adata_terra=adata_terra,
                           train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                           args=args, holdout_ct=holdout_ct, sid=paths["sid"], paths=paths)


def tokenize_cached(adata_terra, model_dir, tok_cache, nproc=16):
    from datasets import load_from_disk
    from terra.inference import tokenize_adata

    tok_cache = Path(tok_cache)
    if tok_cache.exists():
        tok = load_from_disk(str(tok_cache))
    else:
        tok_cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = tok_cache.parent / "tok_tmp"
        tok = tokenize_adata(adata_terra, model_dir, str(tmp), nproc=nproc)
        tok.save_to_disk(str(tok_cache))
        shutil.rmtree(tmp, ignore_errors=True)
    assert len(tok) == adata_terra.n_obs, f"tokenised {len(tok)} rows vs {adata_terra.n_obs} cells"
    return tok


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="crc")
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", default="Fibroblast")
    p.add_argument("--tokenize", action="store_true")
    a = p.parse_args()

    md = model_dir()
    d = load_dataset(a.dataset_name, a.adata_path, a.holdout_celltype, md)
    print("args:", d.args)
    print("paths:", {k: str(v) for k, v in d.paths.items() if not callable(v)})
    print("adata_hvg:", d.adata_hvg.shape, "| adata_terra:", d.adata_terra.shape)
    print(d.adata_terra.obs[d.args["domains_key"]].value_counts())
    assert d.adata_hvg.n_vars == d.args["n_top_genes"]
    if a.tokenize:
        tok = tokenize_cached(d.adata_terra, md, d.paths["tok_cache"])
        print("tokenised:", len(tok), list(tok.features))
    print("OK")
