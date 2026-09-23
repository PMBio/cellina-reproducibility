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

PX_TO_UM = {"crc": 0.12028, "merfish": 1}
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


def layout(dataset_name, adata_path, holdout_ct, universe="benchmark"):
    sid = Path(adata_path).stem
    # eval_loo.py:27-28: crc lives under datasets/crc/, merfish directly under datasets/.
    base = Path(DATA_ROOT) / ("datasets/crc" if dataset_name == "crc" else "datasets")
    out_dir = base / sid / holdout_ct
    work_dir = out_dir / "terra"
    # terra2k keeps a different cell set, so neither the tokens nor the embeddings are reusable.
    uni = "" if universe == "benchmark" else f"_{universe}"
    return {
        "sid": sid,
        "out_dir": out_dir,
        "work_dir": work_dir,
        "tok_cache": base / sid / f"terra_tok{uni}",
        "emb_cache": lambda variant: work_dir / f"emb_{variant}{uni}.npz",
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


def terra2k_path(sid):
    """The shift-path h5ad (terra's own harmonised HVGs) + its gene list.

    Not a sibling of adata_path: raw_zenodo is a read-only symlink (scripts/terra/make_terra2k.py).
    """
    d = Path(DATA_ROOT) / "datasets/crc/raw_zenodo_terra2k"
    return d / f"{sid}_terra2k.h5ad", d / f"{sid}_terra2k_genes.txt"


def load_dataset(dataset_name, adata_path, holdout_ct, model_dir, universe="benchmark"):
    """-> SimpleNamespace(adata_hvg, adata_terra, train_idx, val_idx, test_idx, args, ...).

    universe="benchmark": the HVG object is `preprocess_*(raw slide)` -- what every other
    method is scored on.  universe="terra2k": it is `preprocess_crc({sid}_terra2k.h5ad)`,
    i.e. terra's own 2000 harmonised HVGs (2000 in -> 2000 out).  The TERRA side is the
    full-panel slide either way, so the encoder still tokenises the whole panel.
    """
    from train_loo import preprocess_crc, preprocess_merfish, split_indices

    args = dataset_args(dataset_name)
    paths = layout(dataset_name, adata_path, holdout_ct, universe)

    raw = sc.read_h5ad(adata_path)
    raw.obs_names_make_unique()                 # so raw and the preprocessed copy share names
    prep = preprocess_crc if dataset_name == "crc" else preprocess_merfish
    if universe == "terra2k":
        assert dataset_name == "crc", "terra2k exists for crc only"
        h5, gfile = terra2k_path(paths["sid"])
        genes = [g for g in gfile.read_text().split() if g]
        assert len(genes) == args["n_top_genes"], f"{gfile}: {len(genes)} genes, expected 2000"
        src = sc.read_h5ad(h5)
        src.obs_names_make_unique()
        assert list(src.var_names.astype(str)) == genes, f"{h5} is not on the {gfile} gene axis"
        adata_hvg = prep(src, n_top_genes=len(genes),
                         labels_key=args["labels_key"], domains_key=args["domains_key"])
        assert set(adata_hvg.var_names.astype(str)) <= set(genes), "terra2k HVG selection moved"
        if adata_hvg.n_vars != len(genes):
            msg = f"[terra2k] preprocess dropped {len(genes) - adata_hvg.n_vars}/{len(genes)} genes"
            assert "_max" in paths["sid"], msg + " -- expected only on a --max-cells smoke subset"
            print(msg + " (smoke subset)")
    else:
        adata_hvg = prep(raw.copy(), n_top_genes=args["n_top_genes"],
                         labels_key=args["labels_key"], domains_key=args["domains_key"])
    print(f"[hvg:{universe}] {adata_hvg.n_obs:,} cells x {adata_hvg.n_vars:,} genes")

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
                           args=args, holdout_ct=holdout_ct, sid=paths["sid"], paths=paths,
                           universe=universe)


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


EMB_KEYS = ["cell_emb", "spatial_cell_emb", "neighborhood_emb"]
EMB_KWARGS = dict(emb_layer=None, agg_excluded_genes=None, top_k=None, batch_size=32,
                  include_spatial_cell_emb=True, return_token_embeddings=False,
                  ignore_spc_tokens=True, num_workers=8)


def embed_cached(tok, bundle, cache):
    """embed_dataset over `tok`, cached as an npz keyed by cell_id -> (emb dict, cell ids)."""
    from terra.inference import embed_dataset

    cache = Path(cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print("loaded cached embeddings", cache)
        return {k: z[k] for k in EMB_KEYS}, [str(c) for c in z["cell_id"]]
    emb = embed_dataset(dataset=tok, model_folder_path=str(bundle), **EMB_KWARGS)
    ids = [str(c) for c in tok.with_format(None)["cell_id"]]
    emb = {k: np.asarray(emb[k]) for k in EMB_KEYS}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, cell_id=np.array(ids, dtype=object), **emb)
    print("wrote", cache)
    return emb, ids


# --- neighbour perturbation, shared by inference.py and eval_terra.py ------------------

def perturbation_logfc(adata_hvg, args, holdout_domain, scored_ct):
    """Per-neighbour-cell-type logFC on the HVG axis -> DataFrame (cell type x gene).

    The three canonical lines from notebooks/loo_benchmarks/cellina_node_pert.ipynb cell 9
    (repeated at notebooks/loo_benchmarks/spatialprop/spatialprop_eval_loo.py:263-266):
    every neighbour cell type gets its OWN observed logFC, and only the scored type uses the
    leak-free global one.  Computed on `adata_hvg` because the baselines subset to the 2000
    benchmark HVGs first (train_loo.py:184-186).  `scored_ct` is the holdout for a LOO run
    and any observed cell type when a trained model is scored on it (--eval-celltypes).
    """
    from counterfactual_analysis import get_perturbation_logfc, get_global_perturbation_logfc

    cd, lk, dk = args["control_domains"][0], args["labels_key"], args["domains_key"]
    df = get_perturbation_logfc(adata_hvg, cd, holdout_domain, lk, dk)
    g = get_global_perturbation_logfc(adata_hvg, cd, holdout_domain, lk, dk, scored_ct)
    df.loc[scored_ct, g.index] = g
    assert np.isfinite(df.to_numpy()).all(), "logFC contains non-finite values"
    return df


def token_delta(logfc_df, adata_terra, model_dir, labels_key):
    """-> (delta[n_ct + 1, vocab], ct_code[n_obs]); the extra last row is all-zero.

    Row i holds the top-N_PERT_GENES of cell type i, mapped onto TERRA token ids.  Cell
    types absent from `logfc_df` map to the zero row -- cellina leaves them unmodified
    (cellina/src/cellina/_spatial_utils.py:372-374).
    """
    import pickle

    with open(f"{model_dir}/token_dictionary.pkl", "rb") as f:
        token_dict = pickle.load(f)
    sym2ens = adata_terra.var["ensembl_id"].to_dict()
    cts = [str(c) for c in logfc_df.index]
    delta = np.zeros((len(cts) + 1, max(token_dict.values()) + 1), dtype=np.float64)
    for i, ct in enumerate(cts):
        s = logfc_df.loc[ct]
        top = s.abs().nlargest(N_PERT_GENES).index
        n = 0
        for g in top:
            tid = token_dict.get(sym2ens.get(g))
            if tid is not None:
                delta[i, tid] = s[g]
                n += 1
        print(f"  [{ct}] {n}/{len(top)} perturbation genes mapped to TERRA tokens")
    code = {ct: i for i, ct in enumerate(cts)}
    ct_code = np.array([code.get(str(c), len(cts)) for c in adata_terra.obs[labels_key]],
                       dtype=np.int32)
    return delta, ct_code


def position_codes(tok_subset, coords, focal_rows, ct_code, n_pos):
    """(len(tok_subset), n_pos) cell-type code per token position.

    The tokenizer lays a cell out as `n_seg` equal blocks -- block 0 is the focal cell,
    block s is the s-th nearest neighbour -- and writes one rel_x/rel_y per block in the
    SAME `ordered_neighbors` order (terra/tokenizers/cell_tokenizers.py:1030-1067,
    1144-1181).  So `coords[focal] + rel` identifies each block's cell exactly; the
    asserts below fail loudly if that layout ever stops holding.
    """
    from scipy.spatial import cKDTree

    raw = tok_subset.with_format(None)
    rel = np.stack([np.asarray(raw["rel_x_coord"], dtype=np.float64),
                    np.asarray(raw["rel_y_coord"], dtype=np.float64)], axis=-1)
    focal_rows = np.asarray(focal_rows)
    dist, rows = cKDTree(coords).query(coords[focal_rows][:, None, :] + rel, k=1)
    assert dist.max() < 1e-6, f"token block coords do not land on a cell (max {dist.max():.3e})"
    assert (rows[:, 0] == focal_rows).all(), "block 0 is not the focal cell"
    n_seg = rel.shape[1]
    assert n_pos % n_seg == 0, f"{n_pos} token positions do not split into {n_seg} blocks"
    return np.repeat(ct_code[rows], n_pos // n_seg, axis=1)


def shift_neighbourhood(batch, idx, delta, codes, seq_len_cell):
    """Add each neighbour's own log-space logFC to its tokens.  Cell block untouched."""
    tokens = np.asarray(batch["gene_tokens"])
    expr = np.asarray(batch["gene_expr"], dtype=np.float64)
    nb_tok, nb_expr = tokens[:, seq_len_cell:], expr[:, seq_len_cell:]
    shift = delta[codes[idx][:, seq_len_cell:], nb_tok]
    shift[nb_expr <= 0] = 0.0        # undetected / padded genes are not perturbable
    expr[:, seq_len_cell:] = np.clip(nb_expr + shift, 0.0, None)
    batch["gene_tokens"], batch["gene_expr"] = tokens, expr
    return batch


def map_perturbation(ds, delta, codes, seq_len_cell):
    """Map with the torch format OFF -- what terra's own perturb_dataset does; restore it."""
    fmt = ds.format
    out = ds.with_format(None).map(
        shift_neighbourhood, batched=True, batch_size=256, with_indices=True,
        keep_in_memory=True, load_from_cache_file=False,
        fn_kwargs=dict(delta=delta, codes=codes, seq_len_cell=seq_len_cell))
    out.set_format(type=fmt["type"], columns=fmt["columns"],
                   output_all_columns=fmt["output_all_columns"])
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="crc")
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", default="Fibroblast")
    p.add_argument("--universe", default="benchmark", choices=["benchmark", "terra2k"])
    p.add_argument("--tokenize", action="store_true")
    a = p.parse_args()

    md = model_dir()
    d = load_dataset(a.dataset_name, a.adata_path, a.holdout_celltype, md, universe=a.universe)
    print("args:", d.args)
    print("paths:", {k: str(v) for k, v in d.paths.items() if not callable(v)})
    print("adata_hvg:", d.adata_hvg.shape, "| adata_terra:", d.adata_terra.shape)
    print(d.adata_terra.obs[d.args["domains_key"]].value_counts())
    assert d.adata_hvg.n_vars == d.args["n_top_genes"]
    if a.tokenize:
        tok = tokenize_cached(d.adata_terra, md, d.paths["tok_cache"])
        print("tokenised:", len(tok), list(tok.features))
    print("OK")
