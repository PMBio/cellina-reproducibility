"""Leave-one-out (LOO) benchmark setup for the CRC perturbation task.

This module holds the "plumbing" the notebooks import: CRC preprocessing, the spatial
feature step, the train/val/test split for both **leave-one-cell-type-out** and
**leave-one-region/domain-out**, plus thin glue to feed observed-vs-predicted AnnDatas
into :func:`scripts.metrics.compare_adatas`.

Ported and lightly cleaned from
``cellina-reproducibility/scripts/train_loo.py`` (preprocessing + ``split_indices``).
The metrics themselves live in :mod:`scripts.metrics`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from metrics import compare_adatas

# Defaults for the CRC dataset (mirrors configs/adata_crc_config.py + train_loo.py)
DEFAULT_LABELS_KEY = "coarse_type"
DEFAULT_DOMAINS_KEY = "typ_clean"
DEFAULT_BATCH_KEY = "sid"
DEFAULT_CTRL_DOMAINS = ["REF"]
DEFAULT_HOLDOUT_DOMAINS = ["CRC"]
DEFAULT_SEED = 0
COUNTS_PER_K = 1e4
CRC_STEP_SIZE_PX = 0.12028  # microns-per-pixel scale used to set the spatial bandwidth

# Fine -> coarse cell-type label map for the CRC CosMx data.
# (Inlined from cellina-reproducibility/scripts/_labels_to_coarse.py.)
LABEL_TO_COARSE = {
    "epi1": "Epithelial",
    "epi2": "Epithelial",
    "epi3": "Epithelial",
    "epi4": "Epithelial",
    "fib1": "Fibroblast",
    "fib2": "Fibroblast",
    "EC": "Endothelial",
    "SMC": "Smooth_muscle",
    "BC": "B_cell",
    "PC_IgA": "Plasma_cell",
    "PC_IgG": "Plasma_cell",
    "PC_IgM": "Plasma_cell",
    "TC": "T_cell",
    "mye1": "Myeloid",
    "mye2": "Myeloid",
    "mast": "Mast_cell",
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _preprocess_adata(adata, n_top_genes=2000, labels_key=DEFAULT_LABELS_KEY, domains_key=DEFAULT_DOMAINS_KEY):
    """Shared QC + HVG selection. Stores raw counts in ``adata.layers['counts']``."""
    adata.obs_names_make_unique()

    adata = adata[~adata.obs[domains_key].isna()]
    adata = adata[~adata.obs[labels_key].isna()]

    adata.obs[labels_key] = adata.obs[labels_key].astype("category")
    adata.obs[domains_key] = adata.obs[domains_key].astype("category")

    sc.pp.filter_cells(adata, min_counts=3)
    sc.pp.filter_genes(adata, min_counts=3)

    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, layer="counts", flavor="seurat_v3", n_top_genes=n_top_genes, subset=True)
    adata.X = adata.layers["counts"].copy()

    return adata


def preprocess_crc(adata, n_top_genes=2000, labels_key=DEFAULT_LABELS_KEY, domains_key=DEFAULT_DOMAINS_KEY):
    """CRC-specific preprocessing: coarse labels, clean domains, spatial coords, then QC/HVG."""
    # Coarse cell-type label
    if "coarse_type" not in adata.obs.columns or adata.obs["coarse_type"].isna().any():
        adata.obs["coarse_type"] = adata.obs["ist"].map(LABEL_TO_COARSE).astype("category")

    # Clean spatial domain from the raw `typ` string (REF / TVA / CRC)
    adata.obs["typ_clean"] = adata.obs["typ"].str.extract(r"(REF|TVA|CRC)", expand=False)

    # Spatial coordinates in the standard obsm field
    adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values

    return _preprocess_adata(adata, n_top_genes=n_top_genes, labels_key=labels_key, domains_key=domains_key)


def preprocess_spatial_features(adata, step_size_px=CRC_STEP_SIZE_PX, n_neighbors=200, test_indices=None):
    """Compute cellina spatial neighbour graph + aggregated features.

    Stores the original (unmasked) adjacency in ``adata.obsp['spatial_connectivities_orig']``
    for inference, and recomputes with ``test_indices`` masked out to avoid leaking the
    held-out cells into the training-set spatial features. Resets ``adata.X`` to raw counts.
    """
    sc.pp.normalize_total(adata, target_sum=COUNTS_PER_K)
    sc.pp.log1p(adata)

    from cellina._spatial_utils import spatial_neighbors, compute_spatial_features

    adata.obsp["spatial_connectivities_orig"] = spatial_neighbors(
        adata, bandwidth=100 / step_size_px, max_neighbours=n_neighbors, standardize=False, inplace=False
    )
    # Recompute with test indices masked out (no data leakage into spatial features).
    spatial_neighbors(
        adata, bandwidth=100 / step_size_px, max_neighbours=n_neighbors, standardize=False, test_indices=test_indices
    )
    compute_spatial_features(adata)

    adata.X = adata.layers["counts"].copy()  # cellina expects raw counts in .X
    return adata


# ---------------------------------------------------------------------------
# LOO splits
# ---------------------------------------------------------------------------


def split_indices(
    adata,
    holdout_celltype,
    labels_key=DEFAULT_LABELS_KEY,
    domains_key=DEFAULT_DOMAINS_KEY,
    holdout_domains=DEFAULT_HOLDOUT_DOMAINS,
    seed=DEFAULT_SEED,
    holdout_full_domain=False,
):
    """Create train/val/test splits for the LOO benchmark.

    Two regimes:
      - **leave-one-cell-type-out** (``holdout_full_domain=False``, default): the test set
        is ``holdout_celltype`` cells located in a holdout domain (domain label contains
        any string in ``holdout_domains`` — substring match, e.g. ``'CRC'``).
      - **leave-one-region/domain-out** (``holdout_full_domain=True``): the test set is
        *every* cell in the holdout domain, regardless of cell type.

    Val is a random 10% of the remaining train/val pool. Also annotates
    ``adata.obs['is_holdout']``.
    """
    if holdout_celltype not in adata.obs[labels_key].unique():
        raise ValueError(f"holdout_celltype '{holdout_celltype}' not found in adata.obs['{labels_key}']")

    domain_str = adata.obs[domains_key].astype(str)
    is_holdout_domain = domain_str.apply(lambda d: any(hd in d for hd in holdout_domains))
    is_holdout_ct = adata.obs[labels_key].astype(str) == holdout_celltype

    test_mask = is_holdout_domain if holdout_full_domain else (is_holdout_domain & is_holdout_ct)

    all_idx = np.arange(adata.n_obs)
    test_idx = np.where(test_mask.values)[0]
    trainval_idx = np.setdiff1d(all_idx, test_idx)

    rng = np.random.default_rng(seed)
    n_trainval = trainval_idx.shape[0]
    n_val = max(1, int(0.1 * n_trainval))
    val_idx_rel = rng.choice(np.arange(n_trainval), size=n_val, replace=False)
    val_idx = trainval_idx[val_idx_rel]
    train_idx = np.setdiff1d(trainval_idx, val_idx)

    adata.obs["is_holdout"] = False
    if len(test_idx) > 0:
        adata.obs.iloc[test_idx, adata.obs.columns.get_loc("is_holdout")] = True

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Glue to the metrics API
# ---------------------------------------------------------------------------


def _to_dense(x):
    toarray = getattr(x, "toarray", None)
    if callable(toarray):
        return toarray()
    return np.asarray(x)


def make_comparison_adata(
    control_counts,
    perturbed_counts,
    var_names,
    *,
    control_label="REF",
    perturbed_label="CRC",
    pert_col=DEFAULT_DOMAINS_KEY,
):
    """Assemble a count-like AnnData with a 2-level ``pert_col`` (control + perturbed).

    Used to build both the *observed* pair (real control vs real held-out cells) and a
    *predicted* pair (model reconstruction of control vs model counterfactual), so that
    each can be passed straight to :func:`compare_adatas`.
    """
    control_counts = _to_dense(control_counts)
    perturbed_counts = _to_dense(perturbed_counts)
    X = np.vstack([control_counts, perturbed_counts]).astype(np.float32)

    labels = [control_label] * control_counts.shape[0] + [perturbed_label] * perturbed_counts.shape[0]
    adata = ad.AnnData(X=X)
    adata.obs[pert_col] = pd.Categorical(labels)
    adata.var_names = list(var_names)
    return adata


def run_loo_eval(
    observed,
    predicted,
    pert_col=DEFAULT_DOMAINS_KEY,
    control_label="REF",
    top_n=50,
    metrics=None,
    predicted_logfc_baseline="observed_control",
    **kwargs,
):
    """Thin wrapper over :func:`compare_adatas` with LOO-friendly defaults.

    ``predicted_logfc_baseline='observed_control'`` matches the cellina tutorial, which
    computes both the ground-truth and counterfactual logFC against the *same observed*
    control population.
    """
    if metrics is None:
        metrics = ["pearson_logfc", "rmse_logfc", "spearman_avg", "precision_at_top_n", "edistance_pca"]
    return compare_adatas(
        observed,
        predicted,
        pert_col=pert_col,
        control_label=control_label,
        top_n=top_n,
        metrics=metrics,
        predicted_logfc_baseline=predicted_logfc_baseline,
        **kwargs,
    )
