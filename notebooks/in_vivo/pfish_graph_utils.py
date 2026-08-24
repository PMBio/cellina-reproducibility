"""Cellina-free re-implementation of the pfish_prep.py data-prep steps needed by
the SpatialProp analysis (spprop_analysis.ipynb).

``pfish_prep.py`` pulls ``spatial_neighbors``/``compute_spatial_features`` from the
``cellina`` package, which lives in a conda env (``cellina-graph``) that does not
have ``spatial_gnn`` installed (and vice versa: the ``spatial-prop`` env does not
have ``cellina``). SpatialProp builds its own graph internally and never touches
Cellina's aggregated ``spatial_x`` niche feature, so the only piece actually needed
here is the "who neighbours whom" graph used to define perturbation *context* and
the edge-swap index sets (``idx_control`` / ``idx_target`` / ``donor_idx``).

``spatial_neighbors`` below is a trimmed, dependency-free copy of
``cellina._spatial_utils.spatial_neighbors`` (single-sample case, Gaussian kernel
only) so that the graph — and therefore ``idx_control``/``idx_target`` — is
constructed identically to ``pfish_prep.py``/``pfish_analysis.ipynb``, keeping the
two analyses on the same held-out cells for a fair comparison.
"""

from __future__ import annotations

import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

NORM_TARGET = 178.0
DEFAULT_BANDWIDTH = 250.0
DEFAULT_MAX_NEIGHBOURS = 30

CONN_ORIG_KEY = "spatial_connectivities_orig"


def spatial_neighbors(
    adata: sc.AnnData,
    bandwidth: float,
    max_neighbours: int = 100,
    cutoff: float = 0.1,
    spatial_key: str = "spatial",
    test_indices: np.ndarray | None = None,
) -> csr_matrix:
    """Gaussian-kernel spatial connectivity graph (single sample, dense-ish).

    Vendored from ``cellina._spatial_utils`` (Gaussian kernel branch only) so this
    module has no dependency on the ``cellina`` package. See that module for the
    full-featured version (multiple kernels, ``library_key`` batching, etc.).
    """
    coordinates = adata.obsm[spatial_key]

    if test_indices is not None and len(test_indices) > 0:
        coordinates = coordinates.astype(float).copy()
        extent = np.abs(coordinates).max() + 1.0
        test_arr = np.asarray(test_indices)
        coordinates[test_arr, 0] += extent * 1e6 * (np.arange(len(test_arr)) + 1)

    tree = NearestNeighbors(n_neighbors=max_neighbours + 1, algorithm="ball_tree", metric="euclidean").fit(
        coordinates
    )
    dist = tree.kneighbors_graph(coordinates, mode="distance")

    bw = np.array(bandwidth, dtype=np.float64)
    dist.data = np.exp(-(dist.data ** 2.0) / (2.0 * bw ** 2.0))
    dist.setdiag(0)
    dist.data = dist.data * (dist.data > cutoff)
    dist.eliminate_zeros()

    if dist.shape[0] > 1000:
        dist = dist.astype(np.float32)
    return dist


def load_pfish(path: str) -> sc.AnnData:
    """Load the linearized h5ad and materialise a ``counts`` layer of raw integer counts."""
    adata = sc.read_h5ad(path)
    n_counts = adata.obs["n_counts"].to_numpy()[:, None].astype(np.float64)
    raw = np.asarray(adata.X, dtype=np.float64) * n_counts / NORM_TARGET
    counts = np.rint(raw).astype(np.float32)
    resid = np.abs(raw - counts).max()
    if resid > 0.5:
        print(f"[pfish_graph_utils] WARN: max count-reconstruction residual {resid:.3f} (>0.5)")
    adata.layers["counts"] = counts
    return adata


def filter_cells(adata: sc.AnnData, max_n_perturb: int = 1) -> sc.AnnData:
    """Drop ambiguous multi-guide cells (``n_perturb >= 2``), matching pfish_prep.py."""
    keep = adata.obs["n_perturb"].to_numpy() < (max_n_perturb + 1)
    n_drop = int((~keep).sum())
    print(f"[pfish_graph_utils] dropping {n_drop} cells with n_perturb > {max_n_perturb}")
    return adata[keep].copy()


def build_graph(
    adata: sc.AnnData,
    bandwidth: float = DEFAULT_BANDWIDTH,
    max_neighbours: int = DEFAULT_MAX_NEIGHBOURS,
) -> sc.AnnData:
    """Build the full (unmasked) spatial graph in ``obsp[CONN_ORIG_KEY]``."""
    adata.obsp[CONN_ORIG_KEY] = spatial_neighbors(adata, bandwidth=bandwidth, max_neighbours=max_neighbours)
    deg = np.asarray((adata.obsp[CONN_ORIG_KEY] > 0).sum(1)).ravel()
    print(
        f"[pfish_graph_utils] neighbours/cell: median={np.median(deg):.0f} "
        f"mean={deg.mean():.1f} (bandwidth={bandwidth}, max={max_neighbours})"
    )
    return adata


def label_context(adata: sc.AnnData) -> sc.AnnData:
    """Tag each cell with a binary perturbation-context ``obs['context']``.

    Identical logic to ``pfish_prep.label_context``: a cancer cell is 'perturbed' if
    it carries a guide (``n_perturb>0``); a non-cancer cell is 'perturbed' if it has
    at least one perturbed-cancer neighbour.
    """
    obs = adata.obs
    is_cancer = (obs["celltype2"] == "cancer").to_numpy()
    perturbed_cancer = is_cancer & (obs["n_perturb"].to_numpy() > 0)

    conn = adata.obsp[CONN_ORIG_KEY]
    has_pert_neighbour = np.asarray(conn.dot(perturbed_cancer.astype(np.float64))).ravel() > 0

    context = np.where(
        is_cancer,
        np.where(perturbed_cancer, "perturbed", "control"),
        np.where(has_pert_neighbour, "perturbed", "control"),
    )
    adata.obs["context"] = context
    adata.obs["has_pert_neighbour"] = has_pert_neighbour
    print("[pfish_graph_utils] context counts:\n", adata.obs["context"].value_counts())
    return adata


def make_edge_swap_sets(adata: sc.AnnData, responder: str = "T cells"):
    """Build the edge-swap index sets (Case 1) — identical to ``pfish_prep.make_edge_swap_sets``.

    Returns
    -------
    idx_control : responder cells in a control niche (predict these forward)
    idx_target  : responder cells in a perturbed niche (observed ground truth)
    donor_idx   : the full neighbourhood of idx_target responders (perturbed cancer + others)
    """
    obs = adata.obs
    is_resp = (obs["celltype2"] == responder).to_numpy()
    unperturbed = obs["n_perturb"].to_numpy() == 0
    resp = is_resp & unperturbed

    is_cancer = (obs["celltype2"] == "cancer").to_numpy()
    perturbed_cancer = is_cancer & (obs["n_perturb"].to_numpy() > 0)
    has_pert = adata.obs["has_pert_neighbour"].to_numpy()

    conn = adata.obsp[CONN_ORIG_KEY]
    n_cancer_neighbours = np.asarray(conn.dot(is_cancer.astype(np.float64))).ravel()

    idx_target = np.where(resp & has_pert)[0]
    idx_control = np.where(resp & ~has_pert & (n_cancer_neighbours > 0))[0]

    sub = conn[idx_target]
    neigh = np.unique(sub.nonzero()[1])
    donor_idx = neigh[~is_resp[neigh]]
    n_pert_donors = int(perturbed_cancer[donor_idx].sum())

    print(
        f"[pfish_graph_utils] responder='{responder}'  "
        f"control(niche)={len(idx_control)}  target(niche)={len(idx_target)}  "
        f"donors={len(donor_idx)} (of which perturbed-cancer={n_pert_donors})"
    )
    return idx_control, idx_target, donor_idx
