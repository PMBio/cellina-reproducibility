"""Shared preprocessing for the Perturb-FISH xenograft edge-swap experiment.

Both ``train_cellina.py`` (training) and ``pfish_cellina_edge_swap.ipynb``
(analysis) import from here so the data prep, spatial graph, context labels and
the control/target index sets are defined exactly once and stay consistent.

The Cellina edge-swap counterfactual we build here (paper "Case 1") is:

    responder  = unperturbed T cells
    control-context T = T cells whose cancer neighbours are all unperturbed/Control
    target-context  T = T cells with >=1 perturbed cancer neighbour (CRISPR KO)
    donors      = the perturbed cancer neighbours of the target-context T cells

We then predict the control-context T cells forward into the perturbed niche
and validate the predicted logFC against the observed target-context T cells.
This mirrors ``docs/tutorial.ipynb`` (REF-region -> CRC-region holdout swap).
"""

from __future__ import annotations

import numpy as np
import scanpy as sc

from cellina._spatial_utils import compute_spatial_features, spatial_neighbors

# --- data-specific constants -------------------------------------------------
# X in the linearized h5ad is normalize_total'd (every cell sums to this value);
# obs['n_counts'] holds the original raw total, so raw counts are recoverable.
NORM_TARGET = 178.0
# Coordinate NN spacing is ~99 units; a bandwidth of a few hundred captures a
# local neighbourhood of order ~10-30 cells. Tune with the neighbour-count
# printout in build_spatial() if needed.
DEFAULT_BANDWIDTH = 250.0
DEFAULT_MAX_NEIGHBOURS = 30

CONN_ORIG_KEY = "spatial_connectivities_orig"  # full graph, no test masking
CONN_KEY = "spatial_connectivities"            # (possibly) test-masked graph


def load_pfish(path: str) -> sc.AnnData:
    """Load the linearized h5ad and materialise ``counts`` and ``lognorm`` layers.

    - ``layers['counts']``  : reconstructed raw integer counts (for Cellina/NB).
    - ``layers['lognorm']`` : the stored ``log1p`` layer (log of normalized X).
    - ``X`` is left as the normalized values on load; callers set it explicitly.
    """
    adata = sc.read_h5ad(path)

    # Recover raw counts: X = raw / n_counts * NORM_TARGET  ->  raw = X * n_counts / NORM_TARGET
    n_counts = adata.obs["n_counts"].to_numpy()[:, None].astype(np.float64)
    raw = np.asarray(adata.X, dtype=np.float64) * n_counts / NORM_TARGET
    counts = np.rint(raw).astype(np.float32)
    # sanity: reconstruction should be close to integer before rounding
    resid = np.abs(raw - counts).max()
    if resid > 0.5:
        print(f"[pfish_prep] WARN: max count-reconstruction residual {resid:.3f} (>0.5)")
    adata.layers["counts"] = counts
    adata.layers["lognorm"] = np.asarray(adata.layers["log1p"], dtype=np.float32)
    return adata


def filter_cells(adata: sc.AnnData, max_n_perturb: int = 1) -> sc.AnnData:
    """Drop ambiguous multi-guide cells (``n_perturb >= 2``), matching the EDA."""
    keep = adata.obs["n_perturb"].to_numpy() < (max_n_perturb + 1)
    n_drop = int((~keep).sum())
    print(f"[pfish_prep] dropping {n_drop} cells with n_perturb > {max_n_perturb}")
    return adata[keep].copy()


def build_graph(
    adata: sc.AnnData,
    bandwidth: float = DEFAULT_BANDWIDTH,
    max_neighbours: int = DEFAULT_MAX_NEIGHBOURS,
) -> sc.AnnData:
    """Build the full (unmasked) spatial graph in ``obsp[CONN_ORIG_KEY]``.

    This graph defines who-neighbours-whom and is used by ``label_context`` and
    ``make_edge_swap_sets``. It must be built before the index sets, because the
    test-masked feature graph depends on knowing which cells are held out.
    """
    adata.obsp[CONN_ORIG_KEY] = spatial_neighbors(
        adata, bandwidth=bandwidth, max_neighbours=max_neighbours,
        standardize=False, inplace=False,
    )
    deg = np.asarray((adata.obsp[CONN_ORIG_KEY] > 0).sum(1)).ravel()
    print(f"[pfish_prep] neighbours/cell: median={np.median(deg):.0f} "
          f"mean={deg.mean():.1f} (bandwidth={bandwidth}, max={max_neighbours})")
    return adata


def build_features(
    adata: sc.AnnData,
    bandwidth: float = DEFAULT_BANDWIDTH,
    max_neighbours: int = DEFAULT_MAX_NEIGHBOURS,
    test_indices: np.ndarray | None = None,
) -> sc.AnnData:
    """Compute Cellina neighbour features (``obsm['spatial_x']``).

    Writes a possibly test-masked graph to ``obsp[CONN_KEY]`` (masking held-out
    cells out of the reference to avoid leakage), aggregates log-normalized
    neighbour expression into ``obsm['spatial_x']``, then resets ``X`` to raw
    counts as Cellina's NB likelihood expects.
    """
    spatial_neighbors(
        adata, bandwidth=bandwidth, max_neighbours=max_neighbours,
        standardize=False, test_indices=test_indices,
    )
    adata.X = adata.layers["lognorm"].copy()
    compute_spatial_features(adata, connectivity_key=CONN_KEY)
    adata.X = adata.layers["counts"].copy()
    return adata


def label_context(adata: sc.AnnData) -> sc.AnnData:
    """Tag each cell with a binary perturbation-context ``obs['context']``.

    - cancer cell            -> 'perturbed' if n_perturb>0 else 'control'
    - non-cancer (T / other) -> 'perturbed' if it has a perturbed cancer neighbour,
                                else 'control'

    This doubles as the adversarial ``domains_key`` for training (the microenv.
    signal we want routed to the spatial latent ``s`` rather than identity ``z``).
    """
    obs = adata.obs
    is_cancer = (obs["celltype2"] == "cancer").to_numpy()
    perturbed_cancer = is_cancer & (obs["n_perturb"].to_numpy() > 0)

    conn = adata.obsp[CONN_ORIG_KEY]
    # neighbour is a perturbed cancer cell?  (conn @ indicator > 0)
    has_pert_neighbour = np.asarray(conn.dot(perturbed_cancer.astype(np.float64))).ravel() > 0

    context = np.where(is_cancer,
                       np.where(perturbed_cancer, "perturbed", "control"),
                       np.where(has_pert_neighbour, "perturbed", "control"))
    adata.obs["context"] = context
    adata.obs["has_pert_neighbour"] = has_pert_neighbour
    print("[pfish_prep] context counts:\n", adata.obs["context"].value_counts())
    return adata


def make_edge_swap_sets(adata: sc.AnnData, responder: str = "T cells"):
    """Build the edge-swap index sets (Case 1).

    Returns
    -------
    idx_control : responder cells in a control niche (predict these forward)
    idx_target  : responder cells in a perturbed niche (observed ground truth)
    donor_idx   : perturbed cancer neighbours of idx_target (transplanted niche)
    """
    obs = adata.obs
    is_resp = (obs["celltype2"] == responder).to_numpy()
    unperturbed = obs["n_perturb"].to_numpy() == 0          # drop perturbed responders
    resp = is_resp & unperturbed

    is_cancer = (obs["celltype2"] == "cancer").to_numpy()
    perturbed_cancer = is_cancer & (obs["n_perturb"].to_numpy() > 0)
    has_pert = adata.obs["has_pert_neighbour"].to_numpy()

    # a responder counts as 'control niche' only if it has cancer neighbours,
    # none of them perturbed.
    conn = adata.obsp[CONN_ORIG_KEY]
    n_cancer_neighbours = np.asarray(conn.dot(is_cancer.astype(np.float64))).ravel()

    idx_target = np.where(resp & has_pert)[0]
    idx_control = np.where(resp & ~has_pert & (n_cancer_neighbours > 0))[0]

    # donors = the FULL neighbourhood of the target responders, minus other
    # responder (T) cells. This is the realistic perturbed-context microenvironment
    # (perturbed cancer + surrounding 'other' cells), matching the tutorial. NOT
    # restricted to perturbed cancer only — a pure-cancer niche is out-of-distribution
    # for a T cell and makes the decoder extrapolate to a degenerate output.
    sub = conn[idx_target]
    neigh = np.unique(sub.nonzero()[1])
    donor_idx = neigh[~is_resp[neigh]]
    n_pert_donors = int(perturbed_cancer[donor_idx].sum())

    print(f"[pfish_prep] responder='{responder}'  "
          f"control(niche)={len(idx_control)}  target(niche)={len(idx_target)}  "
          f"donors={len(donor_idx)} (of which perturbed-cancer={n_pert_donors})")
    return idx_control, idx_target, donor_idx
