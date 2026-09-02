"""Shared scoring for the rebuttal sweeps: every metric that scripts/eval_loo.py reports.

Imports the metric implementations from ``scripts/counterfactual_analysis.py`` so the
sweeps and the paper's LOO evaluation stay a single source of truth.

Why more than ``pearson``: ``get_lfc`` builds both vectors as
``gt = log2(T/C)`` and ``cf = log2(P/C)`` -- a SHARED observed-control denominator --
and then selects the gene set on ``|gt|``. Measured on real data with predictors that
know nothing about the healthy->tumour shift (a different cell type's control mean, or
the control mean with gene labels permuted), ``pearson`` still reaches +0.93 on 232
T_cell and +0.98 on 210 Endothelial. ``precision``, ``direction_match_k`` and
``edistance`` null out near zero on the same surrogates, so they carry the signal
``pearson`` cannot. See ``mse_lfc`` for the algebraic reason the denominator matters:
``gt - cf = log2(T) - log2(P)``, i.e. C cancels in a difference but not in a correlation.

``*_pos`` variants repeat the logFC metrics over the top-``n_deg`` genes among those
with ``mean_control > 0``. Genes absent from every control cell get
``log2((T + 1e-6)/1e-6) ~= +20`` from ``safe_log2_fold_change``'s pseudo-count and are
therefore guaranteed top-50 picks with enormous leverage in a 50-point correlation.
"""
import numpy as np
import os
import sys

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from counterfactual_analysis import (  # noqa: E402
    get_lfc, precision, direction_match, compute_mse_lfc, compute_rmse,
    mixing_index, compute_edistance, _normalize_counts,
)

# mixing_index PCAs the stacked (predicted, observed) matrix; at 8 concurrent jobs
# the full Epithelial population (37k + 37k x 2000) is not worth the RAM.
MIXING_MAX_CELLS = 10_000


def _sub(X, n, rng):
    return X if X.shape[0] <= n else X[rng.choice(X.shape[0], n, replace=False)]


def score_all(control, target, counterfactual, adata_full, n_deg=50,
              library_size=1e4, seed=0):
    """Every eval_loo.py metric for one run, plus the C>0-restricted variants.

    Parameters
    ----------
    control, target, counterfactual
        Cell x gene count matrices: observed control cells, observed target cells,
        and the model's predicted counterfactual for the control cells.
    adata_full
        Full AnnData; only used by ``compute_edistance`` (and only when ``use_pca``).
    """
    from scipy.stats import pearsonr, spearmanr

    rng = np.random.default_rng(seed)
    out = {}

    gt, cf, deg = get_lfc(control=control, target=target,
                          counterfactual=counterfactual, n_deg=n_deg)

    def _lfc_metrics(gt_v, cf_v, deg_v, suffix=""):
        m = {}
        m[f"pearson{suffix}"] = pearsonr(gt_v[deg_v], cf_v[deg_v])[0]
        m[f"spearman{suffix}"] = spearmanr(gt_v[deg_v], cf_v[deg_v])[0]
        m[f"precision{suffix}"] = precision(gt_v, cf_v, k=n_deg, use_abs=True)
        m[f"direction_match{suffix}"] = direction_match(gt_v, cf_v, k=n_deg,
                                                        normalize="intersection")
        m[f"direction_match_k{suffix}"] = direction_match(gt_v, cf_v, k=n_deg,
                                                          normalize="k")
        m[f"direction_match_gt{suffix}"] = direction_match(gt_v, cf_v, k=n_deg,
                                                           normalize="gt_topk")
        m[f"mse_lfc{suffix}"] = compute_mse_lfc(gt_vec=gt_v, cf_vec=cf_v, deg=deg_v)
        return m

    out.update(_lfc_metrics(gt, cf, deg))

    # ---- pseudo-count contamination of the DEG set -----------------------
    mean_control = np.nanmean(_normalize_counts(control), axis=0)
    pos = np.where(mean_control > 0)[0]
    out["n_genes_control_eq_0"] = int((mean_control == 0).sum())
    out["n_deg_control_eq_0"] = int((mean_control[deg] == 0).sum())
    if len(pos) > n_deg:
        gt_p, cf_p = gt[pos], cf[pos]
        deg_p = np.argsort(-np.abs(gt_p))[:n_deg]
        out.update(_lfc_metrics(gt_p, cf_p, deg_p, suffix="_pos"))

    # ---- matrix-level metrics: no control denominator anywhere -----------
    out["rmse"] = compute_rmse(observed=target, predicted=counterfactual,
                               deg=deg, library_size=library_size)
    out["mixing_index"] = mixing_index(
        observed=_sub(target, MIXING_MAX_CELLS, rng),
        predicted=_sub(counterfactual, MIXING_MAX_CELLS, rng),
        library_size=library_size)
    out["mixing_index_max_cells"] = MIXING_MAX_CELLS
    # deg=None matches eval_loo.py (all genes); the deg-restricted variant is the
    # one comparable to the logFC metrics above.
    for tag, deg_arg in (("", None), ("_deg", deg)):
        for loc_tag, local in (("_global", False), ("_local", True)):
            out[f"edistance{loc_tag}{tag}"] = compute_edistance(
                adata_full, observed=target, predicted=counterfactual,
                deg=deg_arg, library_size=library_size, local=local)

    return {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                else float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in out.items()}, gt, cf, deg


def save_artifacts(path, control, target, counterfactual, gt, cf, deg,
                   n_cells=2000, seed=0):
    """Per-run dump so a future metric change is an offline re-score, not a re-run."""
    rng = np.random.default_rng(seed)
    np.savez_compressed(
        path,
        mean_control=np.nanmean(_normalize_counts(control), axis=0).astype(np.float32),
        mean_target=np.nanmean(_normalize_counts(target), axis=0).astype(np.float32),
        mean_cf=np.nanmean(_normalize_counts(counterfactual), axis=0).astype(np.float32),
        gt_lfc=gt.astype(np.float32), cf_lfc=cf.astype(np.float32),
        deg=deg.astype(np.int32),
        target_sample=_sub(target, n_cells, rng).astype(np.float32),
        cf_sample=_sub(counterfactual, n_cells, rng).astype(np.float32),
    )
