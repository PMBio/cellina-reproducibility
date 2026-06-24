"""Evaluation metrics and AnnData comparison utilities for perturbation prediction."""

from __future__ import annotations

from typing import Literal
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

__all__ = [
    "compare_adatas",
    "get_perturbation_effects",
    "nb_deviance",
    "nb_deviance_pop_mean",
    "edistance_pca",
    "pearson_logfc",
    "rmse_logfc",
    "spearman_avg",
    "precision_at_top_n",
]

# ---------------------------------------------------------------------------
# Private preprocessing helpers
# ---------------------------------------------------------------------------


def _to_dense(X) -> np.ndarray:
    """Sparse or array-like -> dense float32 2D array."""
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {X.shape}")
    return X


def _log1p_norm(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """CPM normalise then log1p. Input: raw counts [n_cells, n_genes]."""
    X = _to_dense(X)
    if np.any(X < 0):
        raise ValueError("Counts must be non-negative")
    totals = X.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Cannot normalize cells with zero total counts")
    X_norm = np.float32(target_sum) * X / totals
    return np.log1p(X_norm).astype(np.float32, copy=False)


def _top_n_idx(logfc: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n genes with largest absolute logFC."""
    logfc = np.asarray(logfc)
    idx = np.argpartition(np.abs(logfc), -n)[-n:]
    return idx[np.argsort(np.abs(logfc[idx]))[::-1]]


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def pearson_logfc(obs_logfc: np.ndarray, pred_logfc: np.ndarray) -> float:
    """Pearson r between observed and predicted logFC vectors [n_genes]."""
    return float(pearsonr(obs_logfc, pred_logfc).statistic)


def rmse_logfc(obs_logfc: np.ndarray, pred_logfc: np.ndarray) -> float:
    """RMSE between observed and predicted logFC vectors [n_genes]."""
    return float(np.sqrt(np.mean((obs_logfc - pred_logfc) ** 2)))


def spearman_avg(obs_X: np.ndarray, pred_X: np.ndarray) -> float:
    """
    Spearman r between per-gene mean of obs and per-gene mean of pred.
    Inputs are log1p-normed count matrices [n_cells, n_genes].
    """
    return float(spearmanr(obs_X.mean(axis=0), pred_X.mean(axis=0)).statistic)


def precision_at_top_n(obs_logfc: np.ndarray, pred_logfc: np.ndarray, n: int) -> float:
    """Fraction of predicted top-|logFC| genes recovered in the observed top-|logFC| set."""
    obs = np.asarray(obs_logfc, dtype=np.float64)
    pred = np.asarray(pred_logfc, dtype=np.float64)
    if obs.shape != pred.shape:
        raise ValueError(f"obs_logfc and pred_logfc must have the same shape, got {obs.shape} and {pred.shape}")
    if obs.ndim != 1:
        raise ValueError(f"Expected 1D logFC vectors, got shape {obs.shape}")

    k = min(int(n), obs.shape[0])
    obs_top = set(_top_abs_idx(obs, k).tolist())
    pred_top = set(_top_abs_idx(pred, k).tolist())
    return float(len(obs_top & pred_top) / k)


def nb_deviance(
    obs_X: np.ndarray,
    pred_X: np.ndarray,
    theta: np.ndarray | None = None,
    eps: float = 1e-8,
    reduce: str = "mean",
) -> float | np.ndarray:
    """
    Negative-binomial deviance on count-like matrices [n_cells, n_genes].

    When ``theta`` is omitted, per-gene dispersion is estimated from ``obs_X`` with a
    method-of-moments estimator and clipped toward a large finite value for genes that
    appear approximately Poisson or underdispersed.

    Parameters
    ----------
    obs_X
        Observed count matrix [n_cells, n_genes].
    pred_X
        Predicted count matrix [n_cells, n_genes].
    theta
        Per-gene dispersion vector [n_genes]. Estimated from obs_X when None.
    eps
        Small constant for numerical stability.
    reduce
        How to reduce the 2-D deviance array:
        - ``"mean"``: scalar — mean per-cell deviance (default, backward-compatible).
        - ``"per_cell"``: 1-D array [n_cells] — deviance summed over genes per cell.
        - ``"per_gene"``: 1-D array [n_genes] — deviance summed over cells per gene.
    """
    if reduce not in ("mean", "per_cell", "per_gene"):
        raise ValueError(f"reduce must be 'mean', 'per_cell', or 'per_gene', got '{reduce}'")

    obs = np.asarray(obs_X, dtype=np.float64)
    pred = np.asarray(pred_X, dtype=np.float64)
    if obs.shape != pred.shape:
        raise ValueError(f"obs_X and pred_X must have the same shape, got {obs.shape} and {pred.shape}")
    if obs.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got obs_X with shape {obs.shape}")
    if np.any(obs < 0) or np.any(pred < 0):
        raise ValueError("nb_deviance expects non-negative count-like matrices")

    mu = np.clip(pred, a_min=eps, a_max=None)
    theta_vec = _estimate_nb_theta(obs, eps=eps) if theta is None else np.asarray(theta, dtype=np.float64)
    if theta_vec.ndim != 1 or theta_vec.shape[0] != obs.shape[1]:
        raise ValueError(f"theta must have shape ({obs.shape[1]},), got {theta_vec.shape}")
    theta_vec = np.clip(theta_vec, a_min=eps, a_max=1e8)[None, :]

    raw = _nb_deviance_raw(obs, mu, theta_vec)

    if reduce == "mean":
        return float(np.mean(np.maximum(raw.sum(axis=1), 0.0)))
    elif reduce == "per_cell":
        return np.maximum(raw.sum(axis=1), 0.0)
    else:  # per_gene
        return np.maximum(raw.sum(axis=0), 0.0)


def nb_deviance_pop_mean(
    obs_X: np.ndarray,
    pred_X: np.ndarray,
    theta: np.ndarray | None = None,
    eps: float = 1e-8,
) -> float:
    """
    Mean per-gene negative-binomial-style deviance between observed and predicted population means.

    Unlike ``nb_deviance``, this metric is intended for unpaired population comparisons and
    therefore allows different numbers of observed and predicted cells. Dispersion is estimated per gene
    from ``obs_X`` when ``theta`` is omitted.
    """
    obs = np.asarray(obs_X, dtype=np.float64)
    pred = np.asarray(pred_X, dtype=np.float64)
    if obs.ndim != 2 or pred.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got obs_X with shape {obs.shape} and pred_X with shape {pred.shape}")
    if obs.shape[1] != pred.shape[1]:
        raise ValueError(f"obs_X and pred_X must have the same number of genes, got {obs.shape[1]} and {pred.shape[1]}")
    if np.any(obs < 0) or np.any(pred < 0):
        raise ValueError("nb_deviance_pop_mean expects non-negative matrices")

    obs_mean = np.clip(obs.mean(axis=0), a_min=eps, a_max=None)
    pred_mean = np.clip(pred.mean(axis=0), a_min=eps, a_max=None)
    theta_vec = _estimate_nb_theta(obs, eps=eps) if theta is None else np.asarray(theta, dtype=np.float64)
    if theta_vec.ndim != 1 or theta_vec.shape[0] != obs.shape[1]:
        raise ValueError(f"theta must have shape ({obs.shape[1]},), got {theta_vec.shape}")
    theta_vec = np.clip(theta_vec, a_min=eps, a_max=1e8)

    term1 = np.where(obs_mean > 0, obs_mean * np.log(obs_mean / pred_mean), 0.0)
    term2 = (obs_mean + theta_vec) * np.log((obs_mean + theta_vec) / (pred_mean + theta_vec))
    deviance_per_gene = 2.0 * (term1 - term2)
    return float(np.mean(np.maximum(deviance_per_gene, 0.0)))


def edistance_pca(
    obs_X: np.ndarray,
    pred_X: np.ndarray,
    n_pcs: int = 50,
    reference_X: np.ndarray | None = None,
    n_subsample: int = 2000,
    n_iter: int = 10,
    random_state: int | None = 0,
) -> float:
    """
    Energy distance after PCA projection and deterministic cell subsampling.

    Inputs are expected to be log1p-normalized count matrices. PCA is fit on
    ``reference_X`` when provided, otherwise on ``obs_X``.
    """
    reference = obs_X if reference_X is None else reference_X
    n_components = min(int(n_pcs), reference.shape[0], reference.shape[1])

    center = reference.mean(axis=0)
    reference_centered = reference - center
    obs_centered = obs_X - center
    pred_centered = pred_X - center

    _, _, vt = np.linalg.svd(reference_centered, full_matrices=False)
    components = vt[:n_components].T
    obs_pca = obs_centered @ components
    pred_pca = pred_centered @ components
    rng = np.random.default_rng(random_state)
    edists = []
    for _ in range(int(n_iter)):
        obs_sample = _subsample_rows(obs_pca, n_subsample, rng)
        pred_sample = _subsample_rows(pred_pca, n_subsample, rng)
        edists.append(_energy_distance(obs_sample, pred_sample))
    return float(np.mean(edists))


def _nb_deviance_raw(obs: np.ndarray, mu: np.ndarray, theta_vec: np.ndarray) -> np.ndarray:
    """Return 2-D NB deviance array [n_cells, n_genes] without reduction."""
    term1 = np.where(obs > 0, obs * np.log(obs / mu), 0.0)
    term2 = (obs + theta_vec) * np.log((obs + theta_vec) / (mu + theta_vec))
    return 2.0 * (term1 - term2)


def _subsample_rows(x: np.ndarray, n_subsample: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= n_subsample:
        return x
    idx = rng.choice(x.shape[0], size=n_subsample, replace=False)
    return x[idx]


def _mean_pairwise_distance(a: np.ndarray, b: np.ndarray, chunk_size: int = 512) -> float:
    b_sq = np.sum(b * b, axis=1)
    total = 0.0
    for start in range(0, a.shape[0], chunk_size):
        block = a[start:start + chunk_size]
        d2 = np.sum(block * block, axis=1)[:, None] + b_sq[None, :] - 2 * block @ b.T
        np.maximum(d2, 0.0, out=d2)
        total += float(np.sqrt(d2).sum())
    return total / float(a.shape[0] * b.shape[0])


def _energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    value = 2 * _mean_pairwise_distance(a, b) - _mean_pairwise_distance(a, a) - _mean_pairwise_distance(b, b)
    return float(max(0.0, value))


def _estimate_nb_theta(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.asarray(x, dtype=np.float64).mean(axis=0)
    var = np.asarray(x, dtype=np.float64).var(axis=0)
    excess = np.maximum(var - mean, 0.0)
    theta = np.divide(mean * mean, excess + eps)
    return np.clip(theta, a_min=eps, a_max=1e8)


def _top_abs_idx(x: np.ndarray, n: int) -> np.ndarray:
    idx = np.argpartition(np.abs(x), -n)[-n:]
    return idx[np.argsort(np.abs(x[idx]))[::-1]]


# ---------------------------------------------------------------------------
# AnnData comparison API
# ---------------------------------------------------------------------------

_DEFAULT_METRICS = ["pearson_logfc", "rmse_logfc", "spearman_avg", "edistance_pca"]
_EFFECT_METRICS = {"pearson_logfc", "rmse_logfc", "precision_at_top_n"}
_POPULATION_METRICS = {"spearman_avg", "edistance_pca", "nb_deviance_pop_mean"}
_PAIRED_MATRIX_METRICS = {"nb_deviance"}
_AVAILABLE_METRICS = _EFFECT_METRICS | _POPULATION_METRICS | _PAIRED_MATRIX_METRICS
_LOGFC_BASELINES = {"predicted_control", "observed_control"}
_COUNT_SPACE_METRICS = {"nb_deviance_pop_mean", "nb_deviance"}

_METRIC_FN = {
    "pearson_logfc": pearson_logfc,
    "rmse_logfc": rmse_logfc,
    "spearman_avg": spearman_avg,
    "precision_at_top_n": precision_at_top_n,
    "edistance_pca": edistance_pca,
    "nb_deviance_pop_mean": nb_deviance_pop_mean,
    "nb_deviance": nb_deviance,
}


def compare_adatas(
    observed: AnnData,
    predicted: AnnData,
    pert_col: str,
    control_label: str,
    top_n: int = 50,
    metrics: list[str] | None = None,
    n_pcs: int = 50,
    layer: str | None = None,
    normalize: bool = True,
    predicted_logfc_baseline: Literal["predicted_control", "observed_control"] = "predicted_control",
) -> pd.DataFrame:
    """Compare observed and predicted perturbation responses on top-N observed-effect genes.

    Inputs are expected to be count-like (raw counts or decoder rates). When ``normalize``
    is True (default), log-space metrics (logFC-based, ``spearman_avg``, ``edistance_pca``) are
    computed on a ``CPM->1e4->log1p`` transform of the input, while count-space metrics
    (``nb_deviance``, ``nb_deviance_pop_mean``) always use the raw count-like input. Set
    ``normalize=False`` when the caller already supplies log-normalized data for the log-space
    metrics.
    """
    metric_names = _validate_inputs(
        observed,
        predicted,
        pert_col,
        control_label,
        top_n,
        metrics,
        predicted_logfc_baseline,
    )
    observed, predicted = _align_genes(observed, predicted)
    needs_count_metrics = bool(set(metric_names) & _COUNT_SPACE_METRICS)
    needs_log_population_metrics = bool(set(metric_names) & {"spearman_avg", "edistance_pca"})
    needs_effect_metrics = bool(set(metric_names) & _EFFECT_METRICS)

    control_mask = observed.obs[pert_col] == control_label
    obs_ctrl_mean = _log_matrix(observed, control_mask, layer, normalize).mean(axis=0)
    pred_ctrl_mean = None
    if needs_effect_metrics and predicted_logfc_baseline == "predicted_control":
        pred_control_mask = predicted.obs[pert_col] == control_label
        pred_ctrl_mean = _log_matrix(predicted, pred_control_mask, layer, normalize).mean(axis=0)
    rows = []
    labels = []

    for perturbation in pd.unique(observed.obs[pert_col]):
        if perturbation == control_label:
            continue

        obs_mask = observed.obs[pert_col] == perturbation
        pred_mask = predicted.obs[pert_col] == perturbation
        if not bool(obs_mask.any()) or not bool(pred_mask.any()):
            warnings.warn(f"Skipping perturbation {perturbation!r}: missing observed or predicted cells.", stacklevel=2)
            continue

        obs_log = _log_matrix(observed, obs_mask, layer, normalize)
        pred_log = _log_matrix(predicted, pred_mask, layer, normalize)

        obs_logfc = obs_log.mean(axis=0) - obs_ctrl_mean
        idx = _top_n_idx(obs_logfc, top_n)
        obs_logfc_sub = np.asarray(obs_logfc[idx], dtype=np.float64)

        if needs_effect_metrics:
            pred_baseline = pred_ctrl_mean if predicted_logfc_baseline == "predicted_control" else obs_ctrl_mean
            pred_logfc = pred_log.mean(axis=0) - pred_baseline
            pred_logfc_sub = np.asarray(pred_logfc[idx], dtype=np.float64)
        if needs_log_population_metrics:
            obs_log_sub = np.asarray(obs_log[:, idx], dtype=np.float64)
            pred_log_sub = np.asarray(pred_log[:, idx], dtype=np.float64)
        if needs_count_metrics:
            obs_counts = np.asarray(_get_matrix(observed, obs_mask, layer), dtype=np.float64)
            pred_counts = np.asarray(_get_matrix(predicted, pred_mask, layer), dtype=np.float64)
            obs_counts_sub = obs_counts[:, idx]
            pred_counts_sub = pred_counts[:, idx]

        row = {}
        for metric in metric_names:
            if metric == "pearson_logfc":
                row[metric] = pearson_logfc(obs_logfc_sub, pred_logfc_sub)
            elif metric == "rmse_logfc":
                row[metric] = rmse_logfc(obs_logfc_sub, pred_logfc_sub)
            elif metric == "precision_at_top_n":
                row[metric] = precision_at_top_n(obs_logfc, pred_logfc, n=top_n)
            elif metric == "spearman_avg":
                row[metric] = spearman_avg(obs_log_sub, pred_log_sub)
            elif metric == "edistance_pca":
                row[metric] = edistance_pca(obs_log_sub, pred_log_sub, n_pcs=n_pcs)
            elif metric == "nb_deviance_pop_mean":
                row[metric] = nb_deviance_pop_mean(obs_counts_sub, pred_counts_sub)
            elif metric == "nb_deviance":
                row[metric] = nb_deviance(obs_counts_sub, pred_counts_sub)
        rows.append(row)
        labels.append(perturbation)

    return pd.DataFrame(rows, index=pd.Index(labels, name=pert_col), columns=metric_names)


def get_perturbation_effects(
    observed: AnnData,
    predicted: AnnData,
    pert_col: str,
    control_label: str,
    *,
    mode: Literal["logfc", "average"] = "logfc",
    layer: str | None = None,
    predicted_logfc_baseline: Literal["predicted_control", "observed_control"] = "predicted_control",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the per-perturbation effects that :func:`compare_adatas` aggregates.

    This exposes the intermediate quantities — one row per perturbation, one
    column per (shared) gene — so they can be inspected or plotted directly
    (e.g. predicted-vs-observed scatter / histograms) instead of only seeing the
    final metric table.

    Parameters
    ----------
    observed, predicted
        Cell-level AnnDatas, grouped by ``pert_col``.
    pert_col, control_label
        Perturbation column in ``.obs`` and the control level within it.
    mode
        ``"logfc"`` returns mean expression minus the control mean (log fold-change);
        ``"average"`` returns the per-perturbation mean expression.

        Inputs are expected to be count-like; both observed and predicted are always
        ``CPM->1e4->log1p`` normalized so the returned effects live in the shared log space
        used by the scatter / histogram plots.
    layer
        Optional layer to read instead of ``.X``.
    predicted_logfc_baseline
        For ``mode="logfc"``, whether the predicted baseline is the predicted
        control mean (default) or the observed control mean.

    Returns
    -------
    (observed_effects, predicted_effects)
        Two DataFrames sharing the same index (perturbations, excluding the
        control) and columns (shared genes).
    """
    if mode not in {"logfc", "average"}:
        raise ValueError(f"mode must be 'logfc' or 'average', got {mode!r}")
    if pert_col not in observed.obs:
        raise ValueError(f"pert_col {pert_col!r} is not present in observed.obs")
    if pert_col not in predicted.obs:
        raise ValueError(f"pert_col {pert_col!r} is not present in predicted.obs")
    if len(observed.var_names.intersection(predicted.var_names)) == 0:
        raise ValueError("observed and predicted have no shared genes")

    observed, predicted = _align_genes(observed, predicted)
    gene_names = observed.var_names

    obs_ctrl_mean = pred_ctrl_mean = None
    if mode == "logfc":
        if not bool((observed.obs[pert_col] == control_label).any()):
            raise ValueError(f"control_label {control_label!r} is not present in observed.obs[{pert_col!r}]")
        obs_ctrl_mean = _log_matrix(observed, observed.obs[pert_col] == control_label, layer, True).mean(axis=0)
        if predicted_logfc_baseline == "predicted_control":
            if not bool((predicted.obs[pert_col] == control_label).any()):
                raise ValueError(
                    f"control_label {control_label!r} is required in predicted.obs[{pert_col!r}] "
                    "for mode='logfc' with predicted_logfc_baseline='predicted_control'"
                )
            pred_ctrl_mean = _log_matrix(predicted, predicted.obs[pert_col] == control_label, layer, True).mean(axis=0)
        else:
            pred_ctrl_mean = obs_ctrl_mean

    obs_rows, pred_rows, labels = [], [], []
    for perturbation in pd.unique(observed.obs[pert_col]):
        if perturbation == control_label:
            continue
        obs_mask = observed.obs[pert_col] == perturbation
        pred_mask = predicted.obs[pert_col] == perturbation
        if not bool(obs_mask.any()) or not bool(pred_mask.any()):
            warnings.warn(f"Skipping perturbation {perturbation!r}: missing observed or predicted cells.", stacklevel=2)
            continue

        obs_mean = _log_matrix(observed, obs_mask, layer, True).mean(axis=0)
        pred_mean = _log_matrix(predicted, pred_mask, layer, True).mean(axis=0)
        if mode == "logfc":
            obs_rows.append(np.asarray(obs_mean - obs_ctrl_mean, dtype=np.float64))
            pred_rows.append(np.asarray(pred_mean - pred_ctrl_mean, dtype=np.float64))
        else:
            obs_rows.append(np.asarray(obs_mean, dtype=np.float64))
            pred_rows.append(np.asarray(pred_mean, dtype=np.float64))
        labels.append(perturbation)

    index = pd.Index(labels, name=pert_col)
    observed_effects = pd.DataFrame(obs_rows, index=index, columns=gene_names)
    predicted_effects = pd.DataFrame(pred_rows, index=index, columns=gene_names)
    return observed_effects, predicted_effects


# ---------------------------------------------------------------------------
# Private AnnData helpers
# ---------------------------------------------------------------------------


def _get_matrix(adata: AnnData, mask, layer: str | None):
    subset = adata[mask]
    matrix = subset.X if layer is None else subset.layers[layer]
    return _to_dense(matrix)


def _log_matrix(adata: AnnData, mask, layer: str | None, normalize: bool) -> np.ndarray:
    """Count-like subset as a log-space matrix: ``_log1p_norm`` when ``normalize`` else as-is."""
    matrix = _get_matrix(adata, mask, layer)
    if normalize:
        return np.asarray(_log1p_norm(matrix), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _validate_inputs(
    observed: AnnData,
    predicted: AnnData,
    pert_col: str,
    control_label: str,
    top_n: int,
    metrics: list[str] | None,
    predicted_logfc_baseline: str,
) -> list[str]:
    if pert_col not in observed.obs:
        raise ValueError(f"pert_col {pert_col!r} is not present in observed.obs")
    if pert_col not in predicted.obs:
        raise ValueError(f"pert_col {pert_col!r} is not present in predicted.obs")
    if not bool((observed.obs[pert_col] == control_label).any()):
        raise ValueError(f"control_label {control_label!r} is not present in observed.obs[{pert_col!r}]")
    if len(observed.var_names.intersection(predicted.var_names)) == 0:
        raise ValueError("observed and predicted have no shared genes")
    metric_names = list(_DEFAULT_METRICS if metrics is None else metrics)
    unknown = [m for m in metric_names if m not in _AVAILABLE_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) {unknown}. Available metrics: {sorted(_AVAILABLE_METRICS)}"
        )
    if predicted_logfc_baseline not in _LOGFC_BASELINES:
        raise ValueError(
            f"predicted_logfc_baseline must be one of {sorted(_LOGFC_BASELINES)}, got {predicted_logfc_baseline!r}"
        )
    if (
        predicted_logfc_baseline == "predicted_control"
        and set(metric_names) & _EFFECT_METRICS
        and not bool((predicted.obs[pert_col] == control_label).any())
    ):
        raise ValueError(
            f"control_label {control_label!r} is required in predicted.obs[{pert_col!r}] for effect-based metrics"
        )
    return metric_names


def _align_genes(observed: AnnData, predicted: AnnData) -> tuple[AnnData, AnnData]:
    shared = observed.var_names.intersection(predicted.var_names)
    observed_drop = 1 - len(shared) / observed.n_vars
    predicted_drop = 1 - len(shared) / predicted.n_vars
    if max(observed_drop, predicted_drop) > 0.1:
        warnings.warn(
            f"Dropping {observed_drop:.1%} of observed genes and {predicted_drop:.1%} of predicted genes during alignment.",
            stacklevel=2,
        )
    return observed[:, shared], predicted[:, shared]

# TODO @captainPopec this is also where the scib / scGRAPH metrics should go.