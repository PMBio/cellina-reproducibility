import numpy as np

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans


def mixing_index(
    observed: np.ndarray,
    predicted: np.ndarray,
    n_clusters: int = 2,
    n_pcs: int = 50,
    random_state: int = 0,
    normalize_counts: bool = True,
    library_size: float = 1e4,
) -> float:
    """
    Mixing index: fraction of predicted cells correctly co-clustered with true cells.

    Reduces dimensionality with PCA, clusters with KMeans, then computes how
    many predicted cells land in clusters that contain the expected proportion of
    true cells.  A value of 1 indicates perfect mixing; 0 indicates no mixing.

    Parameters
    ----------
    X_pred
        Expression matrix for counterfactual (predicted) cells, shape (n_pred, n_genes).
    X_true
        Expression matrix for real target cells, shape (n_true, n_genes).
    n_clusters
        Number of KMeans clusters.
    n_pcs
        Number of PCA components before clustering.
    random_state
        Random seed.
    """
    observed = _normalize_counts(observed, scale=library_size) if normalize_counts else observed
    predicted = _normalize_counts(predicted, scale=library_size) if normalize_counts else predicted

    n_pred = predicted.shape[0]
    n_true = observed.shape[0]
    expected_proportion = n_pred / n_true

    # Joint PCA on stacked data
    X_combined = np.vstack([predicted, observed])
    n_pcs_eff = min(n_pcs, X_combined.shape[1], X_combined.shape[0] - 1)
    pca = PCA(n_components=n_pcs_eff, random_state=random_state)
    X_reduced = pca.fit_transform(X_combined)

    batch_labels = np.array(["pred"] * n_pred + ["true"] * n_true)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_reduced)

    n_correctly_mixed = 0
    for cluster_id in range(n_clusters):
        in_cluster = cluster_labels == cluster_id
        n_pred_in = (batch_labels[in_cluster] == "pred").sum()
        n_true_in = (batch_labels[in_cluster] == "true").sum()
        n_correctly_mixed += min(n_pred_in, n_true_in * expected_proportion)

    return n_correctly_mixed / n_pred


def _knn_masked_mean(kernel_matrix: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    """
    Mean of a kernel matrix restricted to each row's top-k neighbours.

    Parameters
    ----------
    kernel_matrix
        Square (n, n) similarity matrix (higher = more similar).
    k
        Number of neighbours to retain per row.
    exclude_self
        If True, mask out the diagonal before selecting top-k.
    """
    n = kernel_matrix.shape[0]
    mask = np.zeros_like(kernel_matrix, dtype=bool)

    for i in range(n):
        row = kernel_matrix[i].copy()

        if exclude_self:
            row[i] = -np.inf

        k_eff = min(k, kernel_matrix.shape[1] - int(exclude_self))

        # argpartition is faster than full sort for top-k
        top_idx = np.argpartition(-row, k_eff - 1)[:k_eff]

        mask[i, top_idx] = True

    return kernel_matrix[mask].mean()


def _pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Efficient Euclidean distance matrix using broadcasting.
    """
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)


def _standard_edistance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Standard energy distance between two sample sets.

    E(X, Y) = 2 * E[||x - y||] - E[||x - x'||] - E[||y - y'||]

    A value of 0 means the distributions are identical; larger values indicate
    greater distributional discrepancy.

    Parameters
    ----------
    X, Y
        2-D arrays of shape (n_cells, n_features).
    """
    D_xy = _pairwise_distances(X, Y)
    D_xx = _pairwise_distances(X, X)
    D_yy = _pairwise_distances(Y, Y)

    mean_dist_xy = D_xy.mean()
    mean_dist_xx = D_xx.mean()
    mean_dist_yy = D_yy.mean()

    return float(2 * mean_dist_xy - mean_dist_xx - mean_dist_yy)


def _local_edistance(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """
    Local energy distance restricted to each cell's k nearest neighbours.

    Uses negated Euclidean distances as a similarity kernel so that
    higher kernel values correspond to closer cells, making kNN selection
    consistent with _knn_masked_mean.

    Parameters
    ----------
    X, Y
        2-D arrays of shape (n_cells, n_features).
    k
        Number of nearest neighbours to consider per cell.
    """
    # Compute distances
    D_xx = _pairwise_distances(X, X)
    D_yy = _pairwise_distances(Y, Y)
    D_xy = _pairwise_distances(X, Y)

    # Convert to similarity (negated distance)
    K_xx = -D_xx
    K_yy = -D_yy
    K_xy = -D_xy
    K_yx = -D_xy.T

    mean_xx = _knn_masked_mean(K_xx, k, exclude_self=True)
    mean_yy = _knn_masked_mean(K_yy, k, exclude_self=True)
    mean_xy = _knn_masked_mean(K_xy, k, exclude_self=False)
    mean_yx = _knn_masked_mean(K_yx, k, exclude_self=False)

    return float(mean_xx + mean_yy - mean_xy - mean_yx)


def subsample_cells(X, n=200, seed=None):
    """Randomly subsample rows of a matrix."""
    rng = np.random.default_rng(seed)
    if X.shape[0] > n:
        idx = rng.choice(X.shape[0], n, replace=False)
        return X[idx]
    return X


def e_distance(X, Y, local=False, k=10):
    edist = None
    if local:
        edist = _local_edistance(X, Y, k=k)
    else:
        edist = _standard_edistance(X, Y)
    return edist


def compute_edistance(observed, predicted, normalize_counts=True, log1p=True, library_size=1e4, deg=None, n_iter=10, n_subsample=200, local=False):
    # Normalize observed counts, predicted is already normalized
    observed = _normalize_counts(observed, scale=library_size) if normalize_counts else observed

    top_features = deg if deg is not None else np.arange(observed.shape[1])
    pop_a = np.log1p(observed[:, top_features]) if log1p else observed[:, top_features]
    pop_b = np.log1p(predicted[:, top_features]) if log1p else predicted[:, top_features]

    edists = []
    for _ in range(n_iter):
        Xa_s = subsample_cells(pop_a, n_subsample)
        Xb_s = subsample_cells(pop_b, n_subsample)
        edist = e_distance(Xa_s, Xb_s, local=local)
        edists.append(edist)

    return np.mean(edists)


def permutation_test(X, Y, n_perms=1000, seed=None):
    """Permutation test for E-distance between X and Y."""
    rng = np.random.default_rng(seed)
    observed = e_distance(X, Y)
    combined = np.vstack([X, Y])
    n_x = X.shape[0]
    count = 0
    for _ in range(n_perms):
        rng.shuffle(combined)
        X_perm, Y_perm = combined[:n_x], combined[n_x:]
        if e_distance(X_perm, Y_perm) >= observed:
            count += 1
    pval = (count + 1) / (n_perms + 1)
    return pval


def _to_dense(mat):
    """Return a dense numpy array for adata.X-like objects (handles sparse)."""
    if mat is None:
        return None
    toarray = getattr(mat, "toarray", None)
    if callable(toarray):
        return toarray()
    return np.asarray(mat)


def safe_log2_fold_change(a, b, eps=1e-6):
    """
    Compute log2((a + eps) / (b + eps)) elementwise.
    Use this instead of log2(a - b). eps should be set relative to normalized scale.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.log2((a + eps) / (b + eps))


def precision(vec_true, vec_pred, k=20, use_abs=True):
    """
    Precision@k: fraction of top-k (by magnitude if use_abs) in vec_true
    that are also in top-k of vec_pred.
    """
    if use_abs:
        idx_true = np.argsort(-np.abs(vec_true))[:k]
        idx_pred = np.argsort(-np.abs(vec_pred))[:k]
    else:
        idx_true = np.argsort(-vec_true)[:k]
        idx_pred = np.argsort(-vec_pred)[:k]
    set_true = set(idx_true)
    set_pred = set(idx_pred)
    if len(set_true) == 0:
        return np.nan
    return len(set_true & set_pred) / len(set_true)


def _normalize_counts(x, eps=1e-8, scale=1e4):
    return x / (x.sum(axis=1, keepdims=True) + eps) * scale


def get_baseline_delta(
    adata_control,
    adata_target,
    normalize_counts=False,
    eps=1e-8,
):
    # Take log fold change delta of in-sample control and target populations    
    x = adata_control.layers["counts"].toarray()
    y = adata_target.layers["counts"].toarray()
    if normalize_counts:
        # normalize to proportions
        x = _normalize_counts(x, eps=eps)
        y = _normalize_counts(y, eps=eps)

    # Compute shift vector from epithelial control to holdout
    delta = np.log2((y.mean(axis=0) + eps) / (x.mean(axis=0) + eps))

    return delta


def compute_lfc_metrics(control, target, counterfactual, normalize_counts=True, n_deg=200, direction_match_normalize="intersection"):
    if normalize_counts:
        control = _normalize_counts(control)
        target = _normalize_counts(target)
        counterfactual = _normalize_counts(counterfactual)

    mean_control = np.nanmean(control, axis=0)
    mean_target = np.nanmean(target, axis=0)
    mean_cf = np.nanmean(counterfactual, axis=0)

    # compute log2 fold changes
    gt_vec = safe_log2_fold_change(mean_target, mean_control)
    cf_vec = safe_log2_fold_change(mean_cf, mean_control)

    deg_scores = np.abs(gt_vec)
    top_features = np.argsort(-deg_scores)[:n_deg]
    pear, _ = pearsonr(gt_vec[top_features], cf_vec[top_features])
    spear, _ = spearmanr(gt_vec[top_features], cf_vec[top_features])
    prec = precision(gt_vec, cf_vec, k=n_deg, use_abs=True)
    dir_match = direction_match(gt_vec, cf_vec, k=n_deg, normalize=direction_match_normalize)

    return pear, spear, prec, dir_match, top_features


def compute_rmse(observed, predicted, normalize_counts=True, log1p=True, deg=None, library_size=1e4):
    """
    Compute RMSE between psuedobulked observed and counterfactual counts for holdout cells.
    """
    # Subset to DE genes if deg provided; otherwise use all genes
    top_features = deg if deg is not None else np.arange(observed.shape[1])
    observed_target = observed[:, top_features]
    pred_target = predicted[:, top_features]

    if normalize_counts:
        observed_target = _normalize_counts(observed_target, scale=library_size)
        pred_target = _normalize_counts(pred_target, scale=library_size)
    if log1p:
        observed_target = np.log1p(observed_target)
        pred_target = np.log1p(pred_target)

    observed_pseudobulk = observed_target.sum(axis=0)
    pred_pseudobulk = pred_target.sum(axis=0)

    return np.sqrt(np.mean((observed_pseudobulk - pred_pseudobulk) ** 2))


def direction_match(gt_vec, cf_vec, k, normalize="intersection"):
    """
    Direction match between gt and cf.

    Parameters
    ----------
    normalize : str
        "intersection" -> divide by |intersection| (current behavior)
        "k"            -> divide by k
    """
    # Top-k sets (by absolute logFC)
    gt_topk = set(np.argsort(-np.abs(gt_vec))[:k])
    cf_topk = set(np.argsort(-np.abs(cf_vec))[:k])

    # Intersection
    intersect = list(gt_topk & cf_topk)

    if len(intersect) == 0:
        return 0.0  # or np.nan

    gt_sign = np.sign(gt_vec[intersect])
    cf_sign = np.sign(cf_vec[intersect])

    correct = np.sum(gt_sign == cf_sign)

    if normalize == "intersection":
        return correct / len(intersect)
    elif normalize == "k":
        return correct / k
    else:
        raise ValueError("normalize must be 'intersection' or 'k'")
