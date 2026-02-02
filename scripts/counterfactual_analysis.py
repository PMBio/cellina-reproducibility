import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad

from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

from cellina._utils import make_counterfactual_adata


def prepare_matrix(M, n_pca=50, standardize=False):
    """Optional: standardize and reduce dimensionality before computing distances."""
    M = np.asarray(M)
    if standardize:
        M = StandardScaler(with_mean=True, with_std=True).fit_transform(M)
    if n_pca is not None and M.shape[1] > n_pca:
        M = PCA(n_components=n_pca, random_state=0).fit_transform(M)
    return M


def subsample_cells(X, n=200, seed=None):
    """Randomly subsample rows of a matrix."""
    rng = np.random.default_rng(seed)
    if X.shape[0] > n:
        idx = rng.choice(X.shape[0], n, replace=False)
        return X[idx]
    return X


def e_distance(X, Y):
    """Energy distance using pdist (no self-distances) for within-group terms."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    # cross distances
    d_xy = cdist(X, Y, metric="euclidean")
    # within-group distances excluding self-pairs
    d_xx = pdist(X, metric="euclidean")
    d_yy = pdist(Y, metric="euclidean")
    return 2 * np.mean(d_xy) - np.mean(d_xx) - np.mean(d_yy)


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


# NOTE: Bad for memory, a lot of redundancy in adatas - reimplement later
def get_counterfactual_preds(model, adata, labels_key, model_class):
    results = {}
    for celltype in tqdm(adata[adata.obs["is_holdout"]].obs[labels_key].cat.categories):
        mask_control = (~adata.obs["is_holdout"]) & (adata.obs[labels_key] == celltype)
        idx_control = np.where(mask_control.values)[0]
        mask_target = (adata.obs["is_holdout"]) & (adata.obs[labels_key] == celltype)
        idx_target = np.where(mask_target.values)[0]

        adata_cf = make_counterfactual_adata(
            adata,
            indices_basal=idx_control,
            indices_counterfactual=idx_target,
            spatial_column="spatial_x",
            sample=False,
        )

        # Get normalized counterfactual expression
        adata_cf.obsm["recon_x"] = model.get_normalized_expression(adata_cf)

        # Get normalized ground truth control expressions
        adata_control = adata[mask_control].copy()
        adata_control.obsm["recon_x"] = model.get_normalized_expression(adata_control)

        # Get normalized ground truth target expressions
        adata_target = adata[mask_target].copy()
        adata_target.obsm["recon_x"] = model.get_normalized_expression(adata_target)

        # Get latent representations if applicable
        adata_cf.obsm[f"{model_class}_latent"] = model.get_latent_representation(
            adata=adata_cf
        )
        adata_control.obsm[f"{model_class}_latent"] = model.get_latent_representation(
            adata=adata_control
        )
        adata_target.obsm[f"{model_class}_latent"] = model.get_latent_representation(
            adata=adata_target
        )

        if model_class == "cellina":
            for latent_key in ["z", "s"]:
                adata_cf.obsm[latent_key] = model.get_latent_representation(
                    adata=adata_cf, latent_key=latent_key
                )
                adata_control.obsm[latent_key] = model.get_latent_representation(
                    adata=adata_control, latent_key=latent_key
                )
                adata_target.obsm[latent_key] = model.get_latent_representation(
                    adata=adata_target, latent_key=latent_key
                )

        adata_cf.obs["group"] = "counterfactual"
        adata_control.obs["group"] = "control"
        adata_target.obs["group"] = "target"
        adata_merged = ad.concat([adata_cf, adata_control, adata_target])

        results[celltype] = adata_merged

    return results


def get_edistances(
    adata_dict, model_class, n_subsample=500, n_perms=500, compute_pvals=False
):
    groups = adata_dict["Endothelial"].obs["group"].unique()
    pairs = [
        ("counterfactual", "control"),
        ("counterfactual", "target"),
        ("control", "target"),
    ]
    # pair_names = [f"{a[:3]}-{b[:3]}" for a, b in pairs]  # short column names
    pair_names = [f"{a}-{b}" for a, b in pairs]  # short column names

    celltypes = list(adata_dict.keys())
    edist_df = pd.DataFrame(index=celltypes, columns=pair_names, dtype=float)
    pval_df = pd.DataFrame(index=celltypes, columns=pair_names, dtype=float)

    for ct in tqdm(celltypes, desc="celltypes (E-dist)"):
        ad = adata_dict[ct]
        groups = (
            ad.obs["group"].cat.categories
            if hasattr(ad.obs["group"], "cat")
            else np.unique(ad.obs["group"])
        )
        groups = list(groups)

        # prepare matrices for expected groups
        Xg = {}
        for g in ["counterfactual", "control", "target"]:
            if g in groups:
                M = ad[ad.obs["group"] == g].obsm.get(f"{model_class}_latent", None)
                if M is None:
                    # fallback to recon_x if latent missing
                    M = ad[ad.obs["group"] == g].obsm.get("recon_x", None)
                if M is None:
                    Xg[g] = np.zeros((0, 1))
                else:
                    Xg[g] = prepare_matrix(M, n_pca=None)
            else:
                Xg[g] = np.zeros((0, 1))

        for (a, b), col in zip(pairs, pair_names):
            Xa = Xg[a]
            Xb = Xg[b]
            if Xa.shape[0] < 2 or Xb.shape[0] < 2:
                edist_df.loc[ct, col] = np.nan
                pval_df.loc[ct, col] = np.nan
                continue

            Xa_s = subsample_cells(Xa, n_subsample, seed=0)
            Xb_s = subsample_cells(Xb, n_subsample, seed=0)

            ed = e_distance(Xa_s, Xb_s)
            edist_df.loc[ct, col] = ed

            if compute_pvals:
                p = permutation_test(Xa_s, Xb_s, n_perms=n_perms, seed=0)
                pval_df.loc[ct, col] = p
            else:
                pval_df.loc[ct, col] = np.nan
    return edist_df, pval_df


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


def safe_log2_delta(a, b, eps=1e-6):
    """
    Compute log2(a - b) safely:
      delta = a - b
      clip delta to >= eps to avoid log of <=0
    Returns vector of same length.
    """
    delta = np.asarray(a) - np.asarray(b)
    delta_clipped = np.maximum(delta, eps)
    return np.log2(delta_clipped)


def precision_at_k(vec_true, vec_pred, k=20, use_abs=True):
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


def get_de_correlations(cf_adatas, k=50, eps=1e-6, method="lfc", plot=False):
    """
    For each cell type (adata in cf_adatas), compute:
      - gt_vec = log2(mean(target).X - mean(control).X)  <-- uses normalized adata.X
      - cf_vec = log2(mean(counterfactual).obsm['recon_x'] - mean(control).X)
    Then compute Pearson, Spearman and precision@k between gt_vec and cf_vec.
    Returns (results_list, vectors_dict)
    """
    results = []
    vectors = {}

    for ct, adata in tqdm(cf_adatas.items()):
        groups = adata.obs["group"].values
        var_names = np.asarray(adata.var_names)

        # get counts as dense array
        counts = _to_dense(adata.layers["counts"])  # shape: (n_cells, n_genes)

        # normalize counts so each row sums to 1
        # X_all = np.asarray(adata.obsm.get('recon_x'))
        X_all = counts / (counts.sum(axis=1, keepdims=True) + 1e-8)
        recon_all = _to_dense(
            adata.obsm.get("recon_x")
        )  # model-normalized counterfactuals (may be None)

        # masks
        mask_control = groups == "control"
        mask_target = groups == "target"
        mask_cf = groups == "counterfactual"

        if mask_control.sum() == 0:
            # can't compute without real control baseline
            continue
        if mask_target.sum() == 0 and mask_cf.sum() == 0:
            # nothing to compare
            continue

        # compute group means (ensure arrays)
        mean_control = (
            X_all[mask_control].mean(axis=0)
            if mask_control.sum() > 0
            else np.zeros(X_all.shape[1])
        )
        mean_target = (
            X_all[mask_target].mean(axis=0)
            if mask_target.sum() > 0
            else np.zeros(X_all.shape[1])
        )
        mean_cf = None
        if recon_all is not None and mask_cf.sum() > 0:
            # recon_all rows are aligned with adata.obs order
            mean_cf = recon_all[mask_cf].mean(axis=0)
        else:
            # fallback: if recon_x missing, try to use X for CF (not recommended)
            mean_cf = (
                X_all[mask_cf].mean(axis=0)
                if mask_cf.sum() > 0
                else np.zeros(X_all.shape[1])
            )

        # compute gt and cf: observed perturbed (target) minus real control, and counterfactual minus real control
        if method == "lfc":
            diff_method = safe_log2_fold_change
        elif method == "delta":
            diff_method = safe_log2_delta
        else:
            raise ValueError(f"Unknown method: {method}")

        try:
            gt_vec = diff_method(mean_target, mean_control, eps=eps)
            cf_vec = diff_method(mean_cf, mean_control, eps=eps)
        except Exception:
            # numeric issues: skip
            continue

        vectors[ct] = {"gt": gt_vec, "cf": cf_vec, "genes": var_names}

        # compute Pearson and Spearman on finite entries
        valid = np.isfinite(gt_vec) & np.isfinite(cf_vec)
        if (
            valid.sum() < 2
            or (np.nanstd(gt_vec[valid]) == 0)
            or (np.nanstd(cf_vec[valid]) == 0)
        ):
            pear = np.nan
            spearman = np.nan
        else:
            pear, _ = pearsonr(gt_vec[valid], cf_vec[valid])
            spearman, _ = spearmanr(gt_vec[valid], cf_vec[valid])

        prec = precision_at_k(gt_vec, cf_vec, k=k, use_abs=True)
        results.append(
            {"celltype": ct, "pearson": pear, "spearman": spearman, f"prec@{k}": prec}
        )

        # plotting: create multiplot grid with 3 columns per row, one scatter per cell type
    if plot and len(results) > 0:
        import math

        ordered_cts = [r["celltype"] for r in results]
        n = len(ordered_cts)
        ncols = 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
        )
        axes_flat = axes.flatten()

        for i, ct in enumerate(ordered_cts):
            ax = axes_flat[i]
            gt_vec = vectors[ct]["gt"]
            cf_vec = vectors[ct]["cf"]
            var_names = vectors[ct]["genes"]

            valid = np.isfinite(gt_vec) & np.isfinite(cf_vec)
            # plot all genes (use valid mask for plotting extents)
            ax.scatter(
                gt_vec, cf_vec, s=10, alpha=0.45, color="gray", edgecolors="none"
            )
            # highlight top-k GT genes
            top_idx = np.argsort(-np.abs(gt_vec))[:k]
            ax.scatter(
                gt_vec[top_idx],
                cf_vec[top_idx],
                s=20,
                color="red",
                alpha=0.8,
                label=f"top-{k} GT",
            )

            # diagonal
            try:
                mn = np.nanmin([gt_vec[valid].min(), cf_vec[valid].min()])
                mx = np.nanmax([gt_vec[valid].max(), cf_vec[valid].max()])
            except Exception:
                mn, mx = 0, 1
            ax.plot([mn, mx], [mn, mx], color="black", linewidth=0.8, linestyle="--")

            # metrics for title
            res = next((r for r in results if r["celltype"] == ct), None)
            pear = res["pearson"] if res is not None else np.nan
            prec = res.get(f"prec@{k}") if res is not None else np.nan
            ax.set_title(
                f"{ct}\npearson={np.round(pear, 3) if not pd.isna(pear) else 'nan'}  prec@{k}={np.round(prec, 3) if not pd.isna(prec) else 'nan'}"
            )
            ax.set_xlabel(f"gt (log2 {method})")
            ax.set_ylabel(f"cf (log2 {method})")
            ax.legend(frameon=False, fontsize=8)

        # turn off any unused axes
        for j in range(n, nrows * ncols):
            axes_flat[j].axis("off")

        plt.tight_layout()
        plt.show()

    return results, vectors
