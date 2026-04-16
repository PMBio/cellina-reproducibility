import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad

from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr


def make_counterfactual_adata(
    adata,
    indices_basal,
    indices_counterfactual,
    spatial_column,
    sample: bool = True,
    random_state: int = 0,
):
    """
    Create a counterfactual AnnData keeping everything from the original
    except .obsm[spatial_column], which is replaced with sampled spatial counts.

    Parameters
    ----------
    adata
        Original AnnData.
    indices_basal
        Indices of basal/control cells to keep in .X and obs.
    indices_counterfactual
        Indices of counterfactual cells to generate spatial counts from.
    spatial_column
        Column in .obsm containing spatial information (counts of neighbors).
    sample
        If True, generate NB-distributed counts per gene.
        If False, sample rows from existing neighboring cells with replacement.
    random_state
        Seed for reproducibility.

    Returns
    -------
    adata_cf : AnnData
        Copy of original AnnData with updated .obsm[spatial_column] for basal cells.
    """
    rng = np.random.default_rng(random_state)

    # 1. Subset basal cells
    adata_cf = adata[indices_basal].copy()

    # 2. Get spatial counts of counterfactual cells
    spatial_counts_cf = adata.obsm[spatial_column][indices_counterfactual]

    n_basal = len(indices_basal)
    n_genes = spatial_counts_cf.shape[1]

    # 3. Sampling: if true, compute representative NB dist and sample from it
    if sample:
        mu = spatial_counts_cf.mean(axis=0)
        var = spatial_counts_cf.var(axis=0)
        theta = np.maximum((mu**2) / (var - mu + 1e-8), 1e-8)

        spatial_counts_basal_cf = rng.negative_binomial(
            n=theta, p=theta / (theta + mu), size=(n_basal, n_genes)
        )
    # Otherwise just sample from existing neighbors with replacement
    else:
        indices = rng.integers(low=0, high=spatial_counts_cf.shape[0], size=n_basal)
        spatial_counts_basal_cf = spatial_counts_cf[indices]

    # 4. Replace spatial_column in .obsm
    adata_cf.obsm[spatial_column] = spatial_counts_basal_cf

    # 5. Keep original target cells to compare later if needed
    adata_cf.uns["target_cells"] = adata[indices_counterfactual].X.copy()

    return adata_cf


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
def get_model_preds(
    model,
    adata,
    labels_key,
    model_class,
    counterfactual=False,
    return_normalized=True,
    batch_size=4096,
):
    results = {}
    for celltype in tqdm(adata[adata.obs["is_holdout"]].obs[labels_key].cat.categories):
        mask_control = (~adata.obs["is_holdout"]) & (adata.obs[labels_key] == celltype)
        idx_control = np.where(mask_control.values)[0]
        adata_control = adata[mask_control].copy()

        mask_target = (adata.obs["is_holdout"]) & (adata.obs[labels_key] == celltype)
        idx_target = np.where(mask_target.values)[0]
        adata_target = adata[mask_target].copy()

        if model_class == "cellina":
            adata_cf = make_counterfactual_adata(
                adata,
                indices_basal=idx_control,
                indices_counterfactual=idx_target,
                spatial_column="spatial_x",
                sample=False,
            )
        else:
            adata_cf = (
                adata_target.copy()
            )  # CPA and scvi don't use spatial info, so just copy target

        def _to_array(x):
            # convert sparse/dense to numpy array
            if x is None:
                return None
            toarray = getattr(x, "toarray", None)
            if callable(toarray):
                return toarray()
            return np.asarray(x)

        def _reconstruct(adata_obj):
            # CPA: predict returns None and writes to adata_obj.obsm['CPA_pred']
            if model_class == "cpa":
                out = model.predict(adata_obj, batch_size=batch_size)  # may be None
                # common obsm keys CPA might use; extend if needed
                if ("CPA_pred") in adata_obj.obsm:
                    X = _to_array(adata_obj.obsm["CPA_pred"])
                    # normalize counts so each row sums to 1
                    if return_normalized:
                        X = np.log1p(X)
                        X = X / (X.sum(axis=1, keepdims=True) + 1e-8)
                    return X
                # if predict returned an array-like, use it
                if out is not None:
                    return _to_array(out)
                # fallback: nothing found
                raise RuntimeError(
                    "CPA predict produced no return and no known obsm key found; "
                    "inspect adata_obj.obsm keys: "
                    + ", ".join(list(adata_obj.obsm.keys()))
                )
            # other models: expect an array-like return
            else:
                # scvi / cellina use get_normalized_expression
                if hasattr(model, "get_normalized_expression"):
                    library_size = 1.0 if return_normalized else "latent"
                    out = model.get_normalized_expression(
                        adata_obj, library_size=library_size, batch_size=batch_size
                    )
                return _to_array(out)

        # Get normalized counterfactual / control / target expressions as numpy arrays
        adata_control.obsm["recon_x"] = _reconstruct(adata_control)
        adata_target.obsm["recon_x"] = _reconstruct(adata_target)
        adata_cf.obsm["recon_x"] = _reconstruct(adata_cf)

        # Get latent representations if applicable
        latents_control = model.get_latent_representation(adata=adata_control)
        adata_control.obsm[f"{model_class}_latent"] = (
            latents_control
            if model_class != "cpa"
            else latents_control["latent_after"].X
        )

        latents_target = model.get_latent_representation(adata=adata_target)
        adata_target.obsm[f"{model_class}_latent"] = (
            latents_target if model_class != "cpa" else latents_target["latent_after"].X
        )

        latents_cf = model.get_latent_representation(adata=adata_cf)
        adata_cf.obsm[f"{model_class}_latent"] = (
            latents_cf if model_class != "cpa" else latents_cf["latent_after"].X
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

        if not counterfactual:  # Return counterfactual if requested, otherwise return reconstructed target. Here to ensure compat with CPA
            adata_cf = adata_target.copy()
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


def compute_correlations(control, target, cf, deg=200, eps=1e-6):
    """
    Compute Pearson and Spearman correlations between observed and counterfactual LFCs.

    Parameters
    ----------
    control : np.ndarray (n_control_cells, n_genes), raw counts
    target  : np.ndarray (n_target_cells, n_genes), raw counts
    cf      : np.ndarray (n_control_cells, n_genes), counterfactual normalized expression
    deg     : int, number of top DEGs (by |observed LFC|) to evaluate on
    eps     : float, pseudocount for log2 fold change

    Returns
    -------
    pearson, spearman : float
    """
    control = np.asarray(control, dtype=float)
    target  = np.asarray(target,  dtype=float)
    cf      = np.asarray(cf,      dtype=float)

    # Normalize raw counts to library size 1e4
    control_norm = control / (control.sum(axis=1, keepdims=True) + eps) * 1e4
    target_norm  = target  / (target.sum(axis=1,  keepdims=True) + eps) * 1e4

    mean_control = control_norm.mean(axis=0)
    mean_target  = target_norm.mean(axis=0)
    mean_cf      = cf.mean(axis=0)

    gt_vec = safe_log2_fold_change(mean_target, mean_control, eps=eps)
    cf_vec = safe_log2_fold_change(mean_cf,     mean_control, eps=eps)

    top_features = np.argsort(-np.abs(gt_vec))[:deg]
    gt_top = gt_vec[top_features]
    cf_top = cf_vec[top_features]

    valid = np.isfinite(gt_top) & np.isfinite(cf_top)
    if valid.sum() < 2:
        return np.nan, np.nan

    pear,  _ = pearsonr( gt_top[valid], cf_top[valid])
    spear, _ = spearmanr(gt_top[valid], cf_top[valid])
    return float(pear), float(spear)


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


def _normalize_counts(x, eps=1e-8, scale=1e4):
        return x / (x.sum(axis=1, keepdims=True) + eps) * scale


def get_de_correlations(
    cf_adatas,
    k=50,
    eps=1e-8,
    method="lfc",
    plot=False,
    use_recon=False,
    normalize_counts=False,
):
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

        # masks
        mask_control = groups == "control"
        mask_target = groups == "target"
        mask_cf = groups == "counterfactual"

        # Counterfactuals/predictions are always model-generated
        recon_all = _to_dense(adata.obsm.get("recon_x").copy())
        #recon_all = np.log1p(recon_all)
        recon_all = (
            _normalize_counts(recon_all, eps=eps) if normalize_counts else recon_all
        )
        mean_cf = recon_all[mask_cf]
        mean_cf = mean_cf.mean(axis=0)

        # Control and target can be either from recon or raw counts
        X_all = (
            adata.obsm.get("recon_x").copy()
            if use_recon
            else adata.layers["counts"].copy()
        )
        X_all = _to_dense(X_all)  # ensure dense array
        #X_all = np.log1p(X_all)
        X_all = _normalize_counts(X_all, eps=eps) if normalize_counts else X_all

        # compute group means (ensure arrays)
        mean_control = X_all[mask_control].mean(axis=0)
        mean_target = X_all[mask_target].mean(axis=0)

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

        deg_scores = np.abs(gt_vec)
        top_features = np.argsort(-deg_scores)[:k]

        # compute Pearson and Spearman on finite entries
        valid = np.isfinite(gt_vec) & np.isfinite(cf_vec)
        if (
            valid.sum() < 2
            or (np.nanstd(gt_vec[valid]) == 0)
            or (np.nanstd(cf_vec[valid]) == 0)
        ):
            pear = np.nan
            spear = np.nan
        else:
            pear, _ = pearsonr(gt_vec[top_features], cf_vec[top_features])
            spear, _ = spearmanr(gt_vec[top_features], cf_vec[top_features])

        prec = precision_at_k(gt_vec, cf_vec, k=k, use_abs=True)
        results.append(
            {"celltype": ct, "pearson": pear, "spearman": spear, f"prec@{k}": prec}
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
            pearson = np.round(res["pearson"], 3) if res is not None else np.nan
            spearman = np.round(res["spearman"], 3) if res is not None else np.nan
            prec = np.round(res.get(f"prec@{k}"), 3) if res is not None else np.nan
            ax.set_title(
                f"{ct}\npearson={pearson} spearman={spearman}  prec@{k}={prec}"
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


def get_baseline_delta(
    adata,
    model,
    use_celltypes,
    labels_col,
    library_size="latent",
    normalize_counts=False,
    use_recon=False,
    eps=1e-8,
):
    # Take log fold change delta of in-sample control and CRC populations
    adata_control = adata[
        (adata.obs[labels_col].isin(use_celltypes))
        & (~adata.obs["typ"].str.contains("CRC"))
    ]
    adata_target = adata[
        (adata.obs[labels_col].isin(use_celltypes))
        & (adata.obs["typ"].str.contains("CRC"))
    ]
    if use_recon:
        x = model.get_normalized_expression(adata_control, library_size=library_size)
        y = model.get_normalized_expression(adata_target, library_size=library_size)
    else:
        x = adata_control.layers["counts"].toarray()
        y = adata_target.layers["counts"].toarray()
        if normalize_counts:
            # normalize to proportions
            x = _normalize_counts(x, eps=eps)
            y = _normalize_counts(y, eps=eps)

    # Compute shift vector from epithelial control to holdout
    delta = np.log2((y.mean(axis=0) + eps) / (x.mean(axis=0) + eps))

    return delta


def compare_observed_recon_lfc(adata, labels_key, recon_key="recon_x", eps=1e-8):
    agreements = {}
    for ct in tqdm(adata.obs[labels_key].unique()):
        adata_control = adata[
            (adata.obs[labels_key] == ct) & (~adata.obs["typ"].str.contains("CRC"))
        ]
        adata_target = adata[
            (adata.obs[labels_key] == ct) & (adata.obs["typ"].str.contains("CRC"))
        ]

        # DE between observed control and holdout
        mean_control = np.log1p(adata_control.layers["counts"].toarray()).mean(axis=0)
        mean_target = np.log1p(adata_target.layers["counts"].toarray()).mean(axis=0)
        gt_vec = safe_log2_fold_change(mean_target, mean_control, eps=eps)

        # DE between reconstructed control and holdout
        recon_control = np.log1p(adata_control.obsm.get(recon_key)).mean(axis=0)
        recon_target = np.log1p(adata_target.obsm.get(recon_key)).mean(axis=0)
        recon_vec = safe_log2_fold_change(recon_target, recon_control, eps=eps)

        pear, _ = pearsonr(gt_vec, recon_vec)
        spearman, _ = spearmanr(gt_vec, recon_vec)

        agreements[ct] = {"pearson": pear, "spearman": spearman}

    return agreements


def edist_observed_vs_recon(adata, labels_key, n_subsample=250, n_iter=10, recon_key="recon_x", deg=None):
    holdout_cts = adata.obs[labels_key][adata.obs["is_holdout"]].unique()
    edists = {}
    for ct in holdout_cts:
        target_sub = adata[
            (adata.obs[labels_key] == ct)
            & (adata.obs["typ"].str.contains("CRC"))
        ]
        control_sub = adata[
            (adata.obs[labels_key] == ct)
            & (~adata.obs["typ"].str.contains("CRC"))
        ]
        # Select features that are differentially expressed between control and holdout if deg arg supplied
        if deg is not None:
            mean_control = control_sub.layers["counts"].toarray().mean(axis=0)
            mean_target = target_sub.layers["counts"].toarray().mean(axis=0)
            deg_scores = np.abs(safe_log2_fold_change(mean_target, mean_control, eps=1e-8))
            top_features = np.argsort(-deg_scores)[:deg]  # top k DE genes
            #adata_sub = adata_sub[:, top_features]
        else:
            top_features = np.arange(target_sub.shape[1])  # all features
        
        # First log normalize both populations
        gt_pop = np.log1p(target_sub.layers["counts"].toarray()[:, top_features]) # control_sub.obsm[recon_key][:, top_features]
        recon_pop = np.log1p(target_sub.obsm[recon_key][:, top_features]) # target_sub.obsm[recon_key][:, top_features] 
        dists = []
        for i in tqdm(range(n_iter), desc=f"E-dist {ct}"):
            # Edist between populations
            Xa_s = subsample_cells(gt_pop, n_subsample)
            Xb_s = subsample_cells(recon_pop, n_subsample)
            edist = e_distance(Xa_s, Xb_s)
            dists.append(round(float(edist), 4))
        edists[ct] = {'mean': np.mean(dists), 'std': np.std(dists)}
    return edists
