import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
from scipy.stats import pearsonr, spearmanr


def _normalize_counts(counts, counts_per_k=1e4, eps=1e-8):
    return counts / (counts.sum(axis=1, keepdims=True) + eps) * counts_per_k


def safe_log2_fold_change(a, b, eps=1e-6):
    """Compute log2((a + eps) / (b + eps)) elementwise."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.log2((a + eps) / (b + eps))


def compute_correlations(control, target, counterfactual, normalize_counts=True, deg=200):
    if normalize_counts:
        control = _normalize_counts(control)
        target = _normalize_counts(target)
        counterfactual = _normalize_counts(counterfactual)

    mean_control = np.nanmean(control, axis=0)
    mean_target = np.nanmean(target, axis=0)
    mean_cf = np.nanmean(counterfactual, axis=0)

    gt_vec = safe_log2_fold_change(mean_target, mean_control)
    cf_vec = safe_log2_fold_change(mean_cf, mean_control)

    deg_scores = np.abs(gt_vec)
    top_features = np.argsort(-deg_scores)[:deg]
    pear, _ = pearsonr(gt_vec[top_features], cf_vec[top_features])
    spear, _ = spearmanr(gt_vec[top_features], cf_vec[top_features])
    return float(pear), float(spear)


def subsample_adata(adata, fraction=0.1, random_state=42):
    n_subsample = int(adata.n_obs * fraction)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(adata.n_obs, n_subsample, replace=False)
    return adata[idx].copy()


def compute_microenv_logfc(adata, domains_key, labels_key, ref_label, crc_label):
    """Compute global and cell-type-specific logFC between a CRC microenvironment and reference."""
    adata.X = adata.layers['counts']
    pdata_global = dc.pp.pseudobulk(
        adata=adata, sample_col=domains_key, groups_col=None, mode='sum', layer='counts'
    )
    sc.pp.normalize_total(pdata_global, target_sum=1e4)
    sc.pp.log1p(pdata_global)

    global_logfc_series = pd.Series(
        (pdata_global[pdata_global.obs[domains_key] == crc_label].X
         - pdata_global[pdata_global.obs[domains_key] == ref_label].X).flatten(),
        index=pdata_global.var_names,
    )

    pdata_ct = dc.pp.pseudobulk(
        adata=adata, sample_col=domains_key, groups_col=labels_key, mode='sum', layer='counts'
    )
    sc.pp.normalize_total(pdata_ct, target_sum=1e4)
    sc.pp.log1p(pdata_ct)

    cell_types_with_both = [
        ct for ct in pdata_ct.obs[labels_key].unique()
        if ((pdata_ct.obs[domains_key] == ref_label) & (pdata_ct.obs[labels_key] == ct)).any()
        and ((pdata_ct.obs[domains_key] == crc_label) & (pdata_ct.obs[labels_key] == ct)).any()
    ]

    domain_logfc_df = pd.concat(
        [
            pd.Series(
                (pdata_ct[(pdata_ct.obs[domains_key] == crc_label) & (pdata_ct.obs[labels_key] == ct)].X
                 - pdata_ct[(pdata_ct.obs[domains_key] == ref_label) & (pdata_ct.obs[labels_key] == ct)].X
                ).flatten(),
                index=pdata_ct.var_names,
                name=ct,
            )
            for ct in cell_types_with_both
        ],
        axis=1,
    ).T

    return global_logfc_series, domain_logfc_df


def _pw_concordant_series(logfc_series, pw_df, logfc_threshold=0.5, weight_threshold=1):
    """Genes where logFC sign matches pathway weight sign (concordant direction)."""
    pw_weights = pw_df.set_index('target')['weight']
    merged = logfc_series.rename('logfoldchanges').to_frame().join(pw_weights, how='inner')
    mask = (
        ((merged['logfoldchanges'] >  logfc_threshold) & (merged['weight'] >  weight_threshold)) |
        ((merged['logfoldchanges'] < -logfc_threshold) & (merged['weight'] < -weight_threshold))
    )
    return set(merged.index[mask])


def _all_de_series(logfc_series, logfc_threshold=0.5):
    """Genes passing |logFC| threshold."""
    return set(logfc_series[logfc_series.abs() > logfc_threshold].index)


def build_perturbation_dict(ct_logfc_df, pw_dfs=None, filter_by_pathway=True,
                             logfc_threshold=0.5, weight_threshold=1):
    """Per-cell-type logFC DataFrame -> filtered pd.Series per cell type."""
    result = {}
    for ct in ct_logfc_df.index:
        logfc = ct_logfc_df.loc[ct]
        if filter_by_pathway and pw_dfs:
            genes = set().union(*[
                _pw_concordant_series(logfc, pw, logfc_threshold, weight_threshold)
                for pw in pw_dfs
            ])
        else:
            genes = _all_de_series(logfc, logfc_threshold)
        if not genes:
            continue
        result[ct] = logfc[list(genes)]
    return result


def cf_logfc(cf_expr, ctrl_expr):
    """log2(mean CF / mean control + 1) per gene, 1e4-normalised."""
    ctrl_n = ctrl_expr / (ctrl_expr.sum(1, keepdims=True) + 1e-9) * 1e4
    cf_n   = cf_expr   / (cf_expr.sum(1,  keepdims=True) + 1e-9) * 1e4
    return np.log2(cf_n.mean(0) + 1) - np.log2(ctrl_n.mean(0) + 1)
