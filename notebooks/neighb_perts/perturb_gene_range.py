"""
perturb_gene_range.py — sweep perturbation performance vs. number of perturbed genes.

Runs for a single slide and writes results/gene_range/<slide_id>/results.csv.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../scripts'))

import numpy as np
import pandas as pd
import decoupler as dc
import scanpy as sc
import scvi
from scipy.stats import pearsonr, spearmanr

import scipy.sparse as sp

import cellina
from cellina import CellinaModel, make_neighbor_perturbation
from cellina._spatial_utils import spatial_neighbors, compute_spatial_features
from perturb_utils import load_crc_slide, _get_domain_labels
from configs.cellina_config import MODEL_ARGS, TRAIN_ARGS, PLAN_KWARGS
from counterfactual_analysis import (
    safe_log2_fold_change, precision_at_k,
    e_distance, subsample_cells, _normalize_counts,
)

scvi.settings.seed = 0

EDISTANCE_SUBSAMPLE = 500
EDISTANCE_N_ITER    = 10
TOP_N_PERTURB_VALUES = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
_METRIC_KEYS = ('pearson_r', 'spearman_r', 'precision', 'direction_match',
                'edistance', 'edistance_local', 'rmse_log1p')


# ── Metric helpers ────────────────────────────────────────────────────────────

def direction_match(gt_vec, cf_vec, k):
    gt_topk = set(np.argsort(-np.abs(gt_vec))[:k])
    cf_topk = set(np.argsort(-np.abs(cf_vec))[:k])
    intersect = list(gt_topk & cf_topk)
    if len(intersect) == 0:
        return 0.0
    return float(np.mean(np.sign(gt_vec[intersect]) == np.sign(cf_vec[intersect])))


def compute_metrics(ref_expr, pert_expr, obs_expr, top_n=50, eps=1e-8, scale=1e4):
    ref_norm  = _normalize_counts(ref_expr,  eps=eps, scale=scale)
    obs_norm  = _normalize_counts(obs_expr,  eps=eps, scale=scale)
    pert_norm = _normalize_counts(pert_expr, eps=eps, scale=scale)

    mean_ref  = np.nanmean(ref_norm,  axis=0)
    mean_obs  = np.nanmean(obs_norm,  axis=0)
    mean_pert = np.nanmean(pert_norm, axis=0)

    gt_vec = safe_log2_fold_change(mean_obs,  mean_ref, eps=eps)
    cf_vec = safe_log2_fold_change(mean_pert, mean_ref, eps=eps)

    top_features = np.argsort(-np.abs(gt_vec))[:top_n]
    pear,  _ = pearsonr( gt_vec[top_features], cf_vec[top_features])
    spear, _ = spearmanr(gt_vec[top_features], cf_vec[top_features])
    prec     = precision_at_k(gt_vec, cf_vec, k=top_n, use_abs=True)
    dir_m    = direction_match(gt_vec, cf_vec, k=top_n)

    pop_a = np.log1p(obs_norm[:,  top_features])
    pop_b = np.log1p(pert_norm[:, top_features])
    edists = [
        e_distance(subsample_cells(pop_a, EDISTANCE_SUBSAMPLE),
                   subsample_cells(pop_b, EDISTANCE_SUBSAMPLE))
        for _ in range(EDISTANCE_N_ITER)
    ]
    edists_local = [
        e_distance(subsample_cells(pop_a, EDISTANCE_SUBSAMPLE),
                   subsample_cells(pop_b, EDISTANCE_SUBSAMPLE), local=True)
        for _ in range(EDISTANCE_N_ITER)
    ]
    rmse_log1p = float(np.sqrt(np.mean(
        (np.log1p(obs_norm).sum(0) - np.log1p(pert_norm).sum(0)) ** 2
    )))
    return dict(
        pearson_r=float(pear), spearman_r=float(spear),
        precision=float(prec), direction_match=float(dir_m),
        edistance=float(np.mean(edists)),
        edistance_local=float(np.mean(edists_local)),
        rmse_log1p=rmse_log1p,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--slide_id',   type=int, required=True)
    p.add_argument('--gpu',        type=int, default=0,
                   help='CUDA device index (after CUDA_VISIBLE_DEVICES remapping)')
    p.add_argument('--out_dir',    type=str, default='results/gene_range')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--min_cells',  type=int, default=50)
    p.add_argument('--top_n',      type=int, default=50,
                   help='Number of genes used for metric evaluation window')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    slide_id    = args.slide_id
    labels_key  = 'coarse_type'
    domains_key = 'typ'
    top_n       = args.top_n
    batch_size  = args.batch_size
    min_cells   = args.min_cells
    library_size = 'latent'

    results_dir = os.path.join(args.out_dir, str(slide_id))
    model_path  = os.path.join(args.out_dir, 'trained', f'crc_{slide_id}_ID')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f'=== slide {slide_id} | gpu {args.gpu} | out_dir {args.out_dir} ===')

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print('Loading data...')
    adata = load_crc_slide(slide_id, labels_key=labels_key, domains_key=domains_key)

    ref_label, crc_labels = _get_domain_labels(adata, domains_key)
    print(f'  ref={ref_label!r}, crc={crc_labels}')

    spatial_neighbors(adata, bandwidth=100 / 0.12028, max_neighbours=200, standardize=False)
    compute_spatial_features(adata)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    CellinaModel.setup_anndata(
        adata,
        batch_key=None,
        labels_key=labels_key,
        domains_key=domains_key,
        layer='counts',
        spatial_obsm_key='spatial_x',
    )
    model = CellinaModel(adata, **MODEL_ARGS)

    print(f'Training model for slide {slide_id}...')
    train_args = {
        **TRAIN_ARGS,
        'devices': [args.gpu],
        'batch_size': batch_size,
        'train_size': 0.9,
        'validation_size': 0.1,
    }
    model.train(**train_args, plan_kwargs=PLAN_KWARGS)
    model.save(model_path, overwrite=True)
    print(f'Model saved to {model_path}')

    # ── 3. Pseudobulk logFC — global & cell-type-specific ────────────────────
    print('Computing pseudobulk logFC...')
    pdata_global = dc.pp.pseudobulk(
        adata=adata, sample_col=domains_key, groups_col=None, mode='sum', layer='counts'
    )
    sc.pp.normalize_total(pdata_global, target_sum=1e4)
    sc.pp.log1p(pdata_global)
    _crc_X = pdata_global[pdata_global.obs[domains_key].isin(crc_labels)].X
    _ref_X = pdata_global[pdata_global.obs[domains_key] == ref_label].X
    _crc_mean = np.asarray(_crc_X.mean(axis=0)).flatten() if sp.issparse(_crc_X) else _crc_X.mean(axis=0).flatten()
    _ref_mean = np.asarray(_ref_X.mean(axis=0)).flatten() if sp.issparse(_ref_X) else _ref_X.mean(axis=0).flatten()
    global_logfc_series = pd.Series(_crc_mean - _ref_mean, index=pdata_global.var_names)

    pdata_ct = dc.pp.pseudobulk(
        adata=adata, sample_col=domains_key, groups_col=labels_key, mode='sum', layer='counts'
    )
    sc.pp.normalize_total(pdata_ct, target_sum=1e4)
    sc.pp.log1p(pdata_ct)

    cell_types_with_both = [
        ct for ct in pdata_ct.obs[labels_key].unique()
        if ((pdata_ct.obs[domains_key] == ref_label) & (pdata_ct.obs[labels_key] == ct)).any()
        and (pdata_ct.obs[domains_key].isin(crc_labels) & (pdata_ct.obs[labels_key] == ct)).any()
    ]
    _ct_rows = []
    for _ct in cell_types_with_both:
        _crc_ct = pdata_ct[pdata_ct.obs[domains_key].isin(crc_labels) & (pdata_ct.obs[labels_key] == _ct)].X
        _ref_ct = pdata_ct[(pdata_ct.obs[domains_key] == ref_label)   & (pdata_ct.obs[labels_key] == _ct)].X
        _crc_m  = np.asarray(_crc_ct.mean(axis=0)).flatten() if sp.issparse(_crc_ct) else _crc_ct.mean(axis=0).flatten()
        _ref_m  = np.asarray(_ref_ct.mean(axis=0)).flatten() if sp.issparse(_ref_ct) else _ref_ct.mean(axis=0).flatten()
        _ct_rows.append(pd.Series(_crc_m - _ref_m, index=pdata_ct.var_names, name=_ct))
    domain_logfc_df = pd.concat(_ct_rows, axis=1).T

    # ── 4. Pre-compute fixed expressions per cell type ────────────────────────
    print('Pre-computing per-cell-type expressions...')
    ref_idxs   = {}
    crc_idxs   = {}
    ref_exprs  = {}
    cf_exprs   = {}
    swap_exprs = {}
    cell_types = []

    for ct in sorted(cell_types_with_both):
        ref_idx = np.where(
            (adata.obs[labels_key] == ct) & (adata.obs[domains_key] == ref_label)
        )[0]
        crc_idx = np.where(
            (adata.obs[labels_key] == ct) & (adata.obs[domains_key].isin(crc_labels))
        )[0]
        if len(ref_idx) < min_cells or len(crc_idx) < min_cells:
            print(f'  skip {ct}: ref={len(ref_idx)}, crc={len(crc_idx)}')
            continue
        print(f'  {ct}: ref={len(ref_idx)}, crc={len(crc_idx)}')
        ref_idxs[ct]   = ref_idx
        crc_idxs[ct]   = crc_idx
        ref_exprs[ct]  = model.get_normalized_expression(
            indices=ref_idx, batch_size=batch_size, library_size=library_size)
        cf_exprs[ct]   = model.get_normalized_expression(
            indices=crc_idx, batch_size=batch_size, library_size=library_size)
        swap_exprs[ct] = model.get_counterfactual_expression(
            ref_idx, crc_idx, batch_size=batch_size, library_size=library_size)
        cell_types.append(ct)

    print(f'Evaluating {len(cell_types)} cell types: {cell_types}')

    # ── 5. Counterfactual baseline ─────────────────────────────────────────────
    print('Computing counterfactual baseline...')
    cf_metrics_vals = {k: [] for k in _METRIC_KEYS}
    for ct in cell_types:
        stats = compute_metrics(ref_exprs[ct], swap_exprs[ct], cf_exprs[ct], top_n=top_n)
        for k in _METRIC_KEYS:
            cf_metrics_vals[k].append(stats[k])

    # ── 6. Gene-count sweep ───────────────────────────────────────────────────
    print('Running sweep...')
    # Accumulate per-CT vals for each sweep point
    g_vals_all = []
    c_vals_all = []

    for n in TOP_N_PERTURB_VALUES:
        print(f'\n── top_n_perturb = {n} ──')

        # Global perturbation
        top_genes  = global_logfc_series.abs().nlargest(n).index.tolist()
        logfc_dict = {g: float(global_logfc_series[g]) for g in top_genes}
        make_neighbor_perturbation(adata, perturbations=logfc_dict,
                                   obsm_key_out='spatial_x_cf', base=np.e)
        g_vals = {k: [] for k in _METRIC_KEYS}
        for ct in cell_types:
            pert_expr = model.get_perturbed_expression(
                adata=adata, indices=ref_idxs[ct], spatial_obsm_key='spatial_x_cf',
                batch_size=batch_size, library_size=library_size,
            )
            stats = compute_metrics(ref_exprs[ct], pert_expr, cf_exprs[ct], top_n=top_n)
            for k in _METRIC_KEYS:
                g_vals[k].append(stats[k])
        g_vals_all.append(g_vals)
        print(f'  global:      Pearson r = {np.mean(g_vals["pearson_r"]):.3f}'
              f'  Spearman r = {np.mean(g_vals["spearman_r"]):.3f}'
              f'  prec = {np.mean(g_vals["precision"]):.3f}')

        # Cell-type-specific perturbation
        logfc_series_dict = {}
        for ct in domain_logfc_df.index:
            s    = domain_logfc_df.loc[ct]
            top_g = s.abs().nlargest(n).index.tolist()
            logfc_series_dict[ct] = s[top_g]
        make_neighbor_perturbation(
            adata, perturbations=logfc_series_dict, groupby=labels_key,
            obsm_key_out='spatial_x_cf', base=np.e,
        )
        c_vals = {k: [] for k in _METRIC_KEYS}
        for ct in cell_types:
            pert_expr = model.get_perturbed_expression(
                adata=adata, indices=ref_idxs[ct], spatial_obsm_key='spatial_x_cf',
                batch_size=batch_size, library_size=library_size,
            )
            stats = compute_metrics(ref_exprs[ct], pert_expr, cf_exprs[ct], top_n=top_n)
            for k in _METRIC_KEYS:
                c_vals[k].append(stats[k])
        c_vals_all.append(c_vals)
        print(f'  CT-specific: Pearson r = {np.mean(c_vals["pearson_r"]):.3f}'
              f'  Spearman r = {np.mean(c_vals["spearman_r"]):.3f}'
              f'  prec = {np.mean(c_vals["precision"]):.3f}')

    if 'spatial_x_cf' in adata.obsm:
        del adata.obsm['spatial_x_cf']

    # ── 7. Build tidy CSV ─────────────────────────────────────────────────────
    print('\nBuilding results CSV...')
    records = []

    # Baseline rows
    for j, ct in enumerate(cell_types):
        row = {k: cf_metrics_vals[k][j] for k in _METRIC_KEYS}
        row.update(strategy='edge_swap', cell_type=ct, top_n_perturb=float('nan'))
        records.append(row)
    mean_row = {k: float(np.mean(cf_metrics_vals[k])) for k in _METRIC_KEYS}
    mean_row.update(strategy='edge_swap', cell_type='mean', top_n_perturb=float('nan'))
    records.append(mean_row)

    # Sweep rows
    for i, n in enumerate(TOP_N_PERTURB_VALUES):
        for strategy, vals in [('global', g_vals_all[i]), ('ctspec', c_vals_all[i])]:
            for j, ct in enumerate(cell_types):
                row = {k: vals[k][j] for k in _METRIC_KEYS}
                row.update(strategy=strategy, cell_type=ct, top_n_perturb=n)
                records.append(row)
            mean_row = {k: float(np.mean(vals[k])) for k in _METRIC_KEYS}
            mean_row.update(strategy=strategy, cell_type='mean', top_n_perturb=n)
            records.append(mean_row)

    results_df = pd.DataFrame(records)
    out_path   = os.path.join(results_dir, 'results.csv')
    results_df.to_csv(out_path, index=False)
    print(f'Saved {len(results_df)} rows → {out_path}')


if __name__ == '__main__':
    main()
