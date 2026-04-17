"""
Evaluate leave-one-out model reconstructions for a single holdout cell type.

Usage:
python scripts/eval_loo.py --adata_path /path/to/crc_231.h5ad --holdout_celltype Epithelial --model_class cellina --model_name cond_z_False_sim_seed_0_ood

This script:
 - loads and preprocesses the input AnnData (same logic as train_loo)
 - loads reconstruction and optional counterfactual files produced by train_loo located at
   /data2/a330d/datasets/crc/<adata_basename>/<holdout_celltype>/{model_name}_{holdout}_recon_x.h5ad
   and counterfactual variant
 - aligns loaded reconstructions to the parent adata and stores them in adata.uns['recon_x']
   and adata.uns['counterfactual_x'] (when available)
 - computes log-fold-change vectors for target vs control and cf vs control and returns Pearson
   and Spearman correlations
 - writes results JSON to /data2/a330d/datasets/crc/correlations/<sid>_<model_name>_<holdout_celltype>.json

"""

import os
import sys
import json
import argparse
import numpy as np
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import pandas as pd

DEFAULT_SEED = 0
PRECISION_AT_K = 200
EDISTANCE_SUBSAMPLE = 500

# reuse preprocessing defaults from configs
sys.path.append('./scripts')
from configs.adata_config import ADATA_ARGS
from train_loo import preprocess_adata, split_indices, COUNTS_PER_K, DEFAULT_LABELS_KEY, DEFAULT_DOMAINS_KEY, DEFAULT_BATCH_KEY
from utils import set_seed
from counterfactual_analysis import safe_log2_fold_change, get_baseline_delta, subsample_cells, e_distance, precision_at_k, _normalize_counts, _mixing_index


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", required=True)
    p.add_argument("--model_class", required=True, choices=["cellina", "cpa", "cellina_graph", "baseline", "concert", "scgen"]) 
    p.add_argument("--model_name", required=True)
    p.add_argument("--use_recon", action='store_true', help="Use reconstructions for DE (default False)")
    p.add_argument("--use_cf", action='store_true', help="Use counterfactuals for DE (default False)")
    return p.parse_args()


def _to_dense(x):
    if x is None:
        return None
    toarray = getattr(x, 'toarray', None)
    if callable(toarray):
        return toarray()
    return np.asarray(x)


def load_model_predicted(path):
    """Load an AnnData stored at recon_path and align rows to parent_adata.obs_names.
    Returns a dense numpy array shaped (parent_adata.n_obs, parent_adata.n_vars) where rows
    not present in recon file are filled with np.nan.
    """
    adata = sc.read(path)
    latents = adata.obsm["latents"]

    return adata.X, latents


def direction_match(gt_vec, cf_vec, k):
    """
    Direction match computed ONLY on intersection of top-k DE genes
    (as defined in the STRAND paper).
    """
    # Top-k sets (by absolute logFC)
    gt_topk = set(np.argsort(-np.abs(gt_vec))[:k])
    cf_topk = set(np.argsort(-np.abs(cf_vec))[:k])

    # Intersection
    intersect = list(gt_topk & cf_topk)

    if len(intersect) == 0:
        return 0.0  # or np.nan depending on preference

    gt_sign = np.sign(gt_vec[intersect])
    cf_sign = np.sign(cf_vec[intersect])

    return np.mean(gt_sign == cf_sign)


def compute_lfc_metrics(adata, holdout_celltype, use_recon=True, eps=1e-8, labels_key=DEFAULT_LABELS_KEY):
    labels = adata.obs[labels_key].astype(str)
    # masks
    mask_control = (~adata.obs['is_holdout']) & (labels == holdout_celltype)
    mask_target = (adata.obs['is_holdout']) & (labels == holdout_celltype)

    if use_recon:
        recon = adata.uns.get('recon_x', None)
        if recon is None:
            raise RuntimeError('recon_x not found in adata.uns')
        X_all = recon
    else:
        if 'counts' not in adata.layers:
            raise RuntimeError('adata.layers["counts"] missing; cannot compute observed DE')
        X_all = _to_dense(adata.layers['counts'])
        X_all = _normalize_counts(X_all, eps=eps, scale=COUNTS_PER_K)

    # mean vectors
    if mask_control.sum() == 0 or mask_target.sum() == 0:
        raise RuntimeError('No control or no target cells for this holdout in adata')

    mean_control = np.nanmean(X_all[mask_control.values, :], axis=0)
    mean_target = np.nanmean(X_all[mask_target.values, :], axis=0)

    # counterfactual: may be aligned full matrix or subset stored in adata.uns['counterfactual_x']
    cf = adata.uns['counterfactual_x']
    cf = _normalize_counts(cf, eps=eps, scale=COUNTS_PER_K)
    mean_cf = np.nanmean(cf, axis=0)

    # compute log2 fold changes
    gt_vec = safe_log2_fold_change(mean_target, mean_control, eps=eps)
    cf_vec = safe_log2_fold_change(mean_cf, mean_control, eps=eps)

    deg_scores = np.abs(gt_vec)
    top_features = np.argsort(-deg_scores)[:PRECISION_AT_K]
    pear, _ = pearsonr(gt_vec[top_features], cf_vec[top_features])
    spear, _ = spearmanr(gt_vec[top_features], cf_vec[top_features])
    prec = precision_at_k(gt_vec, cf_vec, k=PRECISION_AT_K, use_abs=True)
    dir_match = direction_match(gt_vec, cf_vec, k=PRECISION_AT_K)
    """
    # Plot scatterplot of gt vs cf log fold changes - highlight top features in a different color
    import matplotlib.pyplot as plt
    plt.scatter(gt_vec, cf_vec)
    plt.xlabel('Ground Truth Log2 Fold Change')
    plt.ylabel('Counterfactual Log2 Fold Change')
    plt.title('GT vs CF Log2 Fold Change')
    
    # highlight top features in a different color
    plt.scatter(gt_vec[top_features], cf_vec[top_features], color='red')
    plt.show()
    """
    return (
        float(pear), 
        float(spear), 
        float(prec), 
        float(dir_match), 
        top_features
    )


def compute_rmse(adata, normalize_counts=True, log1p=True, deg=None):
    """
    Compute RMSE between psuedobulked observed and counterfactual counts for holdout cells.
    """
    mask_target = adata.obs['is_holdout']
    observed_target = adata.layers["counts"][mask_target.values, :]
    observed_target = _to_dense(observed_target)
    pred_target = adata.uns['counterfactual_x']

    # Subset to DE genes if deg provided; otherwise use all genes
    top_features = deg if deg is not None else np.arange(adata.n_vars)
    observed_target = observed_target[:, top_features]
    pred_target = pred_target[:, top_features]

    if normalize_counts:
        observed_target = _normalize_counts(observed_target, scale=COUNTS_PER_K)
        pred_target = _normalize_counts(pred_target, scale=COUNTS_PER_K)
    if log1p:
        observed_target = np.log1p(observed_target)
        pred_target = np.log1p(pred_target)

    observed_pseudobulk = observed_target.sum(axis=0)
    pred_pseudobulk = pred_target.sum(axis=0)

    return np.sqrt(np.mean((observed_pseudobulk - pred_pseudobulk) ** 2))


def get_edistance(adata, n_subsample=EDISTANCE_SUBSAMPLE, n_iter= 10, use_cf=True, deg=None, use_latents=False, local=False):
    mask_control = ~adata.obs['is_holdout']
    mask_target = adata.obs['is_holdout']
    if use_latents:
        # If using latents, e-distance is computed between control and target latents
        pop_a = adata.uns["latents"][mask_control.values, :]
        # If adata.uns['counterfactual_latents'] field exists, use it as pop_b; otherwise fall back to recon latents for target population
        if "counterfactual_latents" in adata.uns:
            pop_b = adata.uns["counterfactual_latents"]
            print("Using counterfactual latents for e-distance computation")
        else:
            pop_b = adata.uns["latents"][mask_target.values, :]
            print("Using target latents for e-distance computation")
    else:
        # If using cell counts, e-distance is computed between target_observed and target_recon or target_counterfactual
        observed_target = adata.layers["counts"][mask_target.values, :]
        observed_target = _to_dense(observed_target)
        observed_target = _normalize_counts(observed_target, scale=COUNTS_PER_K)

        pred_target = adata.uns['counterfactual_x'] if use_cf else adata.uns['recon_x'][mask_target.values, :]
        #pred_target = _normalize_counts(pred_target, scale=COUNTS_PER_K)

        top_features = deg if deg is not None else np.arange(adata.n_vars)
        pop_a = np.log1p(observed_target[:, top_features])
        pop_b = np.log1p(pred_target[:, top_features])
    
    edists = []
    for _ in range(n_iter):
        Xa_s = subsample_cells(pop_a, n_subsample)
        Xb_s = subsample_cells(pop_b, n_subsample)
        edist = e_distance(Xa_s, Xb_s, local=local)
        edists.append(edist)

    return np.mean(edists)


def mixing_index(adata, n_clusters=2, n_pcs=50, random_state=0):
    mask_target = adata.obs['is_holdout']

    observed_target = adata.layers["counts"][mask_target.values, :]
    observed_target = _to_dense(observed_target)
    observed_target = _normalize_counts(observed_target, scale=COUNTS_PER_K)

    pred_target = adata.uns['counterfactual_x']
    pred_target = _to_dense(pred_target)
    pred_target = _normalize_counts(pred_target, scale=COUNTS_PER_K)

    return _mixing_index(pred_target, observed_target, n_clusters=n_clusters, n_pcs=n_pcs, random_state=random_state)


def main():
    args = parse_args()

    adata_path = args.adata_path
    holdout_ct = args.holdout_celltype
    mc = args.model_class.lower()
    model_name = args.model_name
    use_recon = args.use_recon
    use_cf = args.use_cf

    set_seed(DEFAULT_SEED)
    # load adata and preprocess (same as train_loo)
    print('Loading adata', adata_path)
    adata = sc.read(adata_path)
    adata.obs_names_make_unique()

    n_top_genes = ADATA_ARGS.get('n_top_genes', 2000)
    n_neighbors = ADATA_ARGS.get('n_neighbors', 50)
    labels_key = ADATA_ARGS.get('labels_key', DEFAULT_LABELS_KEY)
    domains_key = ADATA_ARGS.get('domains_key', DEFAULT_DOMAINS_KEY)
    batch_key = ADATA_ARGS.get('batch_key', DEFAULT_BATCH_KEY)

    adata = preprocess_adata(adata, 
                             n_top_genes=n_top_genes, 
                             n_neighbors=n_neighbors,
                             )
    # only needed to generate obs['is_holdout'] for evaluation
    _ = split_indices(adata, 
                      args.holdout_celltype, 
                      labels_key=labels_key, 
                      domains_key=domains_key, 
                      seed=DEFAULT_SEED)

    # build expected paths for recon and counterfactual
    sid = os.path.splitext(os.path.basename(adata_path))[0]
    base_dir = os.path.join('/data2/a330d/datasets/crc', sid, holdout_ct)
    recon_fname = f"{model_name}_recon_x.h5ad"
    cf_fname = f"{model_name}_counterfactual_x.h5ad"
    recon_path = os.path.join(base_dir, recon_fname)
    cf_path = os.path.join(base_dir, cf_fname)

    # Load recons (if not baseline)
    cf_loaded = False
    recon, latents = None, None
    if mc != 'baseline':
        recon, latents = load_model_predicted(recon_path)
        adata.uns['recon_x'] = recon
        adata.uns['latents'] = latents
        print('Loaded reconstructions and latents into adata.uns["recon_x"] from', recon_path)
        # If not baseline, subset adata to relevant holdout cell type - we don't need entire adata for eval
        adata = adata[adata.obs[labels_key].astype(str) == holdout_ct]
    
    # If baseline mode, compute baseline counterfactual and skip loading model reconstructions
    if mc == 'baseline':
        print('Baseline mode: constructing baseline counterfactual using training-set CRC shift')
        # use all cell types present in adata to compute global shift
        labels_key = DEFAULT_LABELS_KEY
        all_cts = list(adata.obs[labels_key].dropna().unique())
        # get_baseline_delta does not require a model when use_recon=False
        adata_train = adata[~adata.obs["is_holdout"]].copy()
        delta = get_baseline_delta(
            adata_train,
            model=None,
            use_celltypes=all_cts,
            labels_col=labels_key,
            library_size="latent",
            use_recon=use_recon,
        )
        # apply delta to control cells (holdout celltype & not CRC)
        mask_control = (~adata.obs[domains_key].astype(str).str.contains('CRC', regex=True)) & (adata.obs[labels_key].astype(str) == holdout_ct)
        if 'counts' not in adata.layers:
            raise RuntimeError('adata.layers["counts"] missing; cannot compute baseline counterfactual')
        counts = _to_dense(adata.layers['counts'])
        # For log2, but we compute delta on log1 so just take exp
        #cf_matrix = counts[mask_control.values, :] * (2 ** delta)
        cf_matrix = (counts[mask_control.values, :] + 1) * np.exp(delta) - 1
        cf_matrix = np.clip(cf_matrix, a_min=0, a_max=None)
        cf_matrix = cf_matrix / (cf_matrix.sum(axis=1, keepdims=True) + 1e-8) * COUNTS_PER_K
        # store only control-matching rows (compute_correlations can handle subset shape)
        adata.uns['counterfactual_x'] = cf_matrix
        print('Baseline counterfactual stored in adata.uns["counterfactual_x"]')
    # If not baseline - works for cellina-like, cpa, scgen
    else:
        if use_cf:
            cf_matrix, cf_latents = load_model_predicted(cf_path)
            print('Loaded counterfactual into adata.uns["counterfactual_x"] from', cf_path)
        else:
            mask_target = (adata.obs[domains_key].astype(str).str.contains('CRC', regex=True)) & (adata.obs[labels_key].astype(str) == holdout_ct)
            cf_matrix = recon[mask_target.values, :]
            cf_latents = latents[mask_target.values, :]
            print('Loaded recons into adata.uns["counterfactual_x"]')
        adata.uns['counterfactual_x'] = cf_matrix
        adata.uns['counterfactual_latents'] = cf_latents
        cf_loaded = True

    if not cf_loaded and mc != 'baseline':
        raise FileNotFoundError(f"No counterfactual available for evaluation (tried {cf_path} and CPA fallback).")

    # 1. compute correlations
    pear, spear, prec, dir_match, deg = compute_lfc_metrics(adata, 
                                                                     holdout_ct, 
                                                                     use_recon=use_recon, 
                                                                     labels_key=labels_key)

    # 2. compute edistance between observed and counterfactual OOD populations - cell level
    edist_cells = None
    edist_cells = get_edistance(adata, 
                                n_subsample=EDISTANCE_SUBSAMPLE, 
                                use_cf=use_cf, 
                                deg=deg)

    # 3. compute edistance between control and OOD populations - latent level
    edist_latents = None
    # skip latent edistance if either 1) mc is baseline or 2) sid is 120 and mc is scgen - only have partial recons for scgen-120 (probably) because of RAM/VRAM
    if mc == 'baseline' or (sid == 'crc_120' and mc == 'scgen'):
        print("Skipping latent edistances")
    else:
        edist_latents = get_edistance(adata, 
                                      n_subsample=EDISTANCE_SUBSAMPLE, 
                                      use_latents=True)

    # 4. compute local edistance between observed and counterfactual OOD populations - cell level
    edist_local = None
    edist_local = get_edistance(adata, 
                                n_subsample=EDISTANCE_SUBSAMPLE, 
                                use_cf=use_cf, 
                                deg=deg, 
                                local=True)

    # 5. compute mixing index
    mix_idx = mixing_index(adata, 
                           n_clusters=2, 
                           n_pcs=50, 
                           random_state=DEFAULT_SEED)

    # 6. compute RMSE
    rmse = compute_rmse(adata, normalize_counts=False, log1p=False, deg=None)
    rmse_log1p = compute_rmse(adata, normalize_counts=False, log1p=True, deg=None)

    # save results json
    out_dir = '/data2/a330d/datasets/crc/correlations'
    os.makedirs(out_dir, exist_ok=True)
    model_name_save = model_name
    model_name_save += "-cf" if use_cf else ""
    model_name_save += "-recon" if use_recon else ""
    out_fname = f"{sid}_{model_name_save}_{holdout_ct}"
    out_path = os.path.join(out_dir, f"{out_fname}.json")
    with open(out_path, 'w') as fh:
        json.dump({'pearson': pear, 
                   'spearman': spear, 
                   'edistance_cells': edist_cells, 
                   'edistance_latents': edist_latents,
                   'edistance_local': edist_local,
                   'mixing_index': mix_idx,
                   f'precision@{PRECISION_AT_K}': prec,
                   f'direction_match@{PRECISION_AT_K}': dir_match,
                   'rmse': rmse,
                   'rmse_log1p': rmse_log1p
                   }, fh)

    print('Saved correlations to', out_path)


if __name__ == '__main__':
    main()
