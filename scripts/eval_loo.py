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
from counterfactual_analysis import safe_log2_fold_change, get_baseline_delta
import pandas as pd

DEFAULT_SEED = 0

# reuse preprocessing defaults from configs
sys.path.append('./scripts')
from configs.adata_config import ADATA_ARGS
from train_loo import preprocess_adata, split_indices, COUNTS_PER_K, DEFAULT_LABELS_KEY, DEFAULT_DOMAINS_KEY, DEFAULT_BATCH_KEY
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", required=True)
    p.add_argument("--model_class", required=True, choices=["cellina", "cpa", "cellina_graph", "baseline"]) 
    p.add_argument("--model_name", required=True)
    p.add_argument("--use_recon", action='store_true', default=True, help="Use reconstructions for DE (default True)")
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
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    r = sc.read(path)
    if getattr(r, 'X', None) is not None:
        count_mat = _to_dense(r.X)
    else:
        raise RuntimeError(f"No recognized expression matrix found in {path}")
    return count_mat


def compute_correlations(adata, holdout_celltype, use_recon=True, eps=1e-8, labels_key=DEFAULT_LABELS_KEY):
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
        X_all = X_all / (X_all.sum(axis=1, keepdims=True) + 1e-8) * COUNTS_PER_K

    # mean vectors
    if mask_control.sum() == 0 or mask_target.sum() == 0:
        raise RuntimeError('No control or no target cells for this holdout in adata')

    mean_control = np.nanmean(X_all[mask_control.values, :], axis=0)
    mean_target = np.nanmean(X_all[mask_target.values, :], axis=0)

    # counterfactual: may be aligned full matrix or subset stored in adata.uns['counterfactual_x']
    cf = adata.uns.get('counterfactual_x', None)
    if cf is None:
        raise RuntimeError('counterfactual_x not found in adata.uns')
    mean_cf = np.nanmean(cf, axis=0)

    # compute log2 fold changes
    gt_vec = safe_log2_fold_change(mean_target, mean_control, eps=eps)
    cf_vec = safe_log2_fold_change(mean_cf, mean_control, eps=eps)

    # pick finite entries - compute correlations
    valid = np.isfinite(gt_vec) & np.isfinite(cf_vec)
    pear, _ = pearsonr(gt_vec[valid], cf_vec[valid])
    spear, _ = spearmanr(gt_vec[valid], cf_vec[valid])

    return float(pear), float(spear)


def main():
    args = parse_args()

    adata_path = args.adata_path
    holdout_ct = args.holdout_celltype
    mc = args.model_class.lower()
    model_name = args.model_name
    use_recon = args.use_recon
    if mc=='cpa' or mc=='baseline':
        use_recon = False

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
    recon = load_model_predicted(recon_path) if mc != 'baseline' else None
    if recon is not None:
        adata.uns['recon_x'] = recon
        print('Loaded reconstructions into adata.uns["recon_x"] from', recon_path)

    # If not baseline, subset adata to relevant holdout cell type - we don't need entire adata for eval
    if mc != 'baseline':
        adata = adata[adata.obs[labels_key].astype(str) == holdout_ct]
    
    # If baseline mode, compute baseline counterfactual and skip loading model reconstructions
    if mc == 'baseline':
        print('Baseline mode: constructing baseline counterfactual using training-set CRC shift')
        # use all cell types present in adata to compute global shift
        labels_key = DEFAULT_LABELS_KEY
        all_cts = list(adata.obs[labels_key].dropna().unique())
        # get_baseline_delta does not require a model when use_recon=False
        delta = get_baseline_delta(
            adata,
            model=None,
            use_celltypes=all_cts,
            labels_col=labels_key,
            library_size="latent",
            normalize_counts=False,
            use_recon=use_recon,
            eps=1e-8,
        )
        # apply delta to control cells (holdout celltype & not CRC)
        mask_control = (~adata.obs[domains_key].astype(str).str.contains('CRC', regex=True)) & (adata.obs[labels_key].astype(str) == holdout_ct)
        if 'counts' not in adata.layers:
            raise RuntimeError('adata.layers["counts"] missing; cannot compute baseline counterfactual')
        counts = _to_dense(adata.layers['counts'])
        cf_matrix = counts[mask_control.values, :] + delta
        cf_matrix = cf_matrix / (cf_matrix.sum(axis=1, keepdims=True) + 1e-8) * COUNTS_PER_K
        # store only control-matching rows (compute_correlations can handle subset shape)
        adata.uns['counterfactual_x'] = cf_matrix
        print('Baseline counterfactual stored in adata.uns["counterfactual_x"]')
      
    elif mc == 'cpa':
        # for CPA, place recon of target cells into counterfactual field
        mask_target = adata.obs[domains_key].astype(str).str.contains('CRC', regex=True) & (adata.obs[labels_key].astype(str) == holdout_ct)
        if recon is None:
            raise RuntimeError('recon must be loaded for CPA fallback')
        cf_array = recon[mask_target.values, :]
        adata.uns['counterfactual_x'] = cf_array
        cf_loaded = True
        print('CPA mode: placed recon(target) into adata.uns["counterfactual_x"]')
    else:
        # for cellina-like, try to load dedicated counterfactual file if it exists
        if os.path.exists(cf_path):
            cf_matrix = load_model_predicted(cf_path)
            adata.uns['counterfactual_x'] = cf_matrix
            cf_loaded = True
            print('Loaded counterfactual into adata.uns["counterfactual_x"] from', cf_path)
        else:
            # fallback: if model_name contains 'cellina' attempt to look for recon file and use that as cf
            if 'cellina' in model_name.lower():
                # attempt: many cellina runs saved counterfactuals; if absent, raise informative warning
                print('Warning: counterfactual file not found; continuing without counterfactuals')

    if not cf_loaded and mc != 'baseline':
        raise FileNotFoundError(f"No counterfactual available for evaluation (tried {cf_path} and CPA fallback).")

    # compute correlations
    pear, spear = compute_correlations(adata, holdout_ct, use_recon=use_recon, labels_key=labels_key)

    # save results json
    out_dir = '/data2/a330d/datasets/crc/correlations'
    os.makedirs(out_dir, exist_ok=True)
    out_fname = f"{sid}_{model_name}_{holdout_ct}.json"
    out_path = os.path.join(out_dir, out_fname)
    with open(out_path, 'w') as fh:
        json.dump({'pearson': pear, 'spearman': spear}, fh)

    print('Saved correlations to', out_path)


if __name__ == '__main__':
    main()
