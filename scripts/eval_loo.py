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
PRECISION_AT_K = 50
EDISTANCE_SUBSAMPLE = 500

# reuse preprocessing defaults from configs
sys.path.append('./scripts')
from configs.adata_config import ADATA_ARGS
from train_loo import preprocess_adata, split_indices, COUNTS_PER_K, DEFAULT_LABELS_KEY, DEFAULT_DOMAINS_KEY, DEFAULT_BATCH_KEY
from utils import set_seed
from counterfactual_analysis import subsample_cells, e_distance, precision_at_k


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", required=True)
    p.add_argument("--model_class", required=True, choices=["cellina", "cpa", "cellina_graph", "baseline", "concert"]) 
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


def _normalize_counts(counts, eps=1e-8):
    return counts / (counts.sum(axis=1, keepdims=True) + eps) * COUNTS_PER_K


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
        X_all = _normalize_counts(X_all, eps=eps)

    # mean vectors
    if mask_control.sum() == 0 or mask_target.sum() == 0:
        raise RuntimeError('No control or no target cells for this holdout in adata')

    mean_control = np.nanmean(X_all[mask_control.values, :], axis=0)
    mean_target = np.nanmean(X_all[mask_target.values, :], axis=0)

    # counterfactual: may be aligned full matrix or subset stored in adata.uns['counterfactual_x']
    cf = adata.uns['counterfactual_x']
    mean_cf = np.nanmean(cf, axis=0)

    # compute log2 fold changes
    gt_vec = safe_log2_fold_change(mean_target, mean_control, eps=eps)
    cf_vec = safe_log2_fold_change(mean_cf, mean_control, eps=eps)

    deg_scores = np.abs(gt_vec)
    top_features = np.argsort(-deg_scores)[:PRECISION_AT_K]
    pear, _ = pearsonr(gt_vec[top_features], cf_vec[top_features])
    spear, _ = spearmanr(gt_vec[top_features], cf_vec[top_features])
    prec = precision_at_k(gt_vec, cf_vec, k=PRECISION_AT_K, use_abs=True)

    return float(pear), float(spear), float(prec), top_features


def get_edistance(adata, n_subsample=EDISTANCE_SUBSAMPLE, n_iter= 10, use_recon=True, deg=None, use_latents=False):
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
        # If using cell counts, e-distance is computed between target_observed and target_reconstructed
        observed_target = adata.layers["counts"][mask_target.values, :]
        observed_target = _to_dense(observed_target)
        observed_target = _normalize_counts(observed_target)

        pred_target = adata.uns['recon_x'][mask_target.values, :] if use_recon else adata.uns['counterfactual_x']
        pred_target = _normalize_counts(pred_target)

        top_features = deg if deg is not None else adata.n_vars
        pop_a = np.log1p(observed_target[:, top_features])
        pop_b = np.log1p(pred_target[:, top_features])
    edists = []
    for _ in range(n_iter):
        Xa_s = subsample_cells(pop_a, n_subsample)
        Xb_s = subsample_cells(pop_b, n_subsample)
        edist = e_distance(Xa_s, Xb_s)
        edists.append(edist)

    return np.mean(edists)


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
        delta = get_baseline_delta(
            adata,
            model=None,
            use_celltypes=all_cts,
            labels_col=labels_key,
            library_size="latent",
            normalize_counts=True,
            use_recon=use_recon,
            eps=1e-8,
        )
        # apply delta to control cells (holdout celltype & not CRC)
        mask_control = (~adata.obs[domains_key].astype(str).str.contains('CRC', regex=True)) & (adata.obs[labels_key].astype(str) == holdout_ct)
        if 'counts' not in adata.layers:
            raise RuntimeError('adata.layers["counts"] missing; cannot compute baseline counterfactual')
        counts = _to_dense(adata.layers['counts'])
        cf_matrix = counts[mask_control.values, :] + delta
        cf_matrix = np.clip(cf_matrix, a_min=0, a_max=None)
        cf_matrix = cf_matrix / (cf_matrix.sum(axis=1, keepdims=True) + 1e-8) * COUNTS_PER_K
        # store only control-matching rows (compute_correlations can handle subset shape)
        adata.uns['counterfactual_x'] = cf_matrix
        print('Baseline counterfactual stored in adata.uns["counterfactual_x"]')
      
    elif mc == 'cpa':
        # for CPA, place recon of target cells into counterfactual field
        mask_target = (adata.obs[domains_key].astype(str).str.contains('CRC', regex=True)) & (adata.obs[labels_key].astype(str) == holdout_ct)
        cf_array = recon[mask_target.values, :]
        adata.uns['counterfactual_x'] = cf_array
        cf_loaded = True
        print('CPA mode: placed recon(target) into adata.uns["counterfactual_x"]')
    else:
        # for cellina-like, load counterfactuals or recons depending on input arg
        try:
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
        except Exception as e:
            print(f"Could not load counterfactual from {cf_path}: {e}")
            cf_loaded = False

    if not cf_loaded and mc != 'baseline':
        raise FileNotFoundError(f"No counterfactual available for evaluation (tried {cf_path} and CPA fallback).")

    # compute correlations
    pear, spear, precision_at_k, deg = compute_correlations(adata, holdout_ct, use_recon=use_recon, labels_key=labels_key)

    # compute edistance between gt and predicted OOD populations - cell level
    edist_cells = None
    if mc != 'baseline':
        edist_recon = True if mc in ['cpa', 'cellina', 'cellina_graph'] else False
        edist_cells = get_edistance(adata, n_subsample=EDISTANCE_SUBSAMPLE, use_recon=edist_recon, deg=deg)

    # compute edistance between control and OOD populations - latent level
    edist_latents = None
    if mc != 'baseline':
        edist_latents = get_edistance(adata, n_subsample=EDISTANCE_SUBSAMPLE, use_latents=True)

    # save results json
    out_dir = '/data2/a330d/datasets/crc/correlations'
    os.makedirs(out_dir, exist_ok=True)
    model_name_save = model_name
    model_name_save += "-cf" if use_cf else ""
    model_name_save += "-recon" if use_recon else ""
    out_fname = f"{sid}_{model_name_save}_{holdout_ct}"
    out_path = os.path.join(out_dir, f"{out_fname}.json")
    with open(out_path, 'w') as fh:
        json.dump({'pearson': pear, 'spearman': spear, 'edistance_cells': edist_cells, 'edistance_latents': edist_latents, f'precision@{PRECISION_AT_K}': precision_at_k}, fh)

    print('Saved correlations to', out_path)


if __name__ == '__main__':
    main()
