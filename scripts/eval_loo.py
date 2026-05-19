"""
Evaluate leave-one-celltype-out model reconstructions for a single holdout cell type.

Usage:
python scripts/eval_loo.py --dataset_name crc --adata_path <path/to/adata> --holdout_celltype Fibroblast --model_class cellina --model_name cellina --use_cf

Explain what this script does in bullet points:
- Loads an AnnData object containing spatial single-cell data for a given holdout cell type
- For each holdout domain (e.g. donor region), loads model-predicted counterfactual gene expression profiles for the holdout cell type in that domain
- Computes evaluation metrics comparing the counterfactual predictions to the observed data in the holdout domain, using a control set
- Saves the evaluation results (correlation, precision, direction match, mixing index, edistance, rmse) to a JSON file in a specified output directory
"""

import os
import sys
import json
import argparse
import numpy as np
import scanpy as sc

DATA_ROOT = os.environ.get("DATA_ROOT", ".")

from scipy.stats import pearsonr, spearmanr

DEFAULT_SEED = 0
N_DEG = 50
CRC_INFERENCE_BASE_DIR = os.path.join(DATA_ROOT, "datasets/crc")
MERFISH_INFERENCE_BASE_DIR = os.path.join(DATA_ROOT, "datasets")
OUT_DIR_BASE_PATH = os.path.join(DATA_ROOT, "datasets")

# reuse preprocessing defaults from configs
sys.path.append('./scripts')
from configs.adata_crc_config import ADATA_ARGS as ADATA_CRC_ARGS
from configs.adata_merfish_config import ADATA_ARGS as ADATA_MERFISH_ARGS
from configs.cellina_graph_config import N_NEIGHBORS_GRAPH
from train_loo import preprocess_crc, preprocess_merfish, split_indices, preprocess_spatial_features
from train_loo import COUNTS_PER_K, DEFAULT_LABELS_KEY, DEFAULT_DOMAINS_KEY, DEFAULT_BATCH_KEY, DEFAULT_HVGS, DEFAULT_CTRL_DOMAINS, DEFAULT_HOLDOUT_DOMAINS, DEFAULT_N_NEIGHBORS
from utils import set_seed
from counterfactual_analysis import get_baseline_delta, compute_rmse, compute_edistance, mixing_index, get_lfc, precision, direction_match, compute_mse_lfc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, choices=["crc", "merfish"])
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", required=True)
    p.add_argument("--model_class", required=True, choices=["cellina", "cpa", "cellina_graph", "baseline", "scgen"]) 
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


def get_counterfactual_counts(adata, model_class, labels_key, domains_key, holdout_ct, model_name, base_dir, use_cf, control_domains, holdout_domain, recon=None, latents=None):
    cf_matrix, cf_latents = None, None

    # If baseline mode, compute baseline counterfactual and skip loading model reconstructions
    if model_class == 'baseline':
        print('Loading baseline counterfactuals using training-set shift for holdout domain', holdout_domain)
        adata_train = adata[adata.obs['is_holdout'] == False]
        is_holdout_domain = adata_train.obs[domains_key] == holdout_domain
        adata_rest = adata_train[~is_holdout_domain]
        adata_target = adata_train[is_holdout_domain]
        delta = get_baseline_delta(adata_rest, adata_target)

        # apply delta to control cells (holdout celltype & donor region)
        is_holdout_ct = adata_train.obs[labels_key].astype(str) == holdout_ct
        is_in_control_domains = adata_train.obs[domains_key].isin(control_domains)
        adata_control = adata_train[is_holdout_ct & is_in_control_domains]
        counts = _to_dense(adata_control.layers['counts'])

        # For log2, but we compute delta on log1 so just take exp
        cf_matrix = (counts + 1) * np.exp(delta) - 1
        cf_matrix = np.clip(cf_matrix, a_min=0, a_max=None)
        cf_matrix = cf_matrix / (cf_matrix.sum(axis=1, keepdims=True) + 1e-8) * COUNTS_PER_K
    # If not baseline - works for cellina-like, cpa, scgen
    else:
        print('Loading model-predicted counterfactuals for holdout domain', holdout_domain)
        cf_fname = f"{model_name}_counterfactual_x_{holdout_domain}.h5ad"
        cf_path = os.path.join(base_dir, cf_fname)
        if use_cf:
            cf_matrix, cf_latents = load_model_predicted(cf_path)
        else:
            is_holdout_domain = adata.obs[domains_key] == holdout_domain
            is_holdout_ct = adata.obs[labels_key].astype(str) == holdout_ct
            mask_target = is_holdout_domain & is_holdout_ct
            cf_matrix = recon[mask_target.values, :]
            cf_latents = latents[mask_target.values, :]
        
    return cf_matrix, cf_latents

def main():
    args = parse_args()

    adata_path = args.adata_path
    holdout_ct = args.holdout_celltype
    model_class = args.model_class.lower()
    model_name = args.model_name
    use_recon = args.use_recon
    use_cf = args.use_cf
    dataset_name = args.dataset_name.lower()
    out_dir = f'{OUT_DIR_BASE_PATH}/{dataset_name}/correlations'

    set_seed(DEFAULT_SEED)
    # load adata and preprocess (same as train_loo)
    print('Loading adata', adata_path)
    adata = sc.read(adata_path)
    
    # preprocess using DATA_ARGS
    if dataset_name == 'crc':
        DATA_ARGS = ADATA_CRC_ARGS 
    elif dataset_name == 'merfish':
        DATA_ARGS = ADATA_MERFISH_ARGS
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Supported: crc, merfish")
    
    n_top_genes = DATA_ARGS.get('n_top_genes', DEFAULT_HVGS)
    labels_key = DATA_ARGS.get('labels_key', DEFAULT_LABELS_KEY)
    domains_key = DATA_ARGS.get('domains_key', DEFAULT_DOMAINS_KEY)
    control_domains = DATA_ARGS.get('control_domains', DEFAULT_CTRL_DOMAINS)
    holdout_domains = DATA_ARGS.get('holdout_domains', DEFAULT_HOLDOUT_DOMAINS)
    n_neighbors = N_NEIGHBORS_GRAPH if model_class=='cellina_graph' else DATA_ARGS.get('n_neighbors', DEFAULT_N_NEIGHBORS)

    if dataset_name == 'crc':
        adata = preprocess_crc(adata, n_top_genes=n_top_genes, labels_key=labels_key, domains_key=domains_key)
    elif dataset_name == 'merfish':
        adata = preprocess_merfish(adata, n_top_genes=n_top_genes, labels_key=labels_key, domains_key=domains_key)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Supported: crc, merfish")

    # create splits
    train_idx, val_idx, test_idx = split_indices(adata,
                                                 holdout_ct,
                                                 labels_key=labels_key,
                                                 domains_key=domains_key,
                                                 holdout_domains=holdout_domains,
                                                 seed=DEFAULT_SEED)
    
    print(f"n_obs={adata.n_obs} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    step_size_px = 0.12028 if dataset_name == 'crc' else 0.109
    adata = preprocess_spatial_features(adata, step_size_px=step_size_px, n_neighbors=n_neighbors, test_indices=test_idx)

    # build expected paths for recon and counterfactual
    sid = os.path.splitext(os.path.basename(adata_path))[0]
    if dataset_name == 'crc':
        base_dir = os.path.join(CRC_INFERENCE_BASE_DIR, sid, holdout_ct)
    elif dataset_name == 'merfish':
        base_dir = os.path.join(MERFISH_INFERENCE_BASE_DIR, sid, holdout_ct)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Supported: crc, merfish")

    # Load recons
    recon_fname = f"{model_name}_recon_x.h5ad"
    recon_path = os.path.join(base_dir, recon_fname)    
    recon, latents = None, None
    adata_full = adata.copy()
    if model_class != 'baseline':
        recon, latents = load_model_predicted(recon_path)
        adata.uns['recon_x'] = recon
        adata.uns['latents'] = latents
        print('Loaded reconstructions and latents into adata.uns["recon_x"] from', recon_path)
        # Subset adata - full adata only needed for baseline delta
        adata = adata[adata.obs[labels_key].astype(str) == holdout_ct]

    # Counterfactual evaluation - loop over each target domain
    # Control set remains same across evaluation
    is_holdout_ct = adata.obs[labels_key].astype(str) == holdout_ct
    is_in_control_domains = adata.obs[domains_key].isin(control_domains)
    mask_control = is_holdout_ct & is_in_control_domains
    control = adata.layers['counts'][mask_control.values, :]
    control = _to_dense(control)
    if use_recon:
        control = adata.uns['recon_x'][mask_control.values, :]
    
    for hd in holdout_domains:
        adata.uns['counterfactual_x'], adata.uns['counterfactual_latents'] = get_counterfactual_counts(adata, 
                                                                                                       model_class, 
                                                                                                       labels_key, 
                                                                                                       domains_key, 
                                                                                                       holdout_ct, 
                                                                                                       model_name, 
                                                                                                       base_dir, 
                                                                                                       use_cf, 
                                                                                                       control_domains, 
                                                                                                       hd)

        is_in_holdout_domain = adata.obs[domains_key]==hd
        mask_target = is_holdout_ct & is_in_holdout_domain
        target = adata.layers['counts'][mask_target.values, :]
        target = _to_dense(target)
        if use_recon:
            target = adata.uns['recon_x'][mask_target.values, :]
        counterfactual = adata.uns['counterfactual_x']

        # Compute stats - ground-truth log fold change (lfc) and counterfactual lfc vectors on top DE genes
        gt_lfc, cf_lfc, deg = get_lfc(control=control, target=target, counterfactual=counterfactual, n_deg=N_DEG)

        spear, _ = spearmanr(gt_lfc[deg], cf_lfc[deg])
        pear, _ = pearsonr(gt_lfc[deg], cf_lfc[deg])
        prec = precision(gt_lfc, cf_lfc, k=N_DEG, use_abs=True)
        dir_match = direction_match(gt_lfc, cf_lfc, k=N_DEG, normalize="intersection")
        dir_match_k = direction_match(gt_lfc, cf_lfc, k=N_DEG, normalize="k")
        dir_match_gt = direction_match(gt_lfc, cf_lfc, k=N_DEG, normalize="gt_topk")
        mix_idx = mixing_index(observed=target, predicted=counterfactual, library_size=COUNTS_PER_K)
        edist_global = compute_edistance(adata_full, observed=target, predicted=counterfactual, deg=None, library_size=COUNTS_PER_K)
        edist_local = compute_edistance(adata_full, observed=target, predicted=counterfactual, deg=None, library_size=COUNTS_PER_K, local=True)
        edist_pca_log = compute_edistance(adata_full, observed=target, predicted=counterfactual, deg=None, library_size=COUNTS_PER_K, local=True, use_pca=True)
        edist_pca = compute_edistance(adata_full, observed=target, predicted=counterfactual, deg=None, library_size=COUNTS_PER_K, local=True, use_pca=True, log1p=False)
        rmse = compute_rmse(observed=target, predicted=counterfactual, deg=deg, library_size=COUNTS_PER_K)
        mse_lfc = compute_mse_lfc(gt_vec=gt_lfc, cf_vec=cf_lfc, deg=deg)

        print("Eval stats computed.")

        # Save results json
        os.makedirs(out_dir, exist_ok=True)
        model_name_save = model_name
        model_name_save += "-cf" if use_cf else ""
        model_name_save += "-recon" if use_recon else ""
        out_fname = f"{sid}_{model_name_save}_{holdout_ct}_{hd}"
        out_path = os.path.join(out_dir, f"{out_fname}.json")
        print('Saving evaluation results to', out_path)

        with open(out_path, 'w') as fh:
            stats = {'n_deg': N_DEG,
                    'spearman': spear,
                    'pearson': pear,
                    'precision': prec,
                    'direction_match': dir_match,
                    'direction_match_k': dir_match_k,
                    'direction_match_gt': dir_match_gt,
                    'mixing_index': mix_idx,
                    'edistance_global': edist_global,
                    'edistance_local': edist_local,
                    'edistance_pca_log': edist_pca_log,
                    'edistance_pca': edist_pca,
                    'rmse': rmse,
                    'mse_lfc': mse_lfc,
                    }
            stats = {
                k: float(v) if isinstance(v, np.floating) else v
                for k, v in stats.items()
            }
            json.dump(stats, fh)

        print('Saved correlations')


if __name__ == '__main__':
    main()