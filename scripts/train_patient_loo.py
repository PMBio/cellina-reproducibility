"""
Train a model holding out an entire patient/slide (leave-one-patient-out) and
evaluate counterfactual generation quality across cell types.

This mirrors notebooks/loo_benchmarks/loo_patient.ipynb. Unlike train_loo.py
(which holds out a single cell type within one slide's adata), this script
holds out a patient AND a cell type jointly:
 - loads a single, already-preprocessed adata spanning *multiple* patients/slides
 - test set = (--holdout_celltype cells in --holdout_domains, as in train_loo.py's
   split_indices) UNION (all cells from --holdout_sid), so the model never sees
   holdout_celltype in the holdout domain for ANY patient, and never sees
   holdout_sid at all, during training
 - trains (or loads) a model on that split
 - evaluates counterfactual generation for --holdout_celltype specifically (the
   cell type that model was trained/split for) and computes metrics
   (spearman/pearson/precision/direction-match/mixing-index/e-distance/rmse)
   for the held-out patient, as in the notebook

Usage (example):

python scripts/train_patient_loo.py \
  --dataset_name crc \
  --adata_path /data2/a330d/datasets/crc/processed/crc_cosmx_wt.h5ad \
  --holdout_sid 210 \
  --holdout_celltype Fibroblast \
  --model_class cellina \
  --model_name cellina \
  --exclude_sids 110

Outputs:
 - trained model saved under <DATA_ROOT>/data/ood/trained/loo_patients/{holdout_sid}/{holdout_celltype}/{model_name}/
 - per-job metrics CSV saved to results/loo_patient/{dataset_name}_DEG_{n_deg}_sid{holdout_sid}_ct{holdout_celltype}_{model_name}.csv
   (one CSV per job, since concurrent processes can't safely append to a shared
   file; merge afterward with e.g. pandas.concat over glob("results/loo_patient/*.csv"))
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import h5py

DATA_ROOT = '/data2/a330d'  # os.environ.get("DATA_ROOT", ".")

from pprint import pprint
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_ROOT = os.path.join(DATA_ROOT, "data/ood/trained/loo_patients")
RESULTS_ROOT = REPO_ROOT / "results" / "loo_patient"

DEFAULT_N_DEG = 50
DEFAULT_BATCH_SIZE = 2048

# local utils
from utils import set_seed
from counterfactual_analysis import (
    compute_rmse, compute_edistance, mixing_index, get_lfc, precision,
    direction_match, compute_mse_lfc, _to_dense, get_baseline_delta,
    nb_deviance_pop_mean
)
from train_loo import (
    preprocess_spatial_features,
    COUNTS_PER_K,
    DEFAULT_SEED,
    DEFAULT_LABELS_KEY,
    DEFAULT_DOMAINS_KEY,
    DEFAULT_BATCH_KEY,
    DEFAULT_CTRL_DOMAINS,
    DEFAULT_HOLDOUT_DOMAINS,
    DEFAULT_N_NEIGHBORS,
)

# Import configs
sys.path.append('./scripts')
from configs.cellina_config import MODEL_ARGS as CELLINA_MODEL_ARGS, TRAIN_ARGS as CELLINA_TRAIN_ARGS, PLAN_KWARGS as CELLINA_PLAN_KWARGS
from configs.cpa_config import MODEL_ARGS as CPA_MODEL_ARGS, TRAIN_ARGS as CPA_TRAIN_ARGS, PLAN_KWARGS as CPA_PLAN_KWARGS
from configs.adata_crc_config import ADATA_ARGS as ADATA_CRC_ARGS
from configs.adata_merfish_config import ADATA_ARGS as ADATA_MERFISH_ARGS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, choices=['crc', 'merfish'], help="Name of dataset (used for configs)")
    p.add_argument("--adata_path", required=True, help="Path to a single, already-preprocessed adata spanning multiple patients/slides")
    p.add_argument("--holdout_sid", required=True, help="batch_key value (patient/slide id) to hold out entirely for train/val/test")
    p.add_argument("--holdout_celltype", required=True, help="cell type to additionally hold out (in --holdout_domains, across ALL patients) for train/val/test, as in train_loo.py's split_indices")
    p.add_argument("--model_class", required=True, choices=['cellina', 'cpa', 'baseline'], help="one of: cellina, cpa, baseline (average domain shift, no training)")
    p.add_argument("--model_name", required=True, help="folder name for saving model and results")
    p.add_argument("--inference_only", action='store_true', help="Skip training and only load a previously trained model for evaluation")
    p.add_argument("--exclude_sids", default="", help="Comma-separated batch_key values to drop from adata before splitting (e.g. QC-failed slides)")
    p.add_argument("--n_deg", type=int, default=DEFAULT_N_DEG)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    return p.parse_args()


def _read_elem(group_or_dataset):
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
    return read_elem(group_or_dataset)


def load_adata_lean(adata_path, need_spatial):
    """Load only what this script actually uses from adata_path, instead of sc.read()'s
    full load. crc_cosmx_wt.h5ad carries ~20GB of precomputed embeddings in .obsm
    (Cellina_*, PCA.*, niche_*, scVI, scVIVA, ...) that nothing here reads, plus an
    obsm['spatial_x'] that alone decompresses to ~40GB (it's ~73% dense but stored as
    sparse CSR with int64 indices). A full sc.read() pulls all of that into memory
    before split_indices ever runs, which is enough to OOM well before training starts.
    Only X, layers['counts'], obs, var, uns, and (for the cellina model, which needs
    them for setup_anndata/preprocess_spatial_features) obsm['spatial']/['spatial_x']
    are read here.
    """
    with h5py.File(adata_path, 'r') as f:
        adata = ad.AnnData(X=_read_elem(f['X']), obs=_read_elem(f['obs']), var=_read_elem(f['var']))
        if 'uns' in f:
            adata.uns = _read_elem(f['uns'])
        if 'layers' in f and 'counts' in f['layers']:
            adata.layers['counts'] = _read_elem(f['layers']['counts'])
        if need_spatial and 'obsm' in f:
            for key in ('spatial', 'spatial_x'):
                if key in f['obsm']:
                    adata.obsm[key] = _read_elem(f['obsm'][key])
    return adata


def split_indices(adata, holdout_sid, holdout_celltype, batch_key=DEFAULT_BATCH_KEY,
                   labels_key=DEFAULT_LABELS_KEY, domains_key=DEFAULT_DOMAINS_KEY,
                   holdout_domains=DEFAULT_HOLDOUT_DOMAINS, seed=DEFAULT_SEED):
    """Create train/val/test splits holding out both an entire patient/slide and
    a cell type within the holdout domain(s) (across all patients).

    Test: (holdout_celltype cells in any domain whose label contains a string
          from holdout_domains — substring match, as in train_loo.py's
          split_indices) UNION (all cells whose batch_key equals holdout_sid).
    Val: 10% of remaining trainval (random).
    """
    if holdout_sid not in adata.obs[batch_key].unique():
        raise ValueError(f"holdout_sid '{holdout_sid}' not found in adata.obs['{batch_key}'] values")
    if holdout_celltype not in adata.obs[labels_key].unique():
        raise ValueError(f"holdout_celltype '{holdout_celltype}' not found in adata.obs['{labels_key}'] values")

    domain_str = adata.obs[domains_key].astype(str)
    is_holdout_domain = domain_str.apply(lambda d: any(hd in d for hd in holdout_domains))
    is_holdout_ct = adata.obs[labels_key].astype(str) == holdout_celltype
    is_holdout_slide = adata.obs[batch_key] == holdout_sid

    test_mask = (is_holdout_domain & is_holdout_ct) | is_holdout_slide

    all_idx = np.arange(adata.n_obs)
    test_idx = np.where(test_mask.values)[0]
    trainval_idx = np.setdiff1d(all_idx, test_idx)

    rng = np.random.default_rng(seed)
    n_trainval = trainval_idx.shape[0]
    n_val = max(1, int(0.1 * n_trainval))
    val_idx_rel = rng.choice(np.arange(n_trainval), size=n_val, replace=False)
    val_idx = trainval_idx[val_idx_rel]
    train_idx = np.setdiff1d(trainval_idx, val_idx)

    adata.obs['is_holdout'] = False
    if len(test_idx) > 0:
        adata.obs.iloc[test_idx, adata.obs.columns.get_loc('is_holdout')] = True

    return train_idx, val_idx, test_idx


def train_model(adata, model_class, model_args, train_args, save_dir, plan_kwargs=None,
                 batch_key=DEFAULT_BATCH_KEY, labels_key=DEFAULT_LABELS_KEY, domains_key=DEFAULT_DOMAINS_KEY, splits=None):
    """Train model and save to save_dir. Returns trained model instance."""
    mc = model_class.lower()
    model = None

    if mc == 'cellina':
        import cellina
        from cellina import Cellina as CellinaModel
        print("cellina version: ", cellina.__version__)
        CellinaModel.setup_anndata(adata,
                                   batch_key=batch_key,
                                   labels_key=labels_key,
                                   domains_key=domains_key,
                                   spatial_obsm_key='spatial_x',
                                   layer='counts')
        model = CellinaModel(adata, **model_args)

        train_args['datasplitter_kwargs'] = {
                  "external_indexing": [splits[0], splits[1], splits[2]],
                  }
        if plan_kwargs is not None:
            model.train(**train_args, plan_kwargs=plan_kwargs)
        else:
            model.train(**train_args)

    elif mc == 'cpa':
        import cpa
        adata.obs['dose'] = 1.0  # NOTE: dummy dose for compatibility with CPA model
        adata.obs['data_split'] = 'train'
        adata.obs.iloc[splits[1], adata.obs.columns.get_loc('data_split')] = 'valid'
        adata.obs.iloc[splits[2], adata.obs.columns.get_loc('data_split')] = 'test'
        cpa.CPA.setup_anndata(adata,
                  perturbation_key=domains_key,
                  control_group='REF',
                  dosage_key='dose',
                  batch_key=batch_key,
                  categorical_covariate_keys=[labels_key],
                  is_count_data=True,
                  max_comb_len=1,
                 )
        model = cpa.CPA(adata,
                        split_key='data_split',
                        train_split='train',
                        valid_split='valid',
                        test_split='test',
                        **model_args)
        model.train(**train_args, plan_kwargs=plan_kwargs, save_path=save_dir)

    else:
        raise ValueError(f"Unsupported model_class: {model_class}. Supported: cellina, cpa")

    saved_model_path = save_dir
    print('model save path:', saved_model_path)
    try:
        if hasattr(model, 'save'):
            model.save(saved_model_path, overwrite=True)
        elif hasattr(model, 'save_model'):
            model.save_model(saved_model_path, overwrite=True)
    except Exception as e:
        print("Warning: saving model raised:", e)

    return model


def load_model(save_dir, model_class, adata, splits=None):
    mc = model_class.lower()
    if mc == 'cellina':
        from cellina import Cellina as CellinaModel
        model = CellinaModel.load(save_dir, adata)
    elif mc == 'cpa':
        import cpa
        adata.obs['dose'] = 1.0  # NOTE: dummy dose for compatibility with CPA model
        adata.obs['data_split'] = 'train'
        adata.obs.iloc[splits[1], adata.obs.columns.get_loc('data_split')] = 'valid'
        adata.obs.iloc[splits[2], adata.obs.columns.get_loc('data_split')] = 'test'
        model = cpa.CPA.load(dir_path=save_dir, adata=adata, use_gpu=True)
    else:
        raise ValueError(f"Unsupported model_class: {model_class}. Supported: cellina, cpa")

    print(f"{model_class} loaded model from {save_dir}")
    return model


def generate_counterfactual(model, adata_holdout, model_class, idx_control, neighbor_indices, hd, batch_size,
                             adata_train=None, domains_key=None):
    """Generate raw-count-scale counterfactual expression for idx_control cells shifted towards domain hd."""
    mc = model_class.lower()

    if mc == 'baseline':
        # Average domain shift baseline (mirrors eval_loo.py's model_class == 'baseline' branch):
        # delta = mean log2 fold change between domain hd and all other domains, computed on the
        # training population (all patients/celltypes except the held-out patient/celltype-in-domain).
        # That delta is then applied to the held-out patient's own control cells (idx_control).
        is_holdout_domain = adata_train.obs[domains_key].astype(str) == hd
        adata_rest = adata_train[~is_holdout_domain.values]
        adata_target = adata_train[is_holdout_domain.values]
        delta = get_baseline_delta(adata_rest, adata_target)

        control_counts = _to_dense(adata_holdout.layers['counts'][idx_control, :])
        cf_counts = (control_counts + 1) * (2 ** delta) - 1
        cf_counts = np.clip(cf_counts, a_min=0, a_max=None)
        cf_counts = cf_counts / (cf_counts.sum(axis=1, keepdims=True) + 1e-8) * COUNTS_PER_K
        return cf_counts

    if mc == 'cellina':
        args_gex = {
            "adata": adata_holdout,
            "indices": idx_control,
            "batch_size": batch_size,
            "seed": 0,
            "neighbour_indices": neighbor_indices,
            "precomputed": False,
        }
        cf_counts = model.get_counterfactual_expression(**args_gex)
        return _to_dense(cf_counts)

    if mc == 'cpa':
        from cpa._utils import CPA_REGISTRY_KEYS
        adata_ctrl = adata_holdout[idx_control].copy()
        perturbation_idx = model.pert_encoder[hd]
        adata_ctrl.obsm['perts'][:, 0] = perturbation_idx
        adata_ctrl.obs[CPA_REGISTRY_KEYS.CONTROL_KEY] = 0
        out = model.predict(adata_ctrl, batch_size=batch_size)
        if "CPA_pred" in adata_ctrl.obsm:
            return _to_dense(adata_ctrl.obsm["CPA_pred"])
        return _to_dense(out)

    raise ValueError(f"Unsupported model_class: {model_class}. Supported: cellina, cpa, baseline")


def evaluate_patient(adata, model, model_class, model_name, holdout_sid, dataset_name,
                      holdout_celltype, labels_key, domains_key, batch_key,
                      control_domain, holdout_domains, batch_size, n_deg, step_size_px, n_neighbors):
    """Evaluate counterfactual generation for holdout_celltype/holdout_domains within the held-out patient."""
    adata_holdout = adata[adata.obs[batch_key] == holdout_sid].copy()
    results = []

    mc = model_class.lower()
    adata_train = adata[adata.obs['is_holdout'] == False] if mc == 'baseline' else None

    print(f"{'='*50} Holdout celltype: {holdout_celltype} {'='*50}")
    is_control_region = adata_holdout.obs[domains_key] == control_domain
    is_holdout_ct = adata_holdout.obs[labels_key].astype(str) == holdout_celltype
    mask_control = is_control_region & is_holdout_ct
    idx_control = np.where(mask_control.values)[0]

    for hd in holdout_domains:
        is_holdout_region = adata_holdout.obs[domains_key].astype(str) == hd
        mask_target = is_holdout_ct & is_holdout_region
        idx_target = np.where(mask_target.values)[0]

        if len(idx_target) == 0 or len(idx_control) == 0:
            print(f"Skipping {holdout_celltype}/{hd}: no target or control cells found for slide {holdout_sid}")
            continue
        if mc == 'cellina':
            # Recompute spatial connectivities masking this iteration's test indices, to avoid data leakage
            preprocess_spatial_features(adata_holdout, step_size_px=step_size_px, n_neighbors=n_neighbors, test_indices=idx_target)
            conn = adata_holdout.obsp["spatial_connectivities_orig"]
            sub_conn = conn[idx_target]
            neighbor_indices = np.unique(sub_conn.nonzero()[1])
            neighbor_indices = neighbor_indices[~is_holdout_ct.values[neighbor_indices]]
        else:
            neighbor_indices = None

        counterfactual = generate_counterfactual(model, adata_holdout, model_class, idx_control, neighbor_indices, hd, batch_size,
                                                  adata_train=adata_train, domains_key=domains_key)

        control = _to_dense(adata_holdout.layers['counts'][mask_control.values, :])
        target = _to_dense(adata_holdout.layers['counts'][mask_target.values, :])

        gt_lfc, cf_lfc, deg = get_lfc(control=control, target=target, counterfactual=counterfactual, n_deg=n_deg)

        spear, _ = spearmanr(gt_lfc[deg], cf_lfc[deg])
        pear, _ = pearsonr(gt_lfc[deg], cf_lfc[deg])
        prec = precision(gt_lfc, cf_lfc, k=n_deg, use_abs=True)
        dir_match = direction_match(gt_lfc, cf_lfc, k=n_deg, normalize="intersection")
        dir_match_k = direction_match(gt_lfc, cf_lfc, k=n_deg, normalize="k")
        dir_match_gt = direction_match(gt_lfc, cf_lfc, k=n_deg, normalize="gt_topk")
        mix_idx = mixing_index(observed=target, predicted=counterfactual, library_size=COUNTS_PER_K)
        edist_global = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=COUNTS_PER_K)
        edist_local = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=COUNTS_PER_K, local=True)
        edist_pca_log = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=COUNTS_PER_K, local=True, use_pca=True)
        edist_pca = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=COUNTS_PER_K, local=True, use_pca=True, log1p=False)
        rmse = compute_rmse(observed=target, predicted=counterfactual, deg=deg, library_size=COUNTS_PER_K)
        mse_lfc = compute_mse_lfc(gt_vec=gt_lfc, cf_vec=cf_lfc, deg=deg)
        nb_deviance = nb_deviance_pop_mean(obs_X=target, pred_X=counterfactual)

        results.append(
            dict(
                dataset_name=dataset_name,
                sid=holdout_sid,
                control_domain=control_domain,
                target_domain=hd,
                n_deg=n_deg,
                model_name=model_name,
                holdout_celltype=holdout_celltype,
                spearman=spear,
                pearson=pear,
                precision=prec,
                direction_match=dir_match,
                direction_match_k=dir_match_k,
                direction_match_gt=dir_match_gt,
                mixing_index=mix_idx,
                edistance_global=edist_global,
                edistance_local=edist_local,
                edistance_pca_log=edist_pca_log,
                edistance_pca=edist_pca,
                rmse=rmse,
                mse_lfc=mse_lfc,
                nb_deviance=nb_deviance,
            )
        )

    import gc
    gc.collect()

    return pd.DataFrame(results)


def main():
    args = parse_args()

    dataset_name = args.dataset_name.lower()
    mc = args.model_class.lower()
    model_name = args.model_name
    holdout_sid = str(args.holdout_sid)
    holdout_celltype = args.holdout_celltype

    if mc == 'cellina':
        model_args = CELLINA_MODEL_ARGS.copy()
        train_args = CELLINA_TRAIN_ARGS.copy()
        plan_kwargs = CELLINA_PLAN_KWARGS.copy()
    elif mc == 'cpa':
        model_args = CPA_MODEL_ARGS.copy()
        train_args = CPA_TRAIN_ARGS.copy()
        plan_kwargs = CPA_PLAN_KWARGS.copy()
    elif mc == 'baseline':
        model_args = train_args = plan_kwargs = None
    else:
        raise ValueError(f"Unsupported model_class: {args.model_class}")

    set_seed(DEFAULT_SEED)

    print("Loading adata:", args.adata_path)
    adata = load_adata_lean(args.adata_path, need_spatial=(mc == 'cellina'))

    if dataset_name == 'crc':
        DATA_ARGS = ADATA_CRC_ARGS
        step_size_px = 0.12028
    elif dataset_name == 'merfish':
        DATA_ARGS = ADATA_MERFISH_ARGS
        step_size_px = 0.109
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Supported: crc, merfish")

    labels_key = DATA_ARGS.get('labels_key', DEFAULT_LABELS_KEY)
    domains_key = DATA_ARGS.get('domains_key', DEFAULT_DOMAINS_KEY)
    batch_key = DATA_ARGS.get('batch_key', DEFAULT_BATCH_KEY)
    control_domain = DATA_ARGS.get('control_domains', DEFAULT_CTRL_DOMAINS)[0]
    holdout_domains = DATA_ARGS.get('holdout_domains', DEFAULT_HOLDOUT_DOMAINS)
    n_neighbors = DATA_ARGS.get('n_neighbors', DEFAULT_N_NEIGHBORS)

    # CPA requires batch_key as string; cast unconditionally for consistent comparisons
    adata.obs[batch_key] = adata.obs[batch_key].astype(str)

    if args.exclude_sids:
        exclude = {s.strip() for s in args.exclude_sids.split(',') if s.strip()}
        adata = adata[~adata.obs[batch_key].isin(exclude)].copy()

    train_idx, val_idx, test_idx = split_indices(adata, holdout_sid, holdout_celltype,
                                                  batch_key=batch_key, labels_key=labels_key,
                                                  domains_key=domains_key, holdout_domains=holdout_domains,
                                                  seed=DEFAULT_SEED)
    splits = (train_idx, val_idx, test_idx)
    print(f"n_obs={adata.n_obs} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    save_dir = os.path.join(MODEL_ROOT, holdout_sid, holdout_celltype, model_name)
    os.makedirs(save_dir, exist_ok=True)

    if mc == 'baseline':
        print("model_class=baseline: no training/loading needed; applying average domain shift at eval time.")
        model = None
    elif args.inference_only:
        model = load_model(save_dir, args.model_class, adata, splits=splits)
    else:
        model = train_model(adata, args.model_class, model_args, train_args, save_dir,
                             plan_kwargs=plan_kwargs, batch_key=batch_key, labels_key=labels_key,
                             domains_key=domains_key, splits=splits)

    df_results = evaluate_patient(adata, model, args.model_class, model_name, holdout_sid, dataset_name,
                                   holdout_celltype, labels_key, domains_key, batch_key,
                                   control_domain, holdout_domains, args.batch_size, args.n_deg,
                                   step_size_px, n_neighbors)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    ct_slug = holdout_celltype.replace(' ', '_')
    out_path = RESULTS_ROOT / f"{dataset_name}_DEG_{args.n_deg}_sid{holdout_sid}_ct{ct_slug}_{model_name}.csv"
    df_results.to_csv(out_path, index=False)

    print("Done. Outputs:")
    pprint({
        'save_dir': save_dir,
        'model_name': model_name,
        'results_csv': str(out_path),
    })


if __name__ == '__main__':
    main()
