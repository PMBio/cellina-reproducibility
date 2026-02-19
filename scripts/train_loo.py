"""
Train a model holding out a cell type (leave-one-out) and save reconstructions and optional counterfactuals.

Usage (examples):

python scripts/train_loo.py \
  --adata_path /data2/a330d/datasets/cosmx/crc_wt_cosmx/crc_202.h5ad \
  --holdout_celltype Epithelial \
  --model_class cellina \
  --model_name cond_z_False_sim_seed_0_ood

This script mirrors preprocessing and inference steps used in the notebooks.

Outputs:
 - trained model saved under /data2/a330d/data/ood/trained/{model_name}_{holdout_celltype}/
 - reconstructions saved to <adata_parent_dir>/{model_name}_{holdout_celltype}_recon_x.h5ad
 - (optional) counterfactual reconstructions saved to <adata_parent_dir>/{model_name}_{holdout_celltype}_counterfactual_x.h5ad
"""

import os
import argparse
import json
import numpy as np
import scanpy as sc
import anndata as ad
import sys

from pprint import pprint
from scipy.sparse import csr_matrix

# defaults based on notebooks (counterfactuals.ipynb)
DEFAULT_HVGS = 2000
DEFAULT_N_NEIGHBORS = 50
DEFAULT_BATCH_SIZE = 4096
DEFAULT_SEED = 0
DEFAULT_LABELS_KEY = 'coarse_type'
DEFAULT_DOMAINS_KEY = 'typ'
DEFAULT_BATCH_KEY = 'sid'
MODEL_ROOT = "/data2/a330d/data/ood/trained"

# local utils
from counterfactual_analysis import make_counterfactual_adata
from utils import set_seed

# Import configs
sys.path.append('./scripts')
from configs.cellina_config import MODEL_ARGS as CELLINA_MODEL_ARGS, TRAIN_ARGS as CELLINA_TRAIN_ARGS, PLAN_KWARGS as CELLINA_PLAN_KWARGS, DO_COUNTERFACTUAL as CELLINA_DO_COUNTERFACTUAL
from configs.cpa_config import MODEL_ARGS as CPA_MODEL_ARGS, TRAIN_ARGS as CPA_TRAIN_ARGS, PLAN_KWARGS as CPA_PLAN_KWARGS, DO_COUNTERFACTUAL as CPA_DO_COUNTERFACTUAL
from configs.cellina_graph_config import MODEL_ARGS as CELLINA_GRAPH_MODEL_ARGS, TRAIN_ARGS as CELLINA_GRAPH_TRAIN_ARGS, PLAN_KWARGS as CELLINA_GRAPH_PLAN_KWARGS, DO_COUNTERFACTUAL as CELLINA_GRAPH_DO_COUNTERFACTUAL
from configs.adata_config import ADATA_ARGS, NORMALIZE, LOG1P


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", required=True)
    p.add_argument("--model_class", required=True, choices=['cellina', 'cpa', 'cellina_graph'], help="one of: cellina, cpa")
    p.add_argument("--model_name", default=None, help="folder name for saving model and outputs")

    return p.parse_args()


def _to_array(x):
    if x is None:
        return None
    toarray = getattr(x, "toarray", None)
    if callable(toarray):
        return toarray()
    return np.asarray(x)


def _reconstruct_model_output(model, adata_obj, model_class, return_normalized=False, batch_size=4096):
    """Model-agnostic adapter to obtain reconstructions for adata_obj as numpy array.
    - For scvi/cellina: use get_normalized_expression if available
    - For CPA-like models: call predict() which may write into adata_obj.obsm['CPA_pred'] or return array
    """
    model_class = model_class.lower()
    if model_class in ("cpa",):
        out = None
        # many CPA implementations write predictions into adata.obsm['CPA_pred']
        out = model.predict(adata_obj, batch_size=batch_size)
        if "CPA_pred" in adata_obj.obsm:
            X = _to_array(adata_obj.obsm["CPA_pred"])  # likely raw counts
            if return_normalized:
                X = np.log1p(X)
                X = X / (X.sum(axis=1, keepdims=True) + 1e-8)
            return X
        if out is not None:
            return _to_array(out)
        raise RuntimeError("CPA model produced no output and did not populate adata.obsm['CPA_pred']")

    # other models (cellina or generic models exposing get_normalized_expression)
    if hasattr(model, "get_normalized_expression"):
        #library_size = 1.0 if return_normalized else "latent"
        library_size = 1.0
        out = model.get_normalized_expression(adata_obj, library_size=library_size, batch_size=batch_size)
        return _to_array(out)

    # fallback: try model.predict
    if hasattr(model, "predict"):
        out = model.predict(adata_obj)
        return _to_array(out)

    raise RuntimeError("Model does not expose a known reconstruction API (get_normalized_expression or predict)")


def save_recon_adata(adata_parent, recon_array, out_path):
    # build adata with only obs and var copied, and recon in obsm
    #ad_recon = ad.AnnData(X=np.zeros((adata_parent.n_obs, adata_parent.n_vars), dtype=np.float32))
    ad_recon = ad.AnnData(X=recon_array)
    ad_recon.obs = adata_parent.obs.copy()
    ad_recon.var = adata_parent.var.copy()
    ad_recon.write_h5ad(out_path, compression="gzip")
    return out_path


def split_indices(adata, holdout_celltype, labels_key='coarse_type', domains_key='typ', seed=0):
    """Create train/val/test splits consistent with notebooks.

    Test: holdout_celltype & typ contains 'CRC'
    Val: 10% of remaining trainval (random)
    """
    if holdout_celltype not in adata.obs[labels_key].unique():
        raise ValueError(f"holdout_celltype '{holdout_celltype}' not found in adata.obs['{labels_key}'] values")

    is_tumor_region = adata.obs[domains_key].astype(str).str.contains('CRC', regex=True)
    is_holdout_ct = adata.obs[labels_key].astype(str) == holdout_celltype
    test_mask = is_tumor_region & is_holdout_ct

    all_idx = np.arange(adata.n_obs)
    test_idx = np.where(test_mask.values)[0]
    trainval_idx = np.setdiff1d(all_idx, test_idx)

    rng = np.random.default_rng(seed)
    n_trainval = trainval_idx.shape[0]
    n_val = max(1, int(0.1 * n_trainval))
    val_idx_rel = rng.choice(np.arange(n_trainval), size=n_val, replace=False)
    val_idx = trainval_idx[val_idx_rel]
    train_idx = np.setdiff1d(trainval_idx, val_idx)

    # annotate is_holdout in adata.obs
    adata.obs['is_holdout'] = False
    if len(test_idx) > 0:
        adata.obs.iloc[test_idx, adata.obs.columns.get_loc('is_holdout')] = True

    return train_idx, val_idx, test_idx


def preprocess_adata(adata, n_top_genes=2000, n_neighbors=50, labels_key=DEFAULT_LABELS_KEY, domains_key=DEFAULT_DOMAINS_KEY):
    """Apply preprocessing steps from counterfactuals notebook.
    Modifies and returns adata.
    """
    adata.obs_names_make_unique()

    # ensure coarse_type exists (try mapping from 'ist' if not)
    if 'coarse_type' not in adata.obs.columns or adata.obs['coarse_type'].isna().any():
        try:
            from _labels_to_coarse import LABEL_TO_COARSE as LMAP
            adata.obs['coarse_type'] = adata.obs['ist'].map(LMAP)
            adata.obs['coarse_type'] = adata.obs['coarse_type'].astype('category')
        except Exception:
            pass

    adata.obs["typ_clean"] = (
        adata.obs["typ"]
        .str.extract(r"(REF|TVA|CRC)", expand=False)
    )

    adata = adata[~adata.obs[domains_key].isna()]
    adata = adata[~adata.obs[labels_key].isna()]
    
    sc.pp.filter_cells(adata, min_counts=3)
    sc.pp.filter_genes(adata, min_counts=3)

    adata.layers['counts'] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, layer='counts', flavor='seurat_v3', n_top_genes=n_top_genes, subset=True)

    if 'CenterX_global_px' in adata.obs.columns and 'CenterY_global_px' in adata.obs.columns:
        adata.obsm['spatial'] = adata.obs[['CenterX_global_px', 'CenterY_global_px']].values

    try:
        from cellina._spatial_utils import spatial_neighbors
        spatial_neighbors(adata, bandwidth=np.inf, max_neighbours=n_neighbors, standardize=False)
    except Exception as e:
        print("Warning: spatial_neighbors failed or cellina not available:", e)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.obsm['spatial_x'] = adata.obsp['spatial_connectivities'] @ adata.X / n_neighbors
    # float32
    adata.obsm['spatial_x'] = csr_matrix(adata.obsm['spatial_x']).astype(np.float32)
    
    adata.X = adata.layers['counts'].copy()

    return adata


def train_model(adata, model_class, model_args, train_args, save_dir, plan_kwargs=None, batch_key=DEFAULT_BATCH_KEY, labels_key=DEFAULT_LABELS_KEY, domains_key=DEFAULT_DOMAINS_KEY, splits=None):
    """Train model and save to save_dir. Returns trained model instance."""
    mc = model_class.lower()
    model = None

    if mc == 'cellina':
        from cellina import CellinaModel
        CellinaModel.setup_anndata(adata, 
                                   batch_key=batch_key, 
                                   labels_key=labels_key, 
                                   domains_key=domains_key, 
                                   spatial_obsm_key='spatial_x', 
                                   layer='counts')
        model = CellinaModel(adata, **model_args)

        # Add split info
        train_args['datasplitter_kwargs'] = {
                  "external_indexing": [splits[0], splits[1], splits[2]],
                  }
        if plan_kwargs is not None:
            model.train(**train_args, plan_kwargs=plan_kwargs)
        else:
            model.train(**train_args)

    elif mc == 'cpa':
        try:
            import cpa
            adata.obs['dose'] = 1.0 # NOTE: dummy dose for compatibility with CPA model
            adata.obs['data_split'] = 'train'
            adata.obs.iloc[splits[1], adata.obs.columns.get_loc('data_split')] = 'valid'
            adata.obs.iloc[splits[2], adata.obs.columns.get_loc('data_split')] = 'test'
            cpa.CPA.setup_anndata(adata,
                      perturbation_key=domains_key,
                      control_group='REF',
                      dosage_key='dose',
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
        except Exception as e:
            raise RuntimeError(f"CPA training failed or not supported generically: {e}")

    elif mc == 'cellina_graph':
        from cellina_graph import CellinaModel
        CellinaModel.setup_anndata(adata, 
                                   batch_key=batch_key, 
                                   labels_key=labels_key, 
                                   domains_key=domains_key, 
                                   layer='counts',
                                   spatial_connectivities_key='spatial_connectivities', 
                                   )
        model = CellinaModel(adata, **model_args)
        print(model_args)

        # Add split info
        train_args['datasplitter_kwargs'] = {
                  "external_indexing": [splits[0], splits[1], splits[2]],
                  }
        if plan_kwargs is not None:
            model.train(**train_args, plan_kwargs=plan_kwargs)
        else:
            model.train(**train_args)

    else:
        raise ValueError(f"Unsupported model_class: {model_class}. Supported: cellina, cpa, cellina_graph")

    # try saving model with common APIs
    saved_model_path = save_dir
    print('model save path:', saved_model_path)
    try:
        if hasattr(model, 'save'):
            model.save(saved_model_path, overwrite=True)
        elif hasattr(model, 'save_model'):
            model.save_model(saved_model_path, overwrite=True)
        elif hasattr(model, 'write'):
            model.write(saved_model_path, overwrite=True)
        else:
            try:
                model.save(saved_model_path, save_anndata=False, overwrite=True)
            except Exception:
                pass
    except Exception as e:
        print("Warning: saving model raised:", e)

    return model


def run_inference(model, adata, adata_path, model_class, model_name, holdout_celltype, do_cf=True, batch_size=DEFAULT_BATCH_SIZE):
    """Run reconstructions for full adata and optional counterfactuals. Returns paths."""
    # determine output directory: go one level up from input path and create a folder
    # named after the input file (without .h5ad). Example: abc/raw/crc_231.h5ad -> abc/crc_231/
    print("Running inference and saving outputs...")
    input_parent = os.path.dirname(adata_path)
    parent_of_input = os.path.dirname(input_parent)
    input_basename = os.path.splitext(os.path.basename(adata_path))[0]
    out_dir = os.path.join(parent_of_input, input_basename, holdout_celltype)
    os.makedirs(out_dir, exist_ok=True)

    # full reconstruction
    try:
        recon_all = _reconstruct_model_output(model, adata, model_class, return_normalized=True, batch_size=batch_size)
    except Exception as e:
        print("Failed to get reconstructions for full dataset:", e)
        recon_all = None

    if recon_all is not None:
        out_recon_path = os.path.join(out_dir, f"{model_name}_recon_x.h5ad")
        save_recon_adata(adata, recon_all, out_recon_path)
        print("Saved reconstruction adata:", out_recon_path)
    else:
        out_recon_path = None

    out_cf_path = None
    if do_cf:
        # find target (OOD) indices and control indices
        is_tumor_region = adata.obs['typ'].astype(str).str.contains('CRC', regex=True)
        mask_target = is_tumor_region & (adata.obs['coarse_type'].astype(str) == holdout_celltype)
        idx_target = np.where(mask_target.values)[0]
        mask_control = (~adata.obs['is_holdout']) & (adata.obs['coarse_type'] == holdout_celltype)
        idx_control = np.where(mask_control.values)[0]

        if len(idx_control) == 0 or len(idx_target) == 0:
            print("No control or no target cells found for counterfactual creation; skipping CF inference.")
            out_cf_path = None
        else:
            adata_cf = make_counterfactual_adata(adata, indices_basal=idx_control, indices_counterfactual=idx_target, spatial_column='spatial_x', sample=False)
            try:
                recon_cf = _reconstruct_model_output(model, adata_cf, model_class, return_normalized=True, batch_size=batch_size)
                out_cf_path = os.path.join(out_dir, f"{model_name}_counterfactual_x.h5ad")
                save_recon_adata(adata_cf, recon_cf, out_cf_path)
                print("Saved counterfactual reconstructions:", out_cf_path)
            except Exception as e:
                print("Counterfactual inference failed:", e)
                out_cf_path = None

    return out_recon_path, out_cf_path


def main():
    args = parse_args()

    # choose configs based on model_class
    mc = args.model_class.lower()
    if mc == 'cellina':
        model_args = CELLINA_MODEL_ARGS.copy()
        train_args = CELLINA_TRAIN_ARGS.copy()
        plan_kwargs = CELLINA_PLAN_KWARGS.copy()
        do_cf_default = CELLINA_DO_COUNTERFACTUAL
    elif mc == 'cpa':
        model_args = CPA_MODEL_ARGS.copy()
        train_args = CPA_TRAIN_ARGS.copy()
        plan_kwargs = CPA_PLAN_KWARGS.copy()
        do_cf_default = CPA_DO_COUNTERFACTUAL
    elif mc == 'cellina_graph':
        model_args = CELLINA_GRAPH_MODEL_ARGS.copy()
        train_args = CELLINA_GRAPH_TRAIN_ARGS.copy()
        plan_kwargs = CELLINA_GRAPH_PLAN_KWARGS.copy()
        do_cf_default = CELLINA_GRAPH_DO_COUNTERFACTUAL
    else:
        raise ValueError(f"Unsupported model_class: {args.model_class}")

    # seed for reproducibility
    #np.random.seed(DEFAULT_SEED)
    set_seed(DEFAULT_SEED)

    # load adata
    print("Loading adata:", args.adata_path)
    adata = sc.read(args.adata_path)
    
    # preprocess using ADATA_ARGS
    n_top_genes = ADATA_ARGS.get('n_top_genes', DEFAULT_HVGS)
    n_neighbors = 10 if mc=='cellina-graph' else ADATA_ARGS.get('n_neighbors', DEFAULT_N_NEIGHBORS)
    adata = preprocess_adata(adata, 
                             n_top_genes=n_top_genes, 
                             n_neighbors=n_neighbors,
                             )

    # create splits
    train_idx, val_idx, test_idx = split_indices(adata, args.holdout_celltype, 
                                                 labels_key=ADATA_ARGS.get('labels_key', DEFAULT_LABELS_KEY), 
                                                 domains_key=ADATA_ARGS.get('domains_key', DEFAULT_DOMAINS_KEY), 
                                                 seed=DEFAULT_SEED)
    splits = (train_idx, val_idx, test_idx)
    print(f"n_obs={adata.n_obs} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # decide whether to run counterfactuals from config default
    do_cf = bool(do_cf_default)

    # prepare save dir for model
    sid = args.adata_path.split('/')[-1].split('.')[0]
    model_name = args.model_name
    save_dir = os.path.join(MODEL_ROOT, sid, args.holdout_celltype, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # train
    model = train_model(adata,
                        args.model_class, 
                        model_args, 
                        train_args, 
                        save_dir, 
                        labels_key=ADATA_ARGS.get('labels_key', DEFAULT_LABELS_KEY),
                        domains_key=ADATA_ARGS.get('domains_key', DEFAULT_DOMAINS_KEY),
                        batch_key=ADATA_ARGS.get('batch_key', DEFAULT_BATCH_KEY),
                        plan_kwargs=plan_kwargs, 
                        splits=splits)
    
    # inference
    batch_size = train_args.get('batch_size', DEFAULT_BATCH_SIZE)
    out_recon_path, out_cf_path = run_inference(model, adata, args.adata_path, args.model_class, model_name, args.holdout_celltype, do_cf=do_cf, batch_size=batch_size)

    print("Done. Outputs:")
    pprint({
        'save_dir': save_dir,
        'model_name': model_name,
        'recon_adata': out_recon_path,
        'counterfactual_adata': out_cf_path,
    })


if __name__ == '__main__':
    main()
