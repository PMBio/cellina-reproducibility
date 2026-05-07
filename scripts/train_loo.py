"""
Train a model holding out a cell type (leave-one-out) and save reconstructions and optional counterfactuals.

Usage (examples):

python scripts/train_loo.py \
  --adata_path <path/to/adata> \
  --holdout_celltype Epithelial \
  --model_class cellina \
  --model_name cellina

This script mirrors preprocessing and inference steps used in the notebooks.

Outputs:
 - trained model saved under <DATA_ROOT>/data/ood/trained/{model_name}_{holdout_celltype}/
 - reconstructions saved to <adata_parent_dir>/{model_name}_{holdout_celltype}_recon_x.h5ad
 - (optional) counterfactual reconstructions saved to <adata_parent_dir>/{model_name}_{holdout_celltype}_counterfactual_x.h5ad
"""

import os
import argparse
import numpy as np
import scanpy as sc
import anndata as ad
import sys

DATA_ROOT = os.environ.get("DATA_ROOT", ".")

from pprint import pprint

# defaults based on notebooks (counterfactuals.ipynb)
DEFAULT_HVGS = 2000
DEFAULT_N_NEIGHBORS = 50
DEFAULT_BATCH_SIZE = 512
DEFAULT_SEED = 0
COUNTS_PER_K = 1e4
DEFAULT_LABELS_KEY = 'coarse_type'
DEFAULT_DOMAINS_KEY = 'typ'
DEFAULT_BATCH_KEY = 'sid'
DEFAULT_CTRL_DOMAINS = ['REF']
DEFAULT_HOLDOUT_DOMAINS = ['CRC']
MODEL_ROOT = os.path.join(DATA_ROOT, "data/ood/trained")

# local utils
from counterfactual_analysis import _normalize_counts
from utils import set_seed

# Import configs
sys.path.append('./scripts')
from configs.cellina_config import MODEL_ARGS as CELLINA_MODEL_ARGS, TRAIN_ARGS as CELLINA_TRAIN_ARGS, PLAN_KWARGS as CELLINA_PLAN_KWARGS, DO_COUNTERFACTUAL as CELLINA_DO_COUNTERFACTUAL
from configs.cpa_config import MODEL_ARGS as CPA_MODEL_ARGS, TRAIN_ARGS as CPA_TRAIN_ARGS, PLAN_KWARGS as CPA_PLAN_KWARGS, DO_COUNTERFACTUAL as CPA_DO_COUNTERFACTUAL
from configs.cellina_graph_config import MODEL_ARGS as CELLINA_GRAPH_MODEL_ARGS, TRAIN_ARGS as CELLINA_GRAPH_TRAIN_ARGS, PLAN_KWARGS as CELLINA_GRAPH_PLAN_KWARGS, DO_COUNTERFACTUAL as CELLINA_GRAPH_DO_COUNTERFACTUAL, N_NEIGHBORS_PER_SEED, N_NEIGHBORS_GRAPH
from configs.adata_crc_config import ADATA_ARGS as ADATA_CRC_ARGS
from configs.adata_merfish_config import ADATA_ARGS as ADATA_MERFISH_ARGS
from configs.cellina_mmd_config import MODEL_ARGS as CELLINA_MMD_MODEL_ARGS, TRAIN_ARGS as CELLINA_MMD_TRAIN_ARGS, PLAN_KWARGS as CELLINA_MMD_PLAN_KWARGS, DO_COUNTERFACTUAL as CELLINA_MMD_DO_COUNTERFACTUAL
from configs.scgen_config import MODEL_ARGS as SCGEN_MODEL_ARGS, TRAIN_ARGS as SCGEN_TRAIN_ARGS, PLAN_KWARGS as SCGEN_PLAN_KWARGS, DO_COUNTERFACTUAL as SCGEN_DO_COUNTERFACTUAL
from configs.cellina_ablated_config import MODEL_ARGS as CELLINA_ABLATED_MODEL_ARGS, TRAIN_ARGS as CELLINA_ABLATED_TRAIN_ARGS, PLAN_KWARGS as CELLINA_ABLATED_PLAN_KWARGS, DO_COUNTERFACTUAL as CELLINA_ABLATED_DO_COUNTERFACTUAL

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, choices=['crc', 'merfish'], help="Name of dataset (used for configs)")
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", required=True)
    p.add_argument("--model_class", required=True, choices=['cellina', 'cpa', 'cellina_graph', 'scgen'], help="one of: cellina, cpa, cellina_graph, scgen")
    p.add_argument("--model_name", default=None, help="folder name for saving model and outputs")
    p.add_argument("--inference_only", action='store_true', help="Skip training and only do inference on trained model (default False)")

    return p.parse_args()


def _to_array(x):
    if x is None:
        return None
    toarray = getattr(x, "toarray", None)
    if callable(toarray):
        return toarray()
    return np.asarray(x)


def _reconstruct_model_output(model, adata_obj, model_class, return_normalized=False, batch_size=512):
    """Model-agnostic adapter to obtain reconstructions for adata_obj as numpy array.
    - For scvi/cellina: use get_normalized_expression if available
    - For CPA-like models: call predict() which may write into adata_obj.obsm['CPA_pred'] or return array
    """
    model_class = model_class.lower()
    if model_class == "cpa":
        out = None
        # many CPA implementations write predictions into adata.obsm['CPA_pred']
        out = model.predict(adata_obj, batch_size=batch_size)
        if "CPA_pred" in adata_obj.obsm:
            X = _to_array(adata_obj.obsm["CPA_pred"])  # likely raw counts
            X = _normalize_counts(X, eps=1e-8, scale=COUNTS_PER_K) if return_normalized else X
            return X
        if out is not None:
            return _to_array(out)
        raise RuntimeError("CPA model produced no output and did not populate adata.obsm['CPA_pred']")

    # other models (cellina or generic models exposing get_normalized_expression)
    if "cellina" in model_class:
        library_size = COUNTS_PER_K if return_normalized else "latent"
        out = model.get_normalized_expression(adata_obj, library_size=library_size, batch_size=batch_size)
        return _to_array(out)
    if model_class == "scgen":
        out = model.get_decoded_expression(adata_obj, batch_size=batch_size)
        out = out.clip(min=1e-8)
        out = np.clip(np.expm1(out), 0, None)
        out = _normalize_counts(out, eps=1e-8, scale=COUNTS_PER_K) if return_normalized else out
        return _to_array(out)

    raise RuntimeError("Model does not expose a known reconstruction API (get_normalized_expression or predict)")


def save_recon_adata(adata, X_new, save_path, latents=None):
    # build adata with only obs and var copied, and recon in obsm
    ad_recon = ad.AnnData(X=X_new)
    ad_recon.obs = adata.obs.copy()
    ad_recon.var = adata.var.copy()
    if latents is not None:
        ad_recon.obsm[f"latents"] = latents

    ad_recon.write_h5ad(save_path, compression="gzip")
    return save_path


def split_indices(
    adata,
    holdout_celltype,
    labels_key=DEFAULT_LABELS_KEY,
    domains_key=DEFAULT_DOMAINS_KEY,
    holdout_domains=DEFAULT_HOLDOUT_DOMAINS,
    seed=DEFAULT_SEED,
):
    """Create train/val/test splits consistent with notebooks.

    Test: holdout_celltype in any domain whose label contains a string from
          holdout_domains (substring match — e.g. 'TVA' matches 'TVA1', 'TVA2').
    Val: 10% of remaining trainval (random)
    """
    if holdout_celltype not in adata.obs[labels_key].unique():
        raise ValueError(f"holdout_celltype '{holdout_celltype}' not found in adata.obs['{labels_key}'] values")

    domain_str = adata.obs[domains_key].astype(str)
    is_holdout_domain = domain_str.apply(lambda d: any(hd in d for hd in holdout_domains))
    is_holdout_ct = adata.obs[labels_key].astype(str) == holdout_celltype
    test_mask = is_holdout_domain & is_holdout_ct

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


def _preprocess_adata(adata, n_top_genes=2000, n_neighbors=50, labels_key=DEFAULT_LABELS_KEY, domains_key=DEFAULT_DOMAINS_KEY, step_size_px=0.1):
    """Apply preprocessing steps from counterfactuals notebook.
    Modifies and returns adata.
    """
    adata.obs_names_make_unique()

    adata = adata[~adata.obs[domains_key].isna()]
    adata = adata[~adata.obs[labels_key].isna()]

    adata.obs[labels_key] = adata.obs[labels_key].astype("category")
    adata.obs[domains_key] = adata.obs[domains_key].astype("category")
    
    sc.pp.filter_cells(adata, min_counts=3)
    sc.pp.filter_genes(adata, min_counts=3)

    adata.layers['counts'] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, layer='counts', flavor='seurat_v3', n_top_genes=n_top_genes, subset=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    try:
        from cellina._spatial_utils import spatial_neighbors, compute_spatial_features
        spatial_neighbors(adata, bandwidth=100 / step_size_px, max_neighbours=n_neighbors, standardize=False)
        compute_spatial_features(adata)
    except Exception as e:
        print("Warning: cellina spatial pre-processing failed or cellina not available:", e)
    
    adata.X = adata.layers['counts'].copy()

    return adata


def preprocess_crc(adata, n_top_genes=2000, n_neighbors=50, labels_key=DEFAULT_LABELS_KEY, domains_key=DEFAULT_DOMAINS_KEY):
    # Add label column
    if 'coarse_type' not in adata.obs.columns or adata.obs['coarse_type'].isna().any():
        from _labels_to_coarse import LABEL_TO_COARSE as LMAP
        adata.obs['coarse_type'] = adata.obs['ist'].map(LMAP)
        adata.obs['coarse_type'] = adata.obs['coarse_type'].astype('category')

    # Add domain column
    adata.obs["typ_clean"] = (
        adata.obs["typ"]
        .str.extract(r"(REF|TVA|CRC)", expand=False)
    )
    
    # Add spatial coords in correct field
    adata.obsm['spatial'] = adata.obs[['CenterX_global_px', 'CenterY_global_px']].values

    return _preprocess_adata(adata, 
                             n_top_genes=n_top_genes, 
                             n_neighbors=n_neighbors, 
                             labels_key=labels_key, 
                             domains_key=domains_key, 
                             step_size_px=0.12028)


def preprocess_merfish(adata, n_top_genes=1120, n_neighbors=50, labels_key=DEFAULT_LABELS_KEY, domains_key=DEFAULT_DOMAINS_KEY):
    adata.obsm["spatial"] = adata.obsm["X_spatial_coords"]
    adata.X = adata.raw.X.copy()
    adata.layers['counts'] = adata.raw.X.copy()

    return _preprocess_adata(adata, 
                             n_top_genes=n_top_genes, 
                             n_neighbors=n_neighbors, 
                             labels_key=labels_key, 
                             domains_key=domains_key, 
                             step_size_px=0.109)


def train_model(adata, model_class, model_args, train_args, save_dir, plan_kwargs=None, batch_key=DEFAULT_BATCH_KEY, labels_key=DEFAULT_LABELS_KEY, domains_key=DEFAULT_DOMAINS_KEY, splits=None):
    """Train model and save to save_dir. Returns trained model instance."""
    mc = model_class.lower()
    model = None

    if mc == 'cellina':
        import cellina
        from cellina import CellinaModel
        print("cellina version: ", cellina.__version__)
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

    elif mc == 'scgen':
        import pertpy as pt
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        pt.tl.Scgen.setup_anndata(adata, batch_key=domains_key, labels_key=labels_key)
        model = pt.tl.Scgen(adata, **model_args)
        # Add split info
        train_args['datasplitter_kwargs'] = {
                  "external_indexing": [splits[0], splits[1], splits[2]],
                  }
        model.train(**train_args, plan_kwargs=plan_kwargs)

    else:
        raise ValueError(f"Unsupported model_class: {model_class}. Supported: cellina, cpa, cellina_graph, scgen")

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


def _get_latents(model, adata, model_class, batch_size=DEFAULT_BATCH_SIZE):
    latents = None
    if model_class.lower() in ['cellina', 'cellina_graph']:
        latents = model.get_latent_representation(adata=adata, batch_size=batch_size)
    if model_class.lower() == 'cpa':
        latents = model.get_latent_representation(adata=adata, batch_size=batch_size)
        latents = latents["latent_after"].X
    if model_class.lower() == 'scgen':
        latents = model.get_latent_representation(adata=adata, batch_size=batch_size)
    return latents


def run_inference(model, 
                  adata, 
                  adata_path, 
                  model_class, 
                  model_name, 
                  holdout_celltype, 
                  do_cf=True, 
                  batch_size=DEFAULT_BATCH_SIZE, 
                  labels_key=DEFAULT_LABELS_KEY, 
                  domains_key=DEFAULT_DOMAINS_KEY, 
                  return_normalized=False, 
                  control_domains=DEFAULT_CTRL_DOMAINS,
                  holdout_domains=DEFAULT_HOLDOUT_DOMAINS):
    """Run reconstructions for full adata and optional counterfactuals. Returns paths."""

    print("Running inference and saving outputs...")
    input_parent = os.path.dirname(adata_path)
    parent_of_input = os.path.dirname(input_parent)
    input_basename = os.path.splitext(os.path.basename(adata_path))[0]
    out_dir = os.path.join(parent_of_input, input_basename, holdout_celltype)
    os.makedirs(out_dir, exist_ok=True)

    # full reconstruction
    try:
        recon_all = None        
        recon_all = _reconstruct_model_output(model, 
                                                adata[adata.obs[labels_key] == holdout_celltype], 
                                                model_class, 
                                                return_normalized=return_normalized, 
                                                batch_size=batch_size)
        latents = _get_latents(model, 
                                adata[adata.obs[labels_key] == holdout_celltype], 
                                model_class, 
                                batch_size)
    except Exception as e:
        print('Reconstruction failed:', e)
        recon_all = None

    # Save recon to disk
    out_recon_path = None
    if recon_all is not None:
        out_recon_path = os.path.join(out_dir, f"{model_name}_recon_x.h5ad")
        adata_with_obs = adata[adata.obs[labels_key] == holdout_celltype].copy()
        save_recon_adata(adata_with_obs, 
                         recon_all, 
                         out_recon_path, 
                         latents=latents)
        print('Saved recon to', out_recon_path)

    # Compute counterfactuals
    # for space usage reasons, subset to only relevant (OOD) cell type
    # cellina variants need full adata to sample neighbors correctly
    if model_class.lower() not in ['cellina_graph', 'cellina']:
        adata = adata[adata.obs[labels_key] == holdout_celltype]
    
    if do_cf:
        is_control_region = adata.obs[domains_key].isin(control_domains)
        is_holdout_ct = adata.obs[labels_key].astype(str) == holdout_celltype
        mask_control = is_control_region & is_holdout_ct
        idx_control = np.where(mask_control.values)[0]            
        
        # Loop over holdout domains and generate counterfactuals for each
        for hd in holdout_domains:
            is_holdout_region = adata.obs[domains_key].astype(str) == hd
            mask_target = is_holdout_region & is_holdout_ct
            idx_target = np.where(mask_target.values)[0]

            # "neighbour_indices" are indices of the neighbors of idx_target cells
            conn = adata.obsp["spatial_connectivities"]
            sub_conn = conn[idx_target]                # rows for target cells
            neighbor_indices = sub_conn.nonzero()[1]   # all neighbors at once
            neighbor_indices = np.unique(neighbor_indices)
            # remove neighbors having same ct as holdout_ct
            neighbor_indices = neighbor_indices[~is_holdout_ct.values[neighbor_indices]]
                
            if model_class.lower() == 'cpa':
                from cpa._utils import CPA_REGISTRY_KEYS
                # Subset adata - this is how CPA does counterfactuals
                adata_ctrl = adata[idx_control].copy()
                perturbation_idx = model.pert_encoder[hd]
                # Change perturbation label ctrl -> stimulated
                adata_ctrl.obsm['perts'][:, 0] = perturbation_idx
                # Mark as non-control (control flag = 0)
                adata_ctrl.obs[CPA_REGISTRY_KEYS.CONTROL_KEY] = 0

                # Create counterfactuals - normalizing counts at the end before saving, so set False here
                cf_counts = _reconstruct_model_output(model, adata_ctrl, model_class, return_normalized=False, batch_size=batch_size)
                cf_latents = _get_latents(model, adata_ctrl, model_class, batch_size)
            
            if 'cellina' in model_class.lower():
                args_gex = {
                    "indices": idx_control,
                    "batch_size": batch_size,
                    "seed": 0,
                    "neighbour_indices": neighbor_indices
                }
                if model_class.lower() == 'cellina_graph':
                    args_gex["n_neighbors_per_seed"] = 50
                else:
                    args_gex['precomputed'] = False
                
                cf_counts = model.get_counterfactual_expression(**args_gex)
                args_latents = args_gex.copy()
                cf_latents = model.get_counterfactual_latents(**args_latents)

            if model_class.lower() == 'scgen':
                adata_cf, _ = model.predict(adata_to_predict=adata[idx_control].copy(),
                                            ctrl_key=control_domains[0], 
                                            stim_key=hd)
                adata_cf.X = adata_cf.X.clip(min=1e-8)
                cf_counts = adata_cf.X
                cf_counts = np.clip(np.expm1(cf_counts), 0, None)
                cf_latents = model.get_latent_representation(adata=adata_cf, batch_size=batch_size)
            
            # Save counterfactuals
            cf_counts = _normalize_counts(cf_counts, eps=1e-8, scale=COUNTS_PER_K) if return_normalized else cf_counts
            out_cf_path = os.path.join(out_dir, f"{model_name}_counterfactual_x_{hd}.h5ad")
            save_recon_adata(adata[idx_control], 
                             X_new=cf_counts,
                             latents=cf_latents,
                             save_path=out_cf_path)
            print(f"Saved {model_class} counterfactuals to {out_cf_path}")

    return out_recon_path, out_cf_path


def _load_model(save_dir, model_class, adata, splits=None):
    if model_class.lower() == 'cellina':
        from cellina import CellinaModel
        model = CellinaModel.load(save_dir, adata)
    if model_class.lower() == 'cpa':
        import cpa
        adata.obs['dose'] = 1.0 # NOTE: dummy dose for compatibility with CPA model
        adata.obs['data_split'] = 'train'
        adata.obs.iloc[splits[1], adata.obs.columns.get_loc('data_split')] = 'valid'
        adata.obs.iloc[splits[2], adata.obs.columns.get_loc('data_split')] = 'test'
        model = cpa.CPA.load(dir_path=save_dir,
                     adata=adata,
                     use_gpu=True)
    if model_class.lower() == 'scgen':
        import pertpy as pt
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        model = pt.tl.Scgen.load(save_dir, adata)
    if model_class.lower() == 'cellina_graph':
        from cellina_graph import CellinaModel
        model = CellinaModel.load(save_dir, adata)
    
    print(f"{model_class} loaded model from {save_dir}")
    return model


def subset_adata(adata, proportion=0.3, random_state=0):
    n_cells = adata.n_obs
    n_subsample = int(n_cells * proportion)

    # Randomly choose cell indices
    np.random.seed(random_state)  # for reproducibility
    subsample_idx = np.random.choice(n_cells, n_subsample, replace=False)

    # Create the subsampled AnnData
    adata = adata[subsample_idx].copy()

    return adata


def main():
    args = parse_args()

    # choose configs based on model_class
    dataset_name = args.dataset_name.lower()
    mc = args.model_class.lower()
    model_name = args.model_name
    inference_only = args.inference_only
    normalize_counts = False    
    sid = os.path.basename(args.adata_path).split('.h5ad')[0]
    
    if mc == 'cellina':
        model_args = CELLINA_MODEL_ARGS.copy()
        train_args = CELLINA_TRAIN_ARGS.copy()
        plan_kwargs = CELLINA_PLAN_KWARGS.copy()
        do_cf_default = CELLINA_DO_COUNTERFACTUAL
        if model_name == 'cellina-mmd':
            model_args = CELLINA_MMD_MODEL_ARGS.copy()
            train_args = CELLINA_MMD_TRAIN_ARGS.copy()
            plan_kwargs = CELLINA_MMD_PLAN_KWARGS.copy()
            do_cf_default = CELLINA_MMD_DO_COUNTERFACTUAL
        if model_name == 'cellina-ablated':
            model_args = CELLINA_ABLATED_MODEL_ARGS.copy()
            train_args = CELLINA_ABLATED_TRAIN_ARGS.copy()
            plan_kwargs = CELLINA_ABLATED_PLAN_KWARGS.copy()
            do_cf_default = CELLINA_ABLATED_DO_COUNTERFACTUAL
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
    elif mc == 'scgen':
        model_args = SCGEN_MODEL_ARGS.copy()
        train_args = SCGEN_TRAIN_ARGS.copy()
        plan_kwargs = SCGEN_PLAN_KWARGS.copy()
        do_cf_default = SCGEN_DO_COUNTERFACTUAL
    else:
        raise ValueError(f"Unsupported model_class: {args.model_class}")

    # seed for reproducibility
    set_seed(DEFAULT_SEED)

    # load adata
    print("Loading adata:", args.adata_path)
    adata = sc.read(args.adata_path)

    # Subset the data - Put in for scgen slides 120, 210 otherwise segmentation fault for (probably) RAM/VRAM reasons
    if (sid == 'crc_120' and mc == 'scgen'):
        adata = subset_adata(adata, proportion=0.3, random_state=0)
    if (sid == 'crc_210' and mc == 'scgen'):
        adata = subset_adata(adata, proportion=0.3, random_state=0)

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
    batch_key = DATA_ARGS.get('batch_key', DEFAULT_BATCH_KEY)
    control_domains = DATA_ARGS.get('control_domains', DEFAULT_CTRL_DOMAINS)
    holdout_domains = DATA_ARGS.get('holdout_domains', DEFAULT_HOLDOUT_DOMAINS)
    n_neighbors = N_NEIGHBORS_GRAPH if mc=='cellina_graph' else DATA_ARGS.get('n_neighbors', DEFAULT_N_NEIGHBORS)

    if dataset_name == 'crc':
        adata = preprocess_crc(adata, n_top_genes=n_top_genes, n_neighbors=n_neighbors, labels_key=labels_key, domains_key=domains_key)
    elif dataset_name == 'merfish':
        adata = preprocess_merfish(adata, n_top_genes=n_top_genes, n_neighbors=n_neighbors, labels_key=labels_key, domains_key=domains_key)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Supported: crc, merfish")

    # create splits
    train_idx, val_idx, test_idx = split_indices(adata,
                                                 args.holdout_celltype,
                                                 labels_key=labels_key,
                                                 domains_key=domains_key,
                                                 holdout_domains=holdout_domains,
                                                 seed=DEFAULT_SEED)
    
    splits = (train_idx, val_idx, test_idx)
    print(f"n_obs={adata.n_obs} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # decide whether to run counterfactuals from config default
    do_cf = bool(do_cf_default)

    # prepare save dir for model
    save_dir = os.path.join(MODEL_ROOT, sid, args.holdout_celltype, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # train or load for inference only
    if inference_only:
        model = _load_model(save_dir, 
                            model_class=args.model_class,
                            adata=adata,
                            splits=splits
                            )
    else:
        model = train_model(adata,
                            args.model_class, 
                            model_args, 
                            train_args, 
                            save_dir, 
                            labels_key=labels_key,
                            domains_key=domains_key,
                            batch_key=batch_key,
                            plan_kwargs=plan_kwargs, 
                            splits=splits)
    
    # inference
    batch_size = train_args.get('batch_size', DEFAULT_BATCH_SIZE)
    out_recon_path = run_inference(model, 
                                    adata, 
                                    args.adata_path, 
                                    args.model_class, 
                                    model_name, 
                                    args.holdout_celltype, 
                                    do_cf=do_cf, 
                                    batch_size=batch_size, 
                                    labels_key=labels_key,
                                    domains_key=domains_key,
                                    return_normalized=normalize_counts,
                                    control_domains=control_domains,
                                    holdout_domains=holdout_domains,
                                    )

    print("Done. Outputs:")
    pprint({
        'save_dir': save_dir,
        'model_name': model_name,
        'recon_adata': out_recon_path,
    })


if __name__ == '__main__':
    main()
