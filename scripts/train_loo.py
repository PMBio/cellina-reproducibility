"""
Train a model holding out a cell type (leave-one-out) and save reconstructions and optional counterfactuals.

Usage (examples):

python scripts/train_loo.py \
  --adata_path /data2/a330d/datasets/crc/raw_zenodo/crc_242.h5ad \
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
import numpy as np
import scanpy as sc
import anndata as ad
import sys
import torch

from pprint import pprint
from scipy.sparse import csr_matrix

# defaults based on notebooks (counterfactuals.ipynb)
DEFAULT_HVGS = 2000
DEFAULT_N_NEIGHBORS = 50
DEFAULT_BATCH_SIZE = 512
DEFAULT_SEED = 0
COUNTS_PER_K = 1e4
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
from configs.concert_config import MODEL_ARGS as CONCERT_MODEL_ARGS, TRAIN_ARGS as CONCERT_TRAIN_ARGS, PLAN_KWARGS as CONCERT_PLAN_KWARGS, DO_COUNTERFACTUAL as CONCERT_DO_COUNTERFACTUAL
from configs.cellina_mmd_config import MODEL_ARGS as CELLINA_MMD_MODEL_ARGS, TRAIN_ARGS as CELLINA_MMD_TRAIN_ARGS, PLAN_KWARGS as CELLINA_MMD_PLAN_KWARGS, DO_COUNTERFACTUAL as CELLINA_MMD_DO_COUNTERFACTUAL



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", required=True)
    p.add_argument("--model_class", required=True, choices=['cellina', 'cpa', 'cellina_graph', 'concert', 'cellina_mmd'], help="one of: cellina, cpa, cellina_graph, concert, cellina_mmd")
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
                X = X / (X.sum(axis=1, keepdims=True) + 1e-8) * COUNTS_PER_K
            return X
        if out is not None:
            return _to_array(out)
        raise RuntimeError("CPA model produced no output and did not populate adata.obsm['CPA_pred']")

    # other models (cellina or generic models exposing get_normalized_expression)
    if hasattr(model, "get_normalized_expression"):
        library_size = COUNTS_PER_K if return_normalized else "latent"
        out = model.get_normalized_expression(adata_obj, library_size=library_size, batch_size=batch_size)
        return _to_array(out)

    raise RuntimeError("Model does not expose a known reconstruction API (get_normalized_expression or predict)")


def save_recon_adata(adata_parent, recon_array, out_path, latents=None):
    # build adata with only obs and var copied, and recon in obsm
    #ad_recon = ad.AnnData(X=np.zeros((adata_parent.n_obs, adata_parent.n_vars), dtype=np.float32))
    ad_recon = ad.AnnData(X=recon_array)
    ad_recon.obs = adata_parent.obs.copy()
    ad_recon.var = adata_parent.var.copy()
    if latents is not None:
        ad_recon.obsm[f"latents"] = latents
        
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
        spatial_neighbors(adata, bandwidth=100 / 0.12028, max_neighbours=n_neighbors, standardize=False)
        #spatial_neighbors(adata, bandwidth=np.inf, max_neighbours=n_neighbors, standardize=False)
    except Exception as e:
        print("Warning: spatial_neighbors failed or cellina not available:", e)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    from cellina._spatial_utils import compute_spatial_features
    compute_spatial_features(adata)
    
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

    elif mc == 'concert':
        # CONCERT: instantiate and train using data-derived positional + attribute matrices.
        try:
            import sys
            sys.path.append('/data/a330d/projects/CONCERT/src')
            from concert_map import CONCERT
        except Exception as e:
            raise RuntimeError("CONCERT is not importable; ensure CONCERT code is on PYTHONPATH (see notebooks/concert.ipynb)") from e

        # build pos (spatial + one-hot batch) and simple cell_atts matrix
        # spatial coords
        loc = np.asarray(adata.obsm['spatial']).astype(np.float32)
        from sklearn.preprocessing import MinMaxScaler
        scaler_sp = MinMaxScaler()
        loc_range = 20
        loc = scaler_sp.fit_transform(loc) * loc_range
        loc_dim = loc.shape[1]
        
        # use obs.sid as batch/perturbation codes
        sid_vals = adata.obs[batch_key].astype(str).values
        unique_sids = np.unique(sid_vals)
        sid_to_code = {s: i for i, s in enumerate(sorted(unique_sids))}
        batch_code_full = np.array([sid_to_code[s] for s in sid_vals], dtype=int)

        # coarse_type / labels -> code
        ct_vals = adata.obs[labels_key].astype(str).values
        unique_cts = np.unique(ct_vals)
        ct_to_code = {c: i for i, c in enumerate(sorted(unique_cts))}
        ct_code = np.array([ct_to_code[c] for c in ct_vals], dtype=int)

        # map tissue regions (obs.typ) -> contiguous codes across full adata
        tissue_names = adata.obs[domains_key].astype(str).values
        unique_tissues = np.unique(tissue_names)
        tissue_name_to_code = {name: i for i, name in enumerate(sorted(unique_tissues))}
        tissue_code_full = np.array([tissue_name_to_code[s] for s in tissue_names], dtype=int)

        # full cell attributes (tissue, batch)
        cell_atts_full = np.stack([tissue_code_full, batch_code_full, ct_code], axis=1).astype(int)
        n_batch_full = len(np.unique(batch_code_full))
        batch_full = np.eye(n_batch_full, dtype='float32')[batch_code_full]

        pos = np.concatenate((loc, batch_full), axis=1).astype(np.float32)
        cutoff = np.ones(loc.shape[0], dtype=np.float32) * 0.5

        # Prepare training and testing slices
        train_idx = np.concatenate([splits[0], splits[1]])
        test_idx = splits[2]

        from preprocess import normalize
        adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)

        # build kernel_scale & inducing points to match loc_dim and n_batch_full
        kernel_scale_scalar = 10.0
        kernel_scale_re = np.array([[kernel_scale_scalar] * loc_dim] * n_batch_full, dtype=float)
        inducing_point_steps = 6
        eps = 1e-5
        initial_inducing_points_re = np.mgrid[0:(1+eps):(1./inducing_point_steps), 0:(1+eps):(1./inducing_point_steps)].reshape(2, -1).T * loc_range
        initial_inducing_points_batch = np.zeros((initial_inducing_points_re.shape[0], n_batch_full), dtype=float)
        initial_inducing_points_batch[:, 0] = 1.0
        initial_inducing_points_re = np.concatenate((initial_inducing_points_re, initial_inducing_points_batch), axis=1).astype('float32')
        
        # Prepare kwargs for CONCERT constructor: allow model_args dict to supply specific values
        concert_kwargs = dict(model_args or {})
        # ensure required fields exist (num_genes, cell_atts, initial_inducing_points etc can be provided)
        concert_kwargs.setdefault('cell_atts', cell_atts_full[train_idx])
        concert_kwargs.setdefault('num_genes', adata.n_vars)
        concert_kwargs.setdefault('N_train', len(train_idx))
        concert_kwargs.setdefault('mask_cutoff', cutoff[train_idx])
        concert_kwargs.setdefault('kernel_scale', kernel_scale_re)
        concert_kwargs.setdefault('initial_inducing_points', initial_inducing_points_re)
        concert_kwargs.setdefault('n_batch', n_batch_full)
        
        # instantiate
        model = CONCERT(**concert_kwargs)
        # call train_model with expected signature from notebook
        train_kwargs = dict(train_args or {})
        train_kwargs['model_weights'] = f'{save_dir}/concert_model.pt'
        try:
            model.train_model(
                pos=pos[train_idx],
                ncounts=_to_array(adata.X)[train_idx].astype("float32"),
                raw_counts=_to_array(adata.layers['counts'][train_idx]).astype("float32"),
                size_factors=adata.obs.get('size_factors', None)[train_idx],
                batch=batch_full[train_idx],
                **train_kwargs,
            )
        except Exception as e:
            print("CONCERT training filed with error: ", e)
    else:
        raise ValueError(f"Unsupported model_class: {model_class}. Supported: cellina, cpa, cellina_graph, concert")

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

    # For concert, return everything that will be used for inference to avoid redundant code
    extras = {}
    if mc == 'concert':
        extras = {
            'pos': pos,
            'cell_atts': cell_atts_full,
            'train_idx': train_idx,
            'test_idx': test_idx,
        }
    return model, extras


def run_inference(model, adata, adata_path, model_class, model_name, holdout_celltype, do_cf=True, batch_size=DEFAULT_BATCH_SIZE, labels_key=DEFAULT_LABELS_KEY, extras={}):
    """Run reconstructions for full adata and optional counterfactuals. Returns paths."""
    # determine output directory: go one level up from input path and create a folder
    # named after the input file (without .h5ad). Example: abc/raw/crc_231.h5ad -> abc/crc_231/
    print("Running inference and saving outputs...")
    input_parent = os.path.dirname(adata_path)
    parent_of_input = os.path.dirname(input_parent)
    input_basename = os.path.splitext(os.path.basename(adata_path))[0]
    out_dir = os.path.join(parent_of_input, input_basename, holdout_celltype)
    os.makedirs(out_dir, exist_ok=True)

    # for space usage reasons, subset to only relevant (OOD) cell type
    adata = adata[adata.obs[labels_key] == holdout_celltype] if model_class.lower() != 'concert' else adata

    # full reconstruction
    try:
        recon_all = None
        if model_class.lower() == 'concert':
            # Get positional and attribute matrices as in training
            pos = extras['pos']
            cell_atts = extras['cell_atts']
            train_idx = extras['train_idx']
            test_idx = extras['test_idx']

            sample_indices_train = torch.arange(pos[train_idx].shape[0], dtype=torch.int)
            sample_indices_test  = torch.arange(pos[test_idx].shape[0], dtype=torch.int)

            denoised, _ = model.batching_denoise_counts(X=pos[train_idx], 
                                                        sample_index=sample_indices_train, 
                                                        cell_atts=cell_atts[train_idx], 
                                                        batch_size=batch_size, 
                                                        n_samples=25)
            denoised = denoised / (denoised.sum(axis=1, keepdims=True) + 1e-8) * COUNTS_PER_K
            recon_all = _to_array(denoised)
        else:
            recon_all = _reconstruct_model_output(model, adata, model_class, return_normalized=True, batch_size=batch_size)
    except Exception as e:
        print('Reconstruction failed:', e)
        recon_all = None

    # Save recon to disk
    out_recon_path = None
    if recon_all is not None:
        out_recon_path = os.path.join(out_dir, f"{model_name}_recon_x.h5ad")
        adata_with_obs = adata if model_class!='concert' else adata[extras['train_idx']].copy()
        # Get latents
        latents = model.get_latent_representation(adata=adata, batch_size=batch_size)
        latents = (
            latents
            if model_class not in ["cpa", "concert"] 
            else latents["latent_after"].X
        )
        save_recon_adata(adata_with_obs, recon_all, out_recon_path, latents=latents)
        print('Saved recon to', out_recon_path)

    # Compute counterfactuals
    out_cf_path = None
    if do_cf:
        if model_class.lower() == 'concert':
            # Prepare target cells (holdout & matching coarse_type)
            labels = adata.obs[DEFAULT_LABELS_KEY].astype(str)
            mask_target = (adata.obs['is_holdout']) & (labels == holdout_celltype)
            if mask_target.sum() == 0:
                raise RuntimeError('No target cells for counterfactual in CONCERT inference')

            # call counterfactualPrediction
            perturbed_counts, _ = model.counterfactualPrediction(X=pos[test_idx], 
                                                                    sample_index=sample_indices_test, 
                                                                    cell_atts=cell_atts[test_idx], 
                                                                    batch_size=batch_size, 
                                                                    n_samples=25, 
                                                                    perturb_cell_id=[], 
                                                                    target_cell_tissue=cell_atts[test_idx][:,0], 
                                                                    target_cell_perturbation=cell_atts[test_idx][:,1])
            perturbed_counts = _to_array(perturbed_counts)
            out_cf_path = os.path.join(out_dir, f"{model_name}_counterfactual_x.h5ad")
            try:
                save_recon_adata(adata[mask_target.values].copy(), perturbed_counts, out_cf_path)
                print('Saved CONCERT counterfactuals to', out_cf_path)
            except Exception as e:
                print('Failed to save CONCERT counterfactuals:', e)
        else: # model class is cellina
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
                    # Get latents
                    latents_cf = model.get_latent_representation(adata=adata_cf, batch_size=batch_size)
                    save_recon_adata(adata_cf, recon_cf, out_cf_path, latents=latents_cf)
                    print("Saved counterfactual reconstructions:", out_cf_path)
                except Exception as e:
                    print("Counterfactual inference failed:", e)
                    out_cf_path = None

    return out_recon_path, out_cf_path


def main():
    args = parse_args()

    # choose configs based on model_class
    mc = args.model_class.lower()
    model_name = args.model_name
    print(model_name)
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
    elif mc == 'concert':
        model_args = CONCERT_MODEL_ARGS.copy()
        train_args = CONCERT_TRAIN_ARGS.copy()
        plan_kwargs = CONCERT_PLAN_KWARGS.copy()
        do_cf_default = CONCERT_DO_COUNTERFACTUAL
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
    labels_key = ADATA_ARGS.get('labels_key', DEFAULT_LABELS_KEY)
    domains_key = ADATA_ARGS.get('domains_key', DEFAULT_DOMAINS_KEY)
    batch_key = ADATA_ARGS.get('batch_key', DEFAULT_BATCH_KEY)
    n_neighbors = 10 if mc=='cellina-graph' else ADATA_ARGS.get('n_neighbors', DEFAULT_N_NEIGHBORS)
    adata = preprocess_adata(adata, 
                             n_top_genes=n_top_genes, 
                             n_neighbors=n_neighbors,
                             )

    # create splits
    train_idx, val_idx, test_idx = split_indices(adata, args.holdout_celltype, 
                                                 labels_key=labels_key, 
                                                 domains_key=domains_key, 
                                                 seed=DEFAULT_SEED)
    splits = (train_idx, val_idx, test_idx)
    print(f"n_obs={adata.n_obs} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # decide whether to run counterfactuals from config default
    do_cf = bool(do_cf_default)

    # prepare save dir for model
    sid = args.adata_path.split('/')[-1].split('.')[0]
    save_dir = os.path.join(MODEL_ROOT, sid, args.holdout_celltype, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # train
    model, extras = train_model(adata,
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
    out_recon_path, out_cf_path = run_inference(model, adata, args.adata_path, args.model_class, model_name, args.holdout_celltype, do_cf=do_cf, batch_size=batch_size, labels_key=labels_key, extras=extras)

    print("Done. Outputs:")
    pprint({
        'save_dir': save_dir,
        'model_name': model_name,
        'recon_adata': out_recon_path,
        'counterfactual_adata': out_cf_path,
    })


if __name__ == '__main__':
    main()
