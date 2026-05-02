import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import torch
import anndata as ad

from typing import Dict, Optional
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr

sys.path.append('../scripts')
from train_loo import preprocess_crc, preprocess_merfish
from counterfactual_analysis import compute_rmse, compute_edistance, mixing_index, get_lfc, precision, direction_match, compute_mse_lfc

from perturb_utils import compute_pseudobulk_logfc, total_normalize
from spatialprop_train_loo import clean_all_dirs

#from spatial_gnn.api.perturbation_api import (
#    create_perturbation_input_matrix,
#)
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.models.inference import predict
from spatial_gnn.utils.dataset_utils import (
    create_dataloader_from_dataset,
    load_model_from_path,
)

from configs.adata_crc_config import ADATA_ARGS as ADATA_CRC_ARGS
from configs.adata_merfish_config import ADATA_ARGS as ADATA_MERFISH_ARGS

DATASET_NAME = "merfish"  # Options: ['crc', 'merfish']

CRC_BASE_PATH = "/data2/a330d/datasets/crc/raw_zenodo"
CRC_SLIDES = ['crc_232', 'crc_242', 'crc_231', 'crc_210', 'crc_221', 'crc_120']
CRC_CELLTYPES = [
    "Endothelial",
    "Epithelial",
    "Fibroblast",
    "Myeloid",
    "T_cell",
]

MERFISH_BASE_PATH = "/data/a330d/datasets/MERFISH_mouse_brain"
MERFISH_SLIDES = ['C57BL6J-2.036', 'C57BL6J-2.039', 'C57BL6J-2.041']
MERFISH_CELLTYPES = [
    'glutamatergic neuron',
    'GABAergic neuron',
    'astrocyte',
    'oligodendrocyte',
    'endothelial cell',
]

ADATA_BASE_PATH = CRC_BASE_PATH if DATASET_NAME == "crc" else MERFISH_BASE_PATH
SLIDES = CRC_SLIDES if DATASET_NAME == "crc" else MERFISH_SLIDES
CELLTYPES = CRC_CELLTYPES if DATASET_NAME == "crc" else MERFISH_CELLTYPES
DATA_ARGS = ADATA_CRC_ARGS if DATASET_NAME == "crc" else ADATA_MERFISH_ARGS

node_pert = True
top_n = 50
min_cells = 50
batch_size = 1024
library_size = 1e4
labels_key = DATA_ARGS.get('labels_key')
domains_key = DATA_ARGS.get('domains_key')
n_top_genes = DATA_ARGS.get('n_top_genes')
top_n_perturb = n_top_genes if not node_pert else 200
device = "cuda:1" if torch.cuda.is_available() else "cpu"
n_neighbors = DATA_ARGS.get('n_neighbors')
control_domain = DATA_ARGS.get('control_domains')[0]  # Assuming only one control domain for simplicity
holdout_domains = DATA_ARGS.get('holdout_domains')
out_dir = "/data/a330d/tmp/"
model_base_path = '.'
results_csv_name = f'../results/loo_spatialprop_{DATASET_NAME}_DEG_{top_n}'
results_csv_path = results_csv_name + '.csv' if not node_pert else results_csv_name + '_pert.csv'


def create_perturbation_input_matrix(
    adata: ad.AnnData,
    perturbation_dict: Dict[str, Dict[str, float]],
    mask_key: str = 'perturbed_input',
    save_path: Optional[str] = None,
    normalize_total: bool = True,
    operation: str = 'multiply',
) -> str:
    """
    Store a full perturbed expression matrix in adata.obsm[mask_key] with the same
    normalization as the training data.

    Parameters
    ----------
    operation : {'multiply', 'add'}, default 'multiply'
        How to apply each perturbation value to the existing expression:
        - 'multiply': perturbed = expression * value
        - 'add':      perturbed = expression + value
    """
    if operation not in ('multiply', 'add'):
        raise ValueError(
            f"operation must be 'multiply' or 'add', got '{operation}'"
        )

    perturbed_adata = adata.copy()

    X = perturbed_adata.X
    if issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    perturbed = X.copy()  # start from normalized expression

    for cell_type, gene_values in perturbation_dict.items():
        cell_mask = perturbed_adata.obs['celltype'] == cell_type
        cell_indices = np.where(cell_mask)[0]

        if len(cell_indices) == 0:
            print(f"Warning: No cells found for cell type '{cell_type}'")
            continue

        print(f"Applying perturbations to {len(cell_indices)} cells of type '{cell_type}'")

        for gene_name, value in gene_values.items():
            if gene_name in perturbed_adata.var_names:
                gene_idx = perturbed_adata.var_names.get_loc(gene_name)

                if operation == 'multiply':
                    perturbed[cell_indices, gene_idx] *= value
                    print(f"  - Gene '{gene_name}': multiplier = {value}")
                else:  # 'add'
                    perturbed[cell_indices, gene_idx] += value
                    # clip negatives that can arise from additive perturbation
                    np.clip(
                        perturbed[cell_indices, gene_idx],
                        a_min=0,
                        a_max=None,
                        out=perturbed[cell_indices, gene_idx],
                    )
                    print(f"  - Gene '{gene_name}': addend = {value}")
            else:
                print(f"Warning: Gene '{gene_name}' not found in data")

    if normalize_total:
        target_sum = X.shape[1]
        row_sums = perturbed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid /0
        perturbed = perturbed / row_sums * target_sum
        perturbed_adata.obsm[mask_key] = perturbed

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        perturbed_adata.write(save_path)
        print(f"Saved AnnData with perturbation input to: {save_path}")

    return save_path


def predict_for_holdout(
    adata_path, model_path, exp_name, center_celltypes, use_ids=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 512,
):
    """Run spatial-prop inference restricted to center_celltypes.

    Mirrors the predict_for_holdout wrapper in spatialprop_loo.ipynb.
    whole_tissue=False + num_cells_per_ct_id=100_000 selects all cells of
    the requested type without sampling.
    """
    test_adata = sc.read_h5ad(adata_path)
    model, model_config = load_model_from_path(model_path, device)
    celltypes_to_index = model_config["celltypes_to_index"]

    shared_kwargs = dict(
        subfolder_name="predict_holdout",
        dataset_prefix=exp_name,
        target="expression",
        k_hop=2,
        augment_hop=0,
        node_feature="expression",
        inject_feature=None,
        num_cells_per_ct_id=100_000,
        center_celltypes=center_celltypes,
        whole_tissue=False,
        use_ids=use_ids,
        raw_filepaths=[adata_path],
        celltypes_to_index=celltypes_to_index,
        normalize_total=True,
    )

    test_dataset = SpatialAgingCellDataset(**shared_kwargs)
    test_dataset.process()

    perturbed_test_dataset = SpatialAgingCellDataset(
        **shared_kwargs,
        perturbation_mask_key="perturbed_input",
        use_perturbed_expression=True,
    )
    perturbed_test_dataset.process()

    loader_kwargs = dict(
        batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True,
    )
    _, test_loader = create_dataloader_from_dataset(test_dataset, **loader_kwargs)
    _, pert_loader = create_dataloader_from_dataset(
        perturbed_test_dataset, **loader_kwargs
    )

    return predict(
        model=model,
        adata=test_adata,
        dataloader=test_loader,
        perturbed_dataloader=pert_loader,
        use_ids=use_ids,
        temper_method="distribution_renormalize",
        device=device,
    )


def main():
    results = []
    for slide_id in SLIDES:
        print(f"\n{'='*60}\nProcessing slide {slide_id}\n{'='*60}")
        adata = sc.read_h5ad(f"{ADATA_BASE_PATH}/{slide_id}.h5ad")
        if DATASET_NAME == 'crc':
            adata = preprocess_crc(adata, n_top_genes=n_top_genes, n_neighbors=n_neighbors, labels_key=labels_key, domains_key=domains_key)
        elif DATASET_NAME == 'merfish':
            adata = preprocess_merfish(adata, n_top_genes=n_top_genes, n_neighbors=n_neighbors, labels_key=labels_key, domains_key=domains_key)
        else:
            raise ValueError(f"Unknown dataset_name: {DATASET_NAME}. Supported: crc, merfish")
        sc.pp.normalize_total(adata, target_sum=library_size)
        sc.pp.log1p(adata)

        for holdout_ct in CELLTYPES:
            # Set holdout set - cells having holdout_ct and holdout_domains
            mask_holdout = (adata.obs[labels_key] == holdout_ct) & (adata.obs[domains_key].isin(holdout_domains))
            adata.obs['is_holdout'] = mask_holdout
            print(f"\n{'='*60}")
            print(f"Holdout cell type: {holdout_ct}")
            print(f"{'='*60}")

            # 1. Load data
            if holdout_ct not in adata.obs[labels_key].values:
                print(f"  WARNING: '{holdout_ct}' not in adata — skipping")

            # spatial_gnn expects 'celltype', 'region', and 'mouse_id' columns
            adata.obs["celltype"] = adata.obs[labels_key]
            adata.obs["mouse_id"] = str(slide_id)
            adata.obs["region"] = adata.obs[domains_key]            

            # 2. Get test h5ad file paths
            exp_name = f"{slide_id}_loo_{holdout_ct}"
            out_dir_ct = os.path.join(out_dir, exp_name)
            test_path = os.path.join(out_dir_ct, "adata_test.h5ad")
            perturbed_path = os.path.join(out_dir_ct, "adata_test_perturbed.h5ad")


            for hd in holdout_domains:
                # 3. Compute pseudobulk logFC → perturbation dict
                domain_logfc_df = compute_pseudobulk_logfc(
                    adata, labels_key, domains_key, control_domain=control_domain, holdout_domain=hd
                )
                perturbation_dict = {}
                s = domain_logfc_df.loc[holdout_ct]
                top_genes = s.abs().nlargest(top_n_perturb).index.tolist()
                perturbation_dict[holdout_ct] = np.exp(s[top_genes]).to_dict()

                # 4. Create perturbed input matrix
                adata_test = sc.read_h5ad(test_path)
                create_perturbation_input_matrix(
                    adata_test,
                    perturbation_dict,
                    save_path=perturbed_path,
                    normalize_total=True,
                    operation='add',
                )

                # 5. Run GNN inference restricted to holdout cell type
                trained_model_path = f'{model_base_path}/output/{exp_name}/{slide_id}_{holdout_ct}_loo_expression_2hop_2augment_expression_none/weightedl1_1en03/model.pth'  
                adata_result = predict_for_holdout(
                    perturbed_path,
                    trained_model_path,
                    exp_name,
                    center_celltypes=[holdout_ct],
                    use_ids=[str(slide_id)],
                    device=device,
                    batch_size=batch_size,
                )
                
                # 6. Compute eval stats
                mask_ref = (
                    (adata_result.obs["celltype"] == holdout_ct)
                    & (adata_result.obs["region"] == control_domain)
                )
                mask_crc = (
                    (adata_result.obs["celltype"] == holdout_ct)
                    & (adata_result.obs["region"] == hd)
                )

                n_ref = int(mask_ref.sum())
                n_crc = int(mask_crc.sum())
                print(f"  [spatialprop] {holdout_ct}: ref={n_ref}, crc={n_crc}")

                if n_ref < min_cells or n_crc < min_cells:
                    print(f"  skip {holdout_ct}: too few cells (need ≥ {min_cells})")
                else:
                    ref_expr = total_normalize(adata_result[mask_ref].X, target_sum=library_size)
                    pert_expr = total_normalize(
                        # NOTE: we don't use predicted_tempered to avoid leaking info from the heldout CRC distribution into the perturbation prediction
                        adata_result[mask_crc].layers["predicted_perturbed"],
                        target_sum=library_size,
                    )
                    obs_expr = total_normalize(adata_result[mask_crc].X, target_sum=library_size)

                    control = ref_expr
                    target = obs_expr
                    counterfactual = pert_expr

                    gt_lfc, cf_lfc, deg = get_lfc(control=control, target=target, counterfactual=counterfactual, n_deg=top_n)

                    spear, _ = spearmanr(gt_lfc[deg], cf_lfc[deg])
                    pear, _ = pearsonr(gt_lfc[deg], cf_lfc[deg])
                    prec = precision(gt_lfc, cf_lfc, k=top_n, use_abs=True)
                    dir_match = direction_match(gt_lfc, cf_lfc, k=top_n, normalize="intersection")
                    dir_match_k = direction_match(gt_lfc, cf_lfc, k=top_n, normalize="k")
                    dir_match_gt = direction_match(gt_lfc, cf_lfc, k=top_n, normalize="gt_topk")
                    mix_idx = mixing_index(observed=target, predicted=counterfactual, library_size=library_size)
                    edist_global = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=library_size)
                    edist_local = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=library_size, local=True)
                    edist_pca_log = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=library_size, local=True, use_pca=True)
                    edist_pca = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=library_size, local=True, use_pca=True, log1p=False)
                    rmse = compute_rmse(observed=target, predicted=counterfactual, deg=deg, library_size=library_size)
                    mse_lfc = compute_mse_lfc(gt_vec=gt_lfc, cf_vec=cf_lfc, deg=deg)

                    results.append(
                        dict(
                            dataset_name=DATASET_NAME,
                            sid=slide_id,
                            control_domain=control_domain,
                            target_domain=hd,
                            n_deg=top_n,
                            model_name="spatialprop",
                            holdout_celltype=holdout_ct,
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
                            top_n_perturb=top_n_perturb,
                        )
                    )
            
            # Remove spatialprop-generated data files
            #clean_all_dirs()

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{results_csv_path}", index=False)


if __name__ == "__main__":
    main()