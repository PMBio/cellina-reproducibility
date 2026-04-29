import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import torch

sys.path.append('../scripts')
from train_loo import preprocess_crc, preprocess_merfish
from counterfactual_analysis import compute_lfc_metrics, compute_rmse, compute_edistance, mixing_index
from perturb_utils import compute_pseudobulk_logfc, total_normalize
from spatialprop_train_loo import clean_all_dirs

from spatial_gnn.api.perturbation_api import (
    create_perturbation_input_matrix,
)
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

node_pert = False
top_n = 50
min_cells = 50
batch_size = 1024
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
        sc.pp.normalize_total(adata, target_sum=1e4)
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
                perturbation_dict[holdout_ct] = s[top_genes].to_dict()

                # 4. Create perturbed input matrix
                adata_test = sc.read_h5ad(test_path)
                create_perturbation_input_matrix(
                    adata_test,
                    perturbation_dict,
                    save_path=perturbed_path,
                    normalize_total=True,
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
                    ref_expr = total_normalize(adata_result[mask_ref].X)
                    pert_expr = total_normalize(
                        # NOTE: we don't use predicted_tempered to avoid leaking info from the heldout CRC distribution into the perturbation prediction
                        adata_result[mask_crc].layers["predicted_perturbed"]
                    )
                    obs_expr = total_normalize(adata_result[mask_crc].X)

                    control = ref_expr
                    target = obs_expr
                    counterfactual = pert_expr

                    pear, spear, prec, dir_match, deg = compute_lfc_metrics(control=control, target=target, counterfactual=counterfactual, n_deg=top_n)
                    rmse = compute_rmse(observed=target, predicted=counterfactual, deg=deg)
                    edist_global = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=1e4)
                    edist_local = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=1e4, local=True)
                    edist_pca_log = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=1e4, local=True, use_pca=True)
                    edist_pca = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=1e4, local=True, use_pca=True, log1p=False)
                    mix_idx = mixing_index(observed=target, predicted=counterfactual, library_size=1e4)
                    _, _, _, dir_match_k, _ = compute_lfc_metrics(control=control, target=target, counterfactual=counterfactual, n_deg=top_n, direction_match_normalize="k")
                    _, _, _, dir_match_gt, _ = compute_lfc_metrics(control=control, target=target, counterfactual=counterfactual, n_deg=top_n, direction_match_normalize="gt_topk")

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
                            rmse=np.log10(rmse),
                            top_n_perturb=top_n_perturb,
                        )
                    )
            
            # Remove spatialprop-generated data files
            clean_all_dirs()

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{results_csv_path}", index=False)


if __name__ == "__main__":
    main()