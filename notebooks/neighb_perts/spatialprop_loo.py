#!/usr/bin/env python
"""SpatialProp LOO benchmark script.

For each holdout cell type, trains a spatial-prop GNN with that type's CRC
cells held out, then predicts how their expression changes under the CRC
spatial context (using cell-type-specific logFC perturbations).

Results are written to {out_dir}/spatialprop_results_{slide_id}.csv.

Usage (run from notebooks/neighb_perts/):
    CUDA_VISIBLE_DEVICES=0 python spatialprop_loo.py \\
        --slide_id 242 \\
        --groupby "T_cell,Fibroblast,Endothelial,Myeloid,Epithelial,B_cell" \\
        --top_n_perturb 100 \\
        --top_n 100 \\
        --out_dir results/perturb_benchmark
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../scripts')
from perturb_utils import (
    DEFAULT_GROUPBY,
    compute_cf_logfc,
    compute_pseudobulk_logfc,
    load_crc_slide,
    split_indices,
    total_normalize,
)


from spatial_gnn.api.perturbation_api import (
    create_perturbation_input_matrix,
    train_perturbation_model,
)
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.models.inference import predict
from spatial_gnn.utils.dataset_utils import (
    create_dataloader_from_dataset,
    load_model_from_path,
)


def _clean_incomplete_gnn_dirs(base_dir: str = "data/gnn_datasets") -> None:
    """Remove GNN dataset subdirs that have no manifest.json (crashed mid-processing)."""
    if not os.path.exists(base_dir):
        return
    for top in os.scandir(base_dir):
        if not top.is_dir():
            continue
        for sub in os.scandir(top.path):
            if sub.is_dir() and not os.path.exists(os.path.join(sub.path, "manifest.json")):
                print(f"  Removing incomplete GNN dataset cache: {sub.path}")
                shutil.rmtree(sub.path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slide_id", type=int, default=242)
    p.add_argument(
        "--groupby", type=str, default=None,
        help="Comma-separated list of holdout cell types. "
             "Defaults to: " + ",".join(DEFAULT_GROUPBY),
    )
    p.add_argument("--top_n_perturb", type=int, default=100,
                   help="Number of top logFC genes used to perturb neighbor expression.")
    p.add_argument("--top_n", type=int, default=50,
                   help="Number of top logFC genes used for metric evaluation.")
    p.add_argument("--out_dir", type=str, default="results/perturb_benchmark")
    p.add_argument("--min_cells", type=int, default=50)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)
    return p.parse_args()


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
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    groupby_list = (
        [ct.strip() for ct in args.groupby.split(",")]
        if args.groupby
        else DEFAULT_GROUPBY
    )

    labels_key = "coarse_type"
    domains_key = "typ"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    for holdout_ct in groupby_list:
        print(f"\n{'='*60}")
        print(f"Holdout cell type: {holdout_ct}")
        print(f"{'='*60}")

        # 1. Load data
        adata = load_crc_slide(
            args.slide_id, labels_key=labels_key, domains_key=domains_key
        )

        if holdout_ct not in adata.obs[labels_key].values:
            print(f"  WARNING: '{holdout_ct}' not in adata — skipping")
            continue

        # spatial_gnn expects 'celltype', 'region', and 'mouse_id' columns
        adata.obs["celltype"] = adata.obs[labels_key]
        adata.obs["mouse_id"] = str(args.slide_id)
        adata.obs["region"] = adata.obs[domains_key]

        # 2. Holdout split (marks adata.obs['is_holdout'])
        train_idx, val_idx, test_idx = split_indices(
            adata,
            holdout_celltype=holdout_ct,
            labels_key=labels_key,
            domains_key=domains_key,
        )
        print(
            f"  train={len(train_idx):,}  val={len(val_idx):,}  "
            f"test (holdout)={len(test_idx):,}"
        )

        # 3. Save train/test h5ad files
        exp_name = f"crc_loo_{holdout_ct}"
        out_dir_ct = os.path.join(args.out_dir, exp_name)
        os.makedirs(out_dir_ct, exist_ok=True)

        train_path = os.path.join(out_dir_ct, "adata_train.h5ad")
        test_path = os.path.join(out_dir_ct, "adata_test.h5ad")
        perturbed_path = os.path.join(out_dir_ct, "adata_test_perturbed.h5ad")

        adata[~adata.obs["is_holdout"]].copy().write_h5ad(train_path)
        adata.copy().write_h5ad(test_path)

        # 4. Train spatial-prop GNN on training data
        training_args = dict(
            dataset=f"crc_{args.slide_id}_{holdout_ct}_loo",
            file_path=train_path,
            train_ids=[str(args.slide_id)],
            test_ids=[str(args.slide_id)],
            exp_name=exp_name,
            k_hop=2,
            augment_hop=2,
            center_celltypes="all",
            node_feature="expression",
            inject_feature="none",
            learning_rate=1e-3,
            loss="weightedl1",
            epochs=args.max_epochs,
            normalize_total=True,
            num_cells_per_ct_id=100,
            predict_celltype=False,
            pool="center",
            do_eval=False,
            device=device,
        )
        _clean_incomplete_gnn_dirs()
        _, gene_names, (gnn_model, model_config, trained_model_path) = (
            train_perturbation_model(**training_args)
        )
        print(f"  Model saved to: {trained_model_path}")

        # 5. Compute pseudobulk logFC → perturbation dict
        domain_logfc_df, ref_label, crc_label = compute_pseudobulk_logfc(
            adata, labels_key, domains_key
        )

        perturbation_dict = {}
        for ct in domain_logfc_df.index:
            s = domain_logfc_df.loc[ct]
            top_genes = s.abs().nlargest(args.top_n_perturb).index.tolist()
            perturbation_dict[ct] = s[top_genes].to_dict()

        # 6. Create perturbed input matrix
        adata_test = sc.read_h5ad(test_path)
        create_perturbation_input_matrix(
            adata_test,
            perturbation_dict,
            save_path=perturbed_path,
            normalize_total=True,
        )

        # 7. Run GNN inference restricted to holdout cell type
        adata_result = predict_for_holdout(
            perturbed_path,
            trained_model_path,
            exp_name,
            center_celltypes=[holdout_ct],
            use_ids=[str(args.slide_id)],
            device=device,
            batch_size=args.batch_size,
        )

        # 8. Extract expressions and compute metrics
        mask_ref = (
            (adata_result.obs["celltype"] == holdout_ct)
            & (adata_result.obs["region"] == ref_label)
        )
        mask_crc = (
            (adata_result.obs["celltype"] == holdout_ct)
            & (adata_result.obs["region"] == crc_label)
        )

        n_ref = int(mask_ref.sum())
        n_crc = int(mask_crc.sum())
        print(f"  [spatialprop] {holdout_ct}: ref={n_ref}, crc={n_crc}")

        if n_ref < args.min_cells or n_crc < args.min_cells:
            print(f"  skip {holdout_ct}: too few cells (need ≥ {args.min_cells})")
        else:
            ref_expr = total_normalize(adata_result[mask_ref].X)
            pert_expr = total_normalize(
                # NOTE: we don't use predicted_tempered to avoid leaking info from the heldout CRC distribution into the perturbation prediction
                adata_result[mask_crc].layers["predicted_perturbed"]
            )
            obs_expr = total_normalize(adata_result[mask_crc].X)

            stats = compute_cf_logfc(
                ref_expr, pert_expr, obs_expr,
                top_n=args.top_n,
                gene_names=adata_result.var_names.tolist(),
            )

            results.append(
                dict(
                    slide_id=args.slide_id,
                    holdout_celltype=holdout_ct,
                    cell_type=holdout_ct,
                    method="spatialprop",
                    is_holdout=True,
                    n_ref=n_ref,
                    n_crc=n_crc,
                    pearson_r=stats["pearson_r"],
                    spearman_r=stats["spearman_r"],
                    precision=stats["precision"],
                    mixing_index=stats["mixing_index"],
                    edistance=stats["edistance"],
                    rmse=stats["rmse"],
                    top_n_perturb=args.top_n_perturb,
                    top_n=args.top_n,
                )
            )

        # Cleanup temporary h5ad files
        for fpath in [train_path, test_path, perturbed_path]:
            if os.path.exists(fpath):
                os.remove(fpath)

        # Save incrementally so a crash mid-run preserves completed iterations
        out_path = os.path.join(args.out_dir, f"spatialprop_results_{args.slide_id}.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"  → {len(results)} rows saved to {out_path}")

    # Final save
    out_path = os.path.join(args.out_dir, f"spatialprop_results_{args.slide_id}.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} rows → {out_path}")


if __name__ == "__main__":
    main()
