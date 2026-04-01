#!/usr/bin/env python
"""Cellina LOO benchmark script.

For each holdout cell type, trains a CellinaModel with that type's CRC cells
held out, then evaluates two inference strategies:
  - cellina_perturb: neighbor logFC perturbation via make_neighbor_perturbation
  - cellina_cf:      direct spatial-context swap via get_counterfactual_expression
                     (holdout cell type only)

Results are written to {out_dir}/cellina_results_{slide_id}.csv.

Usage (run from notebooks/neighb_perts/):
    CUDA_VISIBLE_DEVICES=1 python cellina_loo.py \\
        --slide_id 242 \\
        --groupby "Fibroblast,Endothelial,Myeloid,T_cell,Epithelial,B_cell" \\
        --top_n_perturb 100 \\
        --top_n 100 \\
        --out_dir results/perturb_benchmark
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scvi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturb_utils import (
    DEFAULT_GROUPBY,
    compute_cf_logfc,
    compute_pseudobulk_logfc,
    load_crc_slide,
    split_indices,
)

import cellina  # noqa: F401 (version check)
from cellina import CellinaModel, make_neighbor_perturbation
from cellina._spatial_utils import compute_spatial_features, spatial_neighbors

scvi.settings.seed = 0
print(f"cellina {cellina.__version__}")


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
    p.add_argument("--top_n", type=int, default=100,
                   help="Number of top logFC genes used for metric evaluation.")
    p.add_argument("--out_dir", type=str, default="results/perturb_benchmark")
    p.add_argument("--min_cells", type=int, default=50,
                   help="Minimum cells in REF and CRC to evaluate a cell type.")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_epochs", type=int, default=30)
    return p.parse_args()


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
    library_size = 1e4

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

        # 2. Spatial graph + features
        spatial_neighbors(
            adata, bandwidth=100 / 0.12028, max_neighbours=200, standardize=False
        )
        compute_spatial_features(adata)

        # 3. Train / val / test split (marks adata.obs['is_holdout'])
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

        # 4. Train CellinaModel (GPU 1 via devices=[1])
        CellinaModel.setup_anndata(
            adata,
            batch_key=None,
            labels_key=labels_key,
            domains_key=domains_key,
            layer="counts",
            spatial_obsm_key="spatial_x",
        )
        model = CellinaModel(
            adata,
            n_latent=20,
            classifier_lambda=1,
            discriminator_lambda=1,
            condition_on_intrinsic=False,
        )
        model.train(
            max_epochs=args.max_epochs,
            check_val_every_n_epoch=1,
            early_stopping=True,
            early_stopping_patience=5,
            early_stopping_monitor="vae_loss_validation",
            plan_kwargs={"lr": 1e-3, "weight_decay": 0.0001, "normalize_losses": True},
            datasplitter_kwargs={
                "external_indexing": [train_idx, val_idx, test_idx]
            },
            enable_checkpointing=True,
            batch_size=args.batch_size,
            devices=[0],  # CUDA_VISIBLE_DEVICES remaps physical GPU to index 0
        )
        model_save_dir = os.path.join(
            args.out_dir, "trained", f"crc_{args.slide_id}_{holdout_ct}"
        )
        model.save(model_save_dir, overwrite=True)

        # 5. Pseudobulk logFC (CRC vs REF) per cell type
        domain_logfc_df, ref_label, crc_label = compute_pseudobulk_logfc(
            adata, labels_key, domains_key
        )

        # 6. Build per-cell-type logFC series (top top_n_perturb genes by |logfc|)
        logfc_series_dict = {}
        for ct in domain_logfc_df.index:
            s = domain_logfc_df.loc[ct]
            top_genes = s.abs().nlargest(args.top_n_perturb).index.tolist()
            logfc_series_dict[ct] = s[top_genes]

        # 7. Apply neighbor perturbation → stores counterfactual spatial features
        make_neighbor_perturbation(
            adata,
            perturbations=logfc_series_dict,
            groupby=labels_key,
            obsm_key_out="spatial_x_cf",
            base=np.e,
        )

        # ── Method: cellina_perturb ─────────────────────────────────────────
        # Evaluate all cell types that have both REF and CRC representation.
        for ct in sorted(domain_logfc_df.index):
            ref_mask = (
                (adata.obs[labels_key] == ct)
                & (adata.obs[domains_key] == ref_label)
            )
            crc_mask = (
                (adata.obs[labels_key] == ct)
                & (adata.obs[domains_key] == crc_label)
            )
            # For the holdout type, restrict CRC side to held-out cells only
            if ct == holdout_ct:
                crc_mask = crc_mask & adata.obs["is_holdout"]

            ref_idx_ct = np.where(ref_mask.values)[0]
            crc_idx_ct = np.where(crc_mask.values)[0]

            if (
                len(ref_idx_ct) < args.min_cells
                or len(crc_idx_ct) < args.min_cells
            ):
                print(
                    f"  [perturb] skip {ct}: "
                    f"ref={len(ref_idx_ct)}, crc={len(crc_idx_ct)}"
                )
                continue

            print(
                f"  [perturb] {ct}: ref={len(ref_idx_ct)}, crc={len(crc_idx_ct)}"
            )

            ref_expr = sc.pp.normalize_total(
                adata[ref_idx_ct], target_sum=library_size, inplace=False
            )["X"]
            ref_expr = (
                ref_expr.toarray()
                if hasattr(ref_expr, "toarray")
                else np.asarray(ref_expr, dtype=np.float32)
            )

            pert_expr = model.get_perturbed_expression(
                adata=adata,
                indices=ref_idx_ct,
                spatial_obsm_key="spatial_x_cf",
                batch_size=args.batch_size,
                library_size=library_size,
            )
            obs_expr = model.get_normalized_expression(
                indices=crc_idx_ct,
                batch_size=args.batch_size,
                library_size=library_size,
            )

            stats = compute_cf_logfc(
                ref_expr, pert_expr, obs_expr,
                top_n=args.top_n,
                gene_names=adata.var_names.tolist(),
            )

            results.append(
                dict(
                    slide_id=args.slide_id,
                    holdout_celltype=holdout_ct,
                    cell_type=ct,
                    method="cellina_perturb",
                    is_holdout=(ct == holdout_ct),
                    n_ref=len(ref_idx_ct),
                    n_crc=len(crc_idx_ct),
                    pearson_r=stats["pearson_r"],
                    spearman_r=stats["spearman_r"],
                    precision=stats["precision"],
                    mixing_index=stats["mixing_index"],
                    edistance=stats["edistance"],
                    top_n_perturb=args.top_n_perturb,
                    top_n=args.top_n,
                )
            )

        # ── Method: cellina_cf ──────────────────────────────────────────────
        # Counterfactual swap: REF holdout cells get CRC cells' spatial context.
        # Only meaningful for the holdout type (unseen CRC examples).
        ref_mask_ho = (
            (adata.obs[labels_key] == holdout_ct)
            & (adata.obs[domains_key] == ref_label)
        )
        crc_mask_ho = (
            (adata.obs[labels_key] == holdout_ct)
            & (adata.obs[domains_key] == crc_label)
            & adata.obs["is_holdout"]
        )

        ref_idx_ho = np.where(ref_mask_ho.values)[0]
        crc_idx_ho = np.where(crc_mask_ho.values)[0]

        if (
            len(ref_idx_ho) >= args.min_cells
            and len(crc_idx_ho) >= args.min_cells
        ):
            print(
                f"  [cf] {holdout_ct}: ref={len(ref_idx_ho)}, crc={len(crc_idx_ho)}"
            )

            ref_expr = sc.pp.normalize_total(
                adata[ref_idx_ho], target_sum=library_size, inplace=False
            )["X"]
            ref_expr = (
                ref_expr.toarray()
                if hasattr(ref_expr, "toarray")
                else np.asarray(ref_expr, dtype=np.float32)
            )

            swap_expr = model.get_counterfactual_expression(
                ref_idx_ho,
                crc_idx_ho,
                batch_size=args.batch_size,
                library_size=library_size,
            )
            obs_expr = model.get_normalized_expression(
                indices=crc_idx_ho,
                batch_size=args.batch_size,
                library_size=library_size,
            )

            stats = compute_cf_logfc(
                ref_expr, swap_expr, obs_expr,
                top_n=args.top_n,
                gene_names=adata.var_names.tolist(),
            )

            results.append(
                dict(
                    slide_id=args.slide_id,
                    holdout_celltype=holdout_ct,
                    cell_type=holdout_ct,
                    method="cellina_cf",
                    is_holdout=True,
                    n_ref=len(ref_idx_ho),
                    n_crc=len(crc_idx_ho),
                    pearson_r=stats["pearson_r"],
                    spearman_r=stats["spearman_r"],
                    precision=stats["precision"],
                    mixing_index=stats["mixing_index"],
                    edistance=stats["edistance"],
                    top_n_perturb=args.top_n_perturb,
                    top_n=args.top_n,
                )
            )
        else:
            print(
                f"  [cf] skip {holdout_ct}: "
                f"ref={len(ref_idx_ho)}, crc={len(crc_idx_ho)}"
            )

        # Save incrementally so a crash mid-run preserves completed iterations
        out_path = os.path.join(args.out_dir, f"cellina_results_{args.slide_id}.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"  → {len(results)} rows saved to {out_path}")

    # Final save
    out_path = os.path.join(args.out_dir, f"cellina_results_{args.slide_id}.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} rows → {out_path}")


if __name__ == "__main__":
    main()
