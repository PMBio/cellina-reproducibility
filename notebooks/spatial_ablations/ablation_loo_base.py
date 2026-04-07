"""
Ablation study: LOO cell-type × domain_classifier_lambda for CellinaBase (no GCN).

Runs HOLDOUT_CELLTYPES × LINK_PREDICTION_WEIGHTS combinations.
All lpw values are run (including 0 — CellinaBase at 0 is distinct from CellinaGraph at 0).
Appends results to the shared CSV alongside cellina-graph rows, then writes a PDF summary.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import numpy as np
import pandas as pd
import scib_metrics

from cellina import CellinaModel as CellinaBase
from cellina._spatial_utils import spatial_neighbors, compute_spatial_features
from perturb_utils import load_crc_slide, split_indices, compute_cf_logfc
from spatial_ablation_utils import (
    SLIDE_ID, LABELS_KEY, DOMAINS_KEY, TOP_N, MIN_CELLS, BATCH_SIZE,
    LIBRARY_SIZE, DEVICES, CF_METRICS, RUN_METRICS,
    RESULTS_CSV,
    _normalize_total, generate_pdf,
)

# ── Config ────────────────────────────────────────────────────────────────────
HOLDOUT_CELLTYPES = ['Epithelial'] # ["Epithelial", "T_cell", "Myeloid"]  # dry-run: ["Epithelial"]
LINK_PREDICTION_WEIGHTS = [0, 0.1] #  [0, 1e-9, 1e-7, 1e-5, 1e-3, 0.1, 0.01, 1., 10, 100]  # dry-run: [0, 0.1]

MAX_EPOCHS  = 1  # dry-run: 1
RESULTS_PDF = "results/ablation_loo_base.pdf"

_SLT = "cellina-base"  # label written to spatial_loss_type column


# ── Main logic ────────────────────────────────────────────────────────────────
def run_one(adata_base, holdout_celltype, lpw):
    """Train one CellinaBase model and return a list of per-cell-type / per-target result dicts."""
    print(
        f"\n{'='*60}\n"
        f"  holdout={holdout_celltype}  "
        f"domain_classifier_lambda={lpw}\n"
        f"{'='*60}"
    )

    adata = adata_base.copy()

    train_idx, val_idx, test_idx = split_indices(
        adata,
        holdout_celltype=holdout_celltype,
        labels_key=LABELS_KEY,
        domains_key=DOMAINS_KEY,
        holdout_domains=("CRC", "TVA"),
    )
    print(f"  train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")

    CellinaBase.setup_anndata(
        adata,
        batch_key=None,
        labels_key=LABELS_KEY,
        domains_key=DOMAINS_KEY,
        layer="counts",
    )

    model = CellinaBase(
        adata,
        n_latent=20,
        classifier_lambda=1,
        discriminator_lambda=1,
        domain_classifier_lambda=lpw,
        condition_on_intrinsic=False,
    )

    model.train(
        max_epochs=MAX_EPOCHS,
        check_val_every_n_epoch=1,
        early_stopping=True,
        early_stopping_patience=5,
        early_stopping_monitor="vae_loss_validation",
        train_size=0.9,
        validation_size=0.1,
        plan_kwargs={"lr": 1e-3, "weight_decay": 0.0001, "normalize_losses": True},
        datasplitter_kwargs={"external_indexing": [train_idx, val_idx, test_idx]},
        enable_checkpointing=True,
        batch_size=BATCH_SIZE,
        devices=DEVICES,
    )

    # ── Per-run scalars ───────────────────────────────────────────────────────
    marginal_ll = model.get_marginal_ll(adata, indices=model.test_indices_, return_mean=True)
    print(f"  marginal_ll={marginal_ll:.4f}")

    adata.obsm["s"] = model.get_latent_representation(latent_key="s")
    rng = np.random.default_rng(seed=42)
    sub_idx = rng.choice(adata.n_obs, size=int(adata.n_obs * 0.25), replace=False)
    ari_nmi = scib_metrics.nmi_ari_cluster_labels_kmeans(
        labels=adata.obs[DOMAINS_KEY].values[sub_idx],
        X=adata.obsm["s"][sub_idx],
    )
    ari = float(ari_nmi["ari"])
    nmi = float(ari_nmi["nmi"])
    print(f"  ARI={ari:.4f}  NMI={nmi:.4f}")

    # ── Build target domain pools ─────────────────────────────────────────────
    domains = [d for d in adata.obs[DOMAINS_KEY].astype(str).unique() if d != "nan"]
    ref_label  = next(d for d in domains if "REF" in d)
    crc_label  = next(d for d in domains if "CRC" in d)
    tva_labels = [d for d in domains if "TVA" in d]
    target_labels = [crc_label] + tva_labels

    cell_types = [
        ct for ct in adata.obs[LABELS_KEY].cat.categories
        if ((adata.obs[DOMAINS_KEY] == ref_label) & (adata.obs[LABELS_KEY] == ct)).any()
        and any(
            ((adata.obs[DOMAINS_KEY] == tl) & (adata.obs[LABELS_KEY] == ct)).any()
            for tl in target_labels
        )
    ]

    target_pools = {
        tl: np.where(adata.obs[DOMAINS_KEY].astype(str) == tl)[0]
        for tl in target_labels
    }

    rows = []
    for target_label in target_labels:
        target_all_idx = target_pools[target_label]
        target_short = next(k for k in ("CRC", "TVA") if k in target_label)

        for ct in sorted(cell_types):
            ref_mask    = (adata.obs[LABELS_KEY] == ct) & (adata.obs[DOMAINS_KEY] == ref_label)
            target_mask = (adata.obs[LABELS_KEY] == ct) & (adata.obs[DOMAINS_KEY] == target_label)

            if ct == holdout_celltype:
                target_mask = target_mask & adata.obs["is_holdout"]

            ref_idx    = np.where(ref_mask.values)[0]
            target_idx = np.where(target_mask.values)[0]

            if len(ref_idx) < MIN_CELLS or len(target_idx) < MIN_CELLS:
                print(f"  skip {ct} ({target_short}): ref={len(ref_idx)}, target={len(target_idx)}")
                continue

            print(f"  {ct} ({target_short}): ref={len(ref_idx)}, target={len(target_idx)}")

            ref_arr  = adata[ref_idx].layers["counts"]
            ref_expr = ref_arr.toarray() if hasattr(ref_arr, "toarray") else np.asarray(ref_arr)
            ref_expr = _normalize_total(ref_expr)

            tgt_arr  = adata[target_idx].layers["counts"]
            cf_expr  = tgt_arr.toarray() if hasattr(tgt_arr, "toarray") else np.asarray(tgt_arr)
            cf_expr  = _normalize_total(cf_expr)

            pert_expr = model.get_counterfactual_expression(
                indices=ref_idx,
                neighbour_indices=target_all_idx,
                batch_size=BATCH_SIZE,
                n_neighbours=50,
                library_size=LIBRARY_SIZE,
            )

            stats = compute_cf_logfc(
                ref_expr, pert_expr, cf_expr,
                top_n=TOP_N,
                gene_names=adata.var_names.tolist(),
            )

            rows.append(dict(
                holdout_celltype=holdout_celltype,
                spatial_loss_type=_SLT,
                link_prediction_weight=lpw,
                target=target_short,
                cell_type=ct,
                is_holdout=(ct == holdout_celltype),
                n_ref=len(ref_idx),
                n_target=len(target_idx),
                pearson_r=stats["pearson_r"],
                spearman_r=stats["spearman_r"],
                precision=stats["precision"],
                mixing_index=stats["mixing_index"],
                marginal_ll=marginal_ll,
                ari=ari,
                nmi=nmi,
            ))

    return rows


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading data and computing spatial graph...")
    adata_base = load_crc_slide(SLIDE_ID, labels_key=LABELS_KEY, domains_key=DOMAINS_KEY)
    spatial_neighbors(adata_base, bandwidth=100 / 0.12028, max_neighbours=50, standardize=False)
    compute_spatial_features(adata_base)

    # Resume from existing CSV if present
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        done = set(zip(
            existing.loc[existing["spatial_loss_type"] == _SLT, "holdout_celltype"],
            existing.loc[existing["spatial_loss_type"] == _SLT, "link_prediction_weight"],
        ))
        all_rows = existing.to_dict("records")
        print(f"Resuming: {len(done)} cellina-base run(s) already done → {RESULTS_CSV}")
    else:
        done = set()
        all_rows = []

    for holdout_ct in HOLDOUT_CELLTYPES:
        for lpw in LINK_PREDICTION_WEIGHTS:
            if (holdout_ct, lpw) in done:
                print(f"  skip (already done): holdout={holdout_ct}, lpw={lpw}")
                continue

            rows = run_one(adata_base, holdout_ct, lpw)
            all_rows.extend(rows)

            pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)
            print(f"  Saved {len(all_rows)} rows → {RESULTS_CSV}")

    results_df = pd.read_csv(RESULTS_CSV)
    generate_pdf(results_df, RESULTS_PDF, HOLDOUT_CELLTYPES)


if __name__ == "__main__":
    main()
