"""
Ablation study: LOO cell-type × spatial_loss_type × link_prediction_weight.

Runs HOLDOUT_CELLTYPES × SPATIAL_LOSS_TYPES × LINK_PREDICTION_WEIGHTS combinations
(skipping domain_clf at lpw=0, which is equivalent to supcon).
Evaluates counterfactuals for both CRC and TVA targets.
Saves results to CSV after every run, then writes a PDF summary.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import numpy as np
import pandas as pd
from cellina_graph import CellinaModel
from cellina_graph._spatial_utils import spatial_neighbors
from perturb_utils import load_crc_slide, split_indices
from spatial_ablation_utils import (
    SLIDE_ID, LABELS_KEY, DOMAINS_KEY, TOP_N, MIN_CELLS, BATCH_SIZE,
    LIBRARY_SIZE, DEVICES, CF_METRICS, RUN_METRICS, METRICS,
    RESULTS_CSV, SLT_MARKERS, SLT_COLORS,
    _normalize_total, evaluate_model, generate_pdf,
)

# ── Config ───────────────────────────────────────────────────────────────────
HOLDOUT_CELLTYPES       = ["Epithelial", "T_cell", "Myeloid"]
SPATIAL_LOSS_TYPES      = ["supcon", "domain_clf"]
LINK_PREDICTION_WEIGHTS = [0, 0.01, 0.1, 1]

MAX_EPOCHS  = 50
RESULTS_PDF = "results/ablation_loo.pdf"


# ── Main logic ────────────────────────────────────────────────────────────────
def run_one(adata_base, holdout_celltype, link_prediction_weight, spatial_loss_type):
    """Train one model and return a list of per-cell-type / per-target result dicts."""
    print(
        f"\n{'='*60}\n"
        f"  holdout={holdout_celltype}  "
        f"spatial_loss_type={spatial_loss_type}  "
        f"link_prediction_weight={link_prediction_weight}\n"
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

    CellinaModel.setup_anndata(
        adata,
        batch_key=None,
        labels_key=LABELS_KEY,
        domains_key=DOMAINS_KEY,
        layer="counts",
        spatial_connectivities_key="spatial_connectivities",
    )

    model = CellinaModel(
        adata,
        n_latent=20,
        convolution_type="gat",
        n_layers=3,
        classifier_lambda=1,
        discriminator_lambda=1,
        link_prediction_weight=link_prediction_weight,
        spatial_loss_type=spatial_loss_type,
        condition_on_intrinsic=False,
        use_batch_norm=False if spatial_loss_type == "supcon" else True,  # NOTE: batch norm might not be consistent with neigh means in supcon
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

    print("  Training complete. Running evaluation...")
    return evaluate_model(
        model, adata, holdout_celltype,
        spatial_loss_type=spatial_loss_type,
        link_prediction_weight=link_prediction_weight,
        marginal_ll_kwargs={"reduce": "mean"},
        cf_extra_kwargs={"n_neighbors_per_seed": 50},
        batch_size_eval_factor=1,
    )


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(RESULTS_CSV)), exist_ok=True)

    print("Loading data and computing spatial graph...")
    adata_base = load_crc_slide(SLIDE_ID, labels_key=LABELS_KEY, domains_key=DOMAINS_KEY)
    spatial_neighbors(adata_base, bandwidth=100 / 0.12028, max_neighbours=50, standardize=False)

    # Resume from existing CSV if present
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        # Only consider cellina-graph runs (not cellina-base rows) when checking done
        existing_graph = existing[existing["spatial_loss_type"] != "cellina-base"]
        done = set(zip(
            existing_graph["holdout_celltype"],
            existing_graph["link_prediction_weight"],
            existing_graph["spatial_loss_type"],
        ))
        all_rows = existing.to_dict("records")
        print(f"Resuming: {len(done)} cellina-graph run(s) already done → {RESULTS_CSV}")
    else:
        done = set()
        all_rows = []

    combos = [
        (hct, lpw, slt)
        for hct in HOLDOUT_CELLTYPES
        for lpw in LINK_PREDICTION_WEIGHTS
        for slt in SPATIAL_LOSS_TYPES
        if not (lpw == 0 and slt == "domain_clf")
    ]
    todo = [c for c in combos if c not in done]
    print(f"  {len(done)} done, {len(todo)} remaining out of {len(combos)} total combos")

    for i, (holdout_ct, lpw, slt) in enumerate(combos):
        # lpw=0 makes spatial_loss_type irrelevant; run only once under 'supcon'
        if lpw == 0 and slt == "domain_clf":
            continue

        if (holdout_ct, lpw, slt) in done:
            print(f"  skip (already done): holdout={holdout_ct}, lpw={lpw}, slt={slt}")
            continue

        print(f"\n[{i+1}/{len(combos)}] holdout={holdout_ct}  lpw={lpw}  slt={slt}")
        rows = run_one(adata_base, holdout_ct, lpw, slt)
        all_rows.extend(rows)

        pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)
        print(f"  Saved {len(all_rows)} rows → {RESULTS_CSV}")

    results_df = pd.read_csv(RESULTS_CSV)
    generate_pdf(results_df, RESULTS_PDF, HOLDOUT_CELLTYPES)


if __name__ == "__main__":
    main()
