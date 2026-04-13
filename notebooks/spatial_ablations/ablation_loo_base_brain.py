"""
Ablation study: LOO cell-type × domain_classifier_lambda for CellinaBase (no GCN).

MERFISH mouse brain version — analogous to ablation_loo_base.py but uses
load_merfish_brain() and brain-region domains instead of CRC/REF/TVA.
Results are written to a separate CSV/PDF so CRC results are not overwritten.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import numpy as np
import pandas as pd
from cellina import CellinaModel as CellinaBase
from cellina._spatial_utils import spatial_neighbors, compute_spatial_features
from perturb_utils import load_merfish_brain, split_indices
from spatial_ablation_utils import (
    TOP_N, MIN_CELLS, BATCH_SIZE, LIBRARY_SIZE, DEVICES,
    CF_METRICS, RUN_METRICS,
    evaluate_model, generate_pdf,
)

import scvi
from utils import set_seed
scvi.settings.seed = 0
set_seed(0)

# ── Config ────────────────────────────────────────────────────────────────────
LABELS_KEY        = "cell_type"
DOMAINS_KEY       = "major_brain_region"
DONOR_REGION      = "Isocortex"
TARGET_REGIONS    = ["Thalamus", "Hippocampus"]
HOLDOUT_CELLTYPES = ["glutamatergic neuron", "GABAergic neuron", "astrocyte"]
LINK_PREDICTION_WEIGHTS = [0, 1e-9, 1e-7, 1e-5, 1e-3, 0.1, 0.01, 1., 10, 100]
MAX_EPOCHS  = 50
RESULTS_CSV = "results/ablation_loo_brain.csv"
RESULTS_PDF = "results/ablation_loo_base_brain.pdf"

_SLT = "cellina-base"


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
        holdout_domains=tuple(TARGET_REGIONS),
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

    print("  Training complete. Running evaluation...")
    return evaluate_model(
        model, adata, holdout_celltype,
        spatial_loss_type=_SLT,
        link_prediction_weight=lpw,
        ref_domain=DONOR_REGION,
        target_domains=TARGET_REGIONS,
        domains_key=DOMAINS_KEY,
        labels_key=LABELS_KEY,
        marginal_ll_kwargs={"return_mean": True},
        cf_extra_kwargs={"n_neighbours": 50},
        batch_size_eval_factor=1,
    )


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(RESULTS_CSV)), exist_ok=True)

    print("Loading MERFISH brain data and computing spatial graph...")
    adata_base = load_merfish_brain(labels_key=LABELS_KEY, domains_key=DOMAINS_KEY)
    spatial_neighbors(adata_base, bandwidth=100, max_neighbours=50, standardize=False)
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

    combos = [(hct, lpw) for hct in HOLDOUT_CELLTYPES for lpw in LINK_PREDICTION_WEIGHTS]
    todo = [c for c in combos if c not in done]
    print(f"  {len(done)} done, {len(todo)} remaining out of {len(combos)} total combos")

    for i, (holdout_ct, lpw) in enumerate(combos):
        if (holdout_ct, lpw) in done:
            print(f"  skip (already done): holdout={holdout_ct}, lpw={lpw}")
            continue

        print(f"\n[{i+1}/{len(combos)}] holdout={holdout_ct}  lpw={lpw}")
        rows = run_one(adata_base, holdout_ct, lpw)
        all_rows.extend(rows)

        pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)
        print(f"  Saved {len(all_rows)} rows → {RESULTS_CSV}")

    results_df = pd.read_csv(RESULTS_CSV)
    generate_pdf(results_df, RESULTS_PDF, HOLDOUT_CELLTYPES)


if __name__ == "__main__":
    main()
