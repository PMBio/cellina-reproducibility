"""
Ablation study: sweep one lambda parameter while holding all others at 1e-7.

Hyperparameters (batch size, lr, n_layers, etc.) are taken directly from
scripts/configs/cellina_config.py and scripts/configs/cellina_graph_config.py.
Only the ablation-specific lambda keys are overridden.

Usage
-----
    conda run -n cellina      python run_ablations.py --ablation clf
    conda run -n cellina      python run_ablations.py --ablation disc
    conda run -n cellina      python run_ablations.py --ablation domain_clf
    conda run -n cellina_edge python run_ablations.py --ablation graph

Output
------
results/ablation_{clf,disc,domain_clf,graph}.csv
  columns: lambda, seed, metric, score
  metric:  F1_celltype | F1_spatial_domain | marginal_ll
"""
import argparse
import copy
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def log(msg, **kwargs):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True, **kwargs)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Support both repo_root/ablations/ and repo_root/notebooks/ablations/
_parent      = os.path.join(SCRIPT_DIR, "..", "scripts")
_grandparent = os.path.join(SCRIPT_DIR, "..", "..", "scripts")
SCRIPTS_DIR  = _parent if os.path.isdir(_parent) else _grandparent
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

from utils import set_seed
from perturb_utils import load_crc_slide
from configs.cellina_config import (
    MODEL_ARGS as CELLINA_MODEL_ARGS,
    TRAIN_ARGS  as CELLINA_TRAIN_ARGS,
    PLAN_KWARGS as CELLINA_PLAN_KWARGS,
)
from configs.cellina_graph_config import (
    MODEL_ARGS as GRAPH_MODEL_ARGS,
    TRAIN_ARGS  as GRAPH_TRAIN_ARGS,
    PLAN_KWARGS as GRAPH_PLAN_KWARGS,
)

# ── Config ────────────────────────────────────────────────────────────────────

SLIDE_ID    = 242
_repo_root  = os.path.dirname(os.path.abspath(SCRIPTS_DIR))
DATA_DIR    = os.path.join(_repo_root, "data", "crc_wt_cosmx")
LABELS_KEY  = "coarse_type"
DOMAINS_KEY = "typ_clean"
BATCH_KEY   = "sid"

LAMBDA_RANGE = [0, 1e-9, 1e-7, 1e-5, 1e-3, 0.1, 1, 10, 100]
SEEDS        = list(range(5))
N_MC_SAMPLES = 500

BASE_PATH   = SCRIPT_DIR
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    from cellina._spatial_utils import spatial_neighbors, compute_spatial_features

    set_seed(0)

    adata = load_crc_slide(
        slide_id=SLIDE_ID,
        data_dir=DATA_DIR,
        n_top_genes=3000,
        labels_key=LABELS_KEY,
        domains_key=DOMAINS_KEY,
    )

    spatial_neighbors(adata, bandwidth=100 / 0.12028, max_neighbours=200, standardize=False)
    compute_spatial_features(adata)

    # Random 10% holdout
    n_cells  = adata.n_obs
    test_idx = np.random.choice(n_cells, int(n_cells * 0.1), replace=False)
    adata.obs["is_holdout"] = False
    adata.obs.iloc[test_idx, adata.obs.columns.get_loc("is_holdout")] = True

    trainval_idx = np.setdiff1d(np.arange(n_cells), test_idx)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.1, random_state=0, shuffle=True
    )

    return adata, train_idx, val_idx, test_idx


# ── Evaluation helpers ────────────────────────────────────────────────────────

def compute_f1(adata, model, target_col, batch_size):
    z = model.get_latent_representation(latent_key="z", batch_size=batch_size)
    adata.obsm["_z_tmp"] = z

    mask_train = ~adata.obs["is_holdout"]
    mask_test  =  adata.obs["is_holdout"]

    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(adata[mask_train].obsm["_z_tmp"], adata[mask_train].obs[target_col].values)
    y_pred = clf.predict(adata[mask_test].obsm["_z_tmp"])
    return f1_score(adata[mask_test].obs[target_col].values, y_pred, average="macro")


def compute_mll(adata, model):
    return model.get_marginal_ll(
        adata=adata[adata.obs["is_holdout"]],
        n_mc_samples=N_MC_SAMPLES,
        return_mean=True,
    )


# ── Resume helper ─────────────────────────────────────────────────────────────

def load_done(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return set(zip(df["lambda"], df["seed"]))
    return set()


# ── Per-ablation runner ───────────────────────────────────────────────────────

def run_ablation(ablation, adata, train_idx, val_idx, test_idx):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"ablation_{ablation}.csv")
    done = load_done(csv_path)

    if ablation == "graph":
        from cellina_graph import CellinaModel
        CellinaModel.setup_anndata(
            adata,
            batch_key=BATCH_KEY,
            labels_key=LABELS_KEY,
            domains_key=DOMAINS_KEY,
            layer="counts",
            spatial_connectivities_key="spatial_connectivities",
        )
        model_args_base = copy.copy(GRAPH_MODEL_ARGS)
        train_args      = GRAPH_TRAIN_ARGS
        plan_kwargs     = GRAPH_PLAN_KWARGS
    else:
        from cellina import CellinaModel
        CellinaModel.setup_anndata(
            adata,
            batch_key=BATCH_KEY,
            labels_key=LABELS_KEY,
            domains_key=DOMAINS_KEY,
            spatial_obsm_key="spatial_x",
            layer="counts",
        )
        model_args_base = copy.copy(CELLINA_MODEL_ARGS)
        train_args      = CELLINA_TRAIN_ARGS
        plan_kwargs     = CELLINA_PLAN_KWARGS

    batch_size = train_args["batch_size"]
    total = len(LAMBDA_RANGE) * len(SEEDS)
    n = 0

    for lambda_ in LAMBDA_RANGE:
        for seed in SEEDS:
            n += 1
            if (lambda_, seed) in done:
                log(f"[{n}/{total}] skip (done): ablation={ablation} lambda={lambda_} seed={seed}")
                continue

            log(f"[{n}/{total}] ablation={ablation} lambda={lambda_} seed={seed}")
            set_seed(seed)

            model_args = copy.copy(model_args_base)

            if ablation == "clf":
                model_args["classifier_lambda"]   = lambda_
                model_args["discriminator_lambda"] = 1e-7
                model_args["domain_classifier_lambda"] = 1e-7
            elif ablation == "disc":
                model_args["classifier_lambda"]   = 1e-7
                model_args["discriminator_lambda"] = lambda_
                model_args["domain_classifier_lambda"] = 1e-7
            elif ablation == "domain_clf":
                model_args["classifier_lambda"]        = 1e-7
                model_args["discriminator_lambda"]     = 1e-7
                model_args["domain_classifier_lambda"] = lambda_
            elif ablation == "graph":
                model_args["classifier_lambda"]      = 1e-7
                model_args["discriminator_lambda"]   = 1e-7
                model_args["link_prediction_weight"] = lambda_

            model = CellinaModel(adata, **model_args)
            model.train(
                **train_args,
                plan_kwargs=plan_kwargs,
                datasplitter_kwargs={"external_indexing": [train_idx, val_idx, test_idx]},
            )

            save_path = os.path.join(BASE_PATH, "trained", f"{ablation}_{lambda_}_seed_{seed}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path, overwrite=True)

            f1_ct = compute_f1(adata, model, LABELS_KEY,  batch_size)
            f1_sd = compute_f1(adata, model, DOMAINS_KEY, batch_size)
            mll   = compute_mll(adata, model)

            new_rows = [
                {"lambda": lambda_, "seed": seed, "metric": "F1_celltype",       "score": f1_ct},
                {"lambda": lambda_, "seed": seed, "metric": "F1_spatial_domain",  "score": f1_sd},
                {"lambda": lambda_, "seed": seed, "metric": "marginal_ll",         "score": mll},
            ]

            if os.path.exists(csv_path):
                existing = pd.read_csv(csv_path)
                pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True).to_csv(csv_path, index=False)
            else:
                pd.DataFrame(new_rows).to_csv(csv_path, index=False)

            done.add((lambda_, seed))
            log(f"  → F1_ct={f1_ct:.4f}  F1_sd={f1_sd:.4f}  mll={mll:.4f}")

    log(f"Done. Results saved to {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run one lambda-sweep ablation.")
    parser.add_argument(
        "--ablation",
        choices=["clf", "disc", "domain_clf", "graph"],
        required=True,
        help="Which parameter to ablate.",
    )
    args = parser.parse_args()

    log(f"Loading data (CRC slide {SLIDE_ID})...")
    adata, train_idx, val_idx, test_idx = load_data()
    log(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    run_ablation(args.ablation, adata, train_idx, val_idx, test_idx)


if __name__ == "__main__":
    main()
