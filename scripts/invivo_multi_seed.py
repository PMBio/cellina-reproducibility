"""Multi-seed train + eval for the in-vivo Perturb-FISH counterfactual experiment.

Ports the pipelines in ``notebooks/in_vivo/pfish_analysis.ipynb`` (cellina /
cellina-gat) and ``notebooks/in_vivo/spprop_analysis.ipynb`` (spatialprop) into a
script that trains and evaluates ``--num_seeds`` independently-seeded models and
appends one row per (KO, type, baseline, seed) to a results CSV. Aggregation
across seeds/models is left to a separate notebook, as requested -- this script
only produces the raw per-seed rows.

cellina / cellina-gat need the ``cellina-graph`` conda env; spatialprop needs
``spatial-prop`` (they can't coexist in one interpreter), so run this once per
model from the matching env, e.g.:

    conda run -n cellina-graph python scripts/invivo_multi_seed.py \\
        --model cellina --num_seeds 5 \\
        --adata_path notebooks/in_vivo/pfish_xenograf_tumor_linearized.h5ad

    conda run -n cellina-graph python scripts/invivo_multi_seed.py \\
        --model cellina-gat --num_seeds 5 \\
        --adata_path notebooks/in_vivo/pfish_xenograf_tumor_linearized.h5ad

    conda run -n spatial-prop python scripts/invivo_multi_seed.py \\
        --model spatialprop --num_seeds 5 \\
        --adata_path notebooks/in_vivo/pfish_xenograf_tumor_linearized.h5ad

Results go to ``<out_dir>/invivo_multiseed_<model>.csv`` (appended incrementally,
one seed at a time, so a crash partway through a long run doesn't lose earlier
seeds). The ``mean`` baseline is model-agnostic and data-only (same formula and
score for any model given the same seed's ``far`` sample), so it is only ever
computed/saved for ``--model cellina`` -- saving it again for cellina-gat or
spatialprop would just be duplicate rows.

Note on a discrepancy fixed during porting: pfish_analysis.ipynb's random
baseline draws two *independent* random cancer-cell samples per draw index (one
list comprehension for the Pearson-scoring lfc, a separate one for the
counterfactual matrix used in the other metrics), so "draw i" isn't the same
cells in both. spprop_analysis.ipynb draws once per index and reuses it for both
outputs, which is the intended behavior (self-consistent baseline draws) and is
what this script does for every model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_VIVO_DIR = REPO_ROOT / "notebooks" / "in_vivo"
sys.path.insert(0, str(IN_VIVO_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# 9 KOs with a real direct effect on their own cancer cell (ko_inventory.csv
# selection, pfish_analysis.ipynb §4.1) -- hardcoded identically in both source
# notebooks (`EFF9`), since both ultimately use `EFF = EFF9` unfiltered.
EFF9 = ["MAP2K2", "IRAK1", "IRF7", "PELI1", "IRF3", "IRF5", "NFKBIA", "LBP", "MAP2K3"]
N_RANDOM = 5
N_DEG = 10
COUNTS_PER_K = 1e4
FAR_CAP = 6000

MODEL_CHOICES = ["cellina", "cellina-gat", "spatialprop"]


# ---------------------------------------------------------------------------
# shared, model-agnostic math (identical in both source notebooks)
# ---------------------------------------------------------------------------

def lfc(a, b):
    return np.log(a + 1) - np.log(b + 1)


def normalize_counts(x, eps=1e-8, scale=COUNTS_PER_K):
    return x / (x.sum(axis=1, keepdims=True) + eps) * scale


def pseudobulk(x):
    """Row-normalize each cell to 1e4 counts, then average across cells."""
    x = np.asarray(x, float)
    return (x / (x.sum(1, keepdims=True) + 1e-8) * COUNTS_PER_K).mean(0)


def match_pool(nc, a, pool):
    """Depth-match indices `a` to their nearest neighbour in `pool` by log1p(n_counts)."""
    from scipy.spatial import cKDTree

    def cd(idx):
        return np.column_stack([np.log1p(nc[idx])])

    Cc = cd(pool)
    mu, sd = Cc.mean(0), Cc.std(0) + 1e-9
    _, mi = cKDTree((Cc - mu) / sd).query((cd(a) - mu) / sd, k=1)
    return pool[mi]


def compute_data_ground_truth(adata, counts, obs, conn, far):
    """Purely data-derived quantities shared by every model: per-KO OBS shift, the
    shared cross-KO confound, the KO-agnostic mean_shift, and each KO's real
    near-T ground-truth expression (for E-distance). Identical across
    cellina / cellina-gat / spatialprop for the same (adata, far).
    """
    pert = obs["perturbation"].astype(str).to_numpy()
    is_cancer = (obs["celltype2"] == "cancer").to_numpy()
    is_T = (obs["celltype2"] == "T cells").to_numpy()
    pc = is_cancer & (obs["n_perturb"].to_numpy() > 0)
    uT = is_T & (obs["n_perturb"].to_numpy() == 0)
    nc = obs["n_counts"].to_numpy()

    OBS, NEAR_EXPR, kc_by_ko = {}, {}, {}
    for ko in EFF9:
        kc = pc & (pert == ko)
        near = np.asarray(conn.dot(kc.astype(float))).ravel() > 0
        noth = np.asarray(conn.dot((pc & (pert != ko)).astype(float))).ravel() > 0
        t = np.where(uT & near & ~noth)[0]  # exclusive near-KO-T ground truth
        OBS[ko] = lfc(pseudobulk(counts[t]), pseudobulk(counts[match_pool(nc, t, far)]))
        NEAR_EXPR[ko] = normalize_counts(adata.X[t])
        kc_by_ko[ko] = kc
    shared = np.mean([OBS[k] for k in EFF9], axis=0)
    mean_shift = lfc(
        pseudobulk(counts[np.where(uT & (np.asarray(conn.dot(pc.astype(float))).ravel() > 0))[0]]),
        pseudobulk(counts[far]),
    )
    return dict(pert=pert, is_cancer=is_cancer, pc=pc, uT=uT, nc=nc, kc_by_ko=kc_by_ko,
                OBS=OBS, NEAR_EXPR=NEAR_EXPR, shared=shared, mean_shift=mean_shift)


# ---------------------------------------------------------------------------
# shared eval/metrics (identical `_metrics_row` logic in both notebooks)
# ---------------------------------------------------------------------------

def metrics_row(adata, lfc_pred, lfc_observed, predicted_expr, near_expr_k):
    from scipy.stats import pearsonr
    from counterfactual_analysis import compute_edistance, direction_match, compute_mse_lfc

    deg = np.argsort(-np.abs(lfc_observed))[:N_DEG]  # rank by the target actually being scored
    pear, _ = pearsonr(lfc_observed[deg], lfc_pred[deg])
    dir_match_k = direction_match(lfc_observed[deg], lfc_pred[deg], k=N_DEG, normalize="k")
    mse_lfc = compute_mse_lfc(gt_vec=lfc_observed, cf_vec=lfc_pred, deg=deg)
    rmse_lfc = np.sqrt(mse_lfc)
    edist = compute_edistance(adata, observed=near_expr_k, predicted=predicted_expr, deg=None,
                               library_size=COUNTS_PER_K, local=True, use_pca=True)
    return {"Pearson": pear, "Precision": dir_match_k, "E-distance": edist, "RMSE_LFC": rmse_lfc}


def baseline_rows(adata, gt, REAL, COUNTERFACTUALS, baseline_name):
    rows = []
    for k in EFF9:
        rows.append({"KO": k, "type": "full", "baseline": baseline_name,
                     **metrics_row(adata, REAL[k], gt["OBS"][k], COUNTERFACTUALS[k], gt["NEAR_EXPR"][k])})
        rows.append({"KO": k, "type": "KO-specific", "baseline": baseline_name,
                     **metrics_row(adata, REAL[k], gt["OBS"][k] - gt["shared"], COUNTERFACTUALS[k], gt["NEAR_EXPR"][k])})
    return rows


def random_baseline_rows(adata, gt, rand_lfcs, cf_random, baseline_name):
    rows = []
    for k in EFF9:
        full_draws = pd.DataFrame([metrics_row(adata, rand_lfcs[i], gt["OBS"][k], cf_random[i], gt["NEAR_EXPR"][k])
                                    for i in range(len(rand_lfcs))])
        ko_draws = pd.DataFrame([metrics_row(adata, rand_lfcs[i], gt["OBS"][k] - gt["shared"], cf_random[i], gt["NEAR_EXPR"][k])
                                  for i in range(len(rand_lfcs))])
        rows.append({"KO": k, "type": "full", "baseline": baseline_name, **full_draws.mean().to_dict()})
        rows.append({"KO": k, "type": "KO-specific", "baseline": baseline_name, **ko_draws.mean().to_dict()})
    return rows


def mean_baseline_rows(adata, gt, recon):
    """Model-agnostic, data-only baseline. Caller decides whether to save these rows."""
    cf_mean = np.clip((recon + 1) * np.exp(gt["mean_shift"])[None, :] - 1, 0, None)
    rows = []
    for k in EFF9:
        rows.append({"KO": k, "type": "full", "baseline": "mean",
                     **metrics_row(adata, gt["mean_shift"], gt["OBS"][k], cf_mean, gt["NEAR_EXPR"][k])})
        rows.append({"KO": k, "type": "KO-specific", "baseline": "mean",
                     **metrics_row(adata, gt["mean_shift"], gt["OBS"][k] - gt["shared"], cf_mean, gt["NEAR_EXPR"][k])})
    return rows


def pick_device(pref):
    import torch
    if pref == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if pref == "cuda":
        return "cuda"
    try:
        _ = (torch.zeros(1, device="cuda") + 1).cpu()
        return "cuda"
    except Exception as e:
        print("GPU present but unusable:", str(e).splitlines()[0][:70], "-> using CPU")
        return "cpu"


# ---------------------------------------------------------------------------
# cellina / cellina-gat: train + eval for one seed
# (notebooks/in_vivo/pfish_analysis.ipynb, sections 1/2/4)
# ---------------------------------------------------------------------------

def run_seed_cellina(adata_path, seed, variant, work_dir, device, save_mean,
                      max_epochs=100, batch_size=256, n_latent=64):
    import torch
    import pfish_prep as pp
    from cellina import Cellina, CellinaGCN
    from scipy.sparse import issparse, csr_matrix
    from scvi.train._callbacks import EarlyStopping, SaveCheckpoint
    from sklearn.model_selection import train_test_split
    from utils import set_seed

    set_seed(seed)
    bw, mn = pp.DEFAULT_BANDWIDTH, pp.DEFAULT_MAX_NEIGHBOURS

    adata = pp.load_pfish(adata_path)
    adata = pp.filter_cells(adata, max_n_perturb=1)
    adata.obs_names_make_unique()
    pp.build_graph(adata, bandwidth=bw, max_neighbours=mn)
    pp.label_context(adata)
    idx_control, idx_target, donor_idx = pp.make_edge_swap_sets(adata, responder="T cells")
    pp.build_features(adata, bandwidth=bw, max_neighbours=mn, test_indices=idx_target)

    # compute_edistance's PCA fit excludes holdout cells via this column
    adata.obs["is_holdout"] = False
    adata.obs.iloc[idx_target, adata.obs.columns.get_loc("is_holdout")] = True

    if not issparse(adata.layers["counts"]):
        adata.layers["counts"] = csr_matrix(adata.layers["counts"])
    else:
        adata.layers["counts"] = adata.layers["counts"].tocsr()

    out_dir = os.path.join(work_dir, f"cellina_{variant}_multiseed", f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    trainval = np.setdiff1d(np.arange(adata.n_obs), idx_target)
    train_idx, val_idx = train_test_split(trainval, test_size=0.1, random_state=seed, shuffle=True)
    print(f"[cellina-{variant} seed={seed}] train={len(train_idx)} val={len(val_idx)} held-out={len(idx_target)}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if variant == "base":
        Cellina.setup_anndata(adata, labels_key="celltype2", domains_key="context",
                               batch_key=None, spatial_obsm_key="spatial_x", layer="counts")
        model = Cellina(adata, n_latent=n_latent, use_observed_lib_size=True,
                         classifier_lambda=1.0, discriminator_lambda=1.0,
                         gene_likelihood="nb", n_layers=2)
    else:
        CellinaGCN.setup_anndata(adata, batch_key=None, labels_key="celltype2",
                                  domains_key="context", layer="counts",
                                  spatial_connectivities_key="spatial_connectivities")
        model = CellinaGCN(adata, n_latent=n_latent, use_observed_lib_size=True,
                            condition_on_intrinsic=False, classifier_lambda=1.0,
                            discriminator_lambda=1.0, link_prediction_weight=1.0,
                            n_layers=2, convolution_type="gat", gene_likelihood="nb")

    train_args = dict(
        max_epochs=max_epochs, batch_size=batch_size, check_val_every_n_epoch=1,
        early_stopping=True, enable_checkpointing=True, early_stopping_patience=10,
        early_stopping_monitor="vae_loss_validation",
        accelerator=("gpu" if device == "cuda" else "cpu"), devices=([0] if device == "cuda" else 1),
        datasplitter_kwargs={"external_indexing": [train_idx, val_idx, idx_target]},
        callbacks=[
            SaveCheckpoint(monitor="vae_loss_validation", dirpath=os.path.join(out_dir, "ckpt"),
                            load_best_on_end=True),
            EarlyStopping(monitor="vae_loss_validation", patience=10, mode="min"),
        ],
    )
    model.train(**train_args, plan_kwargs={"lr": 1e-3, "normalize_losses": True})
    model.save(os.path.join(out_dir, "model"), overwrite=True)
    model.to_device(device)

    # --- counterfactual eval (pfish_analysis.ipynb §4) ---
    obs = adata.obs
    _counts_layer = adata.layers["counts"]
    counts = np.asarray(_counts_layer.todense() if hasattr(_counts_layer, "todense") else _counts_layer, float)
    ln = np.asarray(adata.layers["lognorm"])
    conn = adata.obsp[pp.CONN_ORIG_KEY].tocsr()
    sx = adata.obsm["spatial_x"]
    sx = np.asarray(sx.todense() if hasattr(sx, "todense") else sx, float)
    nc = obs["n_counts"].to_numpy()
    is_cancer = (obs["celltype2"] == "cancer").to_numpy()

    rng = np.random.default_rng(seed)
    far = idx_control if len(idx_control) <= FAR_CAP else rng.choice(idx_control, FAR_CAP, replace=False)

    recon = normalize_counts(adata.X[far])
    p_recon = normalize_counts(recon).mean(0)

    def insert(sig, cf_layer_key="counts_cf_ko"):
        if variant == "base":
            sxd = sx.copy()
            sxd[far] = sig[None, :]
            adata.obsm["spatial_x_cf"] = sxd
            pr = np.asarray(model.get_perturbed_expression(adata=adata, indices=far, batch_size=2048,
                                                             spatial_obsm_key="spatial_x_cf"), float)
        else:
            sig_prop = np.expm1(sig)
            sig_prop = sig_prop / (sig_prop.sum() + 1e-8)
            cf = (sig_prop[None, :] * nc[:, None]).astype(np.float32)
            adata.layers[cf_layer_key] = csr_matrix(cf)
            pr = np.asarray(model.get_perturbed_expression(adata=adata, indices=far, batch_size=2048,
                                                             cf_layer=cf_layer_key), float)
        pr = normalize_counts(pr)
        return lfc(normalize_counts(pr).mean(0), p_recon), pr

    gt = compute_data_ground_truth(adata, counts, obs, conn, far)

    REAL, COUNTERFACTUALS = {}, {}
    for ko in EFF9:
        REAL[ko], COUNTERFACTUALS[ko] = insert(ln[gt["kc_by_ko"][ko]].mean(0))

    rand_lfcs, cf_random = [], []
    for _ in range(N_RANDOM):
        draw = rng.choice(np.where(is_cancer)[0], 300, replace=False)
        r_lfc, r_cf = insert(ln[draw].mean(0))
        rand_lfcs.append(r_lfc)
        cf_random.append(r_cf)

    model_name = "cellina" if variant == "base" else "cellina-gat"
    baseline_prefix = f"Cellina-{variant}"
    rows = baseline_rows(adata, gt, REAL, COUNTERFACTUALS, baseline_name=baseline_prefix)
    rows += random_baseline_rows(adata, gt, rand_lfcs, cf_random, baseline_name=f"{baseline_prefix}-random")
    if save_mean:
        rows += mean_baseline_rows(adata, gt, recon)
    for r in rows:
        r["seed"] = seed
        r["model"] = model_name
    return rows


# ---------------------------------------------------------------------------
# spatialprop: train + eval for one seed
# (notebooks/in_vivo/spprop_analysis.ipynb)
# ---------------------------------------------------------------------------

def run_seed_spatialprop(adata_path, seed, work_dir, device,
                          max_epochs=100, k_hop=2, augment_hop=2,
                          num_cells_per_ct_id=100, learning_rate=1e-3, loss="weightedl1"):
    import torch
    import scanpy as sc
    import pfish_graph_utils as pg
    from spatial_gnn.api.perturbation_api import train_perturbation_model
    from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
    from spatial_gnn.utils.dataset_utils import create_dataloader_from_dataset
    from spatial_gnn.models.inference import predict
    from utils import set_seed

    set_seed(seed)
    rng = np.random.default_rng(seed)
    bw, mn = pg.DEFAULT_BANDWIDTH, pg.DEFAULT_MAX_NEIGHBOURS

    out_dir = os.path.join(work_dir, "spatialprop_multiseed", f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    exp_name = f"pfish_spatialprop_seed{seed}"

    adata = pg.load_pfish(adata_path)
    adata = pg.filter_cells(adata, max_n_perturb=1)
    adata.obs_names_make_unique()
    pg.build_graph(adata, bandwidth=bw, max_neighbours=mn)
    pg.label_context(adata)
    idx_control, idx_target, donor_idx = pg.make_edge_swap_sets(adata, responder="T cells")
    obs = adata.obs

    # compute_edistance's PCA fit excludes holdout cells via this column
    adata.obs["is_holdout"] = False
    adata.obs.iloc[idx_target, adata.obs.columns.get_loc("is_holdout")] = True

    far = idx_control if len(idx_control) <= FAR_CAP else rng.choice(idx_control, FAR_CAP, replace=False)
    print(f"[spatialprop seed={seed}] far T cells: {len(far):,}")

    adata_sp = adata.copy()
    adata_sp.obs["celltype"] = adata_sp.obs["celltype2"].astype(str)
    adata_sp.obs["mouse_id"] = "pfish"
    adata_sp.obs["region"] = adata_sp.obs["context"].astype(str)
    adata_sp.X = adata_sp.layers["counts"].copy()
    sc.pp.normalize_total(adata_sp, target_sum=1e4)
    sc.pp.log1p(adata_sp)
    adata_sp.obs["is_holdout"] = False
    adata_sp.obs.iloc[idx_target, adata_sp.obs.columns.get_loc("is_holdout")] = True

    train_path = os.path.join(out_dir, "adata_train.h5ad")
    test_path = os.path.join(out_dir, "adata_test.h5ad")
    pert_path = os.path.join(out_dir, "adata_test_perturbed.h5ad")  # overwritten per condition
    adata_sp[~adata_sp.obs["is_holdout"]].copy().write_h5ad(train_path)
    adata_sp.write_h5ad(test_path)

    _cwd0 = os.getcwd()
    try:
        os.chdir(out_dir)
        torch.manual_seed(seed)
        training_args = dict(
            dataset=exp_name, file_path=os.path.abspath(os.path.join(_cwd0, train_path)),
            train_ids=["pfish"], test_ids=["pfish"], exp_name=exp_name, k_hop=k_hop,
            augment_hop=augment_hop, center_celltypes="all", node_feature="expression",
            inject_feature="none", learning_rate=learning_rate, loss=loss, epochs=max_epochs,
            normalize_total=True, num_cells_per_ct_id=num_cells_per_ct_id, predict_celltype=False,
            pool="center", do_eval=False, device=device,
        )
        _, gene_names, (model, model_config, trained_model_path) = train_perturbation_model(**training_args)
        print(f"[spatialprop seed={seed}] saved model -> {trained_model_path}")
    finally:
        os.chdir(_cwd0)

    celltypes_to_index = model_config["celltypes_to_index"]

    def build_dataset(tag, file_path, center_celltypes, n_cells, use_perturbed=False, overwrite=False):
        return SpatialAgingCellDataset(
            root=out_dir, subfolder_name=None, dataset_prefix=f"{exp_name}_{tag}", target="expression",
            k_hop=k_hop, augment_hop=0, node_feature="expression", inject_feature=None,
            num_cells_per_ct_id=n_cells, center_celltypes=center_celltypes, whole_tissue=False,
            use_ids=["pfish"], raw_filepaths=[os.path.abspath(file_path)],
            celltypes_to_index=celltypes_to_index, normalize_total=True,
            perturbation_mask_key="perturbed_input", use_perturbed_expression=use_perturbed,
            overwrite=overwrite,
        )

    def loader_from(ds, n=1):
        ds.process()
        loaders = []
        for _ in range(n):
            _, loader = create_dataloader_from_dataset(ds, batch_size=1024, shuffle=False,
                                                         num_workers=4, pin_memory=(device == "cuda"),
                                                         persistent_workers=True)
            loaders.append(loader)
        return loaders[0] if n == 1 else loaders

    unpert_ds = build_dataset("unpert_T", test_path, ["T cells"], 100_000)
    unpert_loader, unpert_loader_dummy = loader_from(unpert_ds, n=2)
    res0 = predict(model=model, adata=sc.read_h5ad(test_path), dataloader=unpert_loader,
                    perturbed_dataloader=unpert_loader_dummy, use_ids=["pfish"], device=device)
    pred_unpert = np.asarray(res0.layers["predicted_unperturbed"])
    n_pred = np.isfinite(pred_unpert).all(axis=1).sum()
    print(f"[spatialprop seed={seed}] T cells with a prediction: {n_pred} / {(obs['celltype2']=='T cells').sum()}")

    n_genes = adata_sp.X.shape[1]
    ln = np.asarray(adata_sp.X)
    far_row_scale = ln[far].sum(1)

    def to_expr(x, row_scale):
        x_hat = x * (row_scale[:, None] / n_genes)
        return np.clip(np.expm1(x_hat), 0, None)

    pert = obs["perturbation"].astype(str).to_numpy()
    is_cancer = (obs["celltype2"] == "cancer").to_numpy()
    baseline_cancer = ln[is_cancer & (pert == "Control")].mean(0)

    def additive_perturb(cell_mask, delta):
        X = ln.copy()
        X[cell_mask] = X[cell_mask] + delta[None, :]
        X = np.clip(X, 0, None)
        row_sums = X.sum(1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return X / row_sums * n_genes

    def run_condition(tag, delta, overwrite):
        adata_pert = sc.read_h5ad(test_path)
        adata_pert.obsm["perturbed_input"] = additive_perturb(is_cancer, delta)
        adata_pert.write_h5ad(pert_path)
        pert_ds = build_dataset(f"pert_{tag}", pert_path, ["T cells"], 100_000,
                                 use_perturbed=True, overwrite=overwrite)
        pert_loader = loader_from(pert_ds)
        res = predict(model=model, adata=sc.read_h5ad(test_path), dataloader=unpert_loader,
                       perturbed_dataloader=pert_loader, use_ids=["pfish"], device=device)
        return np.asarray(res.layers["predicted_perturbed"])[far]

    def insert(tag, delta, overwrite=False):
        pred_far = run_condition(tag, delta, overwrite)
        control = normalize_counts(adata.X[far])
        cf = normalize_counts(to_expr(pred_far, far_row_scale))
        return lfc(pseudobulk(cf), pseudobulk(control)), cf

    conn = adata.obsp[pg.CONN_ORIG_KEY].tocsr()
    counts = np.asarray(adata.layers["counts"])
    gt = compute_data_ground_truth(adata, counts, obs, conn, far)

    REAL, COUNTERFACTUALS = {}, {}
    for ko in EFF9:
        kc = is_cancer & (pert == ko)
        delta = ln[kc].mean(0) - baseline_cancer
        REAL[ko], COUNTERFACTUALS[ko] = insert(f"real_{ko}", delta)
        print(f"[spatialprop seed={seed}]  real[{ko}] done")

    rand_lfcs, cf_random = [], []
    for i in range(N_RANDOM):
        idx = rng.choice(np.where(is_cancer)[0], 300, replace=False)
        delta = ln[idx].mean(0) - baseline_cancer
        r_lfc, r_cf = insert(f"random_{i}", delta)
        rand_lfcs.append(r_lfc)
        cf_random.append(r_cf)
        print(f"[spatialprop seed={seed}]  random[{i}] done")

    rows = baseline_rows(adata, gt, REAL, COUNTERFACTUALS, baseline_name="SpatialProp")
    rows += random_baseline_rows(adata, gt, rand_lfcs, cf_random, baseline_name="SpatialProp-random")
    for r in rows:
        r["seed"] = seed
        r["model"] = "spatialprop"
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODEL_CHOICES)
    p.add_argument("--num_seeds", type=int, required=True)
    p.add_argument("--adata_path", required=True)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--out_dir", default=str(REPO_ROOT / "results"))
    p.add_argument("--work_dir", default=str(IN_VIVO_DIR),
                    help="where per-seed trained models / intermediate files are stashed")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true",
                    help="rerun seeds that already have rows in the output CSV (default: skip them)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"invivo_multiseed_{args.model}.csv")

    done_seeds = set()
    if os.path.exists(csv_path) and not args.overwrite:
        done_seeds = set(pd.read_csv(csv_path)["seed"].unique().tolist())

    device = pick_device(args.device)
    print(f"model={args.model} device={device} adata_path={args.adata_path}")

    save_mean = args.model == "cellina"
    seeds = range(args.seed_start, args.seed_start + args.num_seeds)

    for seed in seeds:
        if seed in done_seeds:
            print(f"seed={seed} already in {csv_path} -- skipping (pass --overwrite to rerun)")
            continue

        if args.model in ("cellina", "cellina-gat"):
            variant = "base" if args.model == "cellina" else "GAT"
            rows = run_seed_cellina(args.adata_path, seed, variant, args.work_dir, device, save_mean)
        else:
            rows = run_seed_spatialprop(args.adata_path, seed, args.work_dir, device)

        df = pd.DataFrame(rows)
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", header=write_header, index=False)
        print(f"seed={seed}: appended {len(df)} rows -> {csv_path}")

    print("done.")


if __name__ == "__main__":
    main()
