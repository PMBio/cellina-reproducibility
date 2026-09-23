"""Multi-seed train + eval for the in-vivo Perturb-FISH counterfactual experiment.

Ports the pipelines in ``notebooks/in_vivo/pfish_analysis.ipynb`` (cellina /
cellina-gat), ``notebooks/in_vivo/spprop_analysis.ipynb`` (spatialprop), and
``scripts/terra/in_vivo.ipynb`` section 3 onward (terra) into a script that
trains and evaluates ``--num_seeds`` independently-seeded models and appends one
row per (KO, type, baseline, seed) to a results CSV. Aggregation across
seeds/models is left to a separate notebook, as requested -- this script only
produces the raw per-seed rows.

cellina / cellina-gat need the ``cellina-graph`` conda env; spatialprop needs
``spatial-prop``; terra needs ``terra`` (they can't coexist in one interpreter),
so run this once per model from the matching env, e.g.:

    conda run -n cellina-graph python scripts/invivo_multi_seed.py \\
        --model cellina --num_seeds 5 \\
        --adata_path notebooks/in_vivo/pfish_xenograf_tumor_linearized.h5ad

    conda run -n cellina-graph python scripts/invivo_multi_seed.py \\
        --model cellina-gat --num_seeds 5 \\
        --adata_path notebooks/in_vivo/pfish_xenograf_tumor_linearized.h5ad

    conda run -n spatial-prop python scripts/invivo_multi_seed.py \\
        --model spatialprop --num_seeds 5 \\
        --adata_path notebooks/in_vivo/pfish_xenograf_tumor_linearized.h5ad

    conda run -n terra python scripts/invivo_multi_seed.py \\
        --model terra --num_seeds 5

Results go to ``<out_dir>/invivo_multiseed_<model>.csv`` (appended incrementally,
one seed at a time, so a crash partway through a long run doesn't lose earlier
seeds). The ``mean`` baseline is model-agnostic and data-only (same formula and
score for any model given the same seed's ``far`` sample), so it is only ever
computed/saved for ``--model cellina`` -- saving it again for cellina-gat,
spatialprop, or terra would just be duplicate rows.

Note on a discrepancy fixed during porting: pfish_analysis.ipynb's random
baseline draws two *independent* random cancer-cell samples per draw index (one
list comprehension for the Pearson-scoring lfc, a separate one for the
counterfactual matrix used in the other metrics), so "draw i" isn't the same
cells in both. spprop_analysis.ipynb draws once per index and reuses it for both
outputs, which is the intended behavior (self-consistent baseline draws) and is
what this script does for every model.

``--model terra`` assumes the fine-tuned encoder already exists (sections 0-2 of
``scripts/terra/in_vivo.ipynb`` already run: pretrained bundle downloaded, LoRA
fine-tuned, ``epoch_selection.json``/``decoder_split.json`` written). It does
NOT take ``--adata_path`` -- terra's inputs (``pfish_terra.h5ad``/
``pfish_full.h5ad``, the fine-tuned encoder bundle) are fixed, since the encoder
cache is tied to them. Per the request that drove this: only the *decoder* is
retrained per seed. Tokenization, the fine-tuned encoder's embeddings, the
synthetic KO/random signatures, and the per-condition neighbourhood embeddings
(``s`` in ``cat(z, s)``) are all computed once (reusing the on-disk caches
``common.tokenize_cached``/``common.embed_cached`` already maintain) and shared
across every seed; each seed only reruns ``terra.training.decode`` with a fresh
``--seed`` and then decodes the cached embeddings with that seed's checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_VIVO_DIR = REPO_ROOT / "notebooks" / "in_vivo"
TERRA_SCRIPTS_DIR = REPO_ROOT / "scripts" / "terra"
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

MODEL_CHOICES = ["cellina", "cellina-gat", "spatialprop", "terra"]

# --- terra-specific fixed paths (scripts/terra/in_vivo.ipynb sections 0-2's outputs) ---
PFISH_TERRA = IN_VIVO_DIR / "pfish_terra.h5ad"    # 500 genes, encoder-side panel
PFISH_FULL = IN_VIVO_DIR / "pfish_full.h5ad"      # 154 genes, comparable to the other models
TERRA_MODEL_REPO = "lotfollahi-lab/TERRA-96M"
TERRA_PRETRAINED_DIR = REPO_ROOT / "pretrained"
TERRA_PX_TO_UM = 0.108
TERRA_WORK = (IN_VIVO_DIR / "terra_pfish" / "terra").resolve()
TERRA_TOK_CACHE = TERRA_WORK.parent / "terra_tok"
TERRA_SELECTION_PATH = TERRA_WORK / "epoch_selection.json"
TERRA_DECODER_SPLIT_PATH = IN_VIVO_DIR / "terra_pfish" / "decoder_split.json"
TERRA_CF_GT_PATH = IN_VIVO_DIR / "terra_pfish" / "counterfactual_ground_truth.npz"
# fixed reference seed for everything EXCEPT decoder training (tokenization jitter, the 5 random
# cancer-cell draws for the random baseline) -- matches in_vivo.ipynb's own `SEED=0`, so these stay
# identical across every seed in the multi-seed loop, per the request that only decoder training vary.
TERRA_BASE_SEED = 0
TERRA_SEQ_LEN_CELL = 256
TERRA_CLUSTER_SPACING = 10_000.0
TERRA_JITTER = 2.0


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
                            classifier_lambda=1.0,
                            discriminator_lambda=1.0, link_prediction_weight=1.0,
                            n_layers=2, convolution_type="gat", gene_likelihood="nb")

    train_args = dict(
        max_epochs=max_epochs, batch_size=batch_size, check_val_every_n_epoch=1,
        early_stopping=True, enable_checkpointing=True, early_stopping_patience=10,
        early_stopping_monitor="vae_loss_validation",
        accelerator=("gpu" if device == "cuda" else "cpu"), devices=([1] if device == "cuda" else 1),
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
# terra: everything except decoder training, computed once and shared across
# every seed (scripts/terra/in_vivo.ipynb sections 0-2, already run, plus the
# seed-independent parts of section 4: tokenization, the fine-tuned encoder's
# embeddings, the synthetic KO/random signatures, and the per-condition `s`
# neighbourhood embeddings). Nothing here touches decoder training.
# ---------------------------------------------------------------------------

def setup_terra():
    import shutil as _shutil

    import anndata as ad
    import h5py
    import scanpy as sc
    import scipy.sparse as sp

    sys.path.insert(0, str(TERRA_SCRIPTS_DIR))
    import common as terra_common
    from terra import download_pretrained
    from terra.inference import embed_dataset, harmonize_adata, tokenize_adata

    # pfish_terra.h5ad has an /uns/log1p/base entry written with a "null" IOSpec this anndata
    # version has no reader for -- harmless scanpy log1p-base marker, not read by anything here.
    # Teach the registry to decode it as None instead of touching the file (in_vivo.ipynb cell 6).
    from anndata._io.specs.registry import _REGISTRY, IOSpec

    if (h5py.Dataset, IOSpec("null", "0.1.0")) not in _REGISTRY.read:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, _reader):
            return None

    def read_var_names(h5ad_path):
        with h5py.File(h5ad_path, "r") as f:
            idx = f["var"]["_index"][:]
        return [g.decode() if isinstance(g, bytes) else g for g in idx]

    genes_500 = read_var_names(PFISH_TERRA)
    genes_154 = read_var_names(PFISH_FULL)

    model_dir = download_pretrained(TERRA_MODEL_REPO, local_dir=str(TERRA_PRETRAINED_DIR))

    with open(TERRA_SELECTION_PATH) as f:
        sel = json.load(f)
    best_epoch = sel["latest_passing_epoch"]
    assert best_epoch is not None, "no epoch passed the collapse guard -- rerun scripts/terra/in_vivo.ipynb section 2"
    dec_enc_dir = TERRA_WORK / f"lora_ep{best_epoch}" / "lora_bundle"
    assert dec_enc_dir.exists(), f"missing {dec_enc_dir} -- rerun scripts/terra/in_vivo.ipynb section 2"

    with open(TERRA_DECODER_SPLIT_PATH) as f:
        split = json.load(f)

    # --- adata_terra + tok_all: common.tokenize_cached loads from TERRA_TOK_CACHE if it already
    # exists (it does, from section 2), so this does not retokenize -- just rebuilds the harmonized
    # adata (cheap I/O + gene mapping) to satisfy tokenize_cached's row-count sanity check ---
    raw = sc.read_h5ad(str(PFISH_TERRA))
    raw.obs_names_make_unique()
    raw.obs["cell_id"] = raw.obs_names.astype(str)
    raw.obsm["spatial"] = np.asarray(raw.obsm["spatial"], dtype=np.float64) * TERRA_PX_TO_UM
    if sp.issparse(raw.X):
        raw.X = raw.X.tocsr()
    raw.layers["counts"] = raw.X.copy()
    raw = harmonize_adata(raw, gene_mapping_dict_file_path=f"{model_dir}/ensembl_dictionary.pkl",
                           gene_occurrence_count_file_path=f"{model_dir}/gene_count_dictionary.pkl")
    adata_terra = raw
    tok_all = terra_common.tokenize_cached(adata_terra, str(model_dir), TERRA_TOK_CACHE, nproc=16)
    del adata_terra, raw

    # embed_cached loads straight from the cached npz (from section 2) if present -- no re-embedding
    emb, ids = terra_common.embed_cached(tok_all, dec_enc_dir, TERRA_WORK / f"emb_lora_ep{best_epoch}.npz")
    scemb = np.hstack([np.asarray(emb["cell_emb"]), np.asarray(emb["neighborhood_emb"])]).astype(np.float32)

    # --- 154-gene ground truth aligned to tok_all's cell_id order ---
    with h5py.File(PFISH_FULL, "r") as f:
        full_ids = [g.decode() if isinstance(g, bytes) else g for g in f["obs"]["_index"][:]]
        full_X = f["X"][:]  # (154418, 154) int32 raw counts, columns == genes_154 order
    full_pos = {c: i for i, c in enumerate(full_ids)}
    counts_154 = full_X[[full_pos[c] for c in ids]]

    pos_of = {c: i for i, c in enumerate(ids)}

    def to_idx(id_list):
        return np.array([pos_of[c] for c in id_list if c in pos_of], dtype=np.int64)

    train_idx, val_idx, test_idx = to_idx(split["train_ids"]), to_idx(split["val_ids"]), to_idx(split["test_ids"])
    print(f"[terra setup] decoder split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # --- decoder training dataset: built ONCE, reused by every seed's terra.training.decode call ---
    npz_path = TERRA_WORK / "decoder_data_multiseed.npz"
    pack = {}
    for name, idx, expr_key in (("train", train_idx, "train_expression"),
                                 ("val", val_idx, "val_expression"),
                                 ("test", test_idx, "test_expression_gt")):
        pack[f"{name}_embeddings"] = scemb[idx]
        pack[expr_key] = counts_154[idx].astype(np.float32)
        pack[f"{name}_barcodes"] = np.array([ids[i] for i in idx], dtype=object)
        pack[f"{name}_slides"] = np.full(len(idx), "pfish_terra", dtype=object)
    np.savez(npz_path, gene_list=np.array(genes_154, dtype=object), **pack)
    del pack

    # --- counterfactual ground truth, precomputed by notebooks/in_vivo/make_counterfactual_ground_truth.py ---
    gt_npz = np.load(TERRA_CF_GT_PATH, allow_pickle=True)
    assert list(gt_npz["genes_154"]) == genes_154, "counterfactual ground truth is on a different gene order"
    far_ids_all = list(gt_npz["far_ids"])
    recon_all = gt_npz["recon"]
    shared = gt_npz["shared"]
    mean_shift = gt_npz["mean_shift"]
    EFF = list(gt_npz["EFF9"])
    OBS = {ko: gt_npz[f"OBS_{ko}"] for ko in EFF}

    # --- far cells restricted to the harmonized panel; z = unperturbed cell_emb (fixed identity) ---
    keep = [c in pos_of for c in far_ids_all]
    far_ids = [c for c, k in zip(far_ids_all, keep) if k]
    recon_far = recon_all[np.array(keep)]
    p_recon = recon_far.mean(0)
    tok_far = tok_all.select([pos_of[c] for c in far_ids])
    z_emb = embed_dataset(dataset=tok_far, model_folder_path=str(dec_enc_dir),
                           **dict(terra_common.EMB_KWARGS, num_workers=4))
    z = np.asarray(z_emb["cell_emb"], dtype=np.float32)
    print(f"[terra setup] far={len(far_ids)} cells | z (unperturbed cell_emb) {z.shape}")

    # --- 500-gene lognorm KO / random signatures -- fixed TERRA_BASE_SEED, shared across every decoder seed ---
    adata_terra_full = sc.read_h5ad(str(PFISH_TERRA))
    ln500 = np.asarray(adata_terra_full.layers["lognorm"])
    obsT = adata_terra_full.obs
    is_cancer = (obsT["celltype2"] == "cancer").to_numpy()
    pert_col = obsT["perturbation"].astype(str).to_numpy()
    pc = is_cancer & (obsT["n_perturb"].to_numpy() > 0)
    n_counts500 = obsT["n_counts"].to_numpy()

    def ko_signature(ko):
        kc = pc & (pert_col == ko)
        return ln500[kc].mean(0), float(n_counts500[kc].mean())

    rng = np.random.default_rng(TERRA_BASE_SEED)
    cancer_idx = np.where(is_cancer)[0]
    random_sigs = [(ln500[draw].mean(0), float(n_counts500[draw].mean()))
                    for draw in (rng.choice(cancer_idx, 300, replace=False) for _ in range(N_RANDOM))]
    del adata_terra_full

    SIGNATURES = {ko: ko_signature(ko) for ko in EFF}
    SIGNATURES.update({f"random_{i}": random_sigs[i] for i in range(N_RANDOM)})

    # --- tokenize one synthetic 11-cell constellation per signature; keep only its own 256-token block ---
    cluster_n = 11
    rows_, coords_, obs_names_syn = [], [], []
    rng_syn = np.random.default_rng(TERRA_BASE_SEED)
    for ci, (name, (sig, depth)) in enumerate(SIGNATURES.items()):
        sig_prop = np.expm1(sig)
        sig_prop = sig_prop / (sig_prop.sum() + 1e-8)
        synth_counts = np.round(sig_prop * depth).astype(np.int32)
        center = np.array([ci * TERRA_CLUSTER_SPACING, 0.0])
        for j in range(cluster_n):
            rows_.append(synth_counts)
            coords_.append(center + rng_syn.uniform(-TERRA_JITTER, TERRA_JITTER, size=2))
            obs_names_syn.append(f"syn_{name}_{j}")

    syn = ad.AnnData(X=np.stack(rows_).astype(np.int32), var=pd.DataFrame(index=genes_500))
    syn.obs_names = obs_names_syn
    syn.obs["cell_id"] = syn.obs_names
    syn.obsm["spatial"] = np.asarray(coords_)
    syn.layers["counts"] = syn.X.copy()
    syn_h = harmonize_adata(syn, gene_mapping_dict_file_path=f"{model_dir}/ensembl_dictionary.pkl",
                             gene_occurrence_count_file_path=f"{model_dir}/gene_count_dictionary.pkl",
                             min_genes_per_cell=1, min_cells_per_gene=1)
    tmp_tok = TERRA_WORK / ".scratch_syn_tok_multiseed"
    _shutil.rmtree(tmp_tok, ignore_errors=True)
    syn_tok = tokenize_adata(syn_h, str(model_dir), str(tmp_tok), nproc=4)
    _shutil.rmtree(tmp_tok, ignore_errors=True)
    syn_raw = syn_tok.with_format(None)
    syn_ids = [str(c) for c in syn_raw["cell_id"]]

    neigh_block = {}
    for name in SIGNATURES:
        pos = syn_ids.index(f"syn_{name}_0")
        gt_tok = np.asarray(syn_raw["gene_tokens"][pos])[:TERRA_SEQ_LEN_CELL]
        ge_tok = np.asarray(syn_raw["gene_expr"][pos])[:TERRA_SEQ_LEN_CELL]
        neigh_block[name] = (gt_tok, ge_tok)

    # --- s (perturbed neighborhood_emb) per condition, computed ONCE and reused for every decoder seed ---
    def splice_neighbourhood(batch, gt_block, ge_block):
        tokens = np.asarray(batch["gene_tokens"])
        expr = np.asarray(batch["gene_expr"], dtype=np.float32)
        tokens[:, TERRA_SEQ_LEN_CELL:] = np.tile(gt_block, 10)[None, :]
        expr[:, TERRA_SEQ_LEN_CELL:] = np.tile(ge_block, 10)[None, :]
        batch["gene_tokens"], batch["gene_expr"] = tokens, expr
        return batch

    lat_by_condition = {}
    for name in SIGNATURES:
        gt_block, ge_block = neigh_block[name]
        fmt = tok_far.format
        pert_tok = tok_far.with_format(None).map(
            lambda b: splice_neighbourhood(b, gt_block, ge_block),
            batched=True, batch_size=256, keep_in_memory=True, load_from_cache_file=False)
        pert_tok.set_format(type=fmt["type"], columns=fmt["columns"], output_all_columns=fmt["output_all_columns"])
        s_emb = embed_dataset(dataset=pert_tok, model_folder_path=str(dec_enc_dir),
                               **dict(terra_common.EMB_KWARGS, num_workers=4))
        lat_by_condition[name] = np.hstack([z, np.asarray(s_emb["neighborhood_emb"], dtype=np.float32)])
        print(f"[terra setup] embedded condition '{name}' ({len(lat_by_condition)}/{len(SIGNATURES)})")

    # --- eval-side ground truth: adata_154 (is_holdout, for compute_edistance's PCA fit) + per-KO
    # real near-T ground-truth expression (NEAR_IDS_{ko}, added to the ground-truth npz for this) ---
    adata_154 = sc.read_h5ad(str(PFISH_FULL))
    adata_154.obs["is_holdout"] = adata_154.obs_names.astype(str).isin(set(split["test_ids"]))

    near_expr = {}
    for ko in EFF:
        near_ids = [str(c) for c in gt_npz[f"NEAR_IDS_{ko}"]]
        near_expr[ko] = normalize_counts(full_X[[full_pos[c] for c in near_ids]].astype(np.float64))

    return dict(
        genes_154=genes_154, best_epoch=best_epoch, npz_path=npz_path,
        EFF=EFF, OBS=OBS, shared=shared, mean_shift=mean_shift,
        p_recon=p_recon, lat_by_condition=lat_by_condition,
        adata_154=adata_154, near_expr=near_expr,
    )


def run_seed_terra(seed, setup, work_dir, device, decoder_epochs=100):
    import subprocess

    import anndata as ad
    import torch
    from terra.training.decode import apply_count_decoder

    out_dir = os.path.join(work_dir, "terra_multiseed", f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    variant = f"lora_ep{setup['best_epoch']}_seed{seed}"
    ckpt = os.path.join(out_dir, f"count_decoder_{variant}.pt")
    metrics_path = os.path.join(out_dir, f"decoder_metrics_{variant}.json")

    cmd = [sys.executable, "-m", "terra.training.decode",
           "--dataset", str(setup["npz_path"]), "--gene-selection", "all",
           "--loss-type", "nb_libsize", "--disable-slide-batching",
           "--epochs", str(decoder_epochs), "--hidden-dim", "512", "--mlp-depth", "2", "--layer-norm",
           "--early-stop-patience", "5", "--device", ("0" if device == "cuda" else "cpu"),
           "--seed", str(seed), "--output", ckpt, "--metrics-json", metrics_path]
    print(f"[terra seed={seed}]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    gene_list = list(torch.load(ckpt, map_location="cpu")["gene_list"])
    assert gene_list == setup["genes_154"], "decoder gene_list != genes_154"
    m = json.loads(Path(metrics_path).read_text())
    assert abs(m["val"]["pearson_mean"] - m["test"]["pearson_mean"]) > 1e-12, (
        "val.pearson_mean == test.pearson_mean -- suspicious, check the split")
    print(f"[terra seed={seed}] decoder val pearson {m['val']['pearson_mean']:.4f} | "
          f"test pearson (held-out perturbed-niche T cells) {m['test']['pearson_mean']:.4f}")

    def decode_condition(name):
        lat = setup["lat_by_condition"][name]
        a_ = ad.AnnData(X=np.zeros((len(lat), len(setup["genes_154"])), dtype=np.float32),
                         var=pd.DataFrame(index=pd.Index(setup["genes_154"])))
        a_.obsm["decoder_emb"] = lat
        apply_count_decoder(a_, emb_key="decoder_emb", model_folder_path=None, checkpoint_path=str(ckpt),
                             decoded_counts_layer_key="decoded", embed_fallback_key="decoder_emb",
                             device=(0 if device == "cuda" else "cpu"))
        pr = normalize_counts(np.asarray(a_.layers["decoded"]))
        return lfc(pr.mean(0), setup["p_recon"]), pr

    REAL, COUNTERFACTUALS = {}, {}
    for ko in setup["EFF"]:
        REAL[ko], COUNTERFACTUALS[ko] = decode_condition(ko)

    rand_lfcs, cf_random = [], []
    for i in range(N_RANDOM):
        r_lfc, r_cf = decode_condition(f"random_{i}")
        rand_lfcs.append(r_lfc)
        cf_random.append(r_cf)

    gt = dict(OBS=setup["OBS"], shared=setup["shared"], mean_shift=setup["mean_shift"], NEAR_EXPR=setup["near_expr"])
    baseline_prefix = f"terra-lora-ep{setup['best_epoch']}"
    rows = baseline_rows(setup["adata_154"], gt, REAL, COUNTERFACTUALS, baseline_name=baseline_prefix)
    rows += random_baseline_rows(setup["adata_154"], gt, rand_lfcs, cf_random, baseline_name=f"{baseline_prefix}-random")
    for r in rows:
        r["seed"] = seed
        r["model"] = "terra"
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODEL_CHOICES)
    p.add_argument("--num_seeds", type=int, required=True)
    p.add_argument("--adata_path", default=None,
                    help="required for cellina/cellina-gat/spatialprop; unused (and must be omitted) "
                         "for terra, whose inputs are fixed paths tied to its fine-tuned encoder")
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--out_dir", default=str(REPO_ROOT / "results"))
    p.add_argument("--work_dir", default=str(IN_VIVO_DIR),
                    help="where per-seed trained models / intermediate files are stashed")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true",
                    help="rerun seeds that already have rows in the output CSV (default: skip them)")
    args = p.parse_args()
    if args.model == "terra":
        if args.adata_path is not None:
            p.error("--adata_path is not used by --model terra (its inputs are fixed)")
    elif args.adata_path is None:
        p.error(f"--adata_path is required for --model {args.model}")
    return args


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

    terra_setup = setup_terra() if args.model == "terra" and any(s not in done_seeds for s in seeds) else None

    for seed in seeds:
        if seed in done_seeds:
            print(f"seed={seed} already in {csv_path} -- skipping (pass --overwrite to rerun)")
            continue

        if args.model in ("cellina", "cellina-gat"):
            variant = "base" if args.model == "cellina" else "GAT"
            rows = run_seed_cellina(args.adata_path, seed, variant, args.work_dir, device, save_mean)
        elif args.model == "spatialprop":
            rows = run_seed_spatialprop(args.adata_path, seed, args.work_dir, device)
        else:
            rows = run_seed_terra(seed, terra_setup, args.work_dir, device)

        df = pd.DataFrame(rows)
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", header=write_header, index=False)
        print(f"seed={seed}: appended {len(df)} rows -> {csv_path}")

    print("done.")


if __name__ == "__main__":
    main()
