#!/usr/bin/env python
"""Single-run worker for the neighbor-graph sensitivity experiment.

Trains one `Cellina` model on a uniform-kNN spatial graph (Gaussian kernel with
``bandwidth=inf`` -> every kept edge has weight 1, so the only graph knob left is
``k = max_neighbours``), then runs the tutorial's *edge-perturbation* counterfactual
on the held-out CRC Myeloid population and records the Pearson r between observed
and predicted logFC over the top DE genes.

One run == one (k, seed) pair. Pin the GPU with CUDA_VISIBLE_DEVICES; the process
always uses local device index 0. Results are written to
``{outdir}/k{k}_seed{seed}.json``.

Mirrors docs/tutorial.ipynb step-for-step; the only deliberate deviations are:
  * bandwidth = inf (uniform kNN, distance weighting off)
  * max_neighbours = k (swept)                       [training graph]
  * n_neighbours  = k in the edge counterfactual     [rewire size tracks k]
  * within-domain graph: library_key=domains_key builds a separate kNN tree
    per domain (e.g. 210_CRC / 210_REF / 210_TVA), so NO cross-domain edges
    exist. Domains have boundary contact, so a plain kNN would otherwise link
    cells across tissue boundaries. Applied to BOTH the training-feature graph
    and the counterfactual donor-pool graph (train/inference stay consistent),
    which makes the edge-perturbation donor pool CRC-only by construction.
  * seed varies model init + counterfactual sampling; the data split is fixed
    (random_state=0) so only the graph/seed change.
"""
import argparse
import json
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, required=True,
                   help="max_neighbours (neighborhood size) for graph construction.")
    p.add_argument("--seed", type=int, required=True,
                   help="Seed for model init + counterfactual neighbour sampling.")
    p.add_argument("--holdout-ct", type=str, default="Myeloid",
                   help="Cell type (coarse_type) held out in the tumour domain: it "
                        "defines the model test split, is masked from the training "
                        "spatial graph (test_indices), and is excluded from the "
                        "edge-perturbation donor pool.")
    p.add_argument("--outdir", type=str, required=True,
                   help="Directory to write the k{k}_seed{seed}.json result.")
    p.add_argument("--data", type=str,
                   default="/data/ddimitrov/repos/cellina-reproducibility/data/crc_wt_cosmx/crc_210.h5ad",
                   help="Path to the CRC h5ad.")
    p.add_argument("--control-domain", type=str, default="210_REF",
                   help="Control (healthy) domain label in obs['typ'].")
    p.add_argument("--target-domain", type=str, default="210_CRC",
                   help="Target (tumour) domain label in obs['typ']. Any further "
                        "domains on the slide (e.g. 210_TVA) stay in training as "
                        "extra domain classes but are excluded from evaluation.")
    p.add_argument("--ckpt-root", type=str, required=True,
                   help="Root dir for per-run training checkpoints.")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lean-cf", action="store_true",
                   help="Memory-lean edge-perturbation counterfactual. Numerically "
                        "identical to the library path (verified in test_lean_cf.py) "
                        "but never materialises the rewired full graph, which needs "
                        ">446GB at k=10000 on slide 210 and gets OOM-killed.")
    p.add_argument("--n-deg", type=int, default=50,
                   help="Number of top DE genes over which Pearson r is computed.")
    p.add_argument("--pixel-size", type=float, default=0.12028,
                   help="Micron/pixel scale (only used for logging bandwidth-in-microns).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# logFC helpers (verbatim from docs/tutorial.ipynb)
# ---------------------------------------------------------------------------
def _normalize_counts(x, eps=1e-8, scale=1e4):
    import numpy as np
    return x / (x.sum(axis=1, keepdims=True) + eps) * scale


def safe_log2_fold_change(a, b, eps=1e-6):
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    return np.log2((a + eps) / (b + eps))


def lean_counterfactual_expression(model, adata, indices_basal, indices_cf,
                                  n_neighbours, seed, batch_size):
    """Memory-lean stand-in for ``model.get_counterfactual_expression(precomputed=False)``.

    The library path rewires the *whole* connectivity graph -- drop every edge
    incident on a basal cell, add bidirectional basal<->donor edges, recompute
    spatial features for all cells -- then keeps only the basal rows. At k=10000
    on slide 210 that is a ~6e9-edge COO round-trip with filtered and
    concatenated copies: 446GB peak, OOM-killed even alone on a 503GB box.

    ``compute_spatial_features`` is a row-normalised ``C @ X``, and every basal
    row ends up with exactly ``n_neighbours`` distinct donor edges of weight 1
    (its own edges were all stripped), so a basal feature vector is just the
    mean expression of its donors. The donor->basal half-edges and the surviving
    original edges only touch rows that get discarded. So we build only the
    (n_basal x n_obs) block.

    The RNG loop mirrors the library's exactly, so the donors -- and therefore
    the result -- are identical for a given seed. See test_lean_cf.py, which
    asserts this against the real library function.
    """
    import numpy as np
    import scipy.sparse as sp

    indices_basal = np.asarray(indices_basal)
    indices_cf = np.asarray(indices_cf)
    n_cf = len(indices_cf)
    if n_neighbours is None or n_neighbours >= n_cf:
        raise ValueError(
            f"n_neighbours must be a finite value < n_cf ({n_cf}); got {n_neighbours}.")

    rng = np.random.default_rng(seed)
    chosen = np.concatenate([
        rng.choice(indices_cf, size=n_neighbours, replace=False) for _ in indices_basal
    ])
    rows = np.repeat(np.arange(len(indices_basal)), n_neighbours)
    C = sp.csr_matrix((np.ones(len(chosen), dtype=np.float32), (rows, chosen)),
                      shape=(len(indices_basal), adata.n_obs))

    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    feats = (C @ X) / float(n_neighbours)   # degree is exactly n_neighbours per row

    adata_cf = adata[indices_basal].copy()
    adata_cf.obsm["spatial_x"] = sp.csr_matrix(feats).astype(np.float32)
    return model.get_normalized_expression(
        adata=adata_cf, indices=None, batch_size=batch_size,
        library_size="latent", return_numpy=True)


def get_lfc(control, target, counterfactual, normalize_counts=True, n_deg=200):
    import numpy as np
    if normalize_counts:
        control = _normalize_counts(control)
        target = _normalize_counts(target)
        counterfactual = _normalize_counts(counterfactual)

    mean_control = np.nanmean(control, axis=0)
    mean_target = np.nanmean(target, axis=0)
    mean_cf = np.nanmean(counterfactual, axis=0)

    gt_vec = safe_log2_fold_change(mean_target, mean_control)
    cf_vec = safe_log2_fold_change(mean_cf, mean_control)

    deg_scores = np.abs(gt_vec)
    top_features = np.argsort(-deg_scores)[:n_deg]
    return gt_vec, cf_vec, top_features


def main():
    args = parse_args()
    t0 = time.time()

    import numpy as np
    import scanpy as sc
    import torch
    from sklearn.model_selection import train_test_split
    from scipy.stats import pearsonr
    from scvi.train._callbacks import SaveCheckpoint, EarlyStopping

    from cellina import Cellina
    from cellina._spatial_utils import spatial_neighbors, compute_spatial_features

    os.makedirs(args.outdir, exist_ok=True)
    run_tag = f"k{args.k}_seed{args.seed}"
    ckpt_dir = os.path.join(args.ckpt_root, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- annotation keys (tutorial) -------------------------------------
    labels_key = "coarse_type"
    domains_key = "typ"
    batch_key = None
    holdout_ct = args.holdout_ct
    control_domain = args.control_domain
    target_domain = args.target_domain

    # ---- 1. load + preprocess (tutorial) --------------------------------
    adata = sc.read(args.data)
    adata.obs_names_make_unique()

    label_to_coarse = {
        "epi1": "Epithelial", "epi2": "Epithelial", "epi3": "Epithelial", "epi4": "Epithelial",
        "fib1": "Fibroblast", "fib2": "Fibroblast",
        "EC": "Endothelial", "SMC": "Smooth_muscle",
        "BC": "B_cell", "PC_IgA": "Plasma_cell", "PC_IgG": "Plasma_cell", "PC_IgM": "Plasma_cell",
        "TC": "T_cell", "mye1": "Myeloid", "mye2": "Myeloid", "mast": "Mast_cell",
    }
    adata.obs["coarse_type"] = adata.obs["ist"].map(label_to_coarse)

    adata = adata[~adata.obs[domains_key].isna()]
    adata = adata[~adata.obs[labels_key].isna()]
    sc.pp.filter_cells(adata, min_counts=3)
    sc.pp.filter_genes(adata, min_counts=3)

    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, layer="counts", flavor="seurat_v3",
                                n_top_genes=2000, subset=True)

    # ---- data splits (FIXED across seeds: only graph/seed vary) ---------
    # exact-match domains: a third domain (e.g. 210_TVA) is neither control
    # nor target -- it stays in training but out of evaluation.
    is_tumor_region = adata.obs[domains_key] == target_domain
    is_holdout_ct = adata.obs[labels_key] == holdout_ct
    test_mask = is_tumor_region & is_holdout_ct
    test_idx = np.where(test_mask)[0]

    all_idx = np.arange(adata.n_obs)
    trainval_idx = np.setdiff1d(all_idx, test_idx)

    adata.obs["is_holdout"] = False
    adata.obs.iloc[test_idx, adata.obs.columns.get_loc("is_holdout")] = True

    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.1, random_state=0, shuffle=True)

    # ---- spatial features -----------------------------------------------
    # bandwidth = inf  ->  Gaussian kernel gives every kept edge weight 1
    #                      => uniform kNN graph, distance weighting OFF.
    # max_neighbours = k  is the single swept graph-construction knob.
    # library_key = domains_key  ->  per-domain kNN tree: each cell's k nearest
    #                      neighbours are drawn ONLY from its own domain, so no
    #                      cross-domain (CRC<->REF) edges are created. The two
    #                      domains overlap spatially, so this drops the spurious
    #                      boundary edges a plain global kNN would introduce.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    bandwidth = float("inf")
    adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values

    # full graph (used to define the counterfactual donor pool) — within-domain
    adata.obsp["spatial_connectivities_orig"] = spatial_neighbors(
        adata, bandwidth=bandwidth, max_neighbours=args.k,
        standardize=False, library_key=domains_key, inplace=False)

    # test-masked graph -> spatial features (avoids leakage of held-out cells)
    spatial_neighbors(adata, bandwidth=bandwidth, max_neighbours=args.k,
                      standardize=False, library_key=domains_key,
                      test_indices=test_idx)
    compute_spatial_features(adata)

    adata.X = adata.layers["counts"].copy()  # reset to raw counts for cellina

    # ---- reproducibility -------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- 2. train (tutorial hyperparameters) ----------------------------
    Cellina.setup_anndata(
        adata, batch_key=batch_key, labels_key=labels_key, domains_key=domains_key,
        spatial_obsm_key="spatial_x", layer="counts")

    cellina_args = {
        "n_latent": 64,
        "use_observed_lib_size": True,
        "condition_on_intrinsic": False,
        "classifier_lambda": 1.0,
        "discriminator_lambda": 1.0,
        "gene_likelihood": "nb",
        "n_layers": 2,
    }
    model = Cellina(adata, **cellina_args)

    train_args = {
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "check_val_every_n_epoch": 1,
        "early_stopping": True,
        "enable_checkpointing": True,
        "early_stopping_patience": 10,
        "early_stopping_monitor": "vae_loss_validation",
        "devices": [0],  # local index; GPU chosen via CUDA_VISIBLE_DEVICES
        "datasplitter_kwargs": {"external_indexing": [train_idx, val_idx, test_idx]},
        "callbacks": [
            SaveCheckpoint(monitor="vae_loss_validation", dirpath=ckpt_dir,
                           load_best_on_end=True),
            EarlyStopping(monitor="vae_loss_validation", patience=10, mode="min"),
        ],
    }
    plan_kwargs = {"lr": 1e-3, "normalize_losses": True}
    model.train(**train_args, plan_kwargs=plan_kwargs)

    # reload best checkpoint (load_best_on_end already restores weights, but be explicit)
    ckpt_name = os.listdir(ckpt_dir)[0]
    model = Cellina.load(os.path.join(ckpt_dir, ckpt_name), adata=adata)

    # ---- 4.1 edge-perturbation counterfactual ---------------------------
    is_control_region = adata.obs[domains_key] == control_domain
    mask_control = is_control_region & is_holdout_ct
    idx_control = np.where(mask_control.values)[0]

    mask_target = is_tumor_region & is_holdout_ct
    idx_target = np.where(mask_target.values)[0]

    conn = adata.obsp["spatial_connectivities_orig"]
    sub_conn = conn[idx_target]
    neighbor_indices = np.unique(sub_conn.nonzero()[1])
    neighbor_indices = neighbor_indices[~is_holdout_ct.values[neighbor_indices]]

    # within-domain sanity: the donor pool (neighbours of CRC Myeloids) must be
    # entirely in the tumour (CRC) domain — per-domain kNN guarantees this.
    donor_in_crc = int(is_tumor_region.values[neighbor_indices].sum())
    n_donor = int(len(neighbor_indices))
    assert donor_in_crc == n_donor, (
        f"donor pool leaked {n_donor - donor_in_crc}/{n_donor} non-CRC cells "
        f"despite within-domain graph (library_key={domains_key!r})")

    if args.lean_cf:
        counterfactual_counts = lean_counterfactual_expression(
            model, adata, idx_control, neighbor_indices,
            n_neighbours=args.k, seed=args.seed, batch_size=args.batch_size)
    else:
        counterfactual_counts = model.get_counterfactual_expression(
            indices=idx_control,
            batch_size=args.batch_size,
            seed=args.seed,
            neighbour_indices=neighbor_indices,
            precomputed=False,
            n_neighbours=args.k,          # rewire size tracks the swept k
        )

    control = np.array(adata.layers["counts"][mask_control.values, :].todense())
    target = np.array(adata.layers["counts"][mask_target.values, :].todense())

    true_lfc, pred_lfc, deg = get_lfc(
        control=control, target=target, counterfactual=counterfactual_counts,
        n_deg=args.n_deg)
    pearson, _ = pearsonr(true_lfc[deg], pred_lfc[deg])

    result = {
        "k": args.k,
        "seed": args.seed,
        "holdout_ct": holdout_ct,
        "data": args.data,
        "control_domain": control_domain,
        "target_domain": target_domain,
        "pearson": float(pearson),
        "n_deg": args.n_deg,
        "bandwidth": "inf",
        "within_domain": True,
        "library_key": domains_key,
        "n_neighbours_cf": args.k,
        "lean_cf": bool(args.lean_cf),
        "n_control": int(len(idx_control)),
        "n_target": int(len(idx_target)),
        "n_donor_pool": n_donor,
        "n_donor_in_crc": donor_in_crc,
        "best_checkpoint": ckpt_name,
        "runtime_sec": round(time.time() - t0, 1),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    out_path = os.path.join(args.outdir, f"{run_tag}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{run_tag}] pearson={pearson:.4f}  ->  {out_path}  "
          f"({result['runtime_sec']}s)")


if __name__ == "__main__":
    main()
