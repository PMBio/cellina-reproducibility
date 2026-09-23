#!/usr/bin/env python
"""Cellina (node perturbation) scored with the terra_og readout, on TERRA's data and TERRA's shift.

Apples-to-apples row for the supplementary shift table (decisions 2026-09-18, Q7-Q12):
  * same DATA   -- cellina is trained by `scripts/train_loo.py` (unchanged, paper hyper-parameters)
                   on `{sid}_terra2k.h5ad`, i.e. TERRA's own 2000-gene axis, one LOO model per
                   (slide, held-out cell type) exactly like the benchmark;
  * same SHIFT  -- the 50 genes and the global natural-log logFC terra_og perturbed
                   (`results/terra_og/cache/perturbed_genes_{sid}.csv`), added in log space to EVERY
                   neighbour cell via `cellina.make_neighbor_perturbation(add_shift=True, base=e,
                   renormalize=True)` -- cellina's native node perturbation, so this is the analogue
                   of terra_og's `neighb_only` target (the focal cell's own input is untouched);
  * same CELLS  -- the REF cells of the scored type inside terra_og's control population
                   (`{sid}/terra_og_cache/control`, dumped once with --dump-control-ids);
  * same TRUTH / UNIVERSE / METRIC -- `results/terra_og/cache/logfc_{sid}.csv[ct]`, the gene list and
                   `is_perturbed` flag of the terra-frozen neighb_only per-gene CSV, and
                   `score_universe` (precision@50 + chance + spearman) copied verbatim from
                   terra_og_eval.py (terra is not importable in the cellina env).
The per-gene statistic standing in for TERRA's per-gene W2 is |log2 FC| of the mean CP10K expression
between cellina's counterfactual and cellina's own reconstruction of the same cells (the model's
internal shift; `abs_log2fc_model`).  The benchmark's convention, counterfactual vs the OBSERVED
control mean (`abs_log2fc_obs`), is stored alongside.  The random-gene control reuses terra_og's
prevalence-matched partner sets (`random_genes_{sid}_seed{s}.csv`) with the same doses.
Diagnostic 2026-09-18 (crc_232 Fibroblast): two reconstruction draws of the SAME cells already give
P@50 ~0.2 -- |log2FC| ranks lowly expressed (noisy) genes first, and those are the high-|logFC| genes.
So every run also stores (i) `universes_null`: draw-vs-draw |log2FC| scored like a shift, and
(ii) `universes_z`: the shift z-scored per gene by the sd of log2FC across N_DRAWS reconstruction
draws (floor Z_SD_FLOOR), which removes the expression-level confound.

Output (terra_og per-run schema, so terra_summary_tables.shift_rows picks it up):
  results/terra_og/per_run/{sid}_cellina-pert_neighb_only[_random]_{ct}.json + _per_gene.csv

    # terra env, once per slide (needs the `datasets` package):
    python scripts/terra/cellina_on_terra_universe.py --sid crc_232 --dump-control-ids
    # cellina111 env, after train_loo.py trained cellina-W for the fold:
    python scripts/terra/cellina_on_terra_universe.py --sid crc_232 --cell-type Fibroblast --random-seeds 0,1,2,3,4
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from counterfactual_analysis import _normalize_counts, precision, safe_log2_fold_change  # noqa: E402

DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "data"))
RESULTS = REPO / "results" / "terra_og"
MODEL_NAME = "cellina-W"          # train_loo's folder name for the paper's cellina arm
ARM = "cellina-pert"              # the benchmark's name for cellina + node perturbation
TARGET = "neighb_only"
K = 50                            # terra_og_eval.K
BATCH_SIZE = 512                  # notebooks/loo_benchmarks/cellina_node_pert.ipynb
LIBRARY_SIZE = "latent"
COUNTS_PER_K = 1e4
N_DRAWS = 5                       # reconstruction draws for the sampling null / z statistic
Z_SD_FLOOR = 1e-3                 # log2 units


def score_universe(truth, pred, k=K):
    """Verbatim copy of terra_og_eval.score_universe."""
    m = np.isfinite(pred) & np.isfinite(truth)
    t, p = truth[m], pred[m]
    return dict(n_genes=int(m.sum()), chance=k / max(int(m.sum()), 1),
                precision=float(precision(t, p, k=k)),
                spearman=float(spearmanr(p, t).statistic))


def _msd(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def _git_sha():
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def control_ids_csv(sid):
    return RESULTS / "cache" / f"control_cells_{sid}.csv"


def dump_control_ids(sid):
    """terra env: cell ids + cell type of terra_og's control population -> cache CSV."""
    from datasets import load_from_disk
    d = load_from_disk(str(DATA_ROOT / "datasets" / "crc" / sid / "terra_og_cache" / "control"))
    ids = np.array(d.with_format(None)["cell_id"]).astype(str)
    out = control_ids_csv(sid)
    pd.DataFrame({"cell_id": ids}).to_csv(out, index=False)
    print(f"wrote {out}: {len(ids):,} control cells")


def terra2k_path(sid):
    return DATA_ROOT / "datasets" / "crc" / "raw_zenodo_terra2k" / f"{sid}_terra2k.h5ad"


def load_fold(sid, ct):
    """Exactly train_loo.main's preprocessing for the cellina class, so the saved model fits."""
    from train_loo import _load_model, preprocess_crc, preprocess_spatial_features, split_indices
    from configs.adata_crc_config import ADATA_ARGS as A
    from utils import set_seed

    set_seed(0)
    adata = sc.read(terra2k_path(sid))
    adata = preprocess_crc(adata, n_top_genes=A["n_top_genes"], labels_key=A["labels_key"], domains_key=A["domains_key"])
    tr, va, te = split_indices(adata, ct, labels_key=A["labels_key"], domains_key=A["domains_key"],
                               holdout_domains=A["holdout_domains"], seed=0)
    adata = preprocess_spatial_features(adata, step_size_px=0.12028, n_neighbors=A["n_neighbors"], test_indices=te)
    sid2k = terra2k_path(sid).stem                       # train_loo derives the sid from the h5ad stem
    save_dir = DATA_ROOT / "data" / "ood" / "trained" / sid2k / ct / MODEL_NAME
    assert save_dir.exists(), f"{save_dir} missing: run train_loo.py for this fold first"
    model, _ = _load_model(str(save_dir), model_class="cellina", adata=adata, splits=(tr, va, te))
    # the notebook's cellina branch: X = log1p(CP10K) for the perturbation and inference
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=COUNTS_PER_K)
    sc.pp.log1p(adata)
    return adata, model, A, str(save_dir)


def perturb_and_decode(adata, model, logfc, labels_key, idx):
    """Global log-space shift of `logfc` (Series over genes) on all neighbour cells -> decoded CP10K-scale
    counterfactual of the cells `idx`."""
    from cellina import make_neighbor_perturbation
    assert logfc.index.isin(adata.var_names).all(), "perturbed gene missing from the terra2k axis"
    pert = {c: logfc for c in adata.obs[labels_key].astype(str).unique()}   # every neighbour type, same shift
    make_neighbor_perturbation(adata, perturbations=pert, groupby=labels_key, obsm_key_out="spatial_x_cf",
                               base=np.e, renormalize=True, add_shift=True)
    return np.asarray(model.get_perturbed_expression(adata=adata, indices=idx, spatial_obsm_key="spatial_x_cf",
                                                     batch_size=BATCH_SIZE, library_size=LIBRARY_SIZE))


def abs_log2fc(a, b):
    """|log2 FC| of mean CP10K expression, a vs b (cells x genes)."""
    return np.abs(safe_log2_fold_change(_normalize_counts(a).mean(0), _normalize_counts(b).mean(0)))


def mean_cp10k(a):
    return _normalize_counts(a).mean(0)


def zscore_shift(cf, draws):
    """|log2FC(cf, mean draw)| / sd over draws of log2FC(draw_i, mean draw), per gene."""
    m = np.stack([mean_cp10k(d) for d in draws])
    mu = m.mean(0)
    sd = np.std(safe_log2_fold_change(m, mu[None, :]), axis=0, ddof=1)
    return np.abs(safe_log2_fold_change(mean_cp10k(cf), mu)) / np.maximum(sd, Z_SD_FLOOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True)
    ap.add_argument("--cell-type")
    ap.add_argument("--random-seeds", default="", help="e.g. 0,1,2,3,4: terra_og's random-gene partner sets")
    ap.add_argument("--dump-control-ids", action="store_true", help="terra env: write the control cell ids cache")
    ap.add_argument("--results", default=str(RESULTS))
    a = ap.parse_args()
    if a.dump_control_ids:
        dump_control_ids(a.sid)
        return
    assert a.cell_type, "--cell-type required"
    t0 = time.time()
    ct, sid = a.cell_type, a.sid
    results = Path(a.results)
    per_run, cache = results / "per_run", RESULTS / "cache"
    per_run.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.random_seeds.split(",") if s != ""]

    adata, model, A, save_dir = load_fold(sid, ct)
    labels_key, domains_key = A["labels_key"], A["domains_key"]

    # ---- same cells: REF cells of the type inside terra_og's control population
    ctrl_ids = set(pd.read_csv(control_ids_csv(sid))["cell_id"].astype(str))
    is_ref = adata.obs[domains_key].astype(str).isin(A["control_domains"]).to_numpy()
    is_ct = (adata.obs[labels_key].astype(str) == ct).to_numpy()
    in_pop = adata.obs_names.astype(str).isin(ctrl_ids)
    n_ref_ct = int((is_ref & is_ct).sum())
    idx = np.flatnonzero(is_ref & is_ct & in_pop)
    print(f"{sid} {ct}: REF cells of type {n_ref_ct:,}; in terra_og control population {len(idx):,}")
    assert len(idx) >= 20

    # ---- same truth and universe (from the terra-frozen neighb_only run)
    logfc = pd.read_csv(cache / f"logfc_{sid}.csv", index_col=0)
    pg = pd.read_csv(RESULTS / "per_run" / f"{sid}_terra-frozen_{TARGET}_{ct}_per_gene.csv")
    genes = pg["gene"].to_numpy()
    assert pd.Index(genes).isin(adata.var_names).all()
    truth = logfc.loc[genes, ct].abs().to_numpy()
    is_pert_terra = pg["is_perturbed"].to_numpy(bool)

    # ---- same shift
    gene_df = pd.read_csv(cache / f"perturbed_genes_{sid}.csv")
    shift = gene_df.set_index("gene")["logfc_global"]
    pert_set = set(gene_df["gene"])
    is_pert = np.array([g in pert_set for g in genes])
    assert (is_pert == is_pert_terra).all(), "perturbed-gene flag differs from the TERRA run"

    counts = adata.layers["counts"][idx]
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    draws = [np.asarray(model.get_normalized_expression(adata=adata, indices=idx, batch_size=BATCH_SIZE,
                                                        library_size=LIBRARY_SIZE)) for _ in range(N_DRAWS)]
    recon = draws[0]
    gi = adata.var_names.get_indexer(genes)
    uni_of = lambda edited: {"all": np.ones(len(genes), bool), "minus_perturbed": ~edited}

    def score(cf, edited):
        pred_model = abs_log2fc(cf, recon)[gi]
        pred_obs = abs_log2fc(cf, counts)[gi]
        pred_z = zscore_shift(cf, draws)[gi]
        uni = uni_of(edited)
        res = {u: score_universe(truth[m], pred_model[m]) for u, m in uni.items()}
        res_obs = {u: score_universe(truth[m], pred_obs[m]) for u, m in uni.items()}
        res_z = {u: score_universe(truth[m], pred_z[m]) for u, m in uni.items()}
        return res, res_obs, res_z, pred_model, pred_obs, pred_z

    # sampling null: a second reconstruction draw of the same cells, scored as if it were a shift
    pred_null = abs_log2fc(draws[1], recon)[gi]
    res_null = {u: score_universe(truth[m], pred_null[m]) for u, m in uni_of(is_pert).items()}
    print(f"{ct}: sampling null P@50 {res_null['all']['precision']:.3f}/{res_null['minus_perturbed']['precision']:.3f}")

    cf = perturb_and_decode(adata, model, shift, labels_key, idx)
    res, res_obs, res_z, pred_model, pred_obs, pred_z = score(cf, is_pert)
    print(f"{ct}: n={len(idx):,} | |A|={res['all']['n_genes']} P@50={res['all']['precision']:.3f} "
          f"rho={res['all']['spearman']:.3f} | |B|={res['minus_perturbed']['n_genes']} "
          f"P@50={res['minus_perturbed']['precision']:.3f} rho={res['minus_perturbed']['spearman']:.3f} "
          f"| chance {res['all']['chance']:.3f}/{res['minus_perturbed']['chance']:.3f} "
          f"| vs observed control: P@50 {res_obs['all']['precision']:.3f}/{res_obs['minus_perturbed']['precision']:.3f} "
          f"| z-scored: P@50 {res_z['all']['precision']:.3f}/{res_z['minus_perturbed']['precision']:.3f}")

    rand = {}
    for s in seeds:
        rdf = pd.read_csv(cache / f"random_genes_{sid}_seed{s}.csv")
        rshift = pd.Series(np.log(rdf["foldchange"].to_numpy()), index=rdf["gene"])   # = the partner's inherited logfc_global
        redited = np.array([g in set(rdf["gene"]) for g in genes])
        rcf = perturb_and_decode(adata, model, rshift, labels_key, idx)
        rres, rres_obs, rres_z, _, _, _ = score(rcf, redited)
        rand[s] = {"universes": rres, "universes_vs_observed": rres_obs, "universes_z": rres_z}
        print(f"  random seed {s}: P@50 {rres['all']['precision']:.3f}/{rres['minus_perturbed']['precision']:.3f} "
              f"| z {rres_z['all']['precision']:.3f}/{rres_z['minus_perturbed']['precision']:.3f}")

    base = {
        "sid": sid, "arm": ARM, "epoch": None, "perturbation_target": ["neighborhood"],
        "target_label": TARGET, "coarse_type": ct, "n_cells_scored": int(len(idx)),
        "n_cells_control_population": int(len(ctrl_ids)), "n_ref_cells_of_type": n_ref_ct,
        "mechanism": "cellina.make_neighbor_perturbation(add_shift=True, base=e, renormalize=True)",
        "shift_source": "logfc_global (natural log) added to log1p(CP10K) of every neighbour cell",
        "statistic": "abs_log2fc(mean CP10K counterfactual, mean CP10K reconstruction)",
        "n_perturbed_genes": int(len(shift)), "n_perturbed_in_universe": int(is_pert.sum()),
        "k": K, "min_cells": None, "blur": None,
        "universes": res, "universes_vs_observed": res_obs,
        "universes_z": res_z, "universes_null": res_null, "n_draws": N_DRAWS, "z_sd_floor": Z_SD_FLOOR,
        "population_w2": None,
        "model_dir": save_dir, "train_data": str(terra2k_path(sid)),
        "git_sha": _git_sha(), "wall_seconds": time.time() - t0,
    }
    out = {**base, "random_control": None}
    per_run.joinpath(f"{sid}_{ARM}_{TARGET}_{ct}.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame({"gene": genes, "abs_log2fc_model": pred_model, "abs_log2fc_obs": pred_obs,
                  "abs_log2fc_null": pred_null, "z_model": pred_z,
                  "abs_logfc_ct": truth, "is_perturbed": is_pert}).to_csv(
        per_run / f"{sid}_{ARM}_{TARGET}_{ct}_per_gene.csv", index=False)
    if seeds:
        rc = {
            "n_sets": len(seeds), "seeds": seeds,
            "universes": {u: {f"{k}_{stat}": v for k in ("precision", "spearman")
                              for stat, v in _msd([rand[s]["universes"][u][k] for s in seeds]).items()}
                          for u in ("all", "minus_perturbed")},
            "universes_z": {u: {f"{k}_{stat}": v for k in ("precision", "spearman")
                                for stat, v in _msd([rand[s]["universes_z"][u][k] for s in seeds]).items()}
                            for u in ("all", "minus_perturbed")},
            "population_w2": None,
            "per_seed": {str(s): rand[s] for s in seeds},
        }
        out = {**base, "random_control": rc, "wall_seconds": time.time() - t0}
        per_run.joinpath(f"{sid}_{ARM}_{TARGET}_random_{ct}.json").write_text(json.dumps(out, indent=1))
    print(f"done {sid} {ct} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
