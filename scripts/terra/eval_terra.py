"""TERRA-native scoring: paper protocol B, gene-embedding W2 shift (no decoder).

PIPELINE_SPEC.md "`eval_terra.py` CLI" + notebooks/loo_benchmarks/terra/GENE_SHIFT_SPEC.md.
Copied from notebook cell 28 (`score_gene_shift`) with three deliberate changes:
universe (ii) is the benchmark's HVGs (`adata_hvg.var_names`, NOT a recomputed HVG set),
--min-cells defaults to 20 and --k to 100 with 10 random control sets (TERRA paper protocol).

Reads inference.py's caches (`ctrl_tok_{hd}`, `pert_tok_{hd}`, `logfc_{hd}.csv`) and
rebuilds them the same way (notebook cells 23-25) when they are absent.  --eval-celltypes
and --universe terra2k add `_{ct}` / `_terra2k` suffixes to those names (see cache_suffix).

    python scripts/terra/eval_terra.py --dataset_name crc --adata_path ... \
        --holdout_celltype Fibroblast --variant frozen
    python scripts/terra/eval_terra.py --dataset_name crc --adata_path ... \
        --holdout_celltype Fibroblast --variant lora --epoch 5
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # before torch; embed_dataset hardcodes cuda:0

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from datasets import load_from_disk
from geomloss import SamplesLoss
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                   # noqa: E402  (also puts scripts/ on sys.path)
from counterfactual_analysis import (_normalize_counts, get_lfc, precision,   # noqa: E402
                                     safe_log2_fold_change)
from terra.inference import embed_dataset       # noqa: E402

SEED = 0
UNI_HVG = "(ii) n benchmark HVGs"   # = the terra2k genes when --universe terra2k
SEQ_LEN_CELL = 256      # model_config['data']['seq_len_cell']
BLUR = 0.01             # terra infer_token_distance default
N_RANDOM_SETS = 10      # TERRA paper protocol
MIN_UNIVERSE = 1000     # precision@k needs a universe much larger than k: below this it is
                        # reported as null (crc_221 Endothelial has 531 control cells -> 76 HVGs)
EMB_KWARGS = dict(emb_layer=None, agg_excluded_genes=None, top_k=None, batch_size=32,
                  include_spatial_cell_emb=True, return_token_embeddings=True,
                  ignore_spc_tokens=True, num_workers=8)


def _dense(m):
    return np.asarray(m.todense()) if sp.issparse(m) else np.asarray(m)


def cache_suffix(d, scored_ct):
    """`_{ct}` for an observed scored type, `_terra2k` for the shift-path universe (both stack).

    The terra2k part is not cosmetic: that universe has its own cell set and gene axis, so
    inference.py's benchmark ctrl_tok/pert_tok/logfc caches must not be picked up.
    """
    return (("" if scored_ct == d.holdout_ct else f"_{scored_ct}") +
            ("" if d.universe == "benchmark" else f"_{d.universe}"))


def build_inputs(d, tok, model_dir_path, hd, scored_ct, max_cells=None):
    """ctrl/pert tokenised sets + delta, from inference.py's caches or rebuilt identically."""
    a = d.adata_terra
    args, work = d.args, Path(d.paths["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    suf = cache_suffix(d, scored_ct)
    is_ctrl = ((a.obs[args["domains_key"]].isin(args["control_domains"])) &
               (a.obs[args["labels_key"]].astype(str) == scored_ct)).to_numpy()
    is_tgt = ((a.obs[args["domains_key"]].astype(str) == hd) &
              (a.obs[args["labels_key"]].astype(str) == scored_ct)).to_numpy()

    logfc_csv = work / f"logfc_{hd}{suf}.csv"
    if logfc_csv.exists():
        logfc_df = pd.read_csv(logfc_csv, index_col=0)
        assert "logfc" not in logfc_df.columns, f"{logfc_csv} predates the per-cell-type delta"
    else:
        logfc_df = common.perturbation_logfc(d.adata_hvg, args, hd, scored_ct)
        logfc_df.to_csv(logfc_csv)
    delta, ct_code = common.token_delta(logfc_df, a, model_dir_path, args["labels_key"])
    n_mapped = int((delta != 0).sum())

    with open(f"{model_dir_path}/token_dictionary.pkl", "rb") as f:
        token_dict = pickle.load(f)
    sym2ens = a.var["ensembl_id"].to_dict()

    pos_of = {str(c): i for i, c in enumerate(tok.with_format(None)["cell_id"])}
    ctrl_ids = a.obs.loc[is_ctrl, "cell_id"].astype(str).tolist()
    if max_cells:
        ctrl_ids = ctrl_ids[:max_cells]
    ctrl_rows = [pos_of[c] for c in ctrl_ids]

    ctrl_p, pert_p = work / f"ctrl_tok_{hd}{suf}", work / f"pert_tok_{hd}{suf}"
    if ctrl_p.exists() and pert_p.exists() and max_cells is None:
        ctrl_tok, pert_tok = load_from_disk(str(ctrl_p)), load_from_disk(str(pert_p))
        print(f"loaded cached {ctrl_p} / {pert_p}")
    else:
        ctrl_tok = tok.select(ctrl_rows)
        pert_tok = None
    assert [str(c) for c in ctrl_tok.with_format(None)["cell_id"]] == ctrl_ids
    n_pos = len(ctrl_tok.with_format(None)[0]["gene_tokens"])
    codes = common.position_codes(ctrl_tok, a.obsm["spatial"], ctrl_rows, ct_code, n_pos)
    if pert_tok is None:
        pert_tok = common.map_perturbation(ctrl_tok, delta, codes, SEQ_LEN_CELL)
        if max_cells is None:
            ctrl_tok.save_to_disk(str(ctrl_p))
            pert_tok.save_to_disk(str(pert_p))
            print(f"wrote {ctrl_p} / {pert_p}")
    print(f"control cells: {len(ctrl_tok):,} | target ({hd}) cells: {int(is_tgt.sum()):,}")

    # Random-gene controls (notebook cell 26): same logFC values, random neighbour tokens.
    nb_pool = np.unique(np.asarray(ctrl_tok.with_format(None)["gene_tokens"])[:, SEQ_LEN_CELL:])
    nb_pool = nb_pool[nb_pool != 0]
    rand_toks = []
    for i in range(N_RANDOM_SETS):
        rng = np.random.default_rng(SEED + i)
        dr = np.zeros_like(delta)
        for r, row in enumerate(delta):
            vals = row[row != 0]
            if len(vals):
                dr[r, rng.choice(nb_pool, len(vals), replace=False)] = vals
        rand_toks.append(common.map_perturbation(ctrl_tok, dr, codes, SEQ_LEN_CELL))
    print(f"{len(rand_toks)} random-gene control sets: {n_mapped} tokens each, "
          f"drawn from the {len(nb_pool)} tokens present in the control neighbourhoods")

    tok2sym = {token_dict[e]: g for g, e in sym2ens.items() if e in token_dict}
    ctrl_counts = _dense(a.layers["counts"][is_ctrl])
    gt_lfc = pd.Series(
        get_lfc(control=ctrl_counts, target=_dense(a.layers["counts"][is_tgt]),
                counterfactual=ctrl_counts, n_deg=1)[0],
        index=a.var_names.astype(str))
    return ctrl_tok, pert_tok, rand_toks, delta, tok2sym, gt_lfc, n_mapped


CLOUD_CHUNK = 2000      # cells per embed call: token_emb is ~4.3 MB/cell before the slice


def _token_cloud(model_folder, ds):
    """The cell's own token embeddings under FULL neighbourhood attention, flattened to
    (n_cells * 256, 384) and L2-normalised, on the GPU.

    Embedded in chunks: token_emb comes back with all 2816 positions (~4.3 MB per cell) and only
    the first SEQ_LEN_CELL are the cell's own, so a whole slide's control set (20k cells on
    crc_210) would need ~88 GB of host RAM -- and w2_against holds two clouds at once.  Slicing
    each chunk before concatenating keeps the peak at CLOUD_CHUNK * 4.3 MB and changes nothing:
    embed_dataset is per-cell with the neighbourhood carried inside each row.
    """
    parts = []
    for lo in range(0, len(ds), CLOUD_CHUNK):
        out = embed_dataset(dataset=ds.select(range(lo, min(lo + CLOUD_CHUNK, len(ds)))),
                            **dict(EMB_KWARGS, model_folder_path=str(model_folder)))
        # token_emb IS n_emb: all 2816 positions (96M has special_tokens: []), so position i is
        # gene_tokens[i].
        parts.append(np.ascontiguousarray(out["token_emb"][:, :SEQ_LEN_CELL]))
        del out
    emb = torch.from_numpy(np.concatenate(parts) if len(parts) > 1 else parts[0])
    del parts
    emb = emb.reshape(-1, emb.shape[-1]).to("cuda:0", torch.float32)
    return emb / (emb.norm(dim=1, keepdim=True) + 1e-12)      # _l2_normalize_rows


def score_gene_shift(model_folder, ctrl_tok, pert_tok, rand_toks, delta, tok2sym, gt_lfc,
                     hvg_set, min_cells, k):
    t0 = time.time()
    tokens = np.asarray(ctrl_tok.with_format(None)["gene_tokens"])[:, :SEQ_LEN_CELL]
    assert (np.asarray(pert_tok.with_format(None)["gene_tokens"])[:, :SEQ_LEN_CELL]
            == tokens).all(), "cell-segment token layout moved -- clouds are not paired"

    n_per_tok = np.bincount(tokens.ravel(), minlength=delta.shape[1])
    n_per_tok[0] = 0                                            # pad token
    cand = np.flatnonzero(n_per_tok >= min_cells)
    utok = np.array([t for t in cand if tok2sym.get(t) in gt_lfc.index])
    n_dropped = len(cand) - len(utok)
    genes = np.array([tok2sym[t] for t in utok])
    truth = gt_lfc.loc[genes].abs().to_numpy()

    # Per-gene occurrence positions, without materialising an (n_genes, n_cells) mask.
    flat = tokens.ravel()
    order = np.argsort(flat, kind="stable")
    lo = np.searchsorted(flat[order], utok, "left")
    hi = np.searchsorted(flat[order], utok, "right")
    pos = torch.from_numpy(order).to("cuda:0")

    loss_fn = SamplesLoss("sinkhorn", p=2, blur=BLUR, backend="tensorized")
    base = _token_cloud(model_folder, ctrl_tok)

    def w2_against(ds):
        other = _token_cloud(model_folder, ds)
        d = np.array([float(loss_fn(base[pos[lo[i]:hi[i]]], other[pos[lo[i]:hi[i]]]))
                      for i in range(len(utok))])
        del other
        torch.cuda.empty_cache()
        return d

    shifts = {"pipeline": w2_against(pert_tok)}
    for i, rt in enumerate(rand_toks, 1):
        shifts[f"random_{i}"] = w2_against(rt)
    self_w2 = float(loss_fn(base[pos[lo[0]:hi[0]]], base[pos[lo[0]:hi[0]]]))
    del base
    torch.cuda.empty_cache()

    is_pert = np.isin(utok, np.flatnonzero((delta != 0).any(axis=0)))
    in_hvg = np.isin(genes, list(hvg_set))
    uni = {"(i) all token-covered": np.ones(len(utok), bool),
           UNI_HVG: in_hvg,
           "(iii) minus perturbed genes": ~is_pert}

    res, rows = {}, []
    for uname, m in uni.items():
        t, n = truth[m], int(m.sum())
        small = n < MIN_UNIVERSE      # precision@k is not discriminative on a tiny universe
        scored = {s: dict(precision=None if small else precision(t, w[m], k=k),
                          pearson=pearsonr(t, w[m])[0], spearman=spearmanr(t, w[m])[0])
                  for s, w in shifts.items()}
        rnd = [v for s, v in scored.items() if s.startswith("random_")]
        res[uname] = dict(n_genes=n, k=k, chance=None if small else k / n, **scored["pipeline"],
                          **{f"random_{f}_mean": None if (small and f == "precision") else
                             float(np.mean([r[f] for r in rnd]))
                             for f in ("precision", "pearson", "spearman")})
        if small:
            res[uname]["precision_skipped"] = f"universe {n} < MIN_UNIVERSE {MIN_UNIVERSE}"
        for s, v in scored.items():
            rows.append(dict(universe=uname, arm=s, n_genes=n, **v))
        rows.append(dict(universe=uname, arm="chance", n_genes=n,
                         precision=None if small else k / n, pearson=np.nan, spearman=np.nan))

    assert np.isfinite(np.concatenate(list(shifts.values()))).all(), "non-finite W2"
    assert self_w2 < 1e-3, f"W2 of a cloud against itself is {self_w2:.3e}, not ~0"
    assert int((~is_pert).sum()) == len(utok) - int(is_pert.sum()), "universe (iii) size"
    assert 0 < in_hvg.sum() <= len(utok), "universe (ii) size"
    assert all(0 <= r["precision"] <= 1 for r in rows
               if r["precision"] is not None), "precision@k outside [0, 1]"

    per_gene = pd.DataFrame(dict(gene=genes, n_cells=n_per_tok[utok], w2=shifts["pipeline"],
                                 abs_log2fc=truth, is_perturbed=is_pert, in_hvg=in_hvg))
    print(f"gene shift: |universe|={len(utok)} genes in >= {min_cells} control cells | "
          f"{n_dropped} tokens dropped (no ground truth in adata_terra.var) | "
          f"{int(is_pert.sum())} perturbed genes present | self-W2={self_w2:.2e} | "
          f"{len(rand_toks)} random sets | {time.time() - t0:.0f}s")
    return res, per_gene, pd.DataFrame(rows)


def cellina_abs_lfc(cf_path, mean_ctrl, ctrl_names, index):
    """|log2FC(mean CP10K counterfactual, mean CP10K control)| per gene -> Series over `index`.

    The counterfactual is written by scripts/cellina_node_pert.py through
    train_loo.save_recon_adata, i.e. counts-like in .X, obs = the control cells.
    """
    cf = ad.read_h5ad(cf_path)
    assert cf.n_vars == len(index), f"{cf_path.name}: {cf.n_vars} genes vs {len(index)}"
    names = cf.obs_names.astype(str)
    missing = set(ctrl_names) - set(names)
    assert not missing, f"{cf_path.name}: {len(missing)} control cells missing from the counterfactual"
    x = _dense(cf.layers["counts"] if cf.X is None else cf.X)[pd.Index(names).get_indexer(ctrl_names)]
    return pd.Series(np.abs(safe_log2_fold_change(_normalize_counts(x).mean(axis=0), mean_ctrl)),
                     index=index)


def score_cellina_cf(d, name, scored_ct, hd, genes, truth, k):
    """Score cellina's node-perturbation counterfactuals on the same genes / |gt| as TERRA.

    Counterfactuals live under the `{sid}_terra2k` sid because train_loo derives the sid from
    the adata stem.  Returns ({arm: metrics}, {arm: per-gene |log2FC|}) or ({}, {}) if absent.
    """
    args = d.args
    base = Path(common.DATA_ROOT) / "datasets/crc" / f"{d.sid}_terra2k" / scored_ct
    paths = {"cellina": base / f"{name}_counterfactual_x_{hd}.h5ad"}
    # however many random arms cellina_node_pert.py wrote -- not necessarily N_RANDOM_SETS.
    for p in sorted(base.glob(f"{name}-random*_counterfactual_x_{hd}.h5ad")):
        paths[f"cellina_random_{p.name.split('-random')[1].split('_')[0]}"] = p
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        print(f"[cellina] WARNING: skipping {name} -- {len(missing)} file(s) missing in {base} "
              f"(first: {missing[0]})")
        return {}, {}

    is_ctrl = ((d.adata_hvg.obs[args["domains_key"]].isin(args["control_domains"])) &
               (d.adata_hvg.obs[args["labels_key"]].astype(str) == scored_ct)).to_numpy()
    ctrl_names = d.adata_hvg.obs_names[is_ctrl].astype(str)
    mean_ctrl = _normalize_counts(_dense(d.adata_hvg.layers["counts"][is_ctrl])).mean(axis=0)
    index = d.adata_hvg.var_names.astype(str)   # same gene axis (and naming) as `genes`

    vecs = {arm: cellina_abs_lfc(p, mean_ctrl, ctrl_names, index).loc[genes].to_numpy()
            for arm, p in paths.items()}
    small = len(genes) < MIN_UNIVERSE
    scored = {arm: dict(precision=None if small else precision(truth, v, k=k),
                        pearson=pearsonr(truth, v)[0],
                        spearman=spearmanr(truth, v)[0]) for arm, v in vecs.items()}
    pr = scored["cellina"]["precision"]
    print(f"[cellina] {name}: {len(ctrl_names):,} control cells, {len(genes)} scored genes | "
          f"precision@{k}={'null (small universe)' if pr is None else format(pr, '.3f')} "
          f"spearman={scored['cellina']['spearman']:.3f}")
    return scored, vecs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, choices=["crc", "merfish"])
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", required=True)
    p.add_argument("--variant", required=True, choices=["frozen", "lora"])
    p.add_argument("--epoch", type=int, default=None, help="--variant lora: LoRA bundle epoch")
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--k", type=int, default=50)          # top-k for every shift-path metric
    p.add_argument("--max-cells", type=int, default=None, help="smoke test: first N control cells")
    p.add_argument("--eval-celltypes", default=None,
                   help="comma-separated cell types to score (default: the holdout only)")
    p.add_argument("--universe", default="benchmark", choices=["benchmark", "terra2k"],
                   help="HVG axis: the benchmark's 2000 HVGs, or terra's own (shift path)")
    p.add_argument("--cellina-cf", default=None,
                   help="cellina counterfactual model name (e.g. cellina-pert) to score alongside")
    a = p.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    assert (a.epoch is not None) == (a.variant == "lora"), "--epoch is required iff --variant lora"
    model_name = "terra-frozen" if a.variant == "frozen" else f"terra-lora-ep{a.epoch}"
    md = common.model_dir()                      # pretrained TERRA-96M bundle
    d = common.load_dataset(a.dataset_name, a.adata_path, a.holdout_celltype, md, universe=a.universe)
    # lora: one slide-level bundle (self-sup on all cells), not a per-holdout fine-tune.
    encoder = md if a.variant == "frozen" else (
        Path(d.paths["out_dir"]).parent / "terra" / f"lora_ep{a.epoch}" / "lora_bundle")
    assert Path(encoder).exists(), f"encoder bundle missing: {encoder}"
    tok = common.tokenize_cached(d.adata_terra, md, d.paths["tok_cache"])
    hvg_set = set(d.adata_hvg.var_names.astype(str))
    corr_dir = Path(d.paths["corr_dir"])
    corr_dir.mkdir(parents=True, exist_ok=True)
    uni_suf = "" if a.universe == "benchmark" else f"-{a.universe}"

    for ct in (a.eval_celltypes.split(",") if a.eval_celltypes else [a.holdout_celltype]):
        for hd in d.args["holdout_domains"]:
            inputs = build_inputs(d, tok, md, hd, ct, max_cells=a.max_cells)
            n_mapped = inputs[-1]
            res, per_gene, table = score_gene_shift(encoder, *inputs[:-1], hvg_set,
                                                    a.min_cells, a.k)

            if a.cellina_cf:
                m = per_gene["in_hvg"].to_numpy()
                scored, vecs = score_cellina_cf(d, a.cellina_cf, ct, hd, per_gene["gene"].to_numpy()[m],
                                                per_gene["abs_log2fc"].to_numpy()[m], a.k)
                rnd = [v for s, v in scored.items() if s.startswith("cellina_random_")]
                if scored:
                    res[UNI_HVG].update(
                        cellina_model=a.cellina_cf,
                        **{f"cellina_{f}": scored["cellina"][f] for f in ("precision", "pearson", "spearman")},
                        # precision is None on a universe below MIN_UNIVERSE -- keep it null
                        **{f"cellina_random_{f}_mean":
                           None if any(r[f] is None for r in rnd) else float(np.mean([r[f] for r in rnd]))
                           for f in ("precision", "pearson", "spearman")})
                    table = pd.concat([table, pd.DataFrame(
                        [dict(universe=UNI_HVG, arm=s, n_genes=int(m.sum()), **v)
                         for s, v in scored.items()])], ignore_index=True)
                    for arm, v in vecs.items():
                        per_gene.loc[m, f"{arm}_abs_log2fc"] = v
            print(table.round(4).to_string(index=False))

            csv = Path(d.paths["work_dir"]) / f"gene_shift_{a.variant}_{hd}{cache_suffix(d, ct)}.csv"
            per_gene.to_csv(csv, index=False)
            out = corr_dir / f"{d.sid}_{model_name}-native{uni_suf}_{ct}_{hd}.json"
            out.write_text(json.dumps(
                dict(res, variant=a.variant, k=a.k, n_random_sets=N_RANDOM_SETS,
                     min_cells=a.min_cells, n_perturbed_mapped=n_mapped), indent=2))
            print(f"wrote {out}\nwrote {csv}")


if __name__ == "__main__":
    main()
