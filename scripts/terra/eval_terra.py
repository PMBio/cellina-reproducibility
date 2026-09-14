"""TERRA-native scoring: paper protocol B, gene-embedding W2 shift (no decoder).

PIPELINE_SPEC.md "`eval_terra.py` CLI" + notebooks/loo_benchmarks/terra/GENE_SHIFT_SPEC.md.
Copied from notebook cell 28 (`score_gene_shift`) with three deliberate changes:
universe (ii) is the benchmark's HVGs (`adata_hvg.var_names`, NOT a recomputed HVG set),
--min-cells defaults to 20 and --k to 50 (eval_loo's N_DEG).

Reads inference.py's caches (`ctrl_tok_{hd}`, `pert_tok_{hd}`, `logfc_{hd}.csv`) and
rebuilds them the same way (notebook cells 23-25) when they are absent.

    python scripts/terra/eval_terra.py --dataset_name crc --adata_path ... \
        --holdout_celltype Fibroblast --variant {ft,frozen}
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # before torch; embed_dataset hardcodes cuda:0

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from datasets import load_from_disk
from geomloss import SamplesLoss
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                   # noqa: E402  (also puts scripts/ on sys.path)
from counterfactual_analysis import (           # noqa: E402
    get_lfc, get_global_perturbation_logfc, precision,
)
from terra.inference import embed_dataset       # noqa: E402

SEED = 0
SEQ_LEN_CELL = 256      # model_config['data']['seq_len_cell']
BLUR = 0.01             # terra infer_token_distance default
N_RANDOM_SETS = 3
EMB_KWARGS = dict(emb_layer=None, agg_excluded_genes=None, top_k=None, batch_size=32,
                  include_spatial_cell_emb=True, return_token_embeddings=True,
                  ignore_spc_tokens=True, num_workers=8)


def _dense(m):
    return np.asarray(m.todense()) if sp.issparse(m) else np.asarray(m)


def _shift_neighbourhood(batch, d):
    """Notebook cell 25: add the log-space logFC to neighbour tokens only."""
    tokens = np.asarray(batch["gene_tokens"])
    expr = np.asarray(batch["gene_expr"], dtype=np.float64)
    nb_tok, nb_expr = tokens[:, SEQ_LEN_CELL:], expr[:, SEQ_LEN_CELL:]
    shift = d[nb_tok]
    shift[nb_expr <= 0] = 0.0                     # undetected / padded genes are not perturbable
    expr[:, SEQ_LEN_CELL:] = np.clip(nb_expr + shift, 0.0, None)
    batch["gene_tokens"], batch["gene_expr"] = tokens, expr
    return batch


def _map_pert(ds, d):
    """Perturb with the torch format OFF (terra's perturb_dataset does the same), then restore."""
    fmt = ds.format
    out = ds.with_format(None).map(_shift_neighbourhood, batched=True, batch_size=256,
                                   fn_kwargs={"d": d})
    out.set_format(type=fmt["type"], columns=fmt["columns"],
                   output_all_columns=fmt["output_all_columns"])
    return out


def build_inputs(d, tok, model_dir_path, hd, max_cells=None):
    """ctrl/pert tokenised sets + delta, from inference.py's caches or rebuilt identically."""
    a = d.adata_terra
    args, work = d.args, Path(d.paths["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    is_ctrl = ((a.obs[args["domains_key"]].isin(args["control_domains"])) &
               (a.obs[args["labels_key"]].astype(str) == d.holdout_ct)).to_numpy()
    is_tgt = ((a.obs[args["domains_key"]].astype(str) == hd) &
              (a.obs[args["labels_key"]].astype(str) == d.holdout_ct)).to_numpy()

    logfc_csv = work / f"logfc_{hd}.csv"
    if logfc_csv.exists():
        logfc = pd.read_csv(logfc_csv, index_col=0)["logfc"]
    else:
        logfc = get_global_perturbation_logfc(
            a, control_domain=args["control_domains"][0], holdout_domain=hd,
            labels_key=args["labels_key"], domains_key=args["domains_key"],
            holdout_ct=d.holdout_ct)
        assert np.isfinite(logfc).all(), "logFC contains non-finite values"
        logfc.rename("logfc").to_csv(logfc_csv)
    top_genes = logfc.abs().nlargest(common.N_PERT_GENES).index

    with open(f"{model_dir_path}/token_dictionary.pkl", "rb") as f:
        token_dict = pickle.load(f)
    sym2ens = a.var["ensembl_id"].to_dict()
    vocab = max(token_dict.values()) + 1
    delta = np.zeros(vocab, dtype=np.float64)
    n_mapped = 0
    for g in top_genes:
        tid = token_dict.get(sym2ens.get(g))
        if tid is not None:
            delta[tid] = logfc[g]
            n_mapped += 1
    print(f"{n_mapped}/{common.N_PERT_GENES} perturbation genes mapped to TERRA tokens")

    ctrl_p, pert_p = work / f"ctrl_tok_{hd}", work / f"pert_tok_{hd}"
    if ctrl_p.exists() and pert_p.exists() and max_cells is None:
        ctrl_tok, pert_tok = load_from_disk(str(ctrl_p)), load_from_disk(str(pert_p))
        print(f"loaded cached {ctrl_p} / {pert_p}")
    else:
        pos_of = {str(c): i for i, c in enumerate(tok.with_format(None)["cell_id"])}
        ctrl_ids = a.obs.loc[is_ctrl, "cell_id"].astype(str).tolist()
        if max_cells:
            ctrl_ids = ctrl_ids[:max_cells]
        ctrl_tok = tok.select([pos_of[c] for c in ctrl_ids])
        assert [str(c) for c in ctrl_tok.with_format(None)["cell_id"]] == ctrl_ids
        pert_tok = _map_pert(ctrl_tok, delta)
        if max_cells is None:
            ctrl_tok.save_to_disk(str(ctrl_p))
            pert_tok.save_to_disk(str(pert_p))
            print(f"wrote {ctrl_p} / {pert_p}")
    print(f"control cells: {len(ctrl_tok):,} | target ({hd}) cells: {int(is_tgt.sum()):,}")

    # Random-gene controls (notebook cell 26): same logFC values, random neighbour tokens.
    nb_pool = np.unique(np.asarray(ctrl_tok.with_format(None)["gene_tokens"])[:, SEQ_LEN_CELL:])
    nb_pool = nb_pool[nb_pool != 0]
    vals = delta[delta != 0]
    rand_toks = []
    for i in range(N_RANDOM_SETS):
        dr = np.zeros(vocab, dtype=np.float64)
        dr[np.random.default_rng(SEED + i).choice(nb_pool, len(vals), replace=False)] = vals
        rand_toks.append(_map_pert(ctrl_tok, dr))
    print(f"{len(rand_toks)} random-gene control sets: {len(vals)} tokens each, "
          f"drawn from the {len(nb_pool)} tokens present in the control neighbourhoods")

    tok2sym = {token_dict[e]: g for g, e in sym2ens.items() if e in token_dict}
    ctrl_counts = _dense(a.layers["counts"][is_ctrl])
    gt_lfc = pd.Series(
        get_lfc(control=ctrl_counts, target=_dense(a.layers["counts"][is_tgt]),
                counterfactual=ctrl_counts, n_deg=1)[0],
        index=a.var_names.astype(str))
    return ctrl_tok, pert_tok, rand_toks, delta, tok2sym, gt_lfc, n_mapped


def _token_cloud(model_folder, ds):
    """The cell's own token embeddings under FULL neighbourhood attention, flattened to
    (n_cells * 256, 384) and L2-normalised, on the GPU."""
    out = embed_dataset(dataset=ds, **dict(EMB_KWARGS, model_folder_path=str(model_folder)))
    # token_emb IS n_emb: all 2816 positions (96M has special_tokens: []), so position i is
    # gene_tokens[i]. ~12.5 GB fp32 for 2,881 cells -- slice to 1.1 GB and drop it.
    emb = torch.from_numpy(np.ascontiguousarray(out["token_emb"][:, :SEQ_LEN_CELL]))
    del out
    emb = emb.reshape(-1, emb.shape[-1]).to("cuda:0", torch.float32)
    return emb / (emb.norm(dim=1, keepdim=True) + 1e-12)      # _l2_normalize_rows


def score_gene_shift(model_folder, ctrl_tok, pert_tok, rand_toks, delta, tok2sym, gt_lfc,
                     hvg_set, min_cells, k):
    t0 = time.time()
    tokens = np.asarray(ctrl_tok.with_format(None)["gene_tokens"])[:, :SEQ_LEN_CELL]
    assert (np.asarray(pert_tok.with_format(None)["gene_tokens"])[:, :SEQ_LEN_CELL]
            == tokens).all(), "cell-segment token layout moved -- clouds are not paired"

    n_per_tok = np.bincount(tokens.ravel(), minlength=len(delta))
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

    is_pert = np.isin(utok, np.flatnonzero(delta != 0))
    in_hvg = np.isin(genes, list(hvg_set))
    uni = {"(i) all token-covered": np.ones(len(utok), bool),
           "(ii) n benchmark HVGs": in_hvg,
           "(iii) minus perturbed genes": ~is_pert}

    res, rows = {}, []
    for uname, m in uni.items():
        t, n = truth[m], int(m.sum())
        scored = {s: dict(precision=precision(t, w[m], k=k),
                          pearson=pearsonr(t, w[m])[0], spearman=spearmanr(t, w[m])[0])
                  for s, w in shifts.items()}
        rnd = [v for s, v in scored.items() if s.startswith("random_")]
        res[uname] = dict(n_genes=n, k=k, chance=k / n, **scored["pipeline"],
                          **{f"random_{f}_mean": float(np.mean([r[f] for r in rnd]))
                             for f in ("precision", "pearson", "spearman")})
        for s, v in scored.items():
            rows.append(dict(universe=uname, arm=s, n_genes=n, **v))
        rows.append(dict(universe=uname, arm="chance", n_genes=n, precision=k / n,
                         pearson=np.nan, spearman=np.nan))

    assert np.isfinite(np.concatenate(list(shifts.values()))).all(), "non-finite W2"
    assert self_w2 < 1e-3, f"W2 of a cloud against itself is {self_w2:.3e}, not ~0"
    assert int((~is_pert).sum()) == len(utok) - int(is_pert.sum()), "universe (iii) size"
    assert 0 < in_hvg.sum() <= len(utok), "universe (ii) size"
    assert all(0 <= r["precision"] <= 1 for r in rows), "precision@k outside [0, 1]"

    per_gene = pd.DataFrame(dict(gene=genes, n_cells=n_per_tok[utok], w2=shifts["pipeline"],
                                 abs_log2fc=truth, is_perturbed=is_pert, in_hvg=in_hvg))
    print(f"gene shift: |universe|={len(utok)} genes in >= {min_cells} control cells | "
          f"{n_dropped} tokens dropped (no ground truth in adata_terra.var) | "
          f"{int(is_pert.sum())} perturbed genes present | self-W2={self_w2:.2e} | "
          f"{len(rand_toks)} random sets | {time.time() - t0:.0f}s")
    return res, per_gene, pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, choices=["crc", "merfish"])
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", required=True)
    p.add_argument("--variant", required=True, choices=["ft", "frozen"])
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--k", type=int, default=50)          # eval_loo.N_DEG
    p.add_argument("--max-cells", type=int, default=None, help="smoke test: first N control cells")
    a = p.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model_name = "terra" if a.variant == "ft" else "terra-frozen"
    md = common.model_dir()
    d = common.load_dataset(a.dataset_name, a.adata_path, a.holdout_celltype, md)
    encoder = Path(d.paths["work_dir"]) / "ft_bundle" if a.variant == "ft" else md
    assert Path(encoder).exists(), f"encoder bundle missing: {encoder}"
    tok = common.tokenize_cached(d.adata_terra, md, d.paths["tok_cache"])
    hvg_set = set(d.adata_hvg.var_names.astype(str))
    corr_dir = Path(d.paths["corr_dir"])
    corr_dir.mkdir(parents=True, exist_ok=True)

    for hd in d.args["holdout_domains"]:
        inputs = build_inputs(d, tok, md, hd, max_cells=a.max_cells)
        n_mapped = inputs[-1]
        res, per_gene, table = score_gene_shift(encoder, *inputs[:-1], hvg_set,
                                                a.min_cells, a.k)
        print(table.round(4).to_string(index=False))

        csv = Path(d.paths["work_dir"]) / f"gene_shift_{a.variant}_{hd}.csv"
        per_gene.to_csv(csv, index=False)
        out = corr_dir / f"{d.sid}_{model_name}-native_{a.holdout_celltype}_{hd}.json"
        out.write_text(json.dumps(
            dict(res, variant=a.variant, min_cells=a.min_cells,
                 n_perturbed_mapped=n_mapped), indent=2))
        print(f"wrote {out}\nwrote {csv}")


if __name__ == "__main__":
    main()
