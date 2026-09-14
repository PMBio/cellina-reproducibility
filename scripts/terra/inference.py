#!/usr/bin/env python
"""TERRA LOO inference: embed -> count decoder -> neighbour perturbation -> eval_loo h5ads.

Straight port of `notebooks/loo_benchmarks/terra/terra_node_pert_finetune.ipynb` cells 23-26 + 31
onto the benchmark's own preprocessing (scripts/terra/common.py). See PIPELINE_SPEC.md.

    python scripts/terra/inference.py --dataset_name crc --adata_path $DATA_ROOT/datasets/crc/raw_zenodo/crc_232.h5ad \
        --holdout_celltype Fibroblast --variant frozen [--skip-eval] [--max-cells N]
"""
import os

# embed_dataset() hardcodes cuda:0 -> after this line "cuda:0" IS physical GPU 0. GPU 1 is someone else's.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import logging
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common
from train_loo import save_recon_adata
from counterfactual_analysis import get_global_perturbation_logfc
from terra.inference import embed_dataset
from terra.training.decode import apply_count_decoder

import datasets as hf_datasets

hf_datasets.disable_progress_bars()
for _n in ("terra", "datasets", "accelerate"):
    logging.getLogger(_n).setLevel(logging.WARNING)

EMB_KEYS = ["cell_emb", "spatial_cell_emb", "neighborhood_emb"]
EMB_KWARGS = dict(emb_layer=None, agg_excluded_genes=None, top_k=None, batch_size=32,
                  include_spatial_cell_emb=True, return_token_embeddings=False,
                  ignore_spc_tokens=True, num_workers=8)
SEQ_LEN_CELL = 256   # model_config['data']['seq_len_cell']
SEED = 0


def smoke_subset(dataset_name, adata_path, holdout_ct, args, n, seed=SEED):
    """Write a small copy of the raw h5ad (smoke only) and return its path.

    Keeps a floor of control / holdout-domain cells of the holdout ct so the counterfactual
    still has something to score, fills the rest at random.
    ponytail: labels are recomputed here the way preprocess_crc does, rather than plumbing
    a max_cells argument through common.load_dataset.
    """
    out = Path(common.DATA_ROOT) / "datasets" / f"{Path(adata_path).stem}_max{n}.h5ad"
    if out.exists():
        print("[smoke] reusing", out)
        return str(out)
    a = sc.read_h5ad(adata_path)
    a.obs_names_make_unique()
    if dataset_name == "crc":
        from _labels_to_coarse import LABEL_TO_COARSE
        lab = a.obs["ist"].map(LABEL_TO_COARSE).astype(str)
        dom = a.obs["typ"].str.extract(r"(REF|TVA|CRC)", expand=False).astype(str)
    else:
        lab = a.obs[args["labels_key"]].astype(str)
        dom = a.obs[args["domains_key"]].astype(str)
    rng = np.random.default_rng(seed)
    keep = []
    for doms in [args["control_domains"], args["holdout_domains"]]:
        idx = np.where((lab == holdout_ct).to_numpy() & dom.isin(doms).to_numpy())[0]
        keep.append(rng.choice(idx, min(len(idx), n // 4), replace=False))
    keep = np.unique(np.concatenate(keep))
    rest = np.setdiff1d(np.arange(a.n_obs), keep)
    keep = np.sort(np.concatenate([keep, rng.choice(rest, min(len(rest), max(0, n - len(keep))), replace=False)]))
    out.parent.mkdir(parents=True, exist_ok=True)
    a[keep].copy().write_h5ad(out)
    print(f"[smoke] wrote {out} ({len(keep)} cells)")
    return str(out)


def embed_cached(tok, enc, cache):
    """embed_dataset over `tok`, cached as an npz keyed by cell_id (notebook cell 31)."""
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print("loaded cached embeddings", cache)
        return {k: z[k] for k in EMB_KEYS}, [str(c) for c in z["cell_id"]]
    emb = embed_dataset(dataset=tok, model_folder_path=str(enc), **EMB_KWARGS)
    ids = [str(c) for c in tok.with_format(None)["cell_id"]]
    emb = {k: np.asarray(emb[k]) for k in EMB_KEYS}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, cell_id=np.array(ids, dtype=object), **emb)
    print("wrote", cache)
    return emb, ids


def perturb(d, tok, model_dir, hd, work_dir):
    """Notebook cells 23-25: global logFC -> token delta -> additive shift on neighbour tokens."""
    ctrl_path, pert_path = work_dir / f"ctrl_tok_{hd}", work_dir / f"pert_tok_{hd}"
    a = d.adata_terra
    args = d.args
    logfc = get_global_perturbation_logfc(
        a, control_domain=args["control_domains"][0], holdout_domain=hd,
        labels_key=args["labels_key"], domains_key=args["domains_key"], holdout_ct=d.holdout_ct)
    assert np.isfinite(logfc).all(), "logFC contains non-finite values"
    top_genes = logfc.abs().nlargest(common.N_PERT_GENES).index

    with open(f"{model_dir}/token_dictionary.pkl", "rb") as f:
        token_dict = pickle.load(f)
    sym2ens = a.var["ensembl_id"].to_dict()
    delta = np.zeros(max(token_dict.values()) + 1, dtype=np.float64)
    mapped = []
    for g in top_genes:
        tid = token_dict.get(sym2ens.get(g))
        if tid is not None:
            delta[tid] = logfc[g]
            mapped.append(g)
    print(f"{len(mapped)}/{common.N_PERT_GENES} perturbation genes mapped to TERRA tokens")
    pd.DataFrame({"logfc": logfc, "mapped": logfc.index.isin(mapped)}).to_csv(work_dir / f"logfc_{hd}.csv")

    is_control = ((a.obs[args["domains_key"]].isin(args["control_domains"])) &
                  (a.obs[args["labels_key"]].astype(str) == d.holdout_ct)).to_numpy()
    tok_ids = [str(c) for c in tok.with_format(None)["cell_id"]]
    pos_of = {c: i for i, c in enumerate(tok_ids)}
    ctrl_ids = a.obs.loc[is_control, "cell_id"].astype(str).tolist()
    ctrl_tok = tok.select([pos_of[c] for c in ctrl_ids])
    assert [str(c) for c in ctrl_tok.with_format(None)["cell_id"]] == ctrl_ids

    def shift_neighbourhood(batch):
        # Add the log-space logFC to neighbour tokens only. Cell segment [0:256] untouched.
        tokens = np.asarray(batch["gene_tokens"])
        expr = np.asarray(batch["gene_expr"], dtype=np.float64)
        nb_tok, nb_expr = tokens[:, SEQ_LEN_CELL:], expr[:, SEQ_LEN_CELL:]
        shift = delta[nb_tok]
        shift[nb_expr <= 0] = 0.0        # undetected / padded genes are not perturbable
        expr[:, SEQ_LEN_CELL:] = np.clip(nb_expr + shift, 0.0, None)
        batch["gene_tokens"], batch["gene_expr"] = tokens, expr
        return batch

    # Map with the torch format OFF -- what terra's own perturb_dataset does; restore it after.
    fmt = ctrl_tok.format
    pert_tok = ctrl_tok.with_format(None).map(shift_neighbourhood, batched=True, batch_size=256)
    pert_tok.set_format(type=fmt["type"], columns=fmt["columns"], output_all_columns=fmt["output_all_columns"])

    e0 = np.asarray(ctrl_tok.with_format(None)["gene_expr"], dtype=np.float64)
    e1 = np.asarray(pert_tok.with_format(None)["gene_expr"], dtype=np.float64)
    dc = np.abs(e0[:, :SEQ_LEN_CELL] - e1[:, :SEQ_LEN_CELL]).max()
    dn = np.abs(e0[:, SEQ_LEN_CELL:] - e1[:, SEQ_LEN_CELL:])
    print(f"perturbed {len(pert_tok):,} control cells | cell segment max|change| {dc:.3e} (must be 0) | "
          f"neigh max|change| {dn.max():.3e} | {(dn > 0).sum(1).mean():.1f} tokens shifted/cell")
    assert dc == 0, "cell segment was modified"
    for p, t in ((ctrl_path, ctrl_tok), (pert_path, pert_tok)):
        if p.exists():
            shutil.rmtree(p)
        t.save_to_disk(str(p))
    return is_control, ctrl_tok, pert_tok


def train_decoder(d, scemb, variant, work_dir, epochs):
    """`python -m terra.training.decode` with the notebook's flags, on the benchmark HVGs."""
    ckpt = work_dir / f"count_decoder_{variant}.pt"
    metrics = work_dir / f"decoder_metrics_{variant}.json"
    train_h5ad, test_h5ad = work_dir / f"decoder_train_{variant}.h5ad", work_dir / f"decoder_test_{variant}.h5ad"
    genes = work_dir / "gene_list.txt"
    genes.write_text("\n".join(d.adata_hvg.var_names))
    if not ckpt.exists():
        # Decoder trains on adata_hvg itself, so its gene axis IS the benchmark's, in order.
        a = d.adata_hvg.copy()
        a.obsm["spatial_cell_emb"] = np.asarray(scemb, dtype=np.float32)
        is_holdout = a.obs["is_holdout"].to_numpy().astype(bool)
        a[~is_holdout].copy().write_h5ad(train_h5ad)
        a[is_holdout].copy().write_h5ad(test_h5ad)
        del a
        cmd = [sys.executable, "-m", "terra.training.decode",
               "--train-adata", str(train_h5ad), "--test-adata", str(test_h5ad),
               "--embed-key", "spatial_cell_emb", "--embed-fallback-key", "spatial_cell_emb",
               "--expression-layer", "counts",
               "--gene-selection", "list", "--gene-list-path", str(genes),
               "--loss-type", "nb_libsize", "--disable-slide-batching",
               "--epochs", str(epochs), "--hidden-dim", "512", "--mlp-depth", "2", "--layer-norm",
               "--early-stop-patience", "5", "--device", "0", "--seed", str(SEED),
               "--output", str(ckpt), "--metrics-json", str(metrics)]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(_REPO))
    else:
        print("using existing", ckpt)
    gene_list = list(torch.load(ckpt, map_location="cpu")["gene_list"])
    assert gene_list == list(d.adata_hvg.var_names), "decoder gene_list != adata_hvg.var_names"
    print(f"[{variant}] decoder test pearson (diagnostic ceiling): "
          f"{json.loads(metrics.read_text())['test']['pearson_mean']:.4f}")
    return ckpt


def decode(emb, var_names, ckpt):
    a = ad.AnnData(X=np.zeros((len(emb), len(var_names)), dtype=np.float32),
                   var=pd.DataFrame(index=pd.Index(var_names)))
    a.obsm["spatial_cell_emb"] = np.asarray(emb, dtype=np.float32)
    apply_count_decoder(a, emb_key="spatial_cell_emb", model_folder_path=None, checkpoint_path=str(ckpt),
                        decoded_counts_layer_key="decoded", embed_fallback_key="spatial_cell_emb", device=0)
    return np.asarray(a.layers["decoded"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, choices=["crc", "merfish"])
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", required=True)
    p.add_argument("--variant", required=True, choices=["ft", "frozen"])
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--max-cells", type=int, default=None, help="smoke only: shrink the input h5ad")
    p.add_argument("--decoder-epochs", type=int, default=30)
    a = p.parse_args()

    assert torch.cuda.is_available(), "TERRA requires a GPU"
    model_dir = common.model_dir()
    args = common.dataset_args(a.dataset_name)
    adata_path = (smoke_subset(a.dataset_name, a.adata_path, a.holdout_celltype, args, a.max_cells)
                  if a.max_cells else a.adata_path)

    d = common.load_dataset(a.dataset_name, adata_path, a.holdout_celltype, model_dir)
    paths = d.paths
    work_dir, out_dir = Path(paths["work_dir"]), Path(paths["out_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = work_dir / "ft_bundle" if a.variant == "ft" else Path(model_dir)
    if not enc.exists():
        raise SystemExit(f"encoder bundle missing: {enc} -- run scripts/terra/finetuning.py first")
    model_name = "terra" if a.variant == "ft" else "terra-frozen"

    tok = common.tokenize_cached(d.adata_terra, model_dir, paths["tok_cache"])

    # --- embed every cell (cached, keyed by variant) ---
    emb, ids = embed_cached(tok, enc, paths["emb_cache"](a.variant))
    obs_ids = d.adata_terra.obs["cell_id"].astype(str)
    scemb = pd.DataFrame(emb["spatial_cell_emb"], index=ids).reindex(obs_ids).to_numpy(dtype=np.float32)
    assert np.isfinite(scemb).all(), "spatial_cell_emb: missing/non-finite rows"
    del emb

    ckpt = train_decoder(d, scemb, a.variant, work_dir, a.decoder_epochs)
    var_names = list(d.adata_hvg.var_names)

    # --- recon: every holdout-ct cell, unperturbed (eval_loo needs it even with --use_cf) ---
    is_ct = (d.adata_hvg.obs[args["labels_key"]].astype(str) == a.holdout_celltype).to_numpy()
    recon_path = out_dir / f"{model_name}_recon_x.h5ad"
    recon = decode(scemb[is_ct], var_names, ckpt)
    print(f"decoded recon {recon.shape} (holdout-ct cells x {len(var_names)} HVGs)")
    save_recon_adata(d.adata_hvg[is_ct].copy(), recon, str(recon_path), latents=scemb[is_ct])
    shutil.copyfile(recon_path, out_dir / f"{model_name}-null_recon_x.h5ad")
    del recon

    for hd in args["holdout_domains"]:
        is_control, ctrl_tok, pert_tok = perturb(d, tok, model_dir, hd, work_dir)
        kw = dict(EMB_KWARGS, model_folder_path=str(enc))
        ctrl_base = embed_dataset(dataset=ctrl_tok, **kw)
        ctrl_pert = embed_dataset(dataset=pert_tok, **kw)
        delta_emb = {k: float(np.abs(np.asarray(ctrl_base[k]) - np.asarray(ctrl_pert[k])).max()) for k in EMB_KEYS}
        print("max |delta| per embedding:", {k: f"{v:.3e}" for k, v in delta_emb.items()})
        assert delta_emb["cell_emb"] < 1e-5, "cell_emb moved -- the perturbation leaked into the cell segment"
        assert delta_emb["spatial_cell_emb"] > 1e-5, "spatial_cell_emb did not move"

        ctrl = d.adata_hvg[is_control].copy()
        for name, e in ((model_name, ctrl_pert), (f"{model_name}-null", ctrl_base)):
            lat = np.asarray(e["spatial_cell_emb"], dtype=np.float32)
            x = decode(lat, var_names, ckpt)
            assert x.shape == (is_control.sum(), len(var_names)), x.shape
            save_recon_adata(ctrl, x, str(out_dir / f"{name}_counterfactual_x_{hd}.h5ad"), latents=lat)
            print(f"wrote {name}_counterfactual_x_{hd}.h5ad {x.shape}")

    if a.skip_eval:
        return
    for m in (model_name, f"{model_name}-null"):
        cmd = [sys.executable, "scripts/eval_loo.py", "--dataset_name", a.dataset_name,
               "--adata_path", adata_path, "--holdout_celltype", a.holdout_celltype,
               "--model_class", "terra", "--model_name", m, "--use_cf"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(_REPO))
    for hd in args["holdout_domains"]:
        for m in (model_name, f"{model_name}-null"):
            j = Path(paths["corr_dir"]) / f"{d.sid}_{m}-cf_{a.holdout_celltype}_{hd}.json"
            r = json.loads(j.read_text())
            print(f"{j.name}: pearson={r['pearson']:.4f} precision={r['precision']:.4f}")


if __name__ == "__main__":
    main()
