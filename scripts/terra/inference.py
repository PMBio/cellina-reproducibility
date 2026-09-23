#!/usr/bin/env python
"""TERRA LOO inference: embed -> count decoder -> neighbour perturbation -> eval_loo h5ads.

Ported from the original exploratory notebook (no longer in the repo) onto the benchmark's own
preprocessing (scripts/terra/common.py).  Decoder track of scripts/terra/README.md.

    python scripts/terra/inference.py --dataset_name crc --adata_path $DATA_ROOT/datasets/crc/raw_zenodo/crc_232.h5ad \
        --holdout_celltype Fibroblast --variant frozen [--skip-eval] [--max-cells N]
    python scripts/terra/inference.py ... --variant lora --epoch 5
"""
import os

# embed_dataset() hardcodes cuda:0 -> after this line "cuda:0" IS physical GPU 0. GPU 1 is someone else's.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common
from train_loo import save_recon_adata, split_indices
from terra.inference import embed_dataset
from terra.training.decode import apply_count_decoder

import datasets as hf_datasets

hf_datasets.disable_progress_bars()
for _n in ("terra", "datasets", "accelerate"):
    logging.getLogger(_n).setLevel(logging.WARNING)

EMB_KEYS, EMB_KWARGS = common.EMB_KEYS, common.EMB_KWARGS
SEQ_LEN_CELL = 256   # model_config['data']['seq_len_cell']
DECODER_EMB = ["cell_emb", "neighborhood_emb"]   # cellina's cat(z, s) readout
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
        _smoke_terra2k(Path(adata_path).stem, out)
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
    _smoke_terra2k(Path(adata_path).stem, out)
    return str(out)


def _smoke_terra2k(sid, smoke_path):
    """Mirror a smoke subset onto the terra2k h5ad so `eval_terra.py --universe terra2k` runs."""
    src, _ = common.terra2k_path(sid)
    dst, dst_genes = common.terra2k_path(Path(smoke_path).stem)
    if not src.exists() or dst.exists():
        return
    keep = set(sc.read_h5ad(smoke_path).obs_names.astype(str))
    b = sc.read_h5ad(src)
    b.obs_names_make_unique()
    b = b[[n in keep for n in b.obs_names.astype(str)]].copy()
    b.write_h5ad(dst)
    dst_genes.write_text("\n".join(map(str, b.var_names)))
    print(f"[smoke] wrote {dst} ({b.n_obs} cells x {b.n_vars} genes)")


def perturb(d, tok, model_dir, hd, work_dir, scored_ct):
    """Per-neighbour-cell-type logFC over the benchmark HVGs -> additive shift on neighbour tokens.

    `scored_ct` is the cell type whose control cells are perturbed; it is the holdout for a
    plain LOO run and any observed type under --eval-celltypes (caches then get a _{ct} suffix).
    """
    suf = "" if scored_ct == d.holdout_ct else f"_{scored_ct}"
    ctrl_path, pert_path = work_dir / f"ctrl_tok_{hd}{suf}", work_dir / f"pert_tok_{hd}{suf}"
    a = d.adata_terra
    args = d.args
    logfc_df = common.perturbation_logfc(d.adata_hvg, args, hd, scored_ct)
    delta, ct_code = common.token_delta(logfc_df, a, model_dir, args["labels_key"])
    logfc_df.to_csv(work_dir / f"logfc_{hd}{suf}.csv")

    is_control = ((a.obs[args["domains_key"]].isin(args["control_domains"])) &
                  (a.obs[args["labels_key"]].astype(str) == scored_ct)).to_numpy()
    tok_ids = [str(c) for c in tok.with_format(None)["cell_id"]]
    pos_of = {c: i for i, c in enumerate(tok_ids)}
    ctrl_ids = a.obs.loc[is_control, "cell_id"].astype(str).tolist()
    ctrl_rows = [pos_of[c] for c in ctrl_ids]
    ctrl_tok = tok.select(ctrl_rows)
    assert [str(c) for c in ctrl_tok.with_format(None)["cell_id"]] == ctrl_ids

    n_pos = len(ctrl_tok.with_format(None)[0]["gene_tokens"])
    codes = common.position_codes(ctrl_tok, a.obsm["spatial"], ctrl_rows, ct_code, n_pos)
    pert_tok = common.map_perturbation(ctrl_tok, delta, codes, SEQ_LEN_CELL)

    e0 = np.asarray(ctrl_tok.with_format(None)["gene_expr"], dtype=np.float64)
    e1 = np.asarray(pert_tok.with_format(None)["gene_expr"], dtype=np.float64)
    dc = np.abs(e0[:, :SEQ_LEN_CELL] - e1[:, :SEQ_LEN_CELL]).max()
    dn = np.abs(e0[:, SEQ_LEN_CELL:] - e1[:, SEQ_LEN_CELL:])
    print(f"perturbed {len(pert_tok):,} control cells | cell segment max|change| {dc:.3e} (must be 0) | "
          f"neigh max|change| {dn.max():.3e} | {(dn > 0).sum(1).mean():.1f} tokens shifted/cell")
    assert dc == 0, "cell segment was modified"
    assert len(np.unique(codes[:, SEQ_LEN_CELL:])) > 1, "every neighbour has the same cell-type code"
    for p, t in ((ctrl_path, ctrl_tok), (pert_path, pert_tok)):
        if p.exists():
            shutil.rmtree(p)
        t.save_to_disk(str(p))
    return is_control, ctrl_tok, pert_tok


def train_decoder(d, emb, variant, work_dir, epochs):
    """`python -m terra.training.decode` on the benchmark HVGs, via the NPZ (3-way split) path.

    --train-adata/--test-adata sets has_val=False, which makes decode.py use the HELD-OUT
    set for early stopping and checkpoint selection (decode.py:1621,1659,1882) and folds
    val into train.  The NPZ path (decode.py:1260-1298) takes real val_* keys.
    --drop-zero-variance-genes must stay off: decode.py:1298-1320 silently drops them.
    """
    ckpt = work_dir / f"count_decoder_{variant}.pt"
    metrics = work_dir / f"decoder_metrics_{variant}.json"
    npz = work_dir / f"decoder_data_{variant}.npz"
    cmd = [sys.executable, "-m", "terra.training.decode",
           "--dataset", str(npz), "--gene-selection", "all",
           "--loss-type", "nb_libsize", "--disable-slide-batching",
           "--epochs", str(epochs), "--hidden-dim", "512", "--mlp-depth", "2", "--layer-norm",
           "--early-stop-patience", "5", "--device", "0", "--seed", str(SEED),
           "--output", str(ckpt), "--metrics-json", str(metrics)]
    # Artefacts of the 384-d spatial_cell_emb readout / the val==test split must not be reused.
    for stale in (work_dir / f"decoder_train_{variant}.h5ad", work_dir / f"decoder_test_{variant}.h5ad",
                  work_dir / "gene_list.txt"):
        stale.unlink(missing_ok=True)
    if ckpt.exists() and torch.load(ckpt, map_location="cpu")["config"]["embed_dim"] != emb.shape[1]:
        print(f"[{variant}] cached decoder was trained on a different readout -- retraining")
        ckpt.unlink()
        metrics.unlink(missing_ok=True)
    if not ckpt.exists():
        # adata_hvg IS the benchmark gene axis, in order; splits are the ones cellina uses.
        counts = d.adata_hvg.layers["counts"]
        counts = np.asarray(counts.todense() if sp.issparse(counts) else counts, dtype=np.float32)
        emb = np.asarray(emb, dtype=np.float32)
        obs_names = d.adata_hvg.obs_names.astype(str).to_numpy()
        pack = {}
        for name, idx, expr_key in (("train", d.train_idx, "train_expression"),
                                    ("val", d.val_idx, "val_expression"),
                                    ("test", d.test_idx, "test_expression_gt")):
            pack[f"{name}_embeddings"] = emb[idx]
            pack[expr_key] = counts[idx]
            pack[f"{name}_barcodes"] = obs_names[idx]
            pack[f"{name}_slides"] = np.full(len(idx), d.sid, dtype=object)
        assert not (set(d.train_idx) & set(d.val_idx)), "val cells leaked into train"
        print(f"[{variant}] decoder split: train={len(d.train_idx)} val={len(d.val_idx)} "
              f"test={len(d.test_idx)} | readout {emb.shape[1]}-d")
        np.savez(npz, gene_list=np.array(list(d.adata_hvg.var_names), dtype=object), **pack)
        del counts, pack
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(_REPO))
        npz.unlink()                       # ~1.3 GB of embeddings + counts; rebuilt on demand
        m = json.loads(metrics.read_text())
        m.update(n_train=len(d.train_idx), n_val=len(d.val_idx), n_test=len(d.test_idx))
        metrics.write_text(json.dumps(m, indent=2))
    else:
        print("using existing", ckpt)
    gene_list = list(torch.load(ckpt, map_location="cpu")["gene_list"])
    assert gene_list == list(d.adata_hvg.var_names), "decoder gene_list != adata_hvg.var_names"
    m = json.loads(metrics.read_text())
    assert abs(m["val"]["pearson_mean"] - m["test"]["pearson_mean"]) > 1e-12, (
        "val.pearson_mean == test.pearson_mean -- the decoder is selecting on the held-out set")
    print(f"[{variant}] decoder val pearson {m['val']['pearson_mean']:.4f} | "
          f"test pearson (diagnostic ceiling) {m['test']['pearson_mean']:.4f}")
    return ckpt


def decode(emb, var_names, ckpt):
    a = ad.AnnData(X=np.zeros((len(emb), len(var_names)), dtype=np.float32),
                   var=pd.DataFrame(index=pd.Index(var_names)))
    a.obsm["decoder_emb"] = np.asarray(emb, dtype=np.float32)
    apply_count_decoder(a, emb_key="decoder_emb", model_folder_path=None, checkpoint_path=str(ckpt),
                        decoded_counts_layer_key="decoded", embed_fallback_key="decoder_emb", device=0)
    return np.asarray(a.layers["decoded"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, choices=["crc", "merfish"])
    p.add_argument("--adata_path", required=True)
    p.add_argument("--holdout_celltype", required=True)
    p.add_argument("--variant", required=True, choices=["frozen", "lora"])
    p.add_argument("--epoch", type=int, default=None,
                   help="--variant lora: which LoRA epoch bundle (slide_dir/terra/lora_ep{N})")
    p.add_argument("--eval-celltypes", default=None,
                   help="comma-separated cell types to score with this model's decoder "
                        "(default: the holdout only)")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--max-cells", type=int, default=None, help="smoke only: shrink the input h5ad")
    p.add_argument("--decoder-epochs", type=int, default=100)
    a = p.parse_args()
    assert (a.epoch is not None) == (a.variant == "lora"), "--epoch is for --variant lora only"

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

    # Both encoders are SLIDE-level (the LoRA fine-tune is self-supervised on all cells, the
    # frozen bundle is the pretrained one), so bundle, embeddings and model name carry no
    # holdout tag. Only the decoder is LOO, per scored cell type.
    slide_dir = Path(paths["out_dir"]).parent / "terra"
    if a.variant == "lora":
        ep_dir = slide_dir / f"lora_ep{a.epoch}"
        enc, emb_cache = ep_dir / "lora_bundle", ep_dir / "emb_lora.npz"
        variant, model_name = f"lora_ep{a.epoch}", f"terra-lora-ep{a.epoch}"
    else:
        enc, emb_cache = Path(model_dir), slide_dir / "emb_frozen.npz"
        variant, model_name = "frozen", "terra-frozen"
    if not enc.exists():
        raise SystemExit(f"encoder bundle missing: {enc} -- run scripts/terra/finetuning_lora.py first")

    tok = common.tokenize_cached(d.adata_terra, model_dir, paths["tok_cache"])

    # --- embed every cell (cached per slide x encoder) ---
    emb, ids = common.embed_cached(tok, enc, emb_cache)
    obs_ids = d.adata_terra.obs["cell_id"].astype(str)
    # cellina's readout: cat(z from the cell's own counts, s from its neighbourhood)
    # (cellina/src/cellina/_cellina_module.py:229-252).
    scemb = pd.DataFrame(np.hstack([emb[k] for k in DECODER_EMB]),
                         index=ids).reindex(obs_ids).to_numpy(dtype=np.float32)
    assert np.isfinite(scemb).all(), "cell_emb/neighborhood_emb: missing/non-finite rows"
    del emb

    ckpt_ho = train_decoder(d, scemb, variant, work_dir, a.decoder_epochs)
    var_names = list(d.adata_hvg.var_names)

    # Scored types: the holdout by default. --eval-celltypes scores other types with the SAME
    # encoder (the encoder is never held out) but the decoder is always LOO for the scored
    # type: its own split_indices, trained in its own work_dir, so that type's CRC cells never
    # enter decoder training.
    scored_cts = a.eval_celltypes.split(",") if a.eval_celltypes else [a.holdout_celltype]
    name = model_name
    for ct in scored_cts:
        if ct == a.holdout_celltype:
            ct_dir, ckpt = out_dir, ckpt_ho
        else:
            lay = common.layout(a.dataset_name, adata_path, ct)
            ct_dir, ct_work = Path(lay["out_dir"]), Path(lay["work_dir"])
            ct_work.mkdir(parents=True, exist_ok=True)
            tr, va, te = split_indices(d.adata_hvg, ct, labels_key=args["labels_key"],
                                       domains_key=args["domains_key"],
                                       holdout_domains=args["holdout_domains"], seed=0)
            assert not (set(te) & set(tr)), "scored type's CRC cells leaked into decoder train"
            d_ct = SimpleNamespace(**{**vars(d), "train_idx": tr, "val_idx": va, "test_idx": te})
            ckpt = train_decoder(d_ct, scemb, variant, ct_work, a.decoder_epochs)
        ct_dir.mkdir(parents=True, exist_ok=True)

        # --- recon: every scored-ct cell, unperturbed (eval_loo needs it even with --use_cf) ---
        is_ct = (d.adata_hvg.obs[args["labels_key"]].astype(str) == ct).to_numpy()
        recon_path = ct_dir / f"{name}_recon_x.h5ad"
        recon = decode(scemb[is_ct], var_names, ckpt)
        print(f"decoded recon {recon.shape} ({ct} cells x {len(var_names)} HVGs)")
        save_recon_adata(d.adata_hvg[is_ct].copy(), recon, str(recon_path), latents=scemb[is_ct])
        # the null arm reconstructs the same cells from the same latents: hardlink, do not duplicate
        null_recon = ct_dir / f"{name}-null_recon_x.h5ad"
        null_recon.unlink(missing_ok=True)
        os.link(recon_path, null_recon)
        del recon

        for hd in args["holdout_domains"]:
            is_control, ctrl_tok, pert_tok = perturb(d, tok, model_dir, hd, work_dir, ct)
            kw = dict(EMB_KWARGS, model_folder_path=str(enc))
            ctrl_base = embed_dataset(dataset=ctrl_tok, **kw)
            ctrl_pert = embed_dataset(dataset=pert_tok, **kw)
            delta_emb = {k: float(np.abs(np.asarray(ctrl_base[k]) - np.asarray(ctrl_pert[k])).max()) for k in EMB_KEYS}
            print("max |delta| per embedding:", {k: f"{v:.3e}" for k, v in delta_emb.items()})
            assert delta_emb["cell_emb"] < 1e-5, "cell_emb moved -- the perturbation leaked into the cell segment"
            assert delta_emb["neighborhood_emb"] > 1e-5, "neighborhood_emb did not move"

            # cat(z, s): z always from the UNPERTURBED pass, s from the pass being scored.
            ctrl = d.adata_hvg[is_control].copy()
            z = np.asarray(ctrl_base["cell_emb"], dtype=np.float32)
            for nm, e in ((name, ctrl_pert), (f"{name}-null", ctrl_base)):
                lat = np.hstack([z, np.asarray(e["neighborhood_emb"], dtype=np.float32)])
                x = decode(lat, var_names, ckpt)
                assert x.shape == (is_control.sum(), len(var_names)), x.shape
                save_recon_adata(ctrl, x, str(ct_dir / f"{nm}_counterfactual_x_{hd}.h5ad"), latents=lat)
                print(f"wrote {ct_dir}/{nm}_counterfactual_x_{hd}.h5ad {x.shape}")

        if a.skip_eval:
            continue
        for m in (name, f"{name}-null"):
            cmd = [sys.executable, "scripts/eval_loo.py", "--dataset_name", a.dataset_name,
                   "--adata_path", adata_path, "--holdout_celltype", ct,
                   "--model_class", "terra", "--model_name", m, "--use_cf", "--log_norm_x"]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True, cwd=str(_REPO))
        for hd in args["holdout_domains"]:
            for m in (name, f"{name}-null"):
                j = Path(paths["corr_dir"]) / f"{d.sid}_{m}-cf_{ct}_{hd}.json"
                r = json.loads(j.read_text())
                print(f"{j.name}: pearson={r['pearson']:.4f} precision={r['precision']:.4f}")


if __name__ == "__main__":
    main()
