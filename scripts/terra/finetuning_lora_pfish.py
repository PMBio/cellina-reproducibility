#!/usr/bin/env python
"""Self-supervised (I-JEPA) LoRA fine-tune of TERRA on the in-vivo perturb-FISH slide.

Same recipe as scripts/terra/finetuning_lora.py (TERRA's Xenium tutorial LoRA config, verbatim),
run on ALL cells of the single pfish_terra.h5ad slide -- no holdout, no labels, self-supervised.
Unlike the CRC script this does not go through common.load_dataset() / dataset_args(): pfish has
no cellina LOO-benchmark config (labels_key/domains_key/HVG selection) and is a single slide, so
common.py's CRC/merfish-specific plumbing does not apply. Everything else -- the encoder build,
LoRA config, collapse guard, epoch_selection.json -- is copied as-is from finetuning_lora.py.

  python scripts/terra/finetuning_lora_pfish.py
  python scripts/terra/finetuning_lora_pfish.py --max-steps 10 --epochs 1 --batch-size 32   # smoke

Spatial units: pfish_terra.h5ad's obsm['spatial'] is raw pixel/mask-index coordinates (from
pfish_preprocess.ipynb's coordinates.csv). TERRA's positional encoding expects physical (um)
distances, same as common.py's PX_TO_UM for crc/merfish. The conversion factor here (0.108 um/px)
is confirmed from notebooks/in_vivo/pfish_eda (1).ipynb's own `um_per_pixel = 5.4 / 50` used for
its neighbour-radius analyses on this same 154-gene slide.
"""
import argparse
import json
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--adata_path", default=str(Path(__file__).resolve().parents[2] / "notebooks/in_vivo/pfish_terra.h5ad"))
p.add_argument("--px-to-um", type=float, default=0.108,
               help="pfish_eda (1).ipynb: um_per_pixel = 5.4 / 50")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--lr", type=float, default=1e-4,
               help="peak lr; start/final = lr/10. The tutorial's 1e-3 x LoRA scale alpha/r=16 "
                    "collapsed the encoder on crc_232 within ~75 steps")
p.add_argument("--max-steps", type=int, default=None, help="cap cells to max_steps*batch_size (1 epoch)")
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--guard-cells", type=int, default=20000, help="cells embedded per epoch for the collapse guard")
p.add_argument("--max-cells", type=int, default=None, help="subsample the slide (smoke tests only)")
p.add_argument("--work-dir", default=None,
               help="default: notebooks/in_vivo/terra_pfish/terra (sibling to the other pfish model dirs)")
p.add_argument("--from-run-dir", default=None,
               help="skip training; export/embed/guard the checkpoint_epoch_*.pt in this dir")
p.add_argument("--nproc", type=int, default=16)
A = p.parse_args()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TQDM_DISABLE", "1")

import logging

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import yaml
from datasets import load_from_disk

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common
import terra.training.finetune_self_supervised as fss
from terra.inference import embed_dataset, harmonize_adata
from terra.training.finetune_self_supervised import finetune_self_supervised, prepare_finetuned_model
from terra.utils.helper import init_model, parse_arch_kwargs, parse_protein_init_kwargs

# pfish_terra.h5ad has an /uns/log1p/base entry written with a "null" IOSpec encoding this anndata
# (0.11.4) doesn't have a reader for -- harmless scanpy log1p-base marker, not read by anything
# below. Teach the registry to decode it as None instead of touching the file.
from anndata._io.specs.registry import _REGISTRY, IOSpec

if (h5py.Dataset, IOSpec("null", "0.1.0")) not in _REGISTRY.read:
    @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
    def _read_null(elem, _reader):
        return None

logging.basicConfig(level=logging.WARNING, format="%(message)s")
for name in ("terra.training.finetune_self_supervised", "terra.utils.helper"):
    logging.getLogger(name).setLevel(logging.INFO)

# LoRA touches these submodules; nothing else may move by more than EPS.
TARGETS = ["qkv", "proj", "fc1", "fc2"]
EPS = 1e-6      # "changed" cannot be bitwise: EMA touches every tensor every step.

SID = Path(A.adata_path).stem                                       # "pfish_terra"
SID_DIR = Path(A.work_dir).parent if A.work_dir else _REPO / "notebooks/in_vivo/terra_pfish"
WORK = Path(A.work_dir) if A.work_dir else SID_DIR / "terra"
RUN_DIR = WORK / "lora_run"
SELECTION = WORK / "epoch_selection.json"
TOK_CACHE = SID_DIR / "terra_tok"
if SELECTION.exists():
    print(f"[skip] {SELECTION} exists", flush=True)
    raise SystemExit(0)
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(common.model_dir())
CFG = yaml.safe_load(open(MODEL_DIR / "model_config.yaml"))
TOKD = pickle.load(open(MODEL_DIR / "token_dictionary.pkl", "rb"))


def build_encoder():
    """The encoder exactly as embed.py builds it from the bundle's model_config.yaml."""
    n_special_tokens = len(CFG["meta"]["special_tokens"])
    seq_len = (CFG["data"]["seq_len_cell"] + CFG["data"]["seq_len_neighborhood"]
               + n_special_tokens)
    enc, _ = init_model(
        gt_type=CFG["meta"]["gt_type"], count_encoding=CFG["meta"]["count_encoding"],
        n_value_bins=CFG["meta"]["n_value_bins"], cell_pos_enc=CFG["meta"]["cell_pos_enc"],
        device="cpu", vocab_size=len(TOKD), seq_len=seq_len,
        n_special_tokens=n_special_tokens, n_segments=CFG["data"]["n_segments"],
        enc_emb_dim=CFG["meta"]["enc_emb_dim"], enc_depth=CFG["meta"]["enc_depth"],
        pred_emb_dim=CFG["meta"]["pred_emb_dim"], pred_depth=CFG["meta"]["pred_depth"],
        num_heads=CFG["meta"]["num_heads"], mlp_ratio=CFG["meta"]["mlp_ratio"],
        use_flash_attention=CFG["meta"]["use_flash_attention"],
        api_version=CFG["meta"]["api_version"],
        sep_gene_tokens_neb=CFG["data"]["sep_gene_tokens_neb"],
        predict_gene=CFG["meta"]["predict_gene"], pos_learnable=CFG["meta"]["pos_learnable"],
        n_special_values=CFG["data"].get("n_special_values", 0),
        nz_spc=CFG["data"].get("nz_spc", False),
        mlp_bias=CFG["meta"].get("mlp_bias", True),
        protein_init_kwargs=parse_protein_init_kwargs(CFG, TOKD),
        **parse_arch_kwargs(CFG))
    return enc


# --------------------------------------------------------------------------- #
# 1. the whole slide, harmonised + tokenised (cache reused as-is)
# --------------------------------------------------------------------------- #
def load_pfish_terra(adata_path, model_dir, px_to_um):
    """Mirror common.py's _terra_side() crc branch: pfish gene symbols are already human, so no
    merfish-style upper-case/ortholog step is needed -- just harmonize_adata's own symbol lookup."""
    raw = sc.read_h5ad(adata_path)
    raw.obs_names_make_unique()
    raw.obs["cell_id"] = raw.obs_names.astype(str)

    coords = np.asarray(raw.obsm["spatial"], dtype=np.float64) * px_to_um
    raw.obsm["spatial"] = coords
    ext = np.ptp(coords, axis=0)
    print(f"[terra] tissue extent: {ext[0]:.0f} x {ext[1]:.0f} um (px_to_um={px_to_um})")

    if sp.issparse(raw.X):
        raw.X = raw.X.tocsr()
    raw.layers["counts"] = raw.X.copy()
    raw = harmonize_adata(
        raw,
        gene_mapping_dict_file_path=f"{model_dir}/ensembl_dictionary.pkl",
        gene_occurrence_count_file_path=f"{model_dir}/gene_count_dictionary.pkl",
    )
    if sp.issparse(raw.X):
        raw.X = raw.X.tocsr()
    raw.layers["counts"] = raw.X.copy()     # re-sync after harmonize's gene/cell filtering
    return raw


adata_terra = load_pfish_terra(A.adata_path, str(MODEL_DIR), A.px_to_um)
print(f"[harmonize] {adata_terra.n_obs:,} cells x {adata_terra.n_vars} genes survive TERRA harmonization")
tok_all = common.tokenize_cached(adata_terra, str(MODEL_DIR), TOK_CACHE, nproc=A.nproc)
del adata_terra

FT_TOK = TOK_CACHE
if A.max_steps:
    assert A.epochs == 1, "--max-steps implies a single epoch"
    A.max_cells = min(A.max_steps * A.batch_size, A.max_cells or len(tok_all))
if A.max_cells and A.max_cells < len(tok_all):
    # ponytail: smoke only -- plain random subsample of rows; no cell-id filtering needed
    # because the self-supervised objective is label-free.
    FT_TOK = RUN_DIR / "tok_sub"
    if not FT_TOK.exists():
        rng = np.random.default_rng(A.seed)
        idx = np.sort(rng.choice(len(tok_all), A.max_cells, replace=False))
        tok_all.select(idx).flatten_indices().save_to_disk(str(FT_TOK))
    tok_all = load_from_disk(str(FT_TOK))
N_CELLS = len(tok_all)
print(f"[tok] {N_CELLS} rows from {FT_TOK} | columns {list(tok_all.features)}", flush=True)

# --------------------------------------------------------------------------- #
# 2. patch: fss._build_model calls init_model without the config-dependent extras
#    (nz_spc / n_special_values / protein_init / arch kwargs), same gap finetuning.py patches.
# --------------------------------------------------------------------------- #
_orig_init_model = fss.init_model


def _patched_init_model(**kw):
    kw["nz_spc"] = CFG["data"].get("nz_spc", False)
    kw["mlp_bias"] = CFG["meta"].get("mlp_bias", True)
    kw["n_special_values"] = CFG["data"].get("n_special_values", 0)
    kw["protein_init_kwargs"] = parse_protein_init_kwargs(CFG, TOKD)
    kw.update(parse_arch_kwargs(CFG))
    return _orig_init_model(**kw)


fss.init_model = _patched_init_model

# fss.init_cell_dataset also omits `pad_special_tokens=True`, which embed_dataset always sets:
# with special tokens (112M: ["batch"]) the loader would otherwise look up a per-cell
# `batch_value` -- a corpus batch identity (`spv_{dataset_id}_{batch}`) that new slides cannot
# have.  Padding the slot is exactly what inference does, so train and embed see the same input.
_orig_init_cell_dataset = fss.init_cell_dataset


def _patched_init_cell_dataset(**kw):
    kw.setdefault("pad_special_tokens", True)
    kw.setdefault("truncate_neighbors", CFG["data"].get("truncate_neighbors", False))
    kw.setdefault("tokenized_seq_len_cell", CFG["data"].get("tokenized_seq_len_cell", None))
    return _orig_init_cell_dataset(**kw)


fss.init_cell_dataset = _patched_init_cell_dataset

PRETRAINED = torch.load(MODEL_DIR / "model_checkpoint.pt", map_location="cpu")["target_encoder"]
PRETRAINED = {k.replace("module.", ""): v for k, v in PRETRAINED.items()}
_enc = build_encoder()
_enc.load_state_dict(PRETRAINED)     # strict; raises on any mismatch
n_total = sum(p.numel() for p in _enc.parameters())
print(f"[preflight] ALL KEYS MATCHED | {n_total:,} encoder params", flush=True)
del _enc

# --------------------------------------------------------------------------- #
# 3. log capture + training clock
# --------------------------------------------------------------------------- #
epoch_log = []
_EPOCH_RE = re.compile(r"Epoch \[(\d+)/\d+\], Loss: ([\d.]+), LR: ([\d.e+-]+)")


class _EpochLog(logging.Handler):
    def emit(self, record):
        m = record.getMessage()
        g = _EPOCH_RE.search(m)
        if g:
            epoch_log.append({"line": m, "t": time.time(), "epoch": int(g[1]),
                              "loss": float(g[2]), "lr": float(g[3])})


logging.getLogger("terra.training.finetune_self_supervised").addHandler(_EpochLog())

# Start the clock at the first real step so encoder build / dataloader spin-up is
# excluded from sec_per_step. apply_masks runs once per step inside the loop.
first_step = []
_orig_apply_masks = fss.apply_masks


def _timed_apply_masks(*a, **kw):
    if not first_step:
        first_step.append(time.time())
    return _orig_apply_masks(*a, **kw)


fss.apply_masks = _timed_apply_masks

# --------------------------------------------------------------------------- #
# 4. fine-tune -- Xenium tutorial recipe except lr (see --lr) and warmup (0.1 epoch: our
#    epochs are ~50x longer than the tutorial's, so a 1-epoch warmup never reaches peak)
# --------------------------------------------------------------------------- #
args = {
    "model": {"pretrained_checkpoint_path": str(MODEL_DIR),
              "finetune_checkpoint_path": str(RUN_DIR)},
    "data": {"finetune_dataset": [str(FT_TOK)], "batch_size": A.batch_size,
             "num_workers": 8, "pin_memory": True, "drop_last": True,
             "sample_segments": False, "sample_gene_masks": True},
    "finetune": {"num_epochs": A.epochs, "lr": A.lr, "start_lr": A.lr / 10, "final_lr": A.lr / 10,
                 "warmup_epochs": 0.1, "weight_decay": 0.04, "final_weight_decay": 0.4,
                 "ema_momentum": 0.9995, "final_ema_momentum": 1.0,
                 "loss_fn_type": "smooth_l1", "clip_grad": 2.0, "use_bfloat16": True,
                 "use_peft": True, "peft_method": "lora", "peft_rank": 16,
                 "peft_alpha": 256, "peft_dropout": 0.1, "peft_bias": "none",
                 "peft_target_modules": TARGETS, "save_every": 1},
}
steps_per_epoch = N_CELLS // A.batch_size          # drop_last=True

if A.from_run_dir:
    run_dir = Path(A.from_run_dir)
    wall = peak_gib = float("nan")
    old = next(iter(sorted(run_dir.parents[1].glob("*bundle*/ft_summary.json"))), None)
    prev = json.loads(old.read_text()) if old else {}
    loss_per_epoch = prev.get("loss_per_epoch", [])
    steps_per_epoch = prev.get("steps_per_epoch", steps_per_epoch)
    print(f"[reuse] {run_dir} | prior summary: {old}", flush=True)
else:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    run_dir = Path(finetune_self_supervised(args=args, save_folder_path=str(RUN_DIR),
                                            run_name="run"))
    wall = time.time() - t0
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    assert len(epoch_log) == A.epochs, f"{len(epoch_log)} epoch lines for {A.epochs} epochs"
    loss_per_epoch = [e["loss"] for e in epoch_log]
    print(f"[train] {[e['line'] for e in epoch_log]}", flush=True)
    print(f"[probe] peak_gpu_gib={peak_gib:.1f} sec_per_step={(wall - (first_step[0] - t0)) / max(1, steps_per_epoch * A.epochs):.2f}",
          flush=True)

ckpts = {int(f.stem.split("_")[-1]): f for f in run_dir.glob("checkpoint_epoch_*.pt")}
assert ckpts, f"no checkpoint_epoch_*.pt in {run_dir}"
FINAL_EPOCH = max(ckpts)
print(f"[epochs] checkpoints {sorted(ckpts)} in {run_dir}", flush=True)

(RUN_DIR / "ft_config.json").write_text(json.dumps(
    {"cli": vars(A), "sid": SID, "n_cells": N_CELLS,
     "steps_per_epoch": steps_per_epoch, "finetune_dataset": str(FT_TOK),
     "run_dir": str(run_dir), "terra_args": args}, indent=2, default=str))

# --------------------------------------------------------------------------- #
# 5. per epoch: repack -> verify -> embed every cell -> collapse-guard stats
# --------------------------------------------------------------------------- #
PRE_ONLINE = torch.load(MODEL_DIR / "model_checkpoint.pt", map_location="cpu")["encoder"]
PRE_ONLINE = {k.replace("module.", ""): v for k, v in PRE_ONLINE.items()}


def repack(epoch, bundle):
    """prepare_finetuned_model for one epoch + the structural checks (these still assert:
    they are correctness, not collapse)."""
    if bundle.exists():
        shutil.rmtree(bundle)
    prepare_finetuned_model(finetuned_checkpoint_dir=str(run_dir), pretrained_model_dir=str(MODEL_DIR),
                            output_dir=str(bundle), checkpoint_epoch=epoch, use_peft=True)
    for f in ("model_config.yaml", "token_dictionary.pkl", "ensembl_dictionary.pkl",
              "gene_count_dictionary.pkl", "model_checkpoint.pt"):
        assert (bundle / f).exists(), f"bundle is missing {f}"
    new_sd = torch.load(bundle / "model_checkpoint.pt", map_location="cpu")["target_encoder"]
    new_sd = {k.replace("module.", ""): v for k, v in new_sd.items()}
    build_encoder().load_state_dict(new_sd)     # strict; raises on any mismatch
    assert set(new_sd) == set(PRETRAINED), "repacked key set differs from pretrained"

    deltas = {k: (new_sd[k].float() - PRETRAINED[k].float()).abs().max().item()
              for k in PRETRAINED if PRETRAINED[k].is_floating_point()}
    changed = sorted(k for k, v in deltas.items() if v > EPS)
    lora_changed = [k for k in changed if any(t in k for t in TARGETS)]
    off_target = [k for k in changed if k not in lora_changed]
    assert lora_changed, "no LoRA-target tensor changed -- fine-tuning had no effect"
    per_block = {b: sum(f".blocks.{b}." in k for k in lora_changed)
                 for b in range(CFG["meta"]["enc_depth"])}
    assert not [b for b, n in per_block.items() if n == 0], f"empty LoRA blocks: {per_block}"
    gap = {k: (PRE_ONLINE[k].float() - PRETRAINED[k].float()).abs().max().item() for k in deltas}
    floor = {k: 1e-3 * PRETRAINED[k].float().abs().max().item() for k in deltas}
    unexplained = [k for k in off_target if deltas[k] > 1.01 * gap[k] + max(EPS, floor[k])]
    assert not unexplained, f"off-target tensors moved beyond the pretrained EMA gap: {unexplained[:5]}"
    drift = max((deltas[k] / max(gap[k], 1e-12) for k in off_target), default=0.0)
    print(f"[ep{epoch}] repack {bundle} | {len(changed)}/{len(deltas)} tensors changed: "
          f"{len(lora_changed)} LoRA targets, per-block {per_block}; {len(off_target)} frozen "
          f"tensors EMA-drifted at most {drift:.1%} of the pretrained encoder/target gap", flush=True)
    return {"n_changed_tensors": len(changed), "n_lora_changed": len(lora_changed),
            "lora_changed_per_block": per_block, "n_off_target_drifted": len(off_target),
            "max_off_target_drift_frac_of_ema_gap": drift}


def rank_std(x):
    """(effective rank = exp entropy of the PCA spectrum, mean per-dim std, mean cell-cell cosine)"""
    xc = x - x.mean(0)
    s_ = np.linalg.svd(xc, compute_uv=False) ** 2
    p_ = s_ / s_.sum()
    eff_rank = float(np.exp(-(p_ * np.log(p_ + 1e-12)).sum()))
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    cc = xn @ xn.T
    return eff_rank, float(xc.std(0).mean()), float(cc[np.triu_indices(len(x), 1)].mean())


rng_ = np.random.default_rng(A.seed)
GUARD_ROWS = np.sort(rng_.choice(N_CELLS, min(A.guard_cells, N_CELLS), replace=False))
tok_guard = tok_all.select(GUARD_ROWS)
frozen = embed_dataset(dataset=tok_guard, model_folder_path=str(MODEL_DIR),
                       **dict(common.EMB_KWARGS, num_workers=4))
frozen = {k: np.asarray(frozen[k], dtype=np.float64) for k in common.EMB_KEYS}

CRITERIA = {"eff_rank": "> 0.5x frozen", "mean_dim_std": "> 0.5x frozen",
            "mean_cosine_to_frozen": "> 0.5"}
epochs = {}
for ep in sorted(ckpts):
    ep_dir = WORK / f"lora_ep{ep}"
    stats = {"repack": repack(ep, ep_dir / "lora_bundle")}
    emb = embed_dataset(dataset=tok_guard, model_folder_path=str(ep_dir / "lora_bundle"),
                        **dict(common.EMB_KWARGS, num_workers=4))
    for k in common.EMB_KEYS:
        a_, b_ = frozen[k], np.asarray(emb[k], dtype=np.float64)
        ok = np.isfinite(b_).all()
        cos = float(np.mean((a_ * b_).sum(1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(b_, axis=1))))
        mx = float(np.abs(a_ - b_).max())
        (r0, s0, c0), (r1, s1, c1) = rank_std(a_), rank_std(b_)
        stats[k] = {"finite": bool(ok), "max_abs_delta": mx, "mean_cosine_to_frozen": cos,
                    "eff_rank": [r0, r1], "mean_dim_std": [s0, s1],
                    "mean_cell_cell_cosine": [c0, c1],
                    "pass": {"eff_rank": bool(r1 > 0.5 * r0), "mean_dim_std": bool(s1 > 0.5 * s0),
                             "mean_cosine_to_frozen": bool(cos > 0.5)}}
        print(f"[ep{ep}] {k}: max|d| {mx:.4e} | cos-to-frozen {cos:.4f} | eff rank {r0:.1f}->{r1:.1f} "
              f"| dim std {s0:.4f}->{s1:.4f} | cell-cell cos {c0:.3f}->{c1:.3f}", flush=True)
    stats["passed"] = bool(all(v for k in common.EMB_KEYS for v in stats[k]["pass"].values())
                           and all(stats[k]["finite"] for k in common.EMB_KEYS))
    print(f"[ep{ep}] guard passed={stats['passed']}", flush=True)
    epochs[str(ep)] = stats
    del emb

passing = [e for e in sorted(ckpts) if epochs[str(e)]["passed"]]

# --------------------------------------------------------------------------- #
# 6. epoch_selection.json -- mirrors the CRC queue's format
# --------------------------------------------------------------------------- #
SELECTION.write_text(json.dumps({
    "sid": SID, "adata_path": A.adata_path, "run_dir": str(run_dir),
    "final_epoch": FINAL_EPOCH, "latest_passing_epoch": (max(passing) if passing else None),
    "criteria": CRITERIA, "guard_cells": int(len(GUARD_ROWS)), "n_cells": N_CELLS, "steps_per_epoch": steps_per_epoch,
    "loss_per_epoch": loss_per_epoch, "wall_seconds": wall, "peak_gpu_gib": peak_gib,
    "lr": A.lr, "batch_size": A.batch_size, "epochs": epochs,
    "model_repo": common.MODEL_REPO, "model_dir": str(MODEL_DIR), "px_to_um": A.px_to_um,
}, indent=2))
print(json.dumps({"final_epoch": FINAL_EPOCH,
                  "latest_passing_epoch": (max(passing) if passing else None),
                  "passed": {e: epochs[e]["passed"] for e in epochs},
                  "loss_per_epoch": loss_per_epoch}, indent=2), flush=True)
print("wrote", SELECTION, flush=True)
print("OK", flush=True)
