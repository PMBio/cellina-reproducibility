#!/usr/bin/env python
"""Supervised fine-tune of TERRA block 11 on the dataset's domain label (PIPELINE_SPEC.md).

Generalisation of notebooks/loo_benchmarks/terra/finetune_terra.py: splits now come from
`common.load_dataset` (train on train_idx, validate on val_idx -- the same split cellina uses)
instead of a fresh stratified split, and paths from `common.layout`. Everything else -- the
`init_model` monkeypatch, the load guard, the shuffling fix, the repack and every verification --
is that script verbatim.

  python scripts/terra/finetuning.py --dataset_name crc \
    --adata_path $DATA_ROOT/datasets/crc/raw_zenodo/crc_232.h5ad --holdout_celltype Fibroblast
"""
import argparse
import json
import math
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--dataset_name", required=True, choices=["crc", "merfish"])
p.add_argument("--adata_path", required=True)
p.add_argument("--holdout_celltype", required=True)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--patience", type=int, default=3)
p.add_argument("--batch-size", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max-cells", type=int, default=None, help="subsample the train split (smoke tests only)")
p.add_argument("--work-dir", default=None, help="override layout()['work_dir'] (smoke tests only)")
p.add_argument("--nproc", type=int, default=16)
A = p.parse_args()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TQDM_DISABLE", "1")

import logging

import anndata as ad
import numpy as np
import torch
import yaml
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from terra.training.finetune import finetune
import terra.training.finetune as ft
from terra.utils.helper import init_model, parse_arch_kwargs, parse_protein_init_kwargs
from terra.utils.nested_stratified_group_split import filter_dataset_by_cell_ids

logging.basicConfig(level=logging.WARNING, format="%(message)s")
for name in ("terra.training.finetune", "terra.utils.helper"):
    logging.getLogger(name).setLevel(logging.INFO)

TARGETS = ["11.attn.qkv", "11.attn.proj", "11.mlp.fc1", "11.mlp.fc2"]

paths = common.layout(A.dataset_name, A.adata_path, A.holdout_celltype)
WORK = Path(A.work_dir) if A.work_dir else paths["work_dir"]
RUN_DIR = WORK / "ft_run"
BUNDLE = WORK / "ft_bundle"
if (BUNDLE / "ft_summary.json").exists():
    print(f"[skip] {BUNDLE/'ft_summary.json'} exists", flush=True)
    raise SystemExit(0)
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(common.model_dir())
CFG = yaml.safe_load(open(MODEL_DIR / "model_config.yaml"))
TOKD = pickle.load(open(MODEL_DIR / "token_dictionary.pkl", "rb"))


# --------------------------------------------------------------------------- #
# encoder construction, the embed.py way (checks/check_pretrained_load.py)
# --------------------------------------------------------------------------- #
def build_encoder():
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
# 1. splits -- from common.load_dataset, not a fresh stratified split
# --------------------------------------------------------------------------- #
d = common.load_dataset(A.dataset_name, A.adata_path, A.holdout_celltype, str(MODEL_DIR))
LABEL = d.args["domains_key"]
tok_all = common.tokenize_cached(d.adata_terra, str(MODEL_DIR), paths["tok_cache"], nproc=A.nproc)

obs = d.adata_terra.obs[["cell_id", LABEL]].copy()
obs["cell_id"] = obs["cell_id"].astype(str)
tr_i, va_i = d.train_idx, d.val_idx
if A.max_cells and A.max_cells < len(tr_i):
    # ponytail: smoke-test only -- plain random subsample, no per-class stratification.
    rng = np.random.default_rng(A.seed)
    tr_i = np.sort(rng.choice(tr_i, A.max_cells, replace=False))
    va_i = np.sort(rng.choice(va_i, max(2, A.max_cells // 10), replace=False))
del d           # adata_hvg/adata_terra are not needed past this point

cats = sorted(obs[LABEL].astype(str).unique())
codes = obs[LABEL].astype(str).map({c: i for i, c in enumerate(cats)}).to_numpy()
K = len(cats)


def write_split(idx, path):
    """finetune() reads ONLY obs['cell_id'], obs[label] and uns[f'{label}_num_classes']
    (finetune.py:529,572-583) -- so a 0-var AnnData carrying just those is enough."""
    a = ad.AnnData(obs=obs.iloc[idx][["cell_id"]].assign(**{LABEL: codes[idx]}))
    a.obs_names = a.obs["cell_id"].to_numpy()
    a.uns[f"{LABEL}_num_classes"] = K
    a.write_h5ad(path)
    return a


TRAIN_H5AD, VAL_H5AD = RUN_DIR / "ft_train.h5ad", RUN_DIR / "ft_val.h5ad"
train_adata = write_split(tr_i, TRAIN_H5AD)
val_adata = write_split(va_i, VAL_H5AD)
maj = float(np.bincount(codes[va_i], minlength=K).max() / len(va_i))
print(f"[split] {len(tr_i)} train / {len(va_i)} val | label {LABEL} | classes {cats} -> 0..{K-1} "
      f"| val majority rate {maj:.4f}", flush=True)

# --------------------------------------------------------------------------- #
# 2. tokenized subsets
# --------------------------------------------------------------------------- #
TRAIN_TOK, VAL_TOK = RUN_DIR / "ft_train_tok", RUN_DIR / "ft_val_tok"
fmt = tok_all.format
for idx, out in ((tr_i, TRAIN_TOK), (va_i, VAL_TOK)):
    if out.exists():
        continue
    # ponytail: one temp_dir per call -- filter_dataset_by_cell_ids uses a fixed
    # cache filename and its cleanup misses the num_proc shards, so a shared
    # temp_dir silently serves the previous call's rows.
    tmp = RUN_DIR / f"tokfilter_{out.name}"
    sub = filter_dataset_by_cell_ids(tok_all, obs["cell_id"].to_numpy()[idx].tolist(),
                                     temp_dir=str(tmp))
    assert len(sub) == len(idx), f"{out.name}: {len(sub)} tokenized rows for {len(idx)} cells"
    # finetune.py's non-distributed branch uses DistributedSampler(shuffle=False)
    # (datasets/dataloaders.py:223-229) and the tokenised cells are label-sorted
    # (measured: last 60% of the Fibroblast train order is 100% CRC), so an
    # unshuffled epoch ends with ~500 CRC-only steps and val acc == majority rate.
    # Shuffle once before saving, exactly as finetune_self_supervised.py:97 does.
    sub = sub.shuffle(seed=A.seed).flatten_indices()
    sub.save_to_disk(str(out))
    shutil.rmtree(tmp, ignore_errors=True)
train_dataset, val_dataset = load_from_disk(str(TRAIN_TOK)), load_from_disk(str(VAL_TOK))
for ds, idx in ((train_dataset, tr_i), (val_dataset, va_i)):
    ds.set_format(**fmt)
    assert len(ds) == len(idx), f"stale tokenized subset in {RUN_DIR}: {len(ds)} rows vs {len(idx)} cells"
print(f"[tok] {len(train_dataset)} train / {len(val_dataset)} val rows", flush=True)

# --------------------------------------------------------------------------- #
# 3. patch (plan section 4) -- finetune.py:604 calls init_model with kwargs only
# --------------------------------------------------------------------------- #
_orig_init_model = ft.init_model


def _patched_init_model(**kw):
    kw["nz_spc"] = CFG["data"].get("nz_spc", False)
    kw["mlp_bias"] = CFG["meta"].get("mlp_bias", True)
    kw["n_special_values"] = CFG["data"].get("n_special_values", 0)
    kw["protein_init_kwargs"] = parse_protein_init_kwargs(CFG, TOKD)
    kw.update(parse_arch_kwargs(CFG))
    return _orig_init_model(**kw)


ft.init_model = _patched_init_model

# 4a. pre-flight: the patched construction must strict-load the real checkpoint
PRETRAINED = torch.load(MODEL_DIR / "model_checkpoint.pt", map_location="cpu")["target_encoder"]
PRETRAINED = {k.replace("module.", ""): v for k, v in PRETRAINED.items()}
_enc = build_encoder()
_enc.load_state_dict(PRETRAINED)     # raises on any mismatch
n_total = sum(p.numel() for p in _enc.parameters())
n_train_p = sum(p.numel() for n, p in _enc.named_parameters() if any(t in n for t in TARGETS))
print(f"[preflight] ALL KEYS MATCHED | {n_train_p:,} trainable of {n_total:,}", flush=True)
del _enc


# --------------------------------------------------------------------------- #
# 4b. runtime guard + per-epoch log capture
# --------------------------------------------------------------------------- #
class _LoadGuard(logging.Handler):
    tripped = False

    def emit(self, record):
        if "Encountered exception when loading checkpoint" in record.getMessage():
            _LoadGuard.tripped = True


logging.getLogger("terra.utils.helper").addHandler(_LoadGuard())

epoch_log = []
early = []
_EPOCH_RE = re.compile(r"Epoch \[(\d+)/\d+\].*Train Loss: ([\d.]+), Train Acc: ([\d.]+), "
                       r"Val Loss: ([\d.]+), Val Acc: ([\d.]+)")


class _EpochLog(logging.Handler):
    def emit(self, record):
        m = record.getMessage()
        if m.startswith("Epoch ["):
            e = {"line": m, "t": time.time()}
            g = _EPOCH_RE.search(m)
            if g:
                e.update(epoch=int(g[1]), train_loss=float(g[2]), train_acc=float(g[3]),
                         val_loss=float(g[4]), val_acc=float(g[5]))
            epoch_log.append(e)
        elif m.startswith("Early stopping at epoch"):
            early.append(m)


logging.getLogger("terra.training.finetune").addHandler(_EpochLog())

val_seconds = [0.0]
_orig_validate = ft.validate


def _timed_validate(*a, **kw):
    t0 = time.time()
    out = _orig_validate(*a, **kw)
    val_seconds[0] += time.time() - t0
    return out


ft.validate = _timed_validate

# Start the training clock at the first real step, so setup (encoder build,
# checkpoint load, dataloader spin-up) is excluded from seconds-per-step.
first_step = []
_orig_mask = ft.create_binary_selection_mask


def _timed_mask(*a, **kw):
    if not first_step:
        first_step.append(time.time())
    return _orig_mask(*a, **kw)


ft.create_binary_selection_mask = _timed_mask

# --------------------------------------------------------------------------- #
# 5. fine-tune (plan section 6, hyperparameters verbatim)
# --------------------------------------------------------------------------- #
SAVE_DIR = RUN_DIR / "ft"
args = {
    "model": {"pretrained_checkpoint_path": str(MODEL_DIR),
              "finetune_checkpoint_path": str(SAVE_DIR)},
    "data": {"train_dataset": [str(TRAIN_TOK)], "train_adata": str(TRAIN_H5AD),
             "val_dataset": [str(VAL_TOK)], "val_adata": str(VAL_H5AD),
             "label_name": LABEL, "batch_size": A.batch_size,
             "pin_memory": False, "num_workers": 4},
    "finetune": {
        "use_peft": False,
        "peft_target_modules": TARGETS,
        "use_mlp": False, "hidden_dim": 128,
        "lr": 1e-4, "num_epochs": A.epochs, "patience": A.patience, "save_every_k_epochs": -1,
        "selection_type": "agg_cell",
        "excluded_tokens": None, "top_k": None,
    },
}

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
_, _, accuracy = finetune(args=args, train_adata=train_adata, train_dataset=train_dataset,
                          val_adata=val_adata, val_dataset=val_dataset,
                          save_folder_path=str(SAVE_DIR))
wall = time.time() - t0
peak_gb = torch.cuda.max_memory_allocated() / 2**30

assert not _LoadGuard.tripped, "pretrained weights failed to load -- encoder was RANDOM"
print(f"[guard] _LoadGuard.tripped = {_LoadGuard.tripped}", flush=True)

steps_per_epoch = math.ceil(len(tr_i) / A.batch_size)
train_seconds = (epoch_log[-1]["t"] - first_step[0]) - val_seconds[0] if epoch_log else float("nan")
sec_per_step = train_seconds / (len(epoch_log) * steps_per_epoch) if epoch_log else float("nan")

# --------------------------------------------------------------------------- #
# 6. repack (plan section 5) + verification
# --------------------------------------------------------------------------- #
run = sorted(SAVE_DIR.iterdir())[-1]
ckpt = run / "checkpoint_best.pt" if (run / "checkpoint_best.pt").exists() else run / "checkpoint_last.pt"
if BUNDLE.exists():
    shutil.rmtree(BUNDLE)
shutil.copytree(MODEL_DIR, BUNDLE)
sd = torch.load(ckpt, map_location="cpu")["model"]
enc_sd = {k[len("base_model."):]: v for k, v in sd.items() if k.startswith("base_model.")}
assert enc_sd, "no base_model.* keys -- checkpoint format changed"
torch.save({"target_encoder": enc_sd, "epoch": 0}, BUNDLE / "model_checkpoint.pt")
print(f"[repack] {ckpt} -> {BUNDLE}", flush=True)

build_encoder().load_state_dict(enc_sd)     # strict=True; raises on any mismatch
print("[repack] strict load into an embed.py-built encoder: ALL KEYS MATCHED", flush=True)

assert set(enc_sd) == set(PRETRAINED), "repacked key set differs from pretrained"
changed = [k for k in PRETRAINED if not torch.equal(PRETRAINED[k], enc_sd[k].cpu())]
unexpected = [k for k in changed if not any(t in k for t in TARGETS)]
assert not unexpected, f"non-block-11 tensors changed: {unexpected[:5]}"
assert changed, "nothing changed -- fine-tuning had no effect"
print(f"[diff] {len(changed)}/{len(PRETRAINED)} tensors changed, all in block 11: "
      f"{sorted(changed)}", flush=True)

# --------------------------------------------------------------------------- #
# 7. summary
# --------------------------------------------------------------------------- #
summary = {
    "dataset_name": A.dataset_name, "adata_path": A.adata_path, "sid": paths["sid"],
    "holdout_celltype": A.holdout_celltype, "label": LABEL,
    "epochs_requested": A.epochs, "patience": A.patience,
    "epochs_run": len(epoch_log),
    "stopped_early": bool(early) or len(epoch_log) < A.epochs,
    "early_stopping_line": early[0] if early else None,
    "val_loss_per_epoch": [e.get("val_loss") for e in epoch_log],
    "val_accuracy_per_epoch": [e.get("val_acc") for e in epoch_log],
    "epoch_lines": [e["line"] for e in epoch_log],
    "final_val_accuracy": accuracy,
    "val_majority_class_rate": maj,
    "n_train": len(tr_i), "n_val": len(va_i),
    "classes": cats, "num_classes": K,
    "trainable_params": n_train_p, "total_params": n_total,
    "wall_seconds": wall, "train_seconds": train_seconds,
    "val_seconds": val_seconds[0],
    "steps_per_epoch": steps_per_epoch,
    "sec_per_step": sec_per_step,
    "peak_gpu_gib": peak_gb,
    "checkpoint": str(ckpt),
    "changed_tensors": sorted(changed),
    "val_accuracy_above_majority": bool(accuracy > maj),
    "args": args,
}
(BUNDLE / "ft_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k not in ("args", "changed_tensors", "epoch_lines")},
                 indent=2), flush=True)
# Plan check 9. A --max-cells run is a smoke test, too small/short to clear it.
if A.max_cells:
    print(f"[check9] SKIPPED (smoke run): val acc {accuracy:.4f} vs majority {maj:.4f}", flush=True)
else:
    assert accuracy > maj, f"val accuracy {accuracy:.4f} <= majority rate {maj:.4f}"
print("OK", flush=True)
