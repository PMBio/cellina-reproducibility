#!/usr/bin/env python
"""Supervised fine-tuning arm for the TERRA node-perturbation benchmark.

Implements FINETUNE_PLAN.md sections 4-6: patch the encoder reconstruction in
`terra.training.finetune`, fine-tune block 11 on a domain label, repack the
supervised checkpoint into a bundle `embed_dataset` can read, and verify all of it.

Nothing under the installed `terra` package is modified -- section 4's fix is a
monkeypatch applied here, in-process, before `finetune()` is called.
"""
import argparse
import json
import math
import os
import pickle
import shutil
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model-dir", required=True, help="pretrained TERRA bundle dir")
p.add_argument("--tok-cache", required=True, help="tokenized HF dataset dir (all cells)")
p.add_argument("--pool-h5ad", required=True, help="h5ad of fine-tuning-eligible cells")
p.add_argument("--run-dir", required=True, help="fine-tune outputs + split h5ads")
p.add_argument("--bundle-dir", required=True, help="repacked embeddable bundle")
p.add_argument("--label", default="typ")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch-size", type=int, default=100)
p.add_argument("--val-frac", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max-cells", type=int, default=None, help="subsample the pool (smoke tests only)")
A = p.parse_args()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import h5py
import numpy as np
import torch
import yaml
import anndata as ad
import logging
from datasets import load_from_disk
from sklearn.model_selection import train_test_split

from terra.training.finetune import finetune
import terra.training.finetune as ft
from terra.utils.helper import init_model, parse_arch_kwargs, parse_protein_init_kwargs
from terra.utils.nested_stratified_group_split import filter_dataset_by_cell_ids

MODEL_DIR = Path(A.model_dir)
RUN_DIR = Path(A.run_dir); RUN_DIR.mkdir(parents=True, exist_ok=True)
TARGETS = ["11.attn.qkv", "11.attn.proj", "11.mlp.fc1", "11.mlp.fc2"]

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
# 1. splits
# --------------------------------------------------------------------------- #
with h5py.File(A.pool_h5ad, "r") as f:          # obs only -- the 112M obsm is ignored
    obs = ad.io.read_elem(f["obs"])
obs = obs[["cell_id", A.label]].copy()
obs["cell_id"] = obs["cell_id"].astype(str)

if A.max_cells and A.max_cells < len(obs):
    obs = obs.sample(A.max_cells, random_state=A.seed)

cats = sorted(obs[A.label].astype(str).unique())
codes = obs[A.label].astype(str).map({c: i for i, c in enumerate(cats)}).to_numpy()
K = len(cats)

tr_i, va_i = train_test_split(np.arange(len(obs)), test_size=A.val_frac,
                              stratify=codes, random_state=A.seed)


def write_split(idx, path):
    """finetune() reads ONLY obs['cell_id'], obs[label] and uns[f'{label}_num_classes']
    (finetune.py:529,572-583) -- so a 0-var AnnData carrying just those is enough."""
    a = ad.AnnData(obs=obs.iloc[idx][["cell_id"]].assign(**{A.label: codes[idx]}))
    a.obs_names = a.obs["cell_id"].to_numpy()
    a.uns[f"{A.label}_num_classes"] = K
    a.write_h5ad(path)
    return a


TRAIN_H5AD, VAL_H5AD = RUN_DIR / "ft_train.h5ad", RUN_DIR / "ft_val.h5ad"
train_adata = write_split(tr_i, TRAIN_H5AD)
val_adata = write_split(va_i, VAL_H5AD)
maj = float(np.bincount(codes[va_i], minlength=K).max() / len(va_i))
print(f"[split] {len(tr_i)} train / {len(va_i)} val | classes {cats} -> 0..{K-1} "
      f"| val majority rate {maj:.4f}", flush=True)

# --------------------------------------------------------------------------- #
# 2. tokenized subsets
# --------------------------------------------------------------------------- #
TRAIN_TOK, VAL_TOK = RUN_DIR / "ft_train_tok", RUN_DIR / "ft_val_tok"
full = load_from_disk(A.tok_cache)
fmt = full.format
for idx, out in ((tr_i, TRAIN_TOK), (va_i, VAL_TOK)):
    if out.exists():
        continue
    # ponytail: one temp_dir per call -- filter_dataset_by_cell_ids uses a fixed
    # cache filename and its cleanup misses the num_proc shards, so a shared
    # temp_dir silently serves the previous call's rows.
    tmp = RUN_DIR / f"tokfilter_{out.name}"
    sub = filter_dataset_by_cell_ids(full, obs["cell_id"].to_numpy()[idx].tolist(),
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
for d, idx in ((train_dataset, tr_i), (val_dataset, va_i)):
    d.set_format(**fmt)
    assert len(d) == len(idx), f"stale tokenized subset in {RUN_DIR}: {len(d)} rows vs {len(idx)} cells"
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
logging.basicConfig(level=logging.INFO, format="%(message)s")

epoch_log = []


class _EpochLog(logging.Handler):
    def emit(self, record):
        m = record.getMessage()
        if m.startswith("Epoch ["):
            epoch_log.append({"line": m, "t": time.time()})


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
             "label_name": A.label, "batch_size": A.batch_size,
             "pin_memory": False, "num_workers": 4},
    "finetune": {
        "use_peft": False,
        "peft_target_modules": TARGETS,
        "use_mlp": False, "hidden_dim": 128,
        "lr": 1e-4, "num_epochs": A.epochs, "patience": 10, "save_every_k_epochs": -1,
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
BUNDLE = Path(A.bundle_dir)
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
    "val_accuracy_per_epoch": [e["line"] for e in epoch_log],
    "final_val_accuracy": accuracy,
    "val_majority_class_rate": maj,
    "n_train": len(tr_i), "n_val": len(va_i),
    "classes": cats, "num_classes": K,
    "trainable_params": n_train_p, "total_params": n_total,
    "wall_seconds": wall, "train_seconds": train_seconds,
    "val_seconds": val_seconds[0],
    "steps_per_epoch": steps_per_epoch,
    "seconds_per_step": sec_per_step,
    "peak_gpu_gib": peak_gb,
    "epochs_run": len(epoch_log),
    "checkpoint": str(ckpt),
    "changed_tensors": sorted(changed),
    "val_accuracy_above_majority": bool(accuracy > maj),
    "args": args,
}
(BUNDLE / "ft_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
# Plan check 9. A --max-cells run is a smoke test, too small/short to clear it.
if A.max_cells:
    print(f"[check9] SKIPPED (smoke run): val acc {accuracy:.4f} vs majority {maj:.4f}", flush=True)
else:
    assert accuracy > maj, f"val accuracy {accuracy:.4f} <= majority rate {maj:.4f}"
print("OK", flush=True)
