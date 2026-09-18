"""TERRA-native perturbation evaluation (scripts/terra/TERRA_OG_EVAL_SPEC.md).

One job = one slide x one model arm x one perturbation target.  It embeds the control set
and the perturbed set once, then slices every scored cell type from that pass:

    python scripts/terra/terra_og_eval.py --sid crc_232 --arm frozen --target neighb_only
    python scripts/terra/terra_og_eval.py --sid crc_232 --arm lora --epoch 5 --target ct_neigh
    python scripts/terra/terra_og_eval.py --sid crc_232 --arm frozen --target neighb_only \
        --random-seeds 0,1,2,3,4          # the random-gene control (frozen arm only)

Mechanism: terra.inference.perturb_dataset, perturbation_type="foldchange",
foldchange = exp(logfc_global) per gene (spec §6, §15).  Metrics: precision@50 / Spearman
from per-gene Sinkhorn W2 (spec §8) and population W2 by label (spec §9).

Outputs (spec §12, Q11): JSON/CSV/README under results/terra_og/, arrow caches under
$DATA_ROOT/datasets/crc/{sid}/terra_og_cache/.  Nothing already on disk is read-modified.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # before torch; terra hardcodes cuda:0

import argparse
import json
import logging
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from datasets import load_from_disk
from geomloss import SamplesLoss
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                   # noqa: E402
from counterfactual_analysis import precision                   # noqa: E402
from train_loo import preprocess_crc                            # noqa: E402
import terra                                                    # noqa: E402
from terra.inference import embed_dataset, perturb_dataset      # noqa: E402
from terra.inference.summarize_distance import summarize_w2_by_label   # noqa: E402

REPO = Path(__file__).resolve().parents[2]
try:
    from importlib.metadata import version as _pkg_version
    TERRA_VERSION = _pkg_version("terra-st")
except Exception:  # noqa: BLE001
    TERRA_VERSION = getattr(terra, "__version__", "unknown")
RESULTS = REPO / "results" / "terra_og"
SCORED_CTS = ["Endothelial", "Epithelial", "Fibroblast", "Myeloid", "T_cell"]
TARGETS = {"neighb_only": ["neighborhood"], "ct_neigh": ["cell", "neighborhood"]}
N_PERT_GENES = 50
K = 50
MIN_CELLS = 20          # token-slot occupancy in the index-cell block
BLUR = 0.01             # package default (Q16)
SEQ_LEN_CELL = 256
EMB_CHUNK = 2000        # cells per embed_dataset call: token_emb is 4.3 MB/cell fp32 on host
EMB_KWARGS = dict(emb_layer=None, agg_excluded_genes=None, top_k=None, batch_size=128,
                  include_spatial_cell_emb=True, return_token_embeddings=True,
                  ignore_spc_tokens=True, num_workers=8)
N_PREV_CANDIDATES = 20  # prevalence-matched random partner: nearest-20 by REF slot prevalence
HOLDOUT_CT_PIN = "Endothelial"   # only selects common.layout's cache dir; never subsets rows


# --------------------------------------------------------------------------- logging capture
class _TerraLogCapture(logging.Handler):
    """Records terra's INFO lines so the checkpoint-load message can be asserted (A1b)."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


LOGCAP = _TerraLogCapture()
logging.getLogger("terra").addHandler(LOGCAP)
logging.getLogger("terra").setLevel(logging.INFO)


def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:            # noqa: BLE001
        return "unknown"


def _save_atomic(ds, path):
    """save_to_disk into a private temp dir, then rename into place (first writer wins).
    Same contract as eval_terra._save_atomic; copied so this script does not import the
    superseded shift track."""
    if path.exists():
        print(f"{path} already written by another job; keeping it")
        return
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    ds.save_to_disk(str(tmp))
    try:
        os.rename(tmp, path)
        print(f"wrote {path}")
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"{path} appeared while writing; kept the other job's copy")


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _write_csv(df, path, index=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- data (§2, §3, §4)
def load_expression(sid, args):
    """The 2,000-gene terra2k object with coarse_type / typ_clean, exactly as
    common.load_dataset(universe="terra2k") builds adata_hvg -- without the 18,936-gene
    harmonised twin, which this pipeline never reads (perturb_dataset needs no coordinates)."""
    h5, gfile = common.terra2k_path(sid)
    genes = [g for g in gfile.read_text().split() if g]
    src = sc.read_h5ad(h5)
    src.obs_names_make_unique()
    assert list(src.var_names.astype(str)) == genes, f"{h5} is not on the {gfile} gene axis"
    a = preprocess_crc(src, n_top_genes=len(genes),
                       labels_key=args["labels_key"], domains_key=args["domains_key"])
    assert a.n_vars == len(genes), f"preprocess dropped {len(genes) - a.n_vars} terra2k genes"
    a.X = a.layers["counts"].copy()
    a.obs["cell_id"] = a.obs_names.astype(str)
    return a


def pseudobulk_logfc(adata, mask_target, mask_control):
    """Sum counts -> CP10K -> log1p, then difference (spec §3). Natural log."""
    def prof(m):
        s = np.asarray(adata.X[m].sum(axis=0)).ravel().astype(float)
        return np.log1p(s / s.sum() * 1e4)
    return pd.Series(prof(mask_target) - prof(mask_control), index=adata.var_names.astype(str))


def logfc_table(adata, args, cache_csv):
    dom, lab = args["domains_key"], args["labels_key"]
    is_crc = (adata.obs[dom].astype(str) == args["holdout_domains"][0]).to_numpy()
    is_ref = adata.obs[dom].astype(str).isin(args["control_domains"]).to_numpy()
    assert is_crc.sum() and is_ref.sum(), f"empty CRC/REF masks on {dom}"
    cols = {"global": pseudobulk_logfc(adata, is_crc, is_ref)}
    for ct in SCORED_CTS:
        is_ct = (adata.obs[lab].astype(str) == ct).to_numpy()
        cols[ct] = pseudobulk_logfc(adata, is_crc & is_ct, is_ref & is_ct)
    df = pd.DataFrame(cols)
    df.index.name = "gene"
    _write_csv(df, cache_csv)
    return df, is_crc, is_ref


def select_genes(adata, logfc, token_dict, cache_csv):
    ens = adata.var["ensembl_id"].astype(str)
    ens.index = ens.index.astype(str)
    has_token = ens.map(lambda e: e in token_dict)
    n_no_token = int((~has_token).sum())
    pert = logfc["global"][has_token.to_numpy()].abs().nlargest(N_PERT_GENES).index
    df = pd.DataFrame({
        "gene": pert,
        "ensembl_id": ens[pert].to_numpy(),
        "token": [token_dict[e] for e in ens[pert]],
        "logfc_global": logfc.loc[pert, "global"].to_numpy(),
    })
    df["foldchange"] = np.exp(df["logfc_global"])
    _write_csv(df, cache_csv, index=False)
    return df, ens, has_token, n_no_token


def make_perturb_df(gene_df, targets):
    return pd.DataFrame({
        "perturbed_cell_id": "all",
        "perturbed_ensembl_id": np.repeat(gene_df["ensembl_id"].to_numpy(), len(targets)),
        "perturbation_target": list(targets) * len(gene_df),
        "perturbation_type": "foldchange",
        "foldchange": np.repeat(gene_df["foldchange"].to_numpy(), len(targets)),
    })


def build_perturbed(control, gene_df, targets, model_dir, cache_dir):
    if cache_dir.exists():
        ds = load_from_disk(str(cache_dir))
        print(f"loaded cached {cache_dir}")
        return ds
    t0 = time.time()
    # keep_in_memory: otherwise HF `map` drops cache-*.arrow files into the *shared* control/
    # directory, and concurrent jobs of one slide mmap each other's half-written files
    # (SIGBUS / EFAULT on save_to_disk, seen 2026-09-17 on crc_120/221/231/242).
    ds = perturb_dataset(dataset=control, perturb_df=make_perturb_df(gene_df, targets),
                         model_folder_path=str(model_dir), nproc=1, keep_in_memory=True,
                         return_only_perturbed_cells=False, pad_gene_tokens=True,
                         adjust_positions=False)
    print(f"perturb_dataset({targets}) on {len(control):,} cells: {time.time() - t0:.0f}s")
    _save_atomic(ds, cache_dir)
    return load_from_disk(str(cache_dir)) if cache_dir.exists() else ds


def random_gene_set(seed, gene_df, universe_tokens, prevalence, tok2gene, ens, cache_csv):
    """Prevalence-matched partner per perturbed gene: drawn from its N_PREV_CANDIDATES nearest
    non-perturbed universe genes by REF index-cell slot prevalence; inherits the foldchange."""
    rng = np.random.default_rng(seed)
    pert_tokens = set(gene_df["token"])
    pool = np.array([t for t in universe_tokens if t not in pert_tokens])
    pool_prev = prevalence[pool]
    taken, rows = set(), []
    for _, r in gene_df.iterrows():
        d = np.abs(pool_prev - prevalence[r["token"]])
        order = np.argsort(d, kind="stable")
        cands = [pool[i] for i in order if pool[i] not in taken][:N_PREV_CANDIDATES]
        t = int(rng.choice(cands))
        taken.add(t)
        g = tok2gene[t]
        rows.append(dict(gene=g, ensembl_id=ens[g], token=t, partner_of=r["gene"],
                         logfc_global=r["logfc_global"], foldchange=r["foldchange"],
                         prevalence=prevalence[t], partner_prevalence=prevalence[r["token"]]))
    df = pd.DataFrame(rows)
    _write_csv(df, cache_csv, index=False)
    return df


# --------------------------------------------------------------------------- weights (§7, A1)
def _target_encoder_state(bundle):
    ck = torch.load(Path(bundle) / "model_checkpoint.pt", map_location="cpu", weights_only=False)
    return {k.replace("module.", ""): v for k, v in ck["target_encoder"].items()}


def assert_weights(arm, bundle, frozen_bundle):
    te = _target_encoder_state(bundle)
    bad = [k for k in te if "lora_" in k or k.startswith("base_model.") or "base_layer" in k]
    assert not bad, f"{arm}: LoRA/PEFT-prefixed keys in checkpoint, e.g. {bad[:3]}"
    fr = _target_encoder_state(frozen_bundle)
    assert set(te) == set(fr), "architecture mismatch vs pretrained (key sets differ)"
    n_diff = sum(1 for k in fr if fr[k].shape != te[k].shape
                 or not torch.equal(fr[k].float(), te[k].float()))
    qkv = [k for k in fr if "qkv" in k and k.endswith("weight")]
    num = sum(float((te[k].float() - fr[k].float()).abs().mean()) for k in qkv)
    den = sum(float(fr[k].float().abs().mean()) for k in qkv)
    rel = num / den if den else float("nan")
    if arm == "terra-frozen":
        assert n_diff == 0, f"frozen bundle differs from HF snapshot in {n_diff} tensors"
    else:
        assert n_diff > 0, f"{arm} is bit-identical to pretrained -- bundle did not train"
    print(f"[A1] {arm}: {n_diff}/{len(fr)} tensors differ from pretrained, qkv rel delta {rel:.3f}")
    return dict(n_tensors_differing_from_pretrained=int(n_diff), n_tensors=len(fr),
                qkv_rel_delta=rel)


def assert_runtime_load():
    """A1b: terra's load_checkpoint swallows every exception; the INFO line is the only proof."""
    ok = [l for l in LOGCAP.lines if "Loaded pretrained target encoder" in l
          and "All keys matched successfully" in l]
    bad = [l for l in LOGCAP.lines if "Encountered exception when loading checkpoint" in l]
    assert ok and not bad, f"checkpoint did not load cleanly: {bad or LOGCAP.lines[-5:]}"
    return True


# --------------------------------------------------------------------------- embedding
def embed_all(bundle, ds):
    """One pass: the index cell's own token embeddings under full neighbourhood attention,
    (n_cells, 256, 384) fp32 L2-normalised per row, held on the HOST (17 GB for a 44k-cell
    slide; two of them do not fit a 40 GB GPU next to the model), plus the three pooled
    embeddings (for summarize_w2_by_label).  score_dataset moves one cell type at a time to the GPU."""
    pooled = {k: [] for k in ("cell_emb", "spatial_cell_emb", "neighborhood_emb")}
    cloud = None   # preallocated on the first chunk: torch.cat would briefly double the 17 GB
                   # cloud of a 44k-cell slide (CUDA OOM on 44 GB L40s, crc_231, 2026-09-17)
    for i in range(0, len(ds), EMB_CHUNK):
        out = embed_dataset(dataset=ds.select(range(i, min(i + EMB_CHUNK, len(ds)))),
                            **dict(EMB_KWARGS, model_folder_path=str(bundle)))
        te = torch.from_numpy(np.ascontiguousarray(out["token_emb"][:, :SEQ_LEN_CELL])) \
                  .to("cuda:0", torch.float32)
        if cloud is None:
            cloud = torch.empty((len(ds), te.shape[1], te.shape[2]), dtype=torch.float32)
        cloud[i:i + len(te)] = (te / (te.norm(dim=-1, keepdim=True) + 1e-12)).cpu()
        del te
        for k in pooled:
            pooled[k].append(np.asarray(out[k], dtype=np.float32))
        del out
    return cloud, {k: np.concatenate(v, 0) for k, v in pooled.items()}


# --------------------------------------------------------------------------- metric 1 (§8)
def per_gene_w2(loss_fn, tokens_ct, base_ct, other_ct, utok):
    """tokens_ct (n, 256) int; base/other (n, 256, 384) on GPU -> W2 per token in utok."""
    flat = tokens_ct.ravel()
    order = np.argsort(flat, kind="stable")
    lo = np.searchsorted(flat[order], utok, "left")
    hi = np.searchsorted(flat[order], utok, "right")
    pos = torch.from_numpy(order).to("cuda:0")
    b = base_ct.reshape(-1, base_ct.shape[-1])
    o = other_ct.reshape(-1, other_ct.shape[-1])
    w2 = np.full(len(utok), np.nan)
    for i in range(len(utok)):
        if hi[i] > lo[i]:
            idx = pos[lo[i]:hi[i]]
            w2[i] = float(loss_fn(b[idx], o[idx]))
    self_w2 = float(loss_fn(b[pos[lo[0]:hi[0]]], b[pos[lo[0]:hi[0]]]))
    return w2, hi - lo, self_w2


def score_universe(truth, pred, k=K):
    m = np.isfinite(pred) & np.isfinite(truth)
    t, p = truth[m], pred[m]
    n = int(m.sum())
    return dict(n_genes=n, chance=k / n, precision=float(precision(t, p, k=k)),
                spearman=float(spearmanr(p, t).statistic))


# --------------------------------------------------------------------------- dose (§15, A8)
def dose_stats(control, perturbed):
    ec = np.asarray(control.with_format(None)["gene_expr"], dtype=np.float32)
    ep = np.asarray(perturbed.with_format(None)["gene_expr"], dtype=np.float32)
    d = np.abs(ep - ec)
    edited = d > 0
    return dict(
        gene_expr_max_control=float(ec.max()),
        gene_expr_max_perturbed=float(ep.max()),
        dose_per_cell=float(d.sum(1).mean()),
        edited_slots_cell_block_per_cell=float(edited[:, :SEQ_LEN_CELL].sum(1).mean()),
        edited_slots_neigh_block_per_cell=float(edited[:, SEQ_LEN_CELL:].sum(1).mean()),
        frac_cells_with_cell_block_edit=float(edited[:, :SEQ_LEN_CELL].any(1).mean()),
        frac_cells_with_neigh_block_edit=float(edited[:, SEQ_LEN_CELL:].any(1).mean()),
    ), ec, ep


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sid", default="crc_232")
    ap.add_argument("--arm", choices=["frozen", "lora"], required=True)
    ap.add_argument("--epoch", type=int, default=None, help="LoRA epoch (required for --arm lora)")
    ap.add_argument("--target", choices=list(TARGETS), required=True)
    ap.add_argument("--cell-types", default=",".join(SCORED_CTS))
    ap.add_argument("--random-seeds", default="", help="e.g. 0,1,2,3,4: random-gene control (frozen only)")
    ap.add_argument("--results", default=str(RESULTS))
    a = ap.parse_args()

    t_start = time.time()
    cts = [c for c in a.cell_types.split(",") if c]
    seeds = [int(s) for s in a.random_seeds.split(",") if s != ""]
    if a.arm == "lora":
        assert a.epoch is not None, "--epoch required for --arm lora"
        arm = f"terra-lora-ep{a.epoch}"
    else:
        arm = "terra-frozen"
    assert not (seeds and a.arm != "frozen"), "random-gene controls run on the frozen arm only (Q17)"
    targets = TARGETS[a.target]
    results = Path(a.results)
    cache = results / "cache"
    per_run = results / "per_run"
    slide_dir = Path(common.DATA_ROOT) / "datasets" / "crc" / a.sid
    heavy = slide_dir / "terra_og_cache"                     # arrow caches (Q11)
    heavy.mkdir(parents=True, exist_ok=True)
    frozen_bundle = Path(common.model_dir())
    bundle = frozen_bundle if a.arm == "frozen" else slide_dir / "terra" / f"lora_ep{a.epoch}" / "lora_bundle"
    assert (bundle / "model_checkpoint.pt").exists(), f"missing bundle {bundle}"
    print(f"== {a.sid} {arm} target={a.target} cts={cts} seeds={seeds} bundle={bundle}")
    checks = {}

    # A1: weights ----------------------------------------------------------------------------
    weights = assert_weights(arm, bundle, frozen_bundle)
    checks["A1_weights_differ_as_expected"] = True

    # data -----------------------------------------------------------------------------------
    args = common.dataset_args("crc")
    adata = load_expression(a.sid, args)
    tok = load_from_disk(str(slide_dir / "terra_tok_terra2k"))
    tok_ids = np.array(tok.with_format(None)["cell_id"]).astype(str)
    n_before = adata.n_obs
    adata = adata[adata.obs["cell_id"].isin(set(tok_ids))].copy()
    n_tok_not_in_adata = int(len(set(tok_ids) - set(adata.obs["cell_id"])))
    print(f"[A7] adata {n_before:,} -> {adata.n_obs:,} cells after restricting to the tokenized "
          f"set of {len(tok_ids):,}; {n_tok_not_in_adata} tokenized cells missing from adata")
    checks["A7_population_identical"] = bool(adata.n_obs == len(tok_ids))
    assert checks["A7_population_identical"], "logFC and perturbation refer to different populations"

    logfc, is_crc, is_ref = logfc_table(adata, args, cache / f"logfc_{a.sid}.csv")
    with open(frozen_bundle / "token_dictionary.pkl", "rb") as f:
        token_dict = pickle.load(f)
    gene_df, ens, has_token, n_no_token = select_genes(adata, logfc, token_dict,
                                                       cache / f"perturbed_genes_{a.sid}.csv")
    print(f"perturbed genes: {len(gene_df)} | foldchange {gene_df.foldchange.min():.2f}-"
          f"{gene_df.foldchange.max():.2f} | {n_no_token} terra2k genes without a token")

    # control population (§5) ----------------------------------------------------------------
    lab = args["labels_key"]
    pos_of = {c: i for i, c in enumerate(tok_ids)}
    is_pop = is_ref & adata.obs[lab].astype(str).isin(SCORED_CTS).to_numpy()
    pop_rows = np.sort([pos_of[c] for c in adata.obs.loc[is_pop, "cell_id"]])
    control_dir = heavy / "control"
    if control_dir.exists():
        control = load_from_disk(str(control_dir))
    else:
        _save_atomic(tok.select([int(i) for i in pop_rows]), control_dir)
        control = load_from_disk(str(control_dir))
    ctrl_ids = np.array(control.with_format(None)["cell_id"]).astype(str)
    assert np.array_equal(ctrl_ids, tok_ids[pop_rows]), "control cache holds a different cell set"
    obs = adata.obs.set_index("cell_id").loc[ctrl_ids]
    ct_of_cell = obs[lab].astype(str).to_numpy()
    counts_ct = pd.Series(ct_of_cell).value_counts()
    n_ist_fib = int(((adata.obs["ist"].astype(str).isin(["fib1", "fib2"])) & is_ref).sum()) \
        if "ist" in adata.obs else -1
    print(f"control population: {len(control):,} REF cells | per type: {counts_ct.to_dict()} | "
          f"REF ist in {{fib1,fib2}}: {n_ist_fib} (TERRA notebook: 2,881)")

    # perturbed datasets (§6) ----------------------------------------------------------------
    perturbed = build_perturbed(control, gene_df, targets, frozen_bundle, heavy / f"pert_{a.target}")
    tokens_c = np.asarray(control.with_format(None)["gene_tokens"])
    tokens_p = np.asarray(perturbed.with_format(None)["gene_tokens"])
    checks["A2_tokens_unchanged"] = bool(np.array_equal(tokens_c, tokens_p))
    assert checks["A2_tokens_unchanged"], "foldchange moved a gene token"
    checks["A5_cell_id_order"] = bool(np.array_equal(
        ctrl_ids, np.array(perturbed.with_format(None)["cell_id"]).astype(str)))
    assert checks["A5_cell_id_order"], "cell_id order differs between control and perturbed"
    dose, ec, ep = dose_stats(control, perturbed)
    if a.target == "neighb_only":
        checks["A3_cell_block_untouched"] = bool(np.array_equal(ec[:, :SEQ_LEN_CELL], ep[:, :SEQ_LEN_CELL]))
        assert checks["A3_cell_block_untouched"], "neighb_only edited the cell block"
    del ec, ep
    print(f"[A8] dose: {dose}")

    tokens = tokens_c[:, :SEQ_LEN_CELL]
    n_per_tok = np.bincount(tokens.ravel(), minlength=max(token_dict.values()) + 1)
    n_per_tok[0] = 0
    prevalence = n_per_tok / len(control)
    tok2gene = {token_dict[e]: g for g, e in ens.items() if e in token_dict}
    universe_all = np.array(sorted(t for t in np.flatnonzero(n_per_tok >= MIN_CELLS) if t in tok2gene))
    print(f"universe A (token + >= {MIN_CELLS} slots in the whole control population): {len(universe_all)} genes")

    # random-gene sets (§10) -----------------------------------------------------------------
    rand, rand_tokens = {}, {}
    for s in seeds:
        rdf = random_gene_set(s, gene_df, universe_all, prevalence, tok2gene, ens,
                              cache / f"random_genes_{a.sid}_seed{s}.csv")
        rand_tokens[s] = set(rdf["token"])
        rand[s] = build_perturbed(control, rdf, targets, frozen_bundle,
                                  heavy / f"pert_{a.target}_random_seed{s}")
        assert np.array_equal(tokens_c, np.asarray(rand[s].with_format(None)["gene_tokens"]))
    del tokens_p

    # embeddings + scoring -----------------------------------------------------------------
    # One dataset at a time on the GPU next to the control cloud: (n_cells, 256, 384) fp32 is
    # ~0.4 GB per 1,000 cells, so control + one perturbed set stays under 25 GB for this slide.
    LOGCAP.lines.clear()
    base, pooled_c = embed_all(bundle, control)
    checks["A1b_runtime_checkpoint_load"] = assert_runtime_load()
    base3 = base.view(len(control), SEQ_LEN_CELL, -1)
    loss_fn = SamplesLoss("sinkhorn", p=2, blur=BLUR, backend="tensorized")
    pert_tokens = set(gene_df["token"])

    def population_w2(pp):
        ao = ad.AnnData(obs=pd.DataFrame({lab: pd.Categorical(ct_of_cell)}, index=ctrl_ids))
        for k in pooled_c:
            ao.obsm[k] = pooled_c[k]
            ao.obsm[f"{k}_perturb_case1"] = pp[k]
        df = summarize_w2_by_label(ao, label_key=lab, blur=BLUR, backend="tensorized")
        return df.set_index("label")

    ct_rows = {ct: np.flatnonzero(ct_of_cell == ct) for ct in cts}
    ct_universe = {}
    for ct, rows in ct_rows.items():
        n_ct = np.bincount(tokens[rows].ravel(), minlength=len(n_per_tok))
        n_ct[0] = 0
        utok = np.array([t for t in universe_all if n_ct[t] >= MIN_CELLS])
        ct_universe[ct] = utok

    def score_dataset(cloud, pooled, edited_tokens):
        """-> (pop W2 table, {ct: (w2, n_occ, self_w2, res_uni)}) for one perturbed dataset.
        `edited_tokens` are the genes edited in *this* dataset (the 50 CRC genes, or a random
        set's 50 partners), so universe `minus_perturbed` always removes the dataset's own edits."""
        pop_df = population_w2(pooled)
        cloud3 = cloud.view(len(control), SEQ_LEN_CELL, -1)
        per_ct = {}
        for ct, rows in ct_rows.items():
            if len(rows) < MIN_CELLS:
                print(f"SKIP {ct}: {len(rows)} cells < {MIN_CELLS}")
                continue
            ridx = torch.from_numpy(rows)
            utok = ct_universe[ct]
            genes = np.array([tok2gene[t] for t in utok])
            truth = logfc.loc[genes, ct].abs().to_numpy()
            b_ct, o_ct = base3[ridx].to("cuda:0"), cloud3[ridx].to("cuda:0")
            w2, n_occ, self_w2 = per_gene_w2(loss_fn, tokens[rows], b_ct, o_ct, utok)
            del b_ct, o_ct
            torch.cuda.empty_cache()
            is_pert = np.array([t in edited_tokens for t in utok])
            uni = {"all": np.ones(len(utok), bool), "minus_perturbed": ~is_pert}
            res_uni = {u: score_universe(truth[m], w2[m]) for u, m in uni.items()}
            per_ct[ct] = dict(w2=w2, n_occ=n_occ, self_w2=self_w2, res=res_uni, genes=genes,
                              truth=truth, is_pert=is_pert)
        return pop_df, per_ct

    other, pooled_p = embed_all(bundle, perturbed)
    pop, main_ct = score_dataset(other, pooled_p, pert_tokens)
    del other
    torch.cuda.empty_cache()
    if a.target == "neighb_only":
        checks["A4_cell_emb_w2_zero"] = bool((pop["cell_emb_w2"].abs() < 1e-6).all())
        assert checks["A4_cell_emb_w2_zero"], f"cell_emb_w2 nonzero under neighb_only:\n{pop['cell_emb_w2']}"
    print(pop)
    print(f"embedded + scored control and perturbed: {time.time() - t_start:.0f}s")

    pop_rand, rand_ct = {}, {}
    for s, ds in rand.items():
        cloud, pooled = embed_all(bundle, ds)
        pop_rand[s], rand_ct[s] = score_dataset(cloud, pooled, rand_tokens[s])
        del cloud
        torch.cuda.empty_cache()
        print(f"random seed {s} scored: {time.time() - t_start:.0f}s")
    del base, base3
    torch.cuda.empty_cache()

    # per-run JSON per cell type (§12) -------------------------------------------------------
    W2_KEYS = ("spatial_cell_emb_w2", "neighborhood_emb_w2", "cell_emb_w2")

    def _msd(vals):
        return {"mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")}

    for ct, m in main_ct.items():
        finite = bool(np.isfinite(m["w2"]).all()) and m["self_w2"] < 1e-3
        checks_ct = dict(checks, A6_self_w2_small_all_finite=finite)
        assert finite, f"{ct}: self-W2 {m['self_w2']:.2e} or non-finite W2"
        pop_ct = pop.loc[ct] if ct in pop.index else None
        rc = None
        if seeds:
            rc = {
                "n_sets": len(seeds), "seeds": seeds,
                "universes": {u: {f"{k}_{stat}": v for k in ("precision", "spearman")
                                  for stat, v in _msd([rand_ct[s][ct]["res"][u][k] for s in seeds]).items()}
                              for u in ("all", "minus_perturbed")},
                "population_w2": {k: _msd([float(pop_rand[s].loc[ct, k]) for s in seeds]) for k in W2_KEYS},
                "per_seed": {str(s): {"universes": rand_ct[s][ct]["res"],
                                      "population_w2": {k: float(pop_rand[s].loc[ct, k]) for k in W2_KEYS}}
                             for s in seeds},
            }
        out = {
            "sid": a.sid, "arm": arm, "epoch": a.epoch, "perturbation_target": targets,
            "target_label": a.target, "coarse_type": ct, "n_cells_scored": int(len(ct_rows[ct])),
            "n_cells_control_population": int(len(control)),
            "n_ref_ist_fib1_fib2": n_ist_fib,
            "mechanism": "foldchange", "foldchange_source": "np.exp(logfc_global)",
            "n_perturbed_genes": N_PERT_GENES, "n_perturbed_in_universe": int(m["is_pert"].sum()),
            "k": K, "min_cells": MIN_CELLS, "blur": BLUR,
            "universes": m["res"],
            "population_w2": ({k: float(pop_ct[k]) for k in W2_KEYS} | {"n_cells": int(pop_ct["n_cells"])})
                             if pop_ct is not None else None,
            "random_control": rc,
            "dose": dose, "self_w2": m["self_w2"],
            "checks": checks_ct, "weights": weights,
            "n_terra2k_genes_without_token": n_no_token,
            "terra_version": TERRA_VERSION, "git_sha": _git_sha(),
            "wall_seconds": time.time() - t_start,
        }
        tag = f"{a.sid}_{arm}_{a.target}" + ("_random" if seeds else "") + f"_{ct}.json"
        _write_json(per_run / tag, out)
        pg = pd.DataFrame(dict(gene=m["genes"], token=ct_universe[ct], n_cells=m["n_occ"], w2=m["w2"],
                               abs_logfc_ct=m["truth"], is_perturbed=m["is_pert"]))
        _write_csv(pg, per_run / tag.replace(".json", "_per_gene.csv"), index=False)
        for s in seeds:   # per-seed per-gene W2 of the random sets, so the control can be rescored offline
            rm = rand_ct[s][ct]
            _write_csv(pd.DataFrame(dict(gene=rm["genes"], token=ct_universe[ct], n_cells=rm["n_occ"],
                                         w2=rm["w2"], abs_logfc_ct=rm["truth"], is_perturbed=rm["is_pert"])),
                       per_run / tag.replace(".json", f"_seed{s}_per_gene.csv"), index=False)
        r = m["res"]
        sw2 = out["population_w2"]["spatial_cell_emb_w2"] if pop_ct is not None else float("nan")
        print(f"{ct}: n={len(ct_rows[ct]):,} | |A|={r['all']['n_genes']} P@{K}={r['all']['precision']:.3f} "
              f"rho={r['all']['spearman']:.3f} | |B|={r['minus_perturbed']['n_genes']} "
              f"P@{K}={r['minus_perturbed']['precision']:.3f} rho={r['minus_perturbed']['spearman']:.3f} | "
              f"chance {r['all']['chance']:.3f}/{r['minus_perturbed']['chance']:.3f} | "
              f"spatial_cell_emb_w2={sw2:.4g}"
              + (f" | random P@{K} {rc['universes']['minus_perturbed']['precision_mean']:.3f}" if rc else ""))
    print(f"done {arm} {a.target} in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
