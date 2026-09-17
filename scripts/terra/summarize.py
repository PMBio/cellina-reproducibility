#!/usr/bin/env python
"""Aggregate the TERRA benchmark over all 6 CRC slides -> results/terra_crc_*.csv.

Scored (sid, cell type) pairs and the reference rows come from the cellina-pert rows of
`origin/results:results/loo_cellina_crc_DEG_50_pert_v2.csv`; the TERRA arms per slide come
from `{slide_dir}/terra/epoch_selection.json` (frozen, lora-ep{final}, and lora-ep{K} when
the latest guard-passing epoch K is not the final one).  Missing JSONs are warned about and
skipped, so this is usable while runs land.

    python scripts/terra/summarize.py
"""
import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CELLINA_CSV = "results/loo_cellina_crc_DEG_50_pert_v2.csv"
HD = "CRC"
EXPOSURE = {"frozen": "encoder held out",
            "lora": "encoder fine-tuned on all cells (self-sup LoRA)"}
EXTRA = ["arm", "lora_epoch", "encoder_exposure", "epoch_passed_guard"]
M = ("precision", "pearson", "spearman")


def cellina_df():
    r = subprocess.run(["git", "-C", str(REPO), "show", f"origin/results:{CELLINA_CSV}"],
                       capture_output=True, text=True)
    assert not r.returncode, f"cannot read {CELLINA_CSV} from origin/results: {r.stderr}"
    df = pd.read_csv(io.StringIO(r.stdout))
    return df[df.model_name == "cellina-pert"].reset_index(drop=True)


def arms(slide_dir, sid):
    """[(arm, epoch|None, passed|None)] -- frozen + the LoRA epochs worth scoring."""
    out = [("terra-frozen", None, None)]
    sel = Path(slide_dir) / "terra" / "epoch_selection.json"
    if not sel.exists():
        print(f"WARN: missing {sel}; {sid} gets the frozen arm only", file=sys.stderr)
        return out
    j = json.loads(sel.read_text())
    final, k = j["final_epoch"], j["latest_passing_epoch"]
    for ep in [final] + ([k] if k is not None and k != final else []):
        out.append((f"terra-lora-ep{ep}", ep, j["epochs"][str(ep)]["passed"]))
    return out


def decoder_table(corr, sid, cts, arm_list, cols):
    rows = []
    for ct in cts:
        for arm, ep, passed in arm_list:
            for name in (arm, f"{arm}-null"):
                f = corr / f"{sid}_{name}-cf_{ct}_{HD}.json"
                if not f.exists():
                    print(f"WARN: missing {f.name}")
                    continue
                j = json.loads(f.read_text())
                # eval_loo writes no `nb_deviance` / `top_n_perturb`: they stay empty for the
                # TERRA rows (nb_deviance is not reproducible from this repo) so the column
                # set still lines up with the cellina CSV.
                rows.append({**{c: j.get(c) for c in cols},
                             "dataset_name": "crc", "sid": sid, "control_domain": "REF",
                             "target_domain": HD, "model_name": name, "holdout_celltype": ct,
                             "arm": name, "lora_epoch": ep,
                             "encoder_exposure": EXPOSURE["lora" if ep else "frozen"],
                             "epoch_passed_guard": passed, "source": f.name})
    return rows


def uni2(f):
    """Universe (ii) block of a shift JSON (the benchmark/terra2k HVG universe)."""
    if not f.exists():
        print(f"WARN: missing {f.name}")
        return None
    j = json.loads(f.read_text())
    if "skipped" in j:
        print(f"SKIP: {f.name}: {j['skipped']}")
        return None
    return next((v for k, v in j.items()
                 if isinstance(v, dict) and k.strip().startswith("(ii)")), None)


def vals(u, key=None):
    """Metrics of one arm inside a universe block: '<metric>', '<key>_<metric>' or
    '<key>_<metric>_mean' (the random-set means)."""
    if key is None:
        return {m: u.get(m) for m in M}
    return {m: (u.get(f"{key}_{m}") if f"{key}_{m}" in u else u.get(f"{key}_{m}_mean"))
            for m in M}


def shift_table(corr, sid, cts, arm_list):
    rows = []
    for ct in cts:
        seen = []
        for arm, ep, passed in arm_list:
            u = uni2(corr / f"{sid}_{arm}-native-terra2k_{ct}_{HD}.json")
            if u is None:
                continue
            seen.append(u)
            meta = dict(sid=sid, holdout_celltype=ct, lora_epoch=ep,
                        encoder_exposure=EXPOSURE["lora" if ep else "frozen"],
                        epoch_passed_guard=passed, k=u.get("k"), n_genes=u.get("n_genes"))
            rows.append(dict(meta, arm=arm, **vals(u)))
            rows.append(dict(meta, arm=f"{arm} random mean", **vals(u, "random")))
        if not seen:
            continue
        u = seen[0]
        meta = dict(sid=sid, holdout_celltype=ct, lora_epoch=None,
                    encoder_exposure="encoder held out", epoch_passed_guard=None,
                    k=u.get("k"), n_genes=u.get("n_genes"))
        if any(k.startswith("cellina") for k in u):
            rows.append(dict(meta, arm="cellina-pert", **vals(u, "cellina")))
            rows.append(dict(meta, arm="cellina random mean", **vals(u, "cellina_random")))
        else:
            print(f"WARN: no cellina arm in the {sid}/{ct} shift JSONs")
        rows.append(dict(meta, arm="chance", precision=u.get("chance")))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(REPO / "results"))
    p.add_argument("--corr_dir", default=str(Path(os.environ.get("DATA_ROOT", "/data/ddimitrov/data"))
                                             / "datasets/crc/correlations"))
    a = p.parse_args()
    corr, out = Path(a.corr_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.corr_dir).parent          # $DATA_ROOT/datasets/crc

    ref = cellina_df()
    cols = list(ref.columns)                # exact names AND order of the cellina CSV
    metric_cols = [c for c in cols if c not in
                   ("dataset_name", "sid", "control_domain", "target_domain",
                    "model_name", "holdout_celltype")]

    dec, shift = [], []
    for sid, g in ref.groupby("sid", sort=True):
        cts = list(g.holdout_celltype)
        al = arms(base / sid, sid)
        print(f"[{sid}] {len(cts)} cell types x arms {[x[0] for x in al]}")
        dec += decoder_table(corr, sid, cts, al, metric_cols)
        shift += shift_table(corr, sid, cts, al)

    ref_rows = ref.assign(arm="cellina-pert", lora_epoch=None,
                          encoder_exposure="encoder held out", epoch_passed_guard=None,
                          source=f"origin/results:{CELLINA_CSV}")
    dec = pd.concat([pd.DataFrame(dec), ref_rows], ignore_index=True)[cols + EXTRA + ["source"]]
    dec = dec.sort_values(["sid", "holdout_celltype", "arm"]).reset_index(drop=True)
    shift = pd.DataFrame(shift)
    if len(shift):
        shift = shift[["sid", "holdout_celltype", "arm", "lora_epoch", "encoder_exposure",
                       "epoch_passed_guard", "k", "n_genes", *M]]
        shift = shift.sort_values(["sid", "holdout_celltype", "arm"]).reset_index(drop=True)

    for df, f in ((dec, out / "terra_crc_decoder_DEG_50.csv"),
                  (shift, out / "terra_crc_shift_terra2k.csv")):
        df.to_csv(f, index=False)
        print(f"wrote {f} ({len(df)} rows)")
    if len(dec):
        print("\n## decoder: spearman\n")
        print(dec.pivot_table(index=["sid", "holdout_celltype"], columns="arm",
                              values="spearman", aggfunc="first").round(3).to_string())
    if len(shift):
        print("\n## shift (terra's own 2000 HVGs, favourable to TERRA): precision\n")
        print(shift.pivot_table(index=["sid", "holdout_celltype"], columns="arm",
                                values="precision", aggfunc="first").round(3).to_string())


if __name__ == "__main__":
    main()
