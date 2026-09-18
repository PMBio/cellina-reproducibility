#!/usr/bin/env python
"""Summarise both TERRA tracks in the format the other methods use.

Decoder track -> the schema of `origin/results:results/loo_summary_crc_DEG_50_v4.csv`
(sid, model_name, holdout_celltype, n_deg + 14 metrics), aggregated exactly the way
notebooks/make_table.ipynb aggregates it: one groupby over the 30 (slide x held-out cell type)
folds, mean +/- std, with rmse -> log10(rmse) and rmse_lfc = sqrt(mse_lfc).

Shift track -> the terra_og per-run JSONs (results/terra_og/per_run/*.json), same fold grain,
reported per perturbation target (neighb_only / ct_neigh) and per universe.

Read-only: reads the committed CSV through `git show`, reads result JSONs, writes only new
files under results/terra/.

    python scripts/terra/terra_summary_tables.py
"""
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parents[2] / "data"
CORR = DATA / "datasets" / "crc" / "correlations"
PER_RUN = REPO / "results" / "terra_og" / "per_run"
OUT = REPO / "results" / "terra"
V4 = "results/loo_summary_crc_DEG_50_v4.csv"
CTS = ["Endothelial", "Epithelial", "Fibroblast", "Myeloid", "T_cell"]
SIDS = ["crc_120", "crc_210", "crc_221", "crc_231", "crc_242", "crc_232"]
# make_table.ipynb METRICS (sign = higher is better)
TABLE_METRICS = {"pearson": +1, "direction_match_k": +1, "edistance_pca_log": -1, "rmse_lfc": -1}
EXTRA_METRICS = {"spearman": +1, "precision": +1, "rmse": -1, "nb_deviance": -1}


def v4_df():
    r = subprocess.run(["git", "-C", str(REPO), "show", f"origin/results:{V4}"],
                       capture_output=True, text=True)
    assert not r.returncode, r.stderr
    return pd.read_csv(io.StringIO(r.stdout))


def arm_map():
    """{sid: {'terra-lora-epN': label}} from each slide's epoch_selection.json."""
    out = {}
    for sid in SIDS:
        sel = DATA / "datasets" / "crc" / sid / "terra" / "epoch_selection.json"
        j = json.loads(sel.read_text())
        final, guard = j["final_epoch"], j["latest_passing_epoch"]
        m = {"terra-frozen": ["terra-frozen"], f"terra-lora-ep{final}": ["terra-lora-final"]}
        if guard is None:
            print(f"WARN: {sid} has no guard-passing epoch", file=sys.stderr)
        elif guard == final:
            m[f"terra-lora-ep{final}"].append("terra-lora-guard")
        else:
            m[f"terra-lora-ep{guard}"] = ["terra-lora-guard"]
        out[sid] = m
    return out


PAT = re.compile(r"^(crc_\d+)_(terra-.+?)-cf_(" + "|".join(CTS) + r")_CRC$")


def decoder_rows(amap):
    """TERRA decoder JSONs -> v4 schema rows. Skips the shift track entirely."""
    rows = []
    for f in sorted(CORR.glob("*-cf_*_CRC.json")):
        m = PAT.match(f.stem)
        if not m:
            continue
        sid, arm, ct = m.groups()
        is_null = arm.endswith("-null")
        base = arm[:-5] if is_null else arm
        labels = amap[sid].get(base)
        if labels is None:                       # an epoch we do not score in the table
            continue
        j = json.loads(f.read_text())
        for lab in labels:
            rows.append({"sid": sid, "model_name": lab + ("-null" if is_null else ""),
                         "holdout_celltype": f"{ct}_CRC", "n_deg": j["n_deg"],
                         **{k: v for k, v in j.items() if k != "n_deg"}, "source": f.name})
    return pd.DataFrame(rows)


def derive(df):
    df = df.copy()
    df["rmse_lfc"] = np.sqrt(df["mse_lfc"])
    df["rmse"] = np.log10(df["rmse"])
    return df


def agg(df, metrics):
    g = df.groupby("model_name")[list(metrics)].agg(["mean", "std"])
    n = df.groupby("model_name").size().rename("n_folds")
    return g, n


def fmt(g, n, metrics, order):
    best = {m: (g[(m, "mean")].idxmax() if s > 0 else g[(m, "mean")].idxmin())
            for m, s in metrics.items()}
    lines = []
    for mn in order:
        if mn not in g.index:
            continue
        cells = []
        for m in metrics:
            mu, sd = g.loc[mn, (m, "mean")], g.loc[mn, (m, "std")]
            s = f"{mu:.2f} +/- {sd:.2f}" if abs(mu) < 1000 else f"{mu:.3g} +/- {sd:.3g}"
            cells.append(("*" + s + "*") if best[m] == mn else s)
        lines.append((mn, int(n[mn]), cells))
    return lines


def show(title, lines, metrics):
    w = max(len(l[0]) for l in lines) + 2
    head = f"{'method':<{w}}{'n':>4}  " + "  ".join(f"{m:>22}" for m in metrics)
    print(f"\n### {title}\n")
    print(head)
    print("-" * len(head))
    for mn, n, cells in lines:
        print(f"{mn:<{w}}{n:>4}  " + "  ".join(f"{c:>22}" for c in cells))


# ---------------------------------------------------------------- shift track
def _w2(pw, stat=None):
    """spatial_cell_emb_w2 of a population_w2 block; NaN for arms without embeddings (cellina)."""
    if not pw:
        return float("nan")
    v = pw["spatial_cell_emb_w2"]
    return v[stat] if stat else v


def shift_rows(amap):
    rows = []
    for f in sorted(PER_RUN.glob("*.json")):
        j = json.loads(f.read_text())
        sid, arm, tgt, ct = j["sid"], j["arm"], j["target_label"], j["coarse_type"]
        rnd = j.get("random_control")
        labels = amap[sid].get(arm, [arm])      # non-TERRA arms (cellina-pert) keep their own name
        base = {"sid": sid, "holdout_celltype": f"{ct}_{tgt}", "coarse_type": ct,
                "target": tgt, "n_genes_all": j["universes"]["all"]["n_genes"],
                "n_genes_mp": j["universes"]["minus_perturbed"]["n_genes"],
                "chance": j["universes"]["all"]["chance"],
                "n_perturbed_in_universe": j["n_perturbed_in_universe"]}
        if rnd:                                   # the random-gene control file
            rows.append({**base, "model_name": "terra-random-gene",
                         "precision_all": rnd["universes"]["all"]["precision_mean"],
                         "precision_mp": rnd["universes"]["minus_perturbed"]["precision_mean"],
                         "spearman_all": rnd["universes"]["all"]["spearman_mean"],
                         "spearman_mp": rnd["universes"]["minus_perturbed"]["spearman_mean"],
                         "w2_spatial_cell": _w2(rnd["population_w2"], "mean"),
                         "source": f.name})
            continue
        for lab in labels:
            rows.append({**base, "model_name": lab,
                         "precision_all": j["universes"]["all"]["precision"],
                         "precision_mp": j["universes"]["minus_perturbed"]["precision"],
                         "spearman_all": j["universes"]["all"]["spearman"],
                         "spearman_mp": j["universes"]["minus_perturbed"]["spearman"],
                         "w2_spatial_cell": _w2(j["population_w2"]),
                         "source": f.name})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    amap = arm_map()
    print("arm map per slide:")
    for sid, m in amap.items():
        print(f"  {sid}: " + ", ".join(f"{k} -> {'+'.join(v)}" for k, v in m.items()))

    # ---- decoder -----------------------------------------------------------
    ref = v4_df()
    terra = decoder_rows(amap)
    dec = pd.concat([ref.assign(source="origin/results:" + V4),
                     terra[[c for c in ref.columns] + ["source"]]], ignore_index=True)
    dec = derive(dec)
    dec.to_csv(OUT / "terra_decoder_folds.csv", index=False)

    metrics = {**TABLE_METRICS, **EXTRA_METRICS}
    g, n = agg(dec, metrics)
    ref_order = ["baseline", "cpa", "scgen", "concert", "mintflow", "cellina-ablated",
                 "cellina", "cellina-graph", "cellina-pert", "cellina-gat-pert",
                 "spatialprop-pert"]
    terra_order = ["terra-frozen", "terra-lora-final", "terra-lora-guard"]
    null_order = [t + "-null" for t in terra_order]
    show("Decoder track / Table 1 metrics -- mean +/- std over 30 folds (6 slides x 5 held-out cell types)",
         fmt(g, n, TABLE_METRICS, ref_order + terra_order + null_order), TABLE_METRICS)
    show("Decoder track / secondary metrics (same folds; rmse is log10)",
         fmt(g, n, EXTRA_METRICS, ref_order + terra_order + null_order), EXTRA_METRICS)

    piv = dec[dec.model_name.isin(terra_order + null_order + ["cellina-pert"])].pivot_table(
        index=["sid", "holdout_celltype"], columns="model_name", values="pearson")
    print("\n### Decoder pearson per fold (TERRA arms + cellina-pert)\n")
    print(piv[[c for c in ["cellina-pert"] + terra_order + null_order if c in piv]].round(3).to_string())

    # ---- shift -------------------------------------------------------------
    sh = shift_rows(amap)
    sh.to_csv(OUT / "terra_shift_folds.csv", index=False)
    sh_order = terra_order + ["terra-random-gene", "cellina-pert"]
    for tgt in ["neighb_only", "ct_neigh"]:
        s = sh[sh.target == tgt]
        sm = {"precision_all": +1, "precision_mp": +1, "spearman_all": +1, "chance": 0}
        gg = s.groupby("model_name")[["precision_all", "precision_mp", "spearman_all", "chance"]].agg(["mean", "std"])
        nn = s.groupby("model_name").size()
        show(f"Shift track (terra_og) target={tgt} -- mean +/- std over 30 folds",
             fmt(gg, nn, sm, sh_order), sm)

    # Excess over chance, and the same aggregate with the degenerate universes removed.
    # crc_221 Endothelial (69 genes) / Myeloid (70 genes) have chance ~.70: precision there is
    # uninformative and drags every mean up, so report both with and without them.
    sh["excess_all"] = sh.precision_all - sh.chance
    sh["excess_mp"] = sh.precision_mp - sh.chance
    deg = sh.chance > 0.2
    print("\n### Degenerate universes (chance > 0.2) -- excluded from the clean aggregate\n")
    print(sh.loc[deg, ["sid", "coarse_type", "target", "n_genes_all", "chance"]]
          .drop_duplicates(["sid", "coarse_type", "target"]).to_string(index=False))
    sm2 = {"excess_all": +1, "excess_mp": +1}
    for tgt in ["neighb_only", "ct_neigh"]:
        s2 = sh[(sh.target == tgt) & ~deg]
        gg = s2.groupby("model_name")[list(sm2)].agg(["mean", "std"])
        nn = s2.groupby("model_name").size()
        show(f"Shift track precision MINUS chance, target={tgt}, degenerate folds dropped",
             fmt(gg, nn, sm2, sh_order), sm2)

    # How often do three genuinely different models return the identical precision?
    w = sh[sh.model_name.isin(terra_order)].pivot_table(
        index=["sid", "coarse_type", "target"], columns="model_name", values="precision_all")
    ident = (w[terra_order].nunique(axis=1) == 1)
    print("\n### Folds where frozen / lora-final / lora-guard give a BIT-IDENTICAL precision@50\n")
    print(ident.groupby(level="target").agg(["sum", "size"]).to_string())

    print("\n### Shift precision@50, universe 'all', per fold\n")
    for tgt in ["neighb_only", "ct_neigh"]:
        p = sh[sh.target == tgt].pivot_table(index=["sid", "coarse_type"],
                                             columns="model_name", values="precision_all")
        print(f"-- {tgt}")
        print(p[[c for c in sh_order if c in p]].round(3).to_string(), "\n")

    print(f"wrote {OUT/'terra_decoder_folds.csv'} ({len(dec)} rows)")
    print(f"wrote {OUT/'terra_shift_folds.csv'} ({len(sh)} rows)")


if __name__ == "__main__":
    main()
