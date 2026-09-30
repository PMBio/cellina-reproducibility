#!/usr/bin/env python
"""Aggregate the cellina v1.1.2 LOO sweep (results/cellina112_loo/<dataset>/).

    python scripts/cellina112_loo/summarize.py [--results-root DIR]

Prints, per dataset, the four paper metrics pooled as mean +- std over
slide x cell type x holdout domain (plus per domain), next to the paper's current
`main` Cellina row (cellina 1.1.1: counterfactual spatial features aggregated from raw
counts), and writes summary.md / per_fold.csv under each dataset's results dir.
"""
import argparse, csv, json, math, os, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(REPO, "results", "cellina112_loo")
DATASETS = {"merfish": (["Isocortex", "Fiber_tracts"], 30), "crc": (["CRC"], 30)}
METRICS = [("pearson", "Pearson", lambda d: d.get("pearson")),
           ("direction_match_k", "Signed precision", lambda d: d.get("direction_match_k")),
           ("edistance_pca_log", "E-distance", lambda d: d.get("edistance_pca_log")),
           ("rmse_lfc", "RMSE_LFC", lambda d: math.sqrt(d["mse_lfc"]) if d.get("mse_lfc") is not None else None),
           ("nb_deviance", "NB deviance", lambda d: d.get("nb_deviance"))]
# Paper `main` Cellina rows, pooled over the 30 folds of each dataset
# (merfish: notebooks/make_table.ipynb on results/loo_summary_merfish_DEG_50_v2.csv;
#  crc: results/terra_handoff/loo_summary_crc_DEG_50_v5.csv, model_name 'cellina').
# Paper (cellina 1.1.1 main) rows, recomputed from the per-fold CSVs with the SAME pooling as the
# 1.1.2 sweep below: mean +- sd over all slide x cell-type x holdout-domain folds (n = 30 each).
#   merfish: results/loo_summary_merfish_DEG_50_v2.csv (git 7da94fb; 3 slides x 5 cts x 2 domains),
#            model_name `cellina` (edge) and `cellina-pert` (node perturbation).  NB: make_table.ipynb
#            reports the sd as the average of the two per-domain sds, which is why the paper table
#            shows e.g. 0.829 +- 0.156 instead of the pooled 0.829 +- 0.155.
#   crc:     results/terra_handoff/loo_summary_crc_DEG_50_v5.csv (6 slides x 5 cts x 1 domain).
REFERENCE = {
    "merfish": [("Cellina 1.1.1 main (paper, edge cf from raw counts)",
                 [(0.829, 0.155), (0.473, 0.180), (7.960, 1.324), (6.268, 5.009), (0.043, 0.043)]),
                ("Cellina 1.1.1 main (paper, node perturbation)",
                 [(0.824, 0.155), (0.465, 0.174), (9.075, 1.830), (6.310, 4.948), (0.044, 0.047)])],
    "crc": [("Cellina 1.1.1 main (paper, edge cf from raw counts)",
             [(0.820, 0.175), (0.399, 0.186), (7.501, 1.135), (1.289, 0.643), (0.030, 0.035)])],
}


def parse(fname, domains):
    stem = fname[:-5]
    if "-cf_" not in stem:
        return None
    left, right = stem.split("-cf_", 1)
    sid, model = left.split("_cellina112_", 1)
    model = "cellina112_" + model
    for dom in sorted(domains, key=len, reverse=True):
        if right.endswith("_" + dom):
            return sid, model, right[:-len(dom) - 1], dom
    return None


def ms(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def fmt(m, s, n=None):
    return "-" if m is None else f"{m:.2f} ± {s:.2f}" + (f" (n={n})" if n is not None else "")


def run(dataset, root):
    domains, n_expected = DATASETS[dataset]
    d = os.path.join(root, dataset)
    recs, timings = [], []
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not f.endswith(".json"):
            continue
        blob = json.load(open(os.path.join(d, f)))
        if f.startswith("timing_"):
            timings.append(blob); continue
        p = parse(f, domains)
        if p is None:
            print("unparsable:", f, file=sys.stderr); continue
        rec = dict(zip(("sid", "model", "cell_type", "domain"), p))
        for k, _, g in METRICS:
            rec[k] = g(blob)
        recs.append(rec)
    lines = [f"# Cellina v1.1.2 LOO sweep: {dataset}", "",
             f"Folds: {len(recs)}/{n_expected} metric JSONs (slide x cell type x holdout domain), seed 0, "
             "model `cellina112_0` (scripts/multi_seed/train_loo.py --cf_spatial_layer lognorm, eval scripts/eval_loo.py --use_cf).", "",
             "| Arm | " + " | ".join(l for _, l, _ in METRICS) + " | n |", "|---|" + "---|" * (len(METRICS) + 1)]
    for label, ref in REFERENCE[dataset]:
        lines.append(f"| {label} | " + " | ".join(fmt(m, s) for m, s in ref) + f" | {n_expected} |")
    rows = [("Cellina 1.1.2 (cf from lognorm), pooled", recs)] + \
           [(f"  {dom}", [r for r in recs if r["domain"] == dom]) for dom in domains] if len(domains) > 1 else \
           [("Cellina 1.1.2 (cf from lognorm), pooled", recs)]
    for lab, rs in rows:
        cells = [ms([r[k] for r in rs]) for k, _, _ in METRICS]
        lines.append(f"| {lab} | " + " | ".join(fmt(m, s) for m, s, _ in cells) + f" | {cells[0][2]} |")
    lines += ["", "## Per slide (pooled over cell types and domains)", "",
              "| slide | " + " | ".join(l for _, l, _ in METRICS) + " | n |", "|---|" + "---|" * (len(METRICS) + 1)]
    for sid in sorted({r["sid"] for r in recs}):
        rs = [r for r in recs if r["sid"] == sid]
        cells = [ms([r[k] for r in rs]) for k, _, _ in METRICS]
        lines.append(f"| {sid} | " + " | ".join(fmt(m, s) for m, s, _ in cells) + f" | {cells[0][2]} |")
    if timings:
        tw = [t["train_wall_seconds"] for t in timings if t.get("train_wall_seconds")]
        rss = [t["max_rss_gib"] for t in timings if t.get("max_rss_gib")]
        lines += ["", "## Cost", "",
                  f"train+cf wall (s): median {statistics.median(tw):.0f}, max {max(tw):.0f}; "
                  f"max RSS (GiB): median {statistics.median(rss):.1f}, max {max(rss):.1f}; n={len(timings)} folds, "
                  f"GPUs: {sorted({t.get('gpu_name') for t in timings})}"] if tw and rss else []
    out = "\n".join(lines) + "\n"
    print(out)
    if recs:
        with open(os.path.join(d, "summary.md"), "w") as fh:
            fh.write(out)
        with open(os.path.join(d, "per_fold.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results-root", default=ROOT)
    ap.add_argument("--dataset", choices=["merfish", "crc", "all"], default="all")
    a = ap.parse_args()
    for ds in (["merfish", "crc"] if a.dataset == "all" else [a.dataset]):
        run(ds, a.results_root)
