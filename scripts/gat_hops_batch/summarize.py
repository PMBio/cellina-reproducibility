#!/usr/bin/env python
"""Aggregate the Cellina-GAT hops / batch-size sensitivity test.

Reads the per-fold metric JSONs that run_fold.sbatch copied into
``results/gat_hops_batch/<arm>/`` (plus the ``timing_*.json`` files it writes there)
and prints / saves a per-arm table of the four paper metrics, pooled as
mean +- std over slide x cell type, per holdout domain and averaged.

    python scripts/gat_hops_batch/summarize.py [--results-dir DIR] [--out-dir DIR]

Metric JSON names are ``{sid}_{model_name}-cf_{cell type}_{domain}.json`` (see
scripts/eval_loo.py); the cell type may contain spaces and the domain may contain
an underscore (``Fiber_tracts``), so the filename is split from the left for
sid / model / cell type and the remainder is the domain.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS = os.path.join(REPO, "results", "gat_hops_batch")

ARM_ORDER = ["h1-b256", "h1-b512", "h3-b256", "h1lit-b256", "h1lit-b256-pyg", "h1k20-b256"]
ARM_NOTE = {
    "h1-b256": "A: num_neighbors [-1,0,0], batch 256 (May config)",
    "h1-b512": "B: num_neighbors [-1,0,0], batch 512",
    "h3-b256": "C: num_neighbors [-1,-1,-1], batch 256",
    "h1lit-b256": "A-lit: num_neighbors [-1] (literal May), batch 256",
    "h1lit-b256-pyg": "A-lit + pyg-lib (env cellina112), slide .036 only, timing check",
    "h1k20-b256": "A-k20: num_neighbors [20,0,0] (capped fan-out), batch 256, pyg-lib (cellina112)",
}
DATASETS = {
    # domains, expected metric JSONs per arm (slides x cell types x holdout domains)
    "merfish": (["Isocortex", "Fiber_tracts"], 30),   # 3 x 5 x 2
    "crc": (["CRC"], 30),                             # 6 x 5 x 1
}
DATASETS_ALL_DOMAINS = sorted({d for doms, _ in DATASETS.values() for d in doms})
DOMAINS = DATASETS["merfish"][0]
N_EXPECTED = DATASETS["merfish"][1]

# (key, label, how to derive it from the stats dict)
METRICS = [
    ("pearson", "Pearson", lambda d: d.get("pearson")),
    ("direction_match_k", "Signed precision", lambda d: d.get("direction_match_k")),
    ("edistance_pca_log", "E-distance", lambda d: d.get("edistance_pca_log")),
    ("rmse_lfc", "RMSE_LFC",
     lambda d: math.sqrt(d["mse_lfc"]) if d.get("mse_lfc") is not None else None),
]

# Reference rows from GAT_HOPS_BATCH_TEST.md.  Both are MERFISH, averaged over the
# two holdout domains: the current `main` 2-layer pipeline, and the rebuttal RT6
# numbers, which came from the May pipeline and are NOT directly comparable.
REFERENCE_ROWS_BY_DATASET = {
    "merfish": [
        ("main 2-layer (reference)", (0.83, 0.16), (0.49, 0.16), (8.07, 1.87), (6.22, 4.87)),
        ("rebuttal RT6 (not directly comparable)",
         (0.85, 0.15), (0.52, 0.14), (8.69, 1.45), (5.80, 4.50)),
    ],
    "crc": [
        ("main 2-layer (reference)", (0.84, 0.13), (0.36, 0.20), (7.78, 1.41), (1.24, 0.69)),
    ],
}
REFERENCE_ROWS = REFERENCE_ROWS_BY_DATASET["merfish"]


def parse_metric_filename(fname):
    """-> (sid, model_name, cell_type, domain) or None if it is not a metric JSON.

    Name is ``{sid}_{model}-cf_{ct}_{domain}``.  sid may contain an underscore
    (``crc_231``), so may the cell type (``T_cell``) and the domain (``Fiber_tracts``);
    the model name never does.  Split on ``-cf_`` first, then peel the model name
    off the left and a known domain off the right.
    """
    stem = fname[:-len(".json")]
    if "-cf_" not in stem:
        return None
    left, right = stem.split("-cf_", 1)
    if "_" not in left:
        return None
    sid, model = left.rsplit("_", 1)
    for domain in sorted(DATASETS_ALL_DOMAINS, key=len, reverse=True):
        if right.endswith("_" + domain):
            return sid, model, right[:-len(domain) - 1], domain
    return None


def load(results_dir):
    """-> (records, timings). records: dicts with arm/sid/ct/domain/metric values."""
    records, timings = [], []
    for arm in sorted(os.listdir(results_dir)):
        arm_dir = os.path.join(results_dir, arm)
        if not os.path.isdir(arm_dir):
            continue
        for fname in sorted(os.listdir(arm_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(arm_dir, fname)
            try:
                with open(path) as fh:
                    blob = json.load(fh)
            except (OSError, ValueError) as exc:
                print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
                continue
            if fname.startswith("timing_"):
                blob.setdefault("arm", arm)
                timings.append(blob)
                continue
            parsed = parse_metric_filename(fname)
            if parsed is None:
                print(f"WARNING: unparsable metric filename {fname}", file=sys.stderr)
                continue
            sid, model, cell_type, domain = parsed
            rec = {"arm": arm, "sid": sid, "model_name": model,
                   "cell_type": cell_type, "domain": domain, "file": fname}
            for key, _label, getter in METRICS:
                rec[key] = getter(blob)
            records.append(rec)
    return records, timings


def mean_std(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return statistics.fmean(vals), statistics.stdev(vals), len(vals)


def fmt(mean, std, n):
    if mean is None:
        return "-"
    return f"{mean:.2f} +- {std:.2f} (n={n})"


def build_rows(records):
    """-> list of (row_label, arm, scope, {metric: (mean, std, n)})"""
    by_arm = defaultdict(list)
    for r in records:
        by_arm[r["arm"]].append(r)
    arms = [a for a in ARM_ORDER if a in by_arm] + sorted(set(by_arm) - set(ARM_ORDER))
    rows = []
    for arm in arms:
        recs = by_arm[arm]
        for scope in DOMAINS + ["average"]:
            if scope == "average":
                subset = recs
            else:
                subset = [r for r in recs if r["domain"] == scope]
            stats = {key: mean_std([r[key] for r in subset]) for key, _l, _g in METRICS}
            rows.append((arm, scope, stats))
    return arms, rows


def timing_table(timings):
    by_arm = defaultdict(lambda: {"wall": [], "gpu": [], "rss": [], "gpus": set(), "n": 0})
    for t in timings:
        slot = by_arm[t.get("arm", "?")]
        slot["n"] += 1
        for src, dst in (("train_wall_seconds", "wall"), ("peak_gpu_mib", "gpu"),
                         ("max_rss_kb", "rss")):
            if t.get(src) is not None:
                slot[dst].append(t[src])
        if t.get("gpu_name"):
            slot["gpus"].add(t["gpu_name"])
    out = []
    arms = [a for a in ARM_ORDER if a in by_arm] + sorted(set(by_arm) - set(ARM_ORDER))
    for arm in arms:
        s = by_arm[arm]
        out.append({
            "arm": arm,
            "n": s["n"],
            "median_train_wall_s": statistics.median(s["wall"]) if s["wall"] else None,
            "median_train_wall_hms": (
                _hms(statistics.median(s["wall"])) if s["wall"] else None),
            "peak_gpu_mib": max(s["gpu"]) if s["gpu"] else None,
            "median_peak_gpu_mib": statistics.median(s["gpu"]) if s["gpu"] else None,
            "median_max_rss_gib": (
                round(statistics.median(s["rss"]) / 1048576.0, 2) if s["rss"] else None),
            "gpus": ", ".join(sorted(s["gpus"])) or "-",
        })
    return out


def _hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def render(records, timings, arms, rows):
    labels = [label for _k, label, _g in METRICS]
    lines = []
    lines.append("# Cellina-GAT hops / batch-size sensitivity test")
    lines.append("")
    lines.append("MERFISH leave-one-cell-type-out, top-50-DEG metrics, pooled mean +- std over")
    lines.append("slide x cell type (3 slides x 5 cell types per domain). Seed 0.")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Arm | Config | Domain | " + " | ".join(labels) + " |")
    lines.append("|---|---|---|" + "---|" * len(labels))
    for arm, scope, stats in rows:
        cells = [fmt(*stats[key]) for key, _l, _g in METRICS]
        lines.append(f"| {arm} | {ARM_NOTE.get(arm, '')} | {scope} | " + " | ".join(cells) + " |")
    for name, *vals in REFERENCE_ROWS:
        cells = [f"{m:.2f} +- {s:.2f}" for m, s in vals]
        lines.append(f"| _{name}_ | from GAT_HOPS_BATCH_TEST.md | average | "
                     + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Reference rows are quoted from the spec for orientation. The `main` 2-layer row "
                 "uses the same current pipeline; the RT6 row was produced under the May pipeline "
                 "and is **not directly comparable**.")
    lines.append("")

    lines.append("## Completeness")
    lines.append("")
    lines.append(f"| Arm | metric JSONs | expected | timing JSONs |")
    lines.append("|---|---|---|---|")
    n_by_arm = defaultdict(int)
    for r in records:
        n_by_arm[r["arm"]] += 1
    n_timing = defaultdict(int)
    for t in timings:
        n_timing[t.get("arm", "?")] += 1
    for arm in arms or ARM_ORDER:
        flag = "" if n_by_arm[arm] == N_EXPECTED else "  <- incomplete"
        lines.append(f"| {arm} | {n_by_arm[arm]} | {N_EXPECTED} | {n_timing[arm]}{flag} |")
    lines.append("")

    lines.append("## Cost")
    lines.append("")
    lines.append("| Arm | folds | median train wall | peak GPU (MiB) | median peak GPU (MiB) "
                 "| median max RSS (GiB) | GPU |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in timing_table(timings):
        lines.append("| {arm} | {n} | {wall} | {peak} | {mpeak} | {rss} | {gpus} |".format(
            arm=t["arm"], n=t["n"],
            wall=t["median_train_wall_hms"] or "-",
            peak=t["peak_gpu_mib"] if t["peak_gpu_mib"] is not None else "-",
            mpeak=int(t["median_peak_gpu_mib"]) if t["median_peak_gpu_mib"] is not None else "-",
            rss=t["median_max_rss_gib"] if t["median_max_rss_gib"] is not None else "-",
            gpus=t["gpus"]))
    if not timings:
        lines.append("| - | 0 | - | - | - | - | - |")
    lines.append("")
    return "\n".join(lines)


def write_csv(path, rows, timings):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["arm", "domain", "n"]
        for _k, label, _g in METRICS:
            header += [f"{label}_mean", f"{label}_std"]
        w.writerow(header)
        for arm, scope, stats in rows:
            n = max((stats[k][2] for k, _l, _g in METRICS), default=0)
            row = [arm, scope, n]
            for key, _l, _g in METRICS:
                mean, std, _ = stats[key]
                row += ["" if mean is None else round(mean, 4),
                        "" if std is None else round(std, 4)]
            w.writerow(row)
        for name, *vals in REFERENCE_ROWS:
            row = [name, "average", ""]
            for m, s in vals:
                row += [m, s]
            w.writerow(row)
        w.writerow([])
        w.writerow(["arm", "folds", "median_train_wall_s", "peak_gpu_mib",
                    "median_peak_gpu_mib", "median_max_rss_gib", "gpu"])
        for t in timing_table(timings):
            w.writerow([t["arm"], t["n"], t["median_train_wall_s"], t["peak_gpu_mib"],
                        t["median_peak_gpu_mib"], t["median_max_rss_gib"], t["gpus"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="merfish", choices=sorted(DATASETS),
                    help="crc defaults --results-dir to results/gat_hops_batch/crc")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out-dir", default=None, help="default: --results-dir")
    args = ap.parse_args()
    global DOMAINS, N_EXPECTED, REFERENCE_ROWS
    DOMAINS, N_EXPECTED = DATASETS[args.dataset]
    REFERENCE_ROWS = REFERENCE_ROWS_BY_DATASET[args.dataset]
    if args.results_dir is None:
        args.results_dir = DEFAULT_RESULTS if args.dataset == "merfish" else os.path.join(DEFAULT_RESULTS, args.dataset)
    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir or results_dir)
    if not os.path.isdir(results_dir):
        sys.exit(f"no such results dir: {results_dir}")

    records, timings = load(results_dir)
    arms, rows = build_rows(records)
    text = render(records, timings, arms, rows)
    print(text)

    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "summary.md")
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(md_path, "w") as fh:
        fh.write(text + "\n")
    write_csv(csv_path, rows, timings)
    print(f"\nwrote {md_path}\nwrote {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
