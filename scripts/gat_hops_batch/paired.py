#!/usr/bin/env python
"""Paired per-fold comparison of arms (B vs A, C vs A) on common (slide, cell type, domain)."""
import json, math, os, sys, statistics, csv
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results", "gat_hops_batch")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import parse_metric_filename, METRICS

rows = {}
for arm in ["h1-b256", "h1-b512", "h3-b256"]:
    for f in sorted(os.listdir(os.path.join(RES, arm))):
        if not f.endswith(".json") or f.startswith("timing"):
            continue
        sid, model, ct, dom = parse_metric_filename(f)
        d = json.load(open(os.path.join(RES, arm, f)))
        rows[(arm, sid, ct, dom)] = {k: fn(d) for k, _, fn in METRICS}

with open(os.path.join(RES, "per_fold.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["arm", "sid", "ct", "dom"] + [k for k, _, _ in METRICS])
    for key in sorted(rows):
        w.writerow(list(key) + [f"{rows[key][k]:.4f}" for k, _, _ in METRICS])

def paired(a, b):
    keys = sorted(k[1:] for k in rows if k[0] == a and (b,) + k[1:] in rows)
    out = []
    for k, label, _ in METRICS:
        d = [rows[(b,) + key][k] - rows[(a,) + key][k] for key in keys]
        out.append((label, statistics.mean(d), statistics.pstdev(d), sum(x > 0 for x in d) / len(d)))
    return len(keys), out

lines = ["| Comparison | n (fold x domain) | Metric | mean diff | sd diff | frac > 0 |", "|---|---|---|---|---|---|"]
for a, b, name in [("h1-b256", "h1-b512", "B - A (batch 512 vs 256)"), ("h1-b256", "h3-b256", "C - A (3 hops vs 1)")]:
    n, out = paired(a, b)
    for label, m, s, f in out:
        lines.append(f"| {name} | {n} | {label} | {m:+.3f} | {s:.3f} | {f:.2f} |")
print("\n".join(lines))
