"""Cross-slide summary of results/terra_og/per_run/*.json (spec §12, all slides).

    python scripts/terra/terra_og_summary.py [--universe minus_perturbed]

Writes tables/summary_precision_{universe}.csv and tables/summary_population_w2.csv:
rows = (slide, cell type); columns = arm|target P@50, random-gene mean, chance; plus a
per-arm mean over all (slide, cell type) rows at the bottom of the printout.
Arms are labelled generically: frozen, lora-final (epoch 5), lora-early (the latest passing
epoch < 5, which differs per slide), so slides can be pooled.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "results" / "terra_og"
TARGETS = ["neighb_only", "ct_neigh"]


def arm_label(j):
    if j["arm"] == "terra-frozen":
        return "frozen"
    return "lora-final" if j["epoch"] == 5 else f"lora-early"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--universe", default="minus_perturbed", choices=["all", "minus_perturbed"])
    a = ap.parse_args()
    res = Path(a.results)
    rows = {}
    for f in sorted((res / "per_run").glob("*.json")):
        j = json.loads(f.read_text())
        key = (j["sid"], j["coarse_type"])
        r = rows.setdefault(key, {"sid": j["sid"], "cell_type": j["coarse_type"],
                                  "n_cells": j["n_cells_scored"]})
        u = j["universes"][a.universe]
        r["chance"] = u["chance"]
        r["n_genes"] = u["n_genes"]
        if j.get("random_control"):
            rc = j["random_control"]
            r[f"random|{j['target_label']}"] = rc["universes"][a.universe]["precision_mean"]
            r[f"random|{j['target_label']}|w2"] = rc["population_w2"]["spatial_cell_emb_w2"]["mean"]
        else:
            lab = arm_label(j)
            r[f"{lab}|{j['target_label']}"] = u["precision"]
            r[f"{lab}|{j['target_label']}|rho"] = u["spearman"]
            r[f"{lab}|{j['target_label']}|w2"] = j["population_w2"]["spatial_cell_emb_w2"] \
                if j["population_w2"] else float("nan")
            if lab.startswith("lora"):
                r[f"{lab}|epoch"] = j["epoch"]
    df = pd.DataFrame(list(rows.values())).sort_values(["sid", "cell_type"])
    out = res / "tables"
    out.mkdir(exist_ok=True)
    pcols = ["sid", "cell_type", "n_cells", "chance", "n_genes"] + \
            [c for c in df.columns if "|" in c and not c.endswith(("|w2", "|rho", "|epoch"))]
    wcols = ["sid", "cell_type", "n_cells"] + [c for c in df.columns if c.endswith("|w2")]
    df[pcols].to_csv(out / f"summary_precision_{a.universe}.csv", index=False)
    df[wcols].to_csv(out / "summary_population_w2.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        print(f"P@50, universe={a.universe}")
        print(df[pcols].set_index(["sid", "cell_type"]).to_string())
        print("\nmean over (slide, cell type):")
        print(df[pcols].drop(columns=["sid", "cell_type", "n_cells", "n_genes"]).mean().round(3).to_string())
        print("\nmean chance-excess (P@50 - chance):")
        ex = df[[c for c in pcols if "|" in c]].sub(df["chance"], axis=0).mean().round(3)
        print(ex.to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
