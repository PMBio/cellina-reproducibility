"""Assemble the two publication tables from results/terra_og/per_run/*.json (spec §12).

    python scripts/terra/terra_og_tables.py [--sid crc_232] [--universe minus_perturbed]

{sid}_precision_{universe}.csv  rows = cell types; columns = (arm, target) precision@50, plus the
                   random-gene control mean +- sd (frozen arm) and chance.
{sid}_population_w2.csv    same layout for spatial_cell_emb_w2 and neighborhood_emb_w2.
Missing runs leave NaN cells; nothing is invented.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "results" / "terra_og"
ARMS = ["terra-frozen", "terra-lora-ep5", "terra-lora-ep2"]
TARGETS = ["neighb_only", "ct_neigh"]
CTS = ["Endothelial", "Epithelial", "Fibroblast", "Myeloid", "T_cell"]


def load(per_run, sid):
    runs = {}
    for f in sorted(per_run.glob(f"{sid}_*.json")):
        j = json.loads(f.read_text())
        runs[(j["arm"], j["target_label"], bool(j.get("random_control")), j["coarse_type"])] = j
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="crc_232")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--universe", default="minus_perturbed", choices=["all", "minus_perturbed"])
    a = ap.parse_args()
    res = Path(a.results)
    runs = load(res / "per_run", a.sid)
    arms = sorted({k[0] for k in runs}, key=lambda x: ARMS.index(x) if x in ARMS else 99)
    cts = [c for c in CTS if any(k[3] == c for k in runs)]

    prec, spear, pw2, nw2 = [], [], [], []
    for ct in cts:
        rp, rs, rw, rn = {"cell_type": ct}, {"cell_type": ct}, {"cell_type": ct}, {"cell_type": ct}
        for arm in arms:
            for tg in TARGETS:
                j = runs.get((arm, tg, False, ct)) or runs.get((arm, tg, True, ct))
                col = f"{arm}|{tg}"
                if j is None:
                    rp[col] = rs[col] = rw[col] = rn[col] = float("nan")
                    continue
                u = j["universes"][a.universe]
                rp[col], rs[col] = u["precision"], u["spearman"]
                rp["chance"] = u["chance"]
                rp["n_genes"] = u["n_genes"]
                rw[col] = j["population_w2"]["spatial_cell_emb_w2"] if j["population_w2"] else float("nan")
                rn[col] = j["population_w2"]["neighborhood_emb_w2"] if j["population_w2"] else float("nan")
                rp["n_cells"] = rw["n_cells"] = j["n_cells_scored"]
        for tg in TARGETS:
            j = runs.get(("terra-frozen", tg, True, ct))
            col = f"random-gene|{tg}"
            if j is None:
                rp[col] = rs[col] = rw[col] = rn[col] = float("nan")
                continue
            rc = j["random_control"]
            u = rc["universes"][a.universe]
            rp[col] = u["precision_mean"]
            rp[col + "|sd"] = u["precision_sd"]
            rs[col] = u["spearman_mean"]
            rw[col] = rc["population_w2"]["spatial_cell_emb_w2"]["mean"]
            rw[col + "|sd"] = rc["population_w2"]["spatial_cell_emb_w2"]["sd"]
            rn[col] = rc["population_w2"]["neighborhood_emb_w2"]["mean"]
            rp["random_n_sets"] = rc["n_sets"]
        prec.append(rp); spear.append(rs); pw2.append(rw); nw2.append(rn)

    out = res / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prec).to_csv(out / f"{a.sid}_precision_{a.universe}.csv", index=False)
    pd.DataFrame(spear).to_csv(out / f"{a.sid}_spearman_{a.universe}.csv", index=False)
    pd.DataFrame(pw2).to_csv(out / f"{a.sid}_population_w2.csv", index=False)
    pd.DataFrame(nw2).to_csv(out / f"{a.sid}_population_w2_neighborhood.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(f"precision@50, universe={a.universe}\n{pd.DataFrame(prec).set_index('cell_type')}\n")
        print(f"spatial_cell_emb_w2\n{pd.DataFrame(pw2).set_index('cell_type')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
