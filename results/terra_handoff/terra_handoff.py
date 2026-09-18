#!/usr/bin/env python
"""Build the self-contained hand-off folder for regenerating Table 1 + the bar figures.

Decisions this encodes (Daniel, round 1-2 of the design interview):
  Q1  TERRA LoRA arm = the FINAL epoch only ("terra-lora-final"); the guard-selected epoch is
      dropped from the deliverable (less intervention on our side, and marginally better).
  Q2  The `-null` sanity-check arms do NOT go in the main table.
  Q3  The shift track of record is terra_og; the old `*-native-terra2k*` track is discarded.
  Q4  Degenerate universes (chance > 0.2) stay in the per-fold CSV, flagged, and are excluded
      from the aggregate.
  Q5  TERRA rows are emitted in the canonical `loo_summary_crc_DEG_50_v4.csv` column ORDER
      (which puts edistance_pca BEFORE edistance_pca_log -- the per-method CSVs have it the
      other way round; we align to the summary file).
  Q6  TERRA sits after spatialprop-pert, labelled as a node-perturbation method.
  Q7  Aggregation is identical to make_table.ipynb: groupby -> mean/std over the 30 folds.
  Q10 The shift table reuses make_table.ipynb's `perturbation` dimension by encoding the
      target into holdout_celltype as "{cell type}_{nb-only|ct-nb}" (no underscore in the
      target label, or the notebook's rsplit mangles it).

    python scripts/terra/terra_handoff.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terra_summary_tables import (REPO, TABLE_METRICS, arm_map, decoder_rows,  # noqa: E402
                                  derive, shift_rows, v4_df)

OUT = REPO / "results" / "terra_handoff"
ARMS = ["terra-frozen", "terra-lora-final"]          # Q1 + Q2
TGT = {"neighb_only": "nb-only", "ct_neigh": "ct-nb"}  # Q10: no underscores
RENAME = {"terra-frozen": r"TERRA$_{node-pert}$",
          "terra-lora-final": r"TERRA-LoRA$_{node-pert}$",
          "terra-random-gene": "random gene"}


def md_table(g, n, metrics, order, rename=None):
    best = {m: (g[(m, "mean")].idxmax() if s > 0 else g[(m, "mean")].idxmin())
            for m, s in metrics.items() if g[(m, "mean")].notna().any()}
    head = "| method | n | " + " | ".join(metrics) + " |"
    rows = [head, "|" + "---|" * (len(metrics) + 2)]
    for mn in order:
        if mn not in g.index:
            continue
        cells = []
        for m in metrics:
            mu, sd = g.loc[mn, (m, "mean")], g.loc[mn, (m, "std")]
            s = "--" if pd.isna(mu) else f"{mu:.2f} ± {sd:.2f}"
            cells.append(f"**{s}**" if best.get(m) == mn else s)
        lab = (rename or {}).get(mn, mn)
        rows.append(f"| {lab} | {int(n[mn])} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    amap = arm_map()

    # ---------------- decoder: v5 summary CSV in the canonical v4 schema ----------------
    ref = v4_df()
    terra = decoder_rows(amap)
    terra = terra[terra.model_name.isin(ARMS)]                    # Q1, Q2
    assert len(terra) == 60, f"expected 2 arms x 30 folds, got {len(terra)}"
    v5 = pd.concat([ref, terra[list(ref.columns)]], ignore_index=True)   # Q5: column order
    assert list(v5.columns) == list(ref.columns)
    assert v5.notna().all().all(), "TERRA rows must be as dense as the other methods"
    v5.to_csv(OUT / "loo_summary_crc_DEG_50_v5.csv", index=False)

    dec = derive(v5)
    order = ["baseline", "cpa", "scgen", "concert", "mintflow", "cellina-ablated", "cellina",
             "cellina-graph", "cellina-pert", "cellina-gat-pert", "spatialprop-pert", *ARMS]  # Q6
    g = dec.groupby("model_name")[list(TABLE_METRICS)].agg(["mean", "std"])
    n = dec.groupby("model_name").size()
    t1 = md_table(g, n, TABLE_METRICS, order, RENAME)                     # Q7

    # ---------------- shift: terra_og supplementary table ----------------
    sh = shift_rows(amap)
    sh = sh[sh.model_name.isin(ARMS + ["terra-random-gene"])]             # Q1, Q3
    sh["degenerate"] = sh.chance > 0.2                                    # Q4
    sh["excess_over_chance"] = sh.precision_all - sh.chance
    sup = pd.DataFrame({
        "sid": sh.sid, "model_name": sh.model_name,
        "holdout_celltype": sh.coarse_type.str.replace("T_cell", "T-cell", regex=False)
                            + "_" + sh.target.map(TGT),                   # Q10
        "n_deg": 50,
        "precision": sh.precision_all, "precision_minus_perturbed": sh.precision_mp,
        "excess_over_chance": sh.excess_over_chance, "spearman": sh.spearman_all,
        "chance": sh.chance, "n_genes": sh.n_genes_all,
        "n_perturbed_in_universe": sh.n_perturbed_in_universe,
        "w2_spatial_cell": sh.w2_spatial_cell, "degenerate": sh.degenerate,
    }).sort_values(["sid", "holdout_celltype", "model_name"]).reset_index(drop=True)
    sup.to_csv(OUT / "terra_og_shift_supp.csv", index=False)

    SM = {"precision": +1, "precision_minus_perturbed": +1,
          "excess_over_chance": +1, "spearman": +1}
    blocks = []
    for tgt, lab in TGT.items():
        s = sup[sup.holdout_celltype.str.endswith(lab) & ~sup.degenerate]  # Q4
        gg = s.groupby("model_name")[list(SM)].agg(["mean", "std"])
        nn = s.groupby("model_name").size()
        blocks.append(f"**target = `{lab}`** ({tgt})\n\n"
                      + md_table(gg, nn, SM, ARMS + ["terra-random-gene"], RENAME))
    ident = (sh[sh.model_name.isin(ARMS)]
             .pivot_table(index=["sid", "coarse_type", "target"], columns="model_name",
                          values="precision_all").nunique(axis=1) == 1)

    n_deg_folds = int(sup.degenerate.sum() / 3)
    (OUT / "tables" / "table1_decoder.md").write_text(
        "# Table 1 -- CRC leave-one-cell-type-out, decoder track\n\n"
        "Mean ± std over the 30 folds (6 slides × 5 held-out cell types), identical\n"
        "aggregation to `make_table.ipynb`. `rmse_lfc = sqrt(mse_lfc)`.\n\n" + t1 + "\n")
    (OUT / "tables" / "supp_shift.md").write_text(
        "# Supplementary -- TERRA-native perturbation (terra_og) shift track\n\n"
        "Mean ± std over the folds, per perturbation target. "
        f"{n_deg_folds} degenerate folds excluded\n(crc_221 Endothelial, 76 genes, chance .66; "
        "crc_221 Myeloid, 75 genes, chance .67).\n\n" + "\n\n".join(blocks) + "\n\n"
        f"Arms returning a bit-identical precision@50: "
        f"{int(ident.groupby(level='target').sum().get('ct_neigh', 0))}/30 ct_neigh folds, "
        f"{int(ident.groupby(level='target').sum().get('neighb_only', 0))}/30 neighb_only.\n")

    for f in ("terra_summary_tables.py", "terra_handoff.py", "terra_og_eval.py"):
        shutil.copy(REPO / "scripts" / "terra" / f, OUT / f)

    print(t1, "\n")
    for b in blocks:
        print(b, "\n")
    print(f"v5 rows: {len(v5)} ({len(ref)} existing + {len(terra)} TERRA), "
          f"methods: {v5.model_name.nunique()}")
    print(f"supp rows: {len(sup)}, degenerate flagged: {int(sup.degenerate.sum())}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
