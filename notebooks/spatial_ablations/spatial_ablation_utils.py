"""
Shared constants and helpers for LOO ablation scripts (ablation_loo.py / ablation_loo_base.py).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Shared constants ──────────────────────────────────────────────────────────
SLIDE_ID      = 242
LABELS_KEY    = "coarse_type"
DOMAINS_KEY   = "typ"
TOP_N         = 50
MIN_CELLS     = 50
BATCH_SIZE    = 512
LIBRARY_SIZE  = 1e4
DEVICES       = [1]

CF_METRICS  = ["pearson_r", "spearman_r", "precision", "mixing_index", "edistance", "rmse"]
RUN_METRICS = ["marginal_ll", "ari", "nmi",
               "ari_s_labels", "nmi_s_labels",
               "ari_z_domains", "nmi_z_domains"]
METRICS     = CF_METRICS + RUN_METRICS

# Shared CSV: both ablation scripts append here
RESULTS_CSV = "results/ablation_loo.csv"

SLT_COLORS  = {"supcon": "#4C72B0", "domain_clf": "#e05c5c", "cellina-base": "#2ca02c"}
SLT_MARKERS = {"supcon": "o", "domain_clf": "s", "cellina-base": "^"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _normalize_total(arr, target_sum=LIBRARY_SIZE):
    return arr / arr.sum(axis=1, keepdims=True) * target_sum


def evaluate_model(
    model,
    adata,
    holdout_celltype,
    spatial_loss_type,
    link_prediction_weight,
    marginal_ll_kwargs=None,
    cf_extra_kwargs=None,
    batch_size_eval_factor=1,
    ref_domain=None,
    target_domains=None,
    domains_key=None,
    labels_key=None,
):
    """Post-training evaluation shared by all ablation scripts.

    Parameters
    ----------
    marginal_ll_kwargs : dict, optional
        Extra kwargs for model.get_marginal_ll() — e.g. {"reduce": "mean"} or
        {"return_mean": True} depending on model class.
    cf_extra_kwargs : dict, optional
        Extra kwargs for model.get_counterfactual_expression() — e.g.
        {"n_neighbors_per_seed": 50}.
    batch_size_eval_factor : int, optional
        Multiplier applied to BATCH_SIZE for counterfactual inference.
        Use 1 for cellina-graph (default), 4 for cellina-base (requires less VRAM).
    ref_domain : str, optional
        Name of the reference domain in adata.obs[domains_key]. If None, the
        reference domain is detected by the presence of "REF" in its name (CRC default).
    target_domains : list of str, optional
        Names of target domains. If None, target domains are detected by "CRC"/"TVA"
        in their names (CRC default).
    domains_key : str, optional
        obs column for domain labels. Defaults to the global DOMAINS_KEY ("typ").
    labels_key : str, optional
        obs column for cell-type labels. Defaults to the global LABELS_KEY ("coarse_type").
    """
    from perturb_utils import compute_cf_logfc
    import scib_metrics

    marginal_ll_kwargs     = marginal_ll_kwargs     or {}
    cf_extra_kwargs        = cf_extra_kwargs        or {}
    batch_size_eval_factor = batch_size_eval_factor or 1

    _domains_key = domains_key if domains_key is not None else DOMAINS_KEY
    _labels_key  = labels_key  if labels_key  is not None else LABELS_KEY

    eval_batch_size = BATCH_SIZE * batch_size_eval_factor

    # ── Per-run scalars ───────────────────────────────────────────────────────
    print(f"  Evaluating marginal log-likelihood on test set...")
    marginal_ll = model.get_marginal_ll(
        adata, indices=model.test_indices_, batch_size=eval_batch_size, **marginal_ll_kwargs
    )
    print(f"  marginal_ll={marginal_ll:.4f}")

    adata.obsm["s"] = model.get_latent_representation(latent_key="s", batch_size=eval_batch_size)
    rng = np.random.default_rng(seed=42)
    sub_idx = rng.choice(adata.n_obs, size=int(adata.n_obs * 0.25), replace=False)
    ari_nmi = scib_metrics.nmi_ari_cluster_labels_kmeans(
        labels=adata.obs[_domains_key].values[sub_idx],
        X=adata.obsm["s"][sub_idx],
    )
    ari = float(ari_nmi["ari"])
    nmi = float(ari_nmi["nmi"])
    print(f"  ARI={ari:.4f}  NMI={nmi:.4f}")

    ari_nmi_s_labels = scib_metrics.nmi_ari_cluster_labels_kmeans(
        labels=adata.obs[_labels_key].values[sub_idx],
        X=adata.obsm["s"][sub_idx],
    )
    ari_s_labels = float(ari_nmi_s_labels["ari"])
    nmi_s_labels = float(ari_nmi_s_labels["nmi"])
    print(f"  ARI_s_labels={ari_s_labels:.4f}  NMI_s_labels={nmi_s_labels:.4f}")

    adata.obsm["z"] = model.get_latent_representation(latent_key="z", batch_size=eval_batch_size)
    ari_nmi_z_domains = scib_metrics.nmi_ari_cluster_labels_kmeans(
        labels=adata.obs[_domains_key].values[sub_idx],
        X=adata.obsm["z"][sub_idx],
    )
    ari_z_domains = float(ari_nmi_z_domains["ari"])
    nmi_z_domains = float(ari_nmi_z_domains["nmi"])
    print(f"  ARI_z_domains={ari_z_domains:.4f}  NMI_z_domains={nmi_z_domains:.4f}")

    # ── Build target domain pools ─────────────────────────────────────────────
    domains = [d for d in adata.obs[_domains_key].astype(str).unique() if d != "nan"]
    if ref_domain is not None:
        ref_label = ref_domain
    else:
        ref_label = next(d for d in domains if "REF" in d)

    if target_domains is not None:
        target_labels = list(target_domains)
    else:
        crc_label  = next(d for d in domains if "CRC" in d)
        tva_labels = [d for d in domains if "TVA" in d]
        target_labels = [crc_label] + tva_labels

    cell_types = [
        ct for ct in adata.obs[_labels_key].cat.categories
        if ((adata.obs[_domains_key] == ref_label) & (adata.obs[_labels_key] == ct)).any()
        and any(
            ((adata.obs[_domains_key] == tl) & (adata.obs[_labels_key] == ct)).any()
            for tl in target_labels
        )
    ]

    target_pools = {
        tl: np.where(adata.obs[_domains_key].astype(str) == tl)[0]
        for tl in target_labels
    }

    print(f"  Evaluating {len(cell_types)} cell types across {len(target_labels)} targets...")

    rows = []
    for target_label in target_labels:
        target_all_idx = target_pools[target_label]
        if target_domains is None:
            target_short = next(k for k in ("CRC", "TVA") if k in target_label)
        else:
            target_short = target_label

        for ct in sorted(cell_types):
            ref_mask    = (adata.obs[_labels_key] == ct) & (adata.obs[_domains_key] == ref_label)
            target_mask = (adata.obs[_labels_key] == ct) & (adata.obs[_domains_key] == target_label)

            if ct == holdout_celltype:
                target_mask = target_mask & adata.obs["is_holdout"]

            ref_idx    = np.where(ref_mask.values)[0]
            target_idx = np.where(target_mask.values)[0]

            if len(ref_idx) < MIN_CELLS or len(target_idx) < MIN_CELLS:
                print(f"  skip {ct} ({target_short}): ref={len(ref_idx)}, target={len(target_idx)}")
                continue

            print(f"  {ct} ({target_short}): ref={len(ref_idx)}, target={len(target_idx)}")

            ref_arr  = adata[ref_idx].layers["counts"]
            ref_expr = ref_arr.toarray() if hasattr(ref_arr, "toarray") else np.asarray(ref_arr)
            ref_expr = _normalize_total(ref_expr)

            tgt_arr  = adata[target_idx].layers["counts"]
            cf_expr  = tgt_arr.toarray() if hasattr(tgt_arr, "toarray") else np.asarray(tgt_arr)
            cf_expr  = _normalize_total(cf_expr)

            print(f"    computing counterfactuals...", flush=True)
            pert_expr = model.get_counterfactual_expression(
                indices=ref_idx,
                neighbour_indices=target_all_idx,
                batch_size=BATCH_SIZE * batch_size_eval_factor,
                library_size=LIBRARY_SIZE,
                **cf_extra_kwargs,
            )

            stats = compute_cf_logfc(
                ref_expr, pert_expr, cf_expr,
                top_n=TOP_N,
                gene_names=adata.var_names.tolist(),
            )
            print(f"    pearson_r={stats['pearson_r']:.4f}  spearman_r={stats['spearman_r']:.4f}  prec={stats['precision']:.2f}")

            rows.append(dict(
                holdout_celltype=holdout_celltype,
                spatial_loss_type=spatial_loss_type,
                link_prediction_weight=link_prediction_weight,
                target=target_short,
                cell_type=ct,
                is_holdout=(ct == holdout_celltype),
                n_ref=len(ref_idx),
                n_target=len(target_idx),
                pearson_r=stats["pearson_r"],
                spearman_r=stats["spearman_r"],
                precision=stats["precision"],
                mixing_index=stats["mixing_index"],
                edistance=stats["edistance"],
                rmse=stats["rmse"],
                marginal_ll=marginal_ll,
                ari=ari,
                nmi=nmi,
                ari_s_labels=ari_s_labels,
                nmi_s_labels=nmi_s_labels,
                ari_z_domains=ari_z_domains,
                nmi_z_domains=nmi_z_domains,
            ))

    return rows


def _plot_lines(ax, df, x_col, y_col, title, xlabel="link_prediction_weight"):
    """Plot one line per spatial_loss_type on ax."""
    for slt, grp in df.groupby("spatial_loss_type"):
        grp = grp.sort_values(x_col)
        ax.plot(
            grp[x_col], grp[y_col],
            marker=SLT_MARKERS.get(slt, "o"),
            color=SLT_COLORS.get(slt, "gray"),
            linewidth=1.8,
            label=slt,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(y_col, fontsize=8)
    ax.legend(fontsize=7)


def generate_pdf(results_df, out_path, holdout_celltypes):
    """Per-holdout pages + summary pages."""
    with PdfPages(out_path) as pdf:

        # ── Per-holdout pages ─────────────────────────────────────────────────
        for holdout_ct in holdout_celltypes:
            sub = results_df[results_df["holdout_celltype"] == holdout_ct]
            if sub.empty:
                continue

            for target in sorted(sub["target"].unique()):
                tsub = sub[sub["target"] == target]

                holdout_sub = tsub[tsub["is_holdout"]].copy()
                nonhold_sub = tsub[~tsub["is_holdout"]].copy()

                nonhold_mean = (
                    nonhold_sub
                    .groupby(["link_prediction_weight", "spatial_loss_type"])[CF_METRICS]
                    .mean()
                    .reset_index()
                )
                run_sub = (
                    tsub.drop_duplicates(["link_prediction_weight", "spatial_loss_type"])
                    [["link_prediction_weight", "spatial_loss_type"] + RUN_METRICS]
                )

                n_metrics = len(METRICS)
                ncols = 3
                nrows = (n_metrics + ncols - 1) // ncols

                fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
                axes = axes.flatten()

                for i, metric in enumerate(METRICS):
                    ax = axes[i]
                    if metric in RUN_METRICS:
                        _plot_lines(ax, run_sub, "link_prediction_weight", metric, metric)
                    else:
                        for slt, grp in holdout_sub.groupby("spatial_loss_type"):
                            grp = grp.sort_values("link_prediction_weight")
                            ax.plot(
                                grp["link_prediction_weight"], grp[metric],
                                marker=SLT_MARKERS.get(slt, "o"),
                                color=SLT_COLORS.get(slt, "gray"),
                                linewidth=2,
                                label=f"{holdout_ct} (holdout) [{slt}]",
                            )
                        for slt, grp in nonhold_mean.groupby("spatial_loss_type"):
                            grp = grp.sort_values("link_prediction_weight")
                            ax.plot(
                                grp["link_prediction_weight"], grp[metric],
                                marker=SLT_MARKERS.get(slt, "o"),
                                color=SLT_COLORS.get(slt, "gray"),
                                linewidth=1.5, linestyle="--",
                                label=f"non-holdout mean [{slt}]",
                            )
                        ax.set_title(metric, fontsize=10)
                        ax.set_xlabel("link_prediction_weight", fontsize=8)
                        ax.set_ylabel(metric, fontsize=8)
                        ax.legend(fontsize=6)

                for j in range(n_metrics, len(axes)):
                    axes[j].set_visible(False)

                fig.suptitle(f"Holdout: {holdout_ct}  |  target: {target}", fontsize=13, y=1.01)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # ── Summary pages (one per target) ────────────────────────────────────
        for target in sorted(results_df["target"].unique()):
            tsub = results_df[results_df["target"] == target]

            holdout_agg = (
                tsub[tsub["is_holdout"]]
                .groupby(["link_prediction_weight", "spatial_loss_type"])[CF_METRICS]
                .mean()
                .reset_index()
            )
            id_agg = (
                tsub[~tsub["is_holdout"]]
                .groupby(["link_prediction_weight", "spatial_loss_type"])[CF_METRICS]
                .mean()
                .reset_index()
            )
            run_agg = (
                tsub
                .drop_duplicates(["holdout_celltype", "link_prediction_weight", "spatial_loss_type"])
                .groupby(["link_prediction_weight", "spatial_loss_type"])[RUN_METRICS]
                .mean()
                .reset_index()
            )

            sections = [
                ("LOO cell type (avg)", holdout_agg, CF_METRICS),
                ("ID cell types (avg)",  id_agg,      CF_METRICS),
                ("Per-run scalars",       run_agg,     RUN_METRICS),
            ]

            ncols = 4
            nrows = len(sections)
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))

            for row_i, (section_title, df_agg, metrics) in enumerate(sections):
                for col_i in range(ncols):
                    ax = axes[row_i, col_i]
                    if col_i < len(metrics):
                        metric = metrics[col_i]
                        _plot_lines(ax, df_agg, "link_prediction_weight", metric, metric)
                        if col_i == 0:
                            ax.set_ylabel(section_title, fontsize=9, labelpad=10)
                    else:
                        ax.set_visible(False)

            fig.suptitle(f"Summary  |  target: {target}", fontsize=14, y=1.01)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved PDF → {out_path}")
