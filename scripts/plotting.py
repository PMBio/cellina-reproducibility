import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import decoupler as dc

from itertools import combinations
from scipy.stats import mannwhitneyu
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(y_true, y_pred, label_mapping, normalize=False, cmap="Blues"):
    """
    Plot confusion matrix with option to normalize by true labels.

    Args:
        y_true: ground truth labels (array-like)
        y_pred: predicted labels (array-like)
        label_mapping: dict mapping int -> str for labels
        normalize: bool, if True matrix is row-normalized to proportions
        cmap: matplotlib colormap
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    disp = ConfusionMatrixDisplay(
        cm, display_labels=[label_mapping[i] for i in range(len(label_mapping))]
    )

    _, ax = plt.subplots(figsize=(8, 8))
    # If normalized → show with 2 decimals, else as integers
    fmt = ".2f" if normalize else "d"
    disp.plot(ax=ax, cmap=cmap, values_format=fmt, colorbar=True)

    # Hide sklearn’s default annotations
    for txt in disp.text_.flatten():
        txt.set_visible(False)

    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.show()


def plot_roc_curves(y_true, y_probs, label_mapping, macro_avg=False):
    # Binarize labels for multi-class ROC
    y_bin = label_binarize(y_true, classes=np.arange(len(label_mapping)))
    n_classes = y_bin.shape[1]

    # Compute ROC curve for each class
    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
    plt.figure(figsize=(8, 8))
    for i, label in enumerate(label_mapping.values()):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        roc_auc_dict[i] = roc_auc
        plt.plot(fpr, tpr, lw=1, alpha=0.5, label=f"{label} (AUC={roc_auc:.2f})")

    plt.legend(loc="lower right")
    plt.title("Multi-class ROC curves")

    if macro_avg:
        # Compute macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= n_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)

        # Plot macro-average ROC
        plt.plot(
            all_fpr,
            mean_tpr,
            color="black",
            linestyle="--",
            lw=2,
            label=f"Macro-average (AUC={roc_auc_macro:.2f})",
        )
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-class ROC with Macro-average")
        plt.legend(loc="lower right")
        plt.show()


def plot_custom_umap(
    adata,
    subsample=None,
    recompute=False,
    use_rep=None,
    rep_dims=None,  # <-- NEW: list of indices to use from use_rep
    clean=False,
    random_state=42,
    **kwargs,
):
    """
    Plot UMAP embedding from AnnData.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    subsample : float or None
        Fraction of cells to subsample for plotting.
    recompute : bool
        Whether to recompute neighbors and UMAP.
    use_rep : str or None
        Key in `adata.obsm` to use as input representation.
    rep_dims : list of int or None
        If provided, restrict to these component indices of use_rep.
    clean : bool
        Whether to remove axis labels, ticks, and spines.
    random_state : int
        Seed for reproducible subsampling.
    **kwargs : dict
        Additional args passed to `sc.pl.umap`.
    """
    # Subsample
    if subsample is not None and 0 < subsample < 1:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(adata.n_obs, size=int(adata.n_obs * subsample), replace=False)
        adata_plot = adata[idx].copy()
    else:
        adata_plot = adata

    # Handle rep_dims
    if use_rep is not None and rep_dims is not None:
        rep_matrix = adata_plot.obsm[use_rep][:, rep_dims]
        # Store a temporary representation
        tmp_key = f"{use_rep}_subset"
        adata_plot.obsm[tmp_key] = rep_matrix
        use_rep_final = tmp_key
    else:
        use_rep_final = use_rep

    # Recompute neighbors/UMAP if requested or missing
    if recompute or "X_umap" not in adata_plot.obsm:
        sc.pp.neighbors(adata_plot, use_rep=use_rep_final)
        sc.tl.umap(adata_plot)

    # Plot
    if clean:
        axes = sc.pl.umap(adata_plot, show=False, **kwargs)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    else:
        sc.pl.umap(adata_plot, **kwargs)

    plt.show()


def cliff_delta(x, y):
    """Compute Cliff's delta: non-parametric effect size"""
    n_x = len(x)
    n_y = len(y)
    more = sum(xi > yi for xi in x for yi in y)
    less = sum(xi < yi for xi in x for yi in y)
    delta = (more - less) / (n_x * n_y)
    return delta


def plot_autocorr(
    df,
    plot="violin",
    class_col="class",
    value_col="C",
    title="Autocorrelation score by gene class",
    sig_level=0.05,
    palette=None,
    figsize=(8, 4),
):
    """
    Violin plot with pairwise Wilcoxon tests and Cliff's delta for any number of classes.
    Genes belonging to multiple classes are counted in all relevant groups.
    """

    # --- Explode list column if needed
    df_plot = df.copy()
    if df_plot[class_col].apply(lambda x: isinstance(x, list)).any():
        df_plot = df_plot.explode(class_col)

    # --- Automatically determine groups
    groups = sorted(df_plot[class_col].unique())
    if palette is None:
        # generate a default color palette
        palette = dict(zip(groups, sns.color_palette("Set2", n_colors=len(groups))))

    # --- Pairwise tests
    pairwise_results = []
    for g1, g2 in combinations(groups, 2):
        vals1 = df_plot[df_plot[class_col] == g1][value_col].values
        vals2 = df_plot[df_plot[class_col] == g2][value_col].values
        _, pval = mannwhitneyu(vals1, vals2, alternative="two-sided")
        delta = cliff_delta(vals1, vals2)
        pairwise_results.append((g1, g2, pval, delta))

    # --- Prepare annotation with bold for significant
    text_lines = []
    for g1, g2, pval, delta in pairwise_results:
        if pval < sig_level:
            line = f"$\\bf{{{g1}~vs~{g2}:~p={pval:.2e},~\\delta={delta:.2f}}}$"
        else:
            line = f"{g1} vs {g2}: p={pval:.2e}, δ={delta:.2f}"
        text_lines.append(line)

    # --- Plot violin
    plt.figure(figsize=figsize)

    if plot == "violin":
        sns.violinplot(
            data=df_plot,
            y=class_col,
            x=value_col,
            order=groups,
            palette=palette,
            hue=class_col,
            orient="h",
            inner="box",
            cut=0,  # clip KDE at min/max
        )
    else:
        sns.boxplot(
            data=df_plot,
            y=class_col,
            x=value_col,
            order=groups,
            palette=palette,
            orient="h",
            showfliers=True,  # show outliers as points
        )

    plt.text(
        0.98,
        0.95,
        "\n".join(text_lines),
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

    plt.xlabel(f"{value_col} (autocorrelation score)")
    plt.ylabel("")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pathway_activity(pw_acts, pw_padj, alpha=0.05):
    for sample in pw_acts.index:
        activity = pw_acts.loc[sample]
        padj = pw_padj.loc[sample]

        # Filter by significance
        sig_mask = padj < alpha
        activity = activity[sig_mask]

        if activity.empty or sample=="-1.0":
            continue
        
        df = activity.to_frame().T
        df.index = [activity.name]
        fig = dc.pl.barplot(data=df, name=str(sample), figsize=(9, 5), return_fig=True)
        ax = fig.axes[0]
        ax.set_title(f"Pathway activities for Module {sample}")


def plot_model_comparison(
    dataset_name,
    df,
    metrics,
    grouping,
    model_order,
    palette,
    average_only=False,
    save=False,
    save_path="../figures",
    save_name="ood_summary"
):
    # Compute global y-limits
    ymin = df[metrics].min().min() * 0.3
    ymax = df[metrics].max().max() * 1.05

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    n_deg = df["n_deg"].iloc[0]
    title = f"{dataset_name}: Performance (observed vs predicted) for DEG {n_deg}"
    fig.suptitle(title, fontsize=16)

    # Ensure axes iterable if only one metric
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):

        if not average_only:
            # -------------------------------
            # Per-celltype plot
            # -------------------------------
            celltypes = df[grouping].unique().tolist()
            all_x_labels = list(celltypes) + ["Average"]

            sns.barplot(
                data=df,
                x=grouping,
                y=metric,
                hue="model_name",
                estimator="mean",
                errorbar="se",
                hue_order=model_order,
                ax=ax,
                palette=palette,
                order=celltypes,
            )
            ax.legend_.remove()

            # -------------------------------
            # Add averages manually (WITH SE)
            # -------------------------------
            df_avg = df.groupby("model_name")[metric].mean()
            df_se = df.groupby("model_name")[metric].sem()

            n_models = len(model_order)
            bar_width = 0.8 / n_models

            for i, model in enumerate(model_order):
                avg_val = df_avg.loc[model]
                se_val = df_se.loc[model]

                x_pos = len(celltypes) - 0.4 + i * bar_width + bar_width / 2

                ax.bar(
                    x=x_pos,
                    height=avg_val,
                    width=bar_width,
                    color=palette[model],
                    alpha=0.9,
                    yerr=se_val,
                    capsize=0,
                )

            ax.set_xticks(range(len(all_x_labels)))
            ax.set_xticklabels(all_x_labels, rotation=45)

        else:
            # -------------------------------
            # AVERAGE-ONLY plot (clean)
            # -------------------------------
            df_avg = (
                df.groupby(["model_name"])[metric]
                .agg(["mean", "sem"])
                .reset_index()
            )

            sns.barplot(
                data=df_avg,
                x="model_name",
                y="mean",
                hue="model_name",
                order=model_order,
                palette=palette,
                errorbar=None,  # we add manually
                ax=ax,
            )

            # Add SE manually
            for i, model in enumerate(model_order):
                row = df_avg[df_avg.model_name == model].iloc[0]
                ax.errorbar(
                    x=i,
                    y=row["mean"],
                    yerr=row["sem"],
                    fmt="none",
                    capsize=0,
                    color="black",
                )

            #ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.tick_params(axis="x", rotation=45)

        # -------------------------------
        # Shared styling
        # -------------------------------
        ax.set_ylim(ymin, ymax)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # -------------------------------
    # Global legend (only once)
    # -------------------------------
    if not average_only:
        handles, labels = axes[0].get_legend_handles_labels()
    else:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=palette[m])
            for m in model_order
        ]
        labels = model_order

    fig.legend(
        handles,
        labels,
        ncol=len(labels),
        fontsize=10,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)

    if save:
        plt.savefig(f"{save_path}/{save_name}.svg", format="svg", bbox_inches="tight")
    plt.show()