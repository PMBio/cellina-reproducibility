"""
Subdomain counterfactual analysis and dumbbell plots.

Counterfactuals here refer to edge-swapping counterfactuals (get_counterfactual_expression /
get_counterfactual_latents), distinct from neighbourhood perturbations (make_neighbor_perturbation /
get_perturbed_expression) used in 02_pathway_analysis.ipynb.

For each slide (assuming 01_data_prep.ipynb has been run):
  - Loads adata with microenvironment labels from output/adata_with_microenv.h5ad
  - Loads the trained model from its checkpoint
  - Computes edge-swapping counterfactuals for control cells of each cell type,
    conditioned on (1) the global CRC neighbourhood and (2) each CRC subdomain/microenvironment
  - Evaluates by correlating counterfactual vs. observed log-fold changes (top-200 DEGs)
  - Saves per-slide CSV to ../../results/microenvironments_{slide_id}.csv

After all slides:
  - Per-slide dumbbell SVGs (Pearson + Spearman)
  - Multi-slide aggregate pointplot saved to ../../figures/application/

Usage
-----
    python 03_domain_perturbations.py                        # crc_210 only
    python 03_domain_perturbations.py --slides crc_210 crc_xxx
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))

import hotspot
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scvi.train._callbacks import SaveCheckpoint, EarlyStopping
from cellina import CellinaModel
from cellina._spatial_utils import spatial_neighbors, compute_spatial_features
from counterfactual_analysis import compute_correlations, safe_log2_fold_change
from utils import set_seed
from perturb_utils import load_crc_slide

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 20
plt.rcParams['figure.dpi'] = 100

# ── Config ────────────────────────────────────────────────────────────────────
LABELS_KEY     = 'coarse_type'
DOMAINS_KEY    = 'typ_clean'
RESULTS_PATH   = '../../results'
FIG_SAVE_PATH  = '../../figures/application'
MODEL_BASE_DIR = 'data/cellina-reproducibility/application'   # {slide_id} appended

CELLTYPES = ['Endothelial', 'Epithelial', 'Fibroblast', 'Myeloid', 'T_cell']
DEG = 200


# ── Spatial preprocessing ────────────────────────────────────────────────────

def preprocess_for_cellina(adata, n_neighbors=200):
    """Add spatial neighbourhood features required by CellinaModel.

    load_crc_slide already normalises + log-transforms + selects HVGs.
    This adds the two missing steps: spatial KNN graph and spatial_x features.
    """
    spatial_neighbors(adata, bandwidth=100 / 0.12028,
                      max_neighbours=n_neighbors, standardize=False)
    compute_spatial_features(adata)          # writes adata.obsm['spatial_x']
    adata.X = adata.layers['counts'].copy()  # reset X to raw counts for model


# ── Model training / loading ──────────────────────────────────────────────────

def train_or_load_model(adata, slide_id):
    """Load an existing checkpoint or train a new CellinaModel for the slide."""
    model_base_path = f"{MODEL_BASE_DIR}/{slide_id}"

    if os.path.isdir(model_base_path):
        checkpoints = [f for f in os.listdir(model_base_path) if not f.startswith('.')]
        if checkpoints:
            model = CellinaModel.load(
                f"{model_base_path}/{checkpoints[0]}", adata=adata
            )
            print("Loaded existing model checkpoint")
            return model

    print("No checkpoint found — training model from scratch ...")

    if 'sid' not in adata.obs:
        adata.obs['sid'] = slide_id

    n = adata.n_obs
    test_idx = np.random.choice(n, int(n * 0.1), replace=False)
    trainval_idx = np.setdiff1d(np.arange(n), test_idx)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.1, random_state=0, shuffle=True
    )

    CellinaModel.setup_anndata(
        adata,
        batch_key='sid',
        labels_key=LABELS_KEY,
        domains_key=DOMAINS_KEY,
        spatial_obsm_key='spatial_x',
        layer='counts',
    )

    os.makedirs(model_base_path, exist_ok=True)
    model = CellinaModel(
        adata=adata,
        n_latent=64,
        n_layers=3,
        use_observed_lib_size=True,
        condition_on_intrinsic=False,
        gene_likelihood='nb',
        classifier_lambda=1.,
        discriminator_lambda=1.,
    )
    model.train(
        max_epochs=100,
        batch_size=4096,
        check_val_every_n_epoch=1,
        early_stopping=True,
        datasplitter_kwargs={'external_indexing': [train_idx, val_idx, test_idx]},
        enable_checkpointing=True,
        callbacks=[
            SaveCheckpoint(
                monitor='vae_loss_validation',
                dirpath=model_base_path,
                load_best_on_end=True,
            ),
            EarlyStopping(
                monitor='vae_loss_validation',
                patience=5,
                mode='min',
            ),
        ],
        plan_kwargs={'lr': 1e-3, 'normalize_losses': True},
    )
    print("Model trained and saved")
    return model


# ── Microenvironment labeling ─────────────────────────────────────────────────

def compute_microenvironments(adata, model, domains_key, n_neighbors=30,
                              top_k=1200, fdr_threshold=0.05,
                              min_gene_threshold=100, jobs=8):
    """Run Hotspot on the CRC sub-population to assign microenvironment labels."""
    adata.obsm['cellina_spatial'] = model.get_latent_representation(
        adata=adata, latent_key='s', batch_size=4096
    )

    adata_crc = adata[adata.obs[domains_key].str.contains('CRC', regex=True)].copy()

    if 'nCount_RNA' not in adata_crc.obs:
        adata_crc.obs['nCount_RNA'] = np.asarray(
            adata_crc.layers['counts'].sum(axis=1)
        ).ravel()

    hs = hotspot.Hotspot(
        adata_crc,
        layer_key='counts',
        model='danb',
        latent_obsm_key='cellina_spatial',
        umi_counts_obs_key='nCount_RNA',
    )
    hs.create_knn_graph(weighted_graph=False, n_neighbors=n_neighbors)

    hs_results = hs.compute_autocorrelations(jobs=jobs)
    hs_genes = hs_results.loc[hs_results.FDR < fdr_threshold].head(top_k).index

    hs.compute_local_correlations(hs_genes, jobs=jobs)
    hs.create_modules(min_gene_threshold=min_gene_threshold,
                      core_only=True, fdr_threshold=fdr_threshold)
    module_scores = hs.calculate_module_scores()

    top_modules = module_scores.idxmax(axis=1)
    adata_crc.obs['microenvironment'] = top_modules.apply(lambda x: f"CRC{x}")

    # Propagate back to full adata (non-CRC cells get NaN)
    adata.obs['microenvironment'] = adata_crc.obs['microenvironment'].reindex(adata.obs_names)


# ── Per-slide function ────────────────────────────────────────────────────────

def per_slide(slide_id):
    print(f"\n{'='*60}")
    print(f"Processing slide: {slide_id}")
    print(f"{'='*60}")

    set_seed(0)

    # ── Load adata ────────────────────────────────────────────────────────────
    numeric_id = int(slide_id.split('_')[-1])
    adata = load_crc_slide(
        slide_id=numeric_id,
        data_dir='../../data/crc_wt_cosmx',
    )
    # Derive typ_clean (REF / TVA / CRC) from the raw typ column
    adata.obs[DOMAINS_KEY] = adata.obs['typ'].str.extract(r'(REF|TVA|CRC)', expand=False)
    print(f"Loaded adata: {adata}")

    # ── Add spatial features + reset X to counts ──────────────────────────────
    preprocess_for_cellina(adata)

    # ── Train or load model ───────────────────────────────────────────────────
    model = train_or_load_model(adata, slide_id)

    # ── Compute microenvironment labels via Hotspot ───────────────────────────
    compute_microenvironments(adata, model, DOMAINS_KEY)
    print(f"Microenvironments: {adata.obs['microenvironment'].value_counts().to_dict()}")

    # ── Build results dict and derive microenvironments ───────────────────────
    results = {ct: adata[adata.obs[LABELS_KEY] == ct] for ct in CELLTYPES}
    microenvironments = [
        m for m in adata.obs['microenvironment'].unique() if 'CRC' in str(m)
    ]
    print(f"Microenvironments: {microenvironments}")

    # ── Counterfactual loop (cells 58–64 logic) ───────────────────────────────
    is_tumor_region = adata.obs[DOMAINS_KEY].astype(str).str.contains('CRC', regex=True)

    for ct in CELLTYPES:
        print(f"  Computing counterfactuals for {ct} ...")
        is_celltype = adata.obs[LABELS_KEY].astype(str) == ct
        idx_control = np.where((~is_tumor_region & is_celltype).values)[0]

        results[ct].obsm['recon_x'] = model.get_normalized_expression(
            adata=results[ct], batch_size=4096, library_size=1e4
        )

        # 1. Global CRC counterfactual
        idx_target_global = np.where(is_tumor_region.values)[0]
        args = dict(
            adata=adata,
            indices=idx_control,
            neighbour_indices=idx_target_global,
            batch_size=4096,
            seed=0,
        )
        results[ct].uns['counterfactual_x_global'] = model.get_counterfactual_expression(
            **args, library_size=1e4
        )
        results[ct].uns['counterfactual_latents_global'] = model.get_counterfactual_latents(
            **args, latent_key='shifted'
        )

        # 2. Per-microenvironment counterfactuals
        for microenv in microenvironments:
            is_in_microenv = adata.obs['microenvironment'].astype(str).str.contains(
                microenv, regex=True
            )
            idx_target = np.where(is_in_microenv.values)[0]
            args['neighbour_indices'] = idx_target
            results[ct].uns[f'counterfactual_x_{microenv}'] = model.get_counterfactual_expression(
                **args, library_size=1e4
            )
            results[ct].uns[f'counterfactual_latents_{microenv}'] = model.get_counterfactual_latents(
                **args, latent_key='shifted'
            )

    # ── Compute correlations ──────────────────────────────────────────────────
    summary = []

    for ct, dataset in results.items():
        mask_control = ~dataset.obs[DOMAINS_KEY].astype(str).str.contains('CRC', regex=True)
        control = np.asarray(dataset.layers['counts'].todense()[mask_control])

        mask_target_global = is_tumor_region
        target_global = np.asarray(adata.layers['counts'].todense()[mask_target_global])

        cf_global = dataset.uns['counterfactual_x_global']
        pear_global, spear_global = compute_correlations(control, target_global, cf_global, deg=DEG)
        summary.append({
            'cell_type': ct, 'label': 'CRC_global',
            'pearson': round(pear_global, 4), 'spearman': round(spear_global, 4),
        })

        for microenv in microenvironments:
            is_in_microenv = adata.obs['microenvironment'].astype(str).str.contains(
                microenv, regex=True
            )
            target = np.asarray(adata.layers['counts'].todense()[is_in_microenv])
            cf = dataset.uns[f'counterfactual_x_{microenv}']
            pear, spear = compute_correlations(control, target, cf, deg=DEG)
            summary.append({
                'cell_type': ct, 'label': microenv,
                'pearson': round(pear, 4), 'spearman': round(spear, 4),
            })

    summary_df = pd.DataFrame(summary)
    print(summary_df)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_PATH, exist_ok=True)
    csv_path = f"{RESULTS_PATH}/microenvironments_{slide_id}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # ── Per-slide dumbbell plots ──────────────────────────────────────────────
    plot_dumbbell_single(summary_df, slide_id)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_dumbbell_single(summary_df, slide_id):
    """Single-slide dumbbell plots (Pearson+Spearman side-by-side, and Spearman alone)."""
    os.makedirs(FIG_SAVE_PATH, exist_ok=True)
    cell_types = summary_df['cell_type'].unique()
    colors = {'CRC_global': '#0072B2', 'mean_others': '#D55E00'}

    # Pearson + Spearman side-by-side
    plot_data = []
    for corr_type in ['pearson', 'spearman']:
        tmp = summary_df.copy()
        global_vals = tmp[tmp['label'] == 'CRC_global'].set_index('cell_type')[corr_type]
        mean_others = tmp[tmp['label'] != 'CRC_global'].groupby('cell_type')[corr_type].mean()
        plot_data.append((global_vals, mean_others))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, (global_vals, mean_others), title in zip(
        axes, plot_data, [r"Pearson $r$", r"Spearman $\rho$"]
    ):
        y_pos = np.arange(len(cell_types))
        ax.hlines(y=y_pos, xmin=mean_others.values, xmax=global_vals.values,
                  color='gray', alpha=1, linewidth=2)
        ax.scatter(global_vals.values, y_pos, color=colors['CRC_global'],
                   s=100, marker='o', label='Global CRC')
        ax.scatter(mean_others.values, y_pos, color=colors['mean_others'],
                   s=100, marker='D', label='Within-microenvironment')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cell_types)
        ax.set_xlim(0.4, 0.9)
        ax.set_xlabel(title, fontsize=14)

    fig.suptitle("Global vs. Microenvironment-specific predictions",
                 fontsize=16, fontweight='bold', y=0.9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.86),
               ncol=2, frameon=False, fontsize=14)
    fig.subplots_adjust(top=0.88)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out = f"{FIG_SAVE_PATH}/dumbbell_{slide_id}.svg"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

    # Spearman only
    tmp = summary_df.copy()
    global_vals = tmp[tmp['label'] == 'CRC_global'].set_index('cell_type')['spearman']
    mean_others = tmp[tmp['label'] != 'CRC_global'].groupby('cell_type')['spearman'].mean()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    y_pos = np.arange(len(cell_types))
    ax.hlines(y=y_pos, xmin=mean_others.values, xmax=global_vals.values,
              color='gray', linewidth=2)
    ax.scatter(global_vals.values, y_pos, color='#0072B2', s=100, marker='o', label='CRC global')
    ax.scatter(mean_others.values, y_pos, color='#D55E00', s=100, marker='D', label='CRC subtype (mean)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cell_types)
    ax.set_xlim(0.4, 0.9)
    ax.set_xlabel(r"Spearman $\rho$", fontsize=14)
    fig.suptitle("CRC global vs. subtype predictions", fontsize=16, fontweight='bold', y=0.92)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.88),
               ncol=2, frameon=False, fontsize=14)
    fig.subplots_adjust(top=0.82)
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    out = f"{FIG_SAVE_PATH}/dumbbell_spearman_{slide_id}.svg"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def plot_aggregate_dumbbell(results_path=RESULTS_PATH, fig_save_path=FIG_SAVE_PATH):
    """Aggregate multi-slide pointplot"""
    all_files = glob.glob(f"{results_path}/microenvironments_*.csv")
    if not all_files:
        print(f"No result CSVs found in {results_path}")
        return

    dfs = []
    for fpath in all_files:
        df = pd.read_csv(fpath)
        df['slide_id'] = fpath.split("_")[-1].split(".")[0]
        dfs.append(df)
    dumbbell_df = pd.concat(dfs, ignore_index=True)

    tmp = dumbbell_df.copy()
    tmp['group'] = np.where(tmp['label'] == 'CRC_global', 'CRC global', 'CRC subtype')
    plot_df = tmp[['cell_type', 'group', 'spearman']]

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {'CRC global': '#0072B2', 'CRC subtype': '#D55E00'}

    sns.pointplot(
        data=plot_df, x='spearman', y='cell_type', hue='group',
        palette=palette, dodge=0.45, markers='|', linestyles='none',
        errorbar=('ci', 95), err_kws={'linewidth': 2.5},
        markersize=16, markeredgewidth=3.0, ax=ax,
    )

    ax.set_xlim(0.4, 0.9)
    ax.set_xticks([0.4, 0.6, 0.8])
    ax.set_xlabel(r"Spearman $\rho$", fontsize=26)
    ax.set_ylabel("", fontsize=26)
    ax.tick_params(axis='both', labelsize=22)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=True)

    fig.suptitle("CRC global vs. subtype predictions", fontsize=26, fontweight='bold', y=0.97)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc='upper center', bbox_to_anchor=(0.5, 0.91),
               ncol=2, frameon=False, fontsize=22, handletextpad=0.4, columnspacing=1.0)
    ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    os.makedirs(fig_save_path, exist_ok=True)
    out = f"{fig_save_path}/boxplot_spearman.svg"
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved aggregate dumbbell to {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Submit all: python notebooks/application/03_subdomain_counterfactuals.py --slides 242 232 231 210 221 120"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--slides', nargs='+', default=['crc_210'], # all slides: ['242', '232', '231', '210', '221', '120']
        help='Slide IDs to process (default: crc_210)',
    )
    parser.add_argument(
        '--plot-only', action='store_true',
        help='Skip analysis; regenerate plots from existing CSVs in RESULTS_PATH',
    )
    args = parser.parse_args()

    if args.plot_only:
        for fpath in sorted(glob.glob(f"{RESULTS_PATH}/microenvironments_*.csv")):
            slide_id = os.path.basename(fpath).replace('microenvironments_', '').replace('.csv', '')
            plot_dumbbell_single(pd.read_csv(fpath), slide_id)
        plot_aggregate_dumbbell()
        return

    for slide_id in args.slides:
        per_slide(slide_id)

    plot_aggregate_dumbbell()


if __name__ == '__main__':
    main()
