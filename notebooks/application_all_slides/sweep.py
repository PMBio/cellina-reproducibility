"""Gene-selection sweep for the joint all-slides Hotspot modules.

Question: do the modules -- and the pathways they map to -- depend on how the gene set is
chosen? Two knobs: HOTSPOT_TOP_K (how many autocorrelated genes enter the local-correlation
matrix) and HOTSPOT_MIN_GENES (how small a module may be).

Two stages, both resumable:

  prep   load + join (HVG union) -> per-slide spatial graph -> train (no holdout) -> latents
         -> slide-balanced CRC subsample written to output/sweep/fit_cells.h5ad
  sweep  fit Hotspot once, compute local correlations once on the largest TOP_K, then slice
         that matrix per TOP_K and re-cluster per MIN_GENES

The slicing is the whole reason this is cheap. compute_local_correlations is O(k^2) and
dominates everything (13 min at k=1000); create_modules is seconds. The z-score of a gene
pair depends only on that pair (per-gene DANB fit + the shared KNN graph), so the k=500
matrix is literally the top-left block of the k=1000 one -- asserted below on a small gene
set before the expensive call.

No counterfactuals: module identity flows results['C'] + modules -> decoupler, and
counterfactuals never enter that chain. They are model-validation figures, not inputs.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint
from joint import (BATCH_KEY, DOMAINS_KEY, LABELS_KEY, SLIDES, build_spatial,
                   cap_per_group, load_joint, module_pathway_assignment, slim)

OUT = 'output/sweep'
FIG = '../../figures/application_all_slides'

N_TOP_GENES, MIN_SLIDES = 3_000, 3          # HVG union: top-3000 in >= 3 slides
MAX_EPOCHS = 100
HOTSPOT_CAP_PER_SLIDE = 33_000
HOTSPOT_N_NEIGHBORS = 30

TOP_KS = [500, 750, 1000]
MIN_GENES = [25, 50, 100]
PATHWAYS = ['TGFb', 'NFkB', 'MAPK']

MODEL_DIR = f'{OUT}/model_sweep'
FIT_H5AD = f'{OUT}/fit_cells.h5ad'
LCZ_CSV = f'{OUT}/lcz_top{max(TOP_KS)}.csv.gz'
N_JOBS = int(os.environ.get('N_JOBS', 24))


def prep():
    """Heavy stage. Writes FIT_H5AD + MODEL_DIR."""
    from cellina import Cellina as CellinaModel
    from scvi.train._callbacks import EarlyStopping, SaveCheckpoint
    from utils import set_seed

    set_seed(0)
    t0 = time.time()
    adata = load_joint(SLIDES, n_top_genes=N_TOP_GENES, min_slides=MIN_SLIDES)
    print(f'{adata.n_obs:,} x {adata.n_vars:,} in {time.time() - t0:.0f}s', flush=True)

    t0 = time.time()
    build_spatial(adata)
    C = adata.obsp['spatial_connectivities']
    coo = C.tocoo()
    sid = adata.obs[BATCH_KEY].to_numpy()
    n_cross = int(((sid[coo.row] != sid[coo.col]) & (coo.data != 0)).sum())
    assert n_cross == 0, f'{n_cross} cross-slide edges'
    import resource
    print(f'graph in {time.time() - t0:.0f}s, {C.nnz / adata.n_obs:.1f} edges/cell, '
          f'0 cross-slide, spatial_x {adata.obsm["spatial_x"].nnz / adata.n_obs:.0f} nnz/cell, '
          f'peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} GB',
          flush=True)

    CellinaModel.setup_anndata(adata, batch_key=BATCH_KEY, labels_key=LABELS_KEY,
                               domains_key=DOMAINS_KEY, spatial_obsm_key='spatial_x',
                               layer='counts')
    if os.path.isdir(MODEL_DIR):
        model = CellinaModel.load(MODEL_DIR, adata=adata)
        print('reused', MODEL_DIR, flush=True)
    else:
        t0 = time.time()
        model = CellinaModel(adata=adata, n_latent=64, n_layers=3, use_observed_lib_size=True,
                             condition_on_intrinsic=False, gene_likelihood='nb',
                             classifier_lambda=1., discriminator_lambda=1.)
        # no test split: every cell is embedded and scored downstream, nothing is held out
        model.train(max_epochs=MAX_EPOCHS, batch_size=4096, check_val_every_n_epoch=1,
                    early_stopping=True, devices=[0], train_size=0.9,
                    enable_checkpointing=True,
                    callbacks=[SaveCheckpoint(monitor='vae_loss_validation',
                                              dirpath=f'{OUT}/checkpoints',
                                              load_best_on_end=True),
                               EarlyStopping(monitor='vae_loss_validation', patience=5,
                                             mode='min')],
                    plan_kwargs={'lr': 1e-3, 'normalize_losses': True})
        print(f'trained in {(time.time() - t0) / 60:.1f} min', flush=True)
        model.save(MODEL_DIR, overwrite=True)

    # give_mean=True: the default (False) SAMPLES the posterior, so Hotspot's KNN graph --
    # and therefore the modules -- differ between runs. See 01_data_prep.ipynb.
    adata.obsm['cellina_spatial'] = model.get_latent_representation(
        adata=adata, latent_key='s', give_mean=True, batch_size=4096)

    crc_pos = np.flatnonzero(adata.obs[DOMAINS_KEY].astype(str).str.contains('CRC').to_numpy())
    fit_pos = crc_pos[cap_per_group(adata.obs.iloc[crc_pos], ['sid'],
                                    HOTSPOT_CAP_PER_SLIDE, seed=0)]
    adata_fit = slim(adata, fit_pos,
                     obs_cols=['sid', 'pid', 'typ', DOMAINS_KEY, LABELS_KEY, 'nCount_RNA'],
                     obsm_keys=['cellina_spatial'])
    adata_fit.write_h5ad(FIT_H5AD)
    print(f'wrote {FIT_H5AD}: {adata_fit.n_obs:,} x {adata_fit.n_vars:,}', flush=True)
    print(adata_fit.obs['sid'].value_counts().to_string(), flush=True)


def fit_hotspot(adata_fit):
    import hotspot
    from utils import set_seed
    set_seed(0)
    hs = hotspot.Hotspot(adata_fit, layer_key='counts', model='danb',
                         latent_obsm_key='cellina_spatial',
                         umi_counts_obs_key='nCount_RNA')
    hs.create_knn_graph(weighted_graph=False, n_neighbors=HOTSPOT_N_NEIGHBORS)
    t0 = time.time()
    res = hs.compute_autocorrelations(jobs=N_JOBS)
    print(f'autocorrelations in {(time.time() - t0) / 60:.1f} min', flush=True)
    return hs, res


def check_lcz_sliceable(hs, genes):
    """The one check behind the slicing trick: a sub-block of a larger lcz equals the lcz
    computed on that subset alone. Cheap on ~60 genes, same code path as the real call."""
    big = hs.compute_local_correlations(list(genes[:60]), jobs=N_JOBS)
    small = hs.compute_local_correlations(list(genes[:30]), jobs=N_JOBS)
    sub = big.loc[small.index, small.columns]
    assert np.allclose(sub.values, small.values, atol=1e-8), \
        f'lcz not sliceable, max diff {np.abs(sub.values - small.values).max():.2e}'
    print('lcz slice check ok', flush=True)


def sweep():
    import decoupler as dc

    adata_fit = sc.read_h5ad(FIT_H5AD)
    print(f'fit cells {adata_fit.n_obs:,} x {adata_fit.n_vars:,}', flush=True)
    hs, hs_res = fit_hotspot(adata_fit)
    genes = hs_res.loc[hs_res.FDR < 0.05].index.to_numpy()
    print(f'{len(genes)} genes at FDR<0.05; sweeping top_k {TOP_KS}', flush=True)
    assert len(genes) >= max(TOP_KS), f'only {len(genes)} significant genes'
    hs_res.to_csv(f'{OUT}/hotspot_results.csv')

    if os.path.exists(LCZ_CSV):
        lcz_full = pd.read_csv(LCZ_CSV, index_col=0)
        print('reused', LCZ_CSV, flush=True)
    else:
        check_lcz_sliceable(hs, genes)
        t0 = time.time()
        lcz_full = hs.compute_local_correlations(list(genes[:max(TOP_KS)]), jobs=N_JOBS)
        print(f'local correlations (k={max(TOP_KS)}) in {(time.time() - t0) / 60:.1f} min',
              flush=True)
        lcz_full.to_csv(LCZ_CSV)

    pw_progeny = dc.op.progeny(organism='human')
    pw_hallmark = dc.op.hallmark(organism='human')

    rows = []
    for top_k in TOP_KS:
        g = list(genes[:top_k])
        hs.local_correlation_z = lcz_full.loc[g, g]
        for min_genes in MIN_GENES:
            tag = f'k{top_k}_m{min_genes}'
            mods = hs.create_modules(min_gene_threshold=min_genes, core_only=True,
                                     fdr_threshold=0.05)
            sizes = mods[mods != -1].value_counts().sort_index()
            n_mod = len(sizes)
            if n_mod == 0:
                rows.append({'top_k': top_k, 'min_genes': min_genes, 'n_modules': 0})
                print(f'{tag}: 0 modules', flush=True)
                continue

            df = hs_res[['C']].join(mods.rename('Module'), how='inner')
            df = df[~df['Module'].isna() & (df['Module'] != -1)]
            mgm = df.pivot_table(index='Module', columns=df.index, values='C', fill_value=0)
            mgm.index = [f'CRC{int(m)}' for m in mgm.index]

            acts_p, padj_p = dc.mt.ulm(data=mgm, net=pw_progeny)
            acts_h, padj_h = dc.mt.ulm(data=mgm, net=pw_hallmark)

            # per-slide composition on the balanced fit subsample (no projection needed):
            # a module dominated by one slide is a patient effect
            hs.calculate_module_scores()
            top_mod = hs.module_scores.idxmax(axis=1).map(lambda m: f'CRC{int(m)}')
            comp = pd.crosstab(top_mod, adata_fit.obs['sid'].astype(str))
            comp.to_csv(f'{OUT}/composition_{tag}.csv')
            max_share = (comp.div(comp.sum(axis=1), axis=0)).max(axis=1)

            assign = module_pathway_assignment(acts_p, padj_p, pathways=tuple(PATHWAYS))
            row = {'top_k': top_k, 'min_genes': min_genes, 'n_modules': n_mod,
                   'n_assigned_genes': int((mods != -1).sum()),
                   'n_unassigned_genes': int((mods == -1).sum()),
                   'min_module_size': int(sizes.min()), 'max_module_size': int(sizes.max()),
                   'max_single_slide_share': round(float(max_share.max()), 3)}
            for pw in PATHWAYS:
                best = acts_p[pw].idxmax()
                row[f'{pw}_best_module'] = best
                row[f'{pw}_activity'] = round(float(acts_p.loc[best, pw]), 2)
                row[f'{pw}_padj'] = float(padj_p.loc[best, pw])
                row[f'{pw}_assigned'] = assign[pw]
            rows.append(row)

            long = pd.concat([
                acts_p.T.stack().rename('activity').to_frame().join(
                    padj_p.T.stack().rename('padj')).assign(net='progeny'),
                acts_h.T.stack().rename('activity').to_frame().join(
                    padj_h.T.stack().rename('padj')).assign(net='hallmark'),
            ])
            long.index.names = ['pathway', 'module']
            long.reset_index().to_csv(f'{OUT}/pathways_{tag}.csv', index=False)
            mods.rename('Module').to_csv(f'{OUT}/modules_{tag}.csv')

            print(f'{tag}: {n_mod} modules {list(sizes.values)}, '
                  f'{int((mods == -1).sum())} unassigned, '
                  f'assign={assign}, max slide share {max_share.max():.2f}', flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(f'{OUT}/sweep_summary.csv', index=False)
    print('\n' + summary.to_string(index=False), flush=True)
    plot(summary)


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    piv = summary.pivot(index='min_genes', columns='top_k', values='n_modules')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(piv, annot=True, fmt='d', cmap='viridis', ax=axes[0], cbar=False)
    axes[0].set_title('number of modules')
    sig = summary.pivot(index='min_genes', columns='top_k', values='TGFb_padj')
    sns.heatmap(-np.log10(sig.clip(lower=1e-300)), annot=True, fmt='.1f', cmap='rocket_r',
                ax=axes[1], cbar_kws={'label': r'$-\log_{10}$ padj'})
    axes[1].set_title('best-module TGFb significance')
    plt.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    for ext in ('svg', 'png'):
        fig.savefig(f'{FIG}/sweep_gene_selection.{ext}', bbox_inches='tight', dpi=300)
    print(f'wrote {FIG}/sweep_gene_selection.svg', flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('prep', 'all') and not os.path.exists(FIT_H5AD):
        prep()
    if what in ('sweep', 'all'):
        sweep()
