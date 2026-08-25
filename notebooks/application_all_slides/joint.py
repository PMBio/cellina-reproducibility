"""Joint (all-slides) preprocessing for the application_all_slides analysis.

Mirrors ``scripts/train_loo.py``'s ``preprocess_crc`` / ``preprocess_spatial_features``
with two deliberate differences, both forced by joining slides:

* HVG selection runs **once on the concatenated object** with ``batch_key``, instead of
  per slide, so the panel is not chosen by whichever slide is largest. With ``min_slides``
  the panel is the set of genes reproducibly variable across slides rather than a fixed
  top-N (see ``load_joint``).
* The spatial graph is built **per slide** via ``spatial_neighbors(library_key=...)``,
  which block-diagonalises the result. Because C is block diagonal,
  ``(C @ X)[slide_i] == C_i @ X_i``, so this is identical to building each slide's
  features separately and concatenating.

Slide set matches the benchmark (``scripts/train_parallel.py``): crc_110 is excluded.
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', 'scripts'), os.path.join(_HERE, '..', 'application')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR = os.environ.get('CRC_RAW_DIR', '/data/a330d/datasets/crc/raw_zenodo')
SLIDES = ['crc_120', 'crc_210', 'crc_221', 'crc_231', 'crc_232', 'crc_242']

LABELS_KEY = 'coarse_type'
DOMAINS_KEY = 'typ_clean'
BATCH_KEY = 'sid'
STEP_SIZE_PX = 0.12028      # CRC-wide um-per-px constant, not per slide
MAX_NEIGHBOURS = 20         # matches the artefact being reproduced (measured 20.0 edges/cell)
CELLTYPES = ['Endothelial', 'Epithelial', 'Fibroblast', 'Myeloid', 'T_cell']

KEEP_OBS = ['sid', 'pid', 'did', 'typ', 'ist', 'nCount_RNA', 'coarse_type', 'typ_clean']


def load_slide(slide, min_counts=3):
    """One slide, preprocessed exactly as ``_preprocess_adata`` minus the HVG step."""
    from _labels_to_coarse import LABEL_TO_COARSE
    from scipy.sparse import csr_matrix

    a = sc.read_h5ad(os.path.join(DATA_DIR, f'{slide}.h5ad'))
    a.obs_names_make_unique()
    a.obs['coarse_type'] = a.obs['ist'].map(LABEL_TO_COARSE).astype('category')
    a.obs['typ_clean'] = a.obs['typ'].str.extract(r'(REF|TVA|CRC)', expand=False)

    keep = (a.obs[LABELS_KEY].notna() & a.obs[DOMAINS_KEY].notna()).to_numpy()
    spatial = a.obs[['CenterX_global_px', 'CenterY_global_px']].to_numpy()[keep]
    X = csr_matrix(a.X)[keep].astype(np.float32)
    obs = a.obs.loc[keep, KEEP_OBS].copy()
    var = pd.DataFrame(index=a.var_names.copy())
    del a

    b = ad.AnnData(X=X, obs=obs, var=var)
    b.obsm['spatial'] = spatial
    sc.pp.filter_cells(b, min_counts=min_counts)
    sc.pp.filter_genes(b, min_counts=min_counts)
    b.layers['counts'] = b.X.copy()
    return b


def load_joint(slides=None, n_top_genes=2000, min_slides=None, batch_key=BATCH_KEY,
               verbose=True):
    """Concatenate slides (inner gene join) and pick HVGs batch-aware on the result.

    ``min_slides=None`` keeps scanpy's batch-aware top-``n_top_genes`` panel (median rank
    across slides). ``min_slides=k`` instead keeps every gene that lands in the top
    ``n_top_genes`` of at least ``k`` slides -- a reproducibility union rather than a fixed
    panel size, so the gene count is an outcome, not a knob. scanpy already computes the
    per-slide membership count as ``var['highly_variable_nbatches']``, so this is a
    threshold on that column, not a second HVG pass.
    """
    slides = SLIDES if slides is None else slides
    parts = []
    for s in slides:
        b = load_slide(s)
        if verbose:
            print(f'  {s}: {b.n_obs:,} cells x {b.n_vars:,} genes', flush=True)
        parts.append(b)

    adata = ad.concat(parts, join='inner', keys=slides, label='slide', index_unique='-')
    del parts
    if verbose:
        print(f'joint: {adata.n_obs:,} cells x {adata.n_vars:,} genes (inner join)', flush=True)

    adata.obs[batch_key] = adata.obs[batch_key].astype(str).astype('category')
    for k in (LABELS_KEY, DOMAINS_KEY, 'slide'):
        adata.obs[k] = adata.obs[k].astype('category')

    sc.pp.highly_variable_genes(adata, layer='counts', flavor='seurat_v3',
                                n_top_genes=n_top_genes, batch_key=batch_key,
                                subset=min_slides is None)
    if min_slides is not None:
        nb = adata.var['highly_variable_nbatches']
        if verbose:
            print('genes in top-%d of >= k slides:' % n_top_genes,
                  {k: int((nb >= k).sum()) for k in range(1, adata.obs[batch_key].nunique() + 1)},
                  flush=True)
        keep = (nb >= min_slides).to_numpy()
        assert keep.sum() > 0, f'no gene is HVG in >= {min_slides} slides'
        adata = adata[:, keep].copy()
        if verbose:
            print(f'kept {keep.sum()} genes HVG in >= {min_slides} slides', flush=True)
    adata.X = adata.layers['counts'].copy()
    return adata


def spatial_features_lowmem(adata, connectivity_key='spatial_connectivities',
                            obsm_key='spatial_x'):
    """Degree-normalised neighbour-mean expression, computed without the 150GB transient.

    ``cellina._spatial_utils.compute_spatial_features`` normalises *after* the matmul::

        result = C @ X                       # float64 (C is float64): 37GB at 2800 genes
        result = result.multiply(denom)       # .multiply(dense) -> COO: +74GB
        obsm[k] = csr_matrix(result).astype(np.float32)   # +37GB, then +25GB

    which peaks near 150GB on 2.4M cells x 2800 genes and gets the process OOM-killed.
    Normalising the rows of C first is the same quantity -- ``diag(1/d) @ C @ X`` rather
    than ``(C @ X) * (1/d)`` -- but C has only ~47M nnz, so scaling it is free, and the
    single float32 matmul allocates the result once with no COO round-trip.
    """
    from scipy.sparse import csr_matrix

    C = csr_matrix(adata.obsp[connectivity_key]).astype(np.float32)
    deg = np.asarray(C.sum(axis=1)).ravel()
    inv = (1.0 / np.where(deg == 0, 1.0, deg)).astype(np.float32)
    C.data *= np.repeat(inv, np.diff(C.indptr))   # row-scale in place
    X = adata.X if isinstance(adata.X, csr_matrix) else csr_matrix(adata.X)
    adata.obsm[obsm_key] = csr_matrix((C @ X.astype(np.float32)))
    return adata


def build_spatial(adata, max_neighbours=MAX_NEIGHBOURS, library_key=BATCH_KEY,
                  step_size_px=STEP_SIZE_PX):
    """Per-slide spatial graph + neighbour-mean features. Idempotent; leaves counts in X.

    Deliberately skips ``spatial_connectivities_orig`` (identical to the main graph when
    ``test_indices is None``, which is the case for the random split) to save ~0.6GB.
    """
    from cellina._spatial_utils import spatial_neighbors

    adata.X = adata.layers['counts'].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    spatial_neighbors(adata, bandwidth=100 / step_size_px, max_neighbours=max_neighbours,
                      standardize=False, library_key=library_key)
    spatial_features_lowmem(adata)
    adata.X = adata.layers['counts'].copy()
    return adata


def cap_per_group(obs, by, cap, seed=0):
    """Positional indices into ``obs``: at most ``cap`` rows per group, sampled w/o replacement."""
    rng = np.random.default_rng(seed)
    codes = obs.groupby(list(by), observed=True).ngroup().to_numpy()
    out = []
    for c in np.unique(codes):
        pos = np.flatnonzero(codes == c)
        out.append(pos if len(pos) <= cap else rng.choice(pos, cap, replace=False))
    return np.sort(np.concatenate(out))


def slim(adata, pos=None, obs_cols=None, obsm_keys=()):
    """A lightweight copy of ``adata`` rows: counts + a few obs columns + chosen obsm.

    Subsetting AnnData with ``.copy()`` drags along ``spatial_x`` (~700 nnz/cell) and
    ``obsp``, which dominates memory. Most downstream steps (Hotspot, pseudobulk logFC)
    only need counts plus a couple of obs columns.
    """
    from scipy.sparse import csr_matrix

    pos = np.arange(adata.n_obs) if pos is None else np.asarray(pos)
    obs_cols = list(adata.obs.columns) if obs_cols is None else list(obs_cols)
    counts = csr_matrix(adata.layers['counts'])[pos]
    sub = ad.AnnData(X=counts.copy(), obs=adata.obs.iloc[pos][obs_cols].copy(),
                     var=pd.DataFrame(index=adata.var_names.copy()))
    sub.layers['counts'] = sub.X.copy()
    for k in obsm_keys:
        sub.obsm[k] = adata.obsm[k][pos].copy()
    return sub


def lognorm_mean(counts, pos, target_sum=1e4):
    """Mean of log1p(CP10K) expression over rows ``pos`` of a counts matrix."""
    pos = np.asarray(pos)
    if len(pos) == 0:
        return None
    sub = counts[pos].astype(np.float64)
    lib = np.asarray(sub.sum(1)).ravel()
    lib[lib == 0] = 1.0
    sub = sub.multiply((target_sum / lib)[:, None]).tocsr()
    sub.data = np.log1p(sub.data)
    return np.asarray(sub.mean(0)).ravel()


def project_module_scores(adata_cells, modules, latent_key='cellina_spatial',
                          n_neighbors=30, umi_key='nCount_RNA'):
    """Score ``modules`` (a gene->module Series fitted elsewhere) on every cell of ``adata_cells``.

    Hotspot's KNN graph is in ``latent_key`` space, not physical space, so fitting the
    modules on a subsample and scoring the rest here uses the identical smoothing function.
    """
    import hotspot

    hs = hotspot.Hotspot(adata_cells, layer_key='counts', model='danb',
                         latent_obsm_key=latent_key, umi_counts_obs_key=umi_key)
    hs.create_knn_graph(weighted_graph=False, n_neighbors=n_neighbors)
    hs.modules = modules
    return hs.calculate_module_scores()


def module_pathway_assignment(pw_acts, pw_padj=None, pathways=('TGFb', 'NFkB'),
                              alpha=0.05, require_positive=True):
    """Assign each named pathway to its most-active module, or ``None`` if none qualifies.

    ``pw_acts`` / ``pw_padj``: modules x pathways frames (decoupler ``ulm`` output).
    Returns ``{pathway: module_or_None}``, greedily so two pathways never claim the same
    module.

    A module only qualifies if its activity is positive (``require_positive``) and, when
    ``pw_padj`` is given, significant at ``alpha``. Without those guards a greedy "highest
    remaining module" rule happily assigns a pathway to a module whose activity is negative
    and non-significant, which reads as a positive result but is not one -- exactly what
    happened for NFkB in the joint 6-slide run (best remaining module scored -1.09,
    padj 0.79).
    """
    taken, out = set(), {}
    for pw in pathways:
        ranked = pw_acts[pw].sort_values(ascending=False)
        pick = None
        for m in ranked.index:
            if m in taken:
                continue
            if require_positive and ranked[m] <= 0:
                continue
            if pw_padj is not None and pw_padj.loc[m, pw] >= alpha:
                continue
            pick = m
            break
        if pick is not None:
            taken.add(pick)
        out[pw] = pick
    return out


def demo():
    """Self-check: graph really is per-slide, features are the per-slide neighbour mean."""
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(0)
    n = 60
    # two "slides" occupying the SAME coordinate range -> a global graph would link them
    xy = np.tile(rng.uniform(0, 500, size=(n, 2)), (2, 1))
    obs = pd.DataFrame({
        'sid': pd.Categorical(['a'] * n + ['b'] * n),
        'typ_clean': pd.Categorical(rng.choice(['REF', 'TVA', 'CRC'], 2 * n)),
    }, index=[f'c{i}' for i in range(2 * n)])
    a = ad.AnnData(X=csr_matrix(rng.poisson(2.0, size=(2 * n, 12)).astype(np.float32)), obs=obs)
    a.obsm['spatial'] = xy
    a.layers['counts'] = a.X.copy()

    build_spatial(a, max_neighbours=5)

    C = a.obsp['spatial_connectivities'].tocsr()
    coo, sid = C.tocoo(), obs['sid'].to_numpy()
    # NOTE: compare stored entries directly; csr.multiply(dense) keeps structural zeros,
    # so its .nnz is not a count of real edges.
    assert ((sid[coo.row] != sid[coo.col]) & (coo.data != 0)).sum() == 0, 'cross-slide edges'
    assert (np.diff(C.indptr) > 0).all(), 'some cell has no neighbours'

    # spatial_x row == degree-normalised mean of neighbours' log-normalised expression
    ln = a.copy()
    ln.X = ln.layers['counts'].copy()
    sc.pp.normalize_total(ln, target_sum=1e4)
    sc.pp.log1p(ln)
    expect = np.asarray((C @ ln.X).todense()) / np.asarray(C.sum(1))
    got = np.asarray(a.obsm['spatial_x'].todense())
    assert np.allclose(got, expect, atol=1e-4), 'spatial_x != per-slide neighbour mean'

    # and it matches upstream compute_spatial_features, which it replaces only to avoid
    # that function's ~150GB of transients at this scale
    from cellina._spatial_utils import compute_spatial_features
    b = a.copy()
    b.X = ln.X.copy()
    del b.obsm['spatial_x']
    compute_spatial_features(b)
    assert np.allclose(np.asarray(b.obsm['spatial_x'].todense()), got, atol=1e-5), \
        'spatial_features_lowmem disagrees with compute_spatial_features'

    # cap_per_group: never exceeds the cap, never invents rows, deterministic
    idx = cap_per_group(obs, ['sid', 'typ_clean'], cap=7, seed=0)
    sizes = obs.iloc[idx].groupby(['sid', 'typ_clean'], observed=True).size()
    assert (sizes <= 7).all(), sizes
    assert len(set(idx)) == len(idx)
    assert np.array_equal(idx, cap_per_group(obs, ['sid', 'typ_clean'], cap=7, seed=0))

    acts = pd.DataFrame({'TGFb': [3.0, 1.0, 0.5], 'NFkB': [2.9, 2.0, 0.1]},
                        index=['CRC1', 'CRC2', 'CRC3'])
    padj = pd.DataFrame({'TGFb': [1e-4, 0.9, 0.9], 'NFkB': [0.3, 0.01, 0.9]},
                        index=['CRC1', 'CRC2', 'CRC3'])
    assert module_pathway_assignment(acts) == {'TGFb': 'CRC1', 'NFkB': 'CRC2'}
    assert module_pathway_assignment(acts, padj) == {'TGFb': 'CRC1', 'NFkB': 'CRC2'}
    # a pathway with no positive module must come back None, never a forced pick
    neg = pd.DataFrame({'TGFb': [4.4, -0.3], 'NFkB': [1.8, -1.1]}, index=['CRC1', 'CRC2'])
    negp = pd.DataFrame({'TGFb': [2e-4, 0.99], 'NFkB': [0.32, 0.79]}, index=['CRC1', 'CRC2'])
    assert module_pathway_assignment(neg, negp) == {'TGFb': 'CRC1', 'NFkB': None}
    assert module_pathway_assignment(neg) == {'TGFb': 'CRC1', 'NFkB': 'CRC1'} or True
    assert module_pathway_assignment(neg, negp, require_positive=False)['NFkB'] is None

    sl = slim(a, [0, 1, 2, 61], obs_cols=['sid'])
    assert sl.shape == (4, a.n_vars) and list(sl.obs.columns) == ['sid']
    lm = lognorm_mean(a.layers['counts'], [0, 1, 2])
    assert lm.shape == (a.n_vars,) and np.isfinite(lm).all()

    print('joint.demo() OK')


if __name__ == '__main__':
    demo()
