"""The lean counterfactual must equal cellina's full-graph rewiring exactly.

run_sensitivity.py --lean-cf skips materialising the rewired graph and computes
spatial features for the basal cells alone. Legitimate only if it reproduces the
library result -- same donors from the same RNG stream, same row-normalised
aggregation. This imports the real library function and compares.

    python test_lean_cf.py
"""
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from cellina._spatial_utils import make_counterfactual_adata


def lean_features(adata, idx_basal, idx_cf, k, seed):
    """The construction used by lean_counterfactual_expression()."""
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([
        rng.choice(idx_cf, size=k, replace=False) for _ in idx_basal])
    C = sp.csr_matrix((np.ones(len(chosen), dtype=np.float32),
                       (np.repeat(np.arange(len(idx_basal)), k), chosen)),
                      shape=(len(idx_basal), adata.n_obs))
    return np.asarray(((C @ adata.X) / float(k)).todense())


def main():
    rng = np.random.default_rng(0)
    n_obs, n_var = 200, 12
    adata = AnnData(X=sp.csr_matrix(rng.poisson(2.0, (n_obs, n_var)).astype(np.float32)))
    adata.obsm["spatial_x"] = sp.csr_matrix((n_obs, n_var), dtype=np.float32)
    conn = sp.random(n_obs, n_obs, density=0.05, format="csr", random_state=1)
    conn.data[:] = 1.0
    adata.obsp["spatial_connectivities"] = conn

    idx_basal = np.array([3, 17, 42, 99])
    idx_cf = np.setdiff1d(np.arange(n_obs), idx_basal)[:120]

    for seed in (0, 1, 7):
        for k in (3, 7, 20):
            lib = make_counterfactual_adata(
                adata.copy(), idx_basal, idx_cf, spatial_column="spatial_x",
                precomputed=False, n_neighbours=k, random_state=seed,
                connectivity_key="spatial_connectivities")
            a = np.asarray(lib.obsm["spatial_x"].todense())
            b = lean_features(adata, idx_basal, idx_cf, k, seed)
            assert a.shape == b.shape, (a.shape, b.shape)
            d = np.abs(a - b).max()
            assert np.allclose(a, b, rtol=1e-6, atol=1e-6), f"seed={seed} k={k} diff={d}"
            print(f"seed={seed} k={k:2d}: identical (max|diff|={d:.2e})")
    print("OK: lean counterfactual == library counterfactual")


if __name__ == "__main__":
    main()
