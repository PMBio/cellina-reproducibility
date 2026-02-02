import numpy as np
import torch
import anndata as ad
import pandas as pd

from tqdm import tqdm
from captum.attr import IntegratedGradients, DeepLift, DeepLiftShap


class IGModelWrapper(torch.nn.Module):
    """Wrap model so forward returns selected latent dims of w."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, neighbor_emb, x, y, d, b):
        result = self.model.vae.forward(x=x, y=y, d=d, b=b, neighbor_emb=neighbor_emb)
        return result[0][0]

def run_attribution(
    model,
    dl,
    target_dims,
    n_cell_types,
    method="IG",  # "IG", "DeepLift", or "DeepLiftShap"
    baseline_mode="zeros",
    n_steps=50,  # used only for IG
    n_baselines=10,  # used only for DeepLiftShap
    device="cpu",
    mask_neighbor_types=None,
    n_spatial_features_per_ct=None,
    side_information_key="neighborhood_pseudobulks",
    label_col="label",
    domain_col="domain",
    batch_col="batch",
    reduce=True,
):
    """
    Unified attribution function for IG, DeepLift, DeepLiftShap.
    """
    wrapper = IGModelWrapper(model).to(device)
    # Choose attribution method
    if method == "IG":
        explainer = IntegratedGradients(wrapper)
    elif method == "DeepLift":
        explainer = DeepLift(wrapper)
    elif method == "DeepLiftShap":
        explainer = DeepLiftShap(wrapper)
    else:
        raise ValueError(f"Unsupported method: {method}")

    side_key = side_information_key
    #n_spatial_features_per_ct = dataset.adata.obsm[side_key].shape[1]//n_cell_types
    if reduce:
        att_sums = {d: np.zeros((n_cell_types, n_spatial_features_per_ct)) for d in target_dims}
    else:
        att_sums = {d: [] for d in target_dims}


    for batch in tqdm(dl):
        x, y, d, b = (
            batch.X.float(),
            batch.obs[label_col],
            batch.obs[domain_col],
            batch.obs[batch_col],
        )
        neighbor_emb = batch.obsm[side_key]
        current_batch_size = neighbor_emb.shape[0]


        # Mask non-target neighbor types
        if mask_neighbor_types is not None:
            mask = np.ones((n_cell_types, n_spatial_features_per_ct))
            for ct in range(n_cell_types):
                if ct not in mask_neighbor_types:
                    mask[ct, :] = 0
            mask = torch.tensor(mask.flatten(), dtype=torch.float32, device=device)
            neighbor_emb = neighbor_emb * mask
        
        # Prepare baseline(s)
        if method in ["IG", "DeepLift"]:
            baseline = torch.zeros_like(neighbor_emb, device=device)
        elif method == "DeepLiftShap":
            if baseline_mode == "zeros":
                baseline = torch.zeros((n_baselines, neighbor_emb.shape[1]), device=device)
            elif baseline_mode == "random":
                idxs = np.random.choice(neighbor_emb.shape[0], n_baselines, replace=True)
                baseline = neighbor_emb[idxs]
            else:
                raise NotImplementedError

        # Compute attributions per target dim
        for gene_idx in target_dims:
            kwargs = {
                "inputs": neighbor_emb,
                "additional_forward_args": (x, y, d, b),
                "target": gene_idx,
            }
            if method == "IG":
                kwargs["baselines"] = baseline
                kwargs["n_steps"] = n_steps
                kwargs["internal_batch_size"]: max(1024, current_batch_size)  # type: ignore
            else:
                kwargs["baselines"] = baseline

            atts = explainer.attribute(**kwargs)
            atts_np = atts.detach().cpu().numpy()
            atts_np = atts_np.reshape(
                current_batch_size, n_cell_types, n_spatial_features_per_ct
            )
            if reduce:
                att_sums[gene_idx] += atts_np.sum(axis=0)
            else:
                att_sums[gene_idx].append(atts_np)

    # Stack all collected attributions if not reduced
    if not reduce:
        for gene_idx in target_dims:
            att_sums[gene_idx] = np.concatenate(att_sums[gene_idx], axis=0)

    return att_sums


def attributions_to_adata(att_sums, adata, target_dims):
    """
    Convert per-cell attribution results from `run_attribution(reduce=False)` into an AnnData object.

    Parameters
    ----------
    att_sums : dict
        Dictionary {gene_idx: np.ndarray of shape (n_cells, n_cell_types, n_spatial_features_per_ct)}.
    adata : AnnData
        Original AnnData containing corresponding cells in obs and uns['spatial_var'].
    target_dims : list[int]
        List of gene indices that correspond to the target genes.

    Returns
    -------
    AnnData
        AnnData where:
        - X: (n_cells, len(target_dims) * len(adata.uns['spatial_var']))
        - obs: copied from input `adata.obs`
        - var_names: geneName_spatialVarName
    """
    # --- Basic checks ---
    if "spatial_var" not in adata.uns:
        raise ValueError("adata.uns['spatial_var'] must contain spatial feature names.")
    
    spatial_features = list(adata.uns["spatial_var"])
    gene_names = list(adata.var_names[target_dims])

    # --- Infer shapes from first entry ---
    first_key = target_dims[0]
    first_attr = att_sums[first_key]
    n_cells = first_attr.shape[0]
    n_features_per_gene = np.prod(first_attr.shape[1:])  # flatten (cell_types × spatial_features)

    # --- Allocate target matrix once ---
    X = np.empty((n_cells, len(target_dims) * n_features_per_gene), dtype=np.float32)

    # --- Fill in one gene at a time (in place) ---
    for i, gene_idx in enumerate(target_dims):
        start = i * n_features_per_gene
        end = start + n_features_per_gene
        X[:, start:end] = att_sums[gene_idx].reshape(n_cells, -1)

    # --- Build var names like "GeneA^feature1" ---
    # infer number of total spatial features per cell (n_cell_types * n_spatial_features_per_ct)
    n_spatial_features_total = n_features_per_gene
    if n_spatial_features_total != len(spatial_features):
        # fallback if spatial_features per cell type — flatten pattern
        spatial_features = [f"spatial_{i}" for i in range(n_spatial_features_total)]

    var_names = []
    for gene in gene_names:
        for sf in spatial_features:
            var_names.append(f"{gene}^{sf}")

    # --- Construct AnnData ---
    adata_att = ad.AnnData(
        X=X,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=var_names),
    )
    adata_att.uns["target_genes"] = gene_names
    adata_att.uns["spatial_var"] = spatial_features

    return adata_att
