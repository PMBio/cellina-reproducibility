import torch
import numpy as np
from captum.attr import DeepLift
from scvi import REGISTRY_KEYS
from typing import Optional, List
from tqdm import tqdm

import torch.nn as nn
from cellina._constants import SPATIAL_X_KEY


class SpatialToGeneWrapper(nn.Module):
    """
    Wrapper to expose:
      spatial_x  --> reconstructed gene expression (px_rate)

    Everything else (x, batch_index) is treated as fixed context.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.module.eval()

    def forward(self, spatial_x, x, batch_index):
        """
        spatial_x: (B, n_spatial)
        x: counts
        batch_index: batch indices

        returns:
            px_rate: (B, n_genes)
        """

        # --- run inference with spatial_x overridden
        # We reproduce _get_inference_input but replace spatial_x
        inference_outputs = self.module.inference(
            x=x,
            spatial_x=spatial_x,
            batch_index=batch_index,
        )

        # --- generative step
        generative_outputs = self.module.generative(
            shifted=inference_outputs["shifted"],
            library=inference_outputs["library"],
            batch_index=batch_index,
        )

        return generative_outputs["px_rate"]


@torch.no_grad()
def run_deeplift(
    model,
    adata,
    indices: Optional[List[int]] = None,
    target_genes: Optional[List[int]] = None,
    batch_size: int = 128,
    baseline: str = "zeros",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute DeepLift attributions from spatial features to reconstructed genes.

    Parameters
    ----------
    model
        Trained scvi model (CellinaModule wrapped in scvi.model object)
    adata
        AnnData already subsetted
    target_genes
        List of gene indices to compute attributions for. If None, compute all genes
    batch_size
        Batch size
    baseline
        "zeros" or "mean" for spatial features
    device
        Torch device. Defaults to model device

    Returns
    -------
    attributions
        Tensor of shape (n_cells, n_spatial_features, n_genes)
    """

    module = model.module
    module.eval()

    if device is None:
        device = next(module.parameters()).device

    # --- scvi-style dataloader
    dataloader = model._make_data_loader(
        adata=adata,
        batch_size=batch_size,
        indices=indices,
        shuffle=False,
    )
    n_genes_total = adata.n_vars

    if target_genes is None:
        target_genes = list(range(n_genes_total))

    all_attrs = []

    # --- loop over genes to attribute individually
    for gene_idx in tqdm(target_genes):
        wrapper = SpatialToGeneWrapper(model.module)
        deeplift = DeepLift(wrapper)
        gene_attrs = []

        for tensors in dataloader:
            x = tensors[REGISTRY_KEYS.X_KEY].to(device)
            spatial_x = tensors[SPATIAL_X_KEY].to(device)
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY].to(device)

            # baseline
            if baseline == "zeros":
                spatial_base = torch.zeros_like(spatial_x)
            elif baseline == "mean":
                spatial_base = spatial_x.mean(dim=0, keepdim=True).expand_as(spatial_x)
            else:
                raise ValueError("baseline must be 'zeros' or 'mean'")

            spatial_x.requires_grad_(True)

            # --- attribute gene
            attrs = deeplift.attribute(
                inputs=spatial_x,  # (B, n_spatial)
                baselines=spatial_base,  # same shape
                additional_forward_args=(x, batch_index),
                target=gene_idx,
            )

            # pick only this gene
            attrs_gene = attrs  # already (B, n_spatial), forward returns all genes, Captum computes grads per input
            gene_attrs.append(attrs_gene.cpu())

        gene_attrs_tensor = torch.cat(gene_attrs, dim=0)  # (n_cells, n_spatial)
        all_attrs.append(gene_attrs_tensor.unsqueeze(-1))  # (n_cells, n_spatial, 1)

    attributions = torch.cat(all_attrs, dim=-1)  # (n_cells, n_spatial, n_genes)
    return attributions


def add_deeplift_to_obs(
    adata,
    gene,
    agg="l2",
    key="deeplift_spatial",
):
    # gene_idx = adata.var_names.get_loc(gene)
    gene_idx = adata.uns[key]["gene_names"].index(gene)  # check gene was computed
    X = adata.obsm[key][:, :, gene_idx]

    if agg == "sum_abs":
        vals = np.abs(X).sum(axis=1)
    elif agg == "mean":
        vals = X.mean(axis=1)
    elif agg == "l2":
        vals = np.linalg.norm(X, axis=1)
    else:
        raise ValueError(f"Unknown agg: {agg}")

    adata.obs[f"{key}_{gene}_{agg}"] = vals
