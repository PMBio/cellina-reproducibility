"""SpatialProp LOO edge-perturbation eval script.

Unlike spatialprop_eval_loo.py (which perturbs neighbor *expression values* while
keeping each cell's real spatial graph fixed -- "node perturbation"), this script
perturbs the graph's neighbor *composition*: for each control-domain cell of the
held-out cell type ("AC"), it steers its real k-hop neighborhood toward a real
target-domain cell's neighborhood ("AT") -- the mechanism behind the spatialprop
paper's "microenvironmental steering" experiment (spatial_gnn's own
`batch_steering_cell`, reimplemented here per-pair so many (control, donor) pairs
with different donors can be scored in batched GNN forward calls instead of one
call per pair -- see steer_and_predict_all). No retraining -- this reuses the
checkpoints produced by spatialprop_train_loo.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from torch_geometric.data import Batch

DATA_ROOT = '/data/a330d'  # os.environ.get("DATA_ROOT", ".")

sys.path.append('../../../scripts')
from train_loo import preprocess_crc, preprocess_merfish
from counterfactual_analysis import compute_rmse, compute_edistance, mixing_index, get_lfc, precision, direction_match, compute_mse_lfc, nb_deviance_pop_mean

from perturb_utils import total_normalize
from spatialprop_train_loo import clean_all_dirs
from spatialprop_eval_loo import to_raw_counts, predicted_to_raw_counts

from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.utils.dataset_utils import (
    create_dataloader_from_dataset,
    load_model_from_path,
)
from spatial_gnn.utils.perturbation_utils import predict as steer_predict

from configs.adata_crc_config import ADATA_ARGS as ADATA_CRC_ARGS
from configs.adata_merfish_config import ADATA_ARGS as ADATA_MERFISH_ARGS

DATASET_NAME = "merfish"  # Options: ['crc', 'merfish']

CRC_BASE_PATH = os.path.join(DATA_ROOT, "datasets/crc/raw_zenodo")
CRC_SLIDES = ['crc_232', 'crc_242', 'crc_231', 'crc_210', 'crc_221', 'crc_120']
CRC_CELLTYPES = [
    "Endothelial",
    "Epithelial",
    "Fibroblast",
    "Myeloid",
    "T_cell",
]

MERFISH_BASE_PATH = os.path.join(DATA_ROOT, "datasets/MERFISH_mouse_brain")
MERFISH_SLIDES = ['C57BL6J-2.036', 'C57BL6J-2.039', 'C57BL6J-2.041']
MERFISH_CELLTYPES = [
    'glutamatergic neuron',
    'GABAergic neuron',
    'astrocyte',
    'oligodendrocyte',
    'endothelial cell',
]

ADATA_BASE_PATH = CRC_BASE_PATH if DATASET_NAME == "crc" else MERFISH_BASE_PATH
SLIDES = CRC_SLIDES if DATASET_NAME == "crc" else MERFISH_SLIDES
CELLTYPES = CRC_CELLTYPES if DATASET_NAME == "crc" else MERFISH_CELLTYPES
DATA_ARGS = ADATA_CRC_ARGS if DATASET_NAME == "crc" else ADATA_MERFISH_ARGS

top_n = 50
min_cells = 50
library_size = 1e4
labels_key = DATA_ARGS.get('labels_key')
domains_key = DATA_ARGS.get('domains_key')
n_top_genes = DATA_ARGS.get('n_top_genes')
device = "cuda:1" if torch.cuda.is_available() else "cpu"
control_domain = DATA_ARGS.get('control_domains')[0]  # Assuming only one control domain for simplicity
holdout_domains = DATA_ARGS.get('holdout_domains')
out_dir = os.path.join(DATA_ROOT, "tmp/")
model_base_path = '.'
results_csv_path = f'../../../results/loo_spatialprop_{DATASET_NAME}_DEG_{top_n}_edge_pert.csv'

PROP = 1.0        # fraction of a control cell's neighborhood replaced by its donor's (paper's steering theta)
SEED = 0          # AC<->AT donor pairing and steering draws
batch_size = 1024 # cells scored per GNN forward call


def build_cell_graph_index(dataset):
    """Map original_cell_id -> unbatched PyG Data object, for every graph the dataset produced.

    Reuses create_dataloader_from_dataset's own flat-list loading of processed_dir's
    .pt batches (the path every other script in this codebase relies on) rather than
    SpatialAgingCellDataset's manifest-based __getitem__, which nothing else here
    exercises.
    """
    all_graphs, _ = create_dataloader_from_dataset(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=False, persistent_workers=False,
    )
    return {g.original_cell_id: g for g in all_graphs}


def _steer_graph(control_graph, donor_graph, prop, generator):
    """Return a copy of control_graph steered toward donor_graph's neighbor pool.

    Reimplements spatial_gnn's own batch_steering_cell logic (random-with-replacement
    draws from the donor's node pool, overwriting the first round(prop*n_cells) node
    rows) applied to a single pair, rather than calling batch_steering_cell itself --
    that function only accepts one `target` shared across an entire batch, so it can't
    batch many pairs that each have a *different* donor. This is functionally
    equivalent per pair, just vectorizable across pairs (see steer_and_predict_all).

    Also re-zeroes the center node's own feature row, which batch_steering_cell does
    not guard either: it overwrites node rows in whatever arbitrary order
    k_hop_subgraph originally assigned them, which can include the center's row. The
    GNN was trained with that row always exactly zero (spatial_dataset.py zeroes it at
    dataset-build time); nothing in the model's forward() re-enforces this.
    """
    steered = control_graph.clone()
    n_cells = steered.x.shape[0]
    n_replace = round(prop * n_cells)
    if n_replace > 0:
        rand_idxs = torch.randint(0, donor_graph.x.shape[0], (n_replace,), generator=generator)
        steered.x[:n_replace] = donor_graph.x[rand_idxs]
    steered.x[int(steered.center_node)] = 0
    return steered


def steer_and_predict_all(control_graphs, donor_graphs, model, prop, device, batch_size, generator):
    """Steer many (control, donor) pairs and score them via batched GNN forward calls.

    Each pair's computation is independent, so instead of one forward call per pair
    (thousands of tiny GPU calls for a large cell type), steer batch_size pairs at a
    time and run one forward pass per chunk -- the same batch_size=512/1024 pattern
    the rest of this codebase already uses for inference. Chunking (rather than
    steering all pairs up front) keeps peak memory bounded regardless of how many
    pairs there are.
    """
    preds = []
    with torch.no_grad():
        for i in range(0, len(control_graphs), batch_size):
            chunk = [
                _steer_graph(c, d, prop, generator)
                for c, d in zip(control_graphs[i:i + batch_size], donor_graphs[i:i + batch_size])
            ]
            batch = Batch.from_data_list(chunk).to(device)
            out = steer_predict(model, batch, inject=False)
            preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    results = []
    for slide_id in SLIDES:
        print(f"\n{'='*60}\nProcessing slide {slide_id}\n{'='*60}")
        adata = sc.read_h5ad(f"{ADATA_BASE_PATH}/{slide_id}.h5ad")
        if DATASET_NAME == 'crc':
            adata = preprocess_crc(adata, n_top_genes=n_top_genes, labels_key=labels_key, domains_key=domains_key)
        elif DATASET_NAME == 'merfish':
            adata = preprocess_merfish(adata, n_top_genes=n_top_genes, labels_key=labels_key, domains_key=domains_key)
        else:
            raise ValueError(f"Unknown dataset_name: {DATASET_NAME}. Supported: crc, merfish")
        sc.pp.normalize_total(adata, target_sum=library_size)
        sc.pp.log1p(adata)

        for holdout_ct in CELLTYPES:
            mask_holdout = (adata.obs[labels_key] == holdout_ct) & (adata.obs[domains_key].isin(holdout_domains))
            adata.obs['is_holdout'] = mask_holdout
            print(f"\n{'='*60}")
            print(f"Holdout cell type: {holdout_ct}")
            print(f"{'='*60}")

            if holdout_ct not in adata.obs[labels_key].values:
                print(f"  WARNING: '{holdout_ct}' not in adata — skipping")

            # spatial_gnn expects 'celltype', 'region', and 'mouse_id' columns
            adata.obs["celltype"] = adata.obs[labels_key]
            adata.obs["mouse_id"] = str(slide_id)
            adata.obs["region"] = adata.obs[domains_key]

            exp_name = f"{slide_id}_loo_{holdout_ct}"
            out_dir_ct = os.path.join(out_dir, exp_name)
            test_path = os.path.join(out_dir_ct, "adata_test.h5ad")
            trained_model_path = f'{model_base_path}/output/{exp_name}/{slide_id}_{holdout_ct}_loo_expression_2hop_2augment_expression_none/weightedl1_1en03/model.pth'

            adata_test = sc.read_h5ad(test_path)
            model, model_config = load_model_from_path(trained_model_path, device)
            celltypes_to_index = model_config["celltypes_to_index"]

            for hd in holdout_domains:
                # dataset_prefix must vary with hd: SpatialAgingCellDataset.process()
                # no-ops once its processed_dir already exists, so reusing exp_name
                # across holdout_domains here would silently reuse the first hd's
                # cached graphs for every subsequent one.
                dataset = SpatialAgingCellDataset(
                    subfolder_name="predict_edge_pert",
                    dataset_prefix=f"{exp_name}_{hd}",
                    target="expression",
                    k_hop=2,
                    augment_hop=0,
                    node_feature="expression",
                    inject_feature=None,
                    num_cells_per_ct_id=100_000,
                    center_celltypes=[holdout_ct],
                    whole_tissue=False,
                    use_ids=[str(slide_id)],
                    raw_filepaths=[test_path],
                    celltypes_to_index=celltypes_to_index,
                    normalize_total=True,
                )
                dataset.process()
                cell_graphs = build_cell_graph_index(dataset)

                mask_ac = (adata_test.obs["celltype"] == holdout_ct) & (adata_test.obs["region"] == control_domain)
                mask_at = (adata_test.obs["celltype"] == holdout_ct) & (adata_test.obs["region"] == hd)

                # Only cells whose k-hop subgraph survived extract_khop_subgraphs's
                # min-size filter are available to pair/steer.
                ac_ids = [c for c in adata_test.obs_names[mask_ac.values] if c in cell_graphs]
                at_ids = [c for c in adata_test.obs_names[mask_at.values] if c in cell_graphs]

                n_ac, n_at = len(ac_ids), len(at_ids)
                print(f"  [spatialprop_edge_pert] {holdout_ct} -> {hd}: AC={n_ac}, AT={n_at}")

                if n_ac < min_cells or n_at < min_cells:
                    print(f"  skip {holdout_ct} -> {hd}: too few cells (need >= {min_cells})")
                    continue

                rng = np.random.default_rng(SEED)
                n_pairs = min(n_ac, n_at)
                paired_ac = rng.choice(ac_ids, size=n_pairs, replace=False)
                paired_at = rng.choice(at_ids, size=n_pairs, replace=False)

                torch_gen = torch.Generator().manual_seed(SEED)
                control_graphs = [cell_graphs[ac_id] for ac_id in paired_ac]
                donor_graphs = [cell_graphs[at_id] for at_id in paired_at]
                steered = steer_and_predict_all(
                    control_graphs, donor_graphs, model, PROP, device, batch_size, torch_gen
                )

                # row_scale: the steered prediction is still nominally FOR the control
                # cell's own physical identity (only its neighborhood is hypothetical),
                # so its own real row-sum substitutes for the model's unrecoverable
                # renormalized output scale -- see predicted_to_raw_counts docstring.
                ac_log_norm = adata_test[paired_ac].X
                ac_log_norm = ac_log_norm.toarray() if issparse(ac_log_norm) else np.asarray(ac_log_norm)
                row_scale_ac = ac_log_norm.sum(axis=1)

                n_genes = adata_test.n_vars
                counterfactual = total_normalize(
                    predicted_to_raw_counts(steered, row_scale_ac, n_genes),
                    target_sum=library_size,
                )
                control = total_normalize(to_raw_counts(adata_test[mask_ac].X), target_sum=library_size)
                target = total_normalize(to_raw_counts(adata_test[mask_at].X), target_sum=library_size)

                gt_lfc, cf_lfc, deg = get_lfc(control=control, target=target, counterfactual=counterfactual, n_deg=top_n)

                spear, _ = spearmanr(gt_lfc[deg], cf_lfc[deg])
                pear, _ = pearsonr(gt_lfc[deg], cf_lfc[deg])
                prec = precision(gt_lfc, cf_lfc, k=top_n, use_abs=True)
                dir_match = direction_match(gt_lfc, cf_lfc, k=top_n, normalize="intersection")
                dir_match_k = direction_match(gt_lfc, cf_lfc, k=top_n, normalize="k")
                dir_match_gt = direction_match(gt_lfc, cf_lfc, k=top_n, normalize="gt_topk")
                mix_idx = mixing_index(observed=target, predicted=counterfactual, library_size=library_size)
                edist_global = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=library_size)
                edist_local = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=library_size, local=True)
                edist_pca_log = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=library_size, local=True, use_pca=True)
                edist_pca = compute_edistance(adata, observed=target, predicted=counterfactual, deg=None, library_size=library_size, local=True, use_pca=True, log1p=False)
                rmse = compute_rmse(observed=target, predicted=counterfactual, deg=deg, library_size=library_size)
                mse_lfc = compute_mse_lfc(gt_vec=gt_lfc, cf_vec=cf_lfc, deg=deg)
                nb_deviance = nb_deviance_pop_mean(obs_X=target, pred_X=counterfactual)

                results.append(
                    dict(
                        dataset_name=DATASET_NAME,
                        sid=slide_id,
                        control_domain=control_domain,
                        target_domain=hd,
                        n_deg=top_n,
                        model_name="spatialprop_edge_pert",
                        holdout_celltype=holdout_ct,
                        spearman=spear,
                        pearson=pear,
                        precision=prec,
                        direction_match=dir_match,
                        direction_match_k=dir_match_k,
                        direction_match_gt=dir_match_gt,
                        mixing_index=mix_idx,
                        edistance_global=edist_global,
                        edistance_local=edist_local,
                        edistance_pca_log=edist_pca_log,
                        edistance_pca=edist_pca,
                        rmse=rmse,
                        mse_lfc=mse_lfc,
                        prop=PROP,
                        n_pairs=n_pairs,
                        nb_deviance=nb_deviance,
                    )
                )

            # Remove spatialprop-generated data files
            clean_all_dirs()

    df_results = pd.DataFrame(results)
    # Check if CSV already exists, if so, append to it ; otherwise, create a new CSV
    if os.path.exists(results_csv_path):
        df_results.to_csv(f"{results_csv_path}", mode='a', header=False, index=False)
    else:
        df_results.to_csv(f"{results_csv_path}", index=False)


if __name__ == "__main__":
    main()
