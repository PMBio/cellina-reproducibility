"""SpatialProp LOO train script.
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch

DATA_ROOT = '/data2/a330d' #os.environ.get("DATA_ROOT", ".")

sys.path.append('../../scripts')
from train_loo import preprocess_crc, preprocess_merfish, split_indices

from spatial_gnn.api.perturbation_api import (
    train_perturbation_model,
)
from configs.adata_crc_holdout_config import ADATA_ARGS as ADATA_CRC_ARGS
from configs.adata_merfish_holdout_config import ADATA_ARGS as ADATA_MERFISH_ARGS

DATASET_NAME = "crc"  # or "merfish"

CRC_BASE_PATH = os.path.join(DATA_ROOT, "datasets/crc/raw_zenodo")
CRC_SLIDES = ['crc_242', 'crc_231', 'crc_210', 'crc_221', 'crc_120']
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

max_epochs = 100
batch_size = 512
labels_key = DATA_ARGS.get('labels_key')
domains_key = DATA_ARGS.get('domains_key')
n_top_genes = DATA_ARGS.get('n_top_genes')
n_neighbors = DATA_ARGS.get('n_neighbors')
control_domains = DATA_ARGS.get('control_domains')
holdout_domains = DATA_ARGS.get('holdout_domains')
device = "cuda:1" if torch.cuda.is_available() else "cpu"
out_dir = os.path.join(DATA_ROOT, "tmp/")

def _clean_incomplete_gnn_dirs(base_dir: str = "data/gnn_datasets") -> None:
    """Remove GNN dataset subdirs that have no manifest.json (crashed mid-processing)."""
    if not os.path.exists(base_dir):
        return
    for top in os.scandir(base_dir):
        if not top.is_dir():
            continue
        for sub in os.scandir(top.path):
            if sub.is_dir() and not os.path.exists(os.path.join(sub.path, "manifest.json")):
                print(f"  Removing incomplete GNN dataset cache: {sub.path}")
                shutil.rmtree(sub.path)


def clean_all_dirs(base_dir="data/gnn_datasets"):
    for root, dirs, _ in os.walk(base_dir, topdown=False):
        print(f"Removing cache directories: {root}")
        for d in dirs:
            path = os.path.join(root, d)
            shutil.rmtree(path)


def main():
    for slide_id in SLIDES:
        print(f"\n{'='*60}\nProcessing slide {slide_id}\n{'='*60}")
        adata = sc.read_h5ad(f"{ADATA_BASE_PATH}/{slide_id}.h5ad")
        if DATASET_NAME == 'crc':
            adata = preprocess_crc(adata, n_top_genes=n_top_genes, labels_key=labels_key, domains_key=domains_key)
        elif DATASET_NAME == 'merfish':
            adata = preprocess_merfish(adata, n_top_genes=n_top_genes, labels_key=labels_key, domains_key=domains_key)
        else:
            raise ValueError(f"Unknown dataset_name: {DATASET_NAME}. Supported: crc, merfish")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # 1. Load data
        # spatial_gnn expects 'celltype', 'region', and 'mouse_id' columns
        adata.obs["celltype"] = adata.obs[labels_key]
        adata.obs["mouse_id"] = str(slide_id)
        adata.obs["region"] = adata.obs[domains_key]

        # 2. Holdout split (marks adata.obs['is_holdout'])
        holdout_ct = "Fibroblast" if DATASET_NAME == "crc" else "astrocyte" # Dummy value, is not used because holdout_full_domain=True
        train_idx, val_idx, test_idx = split_indices(adata,
                                                    holdout_celltype=holdout_ct,
                                                    labels_key=labels_key,
                                                    domains_key=domains_key,
                                                    holdout_domains=holdout_domains,
                                                    holdout_full_domain=True,
                                                    seed=0)
        print(
            f"  train={len(train_idx):,}  val={len(val_idx):,}  "
            f"test (holdout)={len(test_idx):,}"
        )

        # 3. Save train/test h5ad files
        exp_name = f"{slide_id}_ood"
        out_dir_ct = os.path.join(out_dir, exp_name)
        os.makedirs(out_dir_ct, exist_ok=True)

        train_path = os.path.join(out_dir_ct, "adata_train.h5ad")
        test_path = os.path.join(out_dir_ct, "adata_test.h5ad")

        adata[~adata.obs["is_holdout"]].copy().write_h5ad(train_path)
        adata.copy().write_h5ad(test_path)

        # 4. Train spatial-prop GNN on training data
        training_args = dict(
            dataset=f"{slide_id}_{holdout_ct}_ood",
            file_path=train_path,
            train_ids=[str(slide_id)],
            test_ids=[str(slide_id)],
            exp_name=exp_name,
            k_hop=2,
            augment_hop=2,
            center_celltypes="all",
            node_feature="expression",
            inject_feature="none",
            learning_rate=1e-3,
            loss="weightedl1",
            epochs=max_epochs,
            normalize_total=True,
            num_cells_per_ct_id=100,
            predict_celltype=False,
            pool="center",
            do_eval=False,
            device=device,
        )
        #_clean_incomplete_gnn_dirs()
        _, gene_names, (gnn_model, model_config, trained_model_path) = (
            train_perturbation_model(**training_args)
        )
        print(f"  Model saved to: {trained_model_path}")
        #clean_all_dirs()

if __name__ == "__main__":
    main()