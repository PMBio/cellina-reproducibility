import argparse
import logging
from datetime import datetime
import os
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
import shutil

from utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(args):
    set_seed(args.seed)

    logging.info("Loading AnnData...")
    adata = sc.read(args.input)
    adata.obs_names_make_unique()

    # map coarse labels (copy mapping from notebook)
    label_to_coarse = {
        "epi1": "Epithelial",
        "epi2": "Epithelial",
        "epi3": "Epithelial",
        "epi4": "Epithelial",
        "fib1": "Fibroblast",
        "fib2": "Fibroblast",
        "EC": "Endothelial",
        "SMC": "Smooth_muscle",
        "BC": "B_cell",
        "PC_IgA": "Plasma_cell",
        "PC_IgG": "Plasma_cell",
        "PC_IgM": "Plasma_cell",
        "TC": "T_cell",
        "mye1": "Myeloid",
        "mye2": "Myeloid",
        "mast": "Mast_cell",
    }
    labels_key = args.labels_key
    domains_key = args.domains_key

    adata.obs[labels_key] = adata.obs.get(labels_key, adata.obs.get('ist')).map(label_to_coarse)
    adata = adata[~adata.obs[domains_key].isna()]
    adata = adata[~adata.obs[labels_key].isna()]

    # basic QC from notebook
    sc.pp.filter_cells(adata, min_counts=3)
    sc.pp.filter_genes(adata, min_counts=3)

    adata.obs[labels_key] = adata.obs[labels_key].astype("category")
    adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values

    # keep original counts in layers and optionally subset HVG as in notebook
    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, layer="counts", flavor="seurat_v3", n_top_genes=2000, subset=True)

    # Build splits (random or OOD as in notebook)
    if args.split == "random":
        n_holdout = int(adata.n_obs * args.holdout_fraction)
        test_idx = np.random.choice(adata.n_obs, n_holdout, replace=False)
    elif args.split == "ood":
        is_tumor_region = adata.obs[domains_key].str.contains("CRC", regex=True)
        #is_holdout_ct = adata.obs[labels_key].isin(args.holdout_celltype)
        is_holdout_ct = adata.obs[labels_key]==args.holdout_celltype
        test_mask = (is_tumor_region) & (is_holdout_ct)
        test_idx = np.where(test_mask)[0]
    else:
        raise ValueError("split must be one of ['random','ood']")

    all_idx = np.arange(adata.n_obs)
    trainval_idx = np.setdiff1d(all_idx, test_idx)

    # train/val split
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(trainval_idx, test_size=args.val_fraction, random_state=args.seed, shuffle=True)

    adata.obs["is_holdout"] = False
    adata.obs.iloc[test_idx, adata.obs.columns.get_loc("is_holdout")] = True
    adata.obs["data_split"] = "train"
    adata.obs.iloc[val_idx, adata.obs.columns.get_loc("data_split")] = "valid"
    adata.obs.iloc[test_idx, adata.obs.columns.get_loc("data_split")] = "ood"

    # CPA specific expectations
    # ensure numeric dose
    adata.obs["dose"] = 1.0
    # set perturbation: ctrl vs perturbed using domain annotation (same as notebook)
    adata.obs["perturbation"] = "ctrl"
    mask = adata.obs[domains_key].str.contains("CRC", regex=True)
    adata.obs.loc[mask, "perturbation"] = "perturbed"

    # ensure counts are CSR for scvi/cpa speed
    try:
        adata.layers["counts"] = csr_matrix(adata.layers["counts"])
    except Exception:
        pass
    adata.X = adata.layers.get("counts", adata.X)

    # Setup CPA anndata
    import cpa
    cpa.CPA.setup_anndata(
        adata,
        perturbation_key="perturbation",
        control_group="ctrl",
        dosage_key="dose",
        categorical_covariate_keys=[labels_key],
        is_count_data=True,
        max_comb_len=1,
    )

    # model and trainer params copied/trimmed from notebook
    model_params = {
        "n_latent": args.n_latent,
        "recon_loss": args.recon_loss,
        "doser_type": "linear",
        "n_hidden_encoder": args.n_hidden_encoder,
        "n_layers_encoder": args.n_layers_encoder,
        "n_hidden_decoder": args.n_hidden_decoder,
        "n_layers_decoder": args.n_layers_decoder,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": False,
        "use_layer_norm_decoder": True,
        "dropout_rate_encoder": 0.0,
        "dropout_rate_decoder": 0.1,
        "variational": False,
        "seed": args.seed,
    }

    trainer_params = {
        "n_epochs_pretrain_ae": args.pretrain_ae,
        "n_epochs_adv_warmup": args.adv_warmup,
        "mixup_alpha": 0.0,
        "adv_steps": None,
        "n_hidden_adv": 64,
        "n_layers_adv": 3,
        "use_batch_norm_adv": True,
        "dropout_rate_adv": 0.3,
        "reg_adv": 20.0,
        "pen_adv": 5.0,
        "lr": args.lr,
        "wd": 4e-07,
        "adv_lr": args.lr,
        "adv_wd": 4e-07,
        "adv_loss": "cce",
        "doser_lr": args.lr,
        "doser_wd": 4e-07,
        "do_clip_grad": True,
        "gradient_clip_value": 1.0,
        "step_size_lr": 10,
    }

    # create model
    model = cpa.CPA(
        adata=adata,
        split_key="data_split",
        train_split="train",
        valid_split="valid",
        test_split="ood",
        **model_params,
    )

    # train
    save_path = os.path.join(args.save_dir, f"cpa_{args.holdout_celltype}")

    # Remove existing directory if it exists
    if os.path.exists(save_path):
        logging.info("Save directory exists, clearing it: %s", save_path)
        shutil.rmtree(save_path)

    # Re-create empty directory
    os.makedirs(save_path, exist_ok=True)

    logging.info("Starting training for %s...", args.holdout_celltype)
    
    model.train(
        max_epochs=args.epochs,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
        plan_kwargs=trainer_params,
        early_stopping_patience=args.early_stopping,
        check_val_every_n_epoch=args.check_val_every,
        save_path=save_path,
    )
    logging.info("Saving model to %s", save_path)
    model.save(dir_path=save_path)
    logging.info("Done.")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to h5ad")
    p.add_argument("--save-dir", default="/data2/a330d/data/cellina-reproducibility", dest="save_dir")
    p.add_argument("--split", choices=["ood", "random"], default="ood")
    p.add_argument("--holdout-fraction", type=float, default=0.1)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n-latent", dest="n_latent", type=int, default=64)
    p.add_argument("--recon-loss", dest="recon_loss", default="nb")
    p.add_argument("--n-hidden-encoder", dest="n_hidden_encoder", type=int, default=128)
    p.add_argument("--n-layers-encoder", dest="n_layers_encoder", type=int, default=2)
    p.add_argument("--n-hidden-decoder", dest="n_hidden_decoder", type=int, default=512)
    p.add_argument("--n-layers-decoder", dest="n_layers_decoder", type=int, default=2)
    p.add_argument("--pretrain-ae", type=int, default=30)
    p.add_argument("--adv-warmup", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--early-stopping", dest="early_stopping", type=int, default=5)
    p.add_argument("--check-val-every", dest="check_val_every", type=int, default=5)
    p.add_argument("--holdout-celltype", default="Epithelial")
    p.add_argument("--labels-key", default="coarse_type", type=str)
    p.add_argument("--domains-key", default="typ", type=str)
    args = p.parse_args()

    raise SystemExit(main(args))