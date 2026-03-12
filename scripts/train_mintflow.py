import sys

sys.path.append('./scripts')

import scanpy as sc
import os
from tqdm import tqdm
import argparse

from utils import set_seed
from profiler import profile_training

import mintflow


set_seed(0)

ADATA_SAVE_PATH = "/data2/a330d/datasets/crc/processed"
LABELS_KEY = 'coarse_type'
DOMAINS_KEY = 'typ'
NUM_EPOCHS = 51
BATCH_SIZE = 2048
PATIENT_ID = 'sid'
N_NEIGHBORS = 5
CHECKPOINT_INTERVAL = 10
X_POS = 'CenterX_global_px'
Y_POS = 'CenterY_global_px'
USE_WANDB = 'False'
MODEL_OUTPUT_PATH = "/data2/a330d/data/ood/trained"
CSV_PATH = "./results/training_stats.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata_path", required=True)

    return p.parse_args()


def preprocess_adata(adata, slide_id):
    adata.obs_names_make_unique()

    from _labels_to_coarse import LABEL_TO_COARSE as LMAP
    adata.obs['coarse_type'] = adata.obs['ist'].map(LMAP)
    adata.obs['coarse_type'] = adata.obs['coarse_type'].astype('category')
    
    adata = adata[~adata.obs[DOMAINS_KEY].isna()]
    adata = adata[~adata.obs[LABELS_KEY].isna()]

    sc.pp.filter_cells(adata, min_counts=3)
    sc.pp.filter_genes(adata, min_counts=3)

    adata.obs[LABELS_KEY] = adata.obs[LABELS_KEY].astype('category')
    adata.obsm['spatial'] = adata.obs[['CenterX_global_px', 'CenterY_global_px']].values
    adata.layers['counts'] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, layer='counts', flavor='seurat_v3', n_top_genes=2000, subset=True)

    adata.X = adata.layers['counts'].copy() # NOTE: use raw counts for training

    adata.obs["sliceID"] = f"slide_{slide_id}"
    adata.obs["batchID"] = f"slide_{slide_id}"

    adata.obs["sliceID"] = adata.obs["sliceID"].astype("category")
    adata.obs["batchID"] = adata.obs["batchID"].astype("category")

    adata.write_h5ad(f"{ADATA_SAVE_PATH}/{slide_id}.h5ad")


def train_mintflow(adata_save_path, dataset_size, slide_id):
    num_epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE
    labels_key = LABELS_KEY
    patient_id = PATIENT_ID
    n_neighbors = N_NEIGHBORS
    x_pos = X_POS
    y_pos = Y_POS
    use_wandb = USE_WANDB
    path_output_files = f"{MODEL_OUTPUT_PATH}/{slide_id}/mintflow/"
    os.makedirs(path_output_files, exist_ok=True)

    # Set up configs
    config_data_train, config_data_evaluation, config_model, config_training = \
        mintflow.get_default_configurations(
            num_tissue_sections_training=1,
            num_tissue_sections_evaluation=1
        )

    train_file = f"{adata_save_path}/{slide_id}.h5ad"

    # configure tissue section 1 =========
    config_data_train['list_tissue']['anndata1']['file'] = train_file
    #   the absolute path to anndata object of tissue section 1 on disk.

    config_data_train['list_tissue']['anndata1']['obskey_cell_type'] = labels_key
    #   meaning that for the 1st tissue section, cell type labels are provided in `broad_celltypes` column of `adata.obs`.

    config_data_train['list_tissue']['anndata1']['obskey_sliceid_to_checkUnique'] = patient_id
    #   meaning that for the 1st tissue section, tissue section ID (i.e. slice ID) is provided in `sid` column of `adata.obs`

    config_data_train['list_tissue']['anndata1']['obskey_x'] = x_pos
    #   meaning that for the 1st tissue section, spatial x coordinates are provided in `CenterX_global_px` column of `adata.obs`

    config_data_train['list_tissue']['anndata1']['obskey_y'] = y_pos
    #   meaning that for the 1st tissue section, spatial y coordinates are provided in `CenterY_global_px` column of `adata.obs`

    config_data_train['list_tissue']['anndata1']['obskey_biological_batch_key'] = patient_id
    #   meaning that for the 1st tissue section, batch identifier is provided in `info_id` column of `adata.obs`

    config_data_train['list_tissue']['anndata1']['config_dataloader_train']['width_window'] = batch_size
    #   For tissue section one, the crop size of the customized dataloader desribed in Supplementary Fig. 16 of the paper.
    #   The larger this number, the larger the tissue crops, and the bigger the subset of cells in each training iteration.
    #      This implies that more GPU memory would be required during training.
    #   In this notebook after calling `mintflow.setup_data` in Sec 4 the crop(s) are shown on tissue, 
    #      with some information on image title which can help you tune this parameter.
    #   In the manuscript we used `width_window` values between 300 and 800 depending on dataset.

    config_data_train['list_tissue']['anndata1']['config_neighbourhood_graph'] = {
        'n_neighs': n_neighbors,
        'set_diag': 'False',
        'delaunay': 'False',
    }
    #   The parameters for creating the neighbourhood graph for training tissue section 1

    # Eval config - same as train
    config_data_evaluation['list_tissue']['anndata1']['file'] = train_file
    config_data_evaluation['list_tissue']['anndata1']['obskey_cell_type'] = labels_key
    config_data_evaluation['list_tissue']['anndata1']['obskey_sliceid_to_checkUnique'] = patient_id
    config_data_evaluation['list_tissue']['anndata1']['obskey_x'] = x_pos
    config_data_evaluation['list_tissue']['anndata1']['obskey_y'] = y_pos
    config_data_evaluation['list_tissue']['anndata1']['obskey_biological_batch_key'] = patient_id
    config_data_evaluation['list_tissue']['anndata1']['config_dataloader_test']['width_window'] = batch_size
    config_data_evaluation['list_tissue']['anndata1']['config_neighbourhood_graph'] = {
        'n_neighs': n_neighbors,
        'set_diag': 'False',
        'delaunay': 'False',
    }


    config_data_train = mintflow.verify_and_postprocess_config_data_train(config_data_train) 
    config_data_evaluation = mintflow.verify_and_postprocess_config_data_evaluation(config_data_evaluation)

    config_model = mintflow.verify_and_postprocess_config_model(
        config_model,
        num_tissue_sections=1
    )

    config_training['num_training_epochs'] = num_epochs
    config_training['flag_enable_wandb'] = use_wandb
    config_training['flag_finaleval_createanndata_alltissuescombined'] = 'True'
    config_training = mintflow.verify_and_postprocess_config_training(config_training)

    dict_all4_configs = {
        "config_data_train": config_data_train,
        "config_data_evaluation": config_data_evaluation,
        "config_model": config_model,
        "config_training": config_training,
    }

    data_mintflow = mintflow.setup_data(dict_all4_configs=dict_all4_configs)


    model = mintflow.setup_model(
        dict_all4_configs=dict_all4_configs,
        data_mintflow=data_mintflow
    )

    trainer = mintflow.Trainer(
        dict_all4_configs=dict_all4_configs,
        model=model,
        data_mintflow=data_mintflow
    )

    def mintflow_train_loop(model, trainer, data_mintflow, dict_all4_configs, path_output_files, checkpoint_interval=10):
        for epoch in tqdm(range(dict_all4_configs["config_training"]["num_training_epochs"]), desc="Training Epochs"):
            trainer.train_one_epoch()

            # On every nth epoch, save a checkpoint
            if epoch % checkpoint_interval == 0:
                mintflow.dump_checkpoint(
                    model=model,
                    data_mintflow=data_mintflow,
                    dict_all4_configs=dict_all4_configs,
                    path_dump=os.path.join(path_output_files, "checkpoint_epoch_{}.pt".format(epoch)),
                )


    profile_training(
        lambda: mintflow_train_loop(model=model,
                            trainer=trainer,
                            data_mintflow=data_mintflow,
                            dict_all4_configs=dict_all4_configs,
                            path_output_files=path_output_files,
                            checkpoint_interval=CHECKPOINT_INTERVAL
                        ),
        model_name="mintflow",
        num_epochs=num_epochs,
        dataset_size=dataset_size,
        adata_path=train_file,
        csv_path=CSV_PATH
    )


def main():
    args = parse_args()
    slide_id = args.adata_path.split('/')[-1].split('.')[0]

    # 1. Load adata
    adata = sc.read(args.adata_path)
    dataset_size = adata.n_obs

    # 2. Preprocess adata and write to disk
    preprocess_adata(adata, slide_id=slide_id)

    # 3. Train mintflow
    train_mintflow(adata_save_path=ADATA_SAVE_PATH, dataset_size=dataset_size, slide_id=slide_id)


if __name__ == "__main__":
    main()