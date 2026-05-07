import sys

sys.path.append('../../scripts')

import scanpy as sc
import os
from tqdm import tqdm
import argparse

import mintflow

from utils import set_seed
from configs.adata_crc_config import ADATA_ARGS as ADATA_CRC_ARGS
from configs.adata_merfish_config import ADATA_ARGS as ADATA_MERFISH_ARGS
from train_loo import preprocess_crc, preprocess_merfish

set_seed(0)
DATASET_NAME = "merfish"  # or "crc"

CRC_BASE_PATH = "/../../datasets/crc/raw_zenodo"
CRC_SLIDES = ['crc_242', 'crc_232', 'crc_231', 'crc_210', 'crc_221', 'crc_120']

MERFISH_BASE_PATH = "/../../datasets/MERFISH_mouse_brain"
MERFISH_SLIDES = ['C57BL6J-2.036', 'C57BL6J-2.039', 'C57BL6J-2.041']

ADATA_BASE_PATH = CRC_BASE_PATH if DATASET_NAME == "crc" else MERFISH_BASE_PATH
SLIDES = CRC_SLIDES if DATASET_NAME == "crc" else MERFISH_SLIDES
DATA_ARGS = ADATA_CRC_ARGS if DATASET_NAME == "crc" else ADATA_MERFISH_ARGS

NUM_EPOCHS = 100
BATCH_SIZE = 2048
LABELS_KEY = DATA_ARGS.get('labels_key')
DOMAINS_KEY = DATA_ARGS.get('domains_key')
N_TOP_GENES = DATA_ARGS.get('n_top_genes')
PATIENT_ID = DATA_ARGS.get('batch_key')
N_NEIGHBORS = 5
CHECKPOINT_INTERVAL = 10
USE_WANDB = 'False'
MODEL_OUTPUT_PATH = "/../../data/ood/trained"
ADATA_SAVE_PATH = f"/../../datasets/{DATASET_NAME}/processed"
X_POS = 'CenterX_global_px'
Y_POS = 'CenterY_global_px'


def train_mintflow(adata_save_path, slide_id):
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


    mintflow_train_loop(model=model,
                            trainer=trainer,
                            data_mintflow=data_mintflow,
                            dict_all4_configs=dict_all4_configs,
                            path_output_files=path_output_files,
                            checkpoint_interval=CHECKPOINT_INTERVAL
                        )


def main():
    for slide_id in SLIDES:
        print(f"\n{'='*60}\nProcessing slide {slide_id}\n{'='*60}")
        adata = sc.read_h5ad(f"{ADATA_BASE_PATH}/{slide_id}.h5ad")
        if DATASET_NAME == 'crc':
            adata = preprocess_crc(adata, n_top_genes=N_TOP_GENES, n_neighbors=N_NEIGHBORS, labels_key=LABELS_KEY, domains_key=DOMAINS_KEY)
        elif DATASET_NAME == 'merfish':
            adata = preprocess_merfish(adata, n_top_genes=N_TOP_GENES, n_neighbors=N_NEIGHBORS, labels_key=LABELS_KEY, domains_key=DOMAINS_KEY)
        else:
            raise ValueError(f"Unknown dataset_name: {DATASET_NAME}. Supported: crc, merfish")
        adata.obs[X_POS] = adata.obsm['spatial'][:, 0]
        adata.obs[Y_POS] = adata.obsm['spatial'][:, 1]
        os.makedirs(ADATA_SAVE_PATH, exist_ok=True)
        adata.write_h5ad(f"{ADATA_SAVE_PATH}/{slide_id}.h5ad")

        # Train mintflow
        train_mintflow(adata_save_path=ADATA_SAVE_PATH, slide_id=slide_id)


if __name__ == "__main__":
    main()