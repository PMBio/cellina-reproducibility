import sys
import os
import pickle
import mintflow
import torch
import scanpy as sc

from tqdm import tqdm

sys.path.append('./scripts')
from utils import set_seed
from profiler import profile_training

set_seed(0)

ADATA_SAVE_PATH = "/data/a330d/datasets/crc/processed"
#SLIDES = ['120', '210', '221', '231', '232', '242']
SLIDES = ['221', '231', '232', '242']
LABELS_KEY = 'coarse_type'
DOMAINS_KEY = 'typ'
NUM_EPOCHS = 51
BATCH_SIZE = 2048
PATIENT_ID = 'sid'
N_NEIGHBORS = 10
CHECKPOINT_INTERVAL = 10
X_POS = 'CenterX_global_px'
Y_POS = 'CenterY_global_px'
USE_WANDB = 'False'
MODEL_OUTPUT_PATH = "/data/a330d/data/ood/trained/mintflow_joint"
CSV_PATH = "./results/training_stats.csv"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def preprocess_adata(adata, slide_ids):
    adatas = {k: None for k in slide_ids}

    for sid in slide_ids:
        adata_sub = adata[adata.obs['sid']==int(sid)].copy()

        adata_sub.obs["sliceID"] = f"slide_{sid}"
        adata_sub.obs["batchID"] = f"slide_{sid}"

        adata_sub.obs["sliceID"] = adata_sub.obs["sliceID"].astype("category")
        adata_sub.obs["batchID"] = adata_sub.obs["batchID"].astype("category")

        adatas[sid] = adata_sub
    
    for sid, dataset in adatas.items():
        dataset.write_h5ad(f"{ADATA_SAVE_PATH}/crc_{sid}.h5ad")

    return sum([a.n_obs for _, a in adatas.items()])


def train_mintflow(adata_save_path, dataset_size, slide_ids):
    num_epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE
    labels_key = LABELS_KEY
    patient_id = PATIENT_ID
    n_neighbors = N_NEIGHBORS
    x_pos = X_POS
    y_pos = Y_POS
    use_wandb = USE_WANDB
    path_output_files = MODEL_OUTPUT_PATH
    os.makedirs(path_output_files, exist_ok=True)

    # Set up configs
    config_data_train, config_data_evaluation, config_model, config_training = \
        mintflow.get_default_configurations(
            num_tissue_sections_training=len(slide_ids),
            num_tissue_sections_evaluation=len(slide_ids)
        )

    for idx, sid in enumerate(slide_ids):
        print(idx, f"{adata_save_path}/crc_{sid}.h5ad")
        # training config
        config_data_train['list_tissue'][f'anndata{idx+1}']['file'] = f"{adata_save_path}/crc_{sid}.h5ad"
        config_data_train['list_tissue'][f'anndata{idx+1}']['obskey_cell_type'] = labels_key
        config_data_train['list_tissue'][f'anndata{idx+1}']['obskey_sliceid_to_checkUnique'] = patient_id
        config_data_train['list_tissue'][f'anndata{idx+1}']['obskey_x'] = x_pos
        config_data_train['list_tissue'][f'anndata{idx+1}']['obskey_y'] = y_pos
        config_data_train['list_tissue'][f'anndata{idx+1}']['obskey_biological_batch_key'] = patient_id
        config_data_train['list_tissue'][f'anndata{idx+1}']['config_dataloader_train']['width_window'] = batch_size
        config_data_train['list_tissue'][f'anndata{idx+1}']['config_neighbourhood_graph'] = {
            'n_neighs': n_neighbors,
            'set_diag': 'False',
            'delaunay': 'False',
        }

        config_data_evaluation['list_tissue'][f'anndata{idx+1}']['file'] = f"{adata_save_path}/crc_{sid}.h5ad"
        config_data_evaluation['list_tissue'][f'anndata{idx+1}']['obskey_cell_type'] = labels_key
        config_data_evaluation['list_tissue'][f'anndata{idx+1}']['obskey_sliceid_to_checkUnique'] = patient_id
        config_data_evaluation['list_tissue'][f'anndata{idx+1}']['obskey_x'] = x_pos
        config_data_evaluation['list_tissue'][f'anndata{idx+1}']['obskey_y'] = y_pos
        config_data_evaluation['list_tissue'][f'anndata{idx+1}']['obskey_biological_batch_key'] = patient_id
        config_data_evaluation['list_tissue'][f'anndata{idx+1}']['config_dataloader_test']['width_window'] = batch_size
        config_data_evaluation['list_tissue'][f'anndata{idx+1}']['config_neighbourhood_graph'] = {
            'n_neighs': n_neighbors,
            'set_diag': 'False',
            'delaunay': 'False',
        }

    config_model['coef_xbarint2notbatchID_loss'] = 1.0
    config_model['coef_xbarspl2notbatchID_loss'] = 1.0

    config_data_train = mintflow.verify_and_postprocess_config_data_train(config_data_train) 
    config_data_evaluation = mintflow.verify_and_postprocess_config_data_evaluation(config_data_evaluation)

    config_model = mintflow.verify_and_postprocess_config_model(
        config_model,
        num_tissue_sections=len(config_data_train)
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
                # get/save the predictions
                predictions = mintflow.predict(
                    device=DEVICE,
                    dict_all4_configs=dict_all4_configs,
                    data_mintflow=data_mintflow,
                    model=model,
                    evalulate_on_sections="all",
                )
                with open(os.path.join(path_output_files, "predictions_epoch_{}.pkl".format(epoch)), 'wb') as f:
                    pickle.dump(predictions, f)

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
        dataset_name="crc",
        dataset_size=dataset_size,
        dataset_path=adata_save_path,
        csv_path=CSV_PATH
    )


def main():
    # 1. Load adata
    #adata = sc.read(f"{ADATA_SAVE_PATH}/crc_cosmx_wt.h5ad")
    

    # 2. Preprocess adata and write to disk
    #dataset_size = preprocess_adata(adata, slide_ids=SLIDES)

    # 3. Train mintflow
    train_mintflow(adata_save_path=ADATA_SAVE_PATH, 
                   #dataset_size=dataset_size, 
                   dataset_size=10,
                   slide_ids=SLIDES)


if __name__ == "__main__":
    main()