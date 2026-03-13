import os
import numpy as np
import scanpy as sc
from benchmark_pipeline import (
    run_simvi_in_subprocess,
    split_by_sample,
    run_pca_baseline,
    run_scvi_model,
    run_cellina_model,
    run_scanvi_model,
    run_scviva_model,
    plot_results_table
)
import scvi
from scvi.train._callbacks import SaveCheckpoint
import torch

METHODS = {
    #"SCVI": run_scvi_model,
    #"SCANVI": run_scanvi_model,
    #"scVIVA": run_scviva_model,
    "CELLINA": run_cellina_model,
    # "PCA": run_pca_baseline,
    # "SIMVI": run_simvi_in_subprocess,
}

class BenchmarkPipelineRunner:
    def __init__(self,
                 dataset_base_path,
                 dataset_name,
                 dataset_defaults_key='default_params',
                 celltype_key=None,
                 niche_key=None,
                 sample_key=None,
                 batch_key=None,
                 K_NN=None,
                 n_holdout=None,
                 n_val_samples=None,
                 train_frac=None,
                 test_samples=None,
                 n_layers=2,
                 n_latent=64,
                 early_stopping_patience=20,
                 max_epochs=None,
                 batch_size=128,
                 n_hidden=300,
                 profiler=True,
                 **kwargs):
        
        try:
            adata = sc.read(get_adata_path(dataset_base_path, dataset_name))
        except FileNotFoundError:
            raw_data_path = f"{dataset_base_path}/{dataset_name}.h5ad"
            print(f"Processed file not found, loading raw data from {raw_data_path}")
            adata = sc.read(raw_data_path)
            adata.layers["counts"] = adata.X.copy()
            
        self.adata_path = get_adata_path(dataset_base_path, dataset_name)

        # Assign constructor arguments to attributes first so explicit args take precedence
        self.sample_key = sample_key
        self.batch_key = batch_key
        self.celltype_key = celltype_key
        self.niche_key = niche_key
        self.K_NN = K_NN
        self.n_holdout = n_holdout
        self.n_val_samples = n_val_samples
        self.train_frac = train_frac
        self.test_samples = test_samples
        self.n_layers = n_layers
        self.n_latent = n_latent
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_hidden = n_hidden

        # Load dataset defaults but only apply them for attributes that are still None
        defaults = {}
        if dataset_defaults_key is not None and dataset_defaults_key in adata.uns:
            try:
                defaults = adata.uns[dataset_defaults_key] or {}
            except Exception:
                defaults = {}
        for k, v in defaults.items():
            if getattr(self, k, None) is None:
                setattr(self, k, v)
        
        # Apply any extra kwargs passed to the constructor
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        self.adata = adata
        self.dataset_base_path = dataset_base_path
        self.dataset_name = dataset_name
        self.dataset_path = f"{dataset_base_path}/{dataset_name}.h5ad"
        self.n_layers = n_layers
        self.n_latent = n_latent
        self.profiler = profiler
        
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def __repr__(self):
        # loop over all attributes and print them
        attrs = [f"{k}={v}\n" for k, v in self.__dict__.items() if not (k.startswith('_') or k.startswith('adata'))]
        # except adata
        
        return f"{self.__class__.__name__}({'- '.join(attrs)})"

    def run_pipeline(self, adata, **kwargs):
        # Allow overriding parameters via kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # For reproducibility
        np.random.seed(0)
        torch.manual_seed(0)
        scvi.settings.seed = 0
        print("Last run with scvi-tools version:", scvi.__version__)
        
        output_path = self.adata_path

        train_indices, validation_indices, test_indices = split_by_sample(
            adata,
            sample_key=self.sample_key,
            n_holdout=self.n_holdout,
            n_val_samples=self.n_val_samples,
            train_frac=self.train_frac,
            test_samples=self.test_samples,
        )
        data_splitter_kwargs = {"external_indexing": [train_indices, validation_indices, test_indices]}
        print("-------- Data Split ---------")
        print("Val Samples:")
        print(adata.obs.iloc[validation_indices][self.sample_key].unique())
        print("Heldout Samples:")
        print(adata.obs.iloc[test_indices][self.sample_key].unique())
        print("Count of cells in each split:")        
        nums = len(train_indices), len(validation_indices), len(test_indices)
        print(nums)
        print("Proportions of samples in each split:")
        nums / np.sum(nums)
        print("-----------------------------")

        for name, method in METHODS.items():
            print(f"Running {name}...")
            
            dirpath = f"../models/{self.dataset_name}/{name}"
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            
            if name in ['SIMVI', 'scVIVA', 'SCVI', 'SCANVI', 'CELLINA']:
                model_kwargs = {'n_hidden': self.n_hidden}
            else:
                model_kwargs = {}
            
            train_kwargs = {
                    'early_stopping_patience': self.early_stopping_patience,
                    'early_stopping': True,
                    'enable_checkpointing':True,
                    'callbacks':[
                    SaveCheckpoint(monitor='validation_loss',#"elbo_validation",
                                   dirpath=dirpath,
                                   load_best_on_end=True),
                    ],
            }
            extra_args = {}
            if name == "CELLINA":
                extra_arg_keys = ['classifier_lambda', 'discriminator_lambda', 'save_lambda_in_key']
                for k in extra_arg_keys:
                    if getattr(self, k, None) is not None:
                        extra_args[k] = getattr(self, k)
            method(
                adata=adata,
                batch_key=self.batch_key,
                sample_key=self.sample_key,
                celltype_key=self.celltype_key,
                niche_key=self.niche_key,
                output_path=output_path,
                data_splitter_kwargs=data_splitter_kwargs,
                n_layers=self.n_layers,
                n_latent=self.n_latent,
                max_epochs=self.max_epochs,
                K_NN=self.K_NN,
                model_kwargs=model_kwargs,
                train_kwargs=train_kwargs,
                batch_size=self.batch_size,
                dataset_name=self.dataset_name,
                dataset_path=self.dataset_path,
                profiler=self.profiler,
                **extra_args
            )

def main(
    batch_key,
    celltype_key,
    niche_key,
    n_layers,
    n_latent,
    early_stopping_patience,
    max_epochs,
    batch_size,
    n_hidden,
    K_NN,
    **kwargs
    ):
    runner = BenchmarkPipelineRunner(
        adata=adata,
        batch_key=batch_key,
        celltype_key=celltype_key,
        niche_key=niche_key,
        n_layers=n_layers,
        n_latent=n_latent,
        early_stopping_patience=early_stopping_patience,
        max_epochs=max_epochs,
        batch_size=batch_size,
        n_hidden=n_hidden,
        K_NN=K_NN,
        **kwargs
    )
    runner.run_pipeline(
        adata,
        **kwargs
    )

if __name__ == "__main__":
    import argparse
    import ast
    parser = argparse.ArgumentParser(description="Run benchmark pipeline with specific parameters.")
    parser.add_argument("--dataset_name", required=True, help="Dataset name")
    parser.add_argument("--batch_key", required=True, help="Batch key")
    parser.add_argument("--celltype_key", required=True, help="Celltype key")
    parser.add_argument("--niche_key", required=True, help="Condition key")
    parser.add_argument("--params", nargs='*', help="Additional key=value parameters", default=[])

    args = parser.parse_args()
    kwargs = {}
    for arg in args.params:
        if '=' in arg:
            k, v = arg.split('=', 1)
            try:
                v = ast.literal_eval(v)
            except Exception:
                pass
            kwargs[k] = v
    main(
        args.dataset_name,
        args.batch_key,
        args.celltype_key,
        args.niche_key,
        **kwargs
    )

def get_adata_path(base_path, dataset):
    return f"{base_path}/{dataset}.h5ad"