import scvi
from scvi.model import SCVI
from scvi.model import SCANVI
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import subprocess
import os
import json

from typing import Optional

from cellina import CellinaModel
from anndata import AnnData

import sys
sys.path.append('./scripts')

from profiler import profile_training

PROFILER_CSV_PATH = "../../results/training_stats.csv"

OBSM_KEYS = [
    "Unintegrated",
    "scVI", 
    "SCANVI", 
    "scVIVA",
    'Cellina_Basal', 
    'Cellina_Spatial', 
    'Cellina_Shifted',
    'SIMVI_Intrinsic', 
    'SIMVI_Spatial',
    'Cellina_MMD_Basal',
    'Cellina_MMD_Spatial',
    'Cellina_MMD_Shifted'
    ]

def split_by_sample(
    adata: AnnData,
    sample_key: str,
    n_holdout: int = 3,
    train_frac: Optional[float] = 0.8,
    n_val_samples: Optional[int] = False,
    test_samples = False,
    seed: int = 42,
    ):
    """
    Split an AnnData object into train/validation/test indices based on sample IDs.

    This function reserves a specified number of samples for the test set. It then
    splits the remaining data into training and validation sets using one of two methods:
    1. By proportion of observations (cells): Use `train_frac`.
    2. By a specific number of samples: Use `n_val_samples`.

    Only one of `train_frac` or `n_val_samples` should be provided.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    sample_key : str
        Key in `adata.obs` identifying sample labels.
    n_holdout : int
        Number of unique samples to reserve for the test (holdout) set.
    train_frac : float, optional
        Fraction of non-test OBSERVATIONS to use for training. The rest become
        validation. Mutually exclusive with `n_val_samples`. Defaults to 0.8.
    n_val_samples : int, optional
        Number of non-test SAMPLES to use for the validation set. The remaining
        non-test samples will form the training set. Mutually exclusive with
        `train_frac`. Defaults to None.
    test_samples
        Which samples to holdout - if None, chosen randomly.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_indices : np.ndarray
        Indices of training observations.
    validation_indices : np.ndarray
        Indices of validation observations.
    test_indices : np.ndarray
        Indices of test (holdout) observations.
    """
    # --- Input Validation ---
    if (train_frac is np.False_ and n_val_samples is np.False_) or \
       (train_frac is not np.False_ and n_val_samples is not np.False_):
        raise ValueError("Please specify either 'train_frac' or 'n_val_samples', but not both.")

    rng = np.random.default_rng(seed)
    all_samples = adata.obs[sample_key].unique()

    if n_holdout >= len(all_samples):
        raise ValueError(
            f"'n_holdout' ({n_holdout}) must be smaller than the total "
            f"number of samples ({len(all_samples)})."
        )

    # --- Test Split ---
    # Randomly select samples for the test set
    # if test_samples is not an iterable
    if not isinstance(test_samples, (list, np.ndarray)):
        test_samples = rng.choice(all_samples, size=n_holdout, replace=False)
    is_test = adata.obs[sample_key].isin(test_samples)
    test_indices = np.nonzero(is_test.to_numpy())[0]

    # --- Train/Validation Split ---
    remaining_samples = np.setdiff1d(all_samples, test_samples)
    
    # Check if there are enough samples left for validation
    if not n_val_samples and n_val_samples >= len(remaining_samples):
        raise ValueError(
            f"'n_val_samples' ({n_val_samples}) must be smaller than the number of "
            f"remaining non-test samples ({len(remaining_samples)})."
        )

    if n_val_samples:
        # --- Method 2: Split by number of validation SAMPLES ---
        if not isinstance(n_val_samples, np.int64) or n_val_samples < 0:
             raise ValueError("'n_val_samples' must be a non-negative integer.")
        
        val_samples = rng.choice(remaining_samples, size=n_val_samples, replace=False)
        train_samples = np.setdiff1d(remaining_samples, val_samples)

        is_train = adata.obs[sample_key].isin(train_samples)
        is_val = adata.obs[sample_key].isin(val_samples)

        train_indices = np.nonzero(is_train.to_numpy())[0]
        validation_indices = np.nonzero(is_val.to_numpy())[0]

    else:  # train_frac is not None
        # --- Method 1: Split by proportion of OBSERVATIONS ---
        if not isinstance(train_frac, float) or not (0.0 < train_frac < 1.0):
             raise ValueError("'train_frac' must be a float between 0 and 1.")

        remaining_indices = np.nonzero(~is_test.to_numpy())[0]
        rng.shuffle(remaining_indices)

        n_train = int(len(remaining_indices) * train_frac)
        train_indices = remaining_indices[:n_train]
        validation_indices = remaining_indices[n_train:]

    # --- Update AnnData object with split information - Needed for DSA ---
    split_labels = np.full(adata.n_obs, "unassigned", dtype=object)
    split_labels[train_indices] = "train"
    split_labels[test_indices] = "test"
    split_labels[validation_indices] = "val"
    adata.obs['split'] = pd.Categorical(split_labels)

    adata.obs["is_supervised"] = True
    adata.obs["is_holdout"] = False
    adata.obs.loc[adata.obs.index[test_indices], "is_holdout"] = True
    adata.obs["is_validation"] = False
    adata.obs.loc[adata.obs.index[validation_indices], "is_validation"] = True

    return train_indices, validation_indices, test_indices


def run_scvi_model(
    adata,
    batch_key,
    sample_key,
    celltype_key, # NOTE
    niche_key, # NOTE
    output_path,
    data_splitter_kwargs,
    n_layers=2,
    n_latent=30,
    max_epochs=None, # NOTE merge into model_kwargs
    K_NN=20, # NOTE
    model_kwargs={},
    train_kwargs={},
    batch_size=256,
    profiler=True,
    dataset_name="",
    dataset_path="",
    ):
    SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    scvi_model = SCVI(adata, gene_likelihood="nb", n_layers=n_layers, n_latent=n_latent, **model_kwargs)
    
    if profiler:
        profile_training(
            lambda: scvi_model.train(datasplitter_kwargs=data_splitter_kwargs, 
                                    batch_size=batch_size, 
                                    max_epochs=max_epochs, 
                                    **train_kwargs),
            model_name="scvi",
            num_epochs=max_epochs,
            dataset_name=dataset_name,
            dataset_size=adata.n_obs,
            dataset_path=dataset_path,
            csv_path=PROFILER_CSV_PATH
        )
    else:
        scvi_model.train(datasplitter_kwargs=data_splitter_kwargs, 
                                 batch_size=batch_size, 
                                 max_epochs=max_epochs, 
                                 **train_kwargs)

    adata.obsm["scVI"] = scvi_model.get_latent_representation(batch_size=batch_size)
    adata.write(output_path)
    

def run_cellina_model(
    adata,
    batch_key,
    sample_key,
    celltype_key, # NOTE
    niche_key, # NOTE
    output_path,
    data_splitter_kwargs,
    n_layers=2,
    n_latent=30,
    max_epochs=None, # NOTE merge into model_kwargs
    K_NN=20, # NOTE
    model_kwargs={},
    train_kwargs={},
    batch_size=256,
    classifier_lambda=1.0,
    discriminator_lambda=1.0,
    save_lambda_in_key=False,
    profiler=True,
    dataset_name="",
    dataset_path="",
):
    print(f"Running Cellina with classifier_lambda={classifier_lambda}, discriminator_lambda={discriminator_lambda}")
    CellinaModel.setup_anndata(adata,
                           batch_key=batch_key,
                           labels_key=celltype_key, 
                           domains_key=niche_key, 
                           spatial_obsm_key="spatial_x",
                           layer='counts')
    model = CellinaModel(
            adata, n_latent=n_latent,
            n_layers=n_layers,
            classifier_lambda=classifier_lambda,
            discriminator_lambda=discriminator_lambda,
            condition_on_intrinsic=False,
            gene_likelihood="nb",
            **model_kwargs
        )
    
    if profiler:
        profile_training(
            lambda: model.train(
                max_epochs=max_epochs,
                plan_kwargs={
                    'lr': 1e-4,
                    'normalize_losses': True,
                },
                datasplitter_kwargs=data_splitter_kwargs,
                batch_size=batch_size,
                **train_kwargs,
            ),
            model_name="cellina",
            num_epochs=max_epochs,
            dataset_name=dataset_name,
            dataset_size=adata.n_obs,
            dataset_path=dataset_path,
            csv_path=PROFILER_CSV_PATH
        )
    else:
        model.train(
            max_epochs=max_epochs,
            plan_kwargs={
                'lr': 1e-4,
                'normalize_losses': True,
            },
            datasplitter_kwargs=data_splitter_kwargs,
            batch_size=batch_size,
            **train_kwargs,
        )
    basal_key = f"Cellina_Basal" if not save_lambda_in_key else f"Cellina_Basal_{classifier_lambda}_{discriminator_lambda}"
    spatial_key = f"Cellina_Spatial" if not save_lambda_in_key else f"Cellina_Spatial_{classifier_lambda}_{discriminator_lambda}"
    shifted_key = f"Cellina_Shifted" if not save_lambda_in_key else f"Cellina_Shifted_{classifier_lambda}_{discriminator_lambda}"
    adata.obsm[basal_key] = model.get_latent_representation(latent_key='z', batch_size=batch_size)
    adata.obsm[spatial_key] = model.get_latent_representation(latent_key='s', batch_size=batch_size)
    adata.obsm[shifted_key] = model.get_latent_representation(batch_size=batch_size)
    adata.write(output_path)       


def run_cellina_mmd_model(
    adata,
    batch_key,
    sample_key,
    celltype_key, # NOTE
    niche_key, # NOTE
    output_path,
    data_splitter_kwargs,
    n_layers=2,
    n_latent=30,
    max_epochs=None, # NOTE merge into model_kwargs
    K_NN=20, # NOTE
    model_kwargs={},
    train_kwargs={},
    batch_size=256,
    classifier_lambda=1.0,
    discriminator_lambda=1.0,
    save_lambda_in_key=False,
    profiler=True,
    dataset_name="",
    dataset_path="",
):
    classifier_lambda = 0.0
    discriminator_lambda = 0.0
    mmd_lambda = 1.0
    supervised = False
    print(f"Running Cellina MMD with classifier_lambda={classifier_lambda}, discriminator_lambda={discriminator_lambda}, mmd_lambda={mmd_lambda}")
    CellinaModel.setup_anndata(adata,
                           batch_key=batch_key,
                           labels_key=celltype_key, 
                           domains_key=niche_key, 
                           spatial_obsm_key="spatial_x",
                           layer='counts')
    model = CellinaModel(
            adata, n_latent=n_latent,
            n_layers=n_layers,
            classifier_lambda=classifier_lambda,
            discriminator_lambda=discriminator_lambda,
            condition_on_intrinsic=False,
            gene_likelihood="nb",
            mmd_lambda=mmd_lambda,
            supervised=supervised,
            **model_kwargs
        )
    
    if profiler:
        profile_training(
            lambda: model.train(
                max_epochs=max_epochs,
                plan_kwargs={
                    'lr': 1e-4,
                    'normalize_losses': True,
                },
                datasplitter_kwargs=data_splitter_kwargs,
                batch_size=batch_size,
                **train_kwargs,
            ),
            model_name="cellina-mmd",
            num_epochs=max_epochs,
            dataset_name=dataset_name,
            dataset_size=adata.n_obs,
            dataset_path=dataset_path,
            csv_path=PROFILER_CSV_PATH
        )
    else:
        model.train(
            max_epochs=max_epochs,
            plan_kwargs={
                'lr': 1e-4,
                'normalize_losses': True,
            },
            datasplitter_kwargs=data_splitter_kwargs,
            batch_size=batch_size,
            **train_kwargs,
        )
    basal_key = "Cellina_MMD_Basal"
    spatial_key = "Cellina_MMD_Spatial"
    shifted_key = "Cellina_MMD_Shifted"
    adata.obsm[basal_key] = model.get_latent_representation(latent_key='z', batch_size=batch_size)
    adata.obsm[spatial_key] = model.get_latent_representation(latent_key='s', batch_size=batch_size)
    adata.obsm[shifted_key] = model.get_latent_representation(batch_size=batch_size)
    adata.write(output_path)       


def run_pca_baseline(
    adata,
    batch_key,
    sample_key,
    niche_key,
    celltype_key,
    output_path,
    data_splitter_kwargs,
    n_layers=2,
    n_latent=30,
    K_NN=20,
    max_epochs=None,
    model_kwargs=None,
    train_kwargs=None,
    batch_size=256,
    profiler=False,
    dataset_name="",
    dataset_path=""
):
    if model_kwargs is None:
        model_kwargs = {}
    if train_kwargs is None:
        train_kwargs = {}

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.pp.highly_variable_genes(adata)
    sc.tl.pca(adata, n_comps=n_latent, **model_kwargs)
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    adata.X = adata.layers['counts'].copy()

    adata.write(output_path)
    
def run_scanvi_model(
    adata,
    batch_key,
    sample_key,
    celltype_key,
    niche_key,
    output_path,
    data_splitter_kwargs,
    n_layers=2,
    n_latent=30,
    max_epochs=None,
    K_NN=20,
    model_kwargs={},
    train_kwargs={},
    batch_size=256,
    profiler=True,
    dataset_name="",
    dataset_path="",
):
    SCANVI.setup_anndata(
        adata,
        layer="counts",
        batch_key=batch_key,
        unlabeled_category="ignore",
        labels_key=celltype_key,
    )
    scanvi_model = SCANVI(
        adata=adata,
        n_layers=n_layers,
        n_latent=n_latent,
        linear_classifier=True,
        **model_kwargs,
    )
    if max_epochs is not None:
        train_kwargs.setdefault("max_epochs", max_epochs)
    
    if profiler:
        profile_training(
            lambda: scanvi_model.train(datasplitter_kwargs=data_splitter_kwargs, batch_size=batch_size, plan_kwargs={"lr": 1e-5}, **train_kwargs),
            model_name="scanvi",
            num_epochs=max_epochs,
            dataset_name=dataset_name,
            dataset_size=adata.n_obs,
            dataset_path=dataset_path,
            csv_path=PROFILER_CSV_PATH
        )
    else:
        scanvi_model.train(datasplitter_kwargs=data_splitter_kwargs, batch_size=batch_size, **train_kwargs)
    # NOTE: Required for scVIVA!!!
    adata.obsm["SCANVI"] = scanvi_model.get_latent_representation(batch_size=batch_size)
    adata.write(output_path)

def run_scviva_model(
    adata,
    batch_key,
    sample_key,
    celltype_key,
    niche_key,
    output_path,
    data_splitter_kwargs,
    n_layers=2,
    n_latent=30,
    max_epochs=None,
    K_NN=20,
    model_kwargs={},
    train_kwargs={},
    batch_size=256,
    profiler=True,
    dataset_name="",
    dataset_path=""
    ):
    setup_kwargs = {
        "sample_key": sample_key,
        "labels_key": celltype_key,
        "cell_coordinates_key": "spatial",
        # NOTE: according to their results, SCANVI embs worked better than scvi
        "expression_embedding_key": "SCANVI",
    }
    scvi.external.SCVIVA.preprocessing_anndata(
        adata,
        k_nn=K_NN,
        **setup_kwargs,
    )
    scvi.external.SCVIVA.setup_anndata(
        adata,
        layer="counts",
        batch_key=batch_key,
        **setup_kwargs,
    )
    nichevae = scvi.external.SCVIVA(adata, n_latent=n_latent, **model_kwargs)

    if max_epochs is not None:
        train_kwargs.setdefault("max_epochs", max_epochs)

    if profiler:
        profile_training(
            lambda: nichevae.train(datasplitter_kwargs=data_splitter_kwargs, batch_size=batch_size, plan_kwargs={"lr": 1e-5}, **train_kwargs),
            model_name="scviva",
            num_epochs=max_epochs,
            dataset_name=dataset_name,
            dataset_size=adata.n_obs,
            dataset_path=dataset_path,
            csv_path=PROFILER_CSV_PATH
        )
    else:
        nichevae.train(datasplitter_kwargs=data_splitter_kwargs, batch_size=batch_size, **train_kwargs)

    adata.obsm["scVIVA"] = nichevae.get_latent_representation()
    adata.write(output_path)

def run_simvi_in_subprocess(
    adata,
    batch_key,
    sample_key,
    celltype_key,
    niche_key,
    output_path,
    conda_env_name='simvi',
    n_layers=2,
    n_latent=30,
    max_epochs=None,
    K_NN=20,
    model_kwargs=None,
    train_kwargs=None,
    batch_size=256,
    data_splitter_kwargs=None,
    profiler=False,
    dataset_name="",
    dataset_path=""
):
    """
    Runs the simVI model in a separate conda environment using subprocess.
    """
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}

    cmd = [
        "conda", "run", "-n", conda_env_name,
        "python", "call_simvi_model.py", # Make sure this script is in your path
        "--adata_path", output_path,
        "--output_path", output_path,
        "--batch_key", batch_key,
        "--celltype_key", celltype_key,
        "--niche_key", niche_key,
        "--n_layers", str(n_layers),
        "--n_latent", str(n_latent),
        "--K_NN", str(K_NN),
        "--batch_size", str(batch_size),
        # 3. Serialize dictionary arguments to JSON strings
        "--model_kwargs", json.dumps(model_kwargs),
        "--train_kwargs", json.dumps(train_kwargs),
    ]

    # Conditionally add max_epochs if it is provided
    if max_epochs is not None:
        cmd.extend(["--max_epochs", str(max_epochs)])

    print(f"\nRunning command:\n{' '.join(cmd)}\n")

    # 4. Execute the command
    # Using capture_output=True and text=True to see the print statements from the script
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    print("Subprocess STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("Subprocess STDERR:")
        print(result.stderr)
            
    print(f"\nSubprocess finished successfully. Final output should be at {output_path}")



### PLOT -> NOTE to move    
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from plottable import Table, ColumnDefinition
from plottable.plots import bar
from plottable.cmap import normed_cmap

_METRIC_TYPE = "Metric Type"
_AGGREGATE_SCORE = "Aggregate score"

def plot_results_table(
    df,
    min_max_scale: bool = True,
    show: bool = True,
    save_dir: str | None = None,
    metric_cmap=mpl.cm.PRGn,
    score_cmap=mpl.cm.YlGnBu,
    sort_col='Bio conservation'
) -> Table:
    """Plot the benchmarking results.

    Parameters
    ----------
    min_max_scale
        Whether to min max scale the results.
    show
        Whether to show the plot.
    save_dir
        The directory to save the plot to. If `None`, the plot is not saved.
    metric_cmap
        The colormap for the individual metric scores (circles).
    score_cmap
        The colormap for the aggregate scores (bars).
    """
    num_embeds = df.shape[0] - 1
    # Use the new metric_cmap parameter for the circles
    cmap_fn = lambda col_data: normed_cmap(col_data, cmap=metric_cmap, num_stds=2.5)
    # TODO: implement min_max scaling
    # df = self.get_results(min_max_scale=min_max_scale)
    # Do not want to plot what kind of metric it is
    plot_df = df.drop(_METRIC_TYPE, axis=0)
    plot_df = plot_df.sort_values(by=sort_col, ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    # Split columns by metric type, using df as it doesn't have the new method col
    score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
    other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]
    column_definitions = [
        ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
    ]
    # Circles for the metric values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=cmap_fn(plot_df[col]),
            group=df.loc[_METRIC_TYPE, col],
            formatter="{:.2f}",
        )
        for i, col in enumerate(other_cols)
    ]
    # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                # Use the new score_cmap parameter for the bars
                "cmap": score_cmap,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=df.loc[_METRIC_TYPE, col],
            border="left" if i == 0 else None,
        )
        for i, col in enumerate(score_cols)
    ]
    # Allow to manipulate text post-hoc (in illustrator)
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)
    if show:
        plt.show()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "scib_results.svg"), facecolor=ax.get_facecolor(), dpi=300)

    return tab


## GET Stuff
import matplotlib.cm as cm
import matplotlib.colors as mcolors
n_colors = 20
blues = [cm.Blues(x) for x in np.linspace(0.01, 1, n_colors)]
blue_cmap = mcolors.LinearSegmentedColormap.from_list("light_to_dark_blue", blues)

# Define colors
bluish_gray = "#a9b7c9"   # desaturated/light bluish-gray
blue = "#2166ac"          # mid to dark saturated blue (from 'Blues')

# Create the custom colormap
bluishgray_to_blue = mcolors.LinearSegmentedColormap.from_list(
    "bluish_gray_to_blue",
    [bluish_gray, blue]
)

n_colors = 20
reds = [cm.Reds(x) for x in np.linspace(0.1, 1.0, n_colors)]
red_cmap = mcolors.LinearSegmentedColormap.from_list("light_to_dark_red", reds)

# Define the endpoints in RGB
gray = "#d3d3d3"           # light gray
grayish_red = "#b22222"    # firebrick (dark muted red)

# Create a colormap from gray to grayish red
gray_to_red = mcolors.LinearSegmentedColormap.from_list(
    "gray_to_grayish_red",
    [gray, grayish_red]
    )


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

def plot_metric_by_split(df, metric, title=None, ylabel=None, palette=None, figsize=(10, 6)):
    """
    Bar plot for a single metric (e.g., f1_macro), across splits, colored by model.

    Args:
        df (pd.DataFrame): Columns must include 'model', 'split', '{metric}_mean', '{metric}_std'.
        metric (str): Base metric name (e.g. 'f1_macro', 'accuracy').
        title (str): Plot title.
        ylabel (str): Y-axis label.
        palette (Colormap or list of colors): Optional colormap or list of colors for models.
        figsize (tuple): Figure size.
    """
    models = df["model"].unique()
    splits = df["split"].unique()
    n_models = len(models)
    n_splits = len(splits)

    # Get mean and std column names
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # Resolve model colors
    if isinstance(palette, mcolors.Colormap):
        model_colors = [palette(i / max(n_models - 1, 1)) for i in range(n_models)]
    else:
        model_colors = sns.color_palette(palette, n_models) if palette else sns.color_palette("tab10", n_models)

    # Bar plot setup
    total_width = 0.8
    bar_width = total_width / n_models
    gap = 0.02
    x_positions = np.arange(n_splits)

    plt.figure(figsize=figsize)

    for i, model in enumerate(models):
        means = []
        stds = []
        for split in splits:
            row = df[(df["model"] == model) & (df["split"] == split)]
            if row.empty:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(row[mean_col].values[0])
                stds.append(row[std_col].values[0])

        # Bar positions: shift each model within each split group
        x = x_positions - total_width / 2 + i * bar_width + bar_width / 2

        plt.bar(
            x,
            means,
            yerr=stds,
            width=bar_width - gap,
            color=model_colors[i],
            edgecolor='black',
            alpha=0.9,
            capsize=5,
            label=model,
        )

    plt.xticks(x_positions, splits)
    plt.ylim(0, 1)
    plt.ylabel(ylabel or metric.replace("_", " ").title())
    plt.title(title or f"{metric.replace('_', ' ').title()} (± std)")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()