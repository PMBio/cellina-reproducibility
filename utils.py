import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scanpy as sc

from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from cellina import CellinaModel

def set_seed(seed):
    random.seed(seed)                  # Python random
    np.random.seed(seed)               # NumPy
    torch.manual_seed(seed)            # PyTorch CPU
    torch.cuda.manual_seed(seed)       # PyTorch GPU (single-GPU)
    torch.cuda.manual_seed_all(seed)   # PyTorch GPU (multi-GPU)
    
    # Ensures deterministic behavior for some operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)



def plot_results(df, alpha_type, target_col):
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x="alpha", y="score", width=0.5)
    sns.stripplot(data=df, x="alpha", y="score", color="black", size=5, jitter=True)
    plt.title(f"F1-score across seeds ({alpha_type}, target={target_col})")
    plt.xlabel(f"{alpha_type} value")
    plt.ylabel("F1-score")
    plt.show()


def evaluate_models(
    adata,
    seeds,
    alpha_type,
    alpha_values,
    target_col,
    root_dir="trained",
):
    """Evaluate trained models via linear classifier on latent space."""
    results = {alpha: [] for alpha in alpha_values}

    for alpha in tqdm(alpha_values, desc=f"Evaluating ({alpha_type})"):
        for seed in seeds:
            save_path = f"{root_dir}/{alpha_type}_{alpha}_seed_{seed}"

            model = CellinaModel.load(save_path, adata)

            adata.obsm['z'] = model.get_latent_representation(latent_key='z')

            X_train = adata[~adata.obs["is_holdout"]].obsm['z']
            y_train = adata[~adata.obs["is_holdout"]].obs[target_col].values

            X_test = adata[adata.obs["is_holdout"]].obsm['z']
            y_test = adata[adata.obs["is_holdout"]].obs[target_col].values

            clf = LogisticRegression(max_iter=500, solver="lbfgs")
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            score = f1_score(y_test, y_pred, average="macro")
            results[alpha].append(score)

    # convert to tidy DataFrame
    df = pd.DataFrame([
        {"alpha": alpha, "score": score}
        for alpha, scores in results.items()
        for score in scores
    ])

    return df