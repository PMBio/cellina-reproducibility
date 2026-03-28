import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import shuffle




def set_seed(seed):
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (single-GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (multi-GPU)

    # Ensures deterministic behavior for some operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def plot_results(df, lambda_type, target_col):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="lambda", y="score", width=0.5)
    sns.stripplot(data=df, x="lambda", y="score", color="black", size=5, jitter=True)
    plt.title(f"F1-score across seeds ({lambda_type}, target={target_col})")
    plt.xlabel(f"{lambda_type} value")
    plt.ylabel("F1-score")
    plt.show()


def evaluate_models(
    adata,
    seeds,
    lambda_type,
    lambda_values,
    target_col,
    latent_key="z",
    root_dir="trained",
    batch_size=1024,
):
    """Evaluate trained models via linear classifier on latent space."""
    from cellina import CellinaModel
    
    results = {lambda_: [] for lambda_ in lambda_values}

    for lambda_ in tqdm(lambda_values, desc=f"Evaluating ({lambda_type})"):
        for seed in seeds:
            save_path = f"{root_dir}/{lambda_type}_{lambda_}_seed_{seed}"

            model = CellinaModel.load(save_path, adata)

            adata.obsm[latent_key] = model.get_latent_representation(
                latent_key=latent_key, batch_size=batch_size
            )

            X_train = adata[~adata.obs["is_holdout"]].obsm[latent_key]
            y_train = adata[~adata.obs["is_holdout"]].obs[target_col].values

            X_test = adata[adata.obs["is_holdout"]].obsm[latent_key]
            y_test = adata[adata.obs["is_holdout"]].obs[target_col].values

            clf = LogisticRegression(max_iter=500, solver="lbfgs")
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            score = f1_score(y_test, y_pred, average="macro")
            results[lambda_].append(score)

    # convert to tidy DataFrame
    df = pd.DataFrame(
        [
            {"lambda": lambda_, "score": score}
            for lambda_, scores in results.items()
            for score in scores
        ]
    )

    return df


def train_linear(X, y, cv_folds=5, n_perm=0):
    # -------------------------------
    # Fit multinomial logistic regression & cross-validated accuracy
    # -------------------------------
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"Cross-validated accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    clf.fit(X, y)

    # -------------------------------
    # Permutation test for significance
    # -------------------------------

    if n_perm > 0:
        perm_scores = []
        for _ in range(n_perm):
            y_perm = shuffle(y, random_state=None)
            perm_score = cross_val_score(
                clf, X, y_perm, cv=cv, scoring="accuracy"
            ).mean()
            perm_scores.append(perm_score)

        perm_scores = np.array(perm_scores)
        p_val = (np.sum(perm_scores >= scores.mean()) + 1) / (n_perm + 1)
        print(f"Permutation test p-value: {p_val:.3f}")

    return clf
