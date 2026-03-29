# scgen default configuration
# These defaults are intended to be imported by scripts/train_loo.py

MODEL_ARGS = {
    # Do not include adata here; train_loo will pass the AnnData when constructing the model
    "n_latent": 64,
}

# Train args mirror the notebook settings. Some keys (like datasplitter external_indexing)
# will be populated at runtime by train_loo if needed.
TRAIN_ARGS = {
    "max_epochs": 100,
    "batch_size": 2048, #4096,
    "early_stopping": True,
    "early_stopping_patience": 25,
    "devices": [1],
}

PLAN_KWARGS = {}

# Enable counterfactual behaviour by default for Cellina
DO_COUNTERFACTUAL = True