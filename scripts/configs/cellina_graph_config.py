# Cellina default configuration taken from notebooks/conditional_z_mll.ipynb
# These defaults are intended to be imported by scripts/train_loo.py

MODEL_ARGS = {
    # Do not include adata here; train_loo will pass the AnnData when constructing the model
    "n_latent": 64,
    "use_observed_lib_size": True,
    "classifier_lambda": 1.0,
    "discriminator_lambda": 1.0,
    "link_prediction_weight": 1.0
}

# Train args mirror the notebook settings. Some keys (like datasplitter external_indexing)
# will be populated at runtime by train_loo if needed.
TRAIN_ARGS = {
    "max_epochs": 2,  #100,
    "batch_size": 512,
    "check_val_every_n_epoch": 1,
    "early_stopping": True,
    "early_stopping_patience": 25,
    "early_stopping_monitor": "validation_loss",
    "enable_checkpointing": True,
    "devices": [1],  # devices left as default; the user or environment should override if needed
}

# Additional plan kwargs sometimes passed to model.train; include a reasonable default
PLAN_KWARGS = {
    "lr": 1e-4,
    "normalize_losses": True,
}

# Enable counterfactual behaviour by default for Cellina
DO_COUNTERFACTUAL = True
