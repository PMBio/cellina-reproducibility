import torch
# Concert (CONCERT) default configuration adapted from notebooks/concert.ipynb
# These defaults are intended to be imported by scripts/train_loo.py

# Model constructor kwargs — values here are sensible defaults and may be overridden
# by passing a dict via the CLI or other orchestration.
MODEL_ARGS = {
    # Architecture sizes
    "encoder_dim": 256,
    "encoder_layers": [128, 64],
    "decoder_layers": [128],
    "GP_dim": 2,
    "Normal_dim": 8,

    # GP / inducing point behaviour
    "fixed_inducing_points": True,
    # initial_inducing_points should be provided at runtime if desired (array)
    # kernel scale may be scalar or array shaped (n_kernels, loc_dim)
    "multi_kernel_mode": True,
    "fixed_gp_params": False,

    "noise": 0.25,
    "encoder_dropout": 0,
    "decoder_dropout": 0,
    "KL_loss": 0.025,
    "init_beta": 10,
    "min_beta": 5,
    "max_beta": 25,
    "dynamicVAE": True,

    # regularization / VAE params
    "noise": 0.25,
    "shared_dispersion": False,

    # device selection left to runtime (None -> default)
    'dtype': torch.float32,
    "device": None,
}

# Training-time kwargs — passed to the model.train_model(...) call
TRAIN_ARGS = {
    "maxiter": 2, #50,
    "patience": 1, #25,
    "batch_size": 512,
    "lr": 1e-4,
    "weight_decay": 1e-6,
    "num_samples": 1,
    "train_size": 0.95,
    # save_model and model filename handled by train_loo; here default False
    "save_model": False,
}

# Planner / optimizer kwargs (if the training wrapper accepts them)
PLAN_KWARGS = {
    "lr": TRAIN_ARGS.get("lr", 1e-4),
}

# Whether to run counterfactual generation by default for this model
DO_COUNTERFACTUAL = True
