# Cellina default configuration taken from notebooks/conditional_z_mll.ipynb
# These defaults are intended to be imported by scripts/train_loo.py

MODEL_ARGS = {
    # Do not include adata here; train_loo will pass the AnnData when constructing the model
    "n_latent": 64,
    "use_observed_lib_size": True,
    "classifier_lambda": 1.0,
    "discriminator_lambda": 1.0,
    "link_prediction_weight": 1.0,
    "n_layers": 2,
    "convolution_type": 'gat',
    "gene_likelihood": 'nb',
}

# Train args mirror the notebook settings. Some keys (like datasplitter external_indexing)
# will be populated at runtime by train_loo if needed.
TRAIN_ARGS = {
    "max_epochs": 100,
    "batch_size": 512,
    "check_val_every_n_epoch": 1,
    "early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_monitor": "vae_loss_validation",
    "enable_checkpointing": True,
    "devices": [0],  # devices left as default; the user or environment should override if needed
}

# Additional plan kwargs sometimes passed to model.train; include a reasonable default
PLAN_KWARGS = {
    "lr": 1e-3,
    'weight_decay': 0.0001,
    "normalize_losses": True,
}

# Enable counterfactual behaviour by default for Cellina
DO_COUNTERFACTUAL = True
N_NEIGHBORS_PER_SEED = 20  # number of neighbors to use when sampling for counterfactual inference in cellina-graph (matches notebooks)
N_NEIGHBORS_GRAPH = 20 # number of neighbors to compute adjacency matrix

# ---------------------------------------------------------------------------
# Environment-variable overrides (GAT hops / batch-size sensitivity test).
#
# All of these default to the values defined above, so importing this module
# without any of the variables set reproduces the committed configuration
# exactly. They exist so that a batch job can vary the GAT receptive field and
# batch size without editing this file.
#
#   GAT_N_LAYERS            int            -> MODEL_ARGS["n_layers"]
#   GAT_NUM_NEIGHBORS       "-1,0,0"       -> MODEL_ARGS["num_neighbors"]
#                                             (key is only added when set, so
#                                              the package default applies when
#                                              it is unset)
#   GAT_BATCH_SIZE          int            -> TRAIN_ARGS["batch_size"]
#   GAT_N_NEIGHBORS_GRAPH   int            -> N_NEIGHBORS_GRAPH
#   GAT_N_NEIGHBORS_PER_SEED int           -> N_NEIGHBORS_PER_SEED
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys

_GAT_OVERRIDES = {}


def _gat_env_int(name, default):
    raw = _os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw.strip())
    _GAT_OVERRIDES[name] = value
    return value


def _gat_env_int_list(name):
    """Parse a comma-separated int list; returns None when unset."""
    raw = _os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    value = [int(tok) for tok in raw.replace(" ", "").split(",") if tok != ""]
    if not value:
        return None
    _GAT_OVERRIDES[name] = value
    return value


MODEL_ARGS["n_layers"] = _gat_env_int("GAT_N_LAYERS", MODEL_ARGS["n_layers"])

_gat_num_neighbors = _gat_env_int_list("GAT_NUM_NEIGHBORS")
if _gat_num_neighbors is not None:
    # Only set the key when explicitly requested; otherwise the cellina package
    # default ([-20] * n_layers, i.e. all neighbours at every hop) applies.
    MODEL_ARGS["num_neighbors"] = _gat_num_neighbors

TRAIN_ARGS["batch_size"] = _gat_env_int("GAT_BATCH_SIZE", TRAIN_ARGS["batch_size"])
N_NEIGHBORS_GRAPH = _gat_env_int("GAT_N_NEIGHBORS_GRAPH", N_NEIGHBORS_GRAPH)
N_NEIGHBORS_PER_SEED = _gat_env_int("GAT_N_NEIGHBORS_PER_SEED", N_NEIGHBORS_PER_SEED)

if _GAT_OVERRIDES:
    print(
        "[cellina_graph_config] env overrides active: "
        f"n_layers={MODEL_ARGS['n_layers']} "
        f"num_neighbors={MODEL_ARGS.get('num_neighbors', '<package default>')} "
        f"batch_size={TRAIN_ARGS['batch_size']} "
        f"N_NEIGHBORS_GRAPH={N_NEIGHBORS_GRAPH} "
        f"N_NEIGHBORS_PER_SEED={N_NEIGHBORS_PER_SEED} "
        f"(from {sorted(_GAT_OVERRIDES)})",
        file=_sys.stderr,
        flush=True,
    )
