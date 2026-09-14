"""Is the 0.84 TERRA, or is it a prior that needs no model at all?
Baselines that use progressively less of TERRA."""
import nbformat, numpy as np, anndata as ad, scipy.sparse as sp, warnings
warnings.filterwarnings("ignore")
nb = nbformat.read("terra_node_pert.ipynb", as_version=4)
g = {"__name__": "__main__"}
for i, c in enumerate(nb.cells):
    if c.cell_type != "code": continue
    if i > 33: break
    src = "\n".join(l for l in c.source.splitlines() if not l.strip().startswith(("%", "!")))
    exec(compile(src, f"cell{i}", "exec"), g)

np, pearsonr, spearmanr = g["np"], g["pearsonr"], g["spearmanr"]
get_lfc, precision, direction_match = g["get_lfc"], g["precision"], g["direction_match"]
control, target, N_DEG = g["control"], g["target"], g["N_DEG"]
true_lfc, deg = g["true_lfc"], g["deg"]
gene_list, decode = g["gene_list"], g["decode"]

def score(name, cf):
    _, v, _ = get_lfc(control=control, target=target, counterfactual=cf, n_deg=N_DEG)
    r, _ = pearsonr(true_lfc[deg], v[deg]); rho, _ = spearmanr(true_lfc[deg], v[deg])
    print(f"{name:46s} pearson={r:+.4f}  spearman={rho:+.4f}  "
          f"dir={direction_match(true_lfc, v, k=N_DEG):.3f}")
    return r

n_cells = control.shape[0]
print("\n=============== IS THE 0.84 TERRA? ===============")
score("D  perturbed  (the reported 0.839)", g["decoded_pert"])
score("C  unperturbed TERRA embeddings", g["decoded_base"])

# B: decoder fed TERRA embeddings of 1033 RANDOM cells (any type, any region)
rng = np.random.default_rng(0)
allemb = g["base_emb"]["spatial_cell_emb"]
rnd = allemb[rng.choice(allemb.shape[0], n_cells, replace=False)]
sub_ctrl = g["sub"][g["is_control"]].copy()
sub_ctrl.obsm["spatial_cell_emb"] = rnd.astype(np.float32)
b = g["apply_count_decoder"](sub_ctrl, emb_key="spatial_cell_emb", model_folder_path=None,
        checkpoint_path=str(g["CKPT"]), decoded_counts_layer_key="decoded",
        embed_fallback_key="spatial_cell_emb", device=0)
score("B  decoder on RANDOM cells' embeddings", np.asarray(b.layers["decoded"]))

# A: no TERRA whatsoever -- the decoder training set's mean expression profile
tr = ad.read_h5ad(g["TRAIN_H5AD"])[:, gene_list]
X = tr.layers["counts"]; X = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
score("A  train-set MEAN profile (no TERRA at all)", np.tile(X.mean(0), (n_cells, 1)))

# A': even cruder -- the observed control's own mean, i.e. predict "no change"
score("A' control's own mean (predict no change)", np.tile(control.mean(0), (n_cells, 1)))
print("==================================================")
