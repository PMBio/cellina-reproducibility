"""Re-run the notebook tail from cache and ask the one question it didn't:
does the *perturbation* explain the 0.839, or does the decoder's reconstruction bias?"""
import nbformat, numpy as np, warnings
warnings.filterwarnings("ignore")
nb = nbformat.read("terra_node_pert.ipynb", as_version=4)
g = {"__name__": "__main__"}
for i, c in enumerate(nb.cells):
    src = "\n".join(l for l in c.source.splitlines() if not l.strip().startswith(("%","!")))
    if c.cell_type != "code":
        continue
    if i > 33:
        break
    exec(compile(src, f"cell{i}", "exec"), g)

np, pearsonr, spearmanr = g["np"], g["pearsonr"], g["spearmanr"]
get_lfc, precision, direction_match = g["get_lfc"], g["precision"], g["direction_match"]
control, target = g["control"], g["target"]
decoded_base, decoded_pert = g["decoded_base"], g["decoded_pert"]
true_lfc, pred_lfc, deg = g["true_lfc"], g["pred_lfc"], g["deg"]
N_DEG = g["N_DEG"]

_, null_lfc, _ = get_lfc(control=control, target=target, counterfactual=decoded_base, n_deg=N_DEG)

def row(name, v):
    r, _ = pearsonr(true_lfc[deg], v[deg])
    rho, _ = spearmanr(true_lfc[deg], v[deg])
    print(f"{name:34s} pearson={r:+.4f}  spearman={rho:+.4f}  "
          f"prec@20={precision(true_lfc, v, k=20):.3f}  dir={direction_match(true_lfc, v, k=N_DEG):.3f}")

print("\n================ NULL DIAGNOSTIC ================")
row("perturbed   (reported result)", pred_lfc)
row("UNperturbed (null / bias only)", null_lfc)

delta = pred_lfc - null_lfc          # what the perturbation actually contributed
print(f"\nperturbation's own contribution to predicted logFC:")
print(f"  mean|pred - null| over top-{N_DEG} DE = {np.abs(delta[deg]).mean():.5f}")
print(f"  mean|pred|                          = {np.abs(pred_lfc[deg]).mean():.5f}")
print(f"  -> perturbation moves {100*np.abs(delta[deg]).mean()/np.abs(pred_lfc[deg]).mean():.2f}% of the signal")
r_delta, _ = pearsonr(true_lfc[deg], delta[deg])
print(f"  pearson(true_lfc, pert-only delta)  = {r_delta:+.4f}   <- is the *movement* in the right direction?")

# shared-denominator artifact: how much of the 0.839 is just -log2(mean_control)?
mc = np.nanmean(g["_normalize_counts"](control) if hasattr(g, "_normalize_counts") else control, axis=0)
with np.errstate(divide="ignore", invalid="ignore"):
    neg_log_ctrl = -np.log2(mc + 1e-9)
ok = np.isfinite(neg_log_ctrl[deg]) & np.isfinite(true_lfc[deg])
r_art, _ = pearsonr(true_lfc[deg][ok], neg_log_ctrl[deg][ok])
print(f"\npearson(true_lfc, -log2(mean_control)) = {r_art:+.4f}   <- shared-denominator artifact size")
print("=================================================")
