"""One LOO run of Cellina-GAT (cellina 1.1.1, directed CF edges) on crc_232 for one seed / holdout.

Mirrors cellina-reproducibility@main scripts/multi_seed/{train_loo,eval_loo}.py: same preprocessing,
split, training (train_model) and counterfactual inference (run_inference) code, imported from a
detached worktree of origin/main. Only differences vs the pipeline:
  * `condition_on_intrinsic` dropped from MODEL_ARGS (removed from cellina >= 1.1; the pipeline
    config still carries it and would crash on this cellina version).
  * `num_neighbors` set explicitly (default [20,20] = grid's kk_k20_h1; pipeline default None -> [-1,-1]).
  * Metrics computed in-process from the saved counterfactual h5ad instead of a second eval_loo call.
python run_seed.py --seed 0 --ct Epithelial --gpu 0
"""
import argparse, json, os, subprocess, sys, time, warnings
import numpy as np, scanpy as sc
from scipy.stats import pearsonr, spearmanr

WT = '/data/ddimitrov/repos/cellina-reproducibility-main'          # git worktree, detached at origin/main
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(WT)                                                        # pipeline scripts do sys.path.append('./scripts')
sys.path.insert(0, os.path.join(WT, 'scripts')); sys.path.insert(0, os.path.join(WT, 'scripts', 'multi_seed'))
import train_loo as TL                                              # scripts/multi_seed/train_loo.py
from eval_loo import load_model_predicted, N_DEG
from counterfactual_analysis import (get_lfc, precision, direction_match, mixing_index, compute_edistance,
                                     compute_rmse, compute_mse_lfc)
from configs.adata_crc_config import ADATA_ARGS
from configs.cellina_graph_config import MODEL_ARGS, TRAIN_ARGS, PLAN_KWARGS, N_NEIGHBORS_GRAPH
import cellina
warnings.filterwarnings('ignore')

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, required=True)
p.add_argument('--ct', required=True)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--num_neighbors', default='20,20')
p.add_argument('--max_epochs', type=int, default=None, help='smoke-test override; None = pipeline TRAIN_ARGS')
p.add_argument('--tag', default='')
p.add_argument('--batch_size', type=int, default=None, help='train (and CF) batch size; None = pipeline TRAIN_ARGS (256)')
p.add_argument('--expect_version', default='1.1.1')
p.add_argument('--expect_branch', default='release/v1.1.1')
a = p.parse_args()

# ---- guard: correct cellina ----
git = lambda *c: subprocess.check_output(['git', '-C', '/data/ddimitrov/repos/cellina', *c], text=True).strip()
branch, commit = git('branch', '--show-current'), git('rev-parse', '--short', 'HEAD')
assert cellina.__version__ == a.expect_version, (cellina.__version__, a.expect_version)
assert branch == a.expect_branch, (branch, a.expect_branch)
assert cellina.__file__.startswith('/data/ddimitrov/repos/cellina/'), cellina.__file__
print(f'cellina {cellina.__version__} @ {branch} {commit} from {cellina.__file__}', flush=True)

model_name = f'cellina-graph_{a.seed}'                              # pipeline: f'{model_name}_{seed}'
name = f'{a.ct}_seed{a.seed}{a.tag}'
adata_path = os.path.join(HERE, 'data', 'crc_232.h5ad')              # symlink -> run_inference writes to HERE/crc_232/<ct>/
out_json = os.path.join(HERE, 'results', f'{name}.json')
print(f'### {name}', flush=True)

# ---- train_loo.main() for dataset crc / model_class cellina_graph ----
TL.set_seed(a.seed)
adata = sc.read(adata_path)
labels_key, domains_key, batch_key = ADATA_ARGS['labels_key'], ADATA_ARGS['domains_key'], ADATA_ARGS['batch_key']
control_domains, holdout_domains = ADATA_ARGS['control_domains'], ADATA_ARGS['holdout_domains']
adata = TL.preprocess_crc(adata, n_top_genes=ADATA_ARGS['n_top_genes'], labels_key=labels_key, domains_key=domains_key)
splits = TL.split_indices(adata, a.ct, labels_key=labels_key, domains_key=domains_key,
                          holdout_domains=holdout_domains, seed=a.seed)
print(f'n_obs={adata.n_obs} train={len(splits[0])} val={len(splits[1])} test={len(splits[2])}', flush=True)
adata = TL.preprocess_spatial_features(adata, step_size_px=0.12028, n_neighbors=N_NEIGHBORS_GRAPH, test_indices=splits[2])

model_args = {k: v for k, v in MODEL_ARGS.items() if k != 'condition_on_intrinsic'}
model_args['num_neighbors'] = [int(x) for x in a.num_neighbors.split(',')]
train_args = {**TRAIN_ARGS, 'devices': [a.gpu], **({'max_epochs': a.max_epochs} if a.max_epochs else {}),
              **({'batch_size': a.batch_size} if a.batch_size else {})}
plan_kwargs = PLAN_KWARGS.copy()
save_dir = os.path.join(HERE, 'results', 'models', a.ct, model_name)
os.makedirs(save_dir, exist_ok=True)
t0 = time.time()
model, extras = TL.train_model(adata, 'cellina_graph', model_args, train_args, save_dir, plan_kwargs=plan_kwargs,
                               batch_key=batch_key, labels_key=labels_key, domains_key=domains_key, splits=splits)
train_min = (time.time() - t0) / 60
hist = model.history['vae_loss_validation']['vae_loss_validation']
print(f'trained {len(hist)} epochs in {train_min:.1f} min, best val {hist.min():.2f}', flush=True)

recon_path, cf_path = TL.run_inference(model, adata, adata_path, 'cellina_graph', model_name, a.ct, do_cf=True,
                                       batch_size=train_args['batch_size'], labels_key=labels_key, domains_key=domains_key,
                                       return_normalized=False, extras=extras, control_domains=control_domains,
                                       holdout_domains=holdout_domains, seed=a.seed)

# ---- eval_loo.main() metric block (use_cf=True, use_recon=False) ----
adata_full = adata.copy()
adata = adata[adata.obs[labels_key].astype(str) == a.ct]
is_holdout_ct = adata.obs[labels_key].astype(str) == a.ct
mask_control = is_holdout_ct & adata.obs[domains_key].isin(control_domains)
control = TL._to_array(adata.layers['counts'][mask_control.values, :])
hd = holdout_domains[0]
counterfactual, _ = load_model_predicted(cf_path, 'cellina_graph')
mask_target = is_holdout_ct & (adata.obs[domains_key] == hd)
target = TL._to_array(adata.layers['counts'][mask_target.values, :])
gt_lfc, cf_lfc, deg = get_lfc(control=control, target=target, counterfactual=counterfactual, n_deg=N_DEG)
C = TL.COUNTS_PER_K
stats = {
    'n_deg': N_DEG,
    'spearman': spearmanr(gt_lfc[deg], cf_lfc[deg])[0],
    'pearson': pearsonr(gt_lfc[deg], cf_lfc[deg])[0],
    'precision': precision(gt_lfc, cf_lfc, k=N_DEG, use_abs=True),
    'direction_match': direction_match(gt_lfc, cf_lfc, k=N_DEG, normalize='intersection'),
    'direction_match_k': direction_match(gt_lfc, cf_lfc, k=N_DEG, normalize='k'),
    'direction_match_gt': direction_match(gt_lfc, cf_lfc, k=N_DEG, normalize='gt_topk'),
    'mixing_index': mixing_index(observed=target, predicted=counterfactual, library_size=C),
    'edistance_global': compute_edistance(adata_full, observed=target, predicted=counterfactual, deg=None, library_size=C),
    'edistance_local': compute_edistance(adata_full, observed=target, predicted=counterfactual, deg=None, library_size=C, local=True),
    'edistance_pca_log': compute_edistance(adata_full, observed=target, predicted=counterfactual, deg=None, library_size=C, local=True, use_pca=True),
    'edistance_pca': compute_edistance(adata_full, observed=target, predicted=counterfactual, deg=None, library_size=C, local=True, use_pca=True, log1p=False),
    'rmse': compute_rmse(observed=target, predicted=counterfactual, deg=deg, library_size=C),
    'mse_lfc': compute_mse_lfc(gt_vec=gt_lfc, cf_vec=cf_lfc, deg=deg),
}
stats = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in stats.items()}
stats.update(seed=a.seed, tag=a.tag, batch_size=train_args['batch_size'], ct=a.ct, model_name=model_name, num_neighbors=model_args['num_neighbors'],
             epochs=int(len(hist)), best_val_loss=float(hist.min()), train_min=round(train_min, 1),
             n_control=int(mask_control.sum()), n_target=int(mask_target.sum()),
             cellina_version=cellina.__version__, cellina_branch=branch, cellina_commit=commit,
             repro_commit=subprocess.check_output(['git', '-C', WT, 'rev-parse', '--short', 'HEAD'], text=True).strip(),
             cf_path=cf_path)
with open(out_json, 'w') as fh:
    json.dump(stats, fh, indent=2)
print(f"pearson={stats['pearson']:.3f} precision={stats['precision']:.3f} direction_match_k={stats['direction_match_k']:.3f} -> {out_json}", flush=True)
