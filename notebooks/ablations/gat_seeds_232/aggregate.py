"""Per-seed table + mean/std across seeds and cell types. python aggregate.py"""
import glob, json, os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CTS = ['Epithelial', 'Fibroblast', 'Myeloid']
rows = [r for r in (json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, 'results', '*_seed*.json')))) if 'pearson' in r]
df = pd.DataFrame(rows)
df['seed'] = df.seed.astype(str) + df.get('tag', pd.Series('', index=df.index)).fillna('')
if df.empty:
    raise SystemExit('no results yet')
df.to_csv(os.path.join(HERE, 'summary.csv'), index=False)
def _ref(csv, label):  # committed GAT rows (results branch csv), crc_232, seed 0
    r = pd.read_csv(os.path.join(HERE, '..', 'gat_hops_232', csv))
    r = r[r.model_name == 'cellina-graph'].assign(ct=lambda d: d.holdout_celltype.str.replace('_CRC', '', regex=False))
    return r[r.ct.isin(CTS)].assign(seed=label)
ref = pd.concat([_ref('reference_sid232.csv', 'PAPER: cellina-graph, loo_summary_crc_DEG_50_v4.csv (hop-fixed, bidir CF)'),
                 _ref('reference_sid232_apr23.csv', 'old GAT: loo_summary_crc_DEG_50.csv @5e3df54 Apr-23 (cellina_graph 0.0.3)')])
old = pd.read_csv(os.path.join(HERE, '..', 'gat_hops_232', 'final_table.csv')).set_index('config').loc['kk_k20_h1']
print(f'== {len(df)}/12 runs; cellina {sorted(df.cellina_version.unique())} @ {sorted(df.cellina_branch.unique())} ==')
for metric, label in [('pearson', 'Pearson'), ('precision', 'Precision')]:
    piv = pd.concat([df, ref])[['seed', 'ct', metric]].pivot(index='seed', columns='ct', values=metric).reindex(columns=CTS)
    piv.loc['grid kk_k20_h1 seed0 (ablation branch)'] = [old[f'{label} {c[:3]}'] for c in CTS]
    piv['avg'] = piv[CTS].mean(axis=1, skipna=False)
    print(f'\n-- {label} --'); print(piv.round(3).to_markdown())
    ours = df[metric]
    print(f'{label}: mean over all seeds x cts = {ours.mean():.3f} ± {ours.std(ddof=1):.3f} (n={len(ours)}); '
          f'per-seed avg: ' + ', '.join(f'{s}: {v:.3f}' for s, v in df.groupby("seed")[metric].mean().items()))
