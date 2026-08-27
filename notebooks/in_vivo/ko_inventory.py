"""Inventory every candidate KO: # perturbed cancer cells, # exclusive near-T ground-truth cells
(default graph), and confound-matched DIRECT effect on cancer (non-circular, selects on input).
Used to decide the evaluation KO set for the publication figures."""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests
import pfish_prep as pp
OUT="./cellina_pfish"
s=np.load(f"{OUT}/edge_swap_split.npz"); bw,mn=float(s["bandwidth"]),int(s["max_neighbours"])
ad=pp.load_pfish("pfish_xenograf_tumor_linearized.h5ad"); ad=pp.filter_cells(ad,max_n_perturb=1); ad.obs_names_make_unique()
pp.build_graph(ad,bandwidth=bw,max_neighbours=mn); pp.label_context(ad)
ic,it,_=pp.make_edge_swap_sets(ad,responder="T cells")
o=ad.obs; npb=o["n_perturb"].to_numpy(); pert=o["perturbation"].astype(str).to_numpy()
isc=(o["celltype2"]=="cancer").to_numpy(); isT=(o["celltype2"]=="T cells").to_numpy(); pc=isc&(npb>0); uT=isT&(npb==0)
conn=ad.obsp[pp.CONN_ORIG_KEY].tocsr(); ln=np.asarray(ad.layers["lognorm"]); nc=o["n_counts"].to_numpy(); genes=np.array(ad.var_names)
ctrl=np.where(isc&(pert=="Control"))[0]; far=ic
def cd(idx): return np.column_stack([np.log1p(nc[idx])])
def match(a,pool): C=cd(pool); mu,sd=C.mean(0),C.std(0)+1e-9; _,mi=cKDTree((C-mu)/sd).query((cd(a)-mu)/sd,k=1); return pool[mi]
def direct_sig(kc):
    a=match(kc,ctrl); A,B=ln[kc],ln[a]; p=np.array([mannwhitneyu(A[:,g],B[:,g]).pvalue for g in range(len(genes))])
    q=multipletests(p,method="fdr_bh")[1]; d=(A.mean(0)-B.mean(0))/(np.sqrt((A.var(0)+B.var(0))/2)+1e-9); return int(((q<0.05)&(np.abs(d)>0.2)).sum())
rows=[]
for ko,cN in pd.Series(pert[pc]).value_counts().items():
    if ko=="Control": continue
    kc=np.where(pc&(pert==ko))[0]
    near=np.asarray(conn.dot((pc&(pert==ko)).astype(float))).ravel()>0
    noth=np.asarray(conn.dot((pc&(pert!=ko)).astype(float))).ravel()>0
    nT=int((uT&near&~noth).sum())
    rows.append(dict(KO=ko,n_cancer=int(cN),n_nearT=nT,direct_sig=direct_sig(kc)))
d=pd.DataFrame(rows).sort_values("n_cancer",ascending=False)
print(d.to_string(index=False))
print("\n--- tiers ---")
print(f"all KOs:                     {len(d)}")
print(f"n_cancer>=60  & n_nearT>=60:  {((d.n_cancer>=60)&(d.n_nearT>=60)).sum()}")
print(f"n_cancer>=100 & n_nearT>=60:  {((d.n_cancer>=100)&(d.n_nearT>=60)).sum()}")
print(f"n_cancer>=100 & n_nearT>=100: {((d.n_cancer>=100)&(d.n_nearT>=100)).sum()}")
print(f"direct_sig>=3  & n_nearT>=60: {((d.direct_sig>=3)&(d.n_nearT>=60)).sum()}  (effective set)")
print(f"direct_sig>=15 & n_nearT>=60: {((d.direct_sig>=15)&(d.n_nearT>=60)).sum()}  (strong set)")
d.to_csv(f"{OUT}/ko_inventory.csv",index=False)
