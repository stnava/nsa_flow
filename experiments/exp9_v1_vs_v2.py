"""Did v1 ever beat PCA?  v1 verbatim, through the corrected leakage-free harness."""
import numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from experiments.data import load_adni
from experiments.common import cv_score, PCALoadings, NSAPCA
from experiments.v1.flow import nsa_flow_orth

K = 5
class V1PCA(BaseEstimator, TransformerMixin):
    """PCA loadings refined by v1's nsa_flow_orth, defaults as shipped."""
    def __init__(self, n_components=5, w=0.5, orth_type="scale_invariant"):
        self.n_components = n_components; self.w = w; self.orth_type = orth_type
    def fit(self, X, y=None):
        L = np.abs(PCA(n_components=self.n_components, svd_solver="randomized",
                       random_state=0).fit(X).components_.T)
        Y = nsa_flow_orth(torch.as_tensor(L, dtype=torch.float64), w=self.w,
                          orth_type=self.orth_type, max_iter=500, verbose=False)
        Y = Y[0] if isinstance(Y, tuple) else Y
        self.components_ = (Y.detach().numpy() if torch.is_tensor(Y)
                            else np.asarray(Y["Y"] if isinstance(Y, dict) else Y))
        return self
    def transform(self, X): return X @ self.components_

X, meta, _ = load_adni("right")
cov = np.column_stack([meta.AGE.to_numpy(float), (meta.SEX.astype(str)=="M").to_numpy(float)])
TASKS={"CN vs DEM":("CN","DEM"),"CN vs MCI":("CN","MCI"),"MCI vs DEM":("MCI","DEM")}
WS=[0.5,0.9,0.99]
rows=[]
for task,(A,B) in TASKS.items():
    m=meta.DX.isin([A,B]).to_numpy(); Xt,yt,ct=X[m],(meta.DX[m]==B).to_numpy(int),cov[m]
    specs=[("PCA",lambda:PCALoadings(K))]
    specs+=[(f"v1 (w={w})",(lambda w=w: V1PCA(K,w))) for w in WS]
    specs+=[(f"v2 (w={w})",(lambda w=w: NSAPCA(K,w))) for w in WS]
    for name,ld in specs:
        try:
            s=cv_score(Xt,yt,ld,n_components=K,n_splits=5,n_repeats=4,seed=0,covariates=ct)
            rows.append(dict(task=task,method=name,auc=s['auc'],sd=s['auc_sd'],
                             overlap=s['overlap'],sparsity=s['sparsity']))
            print(f"{task:11s} {name:12s} auc={s['auc']:.4f} overlap={s['overlap']:.3f}", flush=True)
        except Exception as e:
            print(f"{task:11s} {name:12s} FAILED: {type(e).__name__}: {e}", flush=True)
pd.DataFrame(rows).to_csv('paper/results/e9_v1_vs_v2.csv', index=False)
print("SAVED")
