r"""Is the norm-balance term -- the term the theory fix ADDED -- what costs accuracy?

D = sum_{i!=j} G_ij^2  +  sum_i (G_ii - 1/k)^2
    \_ decorrelation _/   \_ norm balance  _/

v1 penalised only the first.  For Y >= 0, disjoint supports already make Y'Y
diagonal, so decorrelation alone suffices for disjointness; norm balance is the
extra demand that every column carry equal weight -- i.e. the 'orthoNORMAL' part
of the requirement, as against merely 'orthogonal'.  Both variants are solved
with the SAME correct solver (projected gradient + Armijo, autograd), so the
comparison isolates the functional and not v1's bugs.
"""
import numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from experiments.data import load_adni
from experiments.common import cv_score, PCALoadings

F64 = torch.float64

def solve(X0, w, mode, max_iter=3000, tol=1e-9):
    """Projected gradient with Armijo on (1-w)F + w Dvariant, Y >= 0."""
    k = X0.shape[1]
    X0 = X0.to(F64); denom = X0.pow(2).sum()
    eye = torch.eye(k, dtype=F64)
    def E(Y):
        S = Y.T @ Y; G = S / S.trace()
        if mode == "full":
            D = (G - eye / k).pow(2).sum() / (1 - 1 / k)
        elif mode == "offdiag":                      # v1's functional
            D = (G - torch.diag(torch.diagonal(G))).pow(2).sum() / (1 - 1 / k)
        else:
            raise ValueError(mode)
        return (1 - w) * (Y - X0).pow(2).sum() / denom + w * D
    Y = X0.clone().clamp_min(0)
    t = 1.0
    for _ in range(max_iter):
        Yv = Y.detach().requires_grad_(True)
        e = E(Yv); e.backward(); g = Yv.grad
        e0 = float(e)
        for _ in range(60):
            Yn = (Y - t * g).clamp_min(0)
            if float(E(Yn)) <= e0 - 1e-4 * float((Yn - Y).pow(2).sum()) / t:
                break
            t *= 0.5
        else:
            break
        step = float((Yn - Y).norm()) / t
        Y = Yn; t *= 2.0
        if step <= tol:
            break
    return Y.detach()

class Variant(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5, w=0.9, mode="full"):
        self.n_components = n_components; self.w = w; self.mode = mode
    def fit(self, X, y=None):
        L = (PCA(self.n_components, svd_solver="randomized",
                       random_state=0).fit(X).components_.T)
        Y = solve(torch.as_tensor(L, dtype=F64), self.w, self.mode)
        self.components_ = Y.numpy(); return self
    def transform(self, X): return X @ self.components_

X, meta, _ = load_adni("right")
cov = np.column_stack([meta.AGE.to_numpy(float), (meta.SEX.astype(str)=="M").to_numpy(float)])
K = 5
TASKS = {"CN vs DEM": ("CN","DEM"), "CN vs MCI": ("CN","MCI"), "MCI vs DEM": ("MCI","DEM")}
rows = []
for task,(A,B) in TASKS.items():
    m = meta.DX.isin([A,B]).to_numpy()
    Xt, yt, ct = X[m], (meta.DX[m]==B).to_numpy(int), cov[m]
    specs = [("PCA", lambda: PCALoadings(K), "PCA", np.nan)]
    for w in (0.5, 0.9, 0.99):
        for mode in ("full", "offdiag"):
            specs.append((f"{mode} (w={w})", (lambda w=w, mo=mode: Variant(K, w, mo)), mode, w))
    for name, ld, mode, w in specs:
        s = cv_score(Xt, yt, ld, n_components=K, n_splits=5, n_repeats=4, seed=0, covariates=ct)
        rows.append(dict(task=task, method=name, mode=mode, w=w, auc=s['auc'],
                         sd=s['auc_sd'], overlap=s['overlap'], sparsity=s['sparsity']))
        print(f"{task:11s} {name:16s} auc={s['auc']:.4f} overlap={s['overlap']:.3f} "
              f"sparsity={s['sparsity']:.3f}", flush=True)
pd.DataFrame(rows).to_csv('paper/results/e10_norm_balance_ablation.csv', index=False)
print("SAVED")
