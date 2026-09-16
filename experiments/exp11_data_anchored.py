r"""E11 -- data-anchored NSA-Flow against the loadings-anchored form.

The anchored form refines PCA's loadings; the data-anchored form puts X in the
fidelity and keeps V as the only variable.  Both yield a p x k basis with an
out-of-sample extension X_new V, so both are cross-validatable on equal terms.
"""
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from nsa_flow.reconstruct import nsa_flow_data
from .common import NSAPCA, PCALoadings, SparsePCALoadings, cv_score
from .data import load_adni

warnings.filterwarnings("ignore")
F64 = torch.float64
K = 5
WS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
TASKS = {"CN vs DEM": ("CN", "DEM"), "CN vs MCI": ("CN", "MCI"),
         "MCI vs DEM": ("MCI", "DEM")}


class NSAData(BaseEstimator, TransformerMixin):
    """Data-anchored basis: min (1-w)||X - XVV'||^2/||X||^2 + w Dtilde(V), V >= 0."""

    def __init__(self, n_components=5, w=0.5, max_iter=5000):
        self.n_components = n_components
        self.w = w
        self.max_iter = max_iter

    def fit(self, X, y=None):
        r = nsa_flow_data(torch.as_tensor(np.asarray(X), dtype=F64),
                          k=self.n_components, w=self.w, max_iter=self.max_iter)
        self.components_ = r.Y.numpy()
        self.defect_ = r.defect
        self.fidelity_ = r.fidelity
        self.converged_ = r.converged
        return self

    def transform(self, X):
        return np.asarray(X) @ self.components_


def run(n_repeats=10, seed=0):
    X, meta, _ = load_adni("right")
    cov = np.column_stack([meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])
    rows = []
    for task, (a, b) in TASKS.items():
        m = meta.DX.isin([a, b]).to_numpy()
        Xt, yt, ct = X[m], (meta.DX[m] == b).to_numpy(int), cov[m]
        specs = [("PCA", lambda: PCALoadings(K), "PCA", np.nan),
                 ("SparsePCA (a=4)", lambda: SparsePCALoadings(K, alpha=4.0),
                  "SparsePCA", 4.0)]
        specs += [(f"anchored (w={w})", (lambda w=w: NSAPCA(K, w)), "anchored", w)
                  for w in WS]
        specs += [(f"data (w={w})", (lambda w=w: NSAData(K, w)), "data", w)
                  for w in WS]
        for name, ld, family, w in specs:
            s = cv_score(Xt, yt, ld, n_components=K, n_splits=5,
                         n_repeats=n_repeats, seed=seed, covariates=ct)
            s.update(method=name, family=family, w=w, task=task)
            rows.append(s)
            print(f"{task:11s} {name:18s} auc={s['auc']:.4f} "
                  f"overlap={s['overlap']:.3f} sparsity={s['sparsity']:.3f}",
                  flush=True)
    return pd.DataFrame(rows)
