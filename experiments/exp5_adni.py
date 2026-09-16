"""E5 -- ADNI: diagnosis from DKT regional grey-matter volumes.

Right-hemisphere regions (see ``data.load_adni`` for why), age and sex regressed
out inside each training fold, three tasks of increasing difficulty.  CN vs MCI
is the one that matters clinically and the one where components help least.
"""
import warnings

import numpy as np
import pandas as pd

from .common import NMFLoadings, NSAPCA, PCALoadings, SparsePCALoadings, cv_score
from .data import load_adni

warnings.filterwarnings("ignore")
WS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
K = 5
TASKS = {"CN vs DEM": ("CN", "DEM"), "CN vs MCI": ("CN", "MCI"), "MCI vs DEM": ("MCI", "DEM")}


def run(hemisphere="right", n_repeats=10):
    X, meta, regions = load_adni(hemisphere=hemisphere)
    cov = np.column_stack([meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])
    rows = []
    specs = [("PCA", lambda: PCALoadings(K)),
             ("SparsePCA", lambda: SparsePCALoadings(K, alpha=1.0)),
             ("NMF", lambda: NMFLoadings(K))]
    specs += [(f"NSA-PCA (w={w})", (lambda w=w: NSAPCA(K, w))) for w in WS]

    for task, (a, b) in TASKS.items():
        m = meta.DX.isin([a, b]).to_numpy()
        Xt, yt, ct = X[m], (meta.DX[m] == b).to_numpy(int), cov[m]
        for name, loader in specs:
            s = cv_score(Xt, yt, loader, n_components=K, n_splits=5,
                         n_repeats=n_repeats, seed=0, covariates=ct)
            s.update(method=name, family=name.split(" (")[0],
                     w=float(name.split("w=")[1][:-1]) if "w=" in name else np.nan,
                     task=task, dataset="adni", hemisphere=hemisphere,
                     n=int(m.sum()), p=Xt.shape[1], k=K)
            rows.append(s)
    return pd.DataFrame(rows)


def loadings_for_figure(hemisphere="right", w=0.9):
    """Full-data loadings, for the interpretability figure only (never scored)."""
    from sklearn.preprocessing import StandardScaler
    X, meta, regions = load_adni(hemisphere=hemisphere)
    cov = np.column_stack([np.ones(len(X)), meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])
    beta, *_ = np.linalg.lstsq(cov, X, rcond=None)
    Xa = StandardScaler().fit_transform(X - cov @ beta)
    pca = PCALoadings(K).fit(Xa)
    nsa = NSAPCA(K, w).fit(Xa)
    short = [r.replace("volume right ", "").replace("volume left ", "") for r in regions]
    return short, pca.components_, nsa.components_
