"""E8 -- the comparison that decides whether NSA-Flow has a niche at all.

Earlier experiments scored NSA-Flow against PCA, where it trades accuracy for a
large drop in support overlap.  That is the wrong benchmark.  Sparse PCA is
statistically tied with PCA on ADNI *and* halves overlap, so the question is not
"does NSA beat PCA" but "at matched overlap, does NSA beat sparse PCA".

Previous runs fixed ``alpha = 1.0``, which pins sparse PCA to a single point on
its own accuracy/interpretability curve.  Sweeping ``alpha`` traces that curve,
so the two methods can be compared as frontiers rather than as points.  If sparse
PCA's frontier dominates NSA-Flow's everywhere down to the overlap NSA can reach,
NSA-Flow has no unique operating regime on this data.
"""
import warnings

import numpy as np
import pandas as pd

from .common import NSAPCA, PCALoadings, SparsePCALoadings, cv_score
from .data import load_adni

warnings.filterwarnings("ignore")
ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
WS = [0.0, 0.5, 0.75, 0.9, 0.95, 0.99]
K = 5
TASKS = {"CN vs DEM": ("CN", "DEM"), "CN vs MCI": ("CN", "MCI"),
         "MCI vs DEM": ("MCI", "DEM")}


def run(n_repeats=10, seed=0):
    X, meta, _ = load_adni(hemisphere="right")
    cov = np.column_stack([meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])
    specs = [("PCA", lambda: PCALoadings(K), "PCA", np.nan)]
    specs += [(f"SparsePCA (a={a})", (lambda a=a: SparsePCALoadings(K, alpha=a)),
               "SparsePCA", a) for a in ALPHAS]
    specs += [(f"NSA-PCA (w={w})", (lambda w=w: NSAPCA(K, w)), "NSA-PCA", w)
              for w in WS]
    specs += [(f"NSA-align (w={w})", (lambda w=w: NSAPCA(K, w, align=True)),
               "NSA-align", w) for w in WS]

    rows = []
    for task, (a, b) in TASKS.items():
        m = meta.DX.isin([a, b]).to_numpy()
        Xt, yt, ct = X[m], (meta.DX[m] == b).to_numpy(int), cov[m]
        for name, loader, family, knob in specs:
            s = cv_score(Xt, yt, loader, n_components=K, n_splits=5,
                         n_repeats=n_repeats, seed=seed, covariates=ct)
            s.update(method=name, family=family, knob=knob, task=task, seed=seed)
            rows.append(s)
    return pd.DataFrame(rows)


def dominance(df, grid=(0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)):
    """AUC of each family read at matched overlap; positive delta favours NSA."""
    rows = []
    for task in df.task.unique():
        t = df[df.task == task]
        curves = {f: t[t.family == f].sort_values("overlap")
                  for f in ("SparsePCA", "NSA-PCA", "NSA-align")}
        for tgt in grid:
            r = {"task": task, "overlap": tgt}
            for f, c in curves.items():
                r[f] = (np.interp(tgt, c.overlap, c.auc)
                        if c.overlap.min() <= tgt <= c.overlap.max() else np.nan)
            r["reach_sparse"] = float(curves["SparsePCA"].overlap.min())
            rows.append(r)
    out = pd.DataFrame(rows)
    out["nsa_minus_sparse"] = out["NSA-align"] - out["SparsePCA"]
    return out
