"""E4 -- Golub leukemia: ALL vs AML from gene-expression components.

72 samples, 7129 genes.  Every step -- standardisation, component extraction,
NSA refinement, classifier -- is fitted inside the training fold.  With 72
samples, fitting components on the full matrix first would leak and is the usual
reason such numbers do not replicate.
"""
import warnings

import numpy as np
import pandas as pd

from .common import (NMFLoadings, NSAPCA, PCALoadings, SparsePCALoadings, cv_score)
from .data import load_golub

warnings.filterwarnings("ignore")
WS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
K = 5
N_FEATURES = 2000          # top-variance genes, selected inside each fold


def run(n_repeats=10):
    X, y, _ = load_golub()
    X = np.log2(np.clip(X, 1, None))                  # standard for this array data
    rows = []
    specs = [("PCA", lambda: PCALoadings(K)),
             ("SparsePCA", lambda: SparsePCALoadings(K, alpha=1.0)),
             ("NMF", lambda: NMFLoadings(K))]
    specs += [(f"NSA-PCA (w={w})", (lambda w=w: NSAPCA(K, w))) for w in WS]
    for name, loader in specs:
        s = cv_score(X, y, loader, n_components=K, n_splits=5,
                     n_repeats=n_repeats, seed=0, n_features=N_FEATURES)
        s.update(method=name, family=name.split(" (")[0],
                 w=float(name.split("w=")[1][:-1]) if "w=" in name else np.nan,
                 dataset="golub", n=len(y), p=N_FEATURES, k=K)
        rows.append(s)
    return pd.DataFrame(rows)
