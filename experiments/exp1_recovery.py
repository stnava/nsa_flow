"""E1 -- recovery of a planted non-negative, disjoint-support basis.

Does the w dial actually buy identifiability?  The ground truth is a basis in
the w -> 1 solution class, so this measures whether driving D to zero recovers
structure that PCA, sparse PCA and NMF cannot.
"""
import warnings

import numpy as np
import pandas as pd

from .common import (NMFLoadings, NSAPCA, PCALoadings, SparsePCALoadings,
                     matched_cosine, support_f1, sparsity, support_overlap)
from .data import planted_partition

warnings.filterwarnings("ignore")

WS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
NOISES = [0.2, 0.4, 0.8, 1.6]
SEEDS = range(10)


def run():
    rows = []
    for noise in NOISES:
        for seed in SEEDS:
            X, V, _ = planted_partition(p=60, k=6, n=300, noise=noise, seed=seed)
            methods = [("PCA", PCALoadings(6)),
                       ("SparsePCA", SparsePCALoadings(6, alpha=1.0)),
                       ("NMF", NMFLoadings(6))]
            methods += [(f"NSA-PCA (w={w})", NSAPCA(6, w)) for w in WS]
            for name, model in methods:
                L = model.fit(X).components_
                rows.append(dict(
                    method=name, family=name.split(" (")[0], noise=noise, seed=seed,
                    w=float(name.split("w=")[1][:-1]) if "w=" in name else np.nan,
                    cosine=matched_cosine(L, V), support_f1=support_f1(L, V),
                    sparsity=sparsity(L), overlap=support_overlap(L),
                ))
    return pd.DataFrame(rows)


def summarise(df):
    g = (df.groupby(["method", "family", "w", "noise"], dropna=False)
           .agg(cosine=("cosine", "mean"), cosine_sd=("cosine", "std"),
                support_f1=("support_f1", "mean"), sparsity=("sparsity", "mean"),
                overlap=("overlap", "mean"), n=("cosine", "size"))
           .reset_index())
    return g
