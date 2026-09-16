"""E1 -- NSA-Flow as a refinement operator on a planted non-negative basis.

The ground truth is a non-negative disjoint-support basis, i.e. a point in the
``w -> 1`` solution class.  NSA-Flow is defined as refining a *supplied* target
``X0``, so the question it should be asked is not "does PCA+NSA beat sparse PCA"
--- that compares the choice of input basis, not the operator --- but "does NSA
improve whatever basis it is handed".  We therefore run it on loadings from PCA,
NMF and sparse PCA and report each base against its refinement.

The naive head-to-head is also reported, because it is what a reader would
otherwise compute and it is informative about the pipeline as a whole.
"""
import warnings

import numpy as np
import pandas as pd
import torch

from nsa_flow import nsa_flow
from .common import (NMFLoadings, PCALoadings, SparsePCALoadings,
                     matched_cosine, sparsity, support_f1, support_overlap)
from .data import planted_partition

warnings.filterwarnings("ignore")
F64 = torch.float64

WS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
NOISES = [0.2, 0.4, 0.8, 1.6]
SEEDS = range(10)
BASES = {"PCA": lambda k: PCALoadings(k),
         "NMF": lambda k: NMFLoadings(k),
         "SparsePCA": lambda k: SparsePCALoadings(k, alpha=1.0)}


def _metrics(L, V):
    return dict(cosine=matched_cosine(L, V), support_f1=support_f1(L, V),
                sparsity=sparsity(L), overlap=support_overlap(L))


def run():
    """One row per (base, w, noise, seed); ``w = -1`` marks the unrefined base."""
    rows = []
    k = 6
    for noise in NOISES:
        for seed in SEEDS:
            X, V, _ = planted_partition(p=60, k=k, n=300, noise=noise, seed=seed)
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
            for base_name, mk in BASES.items():
                L0 = np.abs(mk(k).fit(Xs).components_)
                rows.append(dict(base=base_name, w=-1.0, refined=False,
                                 noise=noise, seed=seed, **_metrics(L0, V)))
                T = torch.as_tensor(L0, dtype=F64)
                for w in WS:
                    r = nsa_flow(T, w=w, max_iter=20000, tol=1e-11)
                    rows.append(dict(base=base_name, w=w, refined=True,
                                     noise=noise, seed=seed,
                                     defect=r.raw_defect,
                                     eff_rank=r.effective_rank,
                                     **_metrics(r.Y.numpy(), V)))
    return pd.DataFrame(rows)


def summarise(df):
    return (df.groupby(["base", "w", "refined", "noise"], dropna=False)
              .agg(cosine=("cosine", "mean"), cosine_sd=("cosine", "std"),
                   support_f1=("support_f1", "mean"), sparsity=("sparsity", "mean"),
                   overlap=("overlap", "mean"), n=("cosine", "size"))
              .reset_index())


def refinement_table(df):
    """Each base against its best refinement, averaged over noise and seeds."""
    rows = []
    for base in BASES:
        b = df[(df.base == base) & (~df.refined)]
        r = df[(df.base == base) & (df.refined) & (df.w > 0)]
        per_w = r.groupby("w")[["cosine", "support_f1", "overlap"]].mean()
        best_w = per_w.cosine.idxmax()
        rows.append({
            "base": base,
            "cos": b.cosine.mean(), "cos +NSA": per_w.loc[best_w, "cosine"],
            "F1": b.support_f1.mean(), "F1 +NSA": per_w.loc[best_w, "support_f1"],
            "overlap": b.overlap.mean(), "overlap +NSA": per_w.loc[best_w, "overlap"],
            "best $w$": best_w,
        })
    return pd.DataFrame(rows)
