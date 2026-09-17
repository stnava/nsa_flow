"""E7 -- anchoring to X0's O(k) orbit instead of to X0 itself.

``D`` is exactly right-``O(k)`` invariant (``tests/test_theory.py``), so the
anchored fidelity ``||Y - X0||_F^2`` charges full price to preserve a rotation the
orthogonality term cannot see.  Where only the *span* of ``X0`` is trustworthy --
PCA pins its subspace by the eigenvalue gaps but its rotation only by the
variance-ordering convention -- that over-constrains the problem.  Replacing the
numerator with ``min_{Q in O(k)} ||Y - X0 Q||_F^2`` (orthogonal Procrustes, closed
form) quotients the rotation out while keeping ``X0`` as the anchor.

This is E1's design with ``align`` as the only added factor, so the comparison is
paired: identical planted truth, identical base loadings, identical ``w`` grid.
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

WS = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
NOISES = [0.2, 0.4, 0.8, 1.6]
SEEDS = range(10)
BASES = {"PCA": lambda k: PCALoadings(k),
         "NMF": lambda k: NMFLoadings(k),
         "SparsePCA": lambda k: SparsePCALoadings(k, alpha=1.0)}


def _metrics(L, V):
    return dict(cosine=matched_cosine(L, V), support_f1=support_f1(L, V),
                sparsity=sparsity(L), overlap=support_overlap(L))


def run():
    """One row per (base, w, aligned, noise, seed).

    The flag column is ``aligned``, not ``align``: a DataFrame column named
    ``align`` is shadowed by ``DataFrame.align``, so ``df.align`` silently
    returns the bound method rather than the column.
    """
    rows, k = [], 6
    for noise in NOISES:
        for seed in SEEDS:
            X, V, _ = planted_partition(p=60, k=k, n=300, noise=noise, seed=seed)
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
            for base_name, mk in BASES.items():
                L0 = mk(k).fit(Xs).components_
                T = torch.as_tensor(L0, dtype=F64)
                for w in WS:
                    for align in (False, True):
                        r = nsa_flow(T, w=w, align=align, max_iter=20000, tol=1e-11)
                        rows.append(dict(base=base_name, w=w, aligned=align,
                                         noise=noise, seed=seed,
                                         defect=r.raw_defect, iters=r.iters,
                                         seconds=r.seconds,
                                         converged=r.converged,
                                         **_metrics(r.Y.numpy(), V)))
    return pd.DataFrame(rows)


def summarise(df):
    """Paired anchored-vs-aligned comparison at each base's best w."""
    rows = []
    for base in BASES:
        d = df[df.base == base]
        per = d.groupby(["aligned", "w"])[["cosine", "support_f1", "overlap",
                                         "defect", "iters"]].mean()
        for align in (False, True):
            g = per.loc[align]
            bw = g.cosine.idxmax()
            rows.append(dict(base=base, aligned=align, best_w=bw,
                             cosine=g.cosine[bw], support_f1=g.support_f1[bw],
                             overlap=g.overlap[bw], defect=g.defect[bw],
                             iters=g.iters[bw]))
    out = pd.DataFrame(rows)
    # paired per-replicate delta at each base's own best w, for a signed test
    deltas = []
    for base in BASES:
        d = df[df.base == base]
        for align in (True,):
            a = d[d["aligned"]]
            b = d[~d["aligned"]]
            bw_a = a.groupby("w").cosine.mean().idxmax()
            bw_b = b.groupby("w").cosine.mean().idxmax()
            ka = a[a.w == bw_a].set_index(["noise", "seed"]).cosine
            kb = b[b.w == bw_b].set_index(["noise", "seed"]).cosine
            dd = (ka - kb).dropna()
            deltas.append(dict(base=base, mean_delta=dd.mean(), sd=dd.std(),
                               n=len(dd), frac_improved=float((dd > 0).mean())))
    return out, pd.DataFrame(deltas)


def frontier(df, grid=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
    """Recovery at *matched interpretability*, which is the only fair comparison.

    At a matched ``w`` the aligned form reaches higher ``cosine`` but also higher
    ``overlap``: letting ``X0`` rotate makes the fidelity term cheaper to satisfy,
    so at fixed ``w`` less pressure reaches ``D``.  That makes ``w`` non-comparable
    *across* modes even though it stays an exact blend fraction *within* each.
    Interpolating each mode's ``(overlap, cosine)`` curve over the ``w`` grid and
    reading both at the same overlap removes the reparameterisation.
    """
    rows = []
    for base in df.base.unique():
        d = df[df.base == base].groupby(["aligned", "w"])[["cosine", "overlap"]].mean()
        for tgt in grid:
            r = {"base": base, "overlap": tgt}
            for al in (False, True):
                g = d.loc[al].sort_values("overlap")
                r["aligned" if al else "anchored"] = (
                    np.interp(tgt, g.overlap, g.cosine)
                    if g.overlap.min() <= tgt <= g.overlap.max() else np.nan)
            r["delta"] = r["aligned"] - r["anchored"]
            rows.append(r)
    return pd.DataFrame(rows).dropna().reset_index(drop=True)
