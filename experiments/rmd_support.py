r"""Experiment functions for ``paper/nsa_flow_v2.Rmd``.

Each returns a tidy ``DataFrame`` so the document is narrative plus tables and
plots, with the computation here where it can be tested and reused.  Sizes are
chosen so the whole document knits in a few minutes.
"""
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
F64 = torch.float64


# ---------------------------------------------------------------- 1. properties
def properties_table():
    """Does each functional recognise the structures the method is built around?

    ``D`` is orthoNORMality, ``d`` is v1's off-diagonal-only defect, ``C`` is
    orthogonality.  The rows are the cases that separate them.
    """
    from nsa_flow import stiefel_defect, angle_defect
    torch.manual_seed(0)
    p, k = 20, 5
    Q = torch.linalg.qr(torch.randn(p, k, dtype=F64))[0]

    def d_off(V):
        G = V.T @ V
        G = G / G.trace()
        return float((G - torch.diag(torch.diagonal(G))).pow(2).sum())

    cases = {}
    cases["orthonormal"] = Q
    cases["orthogonal, unequal norms"] = Q @ torch.diag(
        torch.tensor([1.0, 1, 1, 1, 8], dtype=F64))
    V = torch.zeros(p, k, dtype=F64)
    for i in range(k):                       # disjoint blocks, unequal scales
        V[i * 4:(i + 1) * 4, i] = torch.rand(4, dtype=F64) * (i + 1) * 5
    cases["disjoint nonneg supports"] = V
    Z = torch.zeros(p, k, dtype=F64)
    Z[:, 0] = torch.rand(p, dtype=F64)
    cases["rank collapse (1 live col)"] = Z
    cases["all columns collinear"] = (torch.rand(p, 1, dtype=F64)
                                      @ torch.ones(1, k, dtype=F64))
    rows = []
    for name, M in cases.items():
        rows.append(dict(case=name, D=float(stiefel_defect(M)), d_v1=d_off(M),
                         C=float(angle_defect(M))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. abs trap
def abs_trap_table(X):
    """Four ways to get a non-negative basis, scored by reconstruction error.

    ``abs()`` is not even the projection onto the non-negative cone: for an entry
    ``-3`` it moves distance 6 where the projection moves 3.  And no one-shot map
    can work, because a signed component encodes a contrast a single
    non-negative column cannot represent.
    """
    from nsa_flow import nsa_flow_data
    from nsa_flow.reconstruct import reconstruction_fidelity
    T = torch.as_tensor(np.asarray(X), dtype=F64)
    S, c = T.T @ T, T.pow(2).sum()
    k = 5
    _, _, Vh = torch.linalg.svd(T, full_matrices=False)
    E = Vh[:k].T

    def err(V):
        return float(reconstruction_fidelity(V, S, c)) ** 0.5

    rows = [
        dict(basis="signed PCA (unconstrained optimum)", nonneg=False, err=err(E)),
        dict(basis="abs(PCA)", nonneg=True, err=err(E.abs())),
        dict(basis="clamp(PCA) -- the true projection", nonneg=True,
             err=err(E.clamp_min(0.0))),
        dict(basis="fitted to the data (w=0)", nonneg=True,
             err=err(nsa_flow_data(T, k=k, w=0.0).Y)),
    ]
    out = pd.DataFrame(rows)
    out["cost_vs_signed"] = out.err / out.err.iloc[0]
    return out


# --------------------------------------------------------------------- 3. dial
def w_dial_table(X, ws=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99), k=5):
    """One parameter, monotone in the quantity it is supposed to control."""
    from nsa_flow import nsa_flow_data, angle_defect
    from nsa_flow.reconstruct import reconstruction_fidelity
    T = torch.as_tensor(np.asarray(X), dtype=F64)
    S, c = T.T @ T, T.pow(2).sum()
    rows = []
    for w in ws:
        r = nsa_flow_data(T, k=k, w=w)
        V = r.Y.numpy()
        sup = np.abs(V) > 1e-10
        live = sup.any(1)
        rows.append(dict(
            w=w, recon_err=float(reconstruction_fidelity(r.Y, S, c)) ** 0.5,
            C=float(angle_defect(r.Y)), sparsity=float((V == 0).mean()),
            overlap=float((sup.sum(1)[live] - 1).mean()) if live.any() else 0.0,
            iters=int(r.iters), converged=bool(r.converged)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 4. homotopy
def homotopy_table(X, k=5, w=0.5):
    """Path resolution, not iteration count, selects the local minimum."""
    from nsa_flow import nsa_flow_data
    from nsa_flow.reconstruct import GramOperator, relax_into_nonneg, _fid
    T = torch.as_tensor(np.asarray(X), dtype=F64)
    ops = GramOperator(S=T.T @ T)
    c = ops.c
    schedules = {
        "1 stage  (mu = 0 only)": [0.0],
        "3 stages": [0.0, 1.0, 1e2],
        "9 stages (default)": [0.0] + [10.0 ** e for e in range(-3, 5)],
    }
    rows = []
    for name, mus in schedules.items():
        for it in (150, 3000):
            t0 = time.time()
            V = relax_into_nonneg(ops, c, k, w, mus=mus, max_iter=it)
            rows.append(dict(schedule=name, iters_per_stage=it,
                             energy=float(_fid(V.clamp_min(0), ops, c)),
                             seconds=time.time() - t0))
    return pd.DataFrame(rows)


# -------------------------------------------------------------- 5. scaling
def scaling_table(n=200, ps=(500, 2000, 8000), k=5, max_iter=60):
    """Matrix-free against forming the Gram matrix."""
    from nsa_flow import nsa_flow_data
    rows = []
    for p in ps:
        torch.manual_seed(0)
        X = torch.rand(n, p, dtype=F64)
        entry = dict(p=p, gram_GB=p * p * 8 / 1e9)
        for mf in (False, True):
            t0 = time.time()
            nsa_flow_data(X, k=k, w=0.5, matrix_free=mf, max_iter=max_iter,
                          init="clamp")
            entry["matrix_free_s" if mf else "gram_s"] = time.time() - t0
        entry["speedup"] = entry["gram_s"] / entry["matrix_free_s"]
        rows.append(entry)
    return pd.DataFrame(rows)


# ------------------------------------------------------- 6. real data frontier
def adni_frontier(n_repeats=2, k=5, ws=(0.0, 0.5, 0.9), alphas=(1.0, 4.0)):
    """Accuracy against interpretability on ADNI, the comparison that matters.

    Every fit -- scaling, covariate adjustment, basis, classifier -- happens
    inside the training fold.  The competitor is sparse PCA, not PCA: sparse PCA
    is the method that already buys interpretability nearly for free.
    """
    from sklearn.base import BaseEstimator, TransformerMixin
    from .common import NMFLoadings, PCALoadings, SparsePCALoadings, cv_score
    from .data import load_adni
    from nsa_flow import nsa_flow_data

    class DataAnchored(BaseEstimator, TransformerMixin):
        def __init__(self, n_components=5, w=0.5):
            self.n_components, self.w = n_components, w

        def fit(self, X, y=None):
            self.components_ = nsa_flow_data(
                torch.as_tensor(np.asarray(X), dtype=F64),
                k=self.n_components, w=self.w).Y.numpy()
            return self

        def transform(self, X):
            return np.asarray(X) @ self.components_

    X, meta, _ = load_adni("right")
    cov = np.column_stack([meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])
    tasks = {"CN vs DEM": ("CN", "DEM"), "CN vs MCI": ("CN", "MCI"),
             "MCI vs DEM": ("MCI", "DEM")}
    specs = [("PCA", lambda: PCALoadings(k), "PCA")]
    specs += [(f"SparsePCA (a={a})", (lambda a=a: SparsePCALoadings(k, alpha=a)),
               "SparsePCA") for a in alphas]
    specs += [("NMF", lambda: NMFLoadings(k), "NMF")]
    specs += [(f"NSA (w={w})", (lambda w=w: DataAnchored(k, w)), "NSA")
              for w in ws]
    rows = []
    for task, (a, b) in tasks.items():
        m = meta.DX.isin([a, b]).to_numpy()
        Xt, yt, ct = X[m], (meta.DX[m] == b).to_numpy(int), cov[m]
        for name, loader, family in specs:
            s = cv_score(Xt, yt, loader, n_components=k, n_splits=5,
                         n_repeats=n_repeats, seed=0, covariates=ct)
            s.update(method=name, family=family, task=task)
            rows.append(s)
    return pd.DataFrame(rows)[["task", "method", "family", "auc", "auc_sd",
                               "overlap", "sparsity", "defect"]]


# ---------------------------------------------------------- 7. reproducibility
def reproducibility_table(n_splits=5, k=5, w=0.5):
    """Do independent subsamples return the same basis?

    PCA's components 2-5 sit in a near-degenerate block on this data (see the
    eigengaps in the document), so Davis-Kahan says their orientation is
    noise-dominated.  A disjointness constraint selects a representative.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from .common import SparsePCALoadings
    from .data import load_adni
    from nsa_flow import nsa_flow_data

    X, meta, _ = load_adni("right")
    cov = np.column_stack([np.ones(len(X)), meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])

    def matched(A, B):
        A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
        Cm = np.abs(A.T @ B)
        r, c = linear_sum_assignment(-Cm)
        return float(Cm[r, c].mean())

    fits = {"PCA": [], "SparsePCA (a=4)": [], "NSA (w=%.2g)" % w: []}
    for tr, _ in KFold(n_splits, shuffle=True, random_state=0).split(X):
        beta, *_ = np.linalg.lstsq(cov[tr], X[tr], rcond=None)
        Xa = StandardScaler().fit_transform(X[tr] - cov[tr] @ beta)
        fits["PCA"].append(PCA(k, random_state=0).fit(Xa).components_.T)
        fits["SparsePCA (a=4)"].append(
            SparsePCALoadings(k, alpha=4.0).fit(Xa).components_)
        fits["NSA (w=%.2g)" % w].append(
            nsa_flow_data(torch.as_tensor(Xa, dtype=F64), k=k, w=w).Y.numpy())
    rows = []
    for name, Ls in fits.items():
        vals = [matched(Ls[i], Ls[j]) for i in range(len(Ls))
                for j in range(i + 1, len(Ls))]
        rows.append(dict(method=name, n_pairs=len(vals),
                         matched_cos=float(np.mean(vals)),
                         sd=float(np.std(vals))))
    return pd.DataFrame(rows)


def adni_spectrum(k=8):
    """Leading eigenvalues and consecutive gap ratios of the analysis matrix."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from .data import load_adni
    X, meta, _ = load_adni("right")
    cov = np.column_stack([np.ones(len(X)), meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])
    beta, *_ = np.linalg.lstsq(cov, X, rcond=None)
    Xa = StandardScaler().fit_transform(X - cov @ beta)
    ev = PCA(k, random_state=0).fit(Xa).explained_variance_
    return pd.DataFrame(dict(component=np.arange(1, k + 1), eigenvalue=ev,
                             gap_ratio=np.r_[ev[:-1] / ev[1:], np.nan]))
