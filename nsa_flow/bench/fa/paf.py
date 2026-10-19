"""Principal-axis factoring, varimax, and the other classical baselines (numpy only).

``paf`` follows the algorithm of ``psych::fa(fm="pa")``: squared-multiple-correlation start, replace the
diagonal of R by the current communalities, eigendecompose, loadings = leading eigenvectors * sqrt(eigenvalues),
stop when the change in total communality drops below ``min_err`` (psych default 1e-3) or after ``max_iter``
(psych default 50).  ``varimax`` is ``stats::varimax`` (Kaiser normalisation on by default).  Both are checked
against R in ``tests/bench/test_fa.py`` when R, psych and lavaan are installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FAResult:
    loadings: np.ndarray                  # p x k
    communality: np.ndarray               # length p, diag(L L')
    iterations: int = 0
    converged: bool = True
    cert: dict = field(default_factory=dict)

    @property
    def uniqueness(self):
        return 1.0 - self.communality


def corr(X):
    """Pearson correlation of the columns of X."""
    X = np.asarray(X, float)
    Z = (X - X.mean(0)) / (X.std(0, ddof=1) + 1e-12)
    return Z.T @ Z / (len(X) - 1)


def smc(R):
    """Squared multiple correlations 1 - 1/diag(R^-1) (psych::smc); ridge-guarded if R is singular."""
    try:
        d = np.diag(np.linalg.inv(R))
    except np.linalg.LinAlgError:
        d = np.diag(np.linalg.inv(R + 1e-6 * np.eye(len(R))))
    return np.clip(1.0 - 1.0 / d, 0.0, 1.0)


def _eig_top(M, k):
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1][:k]
    return vals[order], vecs[:, order]


def orient(L):
    """psych convention: flip each column so its sum is positive, order columns by variance explained."""
    L = np.array(L, float)
    s = np.sign(L.sum(0))
    s[s == 0] = 1.0
    L = L * s
    return L[:, np.argsort((L ** 2).sum(0))[::-1]]


def paf(R, k, min_err=1e-3, max_iter=50) -> FAResult:
    Rm = R.copy()
    np.fill_diagonal(Rm, smc(R))
    comm, err, it = float(np.trace(Rm)), np.inf, 0
    while err > min_err:
        vals, vecs = _eig_top(Rm, k)
        L = vecs * np.sqrt(np.clip(vals, 0, None))
        new = (L ** 2).sum(1)
        np.fill_diagonal(Rm, new)
        err, comm = abs(comm - new.sum()), new.sum()
        it += 1
        if it >= max_iter:
            break
    L = orient(L)
    return FAResult(L, (L ** 2).sum(1), it, err <= min_err, {"converged": err <= min_err})


def varimax(L, normalize=True, eps=1e-5, maxit=1000):
    """``stats::varimax``: returns (rotated loadings, rotation matrix)."""
    x = np.array(L, float)
    nc = x.shape[1]
    if nc < 2:
        return x, np.eye(nc)
    sc = None
    if normalize:
        sc = np.sqrt((x ** 2).sum(1))
        sc[sc == 0] = 1.0
        x = x / sc[:, None]
    p = x.shape[0]
    T, d = np.eye(nc), 0.0
    for _ in range(maxit):
        z = x @ T
        B = x.T @ (z ** 3 - z @ np.diag((z ** 2).sum(0)) / p)
        u, s, vt = np.linalg.svd(B)
        T = u @ vt
        dpast, d = d, s.sum()
        if d < dpast * (1 + eps):
            break
    z = x @ T
    if normalize:
        z = z * sc[:, None]
    return z, T


def paf_varimax(R, k, **kw) -> FAResult:
    res = paf(R, k, **kw)
    Lr = orient(varimax(res.loadings)[0])
    return FAResult(Lr, (Lr ** 2).sum(1), res.iterations, res.converged, res.cert)


def pca_varimax(R, k) -> FAResult:
    vals, vecs = _eig_top(R, k)
    Lr = orient(varimax(vecs * np.sqrt(np.clip(vals, 0, None)))[0])
    return FAResult(Lr, (Lr ** 2).sum(1))


def ml_varimax(X, k, seed=0) -> FAResult:
    """Maximum-likelihood FA (sklearn) on standardised data, then varimax."""
    from sklearn.decomposition import FactorAnalysis
    Z = (X - X.mean(0)) / (X.std(0, ddof=1) + 1e-12)
    L = FactorAnalysis(n_components=k, random_state=seed).fit(Z).components_.T
    Lr = orient(varimax(L)[0])
    return FAResult(Lr, (Lr ** 2).sum(1))


def threshold_loadings(L, tau):
    """Hard simple structure: zero every loading with |lambda| < tau."""
    out = np.where(np.abs(L) >= tau, L, 0.0)
    return out
