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
    phi: np.ndarray | None = None         # factor correlations (None = orthogonal factors, identity)

    @property
    def uniqueness(self):
        return 1.0 - self.communality

    @property
    def factor_corr(self):
        return np.eye(self.loadings.shape[1]) if self.phi is None else self.phi

    def implied(self):
        """Model-implied correlation L Phi L' (diagonal = communalities)."""
        return self.loadings @ self.factor_corr @ self.loadings.T


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


def _result(L, phi, base: FAResult | None = None, cert=None) -> FAResult:
    phi = None if phi is None else np.asarray(phi, float)
    Lm = np.asarray(L, float)
    comm = np.einsum("ik,kl,il->i", Lm, np.eye(Lm.shape[1]) if phi is None else phi, Lm)
    return FAResult(Lm, comm, 0 if base is None else base.iterations, True if base is None else base.converged,
                    {} if cert is None else cert, phi)


def promax(L, m=4):
    """``stats::promax``/``psych::Promax``: varimax, then a power target; returns (pattern loadings, factor correlations)."""
    xx, rot = varimax(L)
    Q = xx * np.abs(xx) ** (m - 1)
    U = np.linalg.lstsq(xx, Q, rcond=None)[0]
    U = U * np.sqrt(np.diag(np.linalg.inv(U.T @ U)))
    Uf = rot @ U
    ui = np.linalg.inv(Uf)
    return np.asarray(L, float) @ Uf, ui @ ui.T


def oblimin(A, gam=0.0, maxit=1000, eps=1e-5):
    """Oblique oblimin rotation by gradient projection (Bernaards & Jennrich); ``gam = 0`` is quartimin.
    Returns (pattern loadings, factor correlations)."""
    A = np.asarray(A, float)
    p, k = A.shape
    T = np.eye(k)

    def vgq(L):
        X = (L ** 2) @ (1 - np.eye(k))
        if gam:
            X = (np.eye(p) - gam / p) @ X
        return (L ** 2 * X).sum() / 4, L * X

    L = A @ np.linalg.inv(T).T
    f, Gq = vgq(L)
    G = -(L.T @ Gq @ np.linalg.inv(T)).T
    al = 1.0
    for _ in range(maxit):
        Gp = G - T @ np.diag((T * G).sum(0))
        s = np.sqrt((Gp ** 2).sum())
        if s < eps:
            break
        al *= 2
        for _ in range(11):
            X = T - al * Gp
            Tt = X / np.sqrt((X ** 2).sum(0))
            L = A @ np.linalg.inv(Tt).T
            ft, Gq = vgq(L)
            if ft < f - 0.5 * s ** 2 * al:
                break
            al /= 2
        T, f = Tt, ft
        G = -(L.T @ Gq @ np.linalg.inv(T)).T
    return A @ np.linalg.inv(T).T, T.T @ T


def paf_promax(R, k, **kw) -> FAResult:
    base = paf(R, k, **kw)
    L, phi = promax(base.loadings)
    return _result(L, phi, base)


def paf_oblimin(R, k, **kw) -> FAResult:
    base = paf(R, k, **kw)
    L, phi = oblimin(base.loadings)
    return _result(L, phi, base)
