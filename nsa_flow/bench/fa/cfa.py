"""Maximum-likelihood confirmatory factor analysis for a simple-structure model (numpy + scipy).

Model: Sigma = Lambda Phi Lambda' + Theta, each item loads on exactly one factor (given by ``assign``),
Phi is a factor correlation matrix (factor variances fixed to 1), Theta diagonal.  Fit indices follow lavaan's
definitions (chi-square = N * F_ML, RMSEA, CFI/TLI against the independence model, SRMR, BIC); they are checked
against ``lavaan::cfa`` in ``tests/bench/test_fa.py`` when R is available.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CFAFit:
    chisq: float
    df: int
    rmsea: float
    cfi: float
    tli: float
    srmr: float
    bic: float
    converged: bool
    loadings: np.ndarray
    phi: np.ndarray
    theta: np.ndarray


def _f_ml(S, Sigma):
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return np.inf
    _, logdet_s = np.linalg.slogdet(S)
    return logdet + np.trace(S @ np.linalg.inv(Sigma)) - logdet_s - S.shape[0]


def fit_cfa(S, n, assign) -> CFAFit | None:
    """Fit the simple-structure CFA to covariance ``S`` (p x p) from ``n`` observations.

    ``assign[i]`` is the factor of item ``i``.  Returns None when the model is not identified
    (a factor with fewer than two items) or the optimiser fails.
    """
    from scipy.optimize import minimize
    S = np.asarray(S, float)
    p = S.shape[0]
    assign = np.asarray(assign)
    fac = sorted(set(assign.tolist()))
    k = len(fac)
    idx = np.array([fac.index(a) for a in assign])
    if any((idx == j).sum() < 2 for j in range(k)) or k < 1:
        return None
    iu = np.triu_indices(k, 1)
    sd = np.sqrt(np.diag(S))

    # starting values: loading sign relative to the first item of the factor
    lam0 = np.zeros(p)
    for j in range(k):
        members = np.where(idx == j)[0]
        ref = members[0]
        for i in members:
            sgn = 1.0 if i == ref else np.sign(S[i, ref]) or 1.0
            lam0[i] = 0.6 * sd[i] * sgn
    x0 = np.concatenate([lam0, np.full(len(iu[0]), 0.3), 0.5 * np.diag(S)])
    bounds = [(None, None)] * p + [(-0.999, 0.999)] * len(iu[0]) + [(1e-4, None)] * p

    def unpack(x):
        lam, ph, th = x[:p], x[p:p + len(iu[0])], x[p + len(iu[0]):]
        L = np.zeros((p, k))
        L[np.arange(p), idx] = lam
        Phi = np.eye(k)
        Phi[iu] = ph
        Phi = Phi + Phi.T - np.eye(k)
        return L, Phi, th

    def fun(x):
        L, Phi, th = unpack(x)
        Sigma = L @ Phi @ L.T + np.diag(th)
        f = _f_ml(S, Sigma)
        if not np.isfinite(f):
            return 1e10, np.zeros_like(x)
        Si = np.linalg.inv(Sigma)
        G = Si @ (Sigma - S) @ Si
        gL = 2 * G @ L @ Phi
        g = np.concatenate([gL[np.arange(p), idx],
                            2 * (L.T @ G @ L)[iu] if k > 1 else np.zeros(0),
                            np.diag(G)])
        return f, g

    r = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9})
    L, Phi, th = unpack(r.x)
    Sigma = L @ Phi @ L.T + np.diag(th)
    F = _f_ml(S, Sigma)
    npar = p + len(iu[0]) + p
    df = p * (p + 1) // 2 - npar
    chisq = n * F
    # independence model: Sigma = diag(S)
    Fb = _f_ml(S, np.diag(np.diag(S)))
    chisq_b, df_b = n * Fb, p * (p - 1) // 2
    rmsea = float(np.sqrt(max(chisq - df, 0.0) / (df * n))) if df > 0 else 0.0
    d_m, d_b = max(chisq - df, 0.0), max(chisq_b - df_b, 0.0)
    cfi = 1.0 - d_m / max(d_m, d_b, 1e-12)
    tli = ((chisq_b / df_b) - (chisq / df)) / ((chisq_b / df_b) - 1.0) if df > 0 else 1.0
    Ds = np.diag(1.0 / sd)
    res = Ds @ (S - Sigma) @ Ds
    srmr = float(np.sqrt((res[np.tril_indices(p)] ** 2).mean()))
    S_ml = S * (n - 1) / n                       # lavaan's log-likelihood uses the ML-scaled covariance
    loglik = -0.5 * n * (np.linalg.slogdet(Sigma)[1] + np.trace(S_ml @ np.linalg.inv(Sigma)) + p * np.log(2 * np.pi))
    bic = -2.0 * loglik + npar * np.log(n)
    return CFAFit(float(chisq), int(df), rmsea, float(cfi), float(tli), srmr, float(bic), bool(r.success), L, Phi, th)
