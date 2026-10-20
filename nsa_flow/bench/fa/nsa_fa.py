"""NSA for factor analysis: three ways to use the NSA-Flow prox, written for this benchmark (public API only).

``nsa_rotation`` (the sensible one): NSA as a fit-preserving ROTATION.  Fit PAF, then ask NSA for a simple-structure
basis of the SAME subspace (``fidelity="subspace"``: only range(L) is anchored, so any rotation is equally faithful) and
recover the factor correlations by least squares from the reduced correlation matrix.  The model-implied correlation
(and so the fit) is that of PAF by construction; what changes is how interpretable the loadings are, to be compared
with varimax / promax / oblimin.  With ``keys`` (+1/-1 per item) the items are reverse-keyed first and a non-negative
anchored subspace is requested ("keyed non-negative": independent-cluster structure, trading some fit for simplicity).
Note: with subspace fidelity the result did not depend on ``w`` in exploratory runs.

``nsa_fa`` (legacy, kept as a secondary row): PAF whose loadings are shrunk toward orthogonal columns at every
iteration with the signed anchored mode.  It is dominated by oblique rotation (higher fit error, no simpler structure),
see docs/BENCHMARKING.md.  ``nsa_fa_post`` regularises once, after ordinary PAF.
"""
from __future__ import annotations

import numpy as np

from .paf import FAResult, _eig_top, orient, paf, smc, varimax


def _regularise(L, w):
    """Signed anchored NSA-Flow refinement of a loading matrix; returns (Y, certificate)."""
    from nsa_flow import nsa_flow
    r = nsa_flow(np.ascontiguousarray(L, dtype=np.float64), w=w, mode="anchored", nonneg=False)
    return r.Y.detach().cpu().numpy(), {"converged": bool(r["converged"]), "defect_D": float(r["defect_D"])}


def nsa_fa(R, k, w=0.5, rotate=True, max_iter=200, eta_h=0.5, tol=1e-5) -> FAResult:
    p = R.shape[0]
    h2 = smc(R)
    off = ~np.eye(p, dtype=bool)
    best, best_e, cert, e_prev, converged = None, np.inf, {}, np.inf, False
    for it in range(1, max_iter + 1):
        Rm = R.copy()
        np.fill_diagonal(Rm, h2)
        vals, vecs = _eig_top(Rm, k)
        L0 = vecs * np.sqrt(np.clip(vals, 0, None))
        Y, cert = _regularise(L0, w) if w > 0 else (L0, {"converged": True, "defect_D": float("nan")})
        if rotate:
            Y = varimax(Y)[0]
        Y = orient(Y)
        e = float(np.linalg.norm((R - Y @ Y.T)[off]) / np.linalg.norm(R[off]))
        if e < best_e:
            best, best_e = Y, e
        new_h2 = np.clip((Y ** 2).sum(1), 0.0, 1.0)
        dh = float(np.abs(new_h2 - h2).max())
        h2 = (1 - eta_h) * h2 + eta_h * new_h2
        if it > 5 and abs(e - e_prev) < tol and dh < 1e-3:
            converged = True
            break
        e_prev = e
    return FAResult(best, (best ** 2).sum(1), it, converged,
                    {**cert, "converged": bool(cert.get("converged", True)) and converged})


def nsa_fa_post(R, k, w=0.5, rotate=True, **paf_kw) -> FAResult:
    base = paf(R, k, **paf_kw)
    Y, cert = _regularise(base.loadings, w) if w > 0 else (base.loadings, {"converged": True})
    if rotate:
        Y = varimax(Y)[0]
    Y = orient(Y)
    return FAResult(Y, (Y ** 2).sum(1), base.iterations, base.converged, cert)


def phi_from(Y, Rh):
    """Pattern ``Y`` rescaled to unit-variance factors and the factor correlations that best reproduce the reduced
    correlation matrix ``Rh`` as ``Y Phi Y'`` (least squares)."""
    Yp = np.linalg.pinv(Y)
    Phi = Yp @ Rh @ Yp.T
    Phi = (Phi + Phi.T) / 2
    d = np.sqrt(np.clip(np.diag(Phi), 1e-8, None))
    return Y * d, Phi / np.outer(d, d)


def nsa_rotation(R, k, w=0.5, keys=None, nonneg=False, **paf_kw) -> FAResult:
    """Fit-preserving NSA rotation of a PAF solution (see module docstring).  ``nonneg`` requires ``keys``."""
    from nsa_flow import nsa_flow
    base = paf(R, k, **paf_kw)
    Rh = R.copy()
    np.fill_diagonal(Rh, base.communality)
    if nonneg:
        if keys is None:
            raise ValueError("keyed non-negative rotation needs item keys (+1/-1 per item)")
        D = np.diag(np.asarray(keys, float))
        L0, Rh = orient(D @ base.loadings), D @ Rh @ D
    else:
        D, L0 = None, base.loadings
    r = nsa_flow(np.ascontiguousarray(L0, dtype=np.float64), w=w, mode="anchored", nonneg=nonneg,
                 fidelity="subspace")
    Y = r.Y.detach().cpu().numpy()
    Y, Phi = phi_from(orient(Y) if not nonneg else Y, Rh)
    if D is not None:
        Y = D @ Y                                   # back to the original item signs
    comm = np.einsum("ik,kl,il->i", Y, Phi, Y)
    cert = {"converged": bool(r["converged"]), "defect_D": float(r["defect_D"])}
    return FAResult(Y, comm, base.iterations, base.converged, cert, Phi)
