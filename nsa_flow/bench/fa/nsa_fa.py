"""NSA.FA: principal-axis factoring whose loadings are regularised toward orthogonal columns by NSA-Flow.

The pipeline is written from scratch for this benchmark (inspired by, not a port of, earlier R experiments).
Each PAF iteration: put the current communalities on the diagonal of R, take the leading eigen-loadings, refine
them with ``nsa_flow(L, w, mode="anchored", nonneg=False)`` (signed, anchored to the eigen-loadings, pulled toward
orthogonal columns), optionally rotate (varimax), then update the communalities with damping.  ``w = 0.5`` is the
default.  Only the public ``nsa_flow`` API is used.

``nsa_fa_post`` regularises once, after ordinary PAF.
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
