r"""Data-anchored NSA-Flow: put the data matrix in the fidelity, keep V as a basis.

The anchored solver in ``solve`` refines a *supplied* loading matrix ``X0``; the
data reaches it only through whatever estimator produced ``X0``.  This module
solves the variant in which the data appears directly,

    minimise   E_w(V) = (1 - w) ||X - X V V'||_F^2 / ||X||_F^2  +  w Dtilde(V)
    subject to V >= 0,

over ``V`` of shape ``[p, k]``.  Scores are *determined* by projection, ``U = XV``,
rather than being a second free block, so ``V`` remains the only variable: one
smooth objective on a convex set, the same spectral projected gradient, the same
gradient-mapping certificate, and an out-of-sample extension ``X_new V`` that a
transductive transform of ``X`` itself cannot provide.

The two terms are more aligned here than in the anchored form.  ``X V V'`` is an
orthogonal projection exactly when ``V'V = I``, so the reconstruction term already
prefers orthonormal columns, which is what ``D`` measures; at ``w = 0`` the
problem is non-negative-constrained PCA-subspace fitting.

Cost.  With ``S = X'X`` formed once, each iteration is ``O(p^2 k)`` -- independent
of ``n``.  Honest caveat: ``||X - X V V'||_F^2`` is quartic in ``V`` and not convex,
unlike the anchored ``||V - X0||_F^2``, so SPG converges to a stationary point
rather than to a global minimum, and the initialisation matters.
"""
import time

import torch

from .energy import stiefel_defect, stiefel_defect_normalised, effective_rank
from .project import project_nonneg
from .solve import NSAResult

__all__ = ["reconstruction_fidelity", "grad_reconstruction_fidelity", "nsa_flow_data",
           "relax_into_nonneg"]


def _gram_terms(V, S):
    A = V.transpose(-2, -1) @ S @ V          # V' X'X V
    B = V.transpose(-2, -1) @ V              # V' V
    return A, B


def reconstruction_fidelity(V, S, c, trS=None):
    r"""``||X - X V V'||_F^2 / ||X||_F^2`` from ``S = X'X`` and ``c = ||X||_F^2``.

    Expanded as ``tr S - 2 tr(V'SV) + tr(V'SV . V'V)``, so ``n`` never appears.
    """
    A, B = _gram_terms(V, S)
    t = S.diagonal(dim1=-2, dim2=-1).sum(-1) if trS is None else trS
    return (t - 2.0 * A.diagonal(dim1=-2, dim2=-1).sum(-1)
            + (A * B.transpose(-2, -1)).sum((-2, -1))) / c


def grad_reconstruction_fidelity(V, S, c):
    r"""``grad = (2/c) [ -2 S V + S V (V'V) + V (V'SV) ]``."""
    A, B = _gram_terms(V, S)
    SV = S @ V
    return (2.0 / c) * (-2.0 * SV + SV @ B + V @ A)


def _smooth_descent(V, obj, grad, max_iter, tol, sigma):
    """BB + Armijo on an unconstrained smooth objective; returns the iterate."""
    E = float(obj(V))
    g = grad(V)
    t = 1.0 / max(float(g.norm()), 1e-12)
    V_prev = g_prev = None
    for _ in range(max_iter):
        if V_prev is not None:
            s_ = V - V_prev
            r_ = g - g_prev
            sr = float((s_ * r_).sum())
            t = float((s_ * s_).sum()) / sr if sr > 0 else 1e12
            t = min(max(t, 1e-12), 1e12)
        ok = False
        for _ in range(60):
            V_new = V - t * g
            d2 = float((V_new - V).pow(2).sum())
            if float(obj(V_new)) <= E - sigma * d2 / t:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        if (d2 ** 0.5) / t <= tol:
            V = V_new
            break
        V_prev, g_prev = V, g
        V = V_new
        E = float(obj(V))
        g = grad(V)
    return V


def relax_into_nonneg(S, c, k, w, mus=None, max_iter=600, tol=1e-10, sigma=1e-4,
                      trS=None):
    r"""Penalty homotopy: follow the solution path from signed PCA into ``V >= 0``.

    Solves a sequence of *smooth unconstrained* problems

        E_w(V) + mu ||min(0, V)||_F^2,      mu = 0, mu_1, mu_2, ... increasing,

    warm-starting each from the last.  At ``mu = 0`` the ``w = 0`` minimiser is the
    signed PCA basis -- the global optimum of an easy problem -- and increasing
    ``mu`` deforms it continuously towards the feasible set.  The penalty is
    ``C^1`` (its gradient ``2 mu min(0, V)`` is continuous), so every subproblem is
    smooth and ordinary descent applies.

    This exists because the alternative -- mapping a signed basis to a
    non-negative one in one shot -- has no variational justification.  ``abs()``
    is not even the projection (for an entry ``-3`` it moves distance 6 where the
    projection moves 3) and it fabricates wrongly-signed mass; and no one-shot map
    can work in principle, because a signed component encodes contrast that a
    single non-negative component cannot represent.
    """
    from .energy import grad_stiefel_defect, stiefel_defect_normalised
    p = S.shape[-1]
    inv_k = 1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0
    if mus is None:
        mus = [0.0] + [10.0 ** e for e in range(-3, 5)]

    evals, evecs = torch.linalg.eigh(S)
    V = evecs[:, -k:].flip(-1).clone()          # signed, NOT abs: the mu=0 optimum

    def make(mu):
        def obj(Vv):
            e = (1.0 - w) * reconstruction_fidelity(Vv, S, c, trS)
            if k > 1 and w != 0.0:
                e = e + w * stiefel_defect_normalised(Vv)
            if mu:
                e = e + mu * Vv.clamp_max(0.0).pow(2).sum()
            return e

        def grad(Vv):
            g = (1.0 - w) * grad_reconstruction_fidelity(Vv, S, c)
            if k > 1 and w != 0.0:
                g = g + (w * inv_k) * grad_stiefel_defect(Vv)
            if mu:
                g = g + 2.0 * mu * Vv.clamp_max(0.0)
            return g
        return obj, grad

    for mu in mus:
        obj, grad = make(mu)
        V = _smooth_descent(V, obj, grad, max_iter, tol, sigma)
    return V


def nsa_flow_data(X, k=None, w=0.5, *, init="relax", max_iter=5000, tol=None,
                  sigma=1e-4, dtype=None, device=None, verbose=False,
                  keep_trace=False):
    """Fit a non-negative, near-orthonormal basis ``V`` reconstructing ``X``.

    Parameters
    ----------
    X : array-like ``[n, p]``
        The data matrix itself.  Only ``X'X`` is used.
    k : int
        Number of components.  Required unless ``init`` is given.
    w : float in ``[0, 1]``
        ``w = 0`` fits the reconstruction alone (non-negative PCA-subspace
        fitting); ``w = 1`` ignores the data.
    """
    Xt = torch.as_tensor(X)
    if not torch.is_floating_point(Xt):
        Xt = Xt.double()
    if dtype is not None:
        Xt = Xt.to(dtype)
    if device is not None:
        Xt = Xt.to(device)
    Xt = Xt.detach()
    if Xt.ndim != 2:
        raise ValueError(f"X must be 2-D [n, p]; got shape {tuple(Xt.shape)}")
    if not torch.isfinite(Xt).all():
        raise ValueError("X contains non-finite values")
    if not (0.0 <= float(w) <= 1.0):
        raise ValueError(f"w must lie in [0, 1]; got {w}")

    n, p = Xt.shape
    S = Xt.transpose(-2, -1) @ Xt
    c = S.diagonal().sum()                    # ||X||_F^2
    if float(c) <= 0:
        raise ValueError("X is all zeros; fidelity is undefined")
    trS = c
    if tol is None:
        tol = 1e-9 if Xt.dtype == torch.float64 else 1e-6

    if isinstance(init, str):
        if k is None:
            raise ValueError("give k when init is a strategy name")
        evals, evecs = torch.linalg.eigh(S)
        E = evecs[:, -k:].flip(-1)
        if init == "relax":
            # Follow the path from signed PCA into the feasible set.
            V = relax_into_nonneg(S, c, k, float(w), trS=trS)
        elif init == "clamp":
            V = E.clamp_min(0.0).clone()        # the actual projection
        elif init == "abs":
            V = E.abs().clone()                 # kept only for the ablation
        elif init == "random":
            V = torch.rand(p, k, dtype=Xt.dtype, device=Xt.device)
        else:
            raise ValueError(f"unknown init strategy {init!r}")
    elif init is None:
        raise ValueError("give either k with an init strategy, or an explicit init")
    else:
        V = torch.as_tensor(init).to(dtype=Xt.dtype, device=Xt.device).detach().clone()
        k = V.shape[-1]
    if V.shape != (p, k):
        raise ValueError(f"init shape {tuple(V.shape)} != [p, k] = {(p, k)}")

    inv_k = 1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0

    def energy_of(Vv):
        f = reconstruction_fidelity(Vv, S, c, trS)
        d = stiefel_defect_normalised(Vv)
        return (1.0 - w) * f + w * d, f, d

    def grad_of(Vv):
        from .energy import grad_stiefel_defect
        g = (1.0 - w) * grad_reconstruction_fidelity(Vv, S, c)
        if k > 1 and w != 0.0:
            g = g + (w * inv_k) * grad_stiefel_defect(Vv)
        return g

    V = project_nonneg(V)
    E, F, D = energy_of(V)
    E = float(E)
    g = grad_of(V)
    t = 1.0 / max(float(g.norm()), 1e-12)
    V_prev = g_prev = None
    trace = [] if keep_trace else None
    gmap, stop, it = float("inf"), "max_iter", 0
    t0 = time.time()

    for it in range(1, max_iter + 1):
        if V_prev is not None:
            s_ = V - V_prev
            r_ = g - g_prev
            sr = float((s_ * r_).sum())
            t = float((s_ * s_).sum()) / sr if sr > 0 else 1e12
            t = min(max(t, 1e-12), 1e12)
        accepted = False
        for _ in range(60):
            V_new = project_nonneg(V - t * g)
            d_ = V_new - V
            dn2 = float((d_ * d_).sum())
            E_new = float(energy_of(V_new)[0])
            if E_new <= E - sigma * dn2 / t:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            stop = "line_search"
            break
        gmap = (dn2 ** 0.5) / t
        V_prev, g_prev = V, g
        V = V_new
        E, F, D = energy_of(V)
        E = float(E)
        g = grad_of(V)
        if trace is not None:
            trace.append(dict(iter=it, energy=E, fidelity=float(F),
                              defect=float(D), grad_map=gmap, step=t))
        if verbose and (it % max(1, max_iter // 10) == 0 or it == 1):
            print(f"    [w={w:.3f} it={it:5d}] E={E:.8e} |Gmap|={gmap:.3e}")
        if gmap <= tol:
            stop = "grad_map"
            break

    return NSAResult(
        Y=V, target=None, w=float(w), energy=E, fidelity=float(F),
        defect=float(D), raw_defect=float(stiefel_defect(V)),
        effective_rank=float(effective_rank(V)),
        scale_ratio=float("nan"), iters=it, converged=stop != "max_iter",
        stop_reason=stop, grad_map=float(gmap), seconds=time.time() - t0,
        w_schedule=[float(w)], trace=trace, nonneg=True, align=False,
    )
