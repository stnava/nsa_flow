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

Cost.  Two routes to the same numbers, chosen by shape:

* ``S = X'X`` formed once, then ``O(p^2 k)`` per iteration, independent of ``n``.
* matrix-free, from ``X V`` (``n x k``) and ``X'(X V)`` (``p x k``), then ``O(n p k)``
  per iteration and ``S`` never formed.

Per-iteration work is ``n p k`` against ``p^2 k``, so matrix-free wins exactly when
``n < p``, which is also where forming ``S`` is unaffordable: it is 0.4 GB at
``p = 7129`` and 3.2 GB at ``p = 20000``.  Before the matrix-free route existed this
module could not complete a ``p = 2000`` fit in three minutes; it now does
``p = 7129`` in a few seconds.  ``matrix_free=None`` picks by shape.

Honest caveat: ``||X - X V V'||_F^2`` is quartic in ``V`` and not convex, unlike the
anchored ``||V - X0||_F^2``, so SPG converges to a stationary point rather than to
a global minimum, and the relaxation path matters (see ``relax_into_nonneg``).
"""
import time

import math
import warnings
import torch

from .angle import angle_defect, grad_angle_defect
from .energy import stiefel_defect, stiefel_defect_normalised, effective_rank
from .project import project_nonneg
from .solve import NSAResult

__all__ = ["reconstruction_fidelity", "grad_reconstruction_fidelity", "nsa_flow_data",
           "relax_into_nonneg", "GramOperator"]


class GramOperator:
    r"""The action of ``S = X'X`` on ``V``, from either ``S`` or ``X``.

    Supplying ``X`` avoids forming ``S`` at all.  ``quad`` returns only ``V'SV``,
    which the matrix-free route gets from ``X V`` alone without the ``X'`` product;
    the line search evaluates the objective many times per accepted step, so that
    saving is worth having.
    """

    __slots__ = ("S", "X", "p", "c", "trS")

    def __init__(self, S=None, X=None):
        if (S is None) == (X is None):
            raise ValueError("give exactly one of S or X")
        self.S, self.X = S, X
        if S is not None:
            self.p = S.shape[-1]
            self.trS = S.diagonal(dim1=-2, dim2=-1).sum(-1)
        else:
            self.p = X.shape[-1]
            self.trS = X.pow(2).sum((-2, -1))
        self.c = self.trS                      # ||X||_F^2 = tr(X'X)

    @property
    def matrix_free(self):
        return self.X is not None

    def quad(self, V):
        """``V'SV`` only."""
        if self.X is not None:
            XV = self.X @ V
            return XV.transpose(-2, -1) @ XV
        return V.transpose(-2, -1) @ (self.S @ V)

    def both(self, V):
        """``(SV, V'SV)``."""
        if self.X is not None:
            XV = self.X @ V
            return self.X.transpose(-2, -1) @ XV, XV.transpose(-2, -1) @ XV
        SV = self.S @ V
        return SV, V.transpose(-2, -1) @ SV

    def leading(self, k):
        """Signed leading ``k`` eigenvectors of ``S`` -- the ``mu = 0`` optimum."""
        if self.X is not None:
            _, _, Vh = torch.linalg.svd(self.X, full_matrices=False)
            return Vh[:k].transpose(-2, -1).clone()
        _, evecs = torch.linalg.eigh(self.S)
        return evecs[..., -k:].flip(-1).clone()


def _as_ops(S_or_ops):
    return S_or_ops if isinstance(S_or_ops, GramOperator) else GramOperator(S=S_or_ops)


def _fid(V, ops, c, trS=None):
    A = ops.quad(V)
    B = V.transpose(-2, -1) @ V
    t = ops.trS if trS is None else trS
    return (t - 2.0 * A.diagonal(dim1=-2, dim2=-1).sum(-1)
            + (A * B.transpose(-2, -1)).sum((-2, -1))) / c


def _grad_fid(V, ops, c):
    SV, A = ops.both(V)
    B = V.transpose(-2, -1) @ V
    return (2.0 / c) * (-2.0 * SV + SV @ B + V @ A)


def reconstruction_fidelity(V, S, c, trS=None):
    r"""``||X - X V V'||_F^2 / ||X||_F^2``, expanded as ``tr S - 2 tr(V'SV) + tr(V'SV . V'V)``.

    ``S`` may be the ``p x p`` Gram matrix or a :class:`GramOperator` wrapping ``X``.
    """
    return _fid(V, _as_ops(S), c, trS)


def grad_reconstruction_fidelity(V, S, c):
    r"""``grad = (2/c) [ -2 S V + S V (V'V) + V (V'SV) ]``."""
    return _grad_fid(V, _as_ops(S), c)


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
    ops = _as_ops(S)
    inv_k = 1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0
    if mus is None:
        # Continuation in mu, which is NOT the `continuation` argument of
        # nsa_flow (that one steps in w and is only a diagnostic).
        # Nine stages, and the resolution of this path is load-bearing: on ADNI a
        # three-stage path yields basis reproducibility 0.708 at 150 iterations
        # and 0.725 at 3000, while this one yields 0.975.  Refining the path is
        # worth 0.25; converging twenty times harder on a coarse path is worth
        # 0.02.  Do not trim it for speed.
        mus = [0.0] + [10.0 ** e for e in range(-3, 5)]

    V = ops.leading(k)                          # signed, NOT abs: the mu=0 optimum

    def make(mu):
        def obj(Vv):
            e = (1.0 - w) * _fid(Vv, ops, c, trS)
            if k > 1 and w != 0.0:
                e = e + w * stiefel_defect_normalised(Vv)
            if mu:
                e = e + mu * Vv.clamp_max(0.0).pow(2).sum()
            return e

        def grad(Vv):
            g = (1.0 - w) * _grad_fid(Vv, ops, c)
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


def _orth_terms(orth, k):
    """Return (value, grad) for the chosen orthogonality functional."""
    if orth == "D":                      # orthoNORMality: ||G - I/k||^2, scaled
        from .energy import grad_stiefel_defect
        inv = 1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0
        return (lambda V: stiefel_defect_normalised(V),
                lambda V: inv * grad_stiefel_defect(V))
    if orth == "C":                      # orthogonality only: mean cos^2
        return angle_defect, grad_angle_defect
    if orth == "Cg":                     # smooth orthogonality; see nsa_flow.angle
        from .angle import gram_offdiag_defect, grad_gram_offdiag_defect
        return gram_offdiag_defect, grad_gram_offdiag_defect
    raise ValueError(f"orth must be 'D', 'C' or 'Cg'; got {orth!r}")


def nsa_flow_data(X, k=None, w=0.5, *, init="relax", orth="C", max_iter=5000,
                  tol=None, sigma=1e-4, dtype=None, device=None, verbose=False,
                  keep_trace=False, matrix_free=None):
    """Fit a non-negative, near-orthonormal basis ``V`` reconstructing ``X``.

    Parameters
    ----------
    X : array-like ``[n, p]``
        The data matrix itself.  Only ``X'X`` is used.
    k : int
        Number of components.  Required unless ``init`` is given.
    matrix_free : bool, optional
        Work from ``X`` without forming ``S = X'X``.  The default picks by shape
        (``p > n``), which is both the cost crossover -- ``n p k`` against
        ``p^2 k`` per iteration -- and the point past which ``S`` stops fitting in
        memory (0.4 GB at ``p = 7129``, 3.2 GB at ``p = 20000``).
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
    if matrix_free is None:
        # per-iteration work is n p k against p^2 k, so this is the exact
        # crossover; it is also where forming S becomes unaffordable
        matrix_free = p > n
    ops = GramOperator(X=Xt) if matrix_free else GramOperator(S=Xt.transpose(-2, -1) @ Xt)
    c = ops.c                                 # ||X||_F^2
    if float(c) <= 0:
        raise ValueError("X is all zeros; fidelity is undefined")
    trS = ops.trS
    if tol is None:
        tol = 1e-9 if Xt.dtype == torch.float64 else 1e-6

    if isinstance(init, str):
        if k is None:
            raise ValueError("give k when init is a strategy name")
        E = ops.leading(k)
        if init == "relax":
            # Follow the path from signed PCA into the feasible set.
            V = relax_into_nonneg(ops, c, k, float(w), trS=trS)
        elif init == "clamp":
            V = E.clamp_min(0.0).clone()        # the actual projection
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

    orth_val, orth_grad = _orth_terms(orth, k)

    def energy_of(Vv):
        f = _fid(Vv, ops, c, trS)
        d = orth_val(Vv)
        return (1.0 - w) * f + w * d, f, d

    def grad_of(Vv):
        g = (1.0 - w) * _grad_fid(Vv, ops, c)
        if k > 1 and w != 0.0:
            g = g + w * orth_grad(Vv)
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
        t_first, dn2_first = t, None
        for _ in range(60):
            V_new = project_nonneg(V - t * g)
            d_ = V_new - V
            dn2 = float((d_ * d_).sum())
            if dn2_first is None:
                dn2_first = dn2
            E_new = float(energy_of(V_new)[0])
            if E_new <= E - sigma * dn2 / t:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            stop = "line_search"
            if not math.isfinite(gmap):
                gmap = (dn2_first ** 0.5) / t_first
            # A finite but large certificate is not stationarity.  The caller
            # cannot be expected to inspect grad_map on every call, so say so.
            if gmap > max(tol, 0.0) * 1e3:
                warnings.warn(
                    "nsa_flow_data: line search stalled after "
                    f"{it} iteration(s) with |Gmap|={gmap:.2e} against "
                    f"tol={tol:.1e}; the returned point is not stationary. "
                    "Inspect stop_reason and grad_map.",
                    RuntimeWarning, stacklevel=3)
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
        angle_defect=float(angle_defect(V)), orth=orth,
        matrix_free=bool(matrix_free),
        effective_rank=float(effective_rank(V)),
        scale_ratio=float("nan"), iters=it,
        converged=stop != "max_iter" and math.isfinite(gmap),
        stop_reason=stop, grad_map=float(gmap), seconds=time.time() - t0,
        w_schedule=[float(w)], trace=trace, nonneg=True, align=False,
    )
