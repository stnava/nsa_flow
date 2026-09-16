"""NSA-Flow solver: spectral projected gradient on a single, calibration-free energy.

The problem solved is

    minimise   E_w(Y) = (1 - w) ||Y - X0||_F^2 / ||X0||_F^2  +  w Dtilde(Y)
    subject to Y >= 0                                        (when ``nonneg``)

with ``Dtilde = D / (1 - 1/k)`` and ``D(Y) = ||G - I/k||_F^2`` for the
trace-normalised Gram matrix ``G = Y'Y / tr(Y'Y)``.

The method is Spectral Projected Gradient (Birgin, Martinez & Raydan 2000):
Barzilai-Borwein step lengths safeguarded by an Armijo backtracking line search
on the projected-gradient step.  For continuously differentiable ``E_w`` and
closed convex feasible set this converges to a stationary point, and the
gradient-mapping norm ``||Y+ - Y|| / t`` is a computable stationarity
certificate.

Cost per iteration is two ``[p,k] x [k,k]`` products plus one Gram: ``O(p k^2)``.
No SVD, eigendecomposition or QR appears in the loop.
"""
import time
import warnings

import torch

from .energy import energy, stiefel_defect, effective_rank, value_and_grad
from .project import project_nonneg

__all__ = ["nsa_flow", "NSAResult"]

_T_MIN, _T_MAX = 1e-12, 1e12
_COMPILED = None


def _fused(use_compile):
    """Return ``value_and_grad``, optionally inductor-compiled (cached)."""
    global _COMPILED
    if not use_compile:
        return value_and_grad
    if _COMPILED is None:
        _COMPILED = torch.compile(value_and_grad, dynamic=True)
    return _COMPILED


class NSAResult(dict):
    """Solver output; a dict with attribute access."""

    __getattr__ = dict.__getitem__

    def __repr__(self):
        return (f"NSAResult(w={self['w']}, iters={self['iters']}, "
                f"energy={self['energy']:.6e}, fidelity={self['fidelity']:.6e}, "
                f"defect={self['defect']:.6e}, eff_rank={self['effective_rank']:.3f}, "
                f"stop={self['stop_reason']}, |Gmap|={self['grad_map']:.2e})")


def _solve_fixed_w(Y, X0, w, denom, nonneg, max_iter, tol, sigma, verbose, trace,
                   vg=value_and_grad):
    """Monotone spectral projected gradient for a single value of ``w``."""
    proj = project_nonneg if nonneg else (lambda A: A)
    k = Y.shape[-1]
    eye_k = torch.eye(k, dtype=Y.dtype, device=Y.device)
    inv_k = 1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0

    Y = proj(Y)
    E, F, Dn, g = vg(Y, X0, w, denom, inv_k, eye_k)
    E = float(E)
    t = 1.0 / max(float(g.norm()), 1e-12)          # scale-free first guess
    Y_prev = g_prev = None
    it = 0
    gmap = float("inf")
    stop = "max_iter"

    for it in range(1, max_iter + 1):
        if Y_prev is not None:                      # Barzilai-Borwein
            s_ = Y - Y_prev
            r_ = g - g_prev
            sr = float((s_ * r_).sum())
            t = float((s_ * s_).sum()) / sr if sr > 0 else _T_MAX
            t = min(max(t, _T_MIN), _T_MAX)

        accepted = False
        for _ in range(60):                         # Armijo backtracking
            Y_new = proj(Y - t * g)
            d_ = Y_new - Y
            dn2 = float((d_ * d_).sum())
            E_new = float(energy(Y_new, X0, w=w, denom=denom))
            if E_new <= E - sigma * dn2 / t:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            # No feasible descent step exists to within working precision.  That
            # is convergence, not failure; ``grad_map`` certifies how stationary.
            stop = "line_search"
            break

        gmap = (dn2 ** 0.5) / t                     # ||Y+ - Y|| / t
        Y_prev, g_prev = Y, g
        Y = Y_new
        E, F, Dn, g = vg(Y, X0, w, denom, inv_k, eye_k)
        E = float(E)

        if trace is not None:
            trace.append(dict(iter=len(trace) + 1, w=float(w), energy=E,
                              fidelity=float(F), defect=float(Dn),
                              grad_map=gmap, step=t))
        if verbose and (it % max(1, max_iter // 10) == 0 or it == 1):
            print(f"    [w={w:.3f} it={it:5d}] E={E:.8e} |Gmap|={gmap:.3e} t={t:.3e}")
        if gmap <= tol:
            stop = "grad_map"
            break

    return Y, E, it, stop, gmap


def nsa_flow(target, w=0.5, *, init=None, nonneg=True, max_iter=5000, tol=None,
             continuation=0, w_start=0.0, sigma=1e-4, dtype=None, device=None,
             verbose=False, keep_trace=False, compile=False):
    """Fit a non-negative, near-orthogonal ``Y`` close to ``target``.

    Parameters
    ----------
    target : array-like ``[p, k]``
        ``X0``, the matrix ``Y`` should stay close to (e.g. PCA loadings).
    w : float in ``[0, 1]``
        Convex weight.  ``w = 0`` returns ``max(0, X0)``; ``w = 1`` ignores the
        data and seeks orthogonal columns (which, under ``nonneg``, means
        disjoint supports).  ``w`` needs no calibration: both energy terms are
        dimensionless and ``O(1)``.
    init : array-like, optional
        Starting point; defaults to ``target``.
    nonneg : bool
        Enforce ``Y >= 0`` by projection.
    continuation : int
        If ``> 0``, solve at ``continuation + 1`` values of ``w`` increasing from
        ``w_start`` to ``w``, warm-starting each from the last, and (with
        ``keep_trace``) record the whole path.  This is a *diagnostic*: across
        every problem family tested, random restarts and cold starts reach the
        same optimum for ``w < 1``, so continuation is not needed to find it.
    compile : bool
        Compile the fused value-and-gradient kernel with ``torch.compile``.
        Worth 3-4x at moderate sizes; costs a few seconds on first call.
    tol : float, optional
        Stop when the projected-gradient mapping norm falls below this.  The
        default is derived from the working precision -- ``1e-9`` in float64 and
        ``1e-6`` in float32 -- because a fixed ``1e-9`` is below float32 machine
        epsilon (``1.2e-7``) and so can never be met, which would silently turn
        every single-precision call into a ``max_iter`` run.

    Notes
    -----
    Conditioning degrades as ``w -> 1``, where the problem approaches the
    combinatorial limit: convergence takes tens of iterations at ``w = 0.5``,
    hundreds at ``w = 0.9``, and thousands or more beyond ``w = 0.99``.  Raise
    ``max_iter`` accordingly and check ``stop_reason``, which reports
    ``"max_iter"`` rather than claiming convergence.

    Returns
    -------
    NSAResult
    """
    X0 = torch.as_tensor(target)
    if not torch.is_floating_point(X0):
        X0 = X0.double()
    if dtype is not None:
        X0 = X0.to(dtype)
    if device is not None:
        X0 = X0.to(device)
    X0 = X0.detach()
    if X0.ndim != 2:
        raise ValueError(f"target must be 2-D [p, k]; got shape {tuple(X0.shape)}")
    if not (0.0 <= float(w) <= 1.0):
        raise ValueError(f"w must lie in [0, 1]; got {w}")
    if float(w) == 1.0:
        warnings.warn(
            "w=1 drops the fidelity term, and D is scale-invariant, so the scale "
            "of the returned Y is unconstrained and arbitrary (observed "
            "||Y||/||X0|| up to ~1e4). Every scaled Stiefel matrix is optimal, so "
            "nothing selects among clusterings either. Use w slightly below 1, or "
            "rescale the result yourself; result['scale_ratio'] reports the drift.",
            RuntimeWarning, stacklevel=2)
    if not torch.isfinite(X0).all():
        raise ValueError("target contains non-finite values")

    if tol is None:
        tol = 1e-9 if X0.dtype == torch.float64 else 1e-6

    p, k = X0.shape
    denom = X0.pow(2).sum()
    if float(denom) <= 0:
        raise ValueError("target is all zeros; fidelity is undefined")

    Y = (X0.clone() if init is None
         else torch.as_tensor(init).to(dtype=X0.dtype, device=X0.device).detach().clone())
    if Y.shape != X0.shape:
        raise ValueError(f"init shape {tuple(Y.shape)} != target shape {tuple(X0.shape)}")

    ws = ([float(w)] if continuation <= 0 else
          torch.linspace(float(w_start), float(w), continuation + 1).tolist())
    vg = _fused(compile)
    trace = [] if keep_trace else None
    t0 = time.time()
    total_iters, stop, gmap = 0, "max_iter", float("inf")
    for wi in ws:
        if verbose:
            print(f"  continuation step w={wi:.4f}")
        Y, E, it, stop, gmap = _solve_fixed_w(Y, X0, wi, denom, nonneg, max_iter,
                                              tol, sigma, verbose, trace, vg)
        total_iters += it

    tot, f, dd = energy(Y, X0, w=float(w), denom=denom, return_parts=True)
    return NSAResult(
        Y=Y, target=X0, w=float(w), energy=float(tot), fidelity=float(f),
        defect=float(dd), raw_defect=float(stiefel_defect(Y)),
        effective_rank=float(effective_rank(Y)),
        scale_ratio=float(Y.norm() / X0.norm()), iters=total_iters,
        converged=stop != "max_iter", stop_reason=stop, grad_map=float(gmap),
        seconds=time.time() - t0,
        w_schedule=ws, trace=trace, nonneg=bool(nonneg),
    )
