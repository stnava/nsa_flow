"""NSA-Flow solver: spectral projected gradient on a single, calibration-free energy.

The problem solved is

    minimise   E_w(Y) = (1 - w) ||Y - X0||_F^2 / ||X0||_F^2  +  w Dtilde(Y)
    subject to Y >= 0                                        (when ``nonneg``)

with ``Dtilde = D / (1 - 1/k)`` and ``D(Y) = ||G - I/k||_F^2`` for the
trace-normalised Gram matrix ``G = Y'Y / tr(Y'Y)``.  With ``align`` the fidelity
numerator becomes ``min_{Q in O(k)} ||Y - X0 Q||_F^2``, a distance to ``X0``'s
right-``O(k)`` orbit rather than to the point ``X0``.

The method is Spectral Projected Gradient (Birgin, Martinez & Raydan 2000):
Barzilai-Borwein step lengths safeguarded by an Armijo backtracking line search
on the projected-gradient step.  For continuously differentiable ``E_w`` and
closed convex feasible set this converges to a stationary point, and the
gradient-mapping norm ``||Y+ - Y|| / t`` is a computable stationarity
certificate.

Cost per iteration is two ``[p,k] x [k,k]`` products plus one Gram: ``O(p k^2)``.
No SVD, eigendecomposition or QR appears in the loop -- except under ``align``,
which adds one ``k x k`` SVD per evaluation for the Procrustes rotation.
"""
import time
import warnings

import math
import warnings

import torch

from .energy import (energy, stiefel_defect, stiefel_defect_normalised,
                     grad_stiefel_defect, effective_rank, value_and_grad)
from .project import project_nonneg
from .subspace import (SubspaceAnchor, negative_mass, subspace_fidelity,
                       grad_subspace_fidelity)

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


def _subspace_vg(anchor, inv_k):
    """value_and_grad with the sign-blind fidelity in place of the anchored one."""
    def vg(Y, X0, w, denom, inv_k_, eye_k, align):
        F = subspace_fidelity(Y, anchor)
        k = Y.shape[-1]
        Dn = (stiefel_defect_normalised(Y) if k > 1
              else torch.zeros_like(F))
        E = (1.0 - w) * F + w * Dn
        g = (1.0 - w) * grad_subspace_fidelity(Y, anchor)
        if k > 1 and w != 0.0:
            g = g + (w * inv_k) * grad_stiefel_defect(Y)
        return E, F, Dn, g
    return vg


def _solve_fixed_w(Y, X0, w, denom, nonneg, max_iter, tol, sigma, verbose, trace,
                   vg=value_and_grad, align=False):
    """Monotone spectral projected gradient for a single value of ``w``."""
    proj = project_nonneg if nonneg else (lambda A: A)
    k = Y.shape[-1]
    eye_k = torch.eye(k, dtype=Y.dtype, device=Y.device)
    inv_k = 1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0

    Y = proj(Y)
    E, F, Dn, g = vg(Y, X0, w, denom, inv_k, eye_k, align)
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
        t_first, dn2_first = t, None
        for _ in range(60):                         # Armijo backtracking
            Y_new = proj(Y - t * g)
            d_ = Y_new - Y
            dn2 = float((d_ * d_).sum())
            if dn2_first is None:
                dn2_first = dn2
            E_new = float(vg(Y_new, X0, w, denom, inv_k, eye_k, align)[0])
            if E_new <= E - sigma * dn2 / t:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            # No feasible descent step exists to within working precision.  That
            # is convergence ONLY IF grad_map says so -- it is also what a
            # stalled start looks like, so measure the certificate rather than
            # leaving the sentinel, and warn when it is far from stationary.
            stop = "line_search"
            if not math.isfinite(gmap):
                gmap = (dn2_first ** 0.5) / t_first
            # A finite but large certificate is not stationarity.  The caller
            # cannot be expected to inspect grad_map on every call, so say so.
            if gmap > max(tol, 0.0) * 1e3:
                warnings.warn(
                    "nsa_flow: line search stalled after "
                    f"{it} iteration(s) with |Gmap|={gmap:.2e} against "
                    f"tol={tol:.1e}; the returned point is not stationary. "
                    "Inspect stop_reason and grad_map.",
                    RuntimeWarning, stacklevel=3)
            break

        gmap = (dn2 ** 0.5) / t                     # ||Y+ - Y|| / t
        Y_prev, g_prev = Y, g
        Y = Y_new
        E, F, Dn, g = vg(Y, X0, w, denom, inv_k, eye_k, align)
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
             verbose=False, keep_trace=False, compile=False, align=False,
             fidelity="auto", neg_mass_tol=0.01):
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
        same optimum for ``w < 1``, so continuation in ``w`` is not needed to
        find it.

        Not to be confused with the continuation in ``mu`` performed by
        ``relax_into_nonneg``, which follows the path from a signed basis into
        the non-negative cone.  That one *is* load-bearing: the stationary point
        reached depends on how finely it is resolved, and a three-stage path
        gives a materially worse answer than the nine-stage default however many
        iterations each stage is given.  The two are independent; this argument
        does nothing for the other.
    fidelity : {"auto", "anchor", "subspace"}
        Which notion of "close to ``target``" to use.

        ``"anchor"`` is ``||Y - X0||_F^2 / ||X0||_F^2``, the entrywise distance.  It
        is the right choice when the target is itself non-negative.

        ``"subspace"`` is ``||(I-P)Y||_F^2 / ||Y||_F^2`` for ``P`` the projector onto
        ``range(X0)`` -- a *sign-blind* fidelity, invariant under
        ``X0 -> X0 M`` for any invertible ``M``.  Use it when the target is signed.
        The anchored distance charges the solution for negative entries a
        non-negative ``Y`` cannot reach; those charges are constant, so they do not
        steer the solution and the optimum degenerates toward ``max(0, X0)``.  For
        PCA input the entrywise target is doubly inappropriate, since eigenvector
        signs are an arbitrary convention.  Because both this term and the defect
        are scale-free, the result is rescaled to ``||X0||_F`` to fix the gauge.

        ``"auto"`` (default) chooses ``"subspace"`` when ``nonneg`` is set and the
        target's negative mass ``||min(0,X0)||_F / ||X0||_F`` exceeds
        ``neg_mass_tol``, and warns when it does so.  The decision is reported as
        ``result["fidelity_mode"]``, and the measured negative mass as
        ``result["target_negative_mass"]``, so it is never silent.
    neg_mass_tol : float
        Threshold on the target's negative mass for ``fidelity="auto"``.
    align : bool
        Anchor to ``X0``'s right-``O(k)`` orbit rather than to ``X0`` itself, by
        replacing ``||Y - X0||_F^2`` with ``min_{Q in O(k)} ||Y - X0 Q||_F^2``
        (orthogonal Procrustes, closed form).  Appropriate when only the *span*
        of ``X0`` is trustworthy: ``D`` is exactly right-``O(k)`` invariant, so
        the anchored form pays to preserve a rotation the orthogonality term
        cannot see.  Costs one ``k x k`` SVD per iteration.  Note the energy is
        then a difference of convex functions and is nonsmooth where ``X0'Y``
        drops rank, so the SPG theory applies on the full-rank set only.
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

    if fidelity not in ("auto", "anchor", "subspace"):
        raise ValueError("fidelity must be 'auto', 'anchor' or 'subspace'; "
                         f"got {fidelity!r}")
    neg_mass = float(negative_mass(X0))
    requested = fidelity
    if fidelity == "auto":
        # Only switch when there is an orthogonality term to work with.  The
        # subspace fidelity is scale-free and indifferent to rank, so on its own
        # at w = 0 it is minimised by any non-negative matrix inside range(X0),
        # including rank-one ones; what keeps the solution non-degenerate is the
        # orthogonality term, whose diagonal charges a collapsed column.
        fidelity = ("subspace"
                    if (nonneg and neg_mass > neg_mass_tol and float(w) > 0.0)
                    else "anchor")
        if fidelity == "subspace":
            warnings.warn(
                f"target has negative mass {neg_mass:.3f} "
                f"(||min(0,X0)||/||X0||) and nonneg=True, so {neg_mass:.0%} of "
                "its magnitude is unreachable: the entrywise fidelity would be "
                "dominated by constant, unimprovable terms and the optimum would "
                "degenerate toward max(0, X0). Switching to the sign-blind "
                "subspace fidelity, which anchors to range(X0) and is invariant "
                "to the column signs and rotation of the target. Pass "
                "fidelity='anchor' to force the entrywise distance, or use "
                "nsa_flow_data if you have the data itself. The choice is "
                "reported in result['fidelity_mode'].",
                RuntimeWarning, stacklevel=2)

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
    if fidelity == "subspace":
        if float(w) == 0.0:
            warnings.warn(
                "fidelity='subspace' with w=0 is degenerate: the term is "
                "scale-free and indifferent to rank, so any non-negative matrix "
                "inside range(X0) is optimal, including rank-one ones. Use w > 0 "
                "so the orthogonality term keeps the solution non-degenerate, or "
                "fidelity='anchor'.", RuntimeWarning, stacklevel=2)
        anchor = SubspaceAnchor(X0)
        if anchor.rank_deficient:
            warnings.warn(
                "target is rank deficient, so the projector onto range(X0) is "
                "not well determined; the subspace fidelity is regularised and "
                "should be interpreted with care.", RuntimeWarning, stacklevel=2)
        inv_k0 = 1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0
        vg = _subspace_vg(anchor, inv_k0)
    else:
        vg = _fused(compile)
    trace = [] if keep_trace else None
    t0 = time.time()
    total_iters, stop, gmap = 0, "max_iter", float("inf")
    for wi in ws:
        if verbose:
            print(f"  continuation step w={wi:.4f}")
        Y, E, it, stop, gmap = _solve_fixed_w(Y, X0, wi, denom, nonneg, max_iter,
                                              tol, sigma, verbose, trace, vg, align)
        total_iters += it

    if fidelity == "subspace":
        # both terms are degree-0 homogeneous, so the scale is a free gauge;
        # fix it at the target's scale rather than leaving it arbitrary
        nY = float(Y.norm())
        if nY > 0:
            Y = Y * (float(X0.norm()) / nY)
        tot, f, dd = vg(Y, X0, float(w), denom, None, None, False)[:3]
        tot, f, dd = float(tot), float(f), float(dd)
    else:
        tot, f, dd = energy(Y, X0, w=float(w), denom=denom, return_parts=True,
                            align=align)
    clamp_ref = X0.clamp_min(0.0)
    clamp_dist = float((Y - clamp_ref).norm() / clamp_ref.norm().clamp_min(1e-300))
    return NSAResult(
        Y=Y, target=X0, w=float(w), energy=float(tot), fidelity=float(f),
        defect=float(dd), raw_defect=float(stiefel_defect(Y)),
        effective_rank=float(effective_rank(Y)),
        scale_ratio=float(Y.norm() / X0.norm()), iters=total_iters, align=bool(align),
        fidelity_mode=fidelity, fidelity_requested=requested,
        target_negative_mass=neg_mass, clamp_distance=clamp_dist,
        converged=stop != "max_iter" and math.isfinite(gmap),
        stop_reason=stop, grad_map=float(gmap),
        seconds=time.time() - t0,
        w_schedule=ws, trace=trace, nonneg=bool(nonneg),
    )
