"""NSA-Flow solver: spectral projected gradient on a single, calibration-free energy.

The problem solved is

    minimise   E_w(Y) = (1 - w) ||Y - X0||_F^2 / ||X0||_F^2  +  w Orth(Y)
    subject to Y >= 0                                        (when ``nonneg``)

where ``Orth`` is one of three orthogonality functionals selected by ``orth``:

* ``"D"`` (default):  ``Dtilde = ||G - I/k||_F^2 / (1 - 1/k)`` for
  the trace-normalised Gram ``G = Y'Y / tr(Y'Y)``.
* ``"Cg"``:  ``||offdiag(V'V)||_F^2 / tr(V'V)^2``, smooth everywhere.
* ``"C"``:  mean squared cosine between column pairs, per-column normalised.

The method is Spectral Projected Gradient (Birgin, Martinez & Raydan 2000):
Barzilai-Borwein step lengths safeguarded by an Armijo backtracking line search
on the projected-gradient step.  For continuously differentiable ``E_w`` and
closed convex feasible set this converges to a stationary point, and the
gradient-mapping norm ``||Y+ - Y|| / t`` is a computable stationarity
certificate.

Convergence is detected by three independent criteria (whichever fires first):

* ``"grad_map"``: ``|Gmap| <= tol`` — tight stationarity certificate.
* ``"plateau"``: energy span over last ``patience`` steps < ``rtol`` (relative)
  — energy has converged to ~7 significant figures; ``max_iter`` is then a
  safety cap rather than an operating parameter.
* ``"line_search"``: step too small — iterate is at a local minimum of the
  line search (emits ``RuntimeWarning`` if far from stationarity).
* ``"max_iter"``: safety cap hit — iterate is NOT certified stationary
  (emits ``RuntimeWarning`` when ``|Gmap| > 1e-3``).

Cost per iteration is two ``[p,k] x [k,k]`` products plus one Gram: ``O(p k^2)``.
No SVD, eigendecomposition or QR appears in the loop -- except under ``align``
(deprecated), which adds one ``k x k`` SVD per evaluation.
"""
import time
import warnings

import math
import warnings

import torch

from .angle import (angle_defect, grad_angle_defect,
                     gram_offdiag_defect, grad_gram_offdiag_defect)
from .energy import (energy, stiefel_defect, stiefel_defect_normalised,
                     grad_stiefel_defect, effective_rank, value_and_grad,
                     aligned_target)
from .project import project_nonneg
from .subspace import (SubspaceAnchor, negative_mass, subspace_fidelity,
                       grad_subspace_fidelity)

__all__ = ["nsa_flow", "NSAResult", "_spg_loop", "_lbfgs_b_loop", "_torch_lbfgs_loop", "_nsa_flow_anchored"]

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

    @property
    def V(self):
        """Loading matrix [p, k]."""
        return self.get("Y")

    @property
    def components(self):
        """Components matrix [k, p] following scikit-learn convention."""
        y = self.get("Y")
        return y.T if y is not None else None

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


def _orth_terms_anchor(orth):
    """Return (orth_val, orth_grad, inv_k_fn) for the chosen orthogonality term.

    ``inv_k_fn(k)`` returns the normalisation constant such that the defect
    equals 1.0 at full collinearity -- the same convention as Dtilde.
    """
    if orth == "D":
        # Default: orthoNORMality, same normalisation as _solve_fixed_w
        return (stiefel_defect_normalised,
                lambda Y: (1.0 / (1.0 - 1.0 / max(Y.shape[-1], 2)))
                           * grad_stiefel_defect(Y),
                lambda k: 1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0)
    if orth == "Cg":
        return (gram_offdiag_defect, grad_gram_offdiag_defect, lambda k: None)
    if orth == "C":
        return (angle_defect, grad_angle_defect, lambda k: None)
    raise ValueError(f"orth must be 'D', 'Cg' or 'C' for nsa_flow; got {orth!r}")


def _custom_orth_vg(orth_val, orth_grad):
    """value_and_grad with a custom orthogonality term replacing Dtilde."""
    def vg(Y, X0, w, denom, inv_k_, eye_k, align):
        k = Y.shape[-1]
        d_ = float(denom)
        R = Y - (aligned_target(X0, Y) if align else X0)
        F = R.pow(2).sum() / d_
        Dn = orth_val(Y) if k > 1 else torch.zeros([], dtype=Y.dtype, device=Y.device)
        E = (1.0 - w) * F + w * Dn
        g = (1.0 - w) * (2.0 * R / d_)
        if k > 1 and w != 0.0:
            g = g + w * orth_grad(Y)
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


def _spg_loop(Y, proj, energy_fn, grad_fn, max_iter, tol, sigma,
              verbose=False, trace=None, caller="", w=None,
              patience=50, rtol=1e-7):
    """Monotone spectral projected gradient loop (shared by signed and data solvers).

    Parameters
    ----------
    Y : Tensor
        Starting point; projected onto the feasible set before the first step.
    proj : callable ``Tensor -> Tensor``
        Feasible-set projection (e.g. ``project_nonneg`` or a masked clamp).
    energy_fn : callable ``Tensor -> float``
        Objective value; called only inside the Armijo backtracking.
    grad_fn : callable ``Tensor -> (E, Tensor)``
        Returns ``(energy, gradient)`` together so the iterate's energy and
        gradient are computed in one call after each accepted step.
    max_iter : int
        Hard upper bound on iterations.  In practice the loop exits via
        ``grad_map``, ``plateau``, or ``line_search`` long before this limit.
    tol : float
        Stop when the projected-gradient mapping norm ``|Y+ - Y|/t`` falls
        below this.  The tight stationarity certificate.
    sigma : float
        Armijo sufficient-decrease constant.
    patience : int
        Plateau window length.  If the energy has not changed by more than
        ``rtol`` (relative) over the last ``patience`` accepted steps, stop
        with ``stop_reason="plateau"``.  Default 30.
    rtol : float
        Relative energy tolerance for plateau detection.
        ``(E_max - E_min) / (1 + |E_min|) < rtol`` triggers the stop.
        Default 1e-5 (energy converged to ~5 significant figures).
    verbose : bool
    trace : list or None
        Append ``dict(iter, energy, grad_map, step)`` if not None.
    caller : str
        Name used in the stall warning (e.g. ``"nsa_flow_signed"``).
    w : float or None
        Logged into ``trace`` if provided.

    Returns
    -------
    Y : Tensor
    E : float
    it : int
    stop : str   ``"grad_map"`` | ``"plateau"`` | ``"line_search"`` | ``"max_iter"``
    gmap : float
    """
    Y = proj(Y)
    E, g = grad_fn(Y)
    E = float(E)
    t = 1.0 / max(float(g.norm()), 1e-12)
    Y_prev = g_prev = None
    gmap, stop, it = float("inf"), "max_iter", 0
    E_window = []                               # for plateau detection

    for it in range(1, max_iter + 1):
        if Y_prev is not None:                      # Barzilai-Borwein with ABB
            s_ = Y - Y_prev
            r_ = g - g_prev
            sr = float((s_ * r_).sum())
            if sr > 0:
                if it % 2 == 0:
                    t = float((s_ * s_).sum()) / sr
                else:
                    t = sr / max(float((r_ * r_).sum()), 1e-12)
            else:
                # Safeguard: when sr <= 0 (non-convex curvature), don't jump to 1e12
                t = min(max(t, 1e-3), 10.0)
            t = min(max(t, _T_MIN), _T_MAX)

        accepted = False
        t_first, dn2_first = t, None
        for _ in range(30):                         # Armijo backtracking
            Y_new = proj(Y - t * g)
            d_ = Y_new - Y
            dn2 = float((d_ * d_).sum())
            if dn2_first is None:
                dn2_first = dn2
            if float(energy_fn(Y_new)) <= E - sigma * dn2 / t:
                accepted = True
                break
            t *= 0.5

        if not accepted:
            stop = "line_search"
            if not math.isfinite(gmap):
                gmap = (dn2_first ** 0.5) / t_first
            if gmap > max(tol, 0.0) * 1e3:
                warnings.warn(
                    f"{caller}: line search stalled after "
                    f"{it} iteration(s) with |Gmap|={gmap:.2e} against "
                    f"tol={tol:.1e}; the returned point is not stationary. "
                    "Inspect stop_reason and grad_map.",
                    RuntimeWarning, stacklevel=3)
            break

        gmap = (dn2 ** 0.5) / t
        Y_prev, g_prev = Y, g
        Y = Y_new
        E, g = grad_fn(Y)
        E = float(E)

        # ── plateau detection ──────────────────────────────────────────────
        E_window.append(E)
        if len(E_window) > patience:
            E_window.pop(0)
        if len(E_window) == patience:
            span = max(E_window) - min(E_window)
            if span / (1.0 + abs(min(E_window))) < rtol:
                stop = "plateau"
                break

        if trace is not None:
            row = dict(iter=len(trace) + 1, energy=E, grad_map=gmap, step=t)
            if w is not None:
                row["w"] = float(w)
            trace.append(row)
        if verbose and (it % max(1, max_iter // 10) == 0 or it == 1):
            w_tag = f" w={w:.3f}" if w is not None else ""
            print(f"    [{w_tag}it={it:5d}] E={E:.8e} |Gmap|={gmap:.3e} t={t:.3e}")
        if gmap <= tol:
            stop = "grad_map"
            break

    if stop == "max_iter" and math.isfinite(gmap) and gmap > 1e-3:
        warnings.warn(
            f"{caller}: reached max_iter={max_iter} with |Gmap|={gmap:.2e}; "
            "the iterate is not stationary.  Increase max_iter or lower tol, "
            "or inspect stop_reason and grad_map.",
            RuntimeWarning, stacklevel=3)

    return Y, E, it, stop, gmap


def _lbfgs_b_loop(Y, bounds, energy_fn, grad_fn, max_iter=2000, tol=1e-5,
                  verbose=False, trace=None, caller="", w=None):
    """Quasi-Newton L-BFGS-B loop for box-constrained optimization."""
    from scipy.optimize import minimize
    import numpy as np
    shape = Y.shape
    device = Y.device
    dtype = Y.dtype

    def f_and_g(y_flat):
        Y_t = torch.as_tensor(y_flat.reshape(shape), dtype=dtype, device=device)
        E, g = grad_fn(Y_t)
        return float(E), g.detach().cpu().numpy().astype(np.float64).flatten()

    y0 = Y.detach().cpu().numpy().astype(np.float64).flatten()
    res = minimize(
        f_and_g, y0, method="L-BFGS-B", jac=True, bounds=bounds,
        options=dict(maxiter=max_iter, ftol=1e-8, gtol=tol if tol is not None else 1e-5)
    )
    Y_opt = torch.as_tensor(res.x.reshape(shape), dtype=dtype, device=device)
    E_opt = float(res.fun)
    gmap = float(np.max(np.abs(res.jac)))
    msg = str(res.message).upper()
    stop = "grad_map" if res.success else ("plateau" if "CONVERGENCE" in msg else "max_iter")
    if trace is not None:
        trace.append(dict(iter=res.nit, energy=E_opt, grad_map=gmap, step=0.0))
    return Y_opt, E_opt, res.nit, stop, gmap


def _torch_lbfgs_loop(Y0, energy_fn, grad_fn, max_iter=200, tol=1e-5,
                      history_size=10, patience=5, rtol=1e-6,
                      mask=None, verbose=False, trace=None, caller="", w=None):
    """Pure PyTorch native quasi-Newton L-BFGS loop via quadratic reparameterization Y = Z**2.

    100% PyTorch native (runs on CPU, CUDA, MPS). Enforces non-negativity smoothly
    without boundary stalling, and uses analytical gradients via the exact chain rule:
        dE/dZ = (2 * Z * dE/dY).contiguous()
    """
    device = Y0.device
    dtype = Y0.dtype
    t0 = time.time()

    # Reparameterize: Y = Z**2 >= 0 (or Z**2 * mask if masked)
    Z_init = torch.sqrt(Y0.clamp_min(1e-8))
    Z = torch.nn.Parameter(Z_init)

    opt = torch.optim.LBFGS(
        [Z], lr=1.0, max_iter=20, history_size=history_size,
        line_search_fn="strong_wolfe", tolerance_grad=tol, tolerance_change=tol
    )

    stop = "max_iter"
    gmap = float("inf")
    E_cur = float("inf")
    E_window = []
    outer_steps = max(5, max_iter // 20)
    total_sub_iters = 0

    prev_E = None
    for step in range(1, outer_steps + 1):
        def closure():
            nonlocal total_sub_iters
            total_sub_iters += 1
            opt.zero_grad()
            if mask is not None:
                Y_cur = Z.pow(2) * mask
                E, g = grad_fn(Y_cur)
                Z.grad = (2.0 * Z * mask * g).contiguous()
            else:
                Y_cur = Z.pow(2)
                E, g = grad_fn(Y_cur)
                Z.grad = (2.0 * Z * g).contiguous()
            return torch.as_tensor(E, dtype=dtype, device=device)

        loss = opt.step(closure)
        E_cur = float(loss.detach())

        with torch.no_grad():
            if mask is not None:
                Y_cur = Z.pow(2) * mask
                _, g_cur = grad_fn(Y_cur)
                gmap = float((torch.clamp_min(Y_cur - g_cur, 0.0) * mask - Y_cur).norm())
            else:
                Y_cur = Z.pow(2)
                _, g_cur = grad_fn(Y_cur)
                gmap = float((torch.clamp_min(Y_cur - g_cur, 0.0) - Y_cur).norm())

        if trace is not None:
            trace.append(dict(iter=total_sub_iters, energy=E_cur, grad_map=gmap, step=1.0))

        if gmap <= tol:
            stop = "grad_map"
            break

        if prev_E is not None and abs(E_cur - prev_E) / (1.0 + abs(E_cur)) < rtol:
            stop = "plateau"
            break
        prev_E = E_cur

        E_window.append(E_cur)
        if len(E_window) > patience:
            E_window.pop(0)
        if len(E_window) == patience:
            span = max(E_window) - min(E_window)
            if span / (1.0 + abs(min(E_window))) < rtol:
                stop = "plateau"
                break

    with torch.no_grad():
        if mask is not None:
            Y_opt = Z.pow(2) * mask
        else:
            Y_opt = Z.pow(2)

    return Y_opt, E_cur, total_sub_iters, stop, gmap


def _nsa_flow_anchored(target, w=0.5, *, init=None, nonneg=True, max_iter=5000, tol=None,
                      continuation=0, w_start=0.0, sigma=1e-4, dtype=None, device=None,
                      verbose=False, keep_trace=False, compile=False, align=False,
                      fidelity="auto", neg_mass_tol=0.01, orth="D", optimizer="spg"):
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
        **Deprecated.** Anchor to ``X0``'s right-``O(k)`` orbit instead of
        ``X0`` itself by replacing ``||Y - X0||_F^2`` with
        ``min_{Q in O(k)} ||Y - X0 Q||_F^2`` (orthogonal Procrustes).  The
        original justification was that ``D`` is right-``O(k)`` invariant, so
        the anchored form over-constrains the problem by paying to preserve a
        rotation the defect cannot see.  That argument does *not* carry over to
        the ``C`` or ``Cg`` defect, which is *not* right-``O(k)`` invariant --
        mixing columns destroys orthogonality.  Setting ``align=True`` raises a
        ``DeprecationWarning`` and will be removed in a future release.
    orth : {\"D\", \"Cg\", \"C\"}
        Which orthogonality functional to minimise.

        ``\"D\"`` (default) is the full Stiefel defect ``||G - I/k||_F^2``, which
        penalises both off-diagonal correlations *and* unequal column norms.  It
        is the right choice when the target is non-negative and column norms
        carry interpretable information.

        ``\"Cg\"`` (smooth orthogonality) is ``||offdiag(V'V)||_F^2 / tr(V'V)^2``
        -- smooth everywhere including at zero columns, zero iff columns are
        mutually orthogonal.  The default for ``nsa_flow_signed``; also
        appropriate here on signed input where column-norm equality is
        undesirable.

        ``\"C\"`` (angle defect) is the mean squared cosine between column pairs,
        normalised per column.  Carries a dead-column penalty (``diagonal=True``).

        Only ``\"D\"`` is compatible with ``compile=True``; the other two call
        angle.py functions that are not yet compiled.
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
    if align:
        warnings.warn(
            "nsa_flow: align=True is deprecated and will be removed in a future "
            "release.  Its justification was D's right-O(k) invariance, which "
            "does not carry over to the C/Cg orthogonality defect; see the "
            "nsa_flow.angle module docstring.  The parameter remains functional "
            "for now so existing experiments continue to run.",
            DeprecationWarning, stacklevel=2)
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
        if orth == "D":
            vg = _fused(compile)
        else:
            ov, og, _ = _orth_terms_anchor(orth)
            vg = _custom_orth_vg(ov, og)
    trace = [] if keep_trace else None
    t0 = time.time()
    total_iters, stop, gmap = 0, "max_iter", float("inf")
    for wi in ws:
        if verbose:
            print(f"  continuation step w={wi:.4f}")
        if optimizer in ("torch_lbfgs", "torch-lbfgs") and not align and nonneg:
            inv_k_val = inv_k0 if fidelity == "subspace" else (1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0)
            eye_k = torch.eye(k, dtype=Y.dtype, device=Y.device)
            def _anc_energy(Y_c):
                return float(vg(Y_c, X0, wi, denom, inv_k_val, eye_k, False)[0])
            def _anc_grad_and_energy(Y_c):
                E_v, _, _, g_v = vg(Y_c, X0, wi, denom, inv_k_val, eye_k, False)
                return float(E_v), g_v
            iter_cap = max_iter if max_iter is not None else 300
            Y, E, it, stop, gmap = _torch_lbfgs_loop(
                Y, _anc_energy, _anc_grad_and_energy, max_iter=iter_cap, tol=tol,
                verbose=verbose, trace=trace, caller="_nsa_flow_anchored", w=wi
            )
        else:
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
        converged=stop in ("grad_map", "plateau") or (
            stop == "line_search" and math.isfinite(gmap)),
        stop_reason=stop, grad_map=float(gmap),
        seconds=time.time() - t0,
        w_schedule=ws, trace=trace, nonneg=bool(nonneg),
    )


def nsa_flow(data_or_target, k=None, w=0.5, *, mode="auto", nonneg=True,
             consolidate=False, optimizer="torch_lbfgs", init=None, max_iter=None,
             tol=None, **kwargs):
    """Unified high-level entry point for NSA-Flow representation learning.

    Automatically inspects input structure, data sign distribution, and task
    parameters to select the appropriate specialized method:

    1. **Data-reconstruction mode** (when data is non-negative, or ``mode='data'``):
       Fits non-negative basis ``V >= 0`` reconstructing ``X ≈ X V V'``.
    2. **Signed-lifting mode** (when data has negative entries, or ``mode='signed'``):
       Lifts data into positive and negative lobes ``V = V+ - V-``, discovering
       sparse contrast components.  Pass ``consolidate=True`` for strictly disjoint
       supports (zero lobe overlap).
    3. **Anchored mode** (when ``k`` is omitted on a target matrix, or ``mode='anchored'``):
       Perturbs an existing target loading matrix (e.g. PCA loadings) toward
       the non-negative Stiefel manifold.

    Parameters
    ----------
    data_or_target : array-like or Tensor
        Input 2D matrix: either data [n, p] or target [p, k].
    k : int, optional
        Number of components to extract. If omitted and mode='auto', treats input
        as an anchored target matrix [p, k].
    w : float, default 0.5
        Trade-off parameter in [0, 1]. w=0 maximizes fidelity, w=1 maximizes orthogonality.
    mode : {"auto", "data", "nonneg", "signed", "contrast", "anchored", "target"}
        Execution mode. Default 'auto' detects from data signs and arguments.
    consolidate : bool, default False
        Guarantees exact disjoint supports per component in signed mode.
    optimizer : {"torch_lbfgs", "spg", "lbfgs"}, default "torch_lbfgs"
        Optimization algorithm:
        - "torch_lbfgs": Native pure PyTorch quasi-Newton via quadratic reparameterization (default).
        - "spg": Monotone Spectral Projected Gradient.
        - "lbfgs": SciPy L-BFGS-B (box constrained).
    init : str or Tensor, optional
        Initial point strategy or tensor.
    max_iter : int, optional
        Maximum iterations. Default adapted to method.
    tol : float, optional
        Stationarity tolerance.
    **kwargs :
        Additional arguments forwarded to the selected solver.

    Returns
    -------
    NSAResult
        Result dictionary-like object with `.Y`, `.energy`, `.fidelity`, `.defect`,
        `.iters`, `.stop_reason`, `.converged`, and metadata.
    """
    X = torch.as_tensor(data_or_target)
    if X.ndim != 2:
        raise ValueError(f"Input must be 2-D [n, p] or [p, k]; got shape {tuple(X.shape)}")

    if "signed" in kwargs:
        if kwargs.pop("signed"):
            mode = "signed"
    if "nonneg" in kwargs:
        if kwargs.pop("nonneg"):
            mode = "data"

    if mode == "auto":
        if k is not None:
            min_val = float(X.min())
            mode = "signed" if min_val < -1e-12 else "data"
        else:
            mode = "anchored"

    if mode in ("data", "nonneg"):
        from .reconstruct import nsa_flow_data
        init_strat = "clamp" if init in (None, "auto") else init
        iter_cap = max_iter if max_iter is not None else (150 if optimizer in ("torch_lbfgs", "torch-lbfgs") else 2000)
        return nsa_flow_data(X, k=k, w=w, init=init_strat, max_iter=iter_cap,
                             tol=tol, optimizer=optimizer, **kwargs)

    elif mode in ("signed", "contrast"):
        from .signed import nsa_flow_signed
        init_strat = init if init is not None else "auto"
        iter_cap = max_iter if max_iter is not None else (150 if optimizer in ("torch_lbfgs", "torch-lbfgs") else 8000)
        tol_val = tol if tol is not None else (1e-5 if optimizer in ("torch_lbfgs", "torch-lbfgs") else None)
        return nsa_flow_signed(X, k=k, w=w, init=init_strat, consolidate=consolidate,
                               max_iter=iter_cap, tol=tol_val, optimizer=optimizer, **kwargs)

    elif mode in ("anchored", "target"):
        iter_cap = max_iter if max_iter is not None else 20000
        anc_opt = optimizer if optimizer != "torch_lbfgs" else "spg"
        return _nsa_flow_anchored(X, w=w, nonneg=nonneg, init=init,
                                  max_iter=iter_cap, tol=tol, optimizer=anc_opt, **kwargs)

    else:
        raise ValueError(f"Unknown mode {mode!r}; choose 'auto', 'data', 'signed', or 'anchored'")
