r"""Constrained optimisers for NSA-Flow, under one contract.

Every solver entry point in this package reduces to the same problem shape:

    minimise  E(Y)   subject to   Y in Omega,   Omega closed convex, P = proj(Omega)

with ``E`` smooth on ``Omega`` and a closed-form gradient.  What differed
between the three entry points was not the problem, it was the bookkeeping --
each had its own loop, its own stopping rule, its own iteration counter and its
own idea of what ``grad_map`` measured.  That is now one implementation per
*algorithm*, behind one signature.

The contract
------------
Every ``_<name>`` routine below takes

    (Y0, energy_fn, grad_fn, proj, *, max_iter, tol, ...)

where ``energy_fn(Y) -> float`` and ``grad_fn(Y) -> (float, Tensor)``, and
returns a :class:`Report`.  Three things are standardised, and they are the
three that were previously incomparable:

**The certificate.**  ``grad_map`` is always
:func:`nsa_flow.diagnostics.gradient_mapping` -- scale-invariant, projected,
identical in every loop.  It is evaluated at the returned iterate, never
inherited from a step length or a sentinel.

**The budget.**  ``max_iter`` caps *gradient evaluations*, not "iterations".
An SPG iteration costs one gradient and a few energies; an L-BFGS step costs
several of each.  Counting accepted steps made the fast-looking optimiser the
one that did the most work per step.  ``Report`` carries ``n_grad`` and
``n_energy`` so cost is always stated in the currency that is actually spent,
and the cap is enforced in that currency by every loop.

**The stop reason.**  One of :data:`~nsa_flow.diagnostics.STOP_REASONS`.
``"grad_map"`` requires ``grad_map <= tol`` and is the strong claim.
``"plateau"`` requires that the energy *and* the certificate have both stopped
moving (:func:`_plateau`) -- testing the energy alone certified points orders
from stationarity, because on the quartic objective ``E`` reaches the float64
floor while ``grad_map`` is still falling.  ``"line_search"`` and ``"max_iter"``
never mean convergence.

Algorithms
----------
``spg``
    Spectral projected gradient (Birgin-Martinez-Raydan): alternating
    Barzilai-Borwein step lengths with Armijo backtracking on the projected
    step.  Pure PyTorch, no factorisation, ``O(pk^2)`` per iteration, runs
    unmodified on CPU, CUDA and MPS.  Converges to a stationary point of the
    constrained problem, which is the object ``grad_map`` certifies.
``fista``
    Projected accelerated gradient with adaptive (function-value) restart and
    backtracking on the Lipschitz estimate.  Same cost per iteration as ``spg``
    with one extra axpy; ``O(1/i^2)`` on the convex part of the path.  Included
    because the anchored objective *is* convex in ``Y``, where acceleration is
    the theoretically right answer and BB is not.
``pqn``
    Two-metric projection (Bertsekas) with an L-BFGS metric on the free set:
    quasi-Newton curvature where the iterate is free, a projected gradient step
    where it is at the bound, and a projected-arc Armijo line search.  Pure
    PyTorch, so it runs on CUDA and MPS, and it is the only method here that
    matches SciPy L-BFGS-B's solution quality without leaving the device.
``lbfgsb``
    L-BFGS-B (Byrd, Lu, Nocedal & Zhu 1995) in pure PyTorch -- generalized
    Cauchy point plus subspace minimisation over the compact limited-memory
    representation.  A genuine bound-constrained quasi-Newton: it identifies the
    whole active set in one projected search rather than one bound per
    iteration, which is what makes it robust on objectives where half the
    coordinates sit at zero.  Runs on CPU, CUDA and MPS in either precision.
    See :mod:`nsa_flow.lbfgsb`.
``scipy_lbfgsb``
    SciPy's Fortran L-BFGS-B, kept as the reference implementation to validate
    the one above against.  Host-side float64 only, and SciPy is not a
    dependency of this package.
``torch_lbfgs``
    ``torch.optim.LBFGS`` on the smooth reparameterisation ``Y = Z^2``.
    GPU-resident and factorisation-free, but it optimises a *different*
    problem: ``dE/dZ = 2Z (dE/dY)`` vanishes wherever ``Z = 0``, so every
    coordinate that starts at zero is a fixed point.  Seeded from
    ``clamp(PCA, 0)`` -- roughly half the entries -- that froze the support at
    the initialisation and made the flow unable to recruit a feature.  The
    floor here (:data:`_Z_FLOOR`, relative to the peak) restores mobility; the
    method is kept honest rather than kept fast.
"""
import math
import time
import warnings

import torch

from .diagnostics import gradient_mapping
from .project import project_nonneg

__all__ = ["Report", "minimise", "OPTIMIZERS", "optimizer_names"]

_T_MIN, _T_MAX = 1e-12, 1e12

#: Relative floor added to the ``Y = Z^2`` initialisation so that coordinates
#: starting at zero retain a usable gradient.  ``1e-4`` of the peak magnitude:
#: large enough that ``dE/dZ = 2Z dE/dY`` is not numerically dead, small enough
#: that the perturbation to the energy is far below the solve tolerance.
_Z_FLOOR = 1e-4


class Report(dict):
    """Uniform optimiser output; a dict with attribute access."""

    __getattr__ = dict.__getitem__

    def __repr__(self):
        return (f"Report(stop={self['stop']}, grad_map={self['grad_map']:.2e}, "
                f"energy={self['energy']:.6e}, iters={self['iters']}, "
                f"n_grad={self['n_grad']}, n_energy={self['n_energy']}, "
                f"seconds={self['seconds']:.3f})")


def _trace_start(trace, E, gmap, t0, w):
    """Record the *initial* point as iteration 0.

    Without it every convergence curve begins after the first accepted step, so
    the first -- usually largest -- energy drop is invisible, and ``trace[0]``
    reads as the starting energy while actually being one step in.  That made
    two optimisers on the same problem appear to start from different points.
    """
    if trace is None:
        return
    row = dict(iter=0, energy=float(E), grad_map=float(gmap), step=float("nan"),
               seconds=time.time() - t0, n_grad=1, n_energy=0)
    if w is not None:
        row["w"] = float(w)
    trace.append(row)


def _report(Y, E, iters, stop, gmap, n_grad, n_energy, t0, E0=None,
            gmap0=None, **extra):
    r = Report(Y=Y, energy=float(E), iters=int(iters), stop=str(stop),
               grad_map=float(gmap), n_grad=int(n_grad), n_energy=int(n_energy),
               energy_start=(float(E0) if E0 is not None else float("nan")),
               grad_map_start=(float(gmap0) if gmap0 is not None else float("nan")),
               seconds=time.time() - t0)
    r.update(extra)
    return r


class _Counter:
    """Wraps the user's callables so budget accounting cannot be forgotten."""

    __slots__ = ("energy_fn", "grad_fn", "n_energy", "n_grad")

    def __init__(self, energy_fn, grad_fn):
        self.energy_fn, self.grad_fn = energy_fn, grad_fn
        self.n_energy = self.n_grad = 0

    def E(self, Y):
        self.n_energy += 1
        return float(self.energy_fn(Y))

    def G(self, Y):
        self.n_grad += 1
        E, g = self.grad_fn(Y)
        return float(E), g


def _plateau(E_window, g_window, patience, rtol, gmap_rtol=0.1):
    """``True`` when *both* the energy and the certificate have stopped moving.

    Testing the energy alone is what made ``plateau`` an unreliable stop.  On the
    quartic data objective the energy reaches the float64 floor -- identical to
    twelve significant figures -- while the gradient mapping is still falling by
    an order of magnitude per few thousand evaluations.  Stopping there returns a
    point whose support is still moving, which is exactly the failure the signed
    module documents (43% movement in the largest basis entry, 2.7% of the
    support flipping, between |Gmap| 4.7e-05 and 1e-09).

    So a plateau now additionally requires that the certificate has improved by
    less than ``gmap_rtol`` (default 10%) across the window.  When both hold,
    nothing further is achievable in this precision and the stop is a genuine
    statement about ``Y``, not only about ``E``.
    """
    if len(E_window) < patience:
        return False
    lo, hi = min(E_window), max(E_window)
    if (hi - lo) / (1.0 + abs(lo)) >= rtol:
        return False
    first, best = g_window[0], min(g_window)
    if not (math.isfinite(first) and first > 0.0):
        return False
    return best >= (1.0 - gmap_rtol) * first


#: How far above ``tol`` the certificate may sit and still permit a stall to be
#: called the numerical floor.  Matches the threshold :func:`minimise` uses for
#: its "not stationary" warning, so the two can never disagree.
STALL_SLACK = 1e3


def _classify_stall(E, E_best_trial, g_window, gmap, tol, dtype, slack=64.0):
    """Was a line-search failure the numerical floor, or a trap?

    A failed line search means no feasible descent step exists at working
    precision.  That is consistent with two very different situations, and
    conflating them is how a solver certifies a point it has not solved:

    * **the floor** -- the iterate really is as good as this precision allows;
    * **a trap** -- a bad direction, a discontinuity, or a scale mismatch in the
      first trial step, where the energy barely moves *because the step was
      wrong*, not because the iterate is good.

    The energy evidence alone cannot tell them apart: both produce a tiny
    energy change.  So the certificate is the gate.  Unless ``grad_map`` is
    within :data:`STALL_SLACK` of ``tol``, no stall is certified -- the earlier
    rule certified a projected quasi-Newton point at ``|Gmap| = 2.8e-01`` after
    four gradient evaluations, which is not a solution by any reading.

    Given that gate, either kind of evidence suffices: the certificate stopped
    improving across the window (the first-order methods' signature), or the
    best trial step moved the energy by less than floating-point noise (the
    quasi-Newton signature, which converges superlinearly and then stops
    abruptly, so its window still shows large improvement at the stall).
    """
    if not (math.isfinite(gmap) and gmap <= STALL_SLACK * max(tol, 0.0)):
        return "line_search"
    if _gmap_stalled(g_window):
        return "plateau"
    if E_best_trial is not None and math.isfinite(E_best_trial):
        eps = float(torch.finfo(dtype).eps)
        if (E_best_trial - E) <= slack * eps * (1.0 + abs(E)):
            return "plateau"
    return "line_search"


def _gmap_stalled(g_window, min_samples=5, gmap_rtol=0.1):
    """``True`` when the certificate has stopped improving over the window.

    Used to classify a line-search failure.  "No feasible descent step exists to
    within working precision" is the definition of the energy being at its
    floor, so the energy half of the plateau test is automatic there; what
    remains to check is whether ``Y`` had also stopped moving.  If it had, the
    stall is the numerical floor and is reported as ``"plateau"``.  If the
    certificate was still falling, the iterate is genuinely trapped -- a
    discontinuity, a bad basin -- and that is reported as ``"line_search"`` with
    no certificate, which is the honest answer.
    """
    if len(g_window) < min_samples:
        return False
    first, best = g_window[0], min(g_window)
    return (math.isfinite(first) and first > 0.0
            and best >= (1.0 - gmap_rtol) * first)


# --------------------------------------------------------------------------
# spectral projected gradient
# --------------------------------------------------------------------------
def _spg(Y, energy_fn, grad_fn, proj, *, max_iter=2000, tol=1e-9, sigma=1e-4,
         patience=50, rtol=1e-12, alternating=True, trace=None, w=None,
         verbose=False, **_):
    t0 = time.time()
    cnt = _Counter(energy_fn, grad_fn)
    Y = proj(Y)
    E, g = cnt.G(Y)
    t = 1.0 / max(float(g.norm()), 1e-12)
    Y_prev = g_prev = None
    gmap = gradient_mapping(Y, g, proj)
    E0, gmap0 = E, gmap
    stop, it, window, gwin = "max_iter", 0, [], []
    _trace_start(trace, E, gmap, t0, w)

    while cnt.n_grad < max_iter:
        it += 1
        if gmap <= tol:
            stop = "grad_map"
            break
        if Y_prev is not None:                       # (alternating) Barzilai-Borwein
            s_ = Y - Y_prev
            r_ = g - g_prev
            # one host transfer for all three inner products rather than three;
            # on an accelerator each is a full pipeline stall
            sr, ss, rr = torch.stack([(s_ * r_).sum(), (s_ * s_).sum(),
                                      (r_ * r_).sum()]).tolist()
            if sr > 0:
                if alternating and it % 2 == 0:
                    t = sr / max(rr, 1e-300)                         # BB2
                else:
                    t = ss / sr                                      # BB1
            else:
                # negative curvature along the secant: BB is meaningless here.
                # Keep the previous step rather than jumping to the clamp.
                t = min(max(t, _T_MIN), _T_MAX)
            t = min(max(t, _T_MIN), _T_MAX)

        accepted, E_try = False, None
        for _ in range(60):                          # Armijo on the projected step
            Y_new = proj(Y - t * g)
            d_ = Y_new - Y
            dn2 = float((d_ * d_).sum())
            E_new = cnt.E(Y_new)
            E_try = E_new if E_try is None else min(E_try, E_new)
            if E_new <= E - sigma * dn2 / t:
                accepted = True
                break
            t *= 0.5
            if cnt.n_energy >= 4 * max_iter:         # pathological line search
                break
        if not accepted:
            stop = _classify_stall(E, E_try, gwin, gmap, tol, Y.dtype)
            break

        Y_prev, g_prev = Y, g
        Y = Y_new
        E, g = cnt.G(Y)
        gmap = gradient_mapping(Y, g, proj)

        if trace is not None:
            row = dict(iter=it, energy=E, grad_map=gmap, step=t,
                       seconds=time.time() - t0, n_grad=cnt.n_grad,
                       n_energy=cnt.n_energy)
            if w is not None:
                row["w"] = float(w)
            trace.append(row)
        if verbose and (it % 50 == 0 or it == 1):
            print(f"    [spg it={it:5d}] E={E:.8e} |Gmap|={gmap:.3e} t={t:.3e}")

        window.append(E)
        gwin.append(gmap)
        if len(window) > patience:
            window.pop(0)
            gwin.pop(0)
        if gmap <= tol:
            stop = "grad_map"
            break
        if _plateau(window, gwin, patience, rtol):
            stop = "plateau"
            break

    return _report(Y, E, it, stop, gmap, cnt.n_grad, cnt.n_energy, t0,
                   E0=E0, gmap0=gmap0)


# --------------------------------------------------------------------------
# projected accelerated gradient (FISTA) with adaptive restart
# --------------------------------------------------------------------------
def _fista(Y, energy_fn, grad_fn, proj, *, max_iter=2000, tol=1e-9,
           patience=50, rtol=1e-12, cert_every=5, trace=None, w=None,
           verbose=False, **_):
    """Projected accelerated gradient, backtracked, with function-value restart.

    One gradient evaluation per pass, taken at the extrapolated point ``Z``; the
    loop condition is therefore always making progress against the budget, and a
    restart cannot spin.  The certificate lives at the returned iterate ``Y``,
    whose gradient the method does not otherwise need, so it is sampled on a
    cadence and always re-measured once before returning.
    """
    t0 = time.time()
    cnt = _Counter(energy_fn, grad_fn)
    Y = proj(Y)
    E, g = cnt.G(Y)
    gmap = gradient_mapping(Y, g, proj)
    L = max(float(g.norm()), 1e-12)
    Z, theta = Y, 1.0
    E0, gmap0 = E, gmap
    stop, it, window, gwin = "max_iter", 0, [], []
    _trace_start(trace, E, gmap, t0, w)

    while cnt.n_grad < max_iter:
        it += 1
        if gmap <= tol:
            stop = "grad_map"
            break
        E_Z, g_Z = cnt.G(Z)

        accepted, E_try = False, None
        for _ in range(60):                    # backtrack the Lipschitz estimate
            Y_new = proj(Z - g_Z / L)
            d_ = Y_new - Z
            dn2 = float((d_ * d_).sum())
            E_new = cnt.E(Y_new)
            E_try = E_new if E_try is None else min(E_try, E_new)
            if E_new <= E_Z + float((g_Z * d_).sum()) + 0.5 * L * dn2 + 1e-15:
                accepted = True
                break
            L *= 2.0
        if not accepted:
            stop = _classify_stall(E, E_try, gwin, gmap, tol, Y.dtype)
            break

        if E_new > E:                          # adaptive restart: drop the momentum
            if theta == 1.0:
                # already un-accelerated and still not descending: the quadratic
                # model is satisfied but the step does not help.
                stop = _classify_stall(E, E_try, gwin, gmap, tol, Y.dtype)
                break
            Z, theta = Y, 1.0
            continue

        theta_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * theta * theta))
        Z = proj(Y_new + ((theta - 1.0) / theta_new) * (Y_new - Y))
        theta = theta_new
        Y, E = Y_new, E_new
        L = max(L * 0.8, 1e-12)                # let the estimate relax back down

        if it % cert_every == 0:
            _, g = cnt.G(Y)
            gmap = gradient_mapping(Y, g, proj)

        if trace is not None:
            row = dict(iter=it, energy=E, grad_map=gmap, step=1.0 / L,
                       seconds=time.time() - t0, n_grad=cnt.n_grad,
                       n_energy=cnt.n_energy)
            if w is not None:
                row["w"] = float(w)
            trace.append(row)
        if verbose and (it % 50 == 0 or it == 1):
            print(f"    [fista it={it:5d}] E={E:.8e} |Gmap|={gmap:.3e} L={L:.3e}")

        window.append(E)
        gwin.append(gmap)
        if len(window) > patience:
            window.pop(0)
            gwin.pop(0)
        if gmap <= tol:
            stop = "grad_map"
            break
        if _plateau(window, gwin, patience, rtol):
            stop = "plateau"
            break

    _, g = cnt.G(Y)                            # certify the point actually returned
    gmap = gradient_mapping(Y, g, proj)
    if stop == "grad_map" and gmap > tol:
        stop = "plateau"
    return _report(Y, E, it, stop, gmap, cnt.n_grad, cnt.n_energy, t0,
                   E0=E0, gmap0=gmap0)


# --------------------------------------------------------------------------
# projected quasi-Newton (two-metric projection L-BFGS)
# --------------------------------------------------------------------------
def _pqn(Y, energy_fn, grad_fn, proj, *, max_iter=2000, tol=1e-9, sigma=1e-4,
         history_size=10, patience=50, rtol=1e-12, trace=None, w=None,
         verbose=False, **_):
    r"""Bertsekas two-metric projection with an L-BFGS metric on the free set.

    The reason this exists: on the quartic data objective, SciPy's L-BFGS-B
    reaches a *lower* stationary point than SPG in four times fewer gradient
    evaluations, because it gets the curvature right.  But it is float64 CPU
    only and pays a host-device round trip per evaluation, so it cannot be the
    default for a library that has to run on a GPU.  This is the same idea in
    pure PyTorch.

    At each iterate split the coordinates into

        A = { i : Y_i <= eps  and  grad_i > 0 }        (active at the bound)
        F = complement                                  (free)

    and take ``d_A = -grad_A`` (a plain gradient step, which the projection will
    hold at the bound) and ``d_F = -H_F grad_F`` with ``H_F`` the L-BFGS inverse
    Hessian approximation restricted to ``F``.  Bertsekas' result is that this
    *two-metric* combination retains the descent and identification properties
    that a naive ``P(Y - H grad)`` loses -- scaling the whole gradient by a
    non-diagonal ``H`` and then projecting is not a descent method.

    ``eps`` is tied to the certificate, ``min(eps0, gmap) * max|Y|``, so the
    active set is only trusted as far as the iterate is converged; this is what
    stops the method from freezing a coordinate at the bound early, which is the
    failure mode of the ``Y = Z^2`` reparameterisation.

    Line search is projected-arc Armijo, the same rule SPG uses, so the whole
    method is monotone and every accepted step is certified by the shared
    gradient mapping.
    """
    t0 = time.time()
    cnt = _Counter(energy_fn, grad_fn)
    Y = proj(Y)
    E, g = cnt.G(Y)
    gmap = gradient_mapping(Y, g, proj)
    S, R, rho = [], [], []                       # L-BFGS memory: s, y, 1/(s.y)
    stop, it, window, gwin = "max_iter", 0, [], []
    eps0 = 1e-6
    E0, gmap0 = E, gmap
    _trace_start(trace, E, gmap, t0, w)

    while cnt.n_grad < max_iter:
        it += 1
        if gmap <= tol:
            stop = "grad_map"
            break

        scale = float(Y.abs().max())
        eps = min(eps0, max(gmap, 0.0)) * (scale if scale > 0 else 1.0)
        free = ~((Y <= eps) & (g > 0))
        free_f = free.to(Y.dtype)

        # Two-loop recursion, restricted to the free set and kept entirely
        # on-device.  Every `float()` here is a host synchronisation, and with
        # history_size=10 the scalar version issued ~20 of them per iteration --
        # measured at 1909 us per gradient at p=5000 against SPG's 1003, on a
        # problem whose arithmetic is microseconds.  The scalars stay as 0-dim
        # tensors; only the single line-search comparison below needs a sync.
        q = g * free_f
        alphas = []
        for s_i, y_i, r_i in zip(reversed(S), reversed(R), reversed(rho)):
            a = r_i * (s_i * q).sum()
            q = q - a * y_i
            alphas.append(a)
        if R:
            yy = (R[-1] * R[-1]).sum()
            q = torch.where(yy > 0, q * ((S[-1] * R[-1]).sum() / yy.clamp_min(1e-300)), q)
        for (s_i, y_i, r_i), a in zip(zip(S, R, rho), reversed(alphas)):
            b = r_i * (y_i * q).sum()
            q = q + (a - b) * s_i
        d = -(q * free_f) - g * (1.0 - free_f)

        if float((d * g).sum()) >= 0.0:          # not a descent direction: reset
            S, R, rho = [], [], []
            d = -g
        if not S:
            # With no curvature information the direction is just -grad, whose
            # NORM is arbitrary: the energy is dimensionless, so ||grad|| scales
            # like 1/||Y|| and a unit step is meaningless.  Taking step = 1 here
            # made the first trial move 3e-02 on a problem with ||Y|| = 1.5, the
            # line search failed at once, and the method returned after four
            # gradient evaluations at |Gmap| = 2.8e-01.  Use the same scale-free
            # first step SPG uses, 1/||grad||.
            d = d / max(float(g.norm()), 1e-300)

        step, accepted, E_try = 1.0, False, None
        for _ in range(40):                      # projected-arc Armijo
            Y_new = proj(Y + step * d)
            diff = Y_new - Y
            dn2 = float((diff * diff).sum())
            if dn2 == 0.0:
                break
            E_new = cnt.E(Y_new)
            E_try = E_new if E_try is None else min(E_try, E_new)
            if E_new <= E - sigma * dn2 / step:
                accepted = True
                break
            step *= 0.5
        if not accepted and S:
            # A stall on a quasi-Newton direction is usually the memory, not the
            # point: stale curvature from before an active-set change gives a
            # direction the line search cannot use.  Drop it and retry once with
            # scaled steepest descent before concluding anything.
            S, R, rho = [], [], []
            d = -g / max(float(g.norm()), 1e-300)
            step = 1.0
            for _ in range(40):
                Y_new = proj(Y + step * d)
                diff = Y_new - Y
                dn2 = float((diff * diff).sum())
                if dn2 == 0.0:
                    break
                E_new = cnt.E(Y_new)
                E_try = E_new if E_try is None else min(E_try, E_new)
                if E_new <= E - sigma * dn2 / step:
                    accepted = True
                    break
                step *= 0.5
        if not accepted:
            stop = _classify_stall(E, E_try, gwin, gmap, tol, Y.dtype)
            break

        E_new, g_new = cnt.G(Y_new)
        s_vec, y_vec = Y_new - Y, g_new - g
        sy_t = (s_vec * y_vec).sum()
        sy = float(sy_t)
        if sy > 1e-12 * float(s_vec.norm()) * float(y_vec.norm()):
            S.append(s_vec); R.append(y_vec); rho.append(1.0 / sy_t)
            if len(S) > history_size:
                S.pop(0); R.pop(0); rho.pop(0)
        Y, E, g = Y_new, E_new, g_new
        gmap = gradient_mapping(Y, g, proj)

        if trace is not None:
            row = dict(iter=it, energy=E, grad_map=gmap, step=step,
                       seconds=time.time() - t0, n_grad=cnt.n_grad,
                       n_energy=cnt.n_energy)
            if w is not None:
                row["w"] = float(w)
            trace.append(row)
        if verbose and (it % 20 == 0 or it == 1):
            print(f"    [pqn it={it:5d}] E={E:.8e} |Gmap|={gmap:.3e} "
                  f"free={int(free.sum())}/{free.numel()}")

        window.append(E)
        gwin.append(gmap)
        if len(window) > patience:
            window.pop(0)
            gwin.pop(0)
        if gmap <= tol:
            stop = "grad_map"
            break
        if _plateau(window, gwin, patience, rtol):
            stop = "plateau"
            break

    return _report(Y, E, it, stop, gmap, cnt.n_grad, cnt.n_energy, t0,
                   E0=E0, gmap0=gmap0)


# --------------------------------------------------------------------------
# L-BFGS-B, pure PyTorch
# --------------------------------------------------------------------------
def _tlbfgsb(Y, energy_fn, grad_fn, proj, *, max_iter=2000, tol=1e-9, sigma=1e-4,
             history_size=10, mask=None, bounded=True, trace=None, w=None,
             verbose=False, **_):
    """Adapter onto :func:`nsa_flow.lbfgsb.lbfgsb_minimize`.

    The bounds are the feasible set this package uses -- ``Y >= 0``, plus an
    optional fixed-support mask -- so ``proj`` is not needed for feasibility
    here; it is still passed to the certificate so the reported number is the
    same function every other loop reports.
    """
    from .lbfgsb import lbfgsb_minimize
    t0 = time.time()
    cnt = _Counter(energy_fn, grad_fn)

    # The certificate must see the SAME feasible set the solver optimises over:
    # with a fixed-support mask, an unmasked projection reports every pinned
    # coordinate with a negative gradient as a violation, and the re-solve can
    # never certify (measured: stalled at |Gmap| = 7.5e-02 with converged=False
    # on every consolidate call).  Likewise an unconstrained solve (nonneg=False)
    # must not be clamped by a hard-coded lower bound.
    if mask is not None:
        mk = mask.to(Y.dtype)
        cproj = (lambda A: proj(A) * mk)
    else:
        cproj = proj

    def cert(Yv, gv):
        return gradient_mapping(Yv, gv, cproj)

    Y0 = cproj(Y)
    E0, g0 = cnt.G(Y0)
    gmap0 = cert(Y0, g0)
    _trace_start(trace, E0, gmap0, t0, w)

    def cb(it, f, gmap, n_grad, n_fun):
        if trace is not None:
            row = dict(iter=it, energy=f, grad_map=gmap, step=float("nan"),
                       seconds=time.time() - t0, n_grad=n_grad + cnt.n_grad,
                       n_energy=n_fun + cnt.n_energy)
            if w is not None:
                row["w"] = float(w)
            trace.append(row)
        if verbose and (it % 20 == 0 or it == 1):
            print(f"    [lbfgsb it={it:5d}] E={f:.8e} |Gmap|={gmap:.3e}")

    out = lbfgsb_minimize(
        Y0, lambda Yv: cnt.G(Yv), fun=cnt.E, lower=(0.0 if bounded else None), upper=None,
        mask=mask,
        max_grad=max(max_iter - cnt.n_grad, 1), tol=tol, memory=history_size,
        sigma=sigma, certificate=cert, callback=cb, stall_slack=STALL_SLACK)

    stop = out["stop"]          # classified inside lbfgsb_minimize by the shared rule
    return _report(out["x"], out["f"], out["iters"], stop, out["grad_map"],
                   cnt.n_grad, cnt.n_energy, t0, E0=E0, gmap0=gmap0)


# --------------------------------------------------------------------------
# SciPy L-BFGS-B
# --------------------------------------------------------------------------
def _lbfgsb(Y, energy_fn, grad_fn, proj, *, max_iter=2000, tol=1e-9,
            mask=None, trace=None, w=None, **_):
    t0 = time.time()
    try:
        import numpy as np
        from scipy.optimize import minimize as _sp_min
    except ImportError:                                            # pragma: no cover
        raise ImportError("optimizer='lbfgsb' requires scipy") from None

    cnt = _Counter(energy_fn, grad_fn)
    shape, dtype, device = Y.shape, Y.dtype, Y.device
    Y = proj(Y)
    E0, g0 = cnt.G(Y)
    gmap0 = gradient_mapping(Y, g0, proj)
    _trace_start(trace, E0, gmap0, t0, w)

    def f_and_g(y_flat):
        Yt = torch.as_tensor(y_flat.reshape(shape), dtype=dtype, device=device)
        E, g = cnt.G(Yt)
        return E, g.detach().cpu().numpy().astype(np.float64).ravel()

    if mask is None:
        bounds = [(0.0, None)] * Y.numel()
    else:
        bounds = [(0.0, None) if m else (0.0, 0.0)
                  for m in mask.detach().cpu().numpy().ravel()]

    res = _sp_min(f_and_g, Y.detach().cpu().numpy().astype("float64").ravel(),
                  method="L-BFGS-B", jac=True, bounds=bounds,
                  options=dict(maxfun=max_iter, maxiter=max_iter,
                               ftol=1e-16, gtol=1e-16))
    Y_opt = torch.as_tensor(res.x.reshape(shape), dtype=dtype, device=device)
    E_opt, g_opt = cnt.G(Y_opt)
    gmap = gradient_mapping(Y_opt, g_opt, proj)

    # Classify by the SHARED rule, not by SciPy's status.  SciPy's notion of
    # success is its own ftol/gtol on its own residual; mapping it straight
    # through is how this path used to report a different kind of convergence
    # from every other loop.  The certificate decides; SciPy's status only
    # distinguishes the uncertified cases.
    if gmap <= tol:
        stop = "grad_map"
    elif gmap <= STALL_SLACK * max(tol, 0.0):
        stop = "plateau"                             # at the numerical floor
    elif res.status == 1:                            # iteration / evaluation cap
        stop = "max_iter"
    else:
        stop = "line_search"
    if trace is not None:
        trace.append(dict(iter=int(res.nit), energy=float(E_opt),
                          grad_map=gmap, step=float("nan"),
                          seconds=time.time() - t0, n_grad=cnt.n_grad,
                          n_energy=cnt.n_energy,
                          **({"w": float(w)} if w is not None else {})))
    return _report(Y_opt, E_opt, int(res.nit), stop, gmap,
                   cnt.n_grad, cnt.n_energy, t0, E0=E0, gmap0=gmap0)


# --------------------------------------------------------------------------
# torch.optim.LBFGS on Y = Z^2
# --------------------------------------------------------------------------
def _torch_lbfgs(Y, energy_fn, grad_fn, proj, *, max_iter=2000, tol=1e-9,
                 history_size=10, patience=5, rtol=1e-12, mask=None,
                 trace=None, w=None, verbose=False, **_):
    t0 = time.time()
    cnt = _Counter(energy_fn, grad_fn)
    Y0 = proj(Y)
    dtype, device = Y0.dtype, Y0.device

    # Floor the initialisation away from zero, else dE/dZ = 2Z dE/dY makes every
    # zero coordinate a fixed point and the support can never grow.
    floor = _Z_FLOOR * float(Y0.abs().max().clamp_min(1e-300))
    Z = torch.nn.Parameter(torch.sqrt(Y0 + floor))

    def Y_of(Zv):
        Yv = Zv.pow(2)
        return Yv * mask if mask is not None else Yv

    opt = torch.optim.LBFGS([Z], lr=1.0, max_iter=10, history_size=history_size,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=0.0, tolerance_change=0.0)

    def closure():
        opt.zero_grad(set_to_none=False)
        Yc = Y_of(Z)
        E, g = cnt.G(Yc)
        gz = 2.0 * Z * g
        if mask is not None:
            gz = gz * mask
        Z.grad = gz.contiguous()
        return torch.as_tensor(E, dtype=dtype, device=device)

    stop, it, window, gwin = "max_iter", 0, [], []
    with torch.no_grad():
        Y_cur = Y_of(Z)
    E_cur, g_cur = cnt.G(Y_cur)
    gmap = gradient_mapping(Y_cur, g_cur, proj)
    E0, gmap0 = E_cur, gmap
    _trace_start(trace, E_cur, gmap, t0, w)

    while cnt.n_grad < max_iter:
        it += 1
        if gmap <= tol:
            stop = "grad_map"
            break
        opt.step(closure)
        with torch.no_grad():
            Y_cur = Y_of(Z)
        E_cur, g_cur = cnt.G(Y_cur)
        gmap = gradient_mapping(Y_cur, g_cur, proj)

        if trace is not None:
            row = dict(iter=it, energy=E_cur, grad_map=gmap, step=1.0,
                       seconds=time.time() - t0, n_grad=cnt.n_grad,
                       n_energy=cnt.n_energy)
            if w is not None:
                row["w"] = float(w)
            trace.append(row)
        if verbose and (it % 10 == 0 or it == 1):
            print(f"    [tlbfgs it={it:4d}] E={E_cur:.8e} |Gmap|={gmap:.3e}")

        window.append(E_cur)
        gwin.append(gmap)
        if len(window) > patience:
            window.pop(0)
            gwin.pop(0)
        if gmap <= tol:
            stop = "grad_map"
            break
        if _plateau(window, gwin, patience, rtol):
            stop = "plateau"
            break

    with torch.no_grad():
        Y_out = Y_of(Z)
    return _report(Y_out, E_cur, it, stop, gmap, cnt.n_grad, cnt.n_energy, t0,
                   E0=E0, gmap0=gmap0)


#: ``name -> (routine, supports_mask, pure_torch)``.
OPTIMIZERS = {
    "lbfgsb": (_tlbfgsb, True, True),
    "pqn": (_pqn, False, True),
    "spg": (_spg, False, True),
    "fista": (_fista, False, True),
    "scipy_lbfgsb": (_lbfgsb, True, False),
    "torch_lbfgs": (_torch_lbfgs, True, True),
}

#: Historical spellings kept working.
_ALIASES = {"torch-lbfgs": "torch_lbfgs", "lbfgs": "lbfgsb", "l-bfgs-b": "lbfgsb",
            "scipy-lbfgsb": "scipy_lbfgsb", "SPG": "spg"}


def optimizer_names():
    """The valid ``optimizer`` selectors, in a stable order."""
    return tuple(OPTIMIZERS)


def minimise(Y0, energy_fn, grad_fn, proj=project_nonneg, *, optimizer="spg",
             max_iter=2000, tol=1e-9, mask=None, caller="", **kwargs):
    """See module docstring.  ``proj=None`` means unconstrained."""
    bounded = proj is not None
    if proj is None:
        proj = lambda A: A                                         # noqa: E731
    return _minimise(Y0, energy_fn, grad_fn, proj, optimizer=optimizer,
                     max_iter=max_iter, tol=tol, mask=mask, caller=caller,
                     bounded=bounded, **kwargs)


def _minimise(Y0, energy_fn, grad_fn, proj, *, optimizer, max_iter, tol, mask,
              caller, bounded, **kwargs):
    """Minimise ``E`` over ``{Y : proj(Y) == Y}`` and report under the contract.

    ``max_iter`` is a cap on **gradient evaluations**.  ``mask``, where the
    algorithm supports it, restricts the free variables (used by the signed
    solver's consolidation re-solve); where it does not, it is folded into
    ``proj`` instead, which is equivalent and keeps every algorithm usable.

    Warns -- never silently -- when the returned point is not stationary.
    """
    name = _ALIASES.get(optimizer, optimizer)
    if name not in OPTIMIZERS:
        raise ValueError(
            f"optimizer must be one of {optimizer_names()} "
            f"(aliases: {tuple(_ALIASES)}); got {optimizer!r}")
    fn, supports_mask, _ = OPTIMIZERS[name]
    if name == "torch_lbfgs":
        warnings.warn(
            "optimizer='torch_lbfgs' is deprecated: across every configuration in "
            "experiments/optimizer_study.py it failed to certify convergence, "
            "landed above the best energy, and cost 10-100x the alternatives. "
            "Use the default ('lbfgsb', pure-PyTorch L-BFGS-B) or 'fista'.",
            DeprecationWarning, stacklevel=3)

    if mask is not None and not supports_mask:
        base_proj = proj
        proj = lambda A: base_proj(A) * mask                       # noqa: E731
        Y0 = Y0 * mask
        mask_arg = None
    else:
        mask_arg = mask

    if name == "lbfgsb":
        kwargs["bounded"] = bounded
    rep = fn(Y0, energy_fn, grad_fn, proj, max_iter=max_iter, tol=tol,
             mask=mask_arg, **kwargs)
    rep["optimizer"] = name

    E0, E1 = rep["energy_start"], rep["energy"]
    made_progress = (not math.isfinite(E0)) or (E0 - E1) > 64.0 * float(
        torch.finfo(Y0.dtype).eps) * (1.0 + abs(E0))
    if not made_progress and rep["stop"] != "grad_map":
        warnings.warn(
            f"{caller or 'nsa_flow'}: {name} returned essentially its "
            f"initialisation -- energy {E0:.6e} -> {E1:.6e} over "
            f"{rep['n_grad']} gradient evaluation(s), |Gmap| {rep['grad_map_start']:.2e}"
            f" -> {rep['grad_map']:.2e}. Either the starting point is already "
            "optimal, or the solve is trapped; result['converged'] and "
            "result['certificate'] say which.",
            RuntimeWarning, stacklevel=3)
    if rep["stop"] != "grad_map" and rep["grad_map"] > max(tol, 0.0) * STALL_SLACK:
        warnings.warn(
            f"{caller or 'nsa_flow'}: {name} stopped with stop_reason="
            f"{rep['stop']!r} at |Gmap|={rep['grad_map']:.2e} against tol="
            f"{tol:.1e} after {rep['n_grad']} gradient evaluations; the returned "
            "point is NOT stationary and result['converged'] is False.  Raise "
            "max_iter, loosen tol, or inspect stop_reason and grad_map.",
            RuntimeWarning, stacklevel=3)
    return rep
