r"""L-BFGS-B in pure PyTorch: bound-constrained quasi-Newton, on any device.

Byrd, Lu, Nocedal & Zhu (1995), *A limited memory algorithm for bound
constrained optimization*, SIAM J. Sci. Comput. 16(5).

Why this exists
---------------
Measured over three modes, two problem families and three values of ``w``,
SciPy's L-BFGS-B was the fastest optimiser for every NSA-Flow objective and the
only one that never settled in a worse basin (worst-case energy excess 1.6e-15
on the anchored problem, against 1.0e-02 for spectral projected gradient,
FISTA and two-metric projection alike).  It could not be the library default
for two reasons, neither of which is about the algorithm:

* SciPy is not a dependency -- ``pyproject`` requires only ``torch``;
* it is host-side float64, so every function evaluation is a device round trip
  and a GPU-resident problem cannot use it at all.

Both are properties of *that implementation*, not of the method.  This module is
the method, written against ``torch``, so it runs unmodified on CPU, CUDA and
MPS in either precision and keeps the iterate on the device throughout.

The algorithm, and why it beats projected gradient here
-------------------------------------------------------
A projected gradient method takes ``P(x - t grad)`` and is therefore limited to
a *diagonal* metric: it can only rescale the gradient uniformly.  Two-metric
projection improves on that by using a quasi-Newton metric on the free
variables, but it identifies the active set from the *current* iterate, so it
can only release one bound per iteration.  L-BFGS-B instead:

1. **Generalized Cauchy point.**  Follow the projected steepest-descent path
   ``P(x - t grad)``, which is piecewise linear in ``t``, and minimise the
   quadratic model exactly along it.  This can hit many bounds in a single
   step, so the active set is identified in one shot rather than incrementally.
   That is the property that makes it robust on these objectives, where roughly
   half the coordinates sit at zero.
2. **Subspace minimisation.**  With the active set fixed by the Cauchy point,
   minimise the model over the remaining free variables using the limited-memory
   Hessian, then truncate the step to stay feasible.
3. **Strong-Wolfe line search** along the feasible segment (both endpoints are
   feasible and the box is convex, so no projection is needed inside it), with
   projected backtracking as a fallback; every accepted step is certified by
   the shared gradient mapping.

The limited-memory Hessian is held in the compact representation

    B = theta I - W M W',    W = [Y, theta S],
    M = [[-D, L'], [L, theta S'S]]^{-1}

with ``S, Y`` the correction pairs, ``D = diag(s_i'y_i)`` and ``L_{ij} = s_i'y_j``
for ``i > j``.  Every quantity the Cauchy search needs is then a ``2m``-vector
product with ``m`` the memory size (default 10), so the breakpoint walk costs
``O(m^2)`` per breakpoint and never touches the ``p k`` iterate.

Scope
-----
Written for the feasible set this package actually uses -- ``x >= 0``, and
optionally a fixed-support mask where ``l = u = 0`` -- rather than for general
boxes.  The structure is the general one; only the bound bookkeeping is
specialised, and ``lower``/``upper`` are explicit arguments so that is easy to
widen.
"""
import math

import torch

__all__ = ["lbfgsb_minimize", "set_compile"]

try:
    from . import _native
except Exception:
    _native = None

_COMPILED = {}


def set_compile(enabled=True, backend="inductor"):
    """Compile the pure-tensor pieces of the iteration with ``torch.compile``.

    What is compiled: the subspace step (branchless) and the compact-form
    rebuild.  What is not: the Cauchy walk, whose breakpoint count is
    data-dependent and whose early exit is Python control flow.  Whether this
    is a win is a measurement, not a principle -- see ``experiments/`` -- and it
    is off by default so that a missing compiler backend never costs a user
    anything.
    """
    global _subspace_min_impl, _rebuild_K_impl
    if not enabled:
        _COMPILED.clear()
        return
    if "sub" not in _COMPILED:
        _COMPILED["sub"] = torch.compile(_subspace_min_tensor, dynamic=True,
                                         backend=backend)
        _COMPILED["K"] = torch.compile(_build_K, dynamic=True, backend=backend)


_TINY = 1e-300     # Python-float floor only; NEVER clamp a tensor with this


def _tiny(t):
    """Smallest positive normal of ``t``'s dtype.

    ``clamp_min(1e-300)`` on a float32 tensor clamps to **0.0** -- 1e-300 is
    below float32's smallest normal (1.2e-38) -- so a curvature floored that way
    and read back with ``float()`` divides by exactly zero.  Reported from a
    downstream package on a near-flat objective at iteration 25.  Every tensor
    floor here must be in the tensor's own dtype.
    """
    return float(torch.finfo(t.dtype).tiny)


def _safe_div(num, den, tiny):
    """``num / den`` with ``den`` floored away from zero, sign preserved."""
    if den >= 0.0:
        return num / max(den, tiny)
    return num / min(den, -tiny)


class _CompactLBFGS:
    """The compact limited-memory representation ``B = theta I - W M W'``.

    Stored as the ``[n, 2m]`` matrix ``W`` and the small ``[2m, 2m]`` matrix
    ``M``, both rebuilt whenever a correction pair is added or dropped.  ``m`` is
    the memory size, so ``M`` is at most ``20 x 20`` and its inverse is free
    relative to anything touching the iterate.
    """

    def __init__(self, n, memory, dtype, device):
        self.memory = memory
        self.dtype, self.device = dtype, device
        self.S = []          # correction pairs, most recent last
        self.Yv = []
        self.theta = 1.0
        self.W = torch.zeros(n, 0, dtype=dtype, device=device)
        self.M = torch.zeros(0, 0, dtype=dtype, device=device)

    @property
    def m(self):
        return len(self.S)

    def push(self, s, y):
        """Add a correction pair, skipping it if the curvature is not positive.

        The standard safeguard: ``s'y > eps ||y||^2`` keeps ``B`` positive
        definite.  On a non-convex objective the condition genuinely fails
        sometimes, and accepting the pair anyway is how a quasi-Newton method
        starts producing ascent directions.
        """
        sy = float((s * y).sum())
        yy = float((y * y).sum())
        if not (yy > 0 and sy > 2.2e-16 * yy):
            return False
        self.S.append(s)
        self.Yv.append(y)
        if len(self.S) > self.memory:
            self.S.pop(0)
            self.Yv.pop(0)
        self.theta = yy / sy
        self._rebuild()
        return True

    def reset(self):
        self.S, self.Yv = [], []
        self.theta = 1.0
        self.W = self.W[:, :0]
        self.M = self.M[:0, :0]

    def _rebuild(self):
        m = self.m
        S = torch.stack(self.S, dim=1)               # [n, m]
        Y = torch.stack(self.Yv, dim=1)              # [n, m]
        th = self.theta
        self.W = torch.cat([Y, th * S], dim=1)       # [n, 2m]

        if _native is not None and S.is_cpu:
            fn = _native.lbfgsb_build_M
        else:
            fn = _COMPILED.get("K", _build_K)
        try:
            self.M = fn(S, Y, th)
        except Exception:                            # singular: drop the memory
            self.reset()

    def Bv(self, v):
        """``B v = theta v - W M (W' v)``."""
        if self.m == 0:
            return self.theta * v
        return self.theta * v - self.W @ (self.M @ (self.W.transpose(0, 1) @ v))


def _build_K(S, Y, th):
    """``M = [[-D, L'], [L, th S'S]]^{-1}`` from the correction pairs -- pure tensor."""
    m = S.shape[1]
    SY = S.transpose(0, 1) @ Y                       # (SY)_ij = s_i'y_j
    SS = S.transpose(0, 1) @ S
    strict = torch.ones(m, m, dtype=S.dtype, device=S.device).tril(-1)
    L = SY * strict                                  # i > j
    D = torch.diag_embed(torch.diagonal(SY))
    K = torch.cat([torch.cat([-D, L.transpose(0, 1)], 1),
                   torch.cat([L, th * SS], 1)], 0)
    return torch.linalg.solve(K, torch.eye(2 * m, dtype=S.dtype, device=S.device))


def _cauchy_point(x, g, lo, hi, H, max_breakpoints=512):
    r"""Generalized Cauchy point: minimise the model along ``P(x - t g)``.

    The path is piecewise linear with a breakpoint wherever a coordinate reaches
    a bound, and the model restricted to each segment is a scalar quadratic.  We
    walk the breakpoints in increasing order, and on each segment compare the
    unconstrained minimiser ``dt_min = -f' / f''`` against the segment length; the
    first segment that contains its own minimiser is where the Cauchy point lies.

    Every update is an ``O(m^2)`` operation on ``2m``-vectors, never on the
    ``n``-vector, which is what makes the walk affordable.

    Returns ``(x_cp, c, fixed)`` with ``c = W'(x_cp - x)`` -- the quantity the
    subspace minimisation needs -- and ``fixed`` marking coordinates pinned to a
    bound by the walk.
    """
    inf = float("inf")
    tiny = _tiny(x)
    t = torch.full_like(x, inf)
    if lo is not None:
        t = torch.where(g > 0, (x - lo) / g.clamp_min(tiny), t)
    if hi is not None:
        t = torch.where(g < 0, (x - hi) / g.clamp_max(-tiny), t)
    t = torch.nan_to_num(t, nan=inf, posinf=inf, neginf=inf).clamp_min(0.0)

    moving0 = (t > 0).to(x.dtype)
    d = -g * moving0
    fixed = t <= 0                                    # already at a bound, leaving
    bound = lo if lo is not None else hi
    x_cp = x * moving0 + (bound * (1.0 - moving0) if bound is not None else 0.0)

    two_m = H.W.shape[1]
    p = H.W.transpose(0, 1) @ d if two_m else torch.zeros(0, dtype=x.dtype,
                                                          device=x.device)
    c = torch.zeros_like(p)
    fp = -float((d * d).sum())                        # f'(0)
    fpp = -H.theta * fp
    if two_m:
        fpp = fpp - float(p @ (H.M @ p))
    fpp = max(fpp, tiny)
    dt_min = _safe_div(-fp, fpp, tiny)
    t_old = 0.0

    # Only finite breakpoints of currently-moving coordinates matter, and only
    # the smallest few are ever reached; sorting them once is cheaper than
    # repeatedly scanning for the minimum.
    cand = torch.nonzero(torch.isfinite(t) & (t > 0), as_tuple=False).flatten()
    if cand.numel():
        # The walk's state (p, c, f', f'') evolves by recurrences that are all
        # cumulative sums over the breakpoints in order, so the entire walk is
        # computed at once and the first segment containing its own minimiser
        # is found by a single vectorised comparison.  The sequential Python
        # version visited ~44 breakpoints per iteration with a device sync at
        # each: 55% of a fit's wall time against 10% for the objective itself.
        order = cand[torch.argsort(t[cand])][:max_breakpoints]
        tb = t[order]                                          # [B]
        gb = g[order]
        xb = x[order]
        if lo is not None and hi is not None:
            xcp_b = torch.where(gb > 0, lo[order], hi[order])
        elif lo is not None:
            xcp_b = lo[order]
        else:
            xcp_b = hi[order]
        zb = xcp_b - xb
        t_prev = torch.cat([tb.new_zeros(1), tb[:-1]])
        dtb = tb - t_prev                                      # [B]

        if two_m:
            Wb = H.W[order]                                    # [B, 2m]
            # p_{j-1}: p before breakpoint j is processed
            p_prev = p.unsqueeze(0) + torch.cumsum(gb.unsqueeze(1) * Wb, 0) \
                - gb.unsqueeze(1) * Wb
            # c_j: c after adding dt_j p_{j-1}
            c_j = c.unsqueeze(0) + torch.cumsum(dtb.unsqueeze(1) * p_prev, 0)
            MWb = Wb @ H.M                                     # [B, 2m] (M symmetric)
            wMc = (MWb * c_j).sum(1)
            wMp = (MWb * p_prev).sum(1)
            wMw = (MWb * Wb).sum(1)
            dfpp = -H.theta * gb * gb - 2.0 * gb * wMp - gb * gb * wMw
        else:
            wMc = torch.zeros_like(gb)
            dfpp = -H.theta * gb * gb
        fpp_j = (fpp + torch.cumsum(dfpp, 0)).clamp_min(tiny)    # f'' after j
        fpp_prev = torch.cat([fpp_j.new_full((1,), fpp), fpp_j[:-1]])
        dfp = dtb * fpp_prev + gb * gb + H.theta * gb * zb - gb * wMc
        fp_j = fp + torch.cumsum(dfp, 0)                       # f' after j
        dt_min_j = -fp_j / fpp_j                               # minimiser on segment j+1
        # stop before breakpoint j if dt_min_{j-1} < dt_j (segment j holds the min)
        dt_min_prev = torch.cat([dt_min_j.new_full((1,), dt_min), dt_min_j[:-1]])
        stop_mask = (dt_min_prev < dtb) | (dtb <= 0.0)
        hit = torch.nonzero(stop_mask, as_tuple=False)
        n_fix = int(hit[0]) if hit.numel() else int(order.numel())

        if n_fix > 0:
            fixed_idx = order[:n_fix]
            d[fixed_idx] = 0.0
            x_cp[fixed_idx] = xcp_b[:n_fix]
            fixed[fixed_idx] = True
            t_old = float(tb[n_fix - 1])
            fp = float(fp_j[n_fix - 1])
            fpp = max(float(fpp_j[n_fix - 1]), tiny)
            dt_min = _safe_div(-fp, fpp, tiny)
            if two_m:
                p = p_prev[n_fix - 1] + gb[n_fix - 1] * Wb[n_fix - 1]
                c = c_j[n_fix - 1]

    dt_min = max(dt_min, 0.0)
    t_old = t_old + dt_min
    moving = (t > t_old) & (~fixed)
    x_cp = torch.where(moving, x + t_old * d, x_cp)
    if two_m:
        c = c + dt_min * p
    return x_cp, c, fixed


def _subspace_min_tensor(x, g, x_cp, c, free_f, lo, hi, W, M, th, m):
    """Pure-tensor core of :func:`_subspace_min` (compilable)."""
    z = x_cp - x
    r = g + th * z
    if m:
        r = r - W @ (M @ c)
    r = r * free_f
    if m == 0:
        d_hat = -r / th
    else:
        Wf = W * free_f.unsqueeze(1)
        v = M @ (Wf.transpose(0, 1) @ r)
        N = torch.eye(W.shape[1], dtype=x.dtype, device=x.device) \
            - (M @ (Wf.transpose(0, 1) @ Wf)) / th
        v = torch.linalg.solve(N, v)
        d_hat = -(r / th) - (Wf @ v) / (th * th)
    d_hat = d_hat * free_f
    tiny = float(torch.finfo(x.dtype).tiny)
    inf = torch.full_like(d_hat, float("inf"))
    alpha = torch.ones((), dtype=x.dtype, device=x.device)
    if lo is not None:
        lim = torch.where(d_hat < 0, (lo - x_cp) / d_hat.clamp_max(-tiny), inf)
        alpha = torch.minimum(alpha, lim.clamp_min(0.0).min())
    if hi is not None:
        lim = torch.where(d_hat > 0, (hi - x_cp) / d_hat.clamp_min(tiny), inf)
        alpha = torch.minimum(alpha, lim.clamp_min(0.0).min())
    return x_cp + alpha * d_hat


def _subspace_min(x, g, x_cp, c, fixed, lo, hi, H):
    r"""Minimise the model over the free variables, then truncate to the box.

    The direct primal method of Byrd et al. section 5.1.  With ``z = x_cp - x``
    the reduced gradient at the Cauchy point is ``r = (g + theta z - W M c)``
    restricted to the free set, and the subspace Newton step is obtained from
    the compact form by solving a ``2m x 2m`` system rather than anything of the
    size of the iterate.  Tensor core in :func:`_subspace_min_tensor`.
    """
    free = ~fixed
    if not bool(free.any()):
        return x_cp
    fn = _COMPILED.get("sub", _subspace_min_tensor)
    try:
        return fn(x, g, x_cp, c, free.to(x.dtype), lo, hi, H.W, H.M, H.theta, H.m)
    except Exception:
        if fn is not _subspace_min_tensor:
            return _subspace_min_tensor(x, g, x_cp, c, free.to(x.dtype), lo, hi,
                                        H.W, H.M, H.theta, H.m)
        return x_cp


def _cubic_min(a, fa, ga, b, fb, gb):
    """Minimiser of the cubic through ``(a, fa, ga)`` and ``(b, fb, gb)``."""
    if a == b:
        return None
    d1 = ga + gb - 3.0 * (fa - fb) / (a - b)
    q = d1 * d1 - ga * gb
    if q < 0.0:
        return None
    d2 = math.sqrt(q) * (1.0 if b > a else -1.0)
    denom = gb - ga + 2.0 * d2
    if denom == 0.0:
        return None
    cand = b - (b - a) * ((gb + d2 - d1) / denom)
    lo, hi = (a, b) if a < b else (b, a)
    return cand if lo < cand < hi else None


def _strong_wolfe(phi, a_max, f0, g0, c1=1e-4, c2=0.9, max_eval=20):
    r"""Strong Wolfe line search with cubic interpolation.

    ``phi(a) -> (f, dphi)`` along a feasible segment.  Returns
    ``(alpha, f, dphi, n_eval)``, or ``alpha = None`` if no acceptable step was
    found.

    This is what a quasi-Newton method needs and plain backtracking Armijo does
    not provide.  Armijo enforces only *sufficient decrease*; the **curvature**
    condition ``|phi'(a)| <= c2 |phi'(0)|`` is what guarantees the accepted step
    produces a correction pair with meaningful curvature information.  Without
    it the L-BFGS memory fills with short, uninformative pairs and the method
    degenerates towards steepest descent -- measured here as 272 iterations
    against SciPy's 74 on the same problem, for the same final energy.

    Because the feasible set is convex and both endpoints of the search segment
    are feasible, no projection is needed for ``a`` in ``[0, 1]``, so ``phi`` is
    genuinely smooth and the classical theory applies unmodified.
    """
    n_eval = 0
    a_prev, f_prev, d_prev = 0.0, f0, g0
    a_i = min(1.0, a_max)
    a_lo = a_hi = f_lo = d_lo = None

    while n_eval < max_eval:
        f_i, d_i = phi(a_i)
        n_eval += 1
        if not (math.isfinite(f_i) and math.isfinite(d_i)):
            # overflow along the segment: treat as "too far" and bracket below
            a_lo, f_lo, d_lo, a_hi = a_prev, f_prev, d_prev, a_i
            f_hi = d_hi = None
            break
        if f_i > f0 + c1 * a_i * g0 or (n_eval > 1 and f_i >= f_prev):
            a_lo, f_lo, d_lo, a_hi = a_prev, f_prev, d_prev, a_i
            break
        if abs(d_i) <= -c2 * g0:
            return a_i, f_i, d_i, n_eval
        if d_i >= 0.0:
            a_lo, f_lo, d_lo, a_hi = a_i, f_i, d_i, a_prev
            break
        a_prev, f_prev, d_prev = a_i, f_i, d_i
        if a_i >= a_max - 1e-16:
            return a_i, f_i, d_i, n_eval
        a_i = min(2.0 * a_i, a_max)
    else:
        return None, f0, g0, n_eval

    # zoom: cubic interpolation between the bracketing points, bisection when
    # the cubic has no interior minimiser
    f_hi = d_hi = locals().get("f_hi", None), locals().get("d_hi", None)
    f_hi, d_hi = f_hi
    while n_eval < max_eval and abs(a_hi - a_lo) > 1e-16:
        a_j = None
        if f_hi is not None:
            a_j = _cubic_min(a_lo, f_lo, d_lo, a_hi, f_hi, d_hi)
        if a_j is None:
            a_j = 0.5 * (a_lo + a_hi)
        f_j, d_j = phi(a_j)
        n_eval += 1
        if not (math.isfinite(f_j) and math.isfinite(d_j)):
            a_hi, f_hi, d_hi = a_j, None, None
            continue
        if f_j > f0 + c1 * a_j * g0 or f_j >= f_lo:
            a_hi, f_hi, d_hi = a_j, f_j, d_j
        else:
            if abs(d_j) <= -c2 * g0:
                return a_j, f_j, d_j, n_eval
            if d_j * (a_hi - a_lo) >= 0.0:
                a_hi, f_hi, d_hi = a_lo, f_lo, d_lo
            a_lo, f_lo, d_lo = a_j, f_j, d_j
    if a_lo is not None and f_lo < f0:
        return a_lo, f_lo, d_lo, n_eval
    return None, f0, g0, n_eval


def lbfgsb_minimize(x0, fun_grad, fun=None, *, lower=0.0, upper=None, mask=None,
                    max_grad=2000, tol=1e-9, memory=10, sigma=1e-4,
                    certificate=None, callback=None, max_ls=30,
                    patience=50, rtol=1e-12, stall_slack=1e3):
    r"""Minimise ``f`` over a box, in pure PyTorch.

    Parameters
    ----------
    x0 : Tensor
        Starting point, any shape; the box is applied elementwise.
    fun_grad : callable ``Tensor -> (float, Tensor)``
    fun : callable ``Tensor -> float``, optional
        Energy only, for the backtracking fallback.  Without it the fallback
        used ``fun_grad`` and threw the gradient away -- a full gradient's cost
        per trial step, and a count of "gradient evaluations" that disagreed
        with the caller's by exactly that many.
    lower, upper : float, Tensor or None
        Box bounds, broadcast to ``x0``.  ``None`` means unbounded on that side.
    mask : Tensor or None
        Boolean/float; ``0`` pins a coordinate at ``0`` (fixed support).
    max_grad : int
        Budget, in gradient evaluations.
    tol : float
        Stop when ``certificate(x, g) <= tol``.
    certificate : callable ``(x, g) -> float``
        The shared stationarity measure; see :mod:`nsa_flow.diagnostics`.

    Returns
    -------
    dict with ``x, f, n_grad, n_fun, iters, stop, grad_map``.
    """
    x = x0.detach().clone()
    shape, dtype, device = x.shape, x.dtype, x.device

    lo = None if lower is None else torch.as_tensor(
        lower, dtype=dtype, device=device).expand(shape).clone().reshape(-1)
    hi = None if upper is None else torch.as_tensor(
        upper, dtype=dtype, device=device).expand(shape).clone().reshape(-1)
    if mask is not None:
        mb = mask.reshape(-1).to(torch.bool)
        if lo is None:
            lo = torch.zeros(x.numel(), dtype=dtype, device=device)
        lo = torch.where(mb, lo, torch.zeros_like(lo))
        hi = (torch.full_like(lo, float("inf")) if hi is None else hi)
        hi = torch.where(mb, hi, torch.zeros_like(hi))

    def clip(v):
        if lo is not None:
            v = torch.maximum(v, lo)
        if hi is not None:
            v = torch.minimum(v, hi)
        return v

    n_grad = n_fun = 0

    def fg(flat):
        nonlocal n_grad
        n_grad += 1
        f, gr = fun_grad(flat.reshape(shape))
        return float(f), gr.reshape(-1)

    def f_only(flat):
        nonlocal n_fun, n_grad
        n_fun += 1
        if fun is not None:
            return float(fun(flat.reshape(shape)))
        n_grad += 1                                  # honest: it IS a gradient
        f, _ = fun_grad(flat.reshape(shape))
        return float(f)

    xf = clip(x.reshape(-1))
    f, g = fg(xf)
    H = _CompactLBFGS(xf.numel(), memory, dtype, device)
    gmap = (certificate(xf.reshape(shape), g.reshape(shape))
            if certificate else float(g.norm()))
    stop, it = "max_iter", 0
    E_win, g_win = [], []
    eps = float(torch.finfo(dtype).eps)
    noise_floor = math.sqrt(eps)

    def _stalled():
        """Certificate no longer improving across the window (>= 5 samples)."""
        if len(g_win) < 5:
            return False
        first, best = g_win[0], min(g_win)
        return math.isfinite(first) and first > 0.0 and best >= 0.9 * first

    def _classify_stall(f_try):
        """A failed line search is the numerical floor only if the certificate
        is within ``stall_slack`` of ``tol`` (or within the float noise floor)
        AND either it has stopped improving or the best trial step moved f by less
        than float noise.  Otherwise the iterate is trapped and no certificate
        is issued.  Same rule as ``nsa_flow.optim._classify_stall`` so the two
        can never disagree."""
        eps = float(torch.finfo(dtype).eps)
        noise_floor = math.sqrt(eps)
        if not (math.isfinite(gmap) and gmap <= max(stall_slack * max(tol, 0.0), noise_floor)):
            return "line_search"
        if _stalled():
            return "plateau"
        if f_try is not None and math.isfinite(f_try) \
                and (f_try - f) <= 64.0 * eps * (1.0 + abs(f)):
            return "plateau"
        return "line_search"

    while n_grad < max_grad:
        it += 1
        if gmap <= tol:
            stop = "grad_map"
            break

        if _native is not None and xf.is_cpu:
            S_stack = torch.stack(H.S, dim=1) if H.m > 0 else torch.empty(xf.numel(), 0, dtype=dtype, device=device)
            Y_stack = torch.stack(H.Yv, dim=1) if H.m > 0 else torch.empty(xf.numel(), 0, dtype=dtype, device=device)
            d, a_max, fixed = _native.lbfgsb_direction(xf, g, lo, hi, S_stack, Y_stack, H.theta, H.M, 512)
        else:
            x_cp, c, fixed = _cauchy_point(xf, g, lo, hi, H)
            x_bar = _subspace_min(xf, g, x_cp, c, fixed, lo, hi, H)
            d = x_bar - xf
            a_max = None

        if float((d * g).sum()) >= 0.0:
            # The model produced an ascent direction: the memory is stale, which
            # happens after an active-set change on a non-convex objective.
            H.reset()
            x_cp, c, fixed = _cauchy_point(xf, g, lo, hi, H)
            d = clip(x_cp) - xf
            if float((d * g).sum()) >= 0.0:
                d = -g / max(float(g.norm()), _tiny(g))
            a_max = None

        # Both xf and x_bar are feasible and the box is convex, so the whole
        # segment is feasible and phi is smooth: no projection inside the search.
        g0_dir = float((d * g).sum())
        state = {}

        def phi(a):
            xa = clip(xf + a * d)
            fa, ga = fg(xa)
            state["x"], state["g"] = xa, ga
            return fa, float((ga * d).sum())

        if a_max is None:
            a_max = 1.0
            if lo is not None:
                neg = d < 0
                if bool(neg.any()):
                    a_max = max(a_max, float(((lo - xf) / d.clamp_max(-_tiny(xf)))[neg]
                                             .clamp_min(0.0).min()))
        # Leave one evaluation in hand so the fallback's gradient still lands
        # within the budget; the cap is on gradient evaluations and it binds.
        remaining = max(max_grad - n_grad - 1, 1)
        alpha, f_new, _, _ = _strong_wolfe(phi, a_max, f, g0_dir,
                                           max_eval=min(20, remaining))

        f_try = None
        if alpha is None:
            # Fall back to projected-arc backtracking, which asks only for
            # sufficient decrease and so can still make progress where the
            # curvature condition is unreachable (late, at the floor).
            step, accepted = 1.0, False
            for _ in range(max_ls):
                x_new = clip(xf + step * d)
                diff = x_new - xf
                dn2 = float((diff * diff).sum())
                if dn2 == 0.0:
                    break
                fb = f_only(x_new)
                if not math.isfinite(fb):
                    step *= 0.5
                    continue
                f_try = fb if f_try is None else min(f_try, fb)
                if fb <= f - sigma * dn2 / step:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                stop = _classify_stall(f_try)
                break
            if n_grad >= max_grad:
                stop = "max_iter"
                break
            f_new, g_new = fg(x_new)
        else:
            x_new, g_new = state["x"], state["g"]

        if not (math.isfinite(f_new) and bool(torch.isfinite(g_new).all())):
            stop = "line_search"          # the accepted point is not finite: stop, do not step
            break
        H.push(x_new - xf, g_new - g)
        xf, f, g = x_new, f_new, g_new
        gmap = (certificate(xf.reshape(shape), g.reshape(shape))
                if certificate else float(g.norm()))
        if callback is not None:
            callback(it, f, gmap, n_grad, n_fun)
        if gmap <= tol:
            stop = "grad_map"
            break

        # plateau: energy at the float floor AND certificate not improving.
        # Without this the method spins at the float32 floor -- measured at
        # 44,979 gradient evaluations on a problem that was done after ~300.
        E_win.append(f)
        g_win.append(gmap)
        if len(E_win) > patience:
            E_win.pop(0)
            g_win.pop(0)
        if len(E_win) == patience:
            lo_, hi_ = min(E_win), max(E_win)
            if (hi_ - lo_) / (1.0 + abs(lo_)) < rtol and _stalled():
                stop = "plateau"
                break

    if stop == "max_iter" and math.isfinite(gmap) \
            and gmap <= max(stall_slack * max(tol, 0.0), noise_floor) and _stalled():
        stop = "plateau"
    return dict(x=xf.reshape(shape), f=f, n_grad=n_grad, n_fun=n_fun,
                iters=it, stop=stop, grad_map=gmap)
