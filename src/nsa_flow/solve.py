"""NSA-Flow solver: spectral projected gradient on a single, calibration-free energy.

The problem solved is

    minimise   E_w(Y) = (1 - w) ||Y - X0||_F^2 / ||X0||_F^2  +  w Orth(Y)
    subject to Y >= 0                                        (when ``nonneg``)

where ``Orth`` is one of three orthogonality functionals selected by ``orth``:

* ``"D"`` (default):  ``Dtilde = ||G - I/k||_F^2 / (1 - 1/k)`` for
  the trace-normalised Gram ``G = Y'Y / tr(Y'Y)``.
* ``"Cg"``:  ``||offdiag(V'V)||_F^2 / tr(V'V)^2``, smooth everywhere.
* ``"C"``:  mean squared cosine between column pairs, per-column normalised.

The default optimiser is L-BFGS-B implemented in pure PyTorch
(:mod:`nsa_flow.lbfgsb`); ``spg``, ``fista`` and ``pqn`` are available via
``optimizer=``.  All of them are scored by ONE certificate,
:func:`nsa_flow.diagnostics.gradient_mapping` -- scale-invariant, projected --
and stop for one of four reasons:

* ``"grad_map"``: certificate <= ``tol``  (``certificate="stationary"``).
* ``"plateau"``: energy AND certificate both stopped improving
  (``certificate="numerical_floor"``).
* ``"line_search"``: no descent step at working precision, far from
  stationarity -- NOT converged; a ``RuntimeWarning`` names the certificate.
* ``"max_iter"``: the gradient-evaluation budget ran out -- NOT converged.

``result.converged`` is true only in the first two cases.  No SVD or
eigendecomposition appears in the loop; the top-k initialisation is exact and
on-device (:mod:`nsa_flow.linalg`).
"""
import math
import time
import warnings

import torch

from .diagnostics import default_tol, make_result, orth_terms, coerce_tensor
from .energy import (energy, stiefel_defect_normalised, grad_stiefel_defect,
                     value_and_grad, aligned_target)
from .linalg import varimax
from .optim import minimise, optimizer_names
from .project import project_nonneg, simlr_sparseness
from .subspace import (SubspaceAnchor, negative_mass, subspace_fidelity,
                       grad_subspace_fidelity)

__all__ = ["nsa_flow", "NSAResult", "_nsa_flow_anchored", "retract_anchored"]

_COMPILED = None

#: The optimiser every entry point uses unless told otherwise.
#:
#: Chosen by ``experiments/optimizer_study.py`` over three modes x two problem
#: families x three values of ``w`` x two seeds, scored on median wall time,
#: median gradient evaluations, energy above the best found on that instance,
#: and how often the shared certificate was actually earned:
#:
#:     mode      optimizer   med_s  med_n_grad    med_dE    max_dE  certified
#:     anchored  fista       0.022          82  1.38e-16  1.02e-02       100%
#:     anchored  lbfgsb      0.012          65  4.66e-17  1.55e-15       100%
#:     anchored  spg         0.036         114  7.32e-19  1.02e-02       100%
#:     anchored  pqn         0.074         115  0.00e+00  1.02e-02       100%
#:     data      fista       0.057         308  1.45e-15  7.43e-14       100%
#:     data      lbfgsb      0.022         148  6.47e-16  1.43e-14       100%
#:     data      spg         0.112         306  6.27e-16  2.63e-04        83%
#:     data      pqn         0.184         269  0.00e+00  1.14e-05       100%
#:     signed    fista       0.093         370  2.71e-16  9.03e-12       100%
#:     signed    lbfgsb      0.053         252  2.55e-16  1.74e-04       100%
#:     signed    spg         0.233         718  3.30e-16  1.74e-04       100%
#:     signed    pqn         0.191         328  7.16e-18  1.74e-04       100%
#:
#: ``lbfgsb`` is the fastest and the most reliable at avoiding a bad basin, but
#: SciPy is not a dependency of this package (``pyproject`` requires only
#: ``torch``) and it is host-side float64 only, so it cannot be the default for
#: a library that has to run on a GPU.  ``fista`` is pure PyTorch, runs
#: unmodified on CPU, CUDA and MPS in either precision, earned the certificate
#: on every configuration tried, and costs 2-4x ``lbfgsb`` -- against 2-4x again
#: for ``spg`` and ``pqn``.  Its one weakness is shared with ``spg`` and
#: ``pqn``: on one anchored instance all three settled 1.0e-02 above the basin
#: ``lbfgsb`` found.  If you have SciPy, are on CPU in float64, and care about
#: that last margin, pass ``optimizer="lbfgsb"``.
#:
#: ``torch_lbfgs`` remains available and is not recommended: see
#: :mod:`nsa_flow.optim`.
DEFAULT_OPTIMIZER = "lbfgsb"

#: Default budget, in GRADIENT EVALUATIONS (see :mod:`nsa_flow.optim`).  Large
#: because the stopping rule is the certificate, not the cap: a solve that needs
#: 200 evaluations takes 200, and one that needs 12000 is not silently truncated
#: at a point whose support is still moving.
DEFAULT_MAX_GRAD_EVALS = 20000


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


def _custom_orth_vg(orth):
    """value_and_grad with a registry orthogonality term replacing Dtilde."""
    orth_val, orth_grad = orth_terms(orth)

    def vg(Y, X0, w, denom, inv_k_, eye_k, align):
        k = Y.shape[-1]
        d_ = float(denom)
        R = Y - (aligned_target(X0, Y) if align else X0)
        F = R.pow(2).sum() / d_
        Dn = (orth_val(Y) if k > 1
              else torch.zeros((), dtype=Y.dtype, device=Y.device))
        E = (1.0 - w) * F + w * Dn
        g = (1.0 - w) * (2.0 * R / d_)
        if k > 1 and w != 0.0:
            g = g + w * orth_grad(Y)
        return E, F, Dn, g
    return vg


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


def _sparseness_n_keep(p, sparseness):
    """Return ``n_keep`` for quantile ``sparseness`` (or ``None`` when inactive)."""
    if sparseness is None:
        return None
    q = float(sparseness)
    if q <= 0.0:
        return None
    q = min(q, 0.999)
    if p <= 1:
        return p
    return max(1, int(round(p * (1.0 - q))))


def _any_column_denser(Y, n_keep):
    """For 2-D ``[p, k]``, return ``bool``; for 3-D ``[B, p, k]``, return ``BoolTensor[B]``."""
    p, k = Y.shape[-2], Y.shape[-1]
    if n_keep is None or n_keep <= 0 or n_keep >= p or p <= 1 or k == 0 or Y.numel() == 0:
        if Y.ndim == 3:
            return torch.zeros(Y.shape[0], dtype=torch.bool, device=Y.device)
        return False
    mag = Y.detach().abs()
    max_mag = mag.amax(dim=-2, keepdim=True)
    active = (max_mag > 0).squeeze(-2)
    nnz = (mag > (1e-12 * max_mag)).sum(dim=-2)
    denser = (active & (nnz > n_keep)).any(dim=-1)
    if Y.ndim == 2:
        return bool(denser.tolist() if Y.device.type != "cpu" else bool(denser))
    return denser


def _lane_pre_feasible(X, nonneg):
    """Per-lane pre-retraction feasibility mask on ``X.device``."""
    p, k = X.shape[-2], X.shape[-1]
    if p == 0 or k == 0:
        return torch.zeros(X.shape[:-2], dtype=torch.bool, device=X.device)
    fin = torch.isfinite(X).all(dim=(-2, -1))
    frob_pos = torch.linalg.vector_norm(X, dim=(-2, -1)) > 0
    if nonneg:
        cols_ok = (X > 0).any(dim=-2).all(dim=-1)
    else:
        cols_ok = (X.abs() > 0).any(dim=-2).all(dim=-1)
    return fin & frob_pos & cols_ok


def _lane_post_feasible(Y):
    """Per-lane post-retraction check: finite and every column has ``||Y[..., :, j]||_2 > 0``."""
    if Y.shape[-2] == 0 or Y.shape[-1] == 0:
        return torch.zeros(Y.shape[:-2], dtype=torch.bool, device=Y.device)
    col_norms = torch.linalg.vector_norm(Y, dim=-2)
    return (torch.isfinite(col_norms) & (col_norms > 0)).all(dim=-1)


def _apply_scale_equivariant_pre(X, init_t, enabled):
    """When ``enabled=True``, divide ``X`` (and ``init_t``) per lane by ``||X_b||_F / sqrt(k)``."""
    k = X.shape[-1]
    if not enabled or k <= 0 or X.numel() == 0:
        return X, init_t, None
    frob = torch.linalg.vector_norm(X, dim=(-2, -1), keepdim=True)
    scale = frob / math.sqrt(float(k))
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    X_scaled = X / safe_scale
    init_scaled = None if (init_t is None or isinstance(init_t, str)) else (init_t / safe_scale)
    return X_scaled, init_scaled, safe_scale


def _flag_true(t):
    """Evaluate a 0-D boolean tensor without triggering ``Tensor.item`` / ``Tensor.__bool__`` on GPU."""
    return bool(t.tolist()) if t.device.type != "cpu" else bool(t)


def _retract_anchored_batched_torch(X0, w_vec, *, nonneg=True, init=None,
                                    tol=None, max_iter=2000, orth="D",
                                    sigma=1e-4, mask=None, return_stats=True):
    """Batched ``[B, p, k]`` Spectral Projected Gradient in pure PyTorch (CUDA/MPS/CPU).

    Runs all ``B`` anchored problems simultaneously in one vectorized kernel
    stream with per-batch step sizes and active masks.
    """
    B, p, k = X0.shape
    dtype, device = X0.dtype, X0.device
    tol = default_tol(dtype) if tol is None else float(tol)
    if k <= 1 or B == 0:
        Y = X0.clone()
        if not return_stats:
            return Y, None, None, None, None, None
        return Y, [0.0] * B, [0] * B, [0] * B, ["skipped"] * B, [0.0] * B

    if torch.is_tensor(w_vec):
        w_t = w_vec.to(dtype=dtype, device=device).expand(B).reshape(B)
    else:
        w_t = torch.as_tensor(w_vec, dtype=dtype, device=device).expand(B).reshape(B)
    w_3 = w_t.reshape(B, 1, 1)
    denom = X0.pow(2).sum((-2, -1))
    tiny = float(torch.finfo(dtype).tiny)
    denom_safe = denom.clamp_min(tiny)
    inv_k = 1.0 / (1.0 - 1.0 / k)
    eye_k = torch.eye(k, dtype=dtype, device=device)

    mask_t = None if mask is None else mask.to(dtype=dtype, device=device).detach()
    Y = (X0.clone() if init is None else init.to(dtype=dtype, device=device).detach().clone())
    if nonneg:
        Y = Y.clamp_min(0.0)
    if mask_t is not None:
        Y = Y * mask_t

    is_cg = (orth == "Cg")
    if is_cg:
        from .angle import gram_offdiag_defect, grad_gram_offdiag_defect

    def _vg(Yv):
        if is_cg:
            R = Yv - X0
            F = R.pow(2).sum((-2, -1)) / denom_safe
            D = gram_offdiag_defect(Yv)
            E = (1.0 - w_t) * F + w_t * D
            g = (1.0 - w_3) * (2.0 * R / denom_safe.reshape(B, 1, 1)) + w_3 * grad_gram_offdiag_defect(Yv)
            return E, g
        E, _, _, g = value_and_grad(Yv, X0, w=w_t, denom=denom_safe, inv_k=inv_k, eye_k=eye_k)
        return E, g

    def _gmap(Yv, gv):
        nY2 = Yv.pow(2).sum((-2, -1))
        nY = nY2.sqrt()
        step = Yv - nY2.reshape(B, 1, 1) * gv
        if nonneg:
            step = step.clamp_min(0.0)
        if mask_t is not None:
            step = step * mask_t
        return (step - Yv).pow(2).sum((-2, -1)).sqrt() / nY.clamp_min(tiny)

    E, g = _vg(Y)
    gmap = _gmap(Y, g)
    n_grad = torch.ones(B, dtype=torch.int64, device=device) if return_stats else None
    iters = torch.zeros(B, dtype=torch.int64, device=device) if return_stats else None
    active = (gmap > tol) & (denom > 0)
    stops = (["grad_map" if not a else "max_iter" for a in active.tolist()]
             if return_stats else None)

    gn = g.pow(2).sum((-2, -1)).sqrt().clamp_min(tiny)
    t_bb = (1.0 / gn).clamp(1e-12, 1e12)
    Y_prev = None
    g_prev = None
    eps_val = float(torch.finfo(dtype).eps)
    eps_sq = eps_val * eps_val
    stall_cap = max(1e3 * max(tol, 0.0), math.sqrt(eps_val))
    E_pat = E.clone()
    g_pat = gmap.clone()
    g_best = gmap.clone()

    has_active = _flag_true(active.any())
    for it in range(1, max_iter + 1):
        if not has_active:
            break
        if Y_prev is not None:
            s_ = Y - Y_prev
            r_ = g - g_prev
            sr = (s_ * r_).sum((-2, -1))
            ss = (s_ * s_).sum((-2, -1))
            rr = (r_ * r_).sum((-2, -1))
            t_new = (ss / sr.clamp_min(tiny)) if (it % 2 == 1) else (sr / rr.clamp_min(tiny))
            t_bb = torch.where(sr > 0, t_new, torch.full_like(t_bb, 1.0)).clamp(1e-12, 1e12)

        min_d2 = eps_sq * Y.pow(2).sum((-2, -1))

        # First trial (t_bb) succeeds almost always: evaluate value + gradient together
        # so accepted lanes need no second _vg call after the line search.
        Y_try0 = Y - t_bb.reshape(B, 1, 1) * g
        if nonneg:
            Y_try0 = Y_try0.clamp_min(0.0)
        if mask_t is not None:
            Y_try0 = Y_try0 * mask_t
        d2_0 = (Y_try0 - Y).pow(2).sum((-2, -1))
        E_try0, g_try0 = _vg(Y_try0)
        ok0 = active & (d2_0 > min_d2) & torch.isfinite(E_try0) & (E_try0 <= E - sigma * d2_0 / t_bb)
        ok0_3 = ok0.reshape(B, 1, 1)
        Y_acc = torch.where(ok0_3, Y_try0, Y)
        E_acc = torch.where(ok0, E_try0, E)
        g_acc = torch.where(ok0_3, g_try0, g)
        accepted = (~active) | ok0
        exhausted = active & (~ok0) & (d2_0 <= min_d2)
        need = active & (~accepted) & (~exhausted)

        if _flag_true(need.any()):
            t_cur = torch.where(need, t_bb * 0.5, t_bb)
            for _ in range(29):
                need = active & (~accepted) & (~exhausted)
                if not _flag_true(need.any()):
                    break
                Y_try = Y - t_cur.reshape(B, 1, 1) * g
                if nonneg:
                    Y_try = Y_try.clamp_min(0.0)
                if mask_t is not None:
                    Y_try = Y_try * mask_t
                d2 = (Y_try - Y).pow(2).sum((-2, -1))
                too_small = need & (d2 <= min_d2)
                exhausted = exhausted | too_small
                eval_lanes = need & (~too_small)
                if not _flag_true(eval_lanes.any()):
                    break
                E_try, _ = _vg(Y_try)
                ok = eval_lanes & torch.isfinite(E_try) & (E_try <= E - sigma * d2 / t_cur)
                if _flag_true(ok.any()):
                    ok3 = ok.reshape(B, 1, 1)
                    Y_acc = torch.where(ok3, Y_try, Y_acc)
                    E_acc = torch.where(ok, E_try, E_acc)
                    accepted = accepted | ok
                t_cur = torch.where(active & (~accepted) & (~exhausted), t_cur * 0.5, t_cur)

        stalled = active & (~accepted)
        if _flag_true(stalled.any()):
            if return_stats:
                st_idx = torch.nonzero(stalled, as_tuple=False).flatten()
                for b, gm_b in zip(st_idx.tolist(), gmap[st_idx].tolist()):
                    stops[b] = "plateau" if gm_b <= stall_cap else "line_search"
            active = active & (~stalled)
        if not _flag_true(active.any()):
            break

        # Recompute _vg(Y) only if any remaining active lane accepted a backtracked step (ls >= 1)
        used_bt = active & (~ok0)
        Y_prev = Y
        g_prev = g
        act3 = active.reshape(B, 1, 1)
        Y = torch.where(act3, Y_acc, Y)
        if _flag_true(used_bt.any()):
            E_new, g_new = _vg(Y)
            E = torch.where(active, E_new, E)
            g = torch.where(act3, g_new, g)
        else:
            E = torch.where(active, E_acc, E)
            g = torch.where(act3, g_acc, g)

        gmap = torch.where(active, _gmap(Y, g), gmap)
        g_best = torch.minimum(g_best, gmap)
        if return_stats:
            inc = active.to(torch.int64)
            n_grad = n_grad + inc
            iters = iters + inc

        conv = active & (gmap <= tol)
        plat = torch.zeros_like(active)
        if it % 16 == 0:
            plat = (
                active
                & (~conv)
                & (gmap <= stall_cap)
                & ((E_pat - E).abs() <= 64.0 * eps_val * (1.0 + E.abs()))
                & (g_best >= 0.9 * g_pat)
            )
            E_pat = E.clone()
            g_pat = gmap.clone()
            g_best = gmap.clone()

        active_next = active & (~conv) & (~plat)
        if return_stats:
            has_conv, has_plat, has_active = torch.stack([
                conv.any(), plat.any(), active_next.any()
            ]).tolist()
            if has_conv:
                for b in torch.nonzero(conv, as_tuple=False).flatten().tolist():
                    stops[b] = "grad_map"
            if has_plat:
                for b in torch.nonzero(plat, as_tuple=False).flatten().tolist():
                    stops[b] = "plateau"
            active = active_next
        else:
            active = active_next
            if device.type == "cpu" or (it % 4 == 0) or (it == max_iter):
                has_active = _flag_true(active.any())

    if not return_stats:
        return Y, None, None, None, None, None
    return (Y, E.tolist(), n_grad.tolist(), iters.tolist(), stops, gmap.tolist())


def _retract_anchored_2d_core(X0, wf, *, nonneg, init, tol, max_iter, orth,
                              fidelity, sparseness, n_keep, n_keep_cpp, return_info):
    """Core 2-D ``[p, k]`` retraction assuming finite/validated ``X0`` and ``wf``."""
    k = X0.shape[-1]
    if (fidelity == "anchor" and orth in ("D", "Cg")
            and X0.is_cpu and X0.dtype in (torch.float32, torch.float64)):
        from .lbfgsb import _native
        if _native is not None and hasattr(_native, "retract_anchored_cpu"):
            if k <= 1:
                if not torch.isfinite(X0).all():
                    raise ValueError("target contains non-finite values")
                Y = (X0.clone() if n_keep is None else
                     simlr_sparseness(X0, float(sparseness), nonneg=bool(nonneg),
                                      alg="hard", only_if_denser=True))
                if return_info:
                    return Y, {"converged": True, "stop_reason": "skipped", "certificate": "trivial",
                               "grad_map": 0.0, "n_grad": 0, "iters": 0,
                               "sparse_stage": False, "n_keep": n_keep}
                return Y, None
            init_t = None
            if init is not None:
                init_t = (init if isinstance(init, torch.Tensor)
                          else torch.as_tensor(init)).to(dtype=X0.dtype, device=X0.device).detach()
                if init_t.shape != X0.shape:
                    raise ValueError(f"init shape {tuple(init_t.shape)} != target shape {tuple(X0.shape)}")
            Y, _f, n_grad, _n_fun, iters, stop, gmap, sparse_stage, need_tie = _native.retract_anchored_cpu(
                X0.detach(), init_t, wf, bool(nonneg), orth, tol, int(max_iter), 10, None, n_keep_cpp)
            if need_tie and n_keep_cpp > 0:
                Ys = simlr_sparseness(Y, float(sparseness), nonneg=bool(nonneg),
                                      alg="hard", only_if_denser=True)
                mask_t = (Ys != 0).to(dtype=Y.dtype, device=Y.device)
                Y, _f2, ng2, _nf2, it2, stop, gmap, _, _ = _native.retract_anchored_cpu(
                    X0.detach(), Ys.detach(), wf, bool(nonneg), orth, tol, int(max_iter), 10, mask_t, 0)
                n_grad += ng2
                iters += it2
                sparse_stage = True
            if not return_info:
                return Y, None
            cert = ("stationary" if (stop == "grad_map" and gmap <= tol)
                    else ("numerical_floor" if stop == "plateau"
                          else ("trivial" if stop == "skipped" else "none")))
            return Y, {"converged": cert != "none", "stop_reason": stop,
                       "certificate": cert, "grad_map": float(gmap),
                       "n_grad": int(n_grad), "iters": int(iters),
                       "sparse_stage": bool(sparse_stage), "n_keep": n_keep}

    if not _flag_true(torch.isfinite(X0).all()):
        raise ValueError("target contains non-finite values")
    if k <= 1:
        Y = (X0.clone() if n_keep is None else
             simlr_sparseness(X0, float(sparseness), nonneg=bool(nonneg),
                              alg="hard", only_if_denser=True))
        if return_info:
            return Y, {"converged": True, "stop_reason": "skipped", "certificate": "trivial",
                       "grad_map": 0.0, "n_grad": 0, "iters": 0,
                       "sparse_stage": False, "n_keep": n_keep}
        return Y, None

    res = _nsa_flow_anchored(X0, w=wf, init=init, nonneg=nonneg, max_iter=max_iter,
                             tol=tol, fidelity=fidelity, orth=orth, sparseness=sparseness)
    if not return_info:
        return res.Y, None
    return res.Y, {"converged": bool(res.converged), "stop_reason": res.stop_reason,
                   "certificate": res.certificate, "grad_map": float(res.grad_map),
                   "n_grad": int(res.n_grad), "iters": int(res.iters),
                   "sparse_stage": bool(res.get("sparse_stage", False)),
                   "n_keep": res.get("n_keep", n_keep)}


def _retract_anchored_3d_core(X0, init_t, w_tensor, ws_list, *, nonneg, tol,
                              max_iter, orth, fidelity, sparseness, n_keep,
                              n_keep_cpp, return_info):
    """Core 3-D ``[B, p, k]`` retraction assuming finite/validated ``X0`` and weights."""
    B, p, k = X0.shape
    if B == 0:
        Y_empty = X0.clone()
        if not return_info:
            return Y_empty, None
        return Y_empty, {"converged": True, "stop_reasons": [], "stop_reason": "skipped",
                         "certificate": "trivial", "grad_map": 0.0, "grad_maps": [],
                         "n_grad": 0, "iters": 0, "sparse_stage": False,
                         "sparse_stages": [], "n_keep": n_keep}

    if k <= 1:
        if n_keep is None:
            Y = X0.clone()
        else:
            Y = simlr_sparseness(X0, float(sparseness), nonneg=bool(nonneg),
                                 alg="hard", only_if_denser=True)
        if return_info:
            return Y, {"converged": True, "stop_reason": "skipped",
                       "stop_reasons": ["skipped"] * B, "certificate": "trivial",
                       "grad_map": 0.0, "grad_maps": [0.0] * B,
                       "n_grad": 0, "iters": 0,
                       "sparse_stage": False, "sparse_stages": [False] * B, "n_keep": n_keep}
        return Y, None

    if (fidelity == "anchor" and orth in ("D", "Cg")
            and X0.is_cpu and X0.dtype in (torch.float32, torch.float64)):
        from .lbfgsb import _native
        if _native is not None and hasattr(_native, "retract_anchored_batched_cpu"):
            ws_cpp = ws_list if ws_list is not None else w_tensor.detach().cpu().double().flatten().tolist()
            Y, _es, n_grads, iters_v, stops, gmaps, sparse_stages, need_ties = _native.retract_anchored_batched_cpu(
                X0.detach(), init_t, ws_cpp, bool(nonneg), orth, tol, int(max_iter), 10, n_keep_cpp)
            if n_keep_cpp > 0 and any(need_ties):
                tie_idx = [b for b, t in enumerate(need_ties) if t]
                Ys_tie = simlr_sparseness(Y[tie_idx], float(sparseness), nonneg=bool(nonneg),
                                          alg="hard", only_if_denser=True)
                for i_t, b in enumerate(tie_idx):
                    wb = ws_cpp[0] if len(ws_cpp) == 1 else ws_cpp[b]
                    Ys = Ys_tie[i_t]
                    mask_t = (Ys != 0).to(dtype=Y.dtype, device=Y.device)
                    Yb, _f2, ng2, _nf2, it2, st2, gm2, _, _ = _native.retract_anchored_cpu(
                        X0[b].detach(), Ys.detach(), wb, bool(nonneg), orth, tol, int(max_iter), 10, mask_t, 0)
                    Y[b] = Yb
                    n_grads[b] += ng2
                    iters_v[b] += it2
                    stops[b] = st2
                    gmaps[b] = gm2
                    sparse_stages[b] = True
            if not return_info:
                return Y, None
            all_conv = all(s in ("grad_map", "plateau", "skipped") for s in stops)
            worst_gm = max(gmaps) if gmaps else 0.0
            return Y, {"converged": all_conv, "stop_reasons": stops,
                       "stop_reason": stops[0] if len(set(stops)) == 1 else "mixed",
                       "grad_map": float(worst_gm), "grad_maps": gmaps,
                       "n_grad": int(sum(n_grads)), "iters": int(max(iters_v) if iters_v else 0),
                       "sparse_stage": any(sparse_stages), "sparse_stages": list(sparse_stages),
                       "n_keep": n_keep}

    if fidelity == "anchor" and orth in ("D", "Cg"):
        w_vec = (w_tensor.expand(B) if w_tensor is not None
                 else torch.as_tensor(ws_list, dtype=X0.dtype, device=X0.device).expand(B))
        Y, _es, n_grads, iters_v, stops, gmaps = _retract_anchored_batched_torch(
            X0.detach(), w_vec, nonneg=nonneg, init=init_t, tol=tol,
            max_iter=int(max_iter), orth=orth, return_stats=return_info)
        sparse_stages = [False] * B if return_info else None
        if n_keep is not None and n_keep < p:
            dense_mask = _any_column_denser(Y, n_keep)
            if _flag_true(dense_mask.any()):
                Ys_batch = simlr_sparseness(Y[dense_mask], float(sparseness),
                                            nonneg=bool(nonneg), alg="hard",
                                            only_if_denser=True)
                M_batch = (Ys_batch != 0).to(dtype=Y.dtype, device=Y.device)
                X_sub = X0[dense_mask].detach()
                w_sub = w_vec[dense_mask]
                Y2, _e2, ng2, it2, st2, gm2 = _retract_anchored_batched_torch(
                    X_sub, w_sub, nonneg=nonneg, init=Ys_batch, tol=tol,
                    max_iter=int(max_iter), orth=orth, mask=M_batch,
                    return_stats=return_info)
                Y[dense_mask] = Y2 * M_batch
                if return_info:
                    dense_idx = torch.nonzero(dense_mask, as_tuple=False).flatten().tolist()
                    for idx_i, b in enumerate(dense_idx):
                        n_grads[b] += ng2[idx_i]
                        iters_v[b] += it2[idx_i]
                        stops[b] = st2[idx_i]
                        gmaps[b] = gm2[idx_i]
                        sparse_stages[b] = True
        if not return_info:
            return Y, None
        all_conv = all(s in ("grad_map", "plateau", "skipped") for s in stops)
        return Y, {"converged": all_conv, "stop_reasons": stops,
                   "stop_reason": stops[0] if len(set(stops)) == 1 else "mixed",
                   "grad_map": float(max(gmaps) if gmaps else 0.0), "grad_maps": gmaps,
                   "n_grad": int(sum(n_grads)), "iters": int(max(iters_v) if iters_v else 0),
                   "sparse_stage": any(sparse_stages), "sparse_stages": sparse_stages,
                   "n_keep": n_keep}

    # Generic fallback (subspace / C / Coff): per-item loop
    ws_py = ws_list if ws_list is not None else w_tensor.detach().double().flatten().tolist()
    ys, stops, gmaps, n_grads, iters_v, sparse_stages = [], [], [], [], [], []
    for b in range(B):
        wb = ws_py[0] if len(ws_py) == 1 else ws_py[b]
        ib = None if init_t is None else init_t[b]
        rb = _nsa_flow_anchored(X0[b], w=wb, init=ib, nonneg=nonneg, max_iter=max_iter,
                                tol=tol, fidelity=fidelity, orth=orth, sparseness=sparseness)
        ys.append(rb.Y)
        stops.append(rb.stop_reason)
        gmaps.append(float(rb.grad_map))
        n_grads.append(int(rb.n_grad))
        iters_v.append(int(rb.iters))
        sparse_stages.append(bool(rb.get("sparse_stage", False)))
    Y = torch.stack(ys, dim=0)
    if not return_info:
        return Y, None
    all_conv = all(s in ("grad_map", "plateau", "skipped") for s in stops)
    return Y, {"converged": all_conv, "stop_reasons": stops,
               "stop_reason": stops[0] if len(set(stops)) == 1 else "mixed",
               "grad_map": float(max(gmaps) if gmaps else 0.0), "grad_maps": gmaps,
               "n_grad": int(sum(n_grads)), "iters": int(max(iters_v) if iters_v else 0),
               "sparse_stage": any(sparse_stages), "sparse_stages": sparse_stages,
               "n_keep": n_keep}


def retract_anchored(target, w=0.5, *, nonneg=True, init=None, tol=None,
                     max_iter=2000, orth="D", fidelity="anchor",
                     sparseness=None, return_info=False,
                     scale_equivariant=False, allow_infeasible=False):
    r"""Zero-overhead anchored NSA-Flow retraction for 2-D ``[p, k]`` or 3-D ``[B, p, k]``.

    Designed for inner-loop proximal callers (e.g. ``pysimlr``) and batched
    neural layers.  Bypasses ``make_result`` / ``basis_report`` completely on
    the CPU native C++ fast path and returns the retracted tensor ``Y``
    directly (or ``(Y, info)`` when ``return_info=True``).

    Proximal identity
    -----------------
    For ``fidelity="anchor"``, ``orth="D"``, ``k > 1``, ``0 <= w < 1``, and
    ``c != 0``, both the Python and native C++ kernels minimise the exact same
    energy without any hidden constant factor on either term::

        E_w(Y) = (1 - w) ||Y - c||_F^2 / ||c||_F^2  +  w Dtilde(Y)

    Multiplying by ``||c||_F^2 / (2 (1 - w))`` gives the proximal form::

        (||c||_F^2 / (2 (1 - w))) E_w(Y)
            = 1/2 ||Y - c||_F^2  +  lambda * Dtilde(Y),
        with  lambda = w ||c||_F^2 / (2 (1 - w)).

    Additional options
    ------------------
    * ``scale_equivariant=True``: normalises each 2-D matrix or 3-D batch lane
      ``b`` by ``scale_b = ||X0_b||_F / sqrt(k)`` prior to solving (and scales
      ``init`` identically when given), then multiplies ``Y_b`` back by
      ``scale_b`` so ``tol`` acts identically across arbitrary input scales.
    * ``allow_infeasible=True``: instead of raising on non-finite, zero, or
      column-dead lanes, computes a per-lane ``feasible`` ``BoolTensor`` on
      ``target.device``, solves only the feasible slice, scatters results into a
      zero-initialised ``Y``, and sets ``info["feasible"] = feasible`` when
      ``return_info=True``.

    When ``k <= 1`` and ``sparseness`` is ``None`` (or ``<= 0``), skips
    optimisation immediately and returns ``target.clone()`` unchanged with
    ``stop_reason="skipped"`` and ``iters=0``.
    """
    if fidelity == "entrywise":
        fidelity = "anchor"
    if fidelity not in ("anchor", "subspace"):
        raise ValueError(f"fidelity must be 'anchor' or 'subspace'; got {fidelity!r}")
    if orth not in ("D", "Cg", "C", "Coff"):
        orth_terms(orth)  # raises ValueError with valid names

    X0 = target if isinstance(target, torch.Tensor) and torch.is_floating_point(target) else coerce_tensor(target, stacklevel=3)
    if X0.ndim not in (2, 3):
        raise ValueError(f"target must be 2-D [p, k] or 3-D [B, p, k]; got shape {tuple(X0.shape)}")

    p = X0.shape[-2]
    k = X0.shape[-1]
    n_keep = _sparseness_n_keep(p, sparseness)
    n_keep_cpp = 0 if (n_keep is None or n_keep >= p) else int(n_keep)

    if tol is None:
        tol = default_tol(X0.dtype)
    else:
        tol = float(tol)

    if X0.ndim == 2:
        wf = float(w)
        if not (0.0 <= wf <= 1.0):
            raise ValueError(f"w must lie in [0, 1]; got {w}")
        init_checked = init
        if init is not None and not isinstance(init, str):
            init_checked = (init.to(dtype=X0.dtype, device=X0.device).detach()
                            if isinstance(init, torch.Tensor)
                            else coerce_tensor(init, dtype=X0.dtype, device=X0.device, stacklevel=3))
            if init_checked.shape != X0.shape:
                raise ValueError(f"init shape {tuple(init_checked.shape)} != target shape {tuple(X0.shape)}")

        if allow_infeasible:
            feas = _lane_pre_feasible(X0, bool(nonneg))
            if init_checked is not None and not isinstance(init_checked, str):
                feas = feas & torch.isfinite(init_checked).all()
            if not _flag_true(feas):
                Y_zero = torch.zeros_like(X0)
                if not return_info:
                    return Y_zero
                return Y_zero, {"converged": False, "stop_reason": "infeasible",
                                "certificate": "none", "grad_map": float("nan"),
                                "n_grad": 0, "iters": 0, "sparse_stage": False,
                                "n_keep": n_keep, "feasible": feas}

        X_in, init_in, scale = _apply_scale_equivariant_pre(X0, init_checked, bool(scale_equivariant))
        Y, info = _retract_anchored_2d_core(
            X_in, wf, nonneg=nonneg, init=init_in, tol=tol, max_iter=max_iter,
            orth=orth, fidelity=fidelity, sparseness=sparseness,
            n_keep=n_keep, n_keep_cpp=n_keep_cpp, return_info=return_info,
        )
        if scale is not None:
            Y = Y * scale
        if allow_infeasible:
            post_ok = _lane_post_feasible(Y)
            if not _flag_true(post_ok):
                Y = torch.zeros_like(X0)
                if return_info:
                    info = {"converged": False, "stop_reason": "infeasible",
                            "certificate": "none", "grad_map": float("nan"),
                            "n_grad": int(info["n_grad"]), "iters": int(info["iters"]),
                            "sparse_stage": bool(info["sparse_stage"]),
                            "n_keep": n_keep, "feasible": post_ok}
            elif return_info:
                info["feasible"] = post_ok
        return (Y, info) if return_info else Y

    # 3-D batched [B, p, k]
    B = X0.shape[0]
    w_tensor = None
    ws_list = None
    if torch.is_tensor(w):
        w_flat = w.detach().reshape(-1)
        if w_flat.numel() not in (1, B):
            raise ValueError(f"w must be scalar or length B={B}; got length {w_flat.numel()}")
        if w_flat.numel() > 0:
            w_ok = (torch.isfinite(w_flat) & (w_flat >= 0.0) & (w_flat <= 1.0)).all()
            if not _flag_true(w_ok):
                raise ValueError(f"w must lie in [0, 1]; got {w}")
        w_tensor = w_flat.to(dtype=X0.dtype, device=X0.device)
    elif isinstance(w, (list, tuple)):
        ws_list = [float(x) for x in w]
        if len(ws_list) not in (1, B):
            raise ValueError(f"w must be scalar or length B={B}; got length {len(ws_list)}")
        for wf in ws_list:
            if not (0.0 <= wf <= 1.0):
                raise ValueError(f"w must lie in [0, 1]; got {wf}")
    else:
        wf = float(w)
        if not (0.0 <= wf <= 1.0):
            raise ValueError(f"w must lie in [0, 1]; got {w}")
        ws_list = [wf]

    init_t = None
    if init is not None:
        init_t = (init.to(dtype=X0.dtype, device=X0.device).detach()
                  if isinstance(init, torch.Tensor)
                  else coerce_tensor(init, dtype=X0.dtype, device=X0.device, stacklevel=3))
        if init_t.shape != X0.shape:
            raise ValueError(f"init shape {tuple(init_t.shape)} != target shape {tuple(X0.shape)}")

    if not allow_infeasible:
        if B > 0 and not _flag_true(torch.isfinite(X0).all()):
            raise ValueError("target contains non-finite values")
        X_in, init_in, scale = _apply_scale_equivariant_pre(X0, init_t, bool(scale_equivariant))
        Y, info = _retract_anchored_3d_core(
            X_in, init_in, w_tensor, ws_list, nonneg=nonneg, tol=tol,
            max_iter=max_iter, orth=orth, fidelity=fidelity,
            sparseness=sparseness, n_keep=n_keep, n_keep_cpp=n_keep_cpp,
            return_info=return_info,
        )
        if scale is not None:
            Y = Y * scale
        return (Y, info) if return_info else Y

    # allow_infeasible=True on 3-D [B, p, k]
    pre_feas = _lane_pre_feasible(X0, bool(nonneg))
    if init_t is not None and B > 0:
        pre_feas = pre_feas & torch.isfinite(init_t).all(dim=(-2, -1))

    if B == 0 or not _flag_true(pre_feas.any()):
        Y_zero = torch.zeros_like(X0)
        if not return_info:
            return Y_zero
        stops = ["infeasible"] * B
        return Y_zero, {
            "converged": False,
            "stop_reasons": stops,
            "stop_reason": ("infeasible" if B > 0 else "skipped"),
            "grad_map": 0.0,
            "grad_maps": [float("nan")] * B,
            "n_grad": 0,
            "iters": 0,
            "sparse_stage": False,
            "sparse_stages": [False] * B,
            "n_keep": n_keep,
            "feasible": pre_feas,
        }

    all_pre = _flag_true(pre_feas.all())
    if all_pre:
        X_sub = X0
        init_sub = init_t
        w_tensor_sub = w_tensor
        ws_list_sub = ws_list
    else:
        X_sub = X0[pre_feas]
        init_sub = None if init_t is None else init_t[pre_feas]
        if w_tensor is not None:
            w_tensor_sub = w_tensor if w_tensor.numel() == 1 else w_tensor[pre_feas]
            ws_list_sub = None
        else:
            w_tensor_sub = None
            if len(ws_list) == 1:
                ws_list_sub = ws_list
            else:
                pre_idx_w = torch.nonzero(pre_feas, as_tuple=False).flatten().tolist()
                ws_list_sub = [ws_list[i] for i in pre_idx_w]

    X_work, init_work, scale_work = _apply_scale_equivariant_pre(X_sub, init_sub, bool(scale_equivariant))
    Y_sub, info_sub = _retract_anchored_3d_core(
        X_work, init_work, w_tensor_sub, ws_list_sub, nonneg=nonneg, tol=tol,
        max_iter=max_iter, orth=orth, fidelity=fidelity,
        sparseness=sparseness, n_keep=n_keep, n_keep_cpp=n_keep_cpp,
        return_info=return_info,
    )
    if scale_work is not None:
        Y_sub = Y_sub * scale_work

    post_ok = _lane_post_feasible(Y_sub)
    all_post = _flag_true(post_ok.all())
    if not all_post:
        Y_sub = torch.where(post_ok.reshape(-1, 1, 1), Y_sub, torch.zeros_like(Y_sub))

    if all_pre:
        feasible = post_ok
        Y = Y_sub
    else:
        feasible = pre_feas.clone()
        feasible[pre_feas] = post_ok
        Y = torch.zeros_like(X0)
        Y[pre_feas] = Y_sub

    if not return_info:
        return Y

    pre_idx = list(range(B)) if all_pre else torch.nonzero(pre_feas, as_tuple=False).flatten().tolist()
    post_list = [True] * len(pre_idx) if all_post else post_ok.tolist()
    stops_full = ["infeasible"] * B
    gmaps_full = [float("nan")] * B
    sparse_full = [False] * B
    sub_stops = info_sub.get("stop_reasons", [info_sub["stop_reason"]] * len(pre_idx))
    sub_gmaps = info_sub.get("grad_maps", [info_sub["grad_map"]] * len(pre_idx))
    sub_sparse = info_sub.get("sparse_stages", [info_sub["sparse_stage"]] * len(pre_idx))

    feas_stops = []
    feas_gmaps = []
    for i, b in enumerate(pre_idx):
        if post_list[i]:
            stops_full[b] = sub_stops[i]
            gmaps_full[b] = sub_gmaps[i]
            sparse_full[b] = sub_sparse[i]
            feas_stops.append(sub_stops[i])
            feas_gmaps.append(sub_gmaps[i])

    all_conv = bool(feas_stops) and all(s in ("grad_map", "plateau", "skipped") for s in feas_stops)
    stop_summary = stops_full[0] if len(set(stops_full)) == 1 else "mixed"
    worst_gm = float(max(feas_gmaps)) if feas_gmaps else 0.0
    return Y, {
        "converged": all_conv,
        "stop_reasons": stops_full,
        "stop_reason": stop_summary,
        "grad_map": worst_gm,
        "grad_maps": gmaps_full,
        "n_grad": int(info_sub["n_grad"]),
        "iters": int(info_sub["iters"]),
        "sparse_stage": any(sparse_full),
        "sparse_stages": sparse_full,
        "n_keep": n_keep,
        "feasible": feasible,
    }


def _nsa_flow_anchored(target, w=0.5, *, init=None, nonneg=True, max_iter=None, tol=None,
                      continuation=0, w_start=0.0, sigma=1e-4, dtype=None, device=None,
                      verbose=False, keep_trace=False, compile=False, align=False,
                      fidelity="auto", neg_mass_tol=0.01, orth="D", optimizer=None,
                      sparseness=None):
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
        default is derived from the working precision -- ``1e-6`` in float64 and
        ``1e-4`` in float32 -- because a fixed ``1e-9`` is below float32 machine
        epsilon (``1.2e-7``) and so can never be met, which would silently turn
        every single-precision call into a ``max_iter`` run.
    sparseness : float in (0, 1), optional
        Column-wise quantile sparseness target (default ``None``). When given,
        selects the top ``(1 - sparseness)`` support per column after the main
        solve and re-optimises ``E_w(Y)`` restricted to that fixed support.
    dtype : torch.dtype, optional
        Working precision (default ``None`` infers from target).
    device : torch.device, str, or None, optional
        Target computing device (default ``None`` keeps computation on target's device).

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
    X0 = coerce_tensor(target, dtype=dtype, device=device, stacklevel=3)
    if X0.ndim == 3:
        if not torch.isfinite(X0).all():
            raise ValueError("target contains non-finite values")
        if fidelity == "entrywise":
            fidelity = "anchor"
        if fidelity not in ("auto", "anchor", "subspace"):
            raise ValueError("fidelity must be 'auto', 'anchor' or 'subspace'; "
                             f"got {fidelity!r}")
        t0 = time.time()
        fmode = "anchor" if fidelity == "auto" else fidelity
        max_it = DEFAULT_MAX_GRAD_EVALS if max_iter is None else int(max_iter)
        tol_v = default_tol(X0.dtype) if tol is None else float(tol)
        Y_b, info = retract_anchored(
            X0, w=w, nonneg=bool(nonneg), init=init, tol=tol_v,
            max_iter=max_it, orth=orth, fidelity=fmode,
            sparseness=sparseness, return_info=True)
        stop_b = info.get("stop_reason", "grad_map")
        if stop_b == "mixed":
            stop_b = "grad_map" if info.get("converged") else "max_iter"
        res = NSAResult(
            Y=Y_b, w=(float(w) if not (torch.is_tensor(w) or isinstance(w, (list, tuple))) else w),
            orth=str(orth), mode="anchored",
            optimizer=DEFAULT_OPTIMIZER if optimizer is None else str(optimizer),
            energy=float("nan"), fidelity=float("nan"), defect=float("nan"),
            iters=int(info.get("iters", 0)), stop_reason=str(stop_b),
            grad_map=float(info.get("grad_map", 0.0)), tol=tol_v,
            converged=bool(info.get("converged", False)),
            certificate=("trivial" if stop_b == "skipped"
                         else ("stationary" if info.get("converged") else "none")),
            seconds=time.time() - t0,
            n_grad=int(info.get("n_grad", 0)), n_energy=0,
            fidelity_mode=fmode, fidelity_requested=fidelity,
            nonneg=bool(nonneg), align=bool(align), target=X0,
            batched=True,
            sparse_stage=bool(info.get("sparse_stage", False)),
            n_keep=info.get("n_keep", None),
        )
        if "stop_reasons" in info:
            res["stop_reasons"] = info["stop_reasons"]
            res["grad_maps"] = info["grad_maps"]
        if "sparse_stages" in info:
            res["sparse_stages"] = info["sparse_stages"]
        return res
    if X0.ndim != 2:
        raise ValueError(f"target must be 2-D [p, k] or 3-D [B, p, k]; got shape {tuple(X0.shape)}")
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
    if not torch.isfinite(X0).all():
        raise ValueError("target contains non-finite values")

    if tol is None:
        tol = default_tol(X0.dtype)
    optimizer = DEFAULT_OPTIMIZER if optimizer is None else optimizer
    max_iter = DEFAULT_MAX_GRAD_EVALS if max_iter is None else int(max_iter)

    if fidelity == "entrywise":
        fidelity = "anchor"
    if fidelity not in ("auto", "anchor", "subspace"):
        raise ValueError("fidelity must be 'auto', 'anchor' or 'subspace'; "
                         f"got {fidelity!r}")
    requested = fidelity
    orth_terms(orth)

    p, k = X0.shape
    denom = X0.pow(2).sum()
    if float(denom) <= 0:
        raise ValueError("target is all zeros; fidelity is undefined")

    if k <= 1:
        # Orthogonality between columns is vacuous for a single column (k == 1):
        # skip optimisation and return the input unchanged without wasting work.
        t0 = time.time()
        Y = X0.clone()
        ws = ([float(w)] if continuation <= 0 else
              torch.linspace(float(w_start), float(w), continuation + 1).tolist())
        nm = float(negative_mass(X0))
        clamp_ref = X0.clamp_min(0.0)
        clamp_dist = float((Y - clamp_ref).norm() / clamp_ref.norm().clamp_min(torch.finfo(Y.dtype).tiny))
        fmode = "anchor" if fidelity == "auto" else fidelity
        return make_result(
            NSAResult, Y=Y, w=w, orth=orth, mode="anchored",
            optimizer=optimizer, energy=0.0, fidelity=0.0,
            defect=0.0, iters=0, stop="skipped", grad_map=0.0, tol=tol,
            seconds=time.time() - t0,
            energy_start=0.0,
            grad_map_start=0.0,
            target=X0, align=bool(align), nonneg=bool(nonneg),
            fidelity_mode=fmode, fidelity_requested=requested,
            target_negative_mass=nm, clamp_distance=clamp_dist,
            scale_ratio=1.0,
            n_grad=0, n_energy=0, w_schedule=ws,
            trace=([] if keep_trace else None),
        )

    if float(w) == 1.0:
        warnings.warn(
            "w=1 drops the fidelity term, and D is scale-invariant, so the scale "
            "of the returned Y is unconstrained and arbitrary (observed "
            "||Y||/||X0|| up to ~1e4). Every scaled Stiefel matrix is optimal, so "
            "nothing selects among clusterings either. Use w slightly below 1, or "
            "rescale the result yourself; result['scale_ratio'] reports the drift.",
            RuntimeWarning, stacklevel=2)

    neg_mass = None
    if fidelity == "auto":
        neg_mass = float(negative_mass(X0))
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

    if init is None:
        if fidelity == "subspace" and max_iter > 0 and float(w) > 0.0 and k > 1:
            # Subspace fidelity depends only on range(X0), so any right-O(k)
            # rotation X0 @ R is an equally faithful starting point inside
            # range(X0). Rotating via Varimax breaks the Perron-Frobenius
            # global mean mode before non-negative projection.
            Y = varimax(X0)
            if nonneg:
                Y = Y.clamp_min(0.0).clone()
        else:
            Y = X0.clone()
    elif isinstance(init, str):
        if init == "varimax":
            Y = varimax(X0)
            if nonneg:
                Y = Y.clamp_min(0.0).clone()
        else:
            raise ValueError(f"Unknown anchored init strategy {init!r}; choose 'varimax' or a Tensor")
    else:
        Y = torch.as_tensor(init).to(dtype=X0.dtype, device=X0.device).detach().clone()
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
        # "D" has a fused kernel that shares one Gram product between the value
        # and the gradient; everything else goes through the shared registry.
        vg = _fused(compile) if orth == "D" else _custom_orth_vg(orth)

    trace = [] if keep_trace else None
    t0 = time.time()
    proj = project_nonneg if nonneg else None      # None == unconstrained
    eye_k = torch.eye(k, dtype=X0.dtype, device=X0.device)
    inv_k = 1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0
    total_iters = n_grad = n_energy = 0
    stop, gmap = "max_iter", float("inf")

    for wi in ws:
        if verbose:
            print(f"  continuation step w={wi:.4f}")

        def _anc_energy(Yc, _w=wi):
            return vg(Yc, X0, _w, denom, inv_k, eye_k, align)[0]

        def _anc_grad_and_energy(Yc, _w=wi):
            E_v, _, _, g_v = vg(Yc, X0, _w, denom, inv_k, eye_k, align)
            return E_v, g_v

        problem_spec = {
            "mode": "anchored",
            "target": X0,
            "denom": float(denom),
            "w": float(wi),
            "orth": orth,
            "fidelity": fidelity,
            "chol": (anchor.chol if fidelity == "subspace" else None),
            "align": align,
        }
        rep = minimise(Y, _anc_energy, _anc_grad_and_energy, proj,
                       optimizer=optimizer, max_iter=max_iter, tol=tol,
                       sigma=sigma, verbose=verbose, trace=trace,
                       caller="nsa_flow (anchored)", w=wi,
                       problem_spec=problem_spec)
        Y, stop, gmap = rep.Y, rep.stop, rep.grad_map
        total_iters += rep.iters
        n_grad += rep.n_grad
        n_energy += rep.n_energy

    n_keep = _sparseness_n_keep(p, sparseness)
    sparse_stage = False
    if n_keep is not None and _any_column_denser(Y, n_keep):
        q_val = float(sparseness)
        Ys = simlr_sparseness(Y, q_val, nonneg=bool(nonneg), alg="hard",
                              only_if_denser=True)
        mask = (Ys != 0).to(Y.dtype)
        wi_last = ws[-1]

        def _anc_energy_sp(Yc, _w=wi_last):
            return vg(Yc, X0, _w, denom, inv_k, eye_k, align)[0]

        def _anc_grad_and_energy_sp(Yc, _w=wi_last):
            E_v, _, _, g_v = vg(Yc, X0, _w, denom, inv_k, eye_k, align)
            return E_v, g_v

        problem_spec_sp = {
            "mode": "anchored",
            "target": X0,
            "denom": float(denom),
            "w": float(wi_last),
            "orth": orth,
            "fidelity": fidelity,
            "chol": (anchor.chol if fidelity == "subspace" else None),
            "align": align,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rep_sp = minimise(
                Ys, _anc_energy_sp, _anc_grad_and_energy_sp, proj,
                optimizer=optimizer, max_iter=max_iter, tol=tol,
                sigma=sigma, mask=mask, verbose=False, trace=None,
                caller="nsa_flow (anchored, sparse-support)", w=wi_last,
                problem_spec=problem_spec_sp)
        Y = rep_sp.Y * mask
        stop, gmap = rep_sp.stop, rep_sp.grad_map
        total_iters += rep_sp.iters
        n_grad += rep_sp.n_grad
        n_energy += rep_sp.n_energy
        sparse_stage = True

    if fidelity == "subspace":
        # both terms are degree-0 homogeneous, so the scale is a free gauge;
        # fix it at the target's scale rather than leaving it arbitrary
        nY = Y.norm()
        Y = torch.where(nY > 0, Y * (X0.norm() / nY.clamp_min(torch.finfo(Y.dtype).tiny)), Y)
        tot, f, dd = vg(Y, X0, float(w), denom, inv_k, eye_k, False)[:3]
    else:
        tot, f, dd = energy(Y, X0, w=float(w), denom=denom, return_parts=True,
                            align=align)
    clamp_ref = X0.clamp_min(0.0)
    clamp_dist = float((Y - clamp_ref).norm() / clamp_ref.norm().clamp_min(torch.finfo(Y.dtype).tiny))

    res = make_result(
        NSAResult, Y=Y, w=w, orth=orth, mode="anchored",
        optimizer=rep["optimizer"], energy=float(tot), fidelity=float(f),
        defect=float(dd), iters=total_iters, stop=stop, grad_map=gmap, tol=tol,
        seconds=time.time() - t0,
        energy_start=rep["energy_start"],
        grad_map_start=rep["grad_map_start"],
        target=X0, align=bool(align), nonneg=bool(nonneg),
        fidelity_mode=fidelity, fidelity_requested=requested,
        target_negative_mass=(neg_mass if neg_mass is not None else float(negative_mass(X0))), clamp_distance=clamp_dist,
        scale_ratio=float(Y.norm() / X0.norm()),
        n_grad=n_grad, n_energy=n_energy, w_schedule=ws, trace=trace,
    )
    res["sparse_stage"] = sparse_stage
    res["n_keep"] = n_keep
    return res


def nsa_flow(data_or_target, k=None, w=0.5, *, mode="auto", nonneg=None,
             consolidate=False, optimizer=None, init=None, max_iter=None,
             tol=None, fidelity="auto", sparseness=None, dtype=None,
             device=None, **kwargs):
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
    optimizer : {"lbfgsb", "lbfgsb_cpp", "spg", "scipy_lbfgsb"}, default "lbfgsb"
        Optimization algorithm:
        - "lbfgsb": Bound-constrained L-BFGS-B with C++ acceleration when available (default).
        - "lbfgsb_cpp": Explicit C++ L-BFGS-B kernel.
        - "spg": Spectral projected gradient (alternating Barzilai-Borwein).
        - "scipy_lbfgsb": SciPy L-BFGS-B reference solver.
    init : str or Tensor, optional
        Initial point strategy or tensor.
    max_iter : int, optional
        Maximum gradient evaluations (default 20000).
    tol : float, optional
        Stationarity tolerance on the scale-invariant gradient mapping.
    fidelity : {"auto", "anchor", "subspace"}, default "auto"
        Anchored mode only: which notion of "close to the target" to use.
        ``"anchor"`` is the entrywise Euclidean distance -- the one that makes
        the anchored solve a proximal operator, which is what a proximal-
        gradient outer loop (e.g. SiMLR) requires.  ``"auto"`` switches to the
        sign-blind ``"subspace"`` fidelity when the target has negative mass
        > ``neg_mass_tol``, which is right for one-shot basis recovery from PCA
        loadings and wrong for a prox.  Explicit here, not in ``**kwargs``,
        because downstream code detects the capability by signature.
    sparseness : float in (0, 1), optional
        Column-wise quantile sparseness target (default ``None``).
    dtype : torch.dtype, optional
        Working precision (default ``None`` infers from input).
    device : torch.device, str, or None, optional
        Target computing device. If None, computation stays on the device of
        the input tensor. If explicitly specified, moves inputs and executes
        all downstream solver routines on that device.
    **kwargs :
        Additional arguments forwarded to the selected solver.

    Returns
    -------
    NSAResult
        Result dictionary-like object with `.Y`, `.energy`, `.fidelity`, `.defect`,
        `.iters`, `.stop_reason`, `.converged`, and metadata.
    """
    X = coerce_tensor(data_or_target, dtype=dtype, device=device, stacklevel=3)
    if X.ndim == 3:
        if mode in ("auto", "anchored", "target") and k is None:
            return _nsa_flow_anchored(X, w=w, nonneg=(True if nonneg is None else nonneg),
                                      init=init, fidelity=fidelity,
                                      max_iter=(DEFAULT_MAX_GRAD_EVALS if max_iter is None else int(max_iter)),
                                      tol=tol, optimizer=(DEFAULT_OPTIMIZER if optimizer is None else optimizer),
                                      sparseness=sparseness, dtype=dtype, device=device, **kwargs)
        raise ValueError(f"3-D [B, p, k] input is supported in anchored mode only; got mode={mode!r}, k={k!r}")
    if X.ndim != 2:
        raise ValueError(f"Input must be 2-D [n, p] or [p, k] (or 3-D [B, p, k] for anchored mode); got shape {tuple(X.shape)}")

    optimizer = DEFAULT_OPTIMIZER if optimizer is None else optimizer

    if fidelity == "entrywise":
        fidelity = "anchor"

    # ``signed=`` and ``nonneg=`` are mode selectors.  An explicit
    # ``nonneg=True`` with ``k`` means "I want a non-negative basis for this
    # data": route to the data-reconstruction solver, whose objective
    # ||X - X V V'||^2 with V >= 0 is well defined on signed X.  An explicit
    # ``nonneg=False`` means the signed lifting.  ``None`` (default) lets the
    # data's sign decide, and in anchored mode means ``nonneg=True``.
    if kwargs.pop("signed", False):
        mode = "signed"
    if mode == "auto" and k is not None and nonneg is not None:
        mode = "data" if nonneg else "signed"

    if mode == "auto":
        mode = ("anchored" if k is None else
                ("signed" if float(X.min()) < -1e-12 else "data"))

    # One budget rule for every mode and every optimiser.  ``max_iter`` counts
    # GRADIENT EVALUATIONS (see nsa_flow.optim), which is the only unit in which
    # an SPG step and an L-BFGS step cost the same thing.
    iter_cap = DEFAULT_MAX_GRAD_EVALS if max_iter is None else int(max_iter)

    if fidelity != "auto" and mode not in ("anchored", "target"):
        raise ValueError(
            f"fidelity={fidelity!r} applies to the anchored mode only; "
            f"mode={mode!r} uses the data reconstruction term.")

    if mode in ("data", "nonneg"):
        from .reconstruct import nsa_flow_data
        init_strat = "clamp" if init in (None, "auto") else init
        return nsa_flow_data(X, k=k, w=w, init=init_strat, max_iter=iter_cap,
                             tol=tol, optimizer=optimizer, sparseness=sparseness,
                             dtype=dtype, device=device, **kwargs)

    if mode in ("signed", "contrast"):
        from .signed import nsa_flow_signed
        init_strat = init if init is not None else "auto"
        return nsa_flow_signed(X, k=k, w=w, init=init_strat, consolidate=consolidate,
                               max_iter=iter_cap, tol=tol, optimizer=optimizer,
                               sparseness=sparseness, dtype=dtype, device=device, **kwargs)

    if mode in ("anchored", "target"):
        return _nsa_flow_anchored(X, w=w, nonneg=(True if nonneg is None else nonneg),
                                  init=init, fidelity=fidelity,
                                  max_iter=iter_cap, tol=tol, optimizer=optimizer,
                                  sparseness=sparseness, dtype=dtype, device=device, **kwargs)

    raise ValueError(
        f"Unknown mode {mode!r}; choose 'auto', 'data', 'signed' or 'anchored'")

