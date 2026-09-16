r"""Signed NSA-Flow: each component a contrast of two non-negative parts.

A non-negative component cannot encode a contrast.  On the ADNI analysis matrix
the leading eigenvalue (15.6) is global size and components 2-5 (1.61, 1.38,
1.23, 1.00) are regional contrasts -- exactly the shape a single non-negative
column cannot represent -- which is the structural ceiling the purely
non-negative basis keeps hitting.

Lift instead:  V = V+ - V-,   V+ >= 0, V- >= 0,

so a component reads as "these regions minus those regions".  Both parts stay
non-negative and interpretable, and the representational capacity of a signed
basis is restored.  Write ``W = [V+ | V-]`` of shape ``[p, 2k]``.  Then

    minimise  (1-w) ||X - X V V'||_F^2 / ||X||_F^2  +  w Dtilde(W),   W >= 0

with ``V = V+ - V-``.  ``D`` is applied here to the ``2k`` *parts* rather than to ``V``.  MEASUREMENT SAYS
THAT IS THE WRONG CHOICE, and the module is kept in this form only to record why.
``D`` requires ``G_ii = 1/(2k)``, i.e. all ``2k`` parts of EQUAL norm.  But a
contrast is legitimately asymmetric -- a mostly-positive component has a small
negative lobe -- so on ADNI the part-norm ratio at ``w = 0`` is ~4e8, ``D(W) = 0.18``
against a maximum of ``0.9``, and raising ``w`` inflates the near-empty lobes to
equalise them.  Lobe overlap therefore RISES from 3e-9 to 2.55 between ``w = 0``
and ``w = 0.25``, the opposite of the intent.  It does reach 2e-3 by ``w = 0.999``,
but reconstruction error is 0.89 there, worse than the crude ``clamp(PCA)`` at 0.86.

The diagnosis is that two different requirements were conflated: "the ``k``
components are orthonormal" and "each component's two lobes are disjoint".  The
former is a statement about ``V``, so it wants ``Dtilde(V)``; the latter is a
statement about lobe pairs, so it wants a separate ``sum_i <v+_i, v-_i>`` term.
Putting ``D`` on ``W`` asks for something neither of them requires -- that every
lobe carry equal weight -- and that false requirement is what does the damage.
This is also the first setting where the norm-balance half of ``D`` is harmful;
on a purely non-negative basis it was measurably free (``exp10``).

Gradient.  With ``F(V)`` the reconstruction term, ``dF/dV+ = dF/dV`` and
``dF/dV- = -dF/dV`` by the chain rule, so the data term costs one extra sign flip
and the defect term acts on ``W`` directly.
"""
import time

import torch

from .energy import (grad_stiefel_defect, stiefel_defect,
                     stiefel_defect_normalised, effective_rank)
from .project import project_nonneg
from .reconstruct import (grad_reconstruction_fidelity, reconstruction_fidelity,
                          relax_into_nonneg)
from .solve import NSAResult

__all__ = ["nsa_flow_signed"]


def _split(W):
    k = W.shape[-1] // 2
    return W[..., :k], W[..., k:]


def nsa_flow_signed(X, k=None, w=0.5, *, init="relax", max_iter=5000, tol=None,
                    sigma=1e-4, dtype=None, device=None, verbose=False,
                    keep_trace=False):
    """Fit ``V = V+ - V-`` with ``[V+|V-] >= 0`` near-disjoint, reconstructing ``X``.

    Returns an ``NSAResult`` whose ``Y`` is the signed ``V`` of shape ``[p, k]``;
    ``parts`` holds the ``[p, 2k]`` non-negative ``W = [V+|V-]``.
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
        raise ValueError(f"X must be 2-D [n, p]; got {tuple(Xt.shape)}")
    if not torch.isfinite(Xt).all():
        raise ValueError("X contains non-finite values")
    if not (0.0 <= float(w) <= 1.0):
        raise ValueError(f"w must lie in [0, 1]; got {w}")

    n, p = Xt.shape
    S = Xt.transpose(-2, -1) @ Xt
    c = S.diagonal().sum()
    if float(c) <= 0:
        raise ValueError("X is all zeros; fidelity is undefined")
    if tol is None:
        tol = 1e-9 if Xt.dtype == torch.float64 else 1e-6

    if isinstance(init, str):
        if k is None:
            raise ValueError("give k when init is a strategy name")
        evals, evecs = torch.linalg.eigh(S)
        E = evecs[:, -k:].flip(-1)
        if init == "split":
            # The exact signed basis, losslessly: V+ = max(0,E), V- = max(0,-E).
            W = torch.cat([E.clamp_min(0.0), (-E).clamp_min(0.0)], dim=-1).clone()
        elif init == "relax":
            V0 = relax_into_nonneg(S, c, k, float(w), trS=c)
            W = torch.cat([V0.clamp_min(0.0), torch.zeros_like(V0)], dim=-1).clone()
        else:
            raise ValueError(f"unknown init strategy {init!r}")
    else:
        W = torch.as_tensor(init).to(dtype=Xt.dtype, device=Xt.device).detach().clone()
        k = W.shape[-1] // 2
    if W.shape != (p, 2 * k):
        raise ValueError(f"init shape {tuple(W.shape)} != [p, 2k] = {(p, 2 * k)}")

    inv_k = 1.0 / (1.0 - 1.0 / (2 * k))

    def parts_energy(Wv):
        Vp, Vm = _split(Wv)
        f = reconstruction_fidelity(Vp - Vm, S, c, c)
        d = stiefel_defect_normalised(Wv)
        return (1.0 - w) * f + w * d, f, d

    def parts_grad(Wv):
        Vp, Vm = _split(Wv)
        gV = (1.0 - w) * grad_reconstruction_fidelity(Vp - Vm, S, c)
        g = torch.cat([gV, -gV], dim=-1)
        if w != 0.0:
            g = g + (w * inv_k) * grad_stiefel_defect(Wv)
        return g

    W = project_nonneg(W)
    E, F, D = parts_energy(W)
    E = float(E)
    g = parts_grad(W)
    t = 1.0 / max(float(g.norm()), 1e-12)
    W_prev = g_prev = None
    trace = [] if keep_trace else None
    gmap, stop, it = float("inf"), "max_iter", 0
    t0 = time.time()

    for it in range(1, max_iter + 1):
        if W_prev is not None:
            s_ = W - W_prev
            r_ = g - g_prev
            sr = float((s_ * r_).sum())
            t = float((s_ * s_).sum()) / sr if sr > 0 else 1e12
            t = min(max(t, 1e-12), 1e12)
        accepted = False
        for _ in range(60):
            W_new = project_nonneg(W - t * g)
            d_ = W_new - W
            dn2 = float((d_ * d_).sum())
            if float(parts_energy(W_new)[0]) <= E - sigma * dn2 / t:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            stop = "line_search"
            break
        gmap = (dn2 ** 0.5) / t
        W_prev, g_prev = W, g
        W = W_new
        E, F, D = parts_energy(W)
        E = float(E)
        g = parts_grad(W)
        if trace is not None:
            trace.append(dict(iter=it, energy=E, fidelity=float(F),
                              defect=float(D), grad_map=gmap, step=t))
        if verbose and (it % max(1, max_iter // 10) == 0 or it == 1):
            print(f"    [w={w:.3f} it={it:5d}] E={E:.8e} |Gmap|={gmap:.3e}")
        if gmap <= tol:
            stop = "grad_map"
            break

    Vp, Vm = _split(W)
    V = Vp - Vm
    r = NSAResult(
        Y=V, target=None, w=float(w), energy=E, fidelity=float(F),
        defect=float(D), raw_defect=float(stiefel_defect(W)),
        effective_rank=float(effective_rank(W)),
        scale_ratio=float("nan"), iters=it, converged=stop != "max_iter",
        stop_reason=stop, grad_map=float(gmap), seconds=time.time() - t0,
        w_schedule=[float(w)], trace=trace, nonneg=True, align=False,
    )
    r["parts"] = W
    r["lobe_overlap"] = float((Vp * Vm).sum())        # -> 0 as D(W) -> 0
    return r
