"""NSA-Flow energy: a scale-invariant Stiefel-approximation term plus data fidelity.

All functions accept ``[p, k]`` or batched ``[B, p, k]`` tensors and are autograd
compatible, but closed-form gradients are provided because the solver does not
need a tape.

Notation
--------
``S = Y'Y`` (Gram), ``t = tr S = ||Y||_F^2``, ``G = S / t`` (normalised Gram,
``tr G = 1``), ``lambda_i`` the eigenvalues of ``G``.
"""
import torch

__all__ = [
    "gram", "stiefel_defect", "stiefel_defect_normalised", "grad_stiefel_defect",
    "fidelity", "grad_fidelity", "energy", "grad_energy",
    "effective_rank", "defect_floor",
    "procrustes_rotation", "aligned_target",
]


def gram(Y):
    return Y.transpose(-2, -1) @ Y


def _trace(S):
    return S.diagonal(dim1=-2, dim2=-1).sum(-1)


def stiefel_defect(Y, eps=0.0):
    r"""``D(Y) = ||G - I/k||_F^2 = ||Y'Y||_F^2 / ||Y||_F^4 - 1/k``.

    Scale invariant, invariant under ``Y -> UYV`` for orthogonal ``U, V``, and
    zero exactly on ``R_{>0} . St(p, k)``.  Bounds: ``0 <= D <= 1 - 1/k``, with
    the upper bound attained only at rank one.  If ``rank(Y) = r < k`` then
    ``D >= 1/r - 1/k``, so rank collapse is penalised rather than rewarded.

    Equivalently ``D = k Var(lambda_i) = 1/EffectiveRank - 1/k``, and it splits as

        D = sum_{i != j} G_ij^2  +  sum_i (G_ii - 1/k)^2
            \_______________/      \___________________/
              decorrelation            norm balance

    The first term alone is the older "invariant orthogonality defect"; dropping
    the second is what made that functional blind to conditioning, sensitive to
    the choice of basis, and minimised by rank-deficient matrices.

    A consequence used for interpretability: since the first sum is at most ``D``,

        max_{i != j} |<y_i, y_j>|  <=  sqrt(D) . ||Y||_F^2

    so small ``D`` gives quantitatively near-disjoint column supports when
    ``Y >= 0``.  Computed as ``||G - I/k||_F^2`` directly, which is manifestly
    non-negative and avoids the cancellation in ``||G||_F^2 - 1/k``.
    """
    k = Y.shape[-1]
    S = gram(Y)
    t = _trace(S)
    if eps:
        t = t.clamp_min(eps)
    G = S / t.reshape(*t.shape, 1, 1)
    I_k = torch.eye(k, dtype=Y.dtype, device=Y.device) / k
    return (G - I_k).pow(2).sum((-2, -1))


def defect_floor(p, k):
    """Greatest lower bound of ``D`` on ``R^{p x k}``: ``0`` if ``k <= p`` else ``1/p - 1/k``."""
    return 0.0 if k <= p else 1.0 / p - 1.0 / k


def stiefel_defect_normalised(Y, eps=0.0):
    """``D`` rescaled to ``[0, 1]`` so that ``w`` is a dimensionless convex weight.

    For ``k == 1`` the defect is identically zero and this returns zero.
    """
    k = Y.shape[-1]
    if k == 1:
        return torch.zeros(Y.shape[:-2], dtype=Y.dtype, device=Y.device)
    return stiefel_defect(Y, eps=eps) / (1.0 - 1.0 / k)


def grad_stiefel_defect(Y, eps=0.0):
    r"""``grad D = (4 / t^2) [ Y S - (N / t) Y ]`` with ``N = ||S||_F^2``.

    Satisfies ``<grad D, Y> = 0`` (Euler, ``D`` is degree-0 homogeneous), so the
    defect term cannot alter ``||Y||_F``; and ``||grad D||_F <= 8 / ||Y||_F``.
    """
    S = gram(Y)
    t = _trace(S)
    if eps:
        t = t.clamp_min(eps)
    N = (S * S).sum((-2, -1))
    t_ = t.reshape(*t.shape, 1, 1)
    N_ = N.reshape(*N.shape, 1, 1)
    return (4.0 / t_.pow(2)) * (Y @ S - (N_ / t_) * Y)


def effective_rank(Y):
    """Participation ratio ``1 / sum(lambda_i^2) = k / (k D + 1)``; in ``[1, k]``."""
    k = Y.shape[-1]
    return k / (k * stiefel_defect(Y) + 1.0)


def procrustes_rotation(X0, Y):
    r"""``argmin_{Q in O(k)} ||Y - X0 Q||_F``, the orthogonal Procrustes solution.

    With ``M = X0'Y = U Sigma V'`` the minimiser is ``Q = U V'``, and the
    attained value is ``||Y||_F^2 + ||X0||_F^2 - 2 ||M||_*`` (nuclear norm).
    Costs one ``k x k`` SVD, negligible beside the ``O(p k^2)`` solver step.

    Unique iff ``M`` has full rank, which holds generically for full-rank
    ``X0, Y``; see ``aligned_target`` for the consequence.
    """
    M = X0.transpose(-2, -1) @ Y
    U, _, Vh = torch.linalg.svd(M)
    return U @ Vh


def aligned_target(X0, Y):
    r"""``X0 Q`` for the Procrustes ``Q``: the representative of ``X0``'s
    right-``O(k)`` orbit closest to ``Y``.

    Rationale.  ``D`` is exactly right-``O(k)`` invariant, ``D(YQ) = D(Y)``, so the
    orthogonality term is blind to the rotation that the anchored fidelity
    ``||Y - X0||_F^2`` charges full price to preserve.  When only the *span* of
    ``X0`` is trustworthy -- PCA fixes its subspace by the eigenvalue gaps but its
    rotation only by the variance-ordering convention -- charging for rotation
    over-constrains the problem.  Quotienting it out leaves the anchor in place
    while giving ``D`` a whole ``O(k)`` orbit of equally faithful representatives
    to find a disjoint one in.

    ``||X0 Q||_F = ||X0||_F``, so the fidelity denominator is unchanged and the
    term stays dimensionless with the same normalisation.

    Caveat, stated because it is a genuine cost.  The aligned fidelity equals
    ``(||Y||_F^2 + ||X0||_F^2 - 2||X0'Y||_*) / ||X0||_F^2``, a difference of convex
    functions: it is neither convex nor globally differentiable, being nonsmooth
    exactly where ``X0'Y`` drops rank.  The SPG convergence theory therefore
    applies on the full-rank set rather than everywhere, unlike the anchored form.
    """
    return X0 @ procrustes_rotation(X0, Y)


def fidelity(Y, X0, denom=None, align=False):
    """``F(Y) = ||Y - X0||_F^2 / ||X0||_F^2`` -- dimensionless, zero at ``Y = X0``.

    With ``align``, ``X0`` is replaced by its Procrustes-closest rotation, making
    ``F`` a distance to ``X0``'s right-``O(k)`` orbit instead of to the point
    ``X0``; it is then zero on that whole orbit.
    """
    d = _trace(gram(X0)) if denom is None else denom
    if align:
        X0 = aligned_target(X0, Y)
    return (Y - X0).pow(2).sum((-2, -1)) / d


def grad_fidelity(Y, X0, denom=None, align=False):
    """``grad F = 2 (Y - X0) / ||X0||_F^2``, with ``X0 -> X0 Q(Y)`` under ``align``.

    The derivative of the inner minimisation vanishes at its own optimum
    (envelope theorem), so no derivative of the SVD is needed and the numerical
    hazard of differentiating repeated singular values never arises.
    """
    d = _trace(gram(X0)) if denom is None else denom
    d_ = d.reshape(*d.shape, 1, 1) if torch.is_tensor(d) else d
    if align:
        X0 = aligned_target(X0, Y)
    return 2.0 * (Y - X0) / d_


def energy(Y, X0, w=0.5, denom=None, return_parts=False, align=False):
    r"""``E_w(Y) = (1 - w) F(Y) + w Dtilde(Y)``.

    Both terms are dimensionless and ``O(1)``, so ``w in [0, 1]`` is a genuine
    convex weight requiring no data-dependent calibration constants.
    """
    f = fidelity(Y, X0, denom=denom, align=align)
    d = stiefel_defect_normalised(Y)
    tot = (1.0 - w) * f + w * d
    if return_parts:
        return tot, f, d
    return tot


def grad_energy(Y, X0, w=0.5, denom=None, align=False):
    k = Y.shape[-1]
    g = (1.0 - w) * grad_fidelity(Y, X0, denom=denom, align=align)
    if k > 1 and w != 0.0:
        g = g + (w / (1.0 - 1.0 / k)) * grad_stiefel_defect(Y)
    return g


def value_and_grad(Y, X0, w=0.5, denom=None, inv_k=None, eye_k=None, align=False):
    r"""Fused ``(E_w, F, Dtilde, grad E_w)`` sharing a single Gram product.

    ``energy`` and ``grad_energy`` each form ``Y'Y``; computing them together
    halves the work and, more importantly for small matrices, roughly halves the
    number of kernel launches.  ``inv_k`` and ``eye_k`` may be supplied by the
    caller to hoist per-call allocations out of a loop.
    """
    k = Y.shape[-1]
    S = gram(Y)
    t = _trace(S)
    t_ = t.reshape(*t.shape, 1, 1)
    G = S / t_
    if eye_k is None:
        eye_k = torch.eye(k, dtype=Y.dtype, device=Y.device)
    Gc = G - eye_k / k
    D = Gc.pow(2).sum((-2, -1))

    d = _trace(gram(X0)) if denom is None else denom
    d_ = d.reshape(*d.shape, 1, 1) if torch.is_tensor(d) else d
    R = Y - (aligned_target(X0, Y) if align else X0)
    F = R.pow(2).sum((-2, -1)) / d

    if k == 1:
        Dn = torch.zeros_like(F)
        Etot = (1.0 - w) * F
        grad = (1.0 - w) * (2.0 * R / d_)
        return Etot, F, Dn, grad

    scale = 1.0 / (1.0 - 1.0 / k) if inv_k is None else inv_k
    Dn = D * scale
    Etot = (1.0 - w) * F + w * Dn
    # grad D = (4/t^2)[Y S - (N/t) Y] = (4/t)[Y G - (||G||_F^2) Y]
    N2 = (G * G).sum((-2, -1)).reshape(*t.shape, 1, 1)
    grad = (1.0 - w) * (2.0 * R / d_) + (w * scale) * (4.0 / t_) * (Y @ G - N2 * Y)
    return Etot, F, Dn, grad
