"""Projections used by NSA-Flow.

The polar factor is computed with an explicit, numerically robust derivative.
Routing it through ``torch.linalg.svd`` and letting autograd differentiate the
factorisation fails exactly where the polar factor is best behaved: SVD backward
divides by ``sigma_i^2 - sigma_j^2``, which vanishes whenever two singular values
coincide -- as they do, for instance, at any orthogonal initialisation.  The
polar factor itself is smooth wherever ``Y`` has full column rank; its derivative
divides by ``h_i + h_j > 0`` instead, and so never degenerates.
"""
import torch

__all__ = ["project_nonneg", "project_scaled_stiefel", "polar_factor"]


def project_nonneg(Y):
    """Euclidean projection onto ``{Y >= 0}``."""
    return Y.clamp_min(0.0)


class _PolarFactor(torch.autograd.Function):
    r"""``U`` in the polar decomposition ``Y = U H``, ``U'U = I``, ``H = (Y'Y)^{1/2}``.

    Forward: eigendecompose ``S = Y'Y = Q diag(lambda) Q'`` and set
    ``U = Y Q diag(lambda^{-1/2}) Q'``.

    Backward: differentiating ``Y = U H`` gives
    ``dU = U Omega + (I - U U') dY H^{-1}`` with ``Omega`` skew solving the
    Sylvester equation ``Omega H + H Omega = U' dY - dY' U``.  Taking adjoints,

        Ybar = 2 U Psi + (I - U U') Ubar H^{-1},
        Psi H + H Psi = skew(U' Ubar),

    which in the eigenbasis of ``H`` is ``Psi_ij = R_ij / (h_i + h_j)`` with
    ``h_i = sqrt(lambda_i) > 0``.  The denominator is a *sum* of singular values,
    so repeated singular values are harmless.
    """

    @staticmethod
    def forward(ctx, Y, eps):
        S = Y.transpose(-2, -1) @ Y
        lam, Q = torch.linalg.eigh(S)
        lam = lam.clamp_min(eps)
        h = lam.sqrt()
        inv_h = h.reciprocal()
        H_inv = (Q * inv_h.unsqueeze(-2)) @ Q.transpose(-2, -1)
        U = Y @ H_inv
        ctx.save_for_backward(U, Q, h)
        return U

    @staticmethod
    def backward(ctx, U_bar):
        U, Q, h = ctx.saved_tensors
        B = U.transpose(-2, -1) @ U_bar
        R = 0.5 * (B - B.transpose(-2, -1))                     # skew part
        R_t = Q.transpose(-2, -1) @ R @ Q                       # eigenbasis of H
        denom = h.unsqueeze(-1) + h.unsqueeze(-2)               # h_i + h_j > 0
        Psi = Q @ (R_t / denom) @ Q.transpose(-2, -1)
        H_inv = (Q * h.reciprocal().unsqueeze(-2)) @ Q.transpose(-2, -1)
        tangential = U_bar - U @ (U.transpose(-2, -1) @ U_bar)   # (I - UU') Ubar
        return 2.0 * (U @ Psi) + tangential @ H_inv, None


def polar_factor(Y, eps=1e-30):
    """Orthonormal polar factor of ``Y`` (columns orthonormal), safely differentiable.

    Requires ``Y`` to have full column rank; ``eps`` floors the Gram eigenvalues
    so that a numerically rank-deficient input degrades gracefully rather than
    producing ``inf``.
    """
    return _PolarFactor.apply(Y, eps)


def project_scaled_stiefel(Y, eps=1e-30):
    r"""Euclidean projection onto ``R_{>0} . St(p, k)`` -- the zero set of ``D``.

    ``argmin_{c > 0, Q'Q = I} ||Y - cQ||_F^2`` is attained at ``Q = U`` (the polar
    factor) and ``c = (sum_i sigma_i) / k = <U, Y> / k``, giving

        P(Y) = (<U, Y> / k) U .

    This is *not* ``(||Y||_F / sqrt(k)) U``: rescaling the polar factor to
    preserve the Frobenius norm is a different, non-optimal map.
    """
    k = Y.shape[-1]
    U = polar_factor(Y, eps=eps)
    c = (U * Y).sum((-2, -1)) / k
    return c.reshape(*c.shape, 1, 1) * U
