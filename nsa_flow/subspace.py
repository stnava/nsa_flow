r"""Sign-blind fidelity: anchor to ``X0``'s column space, not to its entries.

Why this exists.  The anchored energy uses \|Y - X0\|_F^2, which charges the
solution for every negative entry of ``X0`` that a non-negative ``Y`` structurally
cannot match.  Those charges are constant, so they do not steer the solution --
they merely dominate the term and flatten it, and the optimum degenerates toward
``max(0, X0)``.  Measured on ADNI regional volumes, that clamp costs a factor
1.36 in reconstruction error where a basis fitted to the data costs 1.03.

For PCA input the mismatch is sharper still, because eigenvector signs are a
*convention*: there are ``2^k`` equally valid signed PCA bases for one subspace and
the routine returns an arbitrary one.  Entrywise fidelity to that arbitrary
choice is fidelity to nothing.

The fix is to anchor to what PCA actually determines.  With
``P = X0 (X0'X0)^{-1} X0'`` the orthogonal projector onto ``range(X0)``,

    F_sub(Y) = ||(I - P) Y||_F^2 / ||Y||_F^2,

the fraction of ``Y``'s energy lying outside that subspace.  ``P`` depends only on
the column space, so ``F_sub`` is invariant under ``X0 -> X0 M`` for every invertible
``M`` -- sign flips and rotations included.  It is ``0`` when ``Y`` lies in the
subspace and ``1`` when ``Y`` is orthogonal to it, matching the ``[0,1]`` endpoint
convention of the other energy terms.

Two consequences worth stating.  ``F_sub`` is degree-0 homogeneous, so
``<grad F_sub, Y> = 0`` and the term cannot change ``||Y||_F`` -- the same Euler
property the orthogonality defect has.  But that means an energy built from
``F_sub`` and a scale-invariant defect is scale-free *altogether*, so ``||Y||`` is not
determined; callers should fix the gauge, and ``nsa_flow`` does so by rescaling the
result to ``||X0||_F``.

``P`` is never formed: ``P Y = X0 solve(X0'X0, X0' Y)`` costs ``O(p k^2)``.
"""
import torch

__all__ = ["negative_mass", "SubspaceAnchor", "subspace_fidelity",
           "grad_subspace_fidelity"]


def negative_mass(X0):
    r"""``||min(0, X0)||_F / ||X0||_F`` -- how much of the target is unreachable.

    ``0`` for a non-negative target, in which case the anchored fidelity is
    exactly appropriate and nothing here is needed.  Around ``0.5`` for PCA
    loadings, whose signs are a convention.
    """
    num = X0.clamp_max(0.0).pow(2).sum((-2, -1)).sqrt()
    den = X0.pow(2).sum((-2, -1)).sqrt().clamp_min(1e-300)
    return num / den


class SubspaceAnchor:
    """Cached factorisation of ``X0'X0`` so ``P Y`` costs one solve per call."""

    __slots__ = ("X0", "chol", "rank_deficient")

    def __init__(self, X0, eps=1e-12):
        self.X0 = X0
        G = X0.transpose(-2, -1) @ X0
        k = G.shape[-1]
        eye = torch.eye(k, dtype=G.dtype, device=G.device)
        # ridge only if needed: a rank-deficient target has an ill-defined
        # projector, and silently returning a wrong one would be worse than
        # saying so via the flag
        scale = G.diagonal(dim1=-2, dim2=-1).sum(-1) / k
        if not torch.all(scale > 0):
            raise ValueError(
                "target spans no subspace (X0'X0 has zero trace), so the "
                "projector onto range(X0) is undefined; the subspace fidelity "
                "cannot be used with an all-zero target")
        self.rank_deficient = bool(torch.linalg.matrix_rank(G) < k)
        self.chol = torch.linalg.cholesky(G + (eps * scale) * eye)

    def project(self, Y):
        """``P Y`` -- the component of ``Y`` inside ``range(X0)``."""
        rhs = self.X0.transpose(-2, -1) @ Y
        coef = torch.cholesky_solve(rhs, self.chol)
        return self.X0 @ coef


def subspace_fidelity(Y, anchor, eps=1e-30):
    r"""``||(I - P) Y||_F^2 / ||Y||_F^2``, in ``[0, 1]``."""
    if not isinstance(anchor, SubspaceAnchor):
        anchor = SubspaceAnchor(anchor)
    R = Y - anchor.project(Y)
    return R.pow(2).sum((-2, -1)) / Y.pow(2).sum((-2, -1)).clamp_min(eps)


def grad_subspace_fidelity(Y, anchor, eps=1e-30):
    r"""``grad = (2 / ||Y||_F^2) [ (I - P) Y - F_sub(Y) \, Y ]``.

    Satisfies ``<grad, Y> = 0`` exactly, since ``F_sub`` is degree-0 homogeneous.
    """
    if not isinstance(anchor, SubspaceAnchor):
        anchor = SubspaceAnchor(anchor)
    n2 = Y.pow(2).sum((-2, -1), keepdim=True).clamp_min(eps)
    R = Y - anchor.project(Y)
    F = R.pow(2).sum((-2, -1), keepdim=True) / n2
    return (2.0 / n2) * (R - F * Y)
