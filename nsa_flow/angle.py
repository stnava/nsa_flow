r"""Orthogonality, not orthonormality: the squared-cosine defect.

``D = ||G - I/k||_F^2`` splits as decorrelation plus norm balance, and the norm
balance half demands ``G_ii = 1/k``, i.e. columns of EQUAL norm.  That is
orthonormality, which is strictly more than the method needs and is measurably
harmful: a perfectly disjoint non-negative basis whose columns differ in
magnitude scores ``D = 0.2145`` rather than zero, so ``D`` penalises exactly the
structure the method exists to produce.  Real components do differ in magnitude.

v1's off-diagonal-only defect is not the answer either: a matrix with one nonzero
column and ``k-1`` zero columns scores exactly ``0``, because a zero column is
orthogonal to everything.  Pure off-diagonal orthogonality rewards total collapse.

Normalise per column instead and penalise angles.  Write ``vhat_i = v_i / max(|v_i|,
eps)`` and ``Ghat = Vhat' Vhat``; then

    C(V) = ||Ghat - I_k||_F^2 / (k(k-1))

         = (1/(k(k-1))) [ sum_{i != j} cos^2(v_i, v_j)
                          + sum_i (1 - min(1, |v_i|^2/eps^2))^2 ]

with ``cos_ij = <v_i,v_j> / (|v_i||v_j|)``.  KEEP THE IDENTITY.  The second sum is
exactly zero whenever every column has norm at least ``eps``, so the pairwise sum
alone is the whole story in the generic case and looks like the definition -- but
it is not, and an implementation written from the pairwise form alone reproduces
precisely the v1 collapse failure described above.  Each floored column is charged
``1/(k(k-1))``, so one live column and ``k-1`` dead ones scores ``(k-1)/(k(k-1)) =
1/k``, exactly ``0.2`` at ``k = 5``, against ``0`` for the pairwise sum.  That is what
``diagonal=True`` computes and it is the default for this reason.

Properties (all asserted in ``tests/test_angle.py``):

* ``C = 0`` exactly when the columns are mutually orthogonal, at ANY norms.
* ``0 <= C <= 1``, with ``C = 1`` iff every pair is collinear.
* Invariant under per-column rescaling ``V -> V diag(s)``, ``s > 0`` -- a genuine
  gauge freedom of a basis -- and hence under global scaling.
* Invariant under left multiplication by an orthogonal matrix.
* NOT invariant under right ``O(k)``, correctly: mutual orthogonality of columns
  is not preserved by mixing them.  (This is why the Procrustes ``align`` option,
  whose justification was ``D``'s right-``O(k)`` invariance, has no motivation here.)
* For ``V >= 0``, ``C = 0`` iff the columns have pairwise disjoint supports -- so the
  disjointness theorem, and with it the ``w -> 1`` feature-clustering limit,
  carries over unchanged.  It never needed equal norms.

Degeneracy is handled by the fidelity term rather than by this one: a zero column
contributes nothing to ``||X - X V V'||_F^2``, so reconstruction keeps columns
alive while ``C`` decorrelates them.  ``C`` itself is singular as ``|v_i| -> 0``, so
the implementation floors the norms at ``eps``.
"""
import torch

__all__ = ["cosine_matrix", "angle_defect", "grad_angle_defect"]

_EPS = 1e-12


def cosine_matrix(V, eps=_EPS):
    """Column-normalised Gram: ``cos_ij = <v_i, v_j> / (|v_i| |v_j|)``."""
    nrm = V.norm(dim=-2, keepdim=True).clamp_min(eps)
    U = V / nrm
    return U.transpose(-2, -1) @ U


def angle_defect(V, eps=_EPS, diagonal=True):
    r"""``C(V) = ||Ghat - I_k||_F^2 / (k(k-1))``, in ``[0, 1]``.  See the module
    docstring for the equivalent two-sum form; do not simplify this to the
    pairwise sum over ``i != j``, which is a different and worse functional.

    ``diagonal=True`` subtracts the full identity, so a column whose norm has been
    floored contributes ``cos_ii = 0`` against a target of 1 and is charged
    ``1/(k(k-1))``.  That doubles as a dead-column penalty and is why ``C`` scores
    exactly ``1/k`` on a rank-collapsed matrix -- 0.2 at ``k = 5`` -- where the
    pairwise-only defect scores 0.

    ``diagonal=False`` drops that term and measures *only* the pairwise angles.
    Use it when a dead column is a legitimate outcome and something else keeps
    the basis non-degenerate -- in the signed lifting an empty negative lobe is
    correct (global atrophy is one-signed), and the reconstruction term already
    prevents collapse because a dead component reconstructs nothing.
    """
    k = V.shape[-1]
    if k == 1:
        return torch.zeros(V.shape[:-2], dtype=V.dtype, device=V.device)
    C = cosine_matrix(V, eps)
    if diagonal:
        ref = torch.eye(k, dtype=V.dtype, device=V.device)
    else:
        ref = torch.diag_embed(C.diagonal(dim1=-2, dim2=-1))
    return (C - ref).pow(2).sum((-2, -1)) / (k * (k - 1))


def grad_angle_defect(V, eps=_EPS, diagonal=True):
    r"""Closed-form gradient of ``C``.

    With ``u_i = v_i / |v_i|`` and ``M = U'U`` the cosine matrix, write
    ``B = M - I``.  Then ``dC/dU = 4 U B / (k(k-1))``, and the per-column
    normalisation contributes the projection ``(I - u_i u_i') / |v_i|``, giving

        dC/dv_i = (1 / |v_i|) (I - u_i u_i') (dC/du_i).

    The projection is what makes the gradient tangential to each column's sphere,
    i.e. ``<dC/dv_i, v_i> = 0`` for every i -- ``C`` cannot alter any column norm.
    """
    k = V.shape[-1]
    if k == 1:
        return torch.zeros_like(V)
    nrm = V.norm(dim=-2, keepdim=True).clamp_min(eps)
    U = V / nrm
    M = U.transpose(-2, -1) @ U
    eye = torch.eye(k, dtype=V.dtype, device=V.device)
    ref = eye if diagonal else torch.diag_embed(M.diagonal(dim1=-2, dim2=-1))
    dU = (4.0 / (k * (k - 1))) * (U @ (M - ref))
    # remove the radial part of each column, then undo the scaling
    radial = (U * dU).sum(dim=-2, keepdim=True) * U
    return (dU - radial) / nrm
