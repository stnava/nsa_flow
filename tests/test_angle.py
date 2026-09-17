"""Properties of the squared-cosine (orthogonality) defect C."""
import numpy as np
import pytest
import torch

from nsa_flow.angle import angle_defect, cosine_matrix, grad_angle_defect
from nsa_flow.energy import stiefel_defect

F64 = torch.float64


def _orth(p, k):
    return torch.linalg.qr(torch.randn(p, k, dtype=F64))[0]


def test_zero_exactly_on_orthogonal_columns_at_any_norms():
    """C separates orthogonality from orthonormality; D does not."""
    Q = _orth(20, 5)
    for s in ([1, 1, 1, 1, 8], [0.01, 1, 5, 20, 100], [3, 3, 3, 3, 3]):
        V = Q @ torch.diag(torch.tensor(s, dtype=F64))
        assert float(angle_defect(V)) < 1e-24
    # the same matrix is "defective" to D whenever the norms differ
    V = Q @ torch.diag(torch.tensor([1.0, 1, 1, 1, 8], dtype=F64))
    assert float(stiefel_defect(V)) > 0.5


def test_bounded_in_unit_interval_with_collinearity_at_one():
    assert float(angle_defect(_orth(15, 4))) < 1e-24
    collinear = torch.rand(15, 1, dtype=F64) @ torch.ones(1, 4, dtype=F64)
    assert abs(float(angle_defect(collinear)) - 1.0) < 1e-12
    for _ in range(50):
        c = float(angle_defect(torch.randn(15, 4, dtype=F64)))
        assert -1e-15 <= c <= 1.0 + 1e-15


def test_invariant_under_per_column_rescaling_and_left_rotation():
    V = torch.rand(18, 4, dtype=F64)
    c0 = float(angle_defect(V))
    for _ in range(20):
        s = torch.rand(4, dtype=F64) * 10 + 1e-3
        assert abs(float(angle_defect(V * s)) - c0) < 1e-12      # per column
        U = _orth(18, 18)
        assert abs(float(angle_defect(U @ V)) - c0) < 1e-12      # left O(p)


def test_penalises_rank_collapse_where_v1s_functional_rewarded_it():
    """One live column, k-1 dead: v1's off-diagonal defect is exactly 0."""
    p, k = 20, 5
    Z = torch.zeros(p, k, dtype=F64)
    Z[:, 0] = torch.rand(p, dtype=F64)
    G = Z.T @ Z
    G = G / G.trace()
    v1 = float((G - torch.diag(torch.diagonal(G))).pow(2).sum())
    assert v1 < 1e-30                       # v1 rewards total collapse
    assert float(angle_defect(Z)) > 0.1     # C does not


def test_disjoint_nonneg_supports_are_exactly_the_zero_set():
    """The interpretability theorem needs only orthogonality, not equal norms."""
    p, k = 20, 5
    V = torch.zeros(p, k, dtype=F64)
    for i in range(k):                      # disjoint blocks, very unequal scales
        V[i * 4:(i + 1) * 4, i] = torch.rand(4, dtype=F64) * (i + 1) * 5
    assert float(angle_defect(V)) < 1e-24
    assert float(stiefel_defect(V)) > 0.1   # D penalises this ideal basis
    V[0, 1] = 0.5                           # break disjointness
    assert float(angle_defect(V)) > 1e-6


def test_gradient_matches_autograd_and_is_tangential_per_column():
    torch.manual_seed(0)
    V = torch.rand(20, 5, dtype=F64, requires_grad=True)
    angle_defect(V).backward()
    Vd = V.detach()
    g = grad_angle_defect(Vd)
    assert (V.grad - g).abs().max() < 1e-12
    # C cannot change any individual column norm
    assert (g * Vd).sum(0).abs().max() < 1e-12


def test_k_equals_one_is_zero_and_cosine_matrix_is_unit_diagonal():
    V = torch.rand(10, 1, dtype=F64)
    assert float(angle_defect(V)) == 0.0
    assert grad_angle_defect(V).abs().max() == 0.0
    M = cosine_matrix(torch.rand(10, 4, dtype=F64))
    assert (M.diagonal() - 1.0).abs().max() < 1e-12
    assert (M - M.T).abs().max() < 1e-14


def test_off_diagonal_variant_ignores_dead_columns_but_not_overlap():
    """``diagonal=False`` is for settings where a dead column is a correct answer.

    In the signed lifting an empty negative lobe is right -- global atrophy is
    one-signed -- and the reconstruction term, not the orthogonality term, is
    what keeps the basis non-degenerate.
    """
    p, k = 20, 6
    Q = _orth(p, k)
    half = torch.zeros(p, k, dtype=F64)
    half[:, :3] = Q[:, :3]                       # 3 live, 3 dead
    assert float(angle_defect(half, diagonal=True)) > 0.05     # charges the dead
    assert float(angle_defect(half, diagonal=False)) < 1e-24   # correctly zero
    # but genuine overlap is still detected
    bad = half.clone()
    bad[:, 3] = 0.5 * Q[:, 0] + 0.5 * Q[:, 1]
    assert float(angle_defect(bad, diagonal=False)) > 1e-3


@pytest.mark.parametrize("diagonal", [True, False])
def test_both_variants_have_correct_closed_form_gradients(diagonal):
    torch.manual_seed(2)
    V = torch.rand(18, 5, dtype=F64, requires_grad=True)
    angle_defect(V, diagonal=diagonal).backward()
    g = grad_angle_defect(V.detach(), diagonal=diagonal)
    assert (V.grad - g).abs().max() < 1e-12
    assert (g * V.detach()).sum(0).abs().max() < 1e-12     # tangential per column


def test_rank_collapse_scores_exactly_one_over_k():
    """The dead-column charge is 1/(k(k-1)) each, so k-1 dead columns give 1/k.

    Pinned exactly rather than loosely, because the loose form of this assertion
    would also pass for the pairwise-only defect plus any nonzero fudge.  This is
    the value the paper quotes.
    """
    for k in (3, 5, 8):
        Z = torch.zeros(20, k, dtype=F64)
        Z[:, 0] = torch.rand(20, dtype=F64)
        assert abs(float(angle_defect(Z)) - 1.0 / k) < 1e-9
        # and the pairwise-only form rewards it, which is the whole point
        assert float(angle_defect(Z, diagonal=False)) < 1e-30


def test_documented_formula_reproduces_the_implementation():
    """The two-sum equation in the module docstring and in the paper IS the code.

    Written out independently so that simplifying ``angle_defect`` to the
    pairwise sum -- the natural misreading of the defining equation -- fails here
    instead of silently reintroducing the collapse-rewarding functional.
    """
    eps = 1e-12

    def documented(V):
        k = V.shape[-1]
        nrm = V.norm(dim=-2)
        U = V / nrm.clamp_min(eps)
        G = U.T @ U
        pairwise = (G - torch.diag(torch.diagonal(G))).pow(2).sum()
        floored = ((1.0 - torch.clamp(nrm ** 2 / eps ** 2, max=1.0)) ** 2).sum()
        return float((pairwise + floored) / (k * (k - 1)))

    torch.manual_seed(0)
    cases = [torch.rand(12, 5, dtype=F64),                      # generic
             torch.rand(12, 5, dtype=F64) * 1e-6,               # small but > eps
             torch.eye(6, dtype=F64)[:, :4] * 7.0]              # orthogonal
    Z = torch.zeros(10, 4, dtype=F64)                           # one live column
    Z[:, 0] = torch.arange(1.0, 11.0, dtype=F64)
    cases.append(Z)
    Z2 = torch.zeros(10, 5, dtype=F64)                          # two live columns
    Z2[:, :2] = torch.rand(10, 2, dtype=F64)
    cases.append(Z2)
    for V in cases:
        assert abs(float(angle_defect(V)) - documented(V)) < 1e-12
