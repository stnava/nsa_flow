"""Executable statement of the NSA-Flow theory.

Every claim the paper makes about the energy is asserted here.  If a claim is
weakened or an implementation drifts, one of these fails.  Organised to match
the propositions in the paper.
"""
import itertools
import math

import pytest
import torch

from nsa_flow import (
    aligned_target,
    defect_floor,
    effective_rank,
    energy,
    fidelity,
    gram,
    grad_energy,
    grad_fidelity,
    grad_stiefel_defect,
    procrustes_rotation,
    project_scaled_stiefel,
    stiefel_defect,
)
from nsa_flow.energy import value_and_grad

F64 = torch.float64
SHAPES = [(40, 6), (100, 20), (12, 12), (200, 3), (60, 4)]


def _old_defect(Y):
    """The v1 'invariant orthogonality defect', kept only for contrast."""
    S = gram(Y)
    return ((S * S).sum() - S.diagonal().pow(2).sum()) / Y.pow(2).sum() ** 2


# ---------------------------------------------------------------- Proposition 1
@pytest.mark.parametrize("p,k", SHAPES)
def test_bounds_zero_to_one_minus_inv_k(p, k):
    """0 <= D <= 1 - 1/k, both bounds attained."""
    cap = 1.0 - 1.0 / k
    for _ in range(200):
        Y = torch.randn(p, k, dtype=F64) * (1 + 10 * torch.rand(1, dtype=F64))
        d = stiefel_defect(Y).item()
        assert -1e-14 <= d <= cap + 1e-12

    Q = torch.linalg.qr(torch.randn(p, k, dtype=F64))[0]
    assert abs(stiefel_defect(Q).item()) < 1e-14          # lower bound attained

    rank1 = torch.randn(p, 1, dtype=F64) @ torch.randn(1, k, dtype=F64)
    assert abs(stiefel_defect(rank1).item() - cap) < 1e-10  # upper bound attained


def test_defect_is_never_negative_near_the_optimum():
    """The ||G - I/k||^2 form avoids the cancellation that made D go negative."""
    worst = 0.0
    for _ in range(500):
        Q = torch.linalg.qr(torch.randn(50, 7, dtype=F64))[0]
        Y = Q + 10 ** (-torch.rand(1, dtype=F64) * 16) * torch.randn(50, 7, dtype=F64)
        worst = min(worst, stiefel_defect(Y).item())
    assert worst >= 0.0


# ---------------------------------------------------------------- Proposition 2
def test_zero_set_is_exactly_the_scaled_stiefel_manifold():
    p, k = 50, 7
    Q = torch.linalg.qr(torch.randn(p, k, dtype=F64))[0]
    for c in [1e-3, 1.0, 7.3, 1e4]:
        assert abs(stiefel_defect(c * Q).item()) < 1e-13

    # D = 0 forces Y'Y proportional to I
    S = gram(3.7 * Q)
    assert torch.allclose(S, S.diagonal().mean() * torch.eye(k, dtype=F64), atol=1e-11)

    # orthogonal columns with unequal norms are NOT in the zero set --
    # this is exactly what the v1 functional could not see
    scal = torch.ones(k, dtype=F64)
    scal[-1] = 8.0
    Yu = Q * scal
    assert stiefel_defect(Yu).item() > 0.1
    assert abs(_old_defect(Yu).item()) < 1e-14


def test_projection_onto_zero_set_is_the_true_euclidean_projection():
    """P(Y) = (sum sigma_i / k) U V', not (||Y||_F / sqrt(k)) U V'."""
    Y = torch.randn(40, 6, dtype=F64)
    P = project_scaled_stiefel(Y)
    assert abs(stiefel_defect(P).item()) < 1e-13
    best = (Y - P).norm().item()
    # no scaled polar factor is closer
    U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    for c in torch.linspace(0.1, 3.0, 400, dtype=F64):
        assert (Y - c * S.sum() / 6 * (U @ Vh)).norm().item() >= best - 1e-12


# ---------------------------------------------------------------- Proposition 3
def test_invariance_under_scale_and_both_orthogonal_groups():
    Y = torch.rand(60, 8, dtype=F64)
    d0 = stiefel_defect(Y).item()
    assert abs(stiefel_defect(1e5 * Y).item() - d0) < 1e-12
    U = torch.linalg.qr(torch.randn(60, 60, dtype=F64))[0]
    assert abs(stiefel_defect(U @ Y).item() - d0) < 1e-12
    V = torch.linalg.qr(torch.randn(8, 8, dtype=F64))[0]
    assert abs(stiefel_defect(Y @ V).item() - d0) < 1e-12   # full right O(k)


def test_v1_defect_breaks_the_symmetry_of_its_own_constraint():
    """Y -> YV preserves Y'Y = I, so any distance-to-Stiefel must be O(k)-invariant."""
    Y = torch.rand(60, 8, dtype=F64)
    V = torch.linalg.qr(torch.randn(8, 8, dtype=F64))[0]
    assert abs(_old_defect(Y @ V).item() - _old_defect(Y).item()) > 1e-6


# ---------------------------------------------------------------- Proposition 4
@pytest.mark.parametrize("p,k", [(40, 6), (100, 20), (30, 30)])
def test_spectral_identity(p, k):
    """D = k Var(lambda) = 1/EffectiveRank - 1/k."""
    Y = torch.rand(p, k, dtype=F64)
    lam = torch.linalg.eigvalsh(gram(Y))
    lam = lam / lam.sum()
    er = 1.0 / (lam * lam).sum()
    d = stiefel_defect(Y).item()
    assert abs(d - k * lam.var(unbiased=False).item()) < 1e-12
    assert abs(d - (1.0 / er - 1.0 / k).item()) < 1e-12
    assert abs(effective_rank(Y).item() - er.item()) < 1e-10
    assert 1.0 - 1e-12 <= er.item() <= k + 1e-12


# ---------------------------------------------------------------- Proposition 5
def test_decomposition_into_decorrelation_plus_norm_balance():
    """D = sum_{i!=j} G_ij^2 + sum_i (G_ii - 1/k)^2; the v1 defect is the first term."""
    for p, k in SHAPES:
        Y = torch.rand(p, k, dtype=F64) * torch.rand(k, dtype=F64)
        S = gram(Y)
        G = S / S.diagonal().sum()
        imbalance = (G.diagonal() - 1.0 / k).pow(2).sum()
        assert abs(stiefel_defect(Y).item() - _old_defect(Y).item() - imbalance.item()) < 1e-14
        assert _old_defect(Y).item() <= stiefel_defect(Y).item() + 1e-14


@pytest.mark.parametrize("r", [1, 2, 3, 4, 5])
def test_rank_collapse_floor(r):
    """rank(Y) = r < k  =>  D >= 1/r - 1/k.  The v1 defect has no such floor."""
    p, k = 60, 6
    floor = 1.0 / r - 1.0 / k
    worst = math.inf
    for _ in range(300):
        Y = torch.randn(p, r, dtype=F64) @ torch.randn(r, k, dtype=F64)
        worst = min(worst, stiefel_defect(Y).item())
    assert worst >= floor - 1e-10


def test_v1_defect_is_minimised_by_rank_deficient_matrices():
    p, k = 60, 6
    Q = torch.linalg.qr(torch.randn(p, k, dtype=F64))[0].clone()
    Q[:, -1] = 0.0                                   # rank k-1, orthogonal columns
    assert abs(_old_defect(Q).item()) < 1e-14        # v1: a global minimum
    assert stiefel_defect(Q).item() >= 1.0 / (k - 1) - 1.0 / k - 1e-12


# ---------------------------------------------------------------- Proposition 6
@pytest.mark.parametrize("p,k", [(40, 6), (100, 20), (9, 9), (20, 50)])
def test_gradient_matches_autograd(p, k):
    Y = torch.rand(p, k, dtype=F64, requires_grad=True)
    stiefel_defect(Y).backward()
    rel = (Y.grad - grad_stiefel_defect(Y.detach())).norm() / Y.grad.norm()
    assert rel.item() < 1e-11


@pytest.mark.parametrize("p,k", [(40, 6), (100, 20), (9, 9)])
def test_gradient_is_orthogonal_to_the_radial_direction(p, k):
    """<grad D, Y> = 0 (Euler): D provably cannot change ||Y||_F."""
    Y = torch.rand(p, k, dtype=F64)
    g = grad_stiefel_defect(Y)
    assert abs((g * Y).sum().item()) / (g.norm() * Y.norm()).item() < 1e-12


@pytest.mark.parametrize("p,k", [(40, 6), (100, 20), (9, 9)])
def test_gradient_norm_bound(p, k):
    """||grad D||_F <= 8 / ||Y||_F."""
    Y = torch.rand(p, k, dtype=F64)
    assert grad_stiefel_defect(Y).norm().item() <= 8.0 / Y.norm().item() + 1e-12


# ---------------------------------------------------------------- Proposition 7
def test_nonneg_and_orthogonal_iff_disjoint_supports():
    """Exact equivalence, exhaustively sampled."""
    bad = 0
    for _ in range(3000):
        Y = (torch.rand(8, 3, dtype=F64) * (torch.rand(8, 3) < 0.4)).clamp_min(0)
        S = gram(Y)
        diagonal = (S - torch.diag(S.diagonal())).abs().max().item() < 1e-14
        supp = [set((Y[:, i] > 0).nonzero().flatten().tolist()) for i in range(3)]
        disjoint = all(not (supp[i] & supp[j]) for i, j in itertools.combinations(range(3), 2))
        bad += int(diagonal != disjoint)
    assert bad == 0


def test_approximate_disjointness_bound():
    """max_{i!=j} |<y_i,y_j>| <= sqrt(D) ||Y||_F^2 -- the quantitative version."""
    worst = -math.inf
    for _ in range(3000):
        p = int(torch.randint(5, 60, (1,)))
        k = int(torch.randint(2, 8, (1,)))
        Y = torch.rand(p, k, dtype=F64) * (torch.rand(p, k) < 0.5)
        if Y.pow(2).sum() < 1e-12:
            continue
        S = gram(Y)
        off = (S - torch.diag(S.diagonal())).abs().max()
        bound = stiefel_defect(Y).clamp_min(0).sqrt() * Y.pow(2).sum()
        worst = max(worst, (off - bound).item())
    assert worst <= 1e-12


# ---------------------------------------------------------------- Proposition 8
@pytest.mark.parametrize("p,k", [(20, 50), (120, 200), (8, 9)])
def test_wide_matrices_report_the_true_floor(p, k):
    """For k > p orthonormal columns are impossible; inf D = 1/p - 1/k, reported not hidden."""
    floor = defect_floor(p, k)
    assert abs(floor - (1.0 / p - 1.0 / k)) < 1e-15
    best = math.inf
    for _ in range(200):
        U, _, Vh = torch.linalg.svd(torch.randn(p, k, dtype=F64), full_matrices=False)
        best = min(best, stiefel_defect(U @ Vh).item())
    assert abs(best - floor) < 1e-10
    assert best > 0.0


# ------------------------------------------------------- calibration of the energy
@pytest.mark.parametrize("p,k", SHAPES)
@pytest.mark.parametrize("w", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_both_energy_terms_are_dimensionless_and_order_one(p, k, w):
    """The check that would have caught the v1 calibration bug on day one.

    Each term must be O(1) at initialisation regardless of p, k, the scale of the
    data, or w -- so that w alone controls the trade-off.
    """
    for scale in [1e-4, 1.0, 1e4]:
        X0 = scale * torch.rand(p, k, dtype=F64)
        Y = X0 + 0.3 * scale * torch.randn(p, k, dtype=F64)
        tot, f, d = energy(Y, X0, w=w, return_parts=True)
        assert 0.0 <= f.item() < 10.0, f"fidelity {f.item()} not O(1)"
        assert 0.0 <= d.item() <= 1.0 + 1e-12, f"defect {d.item()} not in [0,1]"
        assert abs(tot.item() - ((1 - w) * f.item() + w * d.item())) < 1e-12


@pytest.mark.parametrize("w", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_energy_is_scale_invariant_in_the_data(w):
    """Rescaling X0 and Y together leaves E_w unchanged: no hidden units."""
    X0 = torch.rand(50, 6, dtype=F64)
    Y = X0 + 0.2 * torch.randn(50, 6, dtype=F64)
    e1 = energy(Y, X0, w=w).item()
    e2 = energy(1e5 * Y, 1e5 * X0, w=w).item()
    assert abs(e1 - e2) < 1e-11 * max(1.0, abs(e1))


@pytest.mark.parametrize("p,k", [(40, 6), (3, 40, 6)[1:], (20, 50), (10, 1)])
def test_fused_value_and_grad_agrees_with_separate_paths(p, k):
    Y = torch.rand(p, k, dtype=F64)
    X0 = torch.rand(p, k, dtype=F64)
    d = X0.pow(2).sum()
    E, F, D, g = value_and_grad(Y, X0, 0.37, d)
    E0, F0, D0 = energy(Y, X0, 0.37, d, return_parts=True)
    assert abs(E.item() - E0.item()) < 1e-14
    assert abs(F.item() - F0.item()) < 1e-14
    assert abs(D.item() - D0.item()) < 1e-14
    assert (g - grad_energy(Y, X0, 0.37, d)).abs().max().item() < 1e-15


def test_energy_gradient_matches_autograd_including_k_equals_one():
    for shape in [(40, 6), (20, 50), (10, 1)]:
        Y = torch.rand(*shape, dtype=F64, requires_grad=True)
        X0 = torch.rand(*shape, dtype=F64)
        energy(Y, X0, w=0.37).backward()
        rel = (Y.grad - grad_energy(Y.detach(), X0, 0.37)).norm() / Y.grad.norm()
        assert rel.item() < 1e-12


# ---------------------------------------------------------------------------
# Procrustes-aligned fidelity: anchor to X0's right-O(k) orbit, not to X0.
# ---------------------------------------------------------------------------

def _rand_orth(k, dtype=torch.float64):
    return torch.linalg.qr(torch.randn(k, k, dtype=dtype))[0]


def test_procrustes_rotation_is_orthogonal_and_optimal():
    """Q is in O(k) and no other orthogonal matrix does better."""
    X0 = torch.rand(40, 5, dtype=torch.float64)
    Y = torch.rand(40, 5, dtype=torch.float64)
    Q = procrustes_rotation(X0, Y)
    assert (Q.T @ Q - torch.eye(5, dtype=torch.float64)).abs().max() < 1e-12
    best = (Y - X0 @ Q).pow(2).sum()
    for _ in range(200):                      # no random competitor beats it
        assert best <= (Y - X0 @ _rand_orth(5)).pow(2).sum() + 1e-12


def test_aligned_fidelity_matches_the_nuclear_norm_closed_form():
    """min_Q ||Y - X0 Q||^2 = ||Y||^2 + ||X0||^2 - 2||X0'Y||_*."""
    X0 = torch.rand(30, 4, dtype=torch.float64)
    Y = torch.rand(30, 4, dtype=torch.float64)
    closed = (Y.pow(2).sum() + X0.pow(2).sum()
              - 2 * torch.linalg.svdvals(X0.T @ Y).sum()) / X0.pow(2).sum()
    assert abs(float(fidelity(Y, X0, align=True)) - float(closed)) < 1e-12


def test_aligned_fidelity_is_a_distance_to_the_orbit():
    """Invariant under X0 -> X0 R, never above the anchored value, zero on the orbit."""
    X0 = torch.rand(30, 4, dtype=torch.float64)
    Y = torch.rand(30, 4, dtype=torch.float64)
    f_al = float(fidelity(Y, X0, align=True))
    assert f_al <= float(fidelity(Y, X0, align=False)) + 1e-14
    for _ in range(20):
        R = _rand_orth(4)
        assert abs(float(fidelity(Y, X0 @ R, align=True)) - f_al) < 1e-12
        # and it vanishes on the whole orbit of X0
        assert float(fidelity(X0 @ R, X0, align=True)) < 1e-12
        # whereas the anchored form does not
    assert float(fidelity(X0 @ _rand_orth(4), X0, align=False)) > 1e-3


def test_aligned_gradient_matches_autograd_via_the_envelope_theorem():
    """No SVD derivative is needed: dF/dY = 2(Y - X0 Q*)/||X0||^2 at the optimum."""
    torch.manual_seed(3)
    X0 = torch.rand(25, 4, dtype=torch.float64)
    Y = torch.rand(25, 4, dtype=torch.float64).requires_grad_(True)
    fidelity(Y, X0, align=True).backward()    # autograd differentiates the SVD too
    closed = grad_fidelity(Y.detach(), X0, align=True)
    assert (Y.grad - closed).abs().max() < 1e-10


def test_defect_is_blind_to_the_rotation_fidelity_pays_for():
    """The premise of aligning: D is right-O(k) invariant, anchored F is not."""
    Y = torch.rand(30, 4, dtype=torch.float64)
    d0 = float(stiefel_defect(Y))
    spread = []
    for _ in range(20):
        R = _rand_orth(4)
        assert abs(float(stiefel_defect(Y @ R)) - d0) < 1e-12
        spread.append(float(fidelity(Y @ R, Y, align=False)))
    assert max(spread) > 1e-2                 # fidelity varies over the orbit
