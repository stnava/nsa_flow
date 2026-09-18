"""Solver behaviour: descent, stationarity, the w limits, determinism, edge cases."""
import itertools
import math

import pytest
import torch

from nsa_flow import (grad_energy, grad_reconstruction_fidelity, nsa_flow,
                      nsa_flow_data, reconstruction_fidelity)
from nsa_flow.reconstruct import GramOperator

F64 = torch.float64


@pytest.fixture
def prob():
    return torch.rand(80, 8, dtype=F64)


# ------------------------------------------------------------------- descent
@pytest.mark.parametrize("w", [0.1, 0.5, 0.9, 0.99])
def test_energy_decreases_monotonically(w, prob):
    r = nsa_flow(prob, w=w, max_iter=3000, keep_trace=True)
    e = [t["energy"] for t in r.trace]
    assert len(e) > 1
    assert all(e[i + 1] <= e[i] + 1e-15 for i in range(len(e) - 1))
    assert e[-1] < e[0]


@pytest.mark.parametrize("w", [0.1, 0.5, 0.9])
def test_terminates_at_a_kkt_point(w, prob):
    """grad = 0 on {Y > 0}; grad >= 0 on {Y = 0}."""
    r = nsa_flow(prob, w=w, max_iter=20000, tol=1e-12)
    g = grad_energy(r.Y, r.target, w=w, denom=r.target.pow(2).sum())
    free, active = r.Y > 1e-13, r.Y <= 1e-13
    scale = max(g.abs().max().item(), 1.0)
    if free.any():
        assert g[free].abs().max().item() <= 1e-7 * scale
    if active.any():
        assert g[active].min().item() >= -1e-9
    assert r.converged and r.stop_reason in ("grad_map", "line_search")


def test_reported_convergence_is_honest(prob):
    """A run that hits max_iter must not claim convergence."""
    r = nsa_flow(prob, w=0.9, max_iter=3, tol=1e-16)
    assert not r.converged and r.stop_reason == "max_iter" and r.iters == 3


# ------------------------------------------------------------------ w limits
def test_w_zero_is_exactly_the_nonneg_projection():
    X = torch.randn(80, 8, dtype=F64)          # signed, so the clamp bites
    r = nsa_flow(X, w=0.0, max_iter=5000, tol=1e-14)
    assert torch.allclose(r.Y, X.clamp_min(0), atol=1e-9)


def test_w_zero_without_nonneg_is_the_identity():
    X = torch.randn(40, 5, dtype=F64)
    r = nsa_flow(X, w=0.0, nonneg=False, max_iter=5000, tol=1e-14)
    assert torch.allclose(r.Y, X, atol=1e-9)


def test_w_one_reaches_disjoint_supports(prob):
    with pytest.warns(RuntimeWarning, match="scale"):
        r = nsa_flow(prob, w=1.0, max_iter=60000, tol=1e-15)
    assert r.raw_defect < 1e-12
    Y, k = r.Y, prob.shape[1]
    thresh = 1e-6 * Y.abs().max()
    supp = [set((Y[:, i] > thresh).nonzero().flatten().tolist()) for i in range(k)]
    overlap = sum(len(supp[i] & supp[j]) for i, j in itertools.combinations(range(k), 2))
    assert overlap == 0
    assert r.effective_rank > k - 1e-6


@pytest.mark.parametrize("w", [0.5, 0.9, 0.99])
def test_defect_decreases_and_fidelity_increases_with_w(w, prob):
    """The trade-off is monotone in w -- the property that makes w meaningful."""
    lo = nsa_flow(prob, w=w - 0.4, max_iter=20000, tol=1e-12)
    hi = nsa_flow(prob, w=w, max_iter=20000, tol=1e-12)
    assert hi.raw_defect <= lo.raw_defect + 1e-12
    assert hi.fidelity >= lo.fidelity - 1e-12


def test_overlap_bound_holds_at_the_solution(prob):
    for w in [0.5, 0.9, 0.99]:
        r = nsa_flow(prob, w=w, max_iter=40000, tol=1e-13)
        S = r.Y.T @ r.Y
        off = (S - torch.diag(S.diagonal())).abs().max().item()
        assert off <= math.sqrt(max(r.raw_defect, 0.0)) * r.Y.pow(2).sum().item() + 1e-10


# --------------------------------------------------------- scale and uniqueness
@pytest.mark.parametrize("w", [0.0, 0.5, 0.9])
def test_scale_stays_bounded_for_moderate_w(w, prob):
    """For w <= 0.9 the fidelity term pins the scale; see the bound in the paper."""
    r = nsa_flow(prob, w=w, max_iter=20000, tol=1e-11)
    assert 0.5 < r.scale_ratio < 1.05
    assert abs(r.scale_ratio - r.Y.norm().item() / prob.norm().item()) < 1e-12


def test_w_one_is_scale_degenerate_and_says_so(prob):
    """D is scale-invariant, so w=1 leaves ||Y|| unconstrained. The solver must
    warn and report the drift rather than silently return a rescaled answer."""
    with pytest.warns(RuntimeWarning, match="scale"):
        r = nsa_flow(prob, w=1.0, max_iter=20000, tol=1e-12)
    assert r.raw_defect < 1e-12            # still a valid D = 0 point
    assert r.scale_ratio > 1.2             # and the scale really has drifted


def test_nonconvergence_near_w_one_is_reported_not_hidden():
    """Conditioning degrades as w -> 1; a run that runs out of iterations must
    say so."""
    X = torch.rand(2000, 20, dtype=F64)
    r = nsa_flow(X, w=0.999, max_iter=200, tol=1e-11)
    assert r.stop_reason == "max_iter" and not r.converged


def test_solution_is_data_scale_equivariant(prob):
    """Scaling the target scales the answer: no hidden absolute units."""
    a = nsa_flow(prob, w=0.7, max_iter=20000, tol=1e-13)
    b = nsa_flow(1e4 * prob, w=0.7, max_iter=20000, tol=1e-13)
    rel = (b.Y / 1e4 - a.Y).norm().item() / a.Y.norm().item()
    assert rel < 1e-6


@pytest.mark.parametrize("w", [0.5, 0.9])
def test_optimum_is_unique_across_random_restarts(w):
    """Empirically E_w has one optimum for w < 1: no restart lottery, no tuning."""
    X0 = torch.rand(60, 6, dtype=F64) * (torch.rand(60, 6) < 0.5) + 1e-3
    ref = nsa_flow(X0, w=w, max_iter=30000, tol=1e-13).energy
    for s in range(8):
        g = torch.Generator().manual_seed(s)
        e = nsa_flow(X0, w=w, init=torch.rand(60, 6, generator=g, dtype=F64),
                     max_iter=30000, tol=1e-13).energy
        assert abs(e - ref) <= 1e-8 * max(abs(ref), 1e-12)


def test_deterministic(prob):
    a = nsa_flow(prob, w=0.6, max_iter=500)
    b = nsa_flow(prob, w=0.6, max_iter=500)
    assert torch.equal(a.Y, b.Y) and a.energy == b.energy


def test_continuation_agrees_with_direct_solve(prob):
    a = nsa_flow(prob, w=0.8, max_iter=30000, tol=1e-13)
    b = nsa_flow(prob, w=0.8, max_iter=30000, tol=1e-13, continuation=10)
    assert abs(a.energy - b.energy) < 1e-9
    assert len(b.w_schedule) == 11


# ------------------------------------------------------------------ edge cases
def test_max_iter_zero_returns_the_initial_point():
    """v1 crashed here with 'NoneType * float'."""
    X = torch.rand(20, 5, dtype=F64)
    r = nsa_flow(X, w=0.5, max_iter=0)
    assert torch.allclose(r.Y, X)
    assert r.iters == 0 and not r.converged


def test_k_equals_one_is_well_defined():
    X = torch.rand(30, 1, dtype=F64)
    r = nsa_flow(X, w=0.9, max_iter=1000)
    assert r.defect == 0.0 and torch.isfinite(r.Y).all()


def test_wide_matrix_does_not_silently_produce_row_orthonormality():
    """k > p: the reported defect must be the true floor, not a hidden zero."""
    p, k = 20, 50
    X = torch.rand(p, k, dtype=F64)
    with pytest.warns(RuntimeWarning):
        r = nsa_flow(X, w=1.0, max_iter=40000, tol=1e-14)
    assert r.raw_defect >= 1.0 / p - 1.0 / k - 1e-9
    assert torch.isfinite(r.Y).all()


def test_single_row_and_single_column_shapes():
    for shape in [(1, 1), (1, 4), (4, 1)]:
        r = nsa_flow(torch.rand(*shape, dtype=F64) + 0.5, w=0.5, max_iter=200)
        assert torch.isfinite(r.Y).all() and r.Y.shape == shape


def test_sparse_target_with_empty_rows():
    X = torch.rand(40, 5, dtype=F64)
    X[:10] = 0.0
    r = nsa_flow(X, w=0.6, max_iter=5000)
    assert torch.isfinite(r.Y).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_respected(dtype):
    X = torch.rand(40, 5, dtype=torch.float64)
    r = nsa_flow(X, w=0.5, max_iter=500, dtype=dtype)
    assert r.Y.dtype == dtype and torch.isfinite(r.Y).all()


def test_nonfinite_input_is_rejected_not_silently_repaired():
    X = torch.rand(10, 3, dtype=F64)
    X[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        nsa_flow(X, w=0.5)


@pytest.mark.parametrize("bad_w", [-0.1, 1.1, 2.0])
def test_w_outside_unit_interval_is_rejected(bad_w):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        nsa_flow(torch.rand(10, 3, dtype=F64), w=bad_w)


def test_zero_target_is_rejected():
    with pytest.raises(ValueError, match="all zeros"):
        nsa_flow(torch.zeros(10, 3, dtype=F64), w=0.5)


def test_wrong_rank_input_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        nsa_flow(torch.rand(3, 10, 3, dtype=F64), w=0.5)


def test_mismatched_init_is_rejected():
    with pytest.raises(ValueError, match="init shape"):
        nsa_flow(torch.rand(10, 3, dtype=F64), w=0.5, init=torch.rand(10, 4, dtype=F64))


def test_numpy_and_list_inputs_accepted():
    import numpy as np
    r = nsa_flow(np.random.rand(20, 4), w=0.5, max_iter=200)
    assert torch.isfinite(r.Y).all()


def test_result_repr_is_informative(prob):
    s = repr(nsa_flow(prob, w=0.5, max_iter=100))
    for field in ["w=", "iters=", "energy=", "defect=", "eff_rank=", "stop="]:
        assert field in s


def test_compiled_kernel_agrees_with_eager(prob):
    a = nsa_flow(prob, w=0.6, max_iter=2000, tol=1e-12)
    b = nsa_flow(prob, w=0.6, max_iter=2000, tol=1e-12, compile=True)
    assert abs(a.energy - b.energy) < 1e-10


# ------------------------------------------------------ dtype-aware tolerance
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("w", [0.3, 0.5, 0.7, 0.9])
def test_default_settings_converge_in_both_precisions(dtype, w):
    """A default call must converge, not silently exhaust max_iter.

    A fixed tol=1e-9 is below float32 machine epsilon (1.2e-7), so it can never
    be met in single precision -- and torch.rand returns float32, so that was
    the common path.  The default tolerance is therefore derived from the dtype.
    """
    X = torch.rand(120, 8, dtype=dtype)
    r = nsa_flow(X, w=w)
    assert r.converged and r.stop_reason != "max_iter", (
        f"{dtype} w={w}: stop={r.stop_reason} after {r.iters} iters, "
        f"|Gmap|={r.grad_map:.2e}")


def test_explicit_tol_is_respected():
    X = torch.rand(60, 6, dtype=F64)
    loose = nsa_flow(X, w=0.7, tol=1e-6)
    tight = nsa_flow(X, w=0.7, tol=1e-12)
    assert loose.iters < tight.iters
    assert loose.grad_map > tight.grad_map


# ---------------------------------------------------------------------------
# Data-anchored variant: min (1-w)||X - XVV'||^2/||X||^2 + w Dtilde(V), V >= 0
# ---------------------------------------------------------------------------

def test_reconstruction_fidelity_and_gradient_match_the_direct_definition():
    """Closed forms agree with ||X - XVV'||^2 and its autograd gradient."""
    torch.manual_seed(0)
    X = torch.rand(50, 12, dtype=torch.float64)
    S, c = X.T @ X, X.pow(2).sum()
    V = torch.rand(12, 4, dtype=torch.float64, requires_grad=True)
    direct = (X - X @ V @ V.T).pow(2).sum() / c
    assert abs(float(reconstruction_fidelity(V.detach(), S, c))
                - float(direct.detach())) < 1e-12
    direct.backward()
    closed = grad_reconstruction_fidelity(V.detach(), S, c)
    assert (V.grad - closed).abs().max() < 1e-10


def test_data_anchored_beats_abs_pca_at_reconstruction():
    """The point of the variant: fitted non-negativity, not abs() of a signed basis."""
    torch.manual_seed(0)
    X = torch.rand(200, 20, dtype=torch.float64)
    X = X - X.mean(0)
    S, c = X.T @ X, X.pow(2).sum()
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    abs_pca = Vh[:5].T.abs()
    fitted = nsa_flow_data(X, k=5, w=0.0).Y
    assert (reconstruction_fidelity(fitted, S, c)
            < reconstruction_fidelity(abs_pca, S, c))


def test_data_anchored_reports_honest_convergence_and_monotone_defect():
    torch.manual_seed(1)
    X = torch.rand(120, 15, dtype=torch.float64)
    prev = float("inf")
    for w in (0.0, 0.5, 0.9, 0.99):
        r = nsa_flow_data(X, k=4, w=w)
        assert r.converged and r.stop_reason != "max_iter"
        assert r.raw_defect <= prev + 1e-12      # more w, less defect
        prev = r.raw_defect


def test_data_anchored_rejects_bad_input():
    X = torch.rand(30, 6, dtype=torch.float64)
    with pytest.raises(ValueError):
        nsa_flow_data(X, k=3, w=1.5)
    with pytest.raises(ValueError):
        nsa_flow_data(torch.zeros(30, 6, dtype=torch.float64), k=3)
    with pytest.raises(ValueError):
        nsa_flow_data(X)                          # neither k nor init
    with pytest.raises(ValueError):
        nsa_flow_data(X.unsqueeze(0), k=3)        # 3-D


# ---------------------------------------------------------------------------
# Matrix-free route: same numbers from X as from S = X'X, without forming S
# ---------------------------------------------------------------------------

def test_matrix_free_matches_the_gram_route_for_value_and_gradient():
    torch.manual_seed(0)
    X = torch.rand(60, 20, dtype=torch.float64)
    S, c = X.T @ X, X.pow(2).sum()
    V = torch.rand(20, 4, dtype=torch.float64)
    ops = GramOperator(X=X)
    assert abs(float(reconstruction_fidelity(V, S, c))
               - float(reconstruction_fidelity(V, ops, c))) < 1e-10
    assert (grad_reconstruction_fidelity(V, S, c)
            - grad_reconstruction_fidelity(V, ops, c)).abs().max() < 1e-10
    # and the mu=0 start agrees up to sign, being eigenvectors either way
    assert (ops.leading(4) * GramOperator(S=S).leading(4)).sum(0).abs().min() > 1 - 1e-8


def test_gram_operator_rejects_ambiguous_construction():
    X = torch.rand(10, 5, dtype=torch.float64)
    with pytest.raises(ValueError):
        GramOperator()                                  # neither
    with pytest.raises(ValueError):
        GramOperator(S=X.T @ X, X=X)                    # both


def test_solver_gives_the_same_answer_by_either_route():
    """The route is an implementation detail; it must not change the result.

    Compared up to a column permutation, which is a genuine symmetry here: both
    ``||X - X V V'||_F`` and the orthogonality defect are permutation invariant,
    so the two routes can order the columns differently while agreeing on every
    scalar the solver reports.

    ``GramOperator.leading()`` uses a canonical sign convention (largest-abs
    entry positive) so that ``init="clamp"`` produces the same starting point
    regardless of whether the SVD or eigh path is taken.
    """
    from scipy.optimize import linear_sum_assignment
    torch.manual_seed(0)
    X = torch.rand(60, 20, dtype=torch.float64)
    a = nsa_flow_data(X, k=4, w=0.5, matrix_free=False)
    b = nsa_flow_data(X, k=4, w=0.5, matrix_free=True)
    assert abs(a.energy - b.energy) < 1e-9
    assert abs(a.fidelity - b.fidelity) < 1e-9
    assert abs(a.defect - b.defect) < 1e-9
    A = a.Y / a.Y.norm(dim=0, keepdim=True)
    B = b.Y / b.Y.norm(dim=0, keepdim=True)
    cos = (A.T @ B).abs().numpy()
    r, c = linear_sum_assignment(-cos)
    assert cos[r, c].min() > 1 - 1e-6           # same basis, possibly reordered


def test_energy_is_invariant_to_column_permutation():
    """Why the previous test must match rather than compare elementwise."""
    torch.manual_seed(1)
    X = torch.rand(40, 12, dtype=torch.float64)
    V = torch.rand(12, 4, dtype=torch.float64)
    S, c = X.T @ X, X.pow(2).sum()
    perm = torch.tensor([2, 0, 3, 1])
    assert abs(float(reconstruction_fidelity(V, S, c))
               - float(reconstruction_fidelity(V[:, perm], S, c))) < 1e-12
    from nsa_flow import angle_defect
    assert abs(float(angle_defect(V)) - float(angle_defect(V[:, perm]))) < 1e-12


def test_route_is_dispatched_by_shape_and_reported():
    """p > n picks matrix-free, which is both the cost crossover and the only
    route that fits in memory at large p."""
    wide = nsa_flow_data(torch.rand(20, 60, dtype=torch.float64), k=3, w=0.5)
    tall = nsa_flow_data(torch.rand(60, 20, dtype=torch.float64), k=3, w=0.5)
    assert wide["matrix_free"] is True
    assert tall["matrix_free"] is False


def test_matrix_free_handles_p_far_larger_than_n():
    """Forming S here would be 0.32 GB; this route never allocates it."""
    torch.manual_seed(0)
    X = torch.rand(40, 6000, dtype=torch.float64)
    r = nsa_flow_data(X, k=4, w=0.5, max_iter=200)
    assert r["matrix_free"] is True
    assert r.Y.shape == (6000, 4)
    assert (r.Y >= 0).all() and torch.isfinite(r.Y).all()
