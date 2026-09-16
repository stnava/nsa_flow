"""Solver behaviour: descent, stationarity, the w limits, determinism, edge cases."""
import itertools
import math

import pytest
import torch

from nsa_flow import energy, grad_energy, nsa_flow, stiefel_defect

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
@pytest.mark.parametrize("w", [0.5, 0.9, 0.99])
def test_scale_stays_bounded(w, prob):
    r = nsa_flow(prob, w=w, max_iter=5000)
    assert 0.1 < r.Y.norm().item() / prob.norm().item() < 10.0


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
