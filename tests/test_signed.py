"""Properties of the signed lifting V = V+ - V-.

The practical requirement is that BOTH lobes be sparse, so most of these assert
per-part structure rather than aggregate sparsity: the aggregate zero-fraction
is satisfied by one empty lobe beside one dense one.
"""
import math

import numpy as np
import pytest
import torch

from nsa_flow import angle_defect, nsa_flow_data, nsa_flow_signed
from nsa_flow.reconstruct import reconstruction_fidelity
from nsa_flow.signed import consolidate_supports, part_sparsity

F64 = torch.float64


@pytest.fixture(scope="module")
def data():
    """Centred data with a planted contrast structure."""
    rng = np.random.default_rng(0)
    p, k, n = 40, 4, 200
    V = np.zeros((p, k))
    for i in range(k):                      # each component contrasts two blocks
        V[i * 5:(i + 1) * 5, i] = rng.random(5) + 0.5
        V[20 + i * 5:20 + (i + 1) * 5, i] = -(rng.random(5) + 0.5)
    X = rng.random((n, k)) @ V.T + 0.05 * rng.standard_normal((n, p))
    return torch.as_tensor(X - X.mean(0), dtype=F64)


# ----------------------------------------------------------------- structure
def test_returns_signed_basis_and_nonnegative_parts(data):
    r = nsa_flow_signed(data, k=4, w=0.5)
    W, V = r["parts"], r.Y
    assert W.shape == (data.shape[1], 8) and V.shape == (data.shape[1], 4)
    assert (W >= 0).all(), "both lobes must be non-negative"
    assert torch.allclose(V, W[:, :4] - W[:, 4:], atol=1e-12)


def test_gradient_of_the_signed_objective_matches_autograd(data):
    """Never verified before this test existed."""
    from nsa_flow.angle import grad_angle_defect
    from nsa_flow.reconstruct import grad_reconstruction_fidelity
    torch.manual_seed(0)
    p, k, w, lobe = data.shape[1], 4, 0.5, 1.0
    S, c = data.T @ data, data.pow(2).sum()
    W = torch.rand(p, 2 * k, dtype=F64, requires_grad=True)
    Vp, Vm = W[:, :k], W[:, k:]
    E = ((1 - w) * reconstruction_fidelity(Vp - Vm, S, c)
         + w * angle_defect(W, diagonal=False) + lobe * (Vp * Vm).sum() / c)
    E.backward()
    Wd = W.detach()
    Vp, Vm = Wd[:, :k], Wd[:, k:]
    gV = (1 - w) * grad_reconstruction_fidelity(Vp - Vm, S, c)
    g = (torch.cat([gV, -gV], -1) + w * grad_angle_defect(Wd, diagonal=False)
         + (lobe / c) * torch.cat([Vm, Vp], -1))
    assert (W.grad - g).abs().max() < 1e-12


# ------------------------------------------------------------------ capacity
def test_lifting_recovers_the_signed_optimum(data):
    """The claim the construction exists to settle: the non-negative ceiling is
    representational, not an optimisation failure."""
    S, c = data.T @ data, data.pow(2).sum()
    _, _, Vh = torch.linalg.svd(data, full_matrices=False)
    signed_err = float(reconstruction_fidelity(Vh[:4].T, S, c)) ** 0.5
    lift_err = float(reconstruction_fidelity(
        nsa_flow_signed(data, k=4, w=0.0, init="split").Y, S, c)) ** 0.5
    nonneg_err = float(reconstruction_fidelity(
        nsa_flow_data(data, k=4, w=0.0).Y, S, c)) ** 0.5
    assert lift_err == pytest.approx(signed_err, rel=1e-3)
    assert lift_err < nonneg_err


# ------------------------------------------------------------------ sparsity
def test_both_lobes_are_sparse_not_just_the_aggregate(data):
    r = nsa_flow_signed(data, k=4, w=0.75, init="split", consolidate=True)
    W = r["parts"]
    p = W.shape[0]
    plus, minus = W[:, :4], W[:, 4:]
    assert float((plus == 0).to(F64).mean()) > 0.6
    assert float((minus == 0).to(F64).mean()) > 0.6
    # and no single part may use most of the features
    assert r["parts_max_nnz"] < p // 2


def test_consolidation_gives_exactly_disjoint_supports(data):
    r = nsa_flow_signed(data, k=4, w=0.75, init="split")
    W = r["parts"]
    assert int((W.abs() > 0).sum(-1).max()) > 1          # relaxed: overlapping
    Wc = consolidate_supports(W)
    assert int((Wc.abs() > 0).sum(-1).max()) <= 1        # rounded: disjoint
    # rounding only removes support, never adds
    assert bool(((Wc != 0) <= (W != 0)).all())


def test_consolidate_option_keeps_disjointness_after_the_polish(data):
    r = nsa_flow_signed(data, k=4, w=0.75, init="split", consolidate=True)
    W = r["parts"]
    assert int((W.abs() > 0).sum(-1).max()) <= 1
    assert (W >= 0).all()
    assert r["consolidated"] is True


def test_consolidation_does_not_destroy_a_component(data):
    """Lobes may die -- a one-signed component is legitimate -- but a component
    losing both lobes would silently reduce the rank."""
    r = nsa_flow_signed(data, k=4, w=0.75, init="split", consolidate=True)
    per_component = (r.Y.abs() > 0).sum(0)
    assert int(per_component.min()) > 0
    assert r.effective_rank > 3.0


def test_sparsity_increases_with_w(data):
    prev = -1.0
    for w in (0.25, 0.5, 0.9):
        W = nsa_flow_signed(data, k=4, w=w, init="split")["parts"]
        z = float((W == 0).to(F64).mean())
        assert z >= prev - 0.05          # monotone up to solver slack
        prev = z


def test_part_sparsity_reports_per_part_not_aggregate():
    """One empty lobe beside one dense one must not read as sparse."""
    W = torch.zeros(20, 4, dtype=F64)
    W[:, 0] = torch.rand(20, dtype=F64)         # dense
    st = part_sparsity(W)
    assert st["n_dead"] == 3
    assert st["max_nnz"] == 20
    assert float((W == 0).to(F64).mean()) == pytest.approx(0.75)  # looks sparse


# --------------------------------------------------------------------- lobes
def test_lobe_penalty_drives_the_two_lobes_apart(data):
    free = nsa_flow_signed(data, k=4, w=0.5, init="split", lobe=0.0)
    penalised = nsa_flow_signed(data, k=4, w=0.5, init="split", lobe=1.0)
    assert penalised["lobe_overlap"] <= free["lobe_overlap"] + 1e-12


# ------------------------------------------------------------- orth variants
@pytest.mark.parametrize("orth", ["Coff", "C", "D"])
def test_every_orth_variant_runs_and_respects_the_constraint(data, orth):
    r = nsa_flow_signed(data, k=4, w=0.5, init="split", orth=orth)
    assert (r["parts"] >= 0).all() and torch.isfinite(r.Y).all()
    # NOT a whitelist of every possible value: that admitted a stalled solve.
    assert math.isfinite(r.grad_map), f"{orth}: grad_map={r.grad_map}"


def test_off_diagonal_default_does_not_charge_an_empty_lobe(data):
    """A one-signed component is correct; C's diagonal would penalise it."""
    W = torch.zeros(20, 4, dtype=F64)
    W[:10, 0] = 1.0
    W[10:, 1] = 1.0
    assert float(angle_defect(W, diagonal=False)) < 1e-24
    assert float(angle_defect(W, diagonal=True)) > 0.1


def test_bad_orth_and_bad_input_raise(data):
    with pytest.raises(ValueError):
        nsa_flow_signed(data, k=4, orth="nope")
    with pytest.raises(ValueError):
        nsa_flow_signed(data, k=4, w=1.5)
    with pytest.raises(ValueError):
        nsa_flow_signed(torch.zeros(10, 5, dtype=F64), k=2)
    with pytest.raises(ValueError):
        nsa_flow_signed(data)                         # neither k nor init array


def test_reports_honest_convergence(data):
    r = nsa_flow_signed(data, k=4, w=0.5, init="split", max_iter=3)
    assert r.stop_reason == "max_iter" and r.converged is False


# --------------------------------------------------- the optimiser itself
# These four mirror guards that already existed for nsa_flow
# (tests/test_solver.py:22-27, :245-256) and nsa_flow_data (:298-306) but were
# never written for the signed path.  Their absence let a solve that exited
# after ONE iteration -- returning its own initialisation with
# grad_map=inf and converged=True -- pass the whole suite.


@pytest.fixture(scope="module")
def nonneg_data():
    """Uncentred, strictly positive input.

    The module fixture is centred, which is exactly why the stall went unseen:
    on centred data init="split" looks healthy.  Both initialisers stall on
    non-negative uncentred input, so the guards below must run on it too.
    """
    rng = np.random.default_rng(1)
    X = 2.0 + rng.random((200, 40)) + 0.3 * rng.standard_normal((200, 40))
    return torch.as_tensor(np.clip(X, 0.05, None), dtype=F64)


@pytest.mark.parametrize("w", [0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("fixture", ["data", "nonneg_data"])
def test_default_settings_actually_run_the_solver(request, fixture, w):
    """At DEFAULT init, every w must take real steps and certify stationarity."""
    X = request.getfixturevalue(fixture)
    r = nsa_flow_signed(X, k=4, w=w)
    assert r.iters > 1, f"{fixture} w={w}: exited after {r.iters} iteration(s)"
    assert math.isfinite(r.grad_map), f"{fixture} w={w}: grad_map={r.grad_map}"
    assert r.stop_reason in ("grad_map", "line_search", "max_iter")


@pytest.mark.parametrize("fixture", ["data", "nonneg_data"])
def test_energy_decreases_from_the_initialisation(request, fixture):
    """The direct check that the solver did something, via the trace."""
    X = request.getfixturevalue(fixture)
    r = nsa_flow_signed(X, k=4, w=0.5, keep_trace=True)
    e = [row["energy"] for row in r.trace]
    assert len(e) > 1, f"{fixture}: only {len(e)} traced iteration(s)"
    assert e[-1] < e[0], f"{fixture}: energy {e[0]:.6e} -> {e[-1]:.6e}"


def test_a_stalled_solve_is_never_silent(nonneg_data):
    """Either stationary, or it warns.  Never grad_map=inf with converged=True."""
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        r = nsa_flow_signed(nonneg_data, k=4, w=0.25, max_iter=2000)
    assert math.isfinite(r.grad_map)
    if r.stop_reason == "line_search" and r.grad_map > 1e-6:
        assert any(issubclass(c.category, RuntimeWarning) for c in caught), (
            f"stalled at |Gmap|={r.grad_map:.2e} without warning")


def test_consolidate_reports_its_own_solve_not_the_previous_one(data):
    """fidelity/defect must describe the returned Y, not the pre-polish point."""
    k = 4
    r = nsa_flow_signed(data, k=k, w=0.5, consolidate=True)
    S = data.T @ data
    c = S.diagonal().sum()
    fresh_f = float(reconstruction_fidelity(r.Y, S, c, c))
    fresh_d = float(angle_defect(r["parts"], diagonal=False))
    assert abs(r.fidelity - fresh_f) < 1e-9, (r.fidelity, fresh_f)
    assert abs(r.defect - fresh_d) < 1e-9, (r.defect, fresh_d)
    assert math.isfinite(r.grad_map)
