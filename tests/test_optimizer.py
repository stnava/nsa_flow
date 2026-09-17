"""Optimiser-level contract, applied uniformly to every solver entry point.

These properties are about the SOLVER, not the functional: did it take steps,
did the energy go down, does the reported certificate mean anything.  They
exist because `nsa_flow_signed` shipped a version that exited after one
iteration -- returning its own initialisation with `grad_map=inf` and
`converged=True` -- and passed a 16-test suite that checked only the algebra of
the lifting.  The guards that would have caught it existed for the other two
solvers and had never been generalised.

Two deliberate choices:

* Every solver is tested through ONE table, so adding a fourth entry point to
  `SOLVERS` is the only work needed to hold it to the same contract.
* Every solver is tested on RAW NON-NEGATIVE data as well as centred.  The
  original signed fixture was centred, and on centred input the stalling
  initialiser looks healthy; the failure only shows on uncentred input.
"""
import math
import warnings

import numpy as np
import pytest
import torch

from nsa_flow import nsa_flow, nsa_flow_data, nsa_flow_signed

F64 = torch.float64
TOL = 1e-9          # the float64 default the solvers pick


# --------------------------------------------------------------- data regimes
@pytest.fixture(scope="module")
def regimes():
    rng = np.random.default_rng(0)
    n, p, k = 120, 30, 4
    V = np.zeros((p, k))
    for i in range(k):                      # planted contrast structure
        V[i * 4:(i + 1) * 4, i] = rng.random(4) + 0.5
        V[16 + i * 3:16 + (i + 1) * 3, i] = -(rng.random(3) + 0.5)
    X = rng.random((n, k)) @ V.T + 0.05 * rng.standard_normal((n, p))
    return {
        # centred: the easy case every previous fixture used
        "centred": torch.as_tensor(X - X.mean(0), dtype=F64),
        # raw non-negative: intrinsic positivity with a floor, like log-intensity
        # expression or nutrient intakes.  This is the regime that stalls.
        "raw_nonneg": torch.as_tensor(np.clip(X - X.min() + 0.5, 0.05, None),
                                      dtype=F64),
        # wide: p > n, where the matrix-free path engages
        "wide": torch.as_tensor(rng.random((40, 150)) + 0.1, dtype=F64),
    }


def _run(name, X, w, keep_trace=False, max_iter=3000):
    """Call one solver on data ``X``; returns an NSAResult."""
    k = 4
    if name == "nsa_flow":
        # the anchored form takes a [p, k] target, not the data matrix
        target = torch.linalg.svd(X - X.mean(0), full_matrices=False)[2][:k].T
        return nsa_flow(target.contiguous(), w=w, max_iter=max_iter,
                        keep_trace=keep_trace)
    fn = nsa_flow_data if name == "nsa_flow_data" else nsa_flow_signed
    return fn(X, k=k, w=w, max_iter=max_iter, keep_trace=keep_trace)


SOLVERS = ["nsa_flow", "nsa_flow_data", "nsa_flow_signed"]
REGIMES = ["centred", "raw_nonneg", "wide"]
WS = [0.25, 0.5, 0.9]


# ------------------------------------------------------------------ contract
@pytest.mark.parametrize("w", WS)
@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_takes_real_steps(regimes, solver, regime, w):
    """It must iterate more than once and report a measured certificate.

    ``iters > 1`` is the assertion whose absence let the signed stall ship: a
    solve that returns its initialisation still satisfies every structural
    property of the output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = _run(solver, regimes[regime], w)
    assert r.iters > 1, f"{solver}/{regime}/w={w}: exited after {r.iters} iter(s)"
    assert math.isfinite(r.grad_map), (
        f"{solver}/{regime}/w={w}: grad_map={r.grad_map} is not a measurement")


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("solver", SOLVERS)
def test_energy_decreases_from_the_initialisation(regimes, solver, regime):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = _run(solver, regimes[regime], 0.5, keep_trace=True)
    e = [row["energy"] for row in r.trace]
    assert len(e) > 1, f"{solver}/{regime}: {len(e)} traced iteration(s)"
    assert e[-1] < e[0], f"{solver}/{regime}: energy {e[0]:.6e} -> {e[-1]:.6e}"


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("solver", SOLVERS)
def test_energy_is_monotone(regimes, solver, regime):
    """SPG with Armijo is a monotone method; a rise means the guard is broken."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = _run(solver, regimes[regime], 0.5, keep_trace=True)
    e = [row["energy"] for row in r.trace]
    for i in range(1, len(e)):
        assert e[i] <= e[i - 1] + 1e-12, (
            f"{solver}/{regime}: energy rose at iter {i}: "
            f"{e[i-1]:.12e} -> {e[i]:.12e}")


@pytest.mark.parametrize("w", WS)
@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("solver", SOLVERS)
def test_a_convergence_claim_is_backed_by_the_certificate(regimes, solver,
                                                          regime, w):
    """``stop_reason`` and ``grad_map`` must agree with each other.

    ``stop_reason="grad_map"`` is a positive claim of stationarity and has to be
    supported.  A stall far from stationarity is permitted -- the geometry can
    genuinely trap the iterate -- but it must not be silent.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = _run(solver, regimes[regime], w)
    tag = f"{solver}/{regime}/w={w}"
    if r.stop_reason == "grad_map":
        assert r.grad_map <= TOL * 10, f"{tag}: claimed grad_map stop at {r.grad_map:.2e}"
    if r.converged:
        assert math.isfinite(r.grad_map), f"{tag}: converged with {r.grad_map}"
    if r.stop_reason == "line_search" and r.grad_map > 1e-6:
        assert any(issubclass(c.category, RuntimeWarning) for c in caught), (
            f"{tag}: stalled at |Gmap|={r.grad_map:.2e} with no warning")


@pytest.mark.parametrize("solver", SOLVERS)
def test_iteration_cap_is_reported_honestly(regimes, solver):
    """A truncated solve must say so rather than claim convergence."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = _run(solver, regimes["centred"], 0.5, max_iter=2)
    assert r.stop_reason == "max_iter" and r.converged is False, (
        f"{solver}: stop={r.stop_reason} converged={r.converged} at max_iter=2")
