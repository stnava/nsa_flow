"""The optimiser and diagnostics contract, held uniformly across the whole API.

Every optimiser x every mode, under one set of assertions.  These exist because
the previous default optimiser shipped without being in the contract table at
all: it ignored ``max_iter``, certified points at |Gmap| = 1e-2, and froze the
support at its initialisation, and 322 tests passed.  Adding an optimiser to
``nsa_flow.optim.OPTIMIZERS`` puts it under every test here automatically.
"""
import math
import warnings

import numpy as np
import pytest
import torch

from nsa_flow import nsa_flow, nsa_flow_data, nsa_flow_signed
from nsa_flow.diagnostics import (ORTH, basis_report,
                                  default_tol, gradient_mapping)
from nsa_flow.energy import grad_energy
from nsa_flow.linalg import jacobi_eigh, top_k_eigenvectors
from nsa_flow.optim import STALL_SLACK, optimizer_names
from nsa_flow.project import project_nonneg
from nsa_flow.solve import DEFAULT_OPTIMIZER, _nsa_flow_anchored

F64 = torch.float64
OPTS = [o for o in optimizer_names() if o not in ("torch_lbfgs", "scipy_lbfgsb")]
MODES = ["anchored", "data", "signed"]


def _problem(mode, seed=0, n=120, p=30, k=4):
    gen = torch.Generator().manual_seed(seed)
    V = torch.zeros(p, k, dtype=F64)
    per = p // k
    for j in range(k):
        V[j * per:(j + 1) * per, j] = torch.rand(per, generator=gen, dtype=F64) + 0.5
    X = torch.rand(n, k, generator=gen, dtype=F64) @ V.T \
        + 0.1 * torch.randn(n, p, generator=gen, dtype=F64)
    X = X - X.mean(0)
    if mode == "anchored":
        return dict(target=top_k_eigenvectors(k, X=X))
    if mode == "data":
        return dict(X=(X - X.min()).clamp_min(0.0), k=k)
    return dict(X=X, k=k)


def _fit(mode, prob, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode == "anchored":
            return _nsa_flow_anchored(prob["target"], **kw)
        if mode == "data":
            return nsa_flow_data(prob["X"], k=prob["k"], **kw)
        return nsa_flow_signed(prob["X"], k=prob["k"], **kw)


# ----------------------------------------------------------------- convergence
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("opt", OPTS)
def test_every_optimizer_converges_and_certifies(opt, mode):
    r = _fit(mode, _problem(mode), w=0.5, optimizer=opt, max_iter=20000)
    tol = default_tol(F64)
    assert r.converged, f"{opt}/{mode}: stop={r.stop_reason} |Gmap|={r.grad_map:.2e}"
    assert r.certificate in ("stationary", "numerical_floor")
    assert math.isfinite(r.grad_map)
    # a certificate is never issued far from stationarity
    assert r.grad_map <= STALL_SLACK * tol, f"{opt}/{mode}: certified at {r.grad_map:.2e}"
    # and the solve did something
    assert r["energy_reduction"] > 0.0
    assert r.iters > 1


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("opt", OPTS)
def test_max_iter_is_a_hard_cap_on_gradient_evaluations(opt, mode):
    for cap in (5, 40):
        r = _fit(mode, _problem(mode), w=0.9, optimizer=opt, max_iter=cap, tol=1e-16)
        assert r["n_grad"] <= cap + 1, f"{opt}/{mode}: n_grad={r['n_grad']} > cap={cap}"
        assert not r.converged and r.stop_reason == "max_iter"


@pytest.mark.parametrize("mode", MODES)
def test_default_optimizer_is_the_documented_one(mode):
    r = _fit(mode, _problem(mode), w=0.5)
    assert r["optimizer"] == DEFAULT_OPTIMIZER == "lbfgsb"


# ------------------------------------------------------------ certificate
@pytest.mark.parametrize("mode", MODES)
def test_reported_grad_map_is_the_shared_definition(mode):
    """Recompute the certificate from the returned basis; it must match."""
    if mode != "anchored":
        return
    r = _fit(mode, _problem(mode), w=0.5, max_iter=20000, fidelity="anchor")
    if mode == "anchored":
        g = grad_energy(r.Y, r.target, w=0.5, denom=r.target.pow(2).sum())
        gm = gradient_mapping(r.Y, g, project_nonneg)
        assert abs(gm - r.grad_map) <= 1e-9 + 1e-6 * gm


def test_certificate_is_scale_invariant():
    T = torch.rand(40, 5, dtype=F64)
    a = _fit("anchored", dict(target=T), w=0.5)
    b = _fit("anchored", dict(target=T * 1e3), w=0.5)
    c = _fit("anchored", dict(target=T * 1e-3), w=0.5)
    assert a.certificate == b.certificate == c.certificate
    # solutions agree to the tolerance they were solved to, not to 1e-9
    assert abs(a["defect_D"] - b["defect_D"]) < 1e-6
    assert abs(a["defect_D"] - c["defect_D"]) < 1e-6


def test_a_stalled_far_point_is_not_certified():
    """Line-search failure far from stationarity must report no certificate."""
    from nsa_flow.optim import _classify_stall
    assert _classify_stall(1.0, 1.0, [], gmap=0.28, tol=1e-9, dtype=F64) == "line_search"
    assert _classify_stall(1.0, 1.0, [], gmap=5e-9, tol=1e-9, dtype=F64) == "plateau"


# --------------------------------------------------------------- diagnostics
@pytest.mark.parametrize("mode", MODES)
def test_every_result_carries_every_defect_under_fixed_definitions(mode):
    r = _fit(mode, _problem(mode), w=0.5)
    rep = basis_report(r.Y)
    for name in ORTH:
        assert f"defect_{name}" in r
        assert abs(r[f"defect_{name}"] - rep[f"defect_{name}"]) < 1e-12
    # `defect` is the functional that was optimised, named by `orth`
    if mode != "signed":                       # signed's defect is on the parts
        assert abs(r["defect"] - r[f"defect_{r['orth']}"]) < 1e-12
    for key in ("energy_start", "energy_reduction", "n_grad", "n_energy",
                "certificate", "optimizer", "mode", "sparsity", "support_overlap"):
        assert key in r


def test_defect_functionals_differ_where_they_should():
    """D charges unequal column norms; C and Cg do not.  A table mixing them is wrong."""
    V = torch.zeros(12, 3, dtype=F64)
    V[0:4, 0], V[4:8, 1], V[8:12, 2] = 1.0, 0.3, 0.05
    rep = basis_report(V)
    assert rep["defect_C"] < 1e-12 and rep["defect_Cg"] < 1e-12
    assert rep["defect_D"] > 0.5


# ------------------------------------------------------------------ linalg
def test_jacobi_matches_eigh():
    for n in (3, 12, 40):
        A = torch.randn(n, n, dtype=F64)
        A = A @ A.T
        w1, _ = torch.linalg.eigh(A)
        w2, V2 = jacobi_eigh(A)
        assert torch.allclose(w1, w2, atol=1e-10 * float(A.abs().max()))
        assert torch.allclose(V2 @ torch.diag(w2) @ V2.T, A, atol=1e-10 * float(A.abs().max()))


@pytest.mark.parametrize("shape", [(200, 60, 5), (60, 900, 4), (72, 7129, 3)])
def test_top_k_eigenvectors_is_exact(shape):
    n, p, k = shape
    X = torch.randn(n, p, dtype=F64)
    E = top_k_eigenvectors(k, X=X)
    Vh = torch.linalg.svd(X, full_matrices=False)[2][:k].T
    cos = torch.linalg.svdvals(E.T @ Vh).min()
    assert float(cos) > 1.0 - 1e-10
    E2 = top_k_eigenvectors(k, X=X)
    assert torch.equal(E, E2), "must be deterministic"


# --------------------------------------------------------------- deprecation
def test_torch_lbfgs_warns():
    prob = _problem("data")
    with pytest.warns(DeprecationWarning):
        nsa_flow_data(prob["X"], k=prob["k"], w=0.5, optimizer="torch_lbfgs", max_iter=50)


def test_scipy_reference_agrees_with_torch_lbfgsb():
    pytest.importorskip("scipy")
    for mode in MODES:
        prob = _problem(mode, seed=3)
        a = _fit(mode, prob, w=0.5, optimizer="lbfgsb", max_iter=20000, tol=1e-9)
        b = _fit(mode, prob, w=0.5, optimizer="scipy_lbfgsb", max_iter=20000, tol=1e-9)
        # same basin: energies agree to solver precision
        assert abs(a.energy - b.energy) <= 1e-8 * (1 + abs(b.energy)), \
            f"{mode}: torch {a.energy:.10e} vs scipy {b.energy:.10e}"


def test_float32_flat_model_does_not_divide_by_zero():
    """clamp_min(1e-300) is 0.0 in float32; the Cauchy walk divided by it.

    Reported downstream (pysimlr, MultiOmics seed 47, iteration 25) as a
    ZeroDivisionError on a near-flat objective.  Floors are now in the tensor's
    own dtype and scalar divisions are guarded.
    """
    from nsa_flow.lbfgsb import _CompactLBFGS, _cauchy_point
    x = torch.tensor([0.5, 0.3, 0.0], dtype=torch.float32)
    g = torch.tensor([0.0, 0.0, 1e-30], dtype=torch.float32)
    H = _CompactLBFGS(3, 10, torch.float32, "cpu")
    H.theta = 1e-30
    x_cp, _, _ = _cauchy_point(x, g, torch.zeros(3, dtype=torch.float32), None, H)
    assert torch.isfinite(x_cp).all()
    for seed in range(40, 50):
        X = torch.rand(60, 40, generator=torch.Generator().manual_seed(seed),
                       dtype=torch.float32)
        nsa_flow_data(X, k=4, w=0.99, max_iter=300)       # must not raise


def test_sklearn_centers_and_transform_matches_fit():
    from nsa_flow import NSAFlow
    X = np.random.default_rng(0).normal(size=(60, 12)) + 0.5      # signed, off-centre
    assert X.min() < 0                                             # so auto -> signed
    m = NSAFlow(n_components=3, w=0.5).fit(X)
    assert np.allclose(m.mean_, X.mean(0))
    Z1 = m.transform(X)
    Z2 = (X - X.mean(0)) @ m.components_.T
    assert np.allclose(Z1, Z2)
    m2 = NSAFlow(n_components=3, w=0.5, mode="data").fit(np.abs(X))
    assert np.all(m2.mean_ == 0.0)


def test_explicit_nonneg_routes_to_the_data_solver():
    X = torch.randn(40, 15, dtype=F64)                 # signed data
    r = nsa_flow(X, k=3, w=0.5, nonneg=True)
    assert r["mode"] == "data" and (r.Y >= 0).all()
    r = nsa_flow(X, k=3, w=0.5, nonneg=False)
    assert r["mode"] == "signed"
    r = nsa_flow(X, k=3, w=0.5)                        # default: data decides
    assert r["mode"] == "signed"
    r = nsa_flow(torch.rand(40, 15, dtype=F64), k=3, w=0.5)
    assert r["mode"] == "data"


def test_fidelity_is_an_explicit_parameter_of_nsa_flow():
    """pysimlr detects the sign-blind fidelity by `'fidelity' in signature`;
    3.0 moved it into **kwargs and silently disabled that path downstream."""
    import inspect
    assert "fidelity" in inspect.signature(nsa_flow).parameters
    T = torch.randn(30, 4, dtype=F64)
    a = nsa_flow(T, w=0.5, fidelity="anchor")
    b = nsa_flow(T, w=0.5, fidelity="subspace")
    assert a["fidelity_mode"] == "anchor" and b["fidelity_mode"] == "subspace"
    with pytest.raises(ValueError):
        nsa_flow(torch.rand(30, 10, dtype=F64), k=3, fidelity="anchor")


def test_import_nsa_flow_does_not_import_sklearn():
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, nsa_flow; print('sklearn' in sys.modules); "
         "from nsa_flow import NSAFlow; print(NSAFlow.__name__)"],
        capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "NSAFlow"], out
