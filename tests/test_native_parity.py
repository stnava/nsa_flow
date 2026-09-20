"""Parity tests verifying C++ native kernel matches pure-Python L-BFGS-B bit-for-bit."""
import os
import pytest
import torch

from experiments.optimizer_study import make_problem
from nsa_flow.lbfgsb import (
    _CompactLBFGS,
    _build_K,
    _cauchy_point,
    _subspace_min,
    _tiny,
)
from nsa_flow.reconstruct import nsa_flow_data
from nsa_flow.signed import nsa_flow_signed
from nsa_flow.solve import _nsa_flow_anchored


def test_native_available_unless_disabled():
    disable = os.environ.get("NSA_FLOW_DISABLE_NATIVE", "0") in ("1", "true", "True")
    if disable:
        with pytest.raises(ImportError):
            import nsa_flow._native  # noqa: F401
    else:
        import nsa_flow._native as native
        assert hasattr(native, "lbfgsb_build_M")
        assert hasattr(native, "lbfgsb_direction")


@pytest.mark.skipif(
    os.environ.get("NSA_FLOW_DISABLE_NATIVE", "0") in ("1", "true", "True"),
    reason="Native extension disabled via NSA_FLOW_DISABLE_NATIVE",
)
def test_m_matrix_parity_random():
    import nsa_flow._native as native
    gen = torch.Generator().manual_seed(123)
    for m in (1, 2, 5, 10):
        n = 100
        S = torch.randn(n, m, generator=gen, dtype=torch.float64)
        Y = torch.randn(n, m, generator=gen, dtype=torch.float64)
        theta = float(torch.rand(1, generator=gen).item() + 0.5)

        M_py = _build_K(S, Y, theta)
        M_c = native.lbfgsb_build_M(S, Y, theta)
        diff = (M_py - M_c).abs().max().item()
        assert diff < 1e-12, f"M parity failed at m={m}: max diff {diff}"


@pytest.mark.skipif(
    os.environ.get("NSA_FLOW_DISABLE_NATIVE", "0") in ("1", "true", "True"),
    reason="Native extension disabled via NSA_FLOW_DISABLE_NATIVE",
)
def test_direction_parity_20_random_states():
    import nsa_flow._native as native
    gen = torch.Generator().manual_seed(42)
    dtype = torch.float64

    for state_idx in range(20):
        n = 80
        m = (state_idx % 6)  # covers m=0, 1, 2, 3, 4, 5
        x = torch.rand(n, generator=gen, dtype=dtype)
        g = torch.randn(n, generator=gen, dtype=dtype)
        lo = torch.zeros(n, dtype=dtype) if (state_idx % 2 == 0) else None
        hi = (torch.rand(n, generator=gen, dtype=dtype) + 2.0) if (state_idx % 3 == 0) else None

        H = _CompactLBFGS(n, 10, dtype, x.device)
        for _ in range(m):
            s = torch.randn(n, generator=gen, dtype=dtype) * 0.1
            y = torch.randn(n, generator=gen, dtype=dtype) * 0.1
            sy = (s * y).sum()
            if sy <= 0:
                y = y + s * 2.0
            H.push(s, y)

        x_cp, c, fixed = _cauchy_point(x, g, lo, hi, H)
        x_bar = _subspace_min(x, g, x_cp, c, fixed, lo, hi, H)
        d_py = x_bar - x

        a_max_py = 1.0
        if lo is not None:
            neg = d_py < 0
            if bool(neg.any()):
                a_max_py = max(a_max_py, float(((lo - x) / d_py.clamp_max(-_tiny(x)))[neg].clamp_min(0.0).min()))
        if hi is not None:
            pos = d_py > 0
            if bool(pos.any()):
                a_max_py = max(a_max_py, float(((hi - x) / d_py.clamp_min(_tiny(x)))[pos].clamp_min(0.0).min()))

        S_stack = torch.stack(H.S, dim=1) if H.m > 0 else torch.empty(n, 0, dtype=dtype)
        Y_stack = torch.stack(H.Yv, dim=1) if H.m > 0 else torch.empty(n, 0, dtype=dtype)

        d_c, a_max_c, fixed_c = native.lbfgsb_direction(x, g, lo, hi, S_stack, Y_stack, H.theta, H.M, 512)

        assert torch.equal(fixed, fixed_c), f"State {state_idx}: fixed mask mismatch"
        d_diff = (d_py - d_c).abs().max().item()
        d_scale = max(1.0, d_py.abs().max().item())
        assert d_diff / d_scale < 1e-11, f"State {state_idx}: d rel diff {d_diff / d_scale} >= 1e-11"
        a_diff = abs(a_max_py - a_max_c)
        assert a_diff < 1e-12, f"State {state_idx}: a_max diff {a_diff} >= 1e-12"


@pytest.mark.parametrize("mode", ["anchored", "data", "signed"])
@pytest.mark.parametrize("family", ["planted", "random"])
def test_six_reference_instances_energy_parity(mode, family, monkeypatch):
    """Parity of full fits on the 6 reference instances to atol=1e-12."""
    dtype = torch.float64
    device = "cpu"
    seed = 0
    w = 0.5
    tol = 1e-9
    budget = 1000

    prob = make_problem(mode, "small", seed=seed, dtype=dtype, device=device, family=family)

    # 1. Run with native extension
    if mode == "anchored":
        r_native = _nsa_flow_anchored(prob["target"], w=w, optimizer="lbfgsb", tol=tol, max_iter=budget)
    elif mode == "data":
        r_native = nsa_flow_data(prob["X"], k=prob["k"], init="clamp", w=w, optimizer="lbfgsb", tol=tol, max_iter=budget)
    else:
        r_native = nsa_flow_signed(prob["X"], k=prob["k"], init="auto", w=w, optimizer="lbfgsb", tol=tol, max_iter=budget)

    # 2. Run with Python fallback by disabling native in lbfgsb
    import nsa_flow.lbfgsb as lbfgsb_mod
    orig_native = lbfgsb_mod._native
    try:
        lbfgsb_mod._native = None
        if mode == "anchored":
            r_py = _nsa_flow_anchored(prob["target"], w=w, optimizer="lbfgsb", tol=tol, max_iter=budget)
        elif mode == "data":
            r_py = nsa_flow_data(prob["X"], k=prob["k"], init="clamp", w=w, optimizer="lbfgsb", tol=tol, max_iter=budget)
        else:
            r_py = nsa_flow_signed(prob["X"], k=prob["k"], init="auto", w=w, optimizer="lbfgsb", tol=tol, max_iter=budget)
    finally:
        lbfgsb_mod._native = orig_native

    e_diff = abs(r_native.energy - r_py.energy)
    assert e_diff < 1e-12, f"{mode} x {family}: energy diff {e_diff:.2e} >= 1e-12 (native={r_native.energy}, py={r_py.energy})"
    assert (r_native.stop_reason == r_py.stop_reason or
            (r_native.stop_reason in ("grad_map", "plateau") and r_py.stop_reason in ("grad_map", "plateau"))), \
        f"{mode} x {family}: stop reason mismatch ({r_native.stop_reason} vs {r_py.stop_reason})"
