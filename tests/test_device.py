"""Tests for `device` parameter support across core NSA-Flow solver entry points.

Verifies:
- R1: API consistency: `device` argument exists in `nsa_flow`, `nsa_flow_data`,
  `nsa_flow_signed`, and `_nsa_flow_anchored`.
- Device preservation when `device=None`.
- Device migration when `device` is explicitly specified.
- R2: Transfer-free execution on accelerator devices (no .cpu() or .numpy() transfers
  during solver optimization iterations).
"""
import inspect
from unittest.mock import patch
import numpy as np
import pytest
import torch

from nsa_flow import (
    nsa_flow,
    nsa_flow_data,
    nsa_flow_signed,
    _nsa_flow_anchored,
)


def _get_available_devices():
    """Return list of available torch devices for testing (cpu + mps/cuda if available)."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


AVAILABLE_DEVICES = _get_available_devices()
ACCELERATOR_DEVICES = [d for d in AVAILABLE_DEVICES if d.type != "cpu"]


# -----------------------------------------------------------------------------
# R1: API Consistency
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("fn", [nsa_flow, nsa_flow_data, nsa_flow_signed, _nsa_flow_anchored])
def test_entry_points_accept_device_parameter(fn):
    sig = inspect.signature(fn)
    assert "device" in sig.parameters, f"{fn.__name__} must accept a `device` parameter"
    assert sig.parameters["device"].default is None, f"{fn.__name__} device default must be None"


# -----------------------------------------------------------------------------
# Device preservation: device=None keeps computation on input tensor's device
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("dev", AVAILABLE_DEVICES)
def test_device_none_anchored_preserves_device(dev):
    # Non-negative target
    dtype = torch.float32 if dev.type == "mps" else torch.float64
    target = torch.randn(20, 4, dtype=dtype, device=dev).abs()
    res = _nsa_flow_anchored(target, max_iter=25, tol=1e-5)
    assert res.Y.device.type == dev.type
    if dev.type != "cpu":
        assert res.Y.device == target.device


@pytest.mark.parametrize("dev", AVAILABLE_DEVICES)
def test_device_none_data_preserves_device(dev):
    dtype = torch.float32 if dev.type == "mps" else torch.float64
    X = torch.randn(25, 8, dtype=dtype, device=dev).abs()
    res = nsa_flow_data(X, k=3, max_iter=25, tol=1e-5)
    assert res.Y.device.type == dev.type
    if dev.type != "cpu":
        assert res.Y.device == X.device


@pytest.mark.parametrize("dev", AVAILABLE_DEVICES)
def test_device_none_signed_preserves_device(dev):
    dtype = torch.float32 if dev.type == "mps" else torch.float64
    X = torch.randn(25, 8, dtype=dtype, device=dev)
    res = nsa_flow_signed(X, k=3, max_iter=25, tol=1e-5)
    assert res.Y.device.type == dev.type
    if dev.type != "cpu":
        assert res.Y.device == X.device
        assert res["parts"].device == X.device


@pytest.mark.parametrize("dev", AVAILABLE_DEVICES)
@pytest.mark.parametrize("mode", ["data", "signed", "anchored"])
def test_device_none_nsa_flow_unified_preserves_device(dev, mode):
    dtype = torch.float32 if dev.type == "mps" else torch.float64
    if mode == "anchored":
        inp = torch.randn(20, 4, dtype=dtype, device=dev).abs()
        res = nsa_flow(inp, mode="anchored", max_iter=25, tol=1e-5)
    elif mode == "data":
        inp = torch.randn(25, 8, dtype=dtype, device=dev).abs()
        res = nsa_flow(inp, k=3, mode="data", max_iter=25, tol=1e-5)
    else:
        inp = torch.randn(25, 8, dtype=dtype, device=dev)
        res = nsa_flow(inp, k=3, mode="signed", max_iter=25, tol=1e-5)
    assert res.Y.device.type == dev.type


def test_device_none_with_numpy_defaults_to_cpu():
    arr = np.random.randn(20, 5).astype(np.float64)
    res = nsa_flow(arr, mode="anchored", max_iter=15)
    assert res.Y.device.type == "cpu"


# -----------------------------------------------------------------------------
# Explicit device: moves inputs and computes on specified device
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("target_dev", ACCELERATOR_DEVICES)
def test_explicit_accelerator_device_from_cpu(target_dev):
    # Input starts on CPU
    target_cpu = torch.randn(20, 4, dtype=torch.float32).abs()
    X_cpu = torch.randn(25, 8, dtype=torch.float32).abs()
    X_signed_cpu = torch.randn(25, 8, dtype=torch.float32)

    r_anc = _nsa_flow_anchored(target_cpu, device=target_dev, max_iter=20, tol=1e-4)
    assert r_anc.Y.device.type == target_dev.type

    r_data = nsa_flow_data(X_cpu, k=3, device=target_dev, max_iter=20, tol=1e-4)
    assert r_data.Y.device.type == target_dev.type

    r_signed = nsa_flow_signed(X_signed_cpu, k=3, device=target_dev, max_iter=20, tol=1e-4)
    assert r_signed.Y.device.type == target_dev.type

    r_uni = nsa_flow(target_cpu, device=target_dev, max_iter=20, tol=1e-4)
    assert r_uni.Y.device.type == target_dev.type


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("target_dev", ACCELERATOR_DEVICES)
def test_explicit_accelerator_device_from_numpy(target_dev):
    arr = np.random.randn(20, 4).astype(np.float32)
    res = nsa_flow(arr, device=target_dev, max_iter=20, tol=1e-4)
    assert res.Y.device.type == target_dev.type


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_explicit_cpu_device_from_accelerator(acc_dev):
    # Input starts on accelerator, target device is explicitly "cpu"
    target_acc = torch.randn(20, 4, dtype=torch.float32, device=acc_dev).abs()
    res = nsa_flow(target_acc, device="cpu", max_iter=20, tol=1e-4)
    assert res.Y.device.type == "cpu"

    r_anc = _nsa_flow_anchored(target_acc, device="cpu", max_iter=20, tol=1e-4)
    assert r_anc.Y.device.type == "cpu"

    X_acc = torch.randn(25, 8, dtype=torch.float32, device=acc_dev).abs()
    r_data = nsa_flow_data(X_acc, k=3, device="cpu", max_iter=20, tol=1e-4)
    assert r_data.Y.device.type == "cpu"

    r_signed = nsa_flow_signed(X_acc, k=3, device="cpu", max_iter=20, tol=1e-4)
    assert r_signed.Y.device.type == "cpu"


# -----------------------------------------------------------------------------
# R2: Transfer-Free Execution During Optimization
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_no_host_transfers_during_optimization(acc_dev):
    """Assert that zero .cpu() or .numpy() transfers happen during solver iterations."""
    orig_cpu = torch.Tensor.cpu
    orig_numpy = torch.Tensor.numpy
    cpu_calls = []
    numpy_calls = []

    def forbidden_cpu(self, *args, **kwargs):
        cpu_calls.append(True)
        return orig_cpu(self, *args, **kwargs)

    def forbidden_numpy(self, *args, **kwargs):
        numpy_calls.append(True)
        return orig_numpy(self, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", forbidden_cpu), patch.object(torch.Tensor, "numpy", forbidden_numpy):
        target = torch.randn(20, 4, dtype=torch.float32, device=acc_dev).abs()
        _nsa_flow_anchored(target, max_iter=25, tol=1e-5)
        assert len(cpu_calls) == 0, "Host transfer via .cpu() detected in _nsa_flow_anchored"
        assert len(numpy_calls) == 0, "NumPy transfer detected in _nsa_flow_anchored"

        X = torch.randn(25, 8, dtype=torch.float32, device=acc_dev).abs()
        nsa_flow_data(X, k=3, max_iter=25, tol=1e-5)
        assert len(cpu_calls) == 0, "Host transfer via .cpu() detected in nsa_flow_data"
        assert len(numpy_calls) == 0, "NumPy transfer detected in nsa_flow_data"

        nsa_flow_signed(X, k=3, max_iter=25, tol=1e-5)
        assert len(cpu_calls) == 0, "Host transfer via .cpu() detected in nsa_flow_signed"
        assert len(numpy_calls) == 0, "NumPy transfer detected in nsa_flow_signed"

        nsa_flow(target, max_iter=25, tol=1e-5)
        assert len(cpu_calls) == 0, "Host transfer via .cpu() detected in nsa_flow"
        assert len(numpy_calls) == 0, "NumPy transfer detected in nsa_flow"


# -----------------------------------------------------------------------------
# Optimizers and Edge Cases on Accelerator
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
@pytest.mark.parametrize("opt", ["lbfgsb", "spg", "fista", "pqn"])
def test_all_optimizers_on_accelerator(acc_dev, opt):
    target = torch.randn(20, 4, dtype=torch.float32, device=acc_dev).abs()
    res = _nsa_flow_anchored(target, optimizer=opt, max_iter=25, tol=1e-4)
    assert res.Y.device.type == acc_dev.type
    assert torch.isfinite(res.Y).all()


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_edge_cases_on_accelerator(acc_dev):
    # 1. Subspace fidelity on signed target
    target_signed = torch.randn(20, 4, dtype=torch.float32, device=acc_dev)
    r_sub = _nsa_flow_anchored(target_signed, fidelity="subspace", max_iter=20, tol=1e-4)
    assert r_sub.Y.device.type == acc_dev.type
    assert r_sub["fidelity_mode"] == "subspace"

    # 2. Sparseness post-processing
    target_pos = torch.randn(20, 4, dtype=torch.float32, device=acc_dev).abs()
    r_sp = _nsa_flow_anchored(target_pos, sparseness=0.5, max_iter=20, tol=1e-4)
    assert r_sp.Y.device.type == acc_dev.type

    # 3. Consolidation in signed mode
    X_s = torch.randn(25, 8, dtype=torch.float32, device=acc_dev)
    r_cons = nsa_flow_signed(X_s, k=3, consolidate=True, max_iter=20, tol=1e-4)
    assert r_cons.Y.device.type == acc_dev.type
    assert r_cons["consolidated"] is True

    # 4. Matrix-free true vs false in data mode
    X_d = torch.randn(25, 8, dtype=torch.float32, device=acc_dev).abs()
    r_mf = nsa_flow_data(X_d, k=3, matrix_free=True, max_iter=20, tol=1e-4)
    r_gram = nsa_flow_data(X_d, k=3, matrix_free=False, max_iter=20, tol=1e-4)
    assert r_mf.Y.device.type == acc_dev.type
    assert r_gram.Y.device.type == acc_dev.type


# -----------------------------------------------------------------------------
# Zero D2H Scalar Sync Verification
# -----------------------------------------------------------------------------
class SyncInterceptor:
    """Context manager intercepting scalar GPU-to-host synchronization calls."""
    def __init__(self):
        self.calls = []
        self._orig_item = torch.Tensor.item
        self._orig_float = torch.Tensor.__float__
        self._orig_bool = torch.Tensor.__bool__
        self._orig_int = torch.Tensor.__int__
        self._orig_cpu = torch.Tensor.cpu

    def __enter__(self):
        interceptor = self

        def item(self):
            interceptor.calls.append(("item", self.device))
            return interceptor._orig_item(self)

        def __float__(self):
            interceptor.calls.append(("__float__", self.device))
            return interceptor._orig_float(self)

        def __bool__(self):
            interceptor.calls.append(("__bool__", self.device))
            return interceptor._orig_bool(self)

        def __int__(self):
            interceptor.calls.append(("__int__", self.device))
            return interceptor._orig_int(self)

        def cpu(self):
            interceptor.calls.append(("cpu", self.device))
            return interceptor._orig_cpu(self)

        torch.Tensor.item = item
        torch.Tensor.__float__ = __float__
        torch.Tensor.__bool__ = __bool__
        torch.Tensor.__int__ = __int__
        torch.Tensor.cpu = cpu
        return self

    def __exit__(self, *args):
        torch.Tensor.item = self._orig_item
        torch.Tensor.__float__ = self._orig_float
        torch.Tensor.__bool__ = self._orig_bool
        torch.Tensor.__int__ = self._orig_int
        torch.Tensor.cpu = self._orig_cpu


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_zero_scalar_syncs_in_inner_optimization_step(acc_dev):
    """Verify that inner optimization steps execute with zero D2H scalar syncs."""
    from nsa_flow.optim import inner_optimization_step, inner_optimization_loop
    from nsa_flow.energy import value_and_grad
    from nsa_flow.project import project_nonneg

    X0 = torch.randn(40, 5, dtype=torch.float32, device=acc_dev).abs()
    Y = X0.clone()
    denom = X0.pow(2).sum()

    def grad_fn(Yc):
        E, _, _, g = value_and_grad(Yc, X0, w=0.5, denom=denom)
        return E, g

    with SyncInterceptor() as sync:
        Y_next, E, g, _ = inner_optimization_step(Y, grad_fn, proj=project_nonneg, lr=0.01)

    assert len(sync.calls) == 0, f"Expected 0 scalar syncs in inner_optimization_step, got {sync.calls}"

    with SyncInterceptor() as sync:
        Y_loop, E, g = inner_optimization_loop(Y, grad_fn, proj=project_nonneg, n_steps=10, lr=0.01)

    assert len(sync.calls) == 0, f"Expected 0 scalar syncs in inner_optimization_loop, got {sync.calls}"


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_zero_scalar_syncs_in_cauchy_and_subspace(acc_dev):
    """Verify that _cauchy_point and _subspace_min execute with zero D2H scalar syncs."""
    from nsa_flow.lbfgsb import _cauchy_point, _subspace_min, _CompactLBFGS

    x = torch.randn(50, device=acc_dev, dtype=torch.float32).clamp_min(0.0)
    g = torch.randn(50, device=acc_dev, dtype=torch.float32)
    lo = torch.zeros_like(x)
    H = _CompactLBFGS(50, 5, torch.float32, acc_dev)

    with SyncInterceptor() as sync:
        xcp, c, fixed = _cauchy_point(x, g, lo, None, H)

    assert len(sync.calls) == 0, f"Expected 0 scalar syncs in _cauchy_point, got {sync.calls}"

    with SyncInterceptor() as sync:
        x_sub = _subspace_min(x, g, xcp, c, fixed, lo, None, H)

    assert len(sync.calls) == 0, f"Expected 0 scalar syncs in _subspace_min, got {sync.calls}"


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_full_solve_sync_reduction_on_accelerator(acc_dev):
    """Verify that full solver iterations have strictly bounded D2H syncs on accelerator."""
    target = torch.randn(25, 4, dtype=torch.float32, device=acc_dev).abs()

    with SyncInterceptor() as sync:
        res_lbfgsb = _nsa_flow_anchored(target, optimizer="lbfgsb", max_iter=10, tol=1e-8)

    cpu_or_np = [c for c in sync.calls if c[0] in ("cpu", "numpy")]
    assert len(cpu_or_np) == 0, f"No bulk host transfers permitted during solve, got {cpu_or_np}"
    # With pipelined step conclusion, 10 steps take <= 40 scalar syncs total (averaging <= 4 syncs/step)
    assert len(sync.calls) <= 40, f"Expected <= 40 scalar syncs for 10 L-BFGS-B steps, got {len(sync.calls)}"
    assert torch.isfinite(res_lbfgsb.Y).all()


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_spg_and_fista_sync_reduction_on_accelerator(acc_dev):
    """Verify SPG and FISTA have minimal D2H scalar syncs on accelerator."""
    target = torch.randn(25, 4, dtype=torch.float32, device=acc_dev).abs()

    with SyncInterceptor() as sync_spg:
        res_spg = _nsa_flow_anchored(target, optimizer="spg", max_iter=10, tol=1e-8)
    assert len(sync_spg.calls) <= 35, f"Expected <= 35 scalar syncs for 10 SPG steps, got {len(sync_spg.calls)}"
    assert torch.isfinite(res_spg.Y).all()

    with SyncInterceptor() as sync_fista:
        res_fista = _nsa_flow_anchored(target, optimizer="fista", max_iter=10, tol=1e-8)
    assert len(sync_fista.calls) <= 40, f"Expected <= 40 scalar syncs for 10 FISTA steps, got {len(sync_fista.calls)}"
    assert torch.isfinite(res_fista.Y).all()


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_float32_barzilai_borwein_underflow_robustness(acc_dev):
    """Verify that Barzilai-Borwein in float32 does not underflow clamp_min to 0.0 or produce NaN."""
    # Create target and solve in float32
    target = torch.randn(30, 4, dtype=torch.float32, device=acc_dev).abs() * 1e-15
    res = _nsa_flow_anchored(target, optimizer="spg", max_iter=15, tol=1e-8)
    assert torch.isfinite(res.Y).all(), "Float32 SPG produced NaN or Inf on near-zero target"


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_relax_and_adaptive_inits_on_accelerator(acc_dev):
    """Verify relax and adaptive continuation inits run on accelerator without transfers."""
    X_d = torch.randn(25, 8, dtype=torch.float32, device=acc_dev).abs()
    r_adapt = nsa_flow_signed(X_d, k=3, init="adaptive", max_iter=10, tol=1e-4)
    assert r_adapt.Y.device.type == acc_dev.type
    assert torch.isfinite(r_adapt.Y).all()

    # Fast relax_into_nonneg with sync interceptor
    from nsa_flow.reconstruct import relax_into_nonneg, GramOperator
    ops = GramOperator(X=X_d)
    c = ops.c
    with SyncInterceptor() as sync:
        V = relax_into_nonneg(ops, c, 3, 0.5, fast=True, trS=c)
    assert V.device.type == acc_dev.type
    assert torch.isfinite(V).all()
    # Verified: _smooth_descent refactored to eliminate redundant obj(V) evaluations and tensor-to-float syncs
    assert len(sync.calls) <= 180, f"Expected <= 180 scalar syncs in fast relax_into_nonneg, got {len(sync.calls)}"


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_jacobi_eigh_on_accelerator(acc_dev):
    """Verify jacobi_eigh runs on accelerator with fused off/total evaluations."""
    from nsa_flow.linalg import jacobi_eigh
    A = torch.randn(10, 10, dtype=torch.float32, device=acc_dev)
    A = A.t() @ A
    with SyncInterceptor() as sync:
        w, V = jacobi_eigh(A, sweeps=10)
    assert w.device.type == acc_dev.type
    assert V.device.type == acc_dev.type
    assert torch.allclose(V.t() @ V, torch.eye(10, dtype=torch.float32, device=acc_dev), atol=1e-4)


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_mps_float64_handling(acc_dev):
    """Verify MPS float64 auto-conversion when dtype is None and explicit ValueError when demanded."""
    if acc_dev.type != "mps":
        pytest.skip("Test specifically covers MPS float64 handling")

    # 1. NumPy float64 array with device="mps" and dtype=None auto-converts to float32 with warning
    arr = np.random.randn(20, 4).astype(np.float64)
    with pytest.warns(UserWarning, match="MPS does not support float64"):
        res = nsa_flow(arr, device=acc_dev, max_iter=10)
    assert res.Y.device.type == "mps"
    assert res.Y.dtype == torch.float32

    # 2. PyTorch float64 tensor with device="mps" and dtype=None auto-converts with warning
    t_f64 = torch.randn(20, 4, dtype=torch.float64)
    with pytest.warns(UserWarning, match="MPS does not support float64"):
        res_anc = _nsa_flow_anchored(t_f64, device=acc_dev, max_iter=10)
    assert res_anc.Y.device.type == "mps"
    assert res_anc.Y.dtype == torch.float32

    # 3. Explicit dtype=torch.float64 on MPS raises actionable ValueError
    with pytest.raises(ValueError, match="MPS does not support float64 precision"):
        _nsa_flow_anchored(t_f64, dtype=torch.float64, device=acc_dev)

    with pytest.raises(ValueError, match="MPS does not support float64 precision"):
        nsa_flow(arr, dtype=torch.float64, device=acc_dev)

    # 4. Explicit dtype=torch.float32 with float64 input succeeds on MPS
    res_f32 = nsa_flow(t_f64, dtype=torch.float32, device=acc_dev, max_iter=10)
    assert res_f32.Y.device.type == "mps"
    assert res_f32.Y.dtype == torch.float32


def test_float16_execution_on_cpu_and_accelerator():
    """Verify float16 half precision solves without native kernel NotImplementedError."""
    for dev in AVAILABLE_DEVICES:
        target = torch.randn(20, 4, dtype=torch.float16, device=dev).abs()
        res = _nsa_flow_anchored(target, dtype=torch.float16, max_iter=5)
        assert res.Y.dtype == torch.float16
        assert res.Y.device.type == dev.type


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_large_dimension_s_eigenvectors_on_accelerator(acc_dev):
    """Verify that large Gram matrix S (p > 512) on accelerator uses subspace iteration rather than stalling."""
    from nsa_flow.linalg import top_k_eigenvectors
    S = torch.randn(600, 600, dtype=torch.float32, device=acc_dev)
    S = S.t() @ S
    with pytest.warns(RuntimeWarning, match="top_k_eigenvectors: dimension of S is large"):
        V = top_k_eigenvectors(3, S=S)
    assert V.device.type == acc_dev.type
    assert V.shape == (600, 3)
    assert torch.isfinite(V).all()


def test_fidelity_entrywise_alias():
    """Verify fidelity='entrywise' is accepted as an alias for 'anchor' across entry points."""
    target = torch.randn(20, 3, dtype=torch.float32).abs()
    r_anc = _nsa_flow_anchored(target, fidelity="entrywise", max_iter=5)
    assert r_anc["fidelity_mode"] == "anchor"

    r_flow = nsa_flow(target, fidelity="entrywise", max_iter=5)
    assert r_flow["fidelity_mode"] == "anchor"


# -----------------------------------------------------------------------------
# Boundary Parameters & Adversarial Stress Tests (Reviewer 3)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("dev", AVAILABLE_DEVICES)
def test_boundary_k1_across_entry_points(dev):
    """Verify k=1 boundary condition across all 4 entry points."""
    dtype = torch.float32 if dev.type == "mps" else torch.float64

    # 1. _nsa_flow_anchored with k=1 target
    target = torch.randn(20, 1, dtype=dtype, device=dev).abs()
    r_anc = _nsa_flow_anchored(target, max_iter=10)
    assert r_anc.Y.shape == (20, 1)
    assert r_anc.Y.device.type == dev.type
    assert torch.isfinite(r_anc.Y).all()

    # 2. nsa_flow_data with k=1
    X = torch.randn(25, 6, dtype=dtype, device=dev).abs()
    r_data = nsa_flow_data(X, k=1, max_iter=10)
    assert r_data.Y.shape == (6, 1)
    assert r_data.Y.device.type == dev.type
    assert torch.isfinite(r_data.Y).all()

    # 3. nsa_flow_signed with k=1
    X_s = torch.randn(25, 6, dtype=dtype, device=dev)
    r_signed = nsa_flow_signed(X_s, k=1, max_iter=10)
    assert r_signed.Y.shape == (6, 1)
    assert r_signed.Y.device.type == dev.type
    assert torch.isfinite(r_signed.Y).all()

    # 4. Unified nsa_flow in anchored and data modes
    r_uni_anc = nsa_flow(target, max_iter=10)
    assert r_uni_anc.Y.shape == (20, 1)
    assert r_uni_anc.Y.device.type == dev.type

    r_uni_data = nsa_flow(X, k=1, max_iter=10)
    assert r_uni_data.Y.shape == (6, 1)
    assert r_uni_data.Y.device.type == dev.type


@pytest.mark.parametrize("dev", AVAILABLE_DEVICES)
def test_boundary_w_values(dev):
    """Verify boundary w=0.0 (pure fidelity) and w=1.0 (pure orthogonality)."""
    dtype = torch.float32 if dev.type == "mps" else torch.float64
    target = torch.randn(20, 3, dtype=dtype, device=dev).abs()

    # w=0.0
    r_w0 = _nsa_flow_anchored(target, w=0.0, max_iter=10)
    assert r_w0.Y.device.type == dev.type
    assert torch.isfinite(r_w0.Y).all()

    # w=1.0
    with pytest.warns(RuntimeWarning, match="w=1 drops the fidelity term"):
        r_w1 = _nsa_flow_anchored(target, w=1.0, max_iter=10)
    assert r_w1.Y.device.type == dev.type
    assert torch.isfinite(r_w1.Y).all()


@pytest.mark.parametrize("dev", AVAILABLE_DEVICES)
def test_boundary_nonneg_false(dev):
    """Verify nonneg=False (unconstrained optimization) on device."""
    dtype = torch.float32 if dev.type == "mps" else torch.float64
    target = torch.randn(20, 3, dtype=dtype, device=dev)

    r_anc = _nsa_flow_anchored(target, nonneg=False, max_iter=10)
    assert r_anc.Y.device.type == dev.type
    assert torch.isfinite(r_anc.Y).all()

    r_uni = nsa_flow(target, nonneg=False, max_iter=10)
    assert r_uni.Y.device.type == dev.type
    assert torch.isfinite(r_uni.Y).all()


def test_coerce_tensor_views_and_grad():
    """Verify coerce_tensor detaches autograd graph and safely handles non-contiguous slice views."""
    from nsa_flow.diagnostics import coerce_tensor
    base = torch.randn(30, 20, dtype=torch.float32, requires_grad=True)
    slice_view = base[::2, 2:8]
    assert not slice_view.is_contiguous()
    assert slice_view.requires_grad

    coerced = coerce_tensor(slice_view, device="cpu")
    assert not coerced.requires_grad
    assert coerced.shape == slice_view.shape

    res = _nsa_flow_anchored(slice_view.abs(), max_iter=10)
    assert torch.isfinite(res.Y).all()


@pytest.mark.skipif(not ACCELERATOR_DEVICES, reason="No accelerator device available")
@pytest.mark.parametrize("acc_dev", ACCELERATOR_DEVICES)
def test_adversarial_aspect_ratios_on_accelerator(acc_dev):
    """Verify extreme aspect ratios (n >> p and p >> n) on accelerator."""
    # Tall skinny target
    tall = torch.randn(400, 2, dtype=torch.float32, device=acc_dev).abs()
    r_tall = _nsa_flow_anchored(tall, max_iter=10)
    assert r_tall.Y.shape == (400, 2)
    assert torch.isfinite(r_tall.Y).all()

    # Short wide target
    wide = torch.randn(2, 40, dtype=torch.float32, device=acc_dev).abs()
    r_wide = _nsa_flow_anchored(wide, max_iter=10)
    assert r_wide.Y.shape == (2, 40)
    assert torch.isfinite(r_wide.Y).all()

    # Data matrix short & wide
    X_wide = torch.randn(5, 100, dtype=torch.float32, device=acc_dev).abs()
    r_data_wide = nsa_flow_data(X_wide, k=3, max_iter=10)
    assert r_data_wide.Y.shape == (100, 3)
    assert torch.isfinite(r_data_wide.Y).all()


@pytest.mark.parametrize("dev", AVAILABLE_DEVICES)
def test_adversarial_inputs_error_handling(dev):
    """Verify graceful handling or raising on empty, 1x1, and invalid targets."""
    dtype = torch.float32 if dev.type == "mps" else torch.float64

    # Empty target raises ValueError
    empty_target = torch.empty((0, 3), dtype=dtype, device=dev)
    with pytest.raises(ValueError):
        _nsa_flow_anchored(empty_target, max_iter=5)

    # 1x1 scalar-like matrix
    single = torch.tensor([[3.14]], dtype=dtype, device=dev)
    res = _nsa_flow_anchored(single, max_iter=5)
    assert res.Y.shape == (1, 1)
    assert torch.isfinite(res.Y).all()

