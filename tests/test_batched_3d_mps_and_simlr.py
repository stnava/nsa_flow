"""Tests for v3.9.0 batched 3-D primitives, MPS Jacobi/polar factor, and per-lane retraction options.

Covers:
- CR-NSA-1: Native 3-D [B, p, k] `_resolve_column_sign_gauge`, `simlr_sparseness`,
  and `_any_column_denser` with zero-sync untied fast path and selective Pass-2 tie-breaking.
- CR-NSA-2: Arbitrary-batch `[..., n, n]` `jacobi_eigh` & `safe_eigh`, and zero-fallback
  3-D `polar_factor` forward + Sylvester backward (CPU float64 gradcheck & MPS float32 parity).
- CR-NSA-3: `scale_equivariant` and `allow_infeasible` options in `retract_anchored` across
  2-D/3-D CPU C++ and PyTorch (CPU/MPS) kernels.
"""
from unittest.mock import patch
import math
import pytest
import torch

import nsa_flow
from nsa_flow import polar_factor, retract_anchored, simlr_sparseness
from nsa_flow.linalg import jacobi_eigh, safe_eigh
from nsa_flow.project import _resolve_column_sign_gauge
from nsa_flow.solve import _any_column_denser, _retract_anchored_batched_torch


F64 = torch.float64
F32 = torch.float32
HAS_MPS = bool(torch.backends.mps.is_available())
DEVICES = [torch.device("cpu")] + ([torch.device("mps")] if HAS_MPS else [])


class ScalarSyncGuard:
    """Intercepts D2H scalar syncs (.item, float, bool, int) and .cpu() transfers."""

    def __init__(self):
        self.calls = []
        self._orig_item = torch.Tensor.item
        self._orig_float = torch.Tensor.__float__
        self._orig_bool = torch.Tensor.__bool__
        self._orig_int = torch.Tensor.__int__
        self._orig_cpu = torch.Tensor.cpu

    def __enter__(self):
        guard = self

        def item(self):
            guard.calls.append(("item", self.device))
            return guard._orig_item(self)

        def __float__(self):
            guard.calls.append(("__float__", self.device))
            return guard._orig_float(self)

        def __bool__(self):
            guard.calls.append(("__bool__", self.device))
            return guard._orig_bool(self)

        def __int__(self):
            guard.calls.append(("__int__", self.device))
            return guard._orig_int(self)

        def cpu(self, *a, **kw):
            guard.calls.append(("cpu", self.device))
            return guard._orig_cpu(self, *a, **kw)

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


def test_version_is_3_9_0():
    assert nsa_flow.__version__ == "3.9.0"


# =============================================================================
# CR-NSA-1: 3-D _resolve_column_sign_gauge, simlr_sparseness, _any_column_denser
# =============================================================================

@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("nonneg", [False, True])
@pytest.mark.parametrize("alg", ["soft", "hard"])
@pytest.mark.parametrize("only_if_denser", [False, True])
@pytest.mark.parametrize("q", [None, 0.0, 0.35, 0.75])
def test_simlr_sparseness_3d_matches_2d_slice_by_slice(dev, nonneg, alg, only_if_denser, q):
    dtype = F32 if dev.type == "mps" else F64
    gen = torch.Generator().manual_seed(17)
    B, p, k = 8, 32, 4
    Y_cpu = torch.randn(B, p, k, generator=gen, dtype=dtype)
    # Inject special column patterns: all-zero col, already-sparse + dust col, mostly-negative col
    Y_cpu[1, :, 0] = 0.0
    Y_cpu[2, :, 1] = 1e-15 * Y_cpu[2, :, 1]
    Y_cpu[2, :3, 1] = torch.tensor([2.5, -1.8, 0.9], dtype=dtype)
    Y_cpu[3, :, 2] = -Y_cpu[3, :, 2].abs() - 0.2

    Y = Y_cpu.to(dev)
    out_3d = simlr_sparseness(Y, sparseness=q, nonneg=nonneg, alg=alg, only_if_denser=only_if_denser)
    out_loop = torch.stack([
        simlr_sparseness(Y[b], sparseness=q, nonneg=nonneg, alg=alg, only_if_denser=only_if_denser)
        for b in range(B)
    ], dim=0)
    assert out_3d.shape == (B, p, k)
    assert out_3d.device.type == dev.type
    assert torch.equal(out_3d, out_loop)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("alg", ["soft", "hard"])
@pytest.mark.parametrize("nonneg", [False, True])
def test_simlr_sparseness_3d_selective_tie_breaking_hadamard(dev, alg, nonneg):
    """Batch mixing untied random matrices with tied 4x4 Walsh-Hadamard & constant columns."""
    dtype = F32 if dev.type == "mps" else F64
    gen = torch.Generator().manual_seed(23)
    B, p, k = 6, 4, 4
    Y_cpu = torch.randn(B, p, k, generator=gen, dtype=dtype)
    H4 = 0.5 * torch.tensor([
        [1.0,  1.0,  1.0,  1.0],
        [1.0, -1.0,  1.0, -1.0],
        [1.0,  1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0,  1.0],
    ], dtype=dtype)
    Y_cpu[1] = H4
    Y_cpu[4] = torch.ones(p, k, dtype=dtype)

    Y = Y_cpu.to(dev)
    out_3d = simlr_sparseness(Y, sparseness=0.5, nonneg=nonneg, alg=alg)
    out_loop = torch.stack([
        simlr_sparseness(Y[b], sparseness=0.5, nonneg=nonneg, alg=alg)
        for b in range(B)
    ], dim=0)
    assert torch.equal(out_3d, out_loop)
    # Hadamard lane 1 must remain full rank 4 on CPU (check on CPU to avoid MPS svd)
    assert int(torch.linalg.matrix_rank(out_3d[1].cpu())) == 4


def test_resolve_column_sign_gauge_3d_matches_2d_with_ties():
    gen = torch.Generator().manual_seed(5)
    Y = torch.randn(5, 6, 3, generator=gen, dtype=F64)
    # Lane 2: col 0 unambiguous positive on row 0; cols 1 and 2 have exact +/- lobe norm ties
    Y[2] = 0.0
    Y[2, 0, 0] = 2.0
    Y[2, 0, 1], Y[2, 1, 1] = 1.0, -1.0
    Y[2, 1, 2], Y[2, 2, 2] = 1.0, -1.0

    out_3d = _resolve_column_sign_gauge(Y)
    out_2d = torch.stack([_resolve_column_sign_gauge(Y[b]) for b in range(5)], dim=0)
    assert torch.equal(out_3d, out_2d)


@pytest.mark.parametrize("alg", ["soft", "hard"])
def test_simlr_sparseness_3d_autograd_gradients(alg):
    gen = torch.Generator().manual_seed(31)
    Y0 = torch.randn(3, 10, 3, generator=gen, dtype=F64)
    W = torch.randn(3, 10, 3, generator=gen, dtype=F64)

    Y_b = Y0.clone().requires_grad_(True)
    loss_b = (simlr_sparseness(Y_b, sparseness=0.5, nonneg=True, alg=alg) * W).sum()
    loss_b.backward()

    grads_2d = []
    for i in range(3):
        Yi = Y0[i].clone().requires_grad_(True)
        (simlr_sparseness(Yi, sparseness=0.5, nonneg=True, alg=alg) * W[i]).sum().backward()
        grads_2d.append(Yi.grad)
    assert torch.allclose(Y_b.grad, torch.stack(grads_2d, dim=0), atol=1e-14)


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_simlr_sparseness_3d_mps_zero_scalar_syncs():
    gen = torch.Generator().manual_seed(9)
    Y = torch.randn(64, 80, 3, generator=gen, dtype=F32).to("mps")
    with ScalarSyncGuard() as guard:
        out_soft = simlr_sparseness(Y, sparseness=0.6, nonneg=True, alg="soft", only_if_denser=True)
        out_hard = simlr_sparseness(Y, sparseness=0.6, nonneg=True, alg="hard", only_if_denser=True)
    assert out_soft.device.type == "mps" and out_hard.device.type == "mps"
    assert len(guard.calls) == 0, f"Unexpected scalar/cpu syncs: {guard.calls}"


def test_any_column_denser_3d_and_batched_torch_sparse_stage():
    gen = torch.Generator().manual_seed(41)
    Y = torch.zeros(4, 20, 3, dtype=F64)
    Y[0, :4, :] = torch.rand(4, 3, generator=gen, dtype=F64) + 0.5     # <= n_keep=6
    Y[1, :10, 1] = torch.rand(10, generator=gen, dtype=F64) + 0.5      # > n_keep=6
    Y[2] = 0.0                                                          # all zero
    Y[3, :6, :] = torch.rand(6, 3, generator=gen, dtype=F64) + 0.5
    Y[3, 6:, :] = 1e-15                                                 # dust only

    mask = _any_column_denser(Y, 6)
    expected = torch.tensor([_any_column_denser(Y[b], 6) for b in range(4)], dtype=torch.bool)
    assert mask.dtype == torch.bool and mask.shape == (4,)
    assert torch.equal(mask, expected)
    assert mask.tolist() == [False, True, False, False]

    # Verify k=1 3-D retract_anchored with sparseness uses 3-D simlr_sparseness
    X1 = torch.randn(5, 20, 1, generator=gen, dtype=F64)
    Y1 = retract_anchored(X1, w=0.5, nonneg=True, sparseness=0.75)
    Y1_ref = torch.stack([retract_anchored(X1[b], w=0.5, nonneg=True, sparseness=0.75) for b in range(5)])
    assert torch.equal(Y1, Y1_ref)


# =============================================================================
# CR-NSA-2: N-D jacobi_eigh, safe_eigh, and 3-D polar_factor (CPU & MPS)
# =============================================================================

@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (5, 3, 3), (4, 6, 6), (2, 3, 5, 5)])
def test_jacobi_eigh_and_safe_eigh_batched(dev, shape):
    dtype = F32 if dev.type == "mps" else F64
    atol = 1e-4 if dtype == F32 else 1e-10
    gen = torch.Generator().manual_seed(sum(shape))
    M = torch.randn(*shape, generator=gen, dtype=dtype)
    A_cpu = M @ M.transpose(-2, -1)
    # Include equal-diagonal 2x2 / nxn test in batch slot 0 when n >= 2
    n = shape[-1]
    if n >= 2 and len(shape) >= 3:
        idx0 = (0,) * (len(shape) - 2)
        eq = torch.eye(n, dtype=dtype) * 2.0
        eq[0, 1] = eq[1, 0] = 0.75
        A_cpu[idx0] = eq

    A = A_cpu.to(dev)
    w_j, V_j = jacobi_eigh(A)
    w_s, V_s = safe_eigh(A)
    w_ref, _ = torch.linalg.eigh(A_cpu.double())

    assert w_j.shape == shape[:-1] and V_j.shape == shape
    assert w_j.device.type == dev.type and V_j.device.type == dev.type
    assert w_s.device.type == dev.type and V_s.device.type == dev.type

    eye = torch.eye(n, dtype=dtype, device=dev).expand_as(V_j)
    for w_got, V_got in ((w_j, V_j), (w_s, V_s)):
        recon = V_got @ torch.diag_embed(w_got) @ V_got.transpose(-2, -1)
        orth = V_got.transpose(-2, -1) @ V_got
        scale = float(A_cpu.abs().max().clamp_min(1.0))
        assert torch.allclose(recon, A, atol=atol * scale)
        assert torch.allclose(orth, eye, atol=atol)
        assert torch.allclose(w_got.cpu().double(), w_ref, atol=atol * scale)


def test_polar_factor_3d_gradcheck_cpu_float64():
    """Verify exact Sylvester autograd backward on 3-D [B, n, k] including repeated singular values."""
    gen = torch.Generator().manual_seed(77)
    Y_gen = torch.randn(3, 6, 3, generator=gen, dtype=F64)
    # Slot 1: exact orthonormal columns -> repeated singular values sigma_1 = sigma_2 = sigma_3 = 1
    Y_gen[1] = torch.linalg.qr(Y_gen[1])[0]
    Y = Y_gen.clone().requires_grad_(True)

    U = polar_factor(Y)
    eye = torch.eye(3, dtype=F64).expand(3, 3, 3)
    assert torch.allclose(U.transpose(-2, -1) @ U, eye, atol=1e-12)
    U_2d = torch.stack([polar_factor(Y_gen[b]) for b in range(3)], dim=0)
    assert torch.allclose(U.detach(), U_2d, atol=1e-13)

    assert torch.autograd.gradcheck(lambda X: polar_factor(X), (Y,), eps=1e-6, atol=1e-9, rtol=1e-7)


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_polar_factor_3d_mps_zero_fallback_and_backward_finite_diff():
    """Verify 3-D polar_factor forward + Sylvester backward on MPS with zero CPU fallback."""
    gen = torch.Generator().manual_seed(88)
    B, n, k = 16, 24, 3
    Y_cpu64 = torch.randn(B, n, k, generator=gen, dtype=F64)
    # Include an equal-column-norm / orthonormal slice to exercise repeated singular values on MPS
    Y_cpu64[0] = torch.linalg.qr(Y_cpu64[0])[0]
    W_cpu64 = torch.randn(B, n, k, generator=gen, dtype=F64)
    dY_cpu64 = torch.randn(B, n, k, generator=gen, dtype=F64)
    dY_cpu64 = dY_cpu64 / dY_cpu64.norm()

    # CPU float64 analytical reference
    Y_ref = Y_cpu64.clone().requires_grad_(True)
    U_ref = polar_factor(Y_ref)
    (U_ref * W_cpu64).sum().backward()
    grad_ref = Y_ref.grad.detach()

    # MPS float32 with strict ban on .cpu() and torch.linalg.svd
    Y_mps = Y_cpu64.float().to("mps").detach().requires_grad_(True)
    W_mps = W_cpu64.float().to("mps")

    def forbidden(*_a, **_kw):
        raise AssertionError("Forbidden CPU fallback or SVD invoked during MPS polar_factor")

    with patch.object(torch.Tensor, "cpu", forbidden), patch("torch.linalg.svd", forbidden):
        U_mps = polar_factor(Y_mps)
        loss = (U_mps * W_mps).sum()
        loss.backward()

    assert U_mps.device.type == "mps"
    assert Y_mps.grad is not None and Y_mps.grad.device.type == "mps"

    eye_mps = torch.eye(k, dtype=F32, device="mps").expand(B, k, k)
    assert torch.allclose(U_mps.detach().transpose(-2, -1) @ U_mps.detach(), eye_mps, atol=1e-5)
    assert torch.allclose(U_mps.detach().cpu().double(), U_ref.detach(), atol=5e-5)
    assert torch.allclose(Y_mps.grad.detach().cpu().double(), grad_ref, atol=1e-4, rtol=1e-4)

    # On-device central finite-difference directional derivative check on MPS
    dY_mps = dY_cpu64.float().to("mps")
    eps_fd = 1e-3
    with torch.no_grad():
        Lp = (polar_factor(Y_mps.detach() + eps_fd * dY_mps) * W_mps).sum()
        Lm = (polar_factor(Y_mps.detach() - eps_fd * dY_mps) * W_mps).sum()
        fd_dir = float(((Lp - Lm) / (2.0 * eps_fd)).cpu())
        ad_dir = float((Y_mps.grad.detach() * dY_mps).sum().cpu())
    assert abs(ad_dir - fd_dir) <= 2e-3 * max(1.0, abs(ad_dir)), (ad_dir, fd_dir)


# =============================================================================
# CR-NSA-3: scale_equivariant and allow_infeasible in retract_anchored
# =============================================================================

@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("nonneg", [True, False])
@pytest.mark.parametrize("q", [None, 0.6])
def test_retract_anchored_scale_equivariant_2d_and_3d(dev, nonneg, q):
    dtype = F32 if dev.type == "mps" else F64
    tol = 1e-4 if dtype == F32 else 1e-7
    rtol = 2e-4 if dtype == F32 else 1e-10
    gen = torch.Generator().manual_seed(101)
    B, p, k = 5, 36, 3
    X_base_cpu = (torch.rand(B, p, k, generator=gen, dtype=dtype) + 0.1
                  if nonneg else torch.randn(B, p, k, generator=gen, dtype=dtype))
    I_base_cpu = (torch.rand(B, p, k, generator=gen, dtype=dtype) + 0.1
                  if nonneg else torch.randn(B, p, k, generator=gen, dtype=dtype))
    scales_cpu = torch.tensor([1e-10, 1e-4, 1.0, 1e4, 1e10], dtype=dtype)

    X_base = X_base_cpu.to(dev)
    I_base = I_base_cpu.to(dev)
    scales = scales_cpu.to(dev).reshape(B, 1, 1)

    Y_unit, info_u = retract_anchored(
        X_base, w=0.5, nonneg=nonneg, init=I_base, tol=tol,
        sparseness=q, scale_equivariant=True, return_info=True,
    )
    Y_scaled, info_s = retract_anchored(
        X_base * scales, w=0.5, nonneg=nonneg, init=I_base * scales, tol=tol,
        sparseness=q, scale_equivariant=True, return_info=True,
    )
    assert info_u["converged"] and info_s["converged"]
    # Non-power-of-two decimal scaling rounds float32 mantissas by 1 ULP
    max_ng_diff = 15 if dtype == F32 else 2
    assert abs(info_u["n_grad"] - info_s["n_grad"]) <= max_ng_diff
    assert torch.allclose(Y_scaled / scales, Y_unit, rtol=rtol, atol=rtol)

    # Exact power-of-two scaling preserves IEEE-754 mantissas bit-identically
    pow2_scales = torch.tensor([2.0 ** -24, 2.0 ** -8, 1.0, 2.0 ** 8, 2.0 ** 24],
                               dtype=dtype, device=dev).reshape(B, 1, 1)
    Y_pow2, info_p2 = retract_anchored(
        X_base * pow2_scales, w=0.5, nonneg=nonneg, init=I_base * pow2_scales,
        tol=tol, sparseness=q, scale_equivariant=True, return_info=True,
    )
    assert info_p2["n_grad"] == info_u["n_grad"]
    assert torch.equal(Y_pow2 / pow2_scales, Y_unit)

    # 2-D parity with pysimlr-style manual scaling ||X||_F / sqrt(k)
    X2 = X_base[0] * scales[0, 0, 0]
    I2 = I_base[0] * scales[0, 0, 0]
    Y2 = retract_anchored(X2, w=0.5, nonneg=nonneg, init=I2, tol=tol,
                          sparseness=q, scale_equivariant=True)
    sc2 = torch.linalg.vector_norm(X2) / math.sqrt(float(k))
    Y2_manual = retract_anchored(X2 / sc2, w=0.5, nonneg=nonneg, init=I2 / sc2,
                                 tol=tol, sparseness=q) * sc2
    assert torch.allclose(Y2, Y2_manual, rtol=rtol, atol=0.0)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("nonneg", [True, False])
@pytest.mark.parametrize("q", [None, 0.65])
def test_retract_anchored_allow_infeasible_3d_and_2d(dev, nonneg, q):
    dtype = F32 if dev.type == "mps" else F64
    tol = 1e-4 if dtype == F32 else 1e-6
    gen = torch.Generator().manual_seed(202)
    B, p, k = 7, 30, 3
    X_cpu = torch.rand(B, p, k, generator=gen, dtype=dtype) + 0.2

    # Lane 0: valid
    # Lane 1: NaN entry
    X_cpu[1, 3, 1] = float("nan")
    # Lane 2: Inf entry
    X_cpu[2, 0, 2] = float("inf")
    # Lane 3: all zeros
    X_cpu[3] = 0.0
    # Lane 4: column 1 is all zeros
    X_cpu[4, :, 1] = 0.0
    # Lane 5: column 2 is strictly negative (infeasible iff nonneg=True)
    X_cpu[5, :, 2] = -torch.rand(p, generator=gen, dtype=dtype) - 0.1
    # Lane 6: valid at scaled magnitude
    X_cpu[6] = X_cpu[6] * 500.0

    ws = [0.2, 0.3, 0.4, 0.5, 0.6, 0.45, 0.7]
    X = X_cpu.to(dev)

    # Default allow_infeasible=False still rejects non-finite input
    with pytest.raises(ValueError, match="non-finite"):
        retract_anchored(X, w=ws, nonneg=nonneg, sparseness=q)

    Y, info = retract_anchored(
        X, w=ws, nonneg=nonneg, tol=tol, sparseness=q,
        scale_equivariant=True, allow_infeasible=True, return_info=True,
    )
    Y_no_info = retract_anchored(
        X, w=ws, nonneg=nonneg, tol=tol, sparseness=q,
        scale_equivariant=True, allow_infeasible=True, return_info=False,
    )
    assert torch.equal(Y, Y_no_info)

    expected_feas = [True, False, False, False, False, (not nonneg), True]
    feas = info["feasible"]
    assert isinstance(feas, torch.Tensor) and feas.dtype == torch.bool
    assert feas.shape == (B,) and feas.device.type == dev.type
    assert feas.cpu().tolist() == expected_feas
    assert info["converged"] is True

    # Infeasible lanes are identically zero; feasible lanes match solving only X[feas]
    assert torch.all(Y[~feas] == 0.0)
    ws_feas = [w_i for w_i, ok in zip(ws, expected_feas) if ok]
    Y_ref = retract_anchored(
        X[feas], w=ws_feas, nonneg=nonneg, tol=tol, sparseness=q,
        scale_equivariant=True, allow_infeasible=False,
    )
    assert torch.allclose(Y[feas], Y_ref, atol=1e-12 if dtype == F64 else 1e-5)

    for b, ok in enumerate(expected_feas):
        if ok:
            assert info["stop_reasons"][b] in ("grad_map", "plateau")
            assert bool((torch.linalg.vector_norm(Y[b], dim=0) > 0).all().cpu())
        else:
            assert info["stop_reasons"][b] == "infeasible"
            assert math.isnan(info["grad_maps"][b])

    # All-infeasible batch returns zeros cleanly without crashing
    X_bad = X[~feas]
    Y_bad, info_bad = retract_anchored(
        X_bad, w=0.5, nonneg=nonneg, allow_infeasible=True, return_info=True,
    )
    assert Y_bad.shape == X_bad.shape and torch.all(Y_bad == 0.0)
    assert info_bad["feasible"].shape == (X_bad.shape[0],)
    assert not bool(info_bad["feasible"].any().cpu())
    assert info_bad["converged"] is False

    # 2-D allow_infeasible=True on infeasible slice
    Y_2d_bad, info_2d_bad = retract_anchored(
        X[4], w=0.5, nonneg=nonneg, allow_infeasible=True, return_info=True,
    )
    assert torch.all(Y_2d_bad == 0.0)
    assert bool(info_2d_bad["feasible"].cpu()) is False


def test_retract_anchored_batched_cpu_cpp_vs_torch_parity():
    """Verify C++ batched kernel and pure-PyTorch batched SPG reach the same optimum with new flags."""
    gen = torch.Generator().manual_seed(303)
    X = torch.rand(6, 40, 3, generator=gen, dtype=F64) + 0.1
    X[2] = 0.0  # one infeasible lane
    scales = torch.tensor([1e-3, 1.0, 1.0, 10.0, 1e3, 0.1], dtype=F64).reshape(6, 1, 1)
    X = X * scales

    Y_cpp, info_cpp = retract_anchored(
        X, w=0.5, nonneg=True, tol=1e-7, sparseness=0.6,
        scale_equivariant=True, allow_infeasible=True, return_info=True,
    )
    import nsa_flow.lbfgsb as lbfgsb_mod
    orig_native = lbfgsb_mod._native
    try:
        lbfgsb_mod._native = None
        Y_pt, info_pt = retract_anchored(
            X, w=0.5, nonneg=True, tol=1e-7, sparseness=0.6,
            scale_equivariant=True, allow_infeasible=True, return_info=True,
        )
    finally:
        lbfgsb_mod._native = orig_native

    assert torch.equal(info_cpp["feasible"], info_pt["feasible"])
    assert info_cpp["feasible"].tolist() == [True, True, False, True, True, True]
    assert torch.all(Y_cpp[2] == 0.0) and torch.all(Y_pt[2] == 0.0)
    rel = (Y_cpp - Y_pt).norm() / Y_cpp.norm()
    assert float(rel) < 5e-4


@pytest.mark.parametrize("dev", DEVICES)
def test_batched_torch_float32_large_batch_never_spins_at_zero_step(dev):
    """Regression: B=200 float32 SPG must reject d2 <= eps^2*||Y||^2 steps and certify plateau
    instead of accepting Y_try == Y (d2 == 0) up to max_iter=2000."""
    gen = torch.Generator().manual_seed(42)
    X = (torch.rand(200, 100, 3, generator=gen, dtype=F32) + 0.05).to(dev)
    if dev.type == "cpu":
        import nsa_flow.lbfgsb as lbfgsb_mod
        orig_native = lbfgsb_mod._native
        try:
            lbfgsb_mod._native = None
            Y, info = retract_anchored(
                X, w=0.5, nonneg=True, sparseness=0.7,
                scale_equivariant=True, allow_infeasible=True, return_info=True,
            )
        finally:
            lbfgsb_mod._native = orig_native
    else:
        Y, info = retract_anchored(
            X, w=0.5, nonneg=True, sparseness=0.7,
            scale_equivariant=True, allow_infeasible=True, return_info=True,
        )
    assert info["converged"] is True
    assert info["iters"] < 250, f"Float32 batched SPG spun to {info['iters']} iters"
    assert torch.isfinite(Y).all()

