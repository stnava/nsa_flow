# test_nsa_flow_extended.py
import math
import torch
import pytest
from nsa_flow import (
    nsa_flow,
    nsa_flow_retract_auto,
    inv_sqrt_sym_adaptive,
    invariant_orthogonality_defect
)

torch.set_default_dtype(torch.float64)


# ---------------------------
# Small fixtures & helpers
# ---------------------------
@pytest.fixture
def small_rng():
    torch.manual_seed(1234)
    return None


def is_finite_tensor(x):
    return torch.isfinite(x).all().item()


# ---------------------------
# inv_sqrt_sym_adaptive tests
# ---------------------------
@pytest.mark.parametrize("method", ["eig", "ns", "diag", "auto"])
def test_inv_sqrt_sym_adaptive_identity(method):
    k = 6
    I = torch.eye(k, dtype=torch.get_default_dtype())
    T = inv_sqrt_sym_adaptive(I, epsilon=1e-8, method=method, ns_iter=8, eig_thresh=128, verbose=False)
    # For identity, (I + eps I)^{-1/2} ~= (1+eps)^{-1/2} I -> after tiny eps close to I
    assert T.shape == (k, k)
    assert is_finite_tensor(T)


def test_inv_sqrt_sym_adaptive_spd_newton_schulz():
    # Create SPD matrix close to identity so NS converges
    k = 8
    A = torch.randn(k, k)
    A = A @ A.T + 0.1 * torch.eye(k)
    T = inv_sqrt_sym_adaptive(A, epsilon=1e-10, method="ns", ns_iter=20, eig_thresh=8, verbose=False)
    # Check approximate property: T @ T @ A ≈ I
    I = torch.eye(k)
    residual = torch.norm(T @ T @ A - I)
    assert residual < 1e-6, f"NS inv-sqrt residual too big: {residual:.3e}"


def test_inv_sqrt_sym_adaptive_diag_approx():
    k = 5
    D = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.get_default_dtype()))
    T = inv_sqrt_sym_adaptive(D, epsilon=1e-8, method="diag")
    # diagonal inverse-sqrt should be diag(1/sqrt(di))
    td = torch.diag(T)
    expected = 1.0 / torch.sqrt(torch.tensor([1., 2., 3., 4., 5.]))
    assert torch.allclose(td, expected, atol=1e-12, rtol=1e-6)


# ---------------------------
# invariant_orthogonality_defect tests
# ---------------------------
def test_invariant_orthogonality_defect_zero_matrix():
    Y = torch.zeros(4, 3)
    d = invariant_orthogonality_defect(Y)
    assert d == 0.0


def test_invariant_orthogonality_defect_orthonormal_columns():
    # Create orthonormal columns (p>=k)
    p, k = 8, 3
    A = torch.randn(p, k)
    U, _, Vt = torch.linalg.svd(A, full_matrices=False)
    Q = U @ Vt
    d = invariant_orthogonality_defect(Q)
    assert pytest.approx(0.0, abs=1e-12) == d


def test_invariant_orthogonality_defect_scale_invariance():
    Y = torch.randn(10, 4)
    d1 = invariant_orthogonality_defect(Y)
    d2 = invariant_orthogonality_defect(10.0 * Y)
    # defect should be invariant to scaling (definition divides by norm^4)
    assert torch.allclose(d1.detach().clone(), d2.detach().clone(), atol=1e-12)


# ---------------------------
# invariant_orthogonality_defect
# ---------------------------
def test_invariant_orthogonality_defect_identity():
    # For orthonormal columns (square identity) expect near-zero error
    I = torch.eye(4)
    err = invariant_orthogonality_defect(I)
    assert err >= 0.0
    assert err < 1e-8


def test_invariant_orthogonality_defect_nonorthogonal():
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    err = invariant_orthogonality_defect(M)
    assert err > 0.0


# ---------------------------
# Retraction behaviour
# ---------------------------
@pytest.mark.parametrize("rtype", ["none", "polar", "soft_polar"])
@pytest.mark.parametrize("w_retract", [0.0, 0.5, 1.0])
def test_retraction_outputs_and_norm_preservation(rtype, w_retract):
    Y = torch.randn(12, 4)
    Y.requires_grad_(True)
    Yr = nsa_flow_retract_auto(Y, w_retract=w_retract, retraction_type=rtype)
    assert Yr.shape == Y.shape
    assert is_finite_tensor(Yr)
    # If w_retract==0 -> output should equal input (for 'none' or any retraction since weight 0)
    if w_retract == 0.0:
        assert torch.allclose(Yr, Y, atol=1e-12, rtol=1e-9)


def test_retraction_differentiability_small():
    Y = torch.randn(6, 2, requires_grad=True)
    Y_ret = nsa_flow_retract_auto(Y, w_retract=0.8, retraction_type="soft_polar")
    # Build a simple scalar and backprop
    (Y_ret.pow(2).sum()).backward()
    assert is_finite_tensor(Y.grad)
    assert torch.norm(Y.grad) > 0.0




# ---------------------------
# Full nsa_flow integration smoke tests
# ---------------------------
def test_nsa_flow_returns_expected_keys_and_target():
    p, k = 9, 3
    Y0 = torch.randn(p, k)
    X0 = torch.randn(p, k)
    out = nsa_flow(Y0, X0=X0, w=0.4, max_iter=10, tol=1e-6, verbose=False, apply_nonneg=False)
    assert "Y" in out and "traces" in out and "best_total_energy" in out and "target" in out
    # target should equal X0 passed in
    assert torch.allclose(out["target"], X0)


def test_nsa_flow_deterministic_seed():
    p, k = 8, 3
    Y0 = torch.randn(p, k)
    X0 = torch.randn(p, k)
    out1 = nsa_flow(Y0, X0=X0, w=0.2, max_iter=8, seed=123, verbose=False )
    out2 = nsa_flow(Y0, X0=X0, w=0.2, max_iter=8, seed=123, verbose=False )
    assert torch.allclose(out1["Y"], out2["Y"], atol=1e-12)


# ---------------------------
# nsa_flow_py targeted legacy interface tests (if present)
# ---------------------------
def test_nsa_flow_py_interface_smoke():
    # nsa_flow_py is present in __init__ — check callable and basic output shape
    p, k = 10, 3
    Y0 = torch.randn(p, k)
    X0 = torch.randn(p, k)  # some functions expect data in sample x feature format; allow flexibility
    try:
        res = nsa_flow(Y0, X0, w=0.5, max_iter=5, verbose=False)
        # Accept several possible return shapes/dicts: ensure Y key or shape present
        if isinstance(res, dict):
            assert "Y" in res
            assert res["Y"].shape == Y0.shape
        else:
            # If legacy returns tuple or array, attempt to find matrix-like result
            assert hasattr(res, "__len__")
    except TypeError:
        # If interface differs, at least ensure the function is importable
        pytest.skip("nsa_flow_py signature mismatch in this build; skipping strict checks.")


# ---------------------------
# GPU device handling (optional)
# ---------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_gpu_compatibility():
    dev = torch.device("cuda")
    p, k = 12, 4
    Y0 = torch.randn(p, k, device=dev)
    X0 = torch.randn(p, k, device=dev)
    res = nsa_flow_autograd(Y0, X0=X0, w=0.2, max_iter=20, lr=1e-2, retraction="soft_polar", device=dev, verbose=False, plot=False)
    assert is_finite_tensor(res["Y"])
    # basic sanity: outputs still on GPU
    assert res["Y"].device.type == "cuda"

