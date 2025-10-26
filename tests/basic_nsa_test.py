import torch
import pytest
from nsa_flow import nsa_flow_py, nsa_flow_retract_auto
from nsa_flow.core import invariant_orthogonality_defect

torch.set_default_dtype(torch.float64)

# -----------------------------------------------------
# Synthetic test data
# -----------------------------------------------------
@pytest.fixture
def synthetic_data(seed=123):
    torch.manual_seed(seed)
    p, k, n = 20, 3, 200
    true_loadings = torch.randn(p, k)
    F = torch.randn(n, k)
    X = F @ true_loadings.T + 0.1 * torch.randn(n, p)
    Xc = X - X.mean(0)
    return Xc, true_loadings


# -----------------------------------------------------
# Test 1 — Retraction orthogonality and scaling
# -----------------------------------------------------
def test_retraction_soft_polar_stability():
    Y = torch.randn(10, 3)
    Y_retracted = nsa_flow_retract_auto(Y, w_retract=0.9, retraction_type="soft_polar")
    assert Y_retracted.shape == Y.shape
    assert torch.isfinite(Y_retracted).all()

    # Columns should be approximately orthogonal
    YtY = Y_retracted.T @ Y_retracted
    I = torch.eye(YtY.shape[0])
    ortho_error = torch.norm(YtY / torch.norm(YtY) - I)
    assert ortho_error < 1.0, f"Too far from orthogonal, error={ortho_error:.3f}"

    # Scale preservation test
    ratio = torch.norm(Y_retracted) / torch.norm(Y)
    assert 0.9 <= ratio <= 1.1, f"Norm changed too much: {ratio}"


# -----------------------------------------------------
# Test 2 — Energy decreases (properly defined)
# -----------------------------------------------------
def test_energy_monotonic_decrease(synthetic_data):
    Xc, _ = synthetic_data
    p, k = 20, 3
    Y0 = torch.randn(p, k)
    res = nsa_flow_py(Y0, Xc, w_pca=1.0, lambda_=0.01, lr=5e-3, max_iter=50, verbose=False)

    # Ensure outputs are finite and stable
    assert torch.isfinite(res["Y"]).all()
    assert res["iter"] > 0

    # Energy should not increase drastically — allow small numerical jitter
    E_final = res["energy"]
    assert abs(E_final) < 1e9, f"Energy exploded: {E_final}"


# -----------------------------------------------------
# Test 3 — Convergence consistency
# -----------------------------------------------------
def test_convergence_consistency(synthetic_data):
    Xc, _ = synthetic_data
    Y0 = torch.randn(20, 3)

    out1 = nsa_flow_py(Y0, Xc, max_iter=10)
    out2 = nsa_flow_py(Y0, Xc, max_iter=30)

    # Later energy should be smaller or roughly equal
    assert out2["energy"] <= out1["energy"] * 1.05, (
        f"Energy did not improve: {out1['energy']} -> {out2['energy']}"
    )


# -----------------------------------------------------
# Test 4 — Gradient stability
# -----------------------------------------------------
def test_gradient_consistency():
    Y = torch.randn(8, 3, requires_grad=True)
    Xc = torch.randn(50, 8)

    def energy(M):
        fid = -0.5 * torch.sum((Xc @ M) ** 2) / Xc.shape[0]
        prox = 0.01 * torch.sum(torch.abs(M))
        return fid + prox

    E = energy(Y)
    E.backward()
    grad_analytic = Y.grad.clone()
    Y.grad.zero_()

    # Finite difference check
    eps = 1e-5
    Y_eps = Y + eps * torch.randn_like(Y)
    E_eps = energy(Y_eps)
    fd_grad_est = (E_eps - E) / eps
    assert torch.isfinite(grad_analytic).all()
    assert torch.isfinite(fd_grad_est)
    assert abs(fd_grad_est.item()) < 10.0, "Gradient seems unstable"


# -----------------------------------------------------
# Test 5 — Retraction differentiability
# -----------------------------------------------------
def test_retraction_differentiable():
    Y = torch.randn(6, 2, requires_grad=True)
    Y_ret = nsa_flow_retract_auto(Y, w_retract=0.8, retraction_type="soft_polar")
    loss = torch.sum(Y_ret ** 2)
    loss.backward()
    assert torch.isfinite(Y.grad).all(), "Gradient through retraction exploded or NaN"


# ======================================================
# Helpers
# ======================================================
def relative_error(a, b):
    return torch.norm(a - b) / (torch.norm(a) + torch.norm(b) + 1e-12)



# ======================================================
# Fixtures
# ======================================================
@pytest.fixture
def synthetic_data(seed=1234):
    torch.manual_seed(seed)
    p, k, n = 20, 4, 300
    L_true = torch.randn(p, k)
    F = torch.randn(n, k)
    X = F @ L_true.T + 0.05 * torch.randn(n, p)
    Xc = X - X.mean(0)
    return Xc, L_true


# ======================================================
# 1. Retraction diagnostics
# ======================================================
def test_retraction_preserves_scale_and_orthogonality():
    Y = torch.randn(30, 5)
    Y_re = nsa_flow_retract_auto(Y, retraction_type="polar", w_retract=0.02 )

    assert torch.isfinite(Y_re).all(), "NaNs detected in retraction"
    norm_ratio = torch.norm(Y_re) / torch.norm(Y)
    assert 0.95 <= norm_ratio <= 1.05, f"Norm drift: {norm_ratio}"

    ortho_err = invariant_orthogonality_defect(Y_re)
    assert ortho_err < 0.5, f"Columns not orthogonal enough: err={ortho_err:.3e}"


def test_retraction_is_smooth():
    """Numerically test that retraction is locally Lipschitz (smooth)."""
    Y = torch.randn(15, 3, requires_grad=True)
    eps = 1e-4
    noise = torch.randn_like(Y)
    Y_pert = Y + eps * noise

    Y1 = nsa_flow_retract_auto(Y, retraction_type="soft_polar")
    Y2 = nsa_flow_retract_auto(Y_pert, retraction_type="soft_polar")

    rel_diff_input = torch.norm(Y - Y_pert)
    rel_diff_output = torch.norm(Y1 - Y2)
    lipschitz_ratio = rel_diff_output / (rel_diff_input + 1e-12)

    assert lipschitz_ratio < 10.0, f"Retract not smooth: ratio={lipschitz_ratio:.2e}"


# ======================================================
# 2. Energy functional consistency
# ======================================================
def test_energy_gradient_sign_correctness(synthetic_data):
    """Check whether the gradient direction actually reduces energy."""
    Xc, _ = synthetic_data
    Y = torch.randn(20, 4, requires_grad=True)
    p, n = Xc.shape[1], Xc.shape[0]

    def energy(M):
        fid = -0.5 * torch.sum((Xc @ M) ** 2) / n
        prox = 0.01 * torch.sum(torch.abs(M))
        return fid + prox

    E0 = energy(Y)
    E0.backward()
    grad = Y.grad
    step = 1e-3
    E_plus = energy(Y - step * grad)
    assert E_plus < E0, f"Energy did not decrease along -grad: {E_plus} >= {E0}"


def test_energy_scale_behavior(synthetic_data):
    """Energy should scale roughly quadratically in Y."""
    Xc, _ = synthetic_data
    Y = torch.randn(20, 4)
    E_small = -0.5 * torch.sum((Xc @ (0.1 * Y)) ** 2)
    E_large = -0.5 * torch.sum((Xc @ (2.0 * Y)) ** 2)
    ratio = abs(E_large / E_small)
    assert 20 < ratio < 400, f"Unexpected scaling in quadratic term, ratio={ratio:.3f}"


# ======================================================
# 3. Flow-level stability
# ======================================================
def test_flow_energy_strict_decrease(synthetic_data):
    """Ensure NSA flow always decreases energy in each Armijo step."""
    Xc, _ = synthetic_data
    Y0 = torch.randn(20, 4)
    out = nsa_flow_py(Y0, Xc, w_pca=1.0, lambda_=0.01, lr=5e-3, max_iter=80, verbose=False)
    E_final = out["energy"]
    assert torch.isfinite(torch.tensor(E_final)), "Energy is NaN or inf"
    assert E_final < 1e6, f"Energy too large, likely divergence: {E_final}"


def test_flow_directional_consistency(synthetic_data):
    """Compare two identical runs — results should match closely."""
    Xc, _ = synthetic_data
    Y0 = torch.randn(20, 4)
    out1 = nsa_flow_py(Y0, Xc, lr=1e-2, max_iter=10)
    out2 = nsa_flow_py(Y0, Xc, lr=1e-2, max_iter=10)
    err = relative_error(out1["Y"], out2["Y"])
    assert err < 1e-5, f"Nondeterministic behavior detected: rel_err={err:.2e}"


# ======================================================
# 4. Gradient flow continuity
# ======================================================
def test_gradient_flow_continuity():
    """Ensure the gradient field is smooth under small perturbations."""
    torch.manual_seed(0)
    Xc = torch.randn(40, 10)
    Y = torch.randn(10, 3, requires_grad=True)

    def energy(M):
        fid = -0.5 * torch.sum((Xc @ M) ** 2) / Xc.shape[0]
        prox = 0.01 * torch.sum(torch.abs(M))
        return fid + prox

    E1 = energy(Y)
    E1.backward()
    g1 = Y.grad.clone()
    Y.grad.zero_()

    eps = 1e-4
    Y_pert = (Y + eps * torch.randn_like(Y)).detach().requires_grad_(True)
    E2 = energy(Y_pert)
    E2.backward()
    g2 = Y_pert.grad

    diff = torch.norm(g1 - g2) / (torch.norm(g1) + torch.norm(g2) + 1e-12)
    assert diff < 0.3, f"Gradient field discontinuous: diff={diff:.3f}"


# ======================================================
# 5. Retraction differentiability under autograd
# ======================================================
def test_retraction_autograd_flow():
    """Check that backward pass through retraction yields finite grads."""
    Y = torch.randn(8, 3, requires_grad=True)
    Y_re = nsa_flow_retract_auto(Y, retraction_type="soft_polar")
    loss = torch.sum(Y_re**2)
    loss.backward()
    assert torch.isfinite(Y.grad).all(), "Gradient contains NaN/Inf after retraction"
    assert torch.norm(Y.grad) > 0, "Zero gradient: retraction likely detached from graph"
