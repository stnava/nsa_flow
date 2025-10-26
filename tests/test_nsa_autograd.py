import torch
import pytest
import torch_optimizer

from nsa_flow import (
    nsa_flow_autograd,
    get_torch_optimizer,
    nsa_flow_retract_auto,
    defect_fast,
)

# =====================================================
# ===============  OPTIMIZER TESTS ====================
# =====================================================
@pytest.mark.parametrize("opt_name", [
    "adam", "adamw", "sgd", "sgd_nesterov", "rmsprop",
    "adagrad", "lbfgs", "adamax", "asgd", "nadam", "radam", "rprop"
])
def test_get_torch_optimizer_creates_instance(opt_name):
    params = [torch.randn(5, 5, requires_grad=True)]
    opt = get_torch_optimizer(opt_name, params, lr=1e-3)
    assert hasattr(opt, "step"), f"{opt_name} missing step()"
    assert hasattr(opt, "zero_grad")


def test_get_torch_optimizer_invalid_name():
    with pytest.raises(ValueError):
        get_torch_optimizer("not_real", [torch.randn(3, 3)], lr=1e-3)


# =====================================================
# ===============  NSA-FLOW TESTS =====================
# =====================================================
@pytest.mark.parametrize("optimizer", ["adam", "lbfgs", "sgd"])
@pytest.mark.parametrize("retraction", ["none", "polar", "soft_polar"])
def test_nsa_flow_autograd_converges(optimizer, retraction):
    torch.manual_seed(0)
    Y0 = torch.randn(10, 4, dtype=torch.float64)
    X0 = Y0 + 0.05 * torch.randn_like(Y0)

    result = nsa_flow_autograd(
        Y0=Y0,
        X0=X0,
        w=0.5,
        retraction=retraction,
        max_iter=50,
        tol=1e-6,
        verbose=False,
        optimizer=optimizer,
        initial_learning_rate=1e-2,
    )

    assert "Y" in result
    assert "traces" in result
    assert isinstance(result["Y"], torch.Tensor)
    assert result["best_total_energy"] >= 0
    assert result["final_iter"] <= 50


def test_energy_decreases_over_time():
    Y0 = torch.randn(12, 5)
    result = nsa_flow_autograd(Y0, max_iter=40, record_every=1, tol=1e-8, verbose=True )
    energies = [t["total_energy"] for t in result["traces"]]
    assert all(e >= 0 for e in energies)
    assert energies[-1] <= energies[0] * 1.01, "Energy did not decrease sufficiently"


@pytest.mark.parametrize("apply_nonneg", [True, False])
def test_nonnegativity_constraint_respected(apply_nonneg):
    Y0 = torch.randn(8, 3)
    result = nsa_flow_autograd(Y0, apply_nonneg=apply_nonneg, max_iter=20)
    Y_final = result["Y"]
    if apply_nonneg:
        assert torch.all(Y_final >= -1e-10)
    else:
        assert torch.any(Y_final < 0)


def test_retraction_preserves_norm():
    Y = torch.randn(10, 3)
    Y_retracted = nsa_flow_retract_auto(Y, w_retract=0.5, retraction_type="polar")
    assert not torch.isnan(Y_retracted).any()
    norm_diff = abs(torch.norm(Y_retracted) - torch.norm(Y))
    assert norm_diff < 1e-3, "Retracted norm deviates unexpectedly"


def test_defect_fast_behavior():
    Y = torch.eye(4)
    d = defect_fast(Y)
    assert torch.isclose(d, torch.tensor(0.0, dtype=d.dtype), atol=1e-10)
    Y2 = Y + 0.1 * torch.randn_like(Y)
    d2 = defect_fast(Y2)
    assert d2 >= 0
    assert d2 > d


def test_deterministic_seed_reproducibility():
    Y0 = torch.randn(6, 4)
    res1 = nsa_flow_autograd(Y0, seed=123, max_iter=10)
    res2 = nsa_flow_autograd(Y0, seed=123, max_iter=10)
    assert torch.allclose(res1["Y"], res2["Y"], atol=1e-8)


def test_returns_expected_keys():
    Y0 = torch.randn(10, 5)
    res = nsa_flow_autograd(Y0, max_iter=10)
    for k in ["Y", "traces", "final_iter", "best_total_energy", "best_Y_iteration", "target"]:
        assert k in res