import os
import math
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa_flow import (
    safe_to_tensor,
    apply_nonnegativity,
    traces_to_dataframe,
    invariant_orthogonality_defect,
    defect_fast,
    fidelity_basic,
    fidelity_scaled,
    fidelity_symmetric,
    compute_energy,
    energy_fidelity,
    inv_sqrt_sym_adaptive,
    nsa_flow_retract_newton_schulz,
    nsa_flow_retract_cayley,
    nsa_flow_retract_auto,
    get_torch_optimizer,
    get_lr_estimation_strategies,
    estimate_learning_rate_for_nsa_flow,
    nsa_flow,
    nsa_flow_autograd,
    nsa_flow_orth,
    SimpleMLP,
    NSAFlowLayer,
    NSAFlowLinear,
    NSAFlowConv2d,
)

def test_scale_invariance():
    p, k = 30, 5
    X = torch.randn(p, k)
    Y = X + 0.1 * torch.randn(p, k)

    lr1 = estimate_learning_rate_for_nsa_flow(Y, X, w=0.5, strategy="armijo")["best_lr"]
    lr2 = estimate_learning_rate_for_nsa_flow(10 * Y, 10 * X, w=0.5, strategy="armijo")["best_lr"]

    assert abs(np.log10(lr1) - np.log10(lr2)) < 0.1, "Learning rate should be approximately scale invariant"

    res1 = nsa_flow_orth(Y, X, max_iter=50, verbose=False)
    res2 = nsa_flow_orth(10 * Y, 10 * X, max_iter=50, verbose=False)

    diff = torch.norm(res1["Y"] / torch.norm(res1["Y"]) - res2["Y"] / torch.norm(res2["Y"]))
    assert diff < 1e-1, "nsa_flow_orth output should be scale invariant"

@pytest.mark.parametrize("strategy", [
    "armijo", "armijo_aggressive", "exponential", "linear", 
    "entropy", "random", "adaptive", "momentum_boost", 
    "poly_decay", "grid", "bayes"
])
def test_lr_strategies(strategy):
    p, k = 20, 4
    X = torch.randn(p, k)
    Y = X + 0.1 * torch.randn(p, k)
    
    res = estimate_learning_rate_for_nsa_flow(
        Y, X, w=0.5, strategy=strategy, verbose=False, plot=False
    )
    assert "best_lr" in res
    assert res["best_lr"] > 0

def test_autograd_scale_invariance():
    p, k = 15, 3
    X = torch.randn(p, k)
    Y = X + 0.05 * torch.randn(p, k)

    res1 = nsa_flow_orth(Y, X, max_iter=30, verbose=False, lr_strategy="armijo")
    res2 = nsa_flow_orth(5 * Y, 5 * X, max_iter=30, verbose=False, lr_strategy="armijo")

    scale_diff = torch.norm(res1["Y"] / torch.norm(res1["Y"]) - res2["Y"] / torch.norm(res2["Y"]))
    assert scale_diff < 1e-1

def test_estimate_learning_rate_for_nsa_flow_modular():
    Y = torch.randn(40, 8)
    X = torch.randn_like(Y)

    res = estimate_learning_rate_for_nsa_flow(
        Y, X,
        w=0.4,
        retraction="soft_polar",
        fidelity_type="symmetric",
        orth_type="scale_invariant",
        strategy="armijo_aggressive",
        plot=False,
        verbose=False
    )

    assert "best_lr" in res
    assert res["best_lr"] > 0, "Learning rate should be positive."
    assert np.isfinite(res["losses"]).all(), "Losses contain NaN or Inf."

def test_retraction_across_w():
    Y = torch.randn(30, 5)
    ws = torch.linspace(0.0, 1.0, steps=5)
    norms = []
    orth = []

    for w in ws:
        Yr = nsa_flow_retract_auto(Y, w_retract=w, retraction_type="soft_polar")
        norms.append(torch.norm(Yr).item())
        orth.append(defect_fast(Yr).item())

    assert np.all(np.isfinite(norms))
    assert np.all(np.isfinite(orth))

def test_nsa_flow_orth_modular():
    Y = torch.randn(50, 10)
    X = torch.randn_like(Y)

    result = nsa_flow_orth(
        Y0=Y,
        X0=X,
        w=0.5,
        retraction="soft_polar",
        fidelity_type="scale_invariant",
        orth_type="scale_invariant",
        max_iter=50,
        lr_strategy="armijo",
        verbose=False,
    )
    
    Y_final = result["Y"]
    assert torch.isfinite(Y_final).all(), "Non-finite values detected in Y."
    assert result["best_total_energy"] <= result["traces"]["total_energy"].iloc[0], "Energy did not decrease"

def test_lr_aggression_monotonicity():
    Y = torch.randn(25, 4)
    X = torch.randn(25, 4)

    strategies = get_lr_estimation_strategies()
    # Skip entropy due to random sampling, and momentum_boost / random which might fluctuate
    stable_strategies = [s for s in strategies if s not in ["entropy", "momentum_boost", "random"]]
    aggressions = [0.0, 0.5, 1.0]

    for strategy in stable_strategies:
        lr_vals = []
        for agg in aggressions:
            res = estimate_learning_rate_for_nsa_flow(
                Y, X, w=0.5, retraction="soft_polar",
                strategy=strategy, aggression=agg,
                verbose=False, plot=False
            )
            lr_vals.append(res["best_lr"])
        
        lr_vals = np.array(lr_vals)
        diffs = np.diff(np.log10(lr_vals + 1e-12))
        assert np.all(diffs >= -1e-2), f"Strategy {strategy} failed monotonicity: {lr_vals}"

def test_mlp_then_nsa_joint_residual(device):
    p, k, n = 8, 4, 100
    hidden = 32
    W_true = torch.randn(p, k, device=device)
    X = torch.randn(n, p, device=device)
    Y_true = F.softplus(X @ W_true)
    Y_obs = Y_true + 0.05 * torch.randn_like(Y_true)

    n_train = int(0.8 * n)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y_obs[:n_train], Y_obs[n_train:]

    # Train simple MLP baseline
    mlp = SimpleMLP(p, k, hidden).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-2)
    for epoch in range(100):
        opt.zero_grad()
        Y_pred = mlp(X_train)
        loss = fidelity_scaled(Y_pred, Y_train)
        loss.backward()
        opt.step()

    # Train NSA refinement
    nsa = NSAFlowLayer(k, w_retract=0.5, apply_nonneg="hard", residual=False).to(device)
    opt_nsa = torch.optim.AdamW(nsa.parameters(), lr=1e-2)
    with torch.no_grad():
        Y0 = mlp(X_train)
    for epoch in range(100):
        opt_nsa.zero_grad()
        Y_ref = nsa(Y0)
        loss = fidelity_scaled(Y_ref, Y_train) + 0.1 * invariant_orthogonality_defect(Y_ref)
        loss.backward()
        opt_nsa.step()

    # Joint model
    joint_model = nn.Sequential(mlp, nsa)
    with torch.no_grad():
        Y_joint = joint_model(X_test)
    assert Y_joint.shape == (n - n_train, k)
    assert (Y_joint >= 0.0).all(), "Outputs must be non-negative"

@pytest.mark.parametrize("retraction_type", [
    "polar", "newton_schulz", "cayley", "ns", "soft_ns", 
    "soft_newton_schulz", "soft_cayley", "soft_polar"
])
def test_retraction_types(retraction_type):
    """Test all implemented retraction types in nsa_flow_retract_auto."""
    Y = torch.randn(15, 3)
    Yr = nsa_flow_retract_auto(Y, w_retract=1.0, retraction_type=retraction_type)
    
    assert Yr.shape == Y.shape
    assert torch.isfinite(Yr).all()
    
    # Check that columns are close to orthogonal
    orth_defect = invariant_orthogonality_defect(Yr).item()
    assert orth_defect < 1e-2

def test_lr_scheduler_integration():
    """Verify that ReduceLROnPlateau scheduler integrates without crash."""
    Y = torch.randn(20, 5)
    X = torch.randn_like(Y)
    
    res = nsa_flow_orth(
        Y, X, w=0.5,
        max_iter=20,
        lr_scheduler=True,
        lr_scheduler_patience=2,
        lr_scheduler_factor=0.5,
        verbose=False
    )
    assert torch.isfinite(res["Y"]).all()
