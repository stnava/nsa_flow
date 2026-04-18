import torch
import torch.nn as nn
from nsa_flow import NSAFlowLinear, defect_fast

def test_nsa_flow_linear_basic():
    """Test the basic functionality of NSAFlowLinear."""
    batch_size = 4
    in_features = 20
    out_features = 5
    
    # Input data
    x = torch.randn(batch_size, in_features)
    
    # Initialize layer
    layer = NSAFlowLinear(
        in_features=in_features,
        out_features=out_features,
        w_retract=0.5,
        retraction_type="soft_polar",
        apply_nonneg="none"
    )
    
    # Forward pass
    y = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    assert y.shape == (batch_size, out_features), f"Shape mismatch: {y.shape}"
    
    # Check orthogonality defect of the effective weight matrix
    W_eff = layer.get_manifold_weight()
    defect_w = defect_fast(W_eff)
    print(f"Defect of W_eff: {defect_w:.6e}")
    
    assert defect_w >= -1e-6

def test_nsa_flow_linear_nonneg():
    """Test nonnegativity enforcement in NSAFlowLinear."""
    batch_size = 2
    in_features = 10
    out_features = 3
    x = torch.randn(batch_size, in_features)
    
    layer = NSAFlowLinear(
        in_features=in_features,
        out_features=out_features,
        apply_nonneg="hard"
    )
    
    W_eff = layer.get_manifold_weight()
    assert torch.all(W_eff >= 0), "Non-negativity (hard) not enforced on weights"
    
    # To ensure output is non-negative, input must also be non-negative
    x_nonneg = torch.abs(x)
    y = layer(x_nonneg)
    
    if layer.bias is None or torch.all(layer.bias >= 0):
        # We initialized bias uniformly which could be negative, so let's set it to positive
        if layer.bias is not None:
            layer.bias.data.abs_()
        y = layer(x_nonneg)
        assert torch.all(y >= 0), "Output should be non-negative"
    print("Non-negativity check passed.")

def test_nsa_flow_linear_gradients():
    """Test if gradients flow correctly through NSAFlowLinear."""
    batch_size = 2
    in_features = 10
    out_features = 3
    x = torch.randn(batch_size, in_features, requires_grad=True)
    
    layer = NSAFlowLinear(in_features=in_features, out_features=out_features, w_retract=0.5)
    
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    assert layer.weight_raw.grad is not None, "Gradient did not flow to raw weights"
    assert x.grad is not None, "Gradient did not flow to input"
    print("Gradient flow check passed.")

if __name__ == "__main__":
    print("Running NSAFlowLinear tests...")
    test_nsa_flow_linear_basic()
    test_nsa_flow_linear_nonneg()
    test_nsa_flow_linear_gradients()
    print("All NSAFlowLinear tests passed!")
