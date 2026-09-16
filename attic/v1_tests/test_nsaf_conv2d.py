import torch
import torch.nn as nn
from nsa_flow import NSAFlowConv2d, defect_fast

def test_nsa_flow_conv2d_basic():
    """Test the basic functionality of NSAFlowConv2d."""
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    h, w = 16, 16
    
    x = torch.randn(batch_size, in_channels, h, w)
    
    layer = NSAFlowConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        w_retract=0.5,
        retraction_type="soft_polar",
        apply_nonneg="none"
    )
    
    y = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    assert y.shape == (batch_size, out_channels, h - kernel_size + 1, w - kernel_size + 1), f"Shape mismatch: {y.shape}"
    
    # Check orthogonality defect of the effective weight matrix
    W_eff = layer.get_manifold_weight()
    # Flatten spatial and in_channels to calculate orthogonal defect across out_channels
    W_flat = W_eff.view(W_eff.size(0), -1).T
    defect_w = defect_fast(W_flat)
    print(f"Defect of W_eff: {defect_w:.6e}")
    
    assert defect_w >= -1e-6

def test_nsa_flow_conv2d_nonneg():
    """Test nonnegativity enforcement in NSAFlowConv2d."""
    batch_size = 2
    in_channels = 2
    out_channels = 4
    kernel_size = 2
    x = torch.randn(batch_size, in_channels, 10, 10)
    
    layer = NSAFlowConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        apply_nonneg="hard"
    )
    
    W_eff = layer.get_manifold_weight()
    assert torch.all(W_eff >= 0), "Non-negativity (hard) not enforced on weights"
    print("Non-negativity check passed.")

def test_nsa_flow_conv2d_gradients():
    """Test if gradients flow correctly through NSAFlowConv2d."""
    batch_size = 2
    in_channels = 3
    out_channels = 5
    kernel_size = 3
    x = torch.randn(batch_size, in_channels, 8, 8, requires_grad=True)
    
    layer = NSAFlowConv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size, 
        w_retract=0.5
    )
    
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    assert layer.weight.grad is not None, "Gradient did not flow to raw weights"
    assert x.grad is not None, "Gradient did not flow to input"
    print("Gradient flow check passed.")

if __name__ == "__main__":
    print("Running NSAFlowConv2d tests...")
    test_nsa_flow_conv2d_basic()
    test_nsa_flow_conv2d_nonneg()
    test_nsa_flow_conv2d_gradients()
    print("All NSAFlowConv2d tests passed!")
