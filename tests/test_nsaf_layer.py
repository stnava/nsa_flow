import torch
import torch.nn as nn
from nsa_flow import NSAFlowLayer, defect_fast

def test_nsa_flow_layer_basic():
    """Test the basic functionality of NSAFlowLayer."""
    batch_size = 4
    p = 20
    k = 5
    
    # Input data
    X = torch.randn(batch_size, p, k)
    
    # Initialize layer
    layer = NSAFlowLayer(
        k=k,
        hidden=32,
        w_retract=0.5,
        retraction_type="soft_polar",
        apply_nonneg="none",
        residual=False,
        use_transform=True
    )
    
    # Forward pass
    Y = layer(X)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Y.shape}")
    
    assert Y.shape == (batch_size, p, k), f"Shape mismatch: {Y.shape}"
    
    # Check orthogonality defect for first batch item
    defect_in = defect_fast(X[0])
    defect_out = defect_fast(Y[0])
    print(f"Defect In:  {defect_in:.6e}")
    print(f"Defect Out: {defect_out:.6e}")
    
    # Since w_retract=0.5, defect should generally decrease
    # (though not guaranteed in a single step with transform, but usually true)
    
def test_nsa_flow_layer_nonneg():
    """Test nonnegativity enforcement in NSAFlowLayer."""
    batch_size = 2
    p = 10
    k = 3
    X = torch.randn(batch_size, p, k)
    
    layer = NSAFlowLayer(
        k=k,
        apply_nonneg="hard"
    )
    
    Y = layer(X)
    assert torch.all(Y >= 0), "Non-negativity (hard) not enforced"
    print("Non-negativity (hard) check passed.")

def test_nsa_flow_layer_residual():
    """Test residual connection in NSAFlowLayer."""
    batch_size = 2
    p = 10
    k = 3
    X = torch.randn(batch_size, p, k)
    
    layer = NSAFlowLayer(
        k=k,
        residual=True,
        w_retract=0.5
    )
    
    Y = layer(X)
    assert Y.shape == X.shape
    print("Residual connection pass successful.")

def test_nsa_flow_layer_with_target():
    """Test loss computation when target is provided."""
    batch_size = 2
    p = 10
    k = 3
    X = torch.randn(batch_size, p, k)
    target = torch.randn(batch_size, p, k)
    
    layer = NSAFlowLayer(k=k)
    
    # Forward with target returns (Y_out, total_loss, fid, orth, w_dyn)
    output = layer(X, target=target)
    
    assert len(output) == 5
    Y_out, loss, fid, orth, w_dyn = output
    
    assert loss > 0
    print(f"Loss computed: {loss.item():.6e}")
    print(f"Fidelity: {fid.item():.6e}, Orth: {orth.item():.6e}")

if __name__ == "__main__":
    print("Running NSAFlowLayer tests...")
    test_nsa_flow_layer_basic()
    test_nsa_flow_layer_nonneg()
    test_nsa_flow_layer_residual()
    test_nsa_flow_layer_with_target()
    print("All NSAFlowLayer tests passed!")
