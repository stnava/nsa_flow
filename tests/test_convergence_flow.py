import torch
import torch.nn as nn
from nsa_flow import NSAFlowLayer, defect_fast
import numpy as np

def test_layer_by_layer_convergence():
    """Verify that multiple layers lead to lower orthogonality defect."""
    torch.manual_seed(42)
    p, k = 50, 10
    batch_size = 1
    X = torch.randn(batch_size, p, k)
    
    # Initialize 5 layers with w=0.5
    layers = nn.ModuleList([
        NSAFlowLayer(k=k, w_retract=0.5, residual=True, use_transform=False)
        for _ in range(5)
    ])
    
    defects = [defect_fast(X[0]).item()]
    print(f"Initial Defect: {defects[0]:.6e}")
    
    Y = X
    for i, layer in enumerate(layers):
        Y = layer(Y)
        d = defect_fast(Y[0]).item()
        defects.append(d)
        print(f"Layer {i+1} Defect: {d:.6e}")
    
    # Check if defect is monotonically decreasing
    for i in range(len(defects)-1):
        assert defects[i+1] <= defects[i] + 1e-8, f"Defect increased at layer {i+1}!"
    
    print("Layer-by-layer convergence check passed.")

if __name__ == "__main__":
    test_layer_by_layer_convergence()
