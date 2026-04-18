import torch
import torch.nn as nn
from nsa_flow import NSAFlowLayer, NSAFlowLinear, defect_fast

def demonstrate_layers():
    torch.manual_seed(42)
    
    # 1. Create a simple dataset: 100 samples, 8 features
    batch_size = 100
    in_features = 8
    out_features = 2
    
    X = torch.randn(batch_size, in_features)
    
    print("==================================================")
    print(f"Input Data X: shape {X.shape}")
    print("Goal: Project to a 2D basis while maintaining orthogonality.")
    print("==================================================\n")

    # -------------------------------------------------------------------
    # APPROACH A: NSAFlowLinear (Parameter-based Squeeze)
    # -------------------------------------------------------------------
    print("--- APPROACH A: NSAFlowLinear ---")
    print("This layer holds a matrix W (8x2). It forces W to be orthogonal.")
    
    # Initialize the layer (w_retract=0.5 pulls the weights halfway to the manifold)
    linear_layer = NSAFlowLinear(in_features, out_features, bias=False, w_retract=0.5)
    
    # Pass the data through the layer
    Z_linear = linear_layer(X)
    
    # Get the actual effective weight matrix used for the projection
    W_eff = linear_layer.get_manifold_weight()
    
    print(f"Output Features (Z) shape: {Z_linear.shape}")
    print(f"Basis Matrix (W_eff) shape: {W_eff.shape}")
    print(f"Orthogonality Defect of Basis (W_eff): {defect_fast(W_eff).item():.6f}")
    print(f"Orthogonality Defect of Features (Z):  {defect_fast(Z_linear).item():.6f}\n")


    # -------------------------------------------------------------------
    # APPROACH B: Standard Linear + NSAFlowLayer (Activation-based Squeeze)
    # -------------------------------------------------------------------
    print("--- APPROACH B: Standard nn.Linear -> NSAFlowLayer ---")
    print("This uses a standard Linear layer to project to 2D, then forces the *features* to be orthogonal.")
    
    # 1. Standard projection
    standard_proj = nn.Linear(in_features, out_features, bias=False)
    
    # 2. NSAFlowLayer to process the 2D activations
    # We turn off the internal MLP transform for a fairer comparison to a linear projection
    act_layer = NSAFlowLayer(k=out_features, w_retract=0.5, use_transform=False)
    
    # Pass data through standard projection
    Z_raw = standard_proj(X)
    
    # Pass the projected 2D data through the NSAFlowLayer
    Z_act = act_layer(Z_raw)
    
    # The basis is just the standard linear weights (transposed to match [in, out])
    W_standard = standard_proj.weight.T
    
    print(f"Output Features (Z) shape: {Z_act.shape}")
    print(f"Basis Matrix (W_standard) shape: {W_standard.shape}")
    print(f"Orthogonality Defect of Basis (W_standard): {defect_fast(W_standard).item():.6f}")
    print(f"Orthogonality Defect of Features (Z):       {defect_fast(Z_act).item():.6f}\n")

if __name__ == "__main__":
    demonstrate_layers()
