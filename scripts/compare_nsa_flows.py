import torch
import torch.nn as nn
from nsa_flow import NSAFlowLayer, NSAFlowLinear, defect_fast

def run_comparison():
    torch.manual_seed(42)
    n_samples, in_features, out_features = 500, 20, 5
    
    # Synthetic dataset
    true_basis = torch.randn(in_features, out_features)
    factors = torch.randn(n_samples, out_features)
    X = factors @ true_basis.T + 0.1 * torch.randn(n_samples, in_features)
    
    print("=== Autoencoder Representation Learning ===")
    print("Goal: Learn a 5D representation of 20D data using different NSA variants.\n")
    
    # 1. Standard PCA / Linear Algebra (Baseline)
    U, S, V = torch.pca_lowrank(X, q=out_features)
    # V is [in_features, out_features]
    Z_pca = X @ V
    recon_pca = Z_pca @ V.T
    mse_pca = torch.nn.functional.mse_loss(recon_pca, X).item()
    orth_pca = defect_fast(V).item()
    print(f"1. Standard PCA (Linear Algebra)")
    print(f"   MSE: {mse_pca:.4f}")
    print(f"   Orth Defect (Basis): {orth_pca:.4e}\n")
    
    # 2. NSAFlowLayer (Activation-based)
    class ActAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encode = nn.Linear(in_features, out_features, bias=False)
            self.nsa = NSAFlowLayer(k=out_features, hidden=16, w_retract=0.5, use_transform=True, apply_nonneg="none")
            self.decode = nn.Linear(out_features, in_features, bias=False)
        def forward(self, x):
            z_raw = self.encode(x)
            z_orth = self.nsa(z_raw)
            return self.decode(z_orth), z_orth

    model_act = ActAE()
    opt_act = torch.optim.Adam(model_act.parameters(), lr=1e-2)
    for _ in range(500):
        opt_act.zero_grad()
        x_rec, z_orth = model_act(X)
        # We need to penalize the orthogonality of the representation
        loss = torch.nn.functional.mse_loss(x_rec, X) + 0.5 * defect_fast(z_orth)
        loss.backward()
        opt_act.step()
    
    _, z_final_act = model_act(X)
    mse_act = torch.nn.functional.mse_loss(model_act(X)[0], X).item()
    orth_act_z = defect_fast(z_final_act).item()
    orth_act_w = defect_fast(model_act.encode.weight.T).item()
    print(f"2. NSAFlowLayer (Activation-based)")
    print(f"   MSE: {mse_act:.4f}")
    print(f"   Orth Defect (Activations): {orth_act_z:.4e}")
    print(f"   Orth Defect (Encoder Weights): {orth_act_w:.4e}\n")

    # 3. NSAFlowLinear (Parameter-based)
    class ParamAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encode = NSAFlowLinear(in_features, out_features, bias=False, w_retract=0.5, apply_nonneg="none")
            self.decode = nn.Linear(out_features, in_features, bias=False)
        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

    model_param = ParamAE()
    opt_param = torch.optim.Adam(model_param.parameters(), lr=1e-2)
    for _ in range(500):
        opt_param.zero_grad()
        x_rec, z_param = model_param(X)
        # No need to penalize representation, the layer handles weight orthogonality!
        loss = torch.nn.functional.mse_loss(x_rec, X)
        loss.backward()
        opt_param.step()
        
    _, z_final_param = model_param(X)
    mse_param = torch.nn.functional.mse_loss(model_param(X)[0], X).item()
    orth_param_w = defect_fast(model_param.encode.get_manifold_weight()).item()
    orth_param_z = defect_fast(z_final_param).item()
    print(f"3. NSAFlowLinear (Parameter-based)")
    print(f"   MSE: {mse_param:.4f}")
    print(f"   Orth Defect (Activations): {orth_param_z:.4e}")
    print(f"   Orth Defect (Encoder Weights): {orth_param_w:.4e}\n")

    from nsa_flow import NSAFlowConv2d
    
    print("=== Convolutional Autoencoder (Orthogonal Filters) ===")
    print("Goal: Learn a latent 2D representation of 1D spatial data using Conv2d.\n")
    
    n_images, channels, size = 100, 3, 16
    X_conv = torch.randn(n_images, channels, size, size)
    
    # 4. NSAFlowConv2d (Filter-based)
    class ConvAE(nn.Module):
        def __init__(self):
            super().__init__()
            # 3 -> 8 channels, kernel 3
            self.encode = NSAFlowConv2d(channels, 8, kernel_size=3, padding=1, w_retract=0.5, apply_nonneg="none")
            self.decode = nn.Conv2d(8, channels, kernel_size=3, padding=1)
        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

    model_conv = ConvAE()
    opt_conv = torch.optim.Adam(model_conv.parameters(), lr=1e-2)
    for _ in range(300):
        opt_conv.zero_grad()
        x_rec, z_conv = model_conv(X_conv)
        loss = torch.nn.functional.mse_loss(x_rec, X_conv)
        loss.backward()
        opt_conv.step()
        
    _, z_final_conv = model_conv(X_conv)
    mse_conv = torch.nn.functional.mse_loss(model_conv(X_conv)[0], X_conv).item()
    
    # Get flattened effective weight
    W_eff_conv = model_conv.encode.get_manifold_weight()
    W_flat_conv = W_eff_conv.view(W_eff_conv.size(0), -1).T
    orth_conv_w = defect_fast(W_flat_conv).item()
    
    # Check activation defect (flattening spatial dims)
    Z_flat_conv = z_final_conv.view(n_images, 8, -1).permute(0, 2, 1).reshape(-1, 8)
    orth_conv_z = defect_fast(Z_flat_conv).item()
    
    print(f"4. NSAFlowConv2d (Parameter-based)")
    print(f"   MSE: {mse_conv:.4f}")
    print(f"   Orth Defect (Activations): {orth_conv_z:.4e}")
    print(f"   Orth Defect (Encoder Filters): {orth_conv_w:.4e}\n")

if __name__ == '__main__':
    run_comparison()
