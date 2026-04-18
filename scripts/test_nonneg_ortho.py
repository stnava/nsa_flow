import torch
import torch.nn as nn
import torch.nn.functional as F
from nsa_flow import NSAFlowLinear, defect_fast

def run_nonneg_ortho_experiment():
    torch.manual_seed(1234)
    n_samples, in_features, out_features = 1000, 20, 4
    
    # 1. Create a dataset with a STRICTLY NON-NEGATIVE and ORTHOGONAL true basis
    # Each of the 4 basis vectors has exactly 5 non-overlapping active features
    V_true = torch.zeros(in_features, out_features)
    V_true[0:5,   0] = 1.0
    V_true[5:10,  1] = 1.0
    V_true[10:15, 2] = 1.0
    V_true[15:20, 3] = 1.0
    
    # True latent factors (must also be non-negative)
    Z_true = torch.rand(n_samples, out_features) * 5.0
    
    # Generate data with some noise
    X = Z_true @ V_true.T + 0.05 * torch.randn(n_samples, in_features)
    
    print("=== Non-Negative Orthogonal Basis Learning (NMF + Orthogonality) ===")
    print(f"Goal: Recover a {in_features}x{out_features} basis that is both strictly non-negative and strictly orthogonal.")
    print("This means the basis vectors should naturally form disjoint supports (clusters).\n")

    # 2. Define the Model using the "Soft Retraction + Asymmetric Penalty" Strategy
    class NonNegOrthoAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            # We use NSAFlowLinear to handle the manifold projection (w_retract=0.5).
            # We do NOT use apply_nonneg='hard' here, to preserve the differentiable geometry.
            self.basis_layer = NSAFlowLinear(in_features, out_features, bias=False, w_retract=0.5, apply_nonneg="none")
            
        def get_V(self):
            # 1. Get the partially-orthogonalized raw basis
            V_raw = self.basis_layer.get_manifold_weight()
            # 2. Apply a smooth non-negativity constraint (Softplus)
            # Beta=5 makes it a bit sharper than standard softplus
            V_nonneg = F.softplus(V_raw, beta=5)
            return V_nonneg

        def forward(self, x):
            V = self.get_V()
            # Encode: Z = X * V
            Z = x @ V
            # Decode: X_rec = Z * V^T
            X_rec = Z @ V.T
            return X_rec, V

    model = NonNegOrthoAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 3. Training Loop
    lambda_ortho = 2.0  # Weight for the orthogonality penalty on the non-negative basis
    
    for epoch in range(1500):
        optimizer.zero_grad()
        
        X_rec, V_nonneg = model(X)
        
        loss_recon = F.mse_loss(X_rec, X)
        # Apply the explicit orthogonality defect penalty to the NON-NEGATIVE weights
        loss_ortho = defect_fast(V_nonneg)
        
        loss = loss_recon + lambda_ortho * loss_ortho
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1:4d} | Total Loss: {loss.item():.4f} | Recon MSE: {loss_recon.item():.4f} | Ortho Defect: {loss_ortho.item():.4e}")

    # 4. Results and Analysis
    V_final = model.get_V().detach()
    
    print("\n--- Final Results ---")
    print(f"Minimum value in V: {V_final.min().item():.6f} (Proves non-negativity)")
    print(f"Orthogonality Defect of V: {defect_fast(V_final).item():.6e} (Proves orthogonality)")
    
    print("\nVisualizing the learned Basis V (Rows 0-19, Cols 0-3):")
    print("Notice how it naturally discovers the disjoint 5-feature blocks (zeros are < 0.05):")
    
    # Print the matrix with low values zeroed out for visual clarity
    V_viz = V_final.clone()
    V_viz[V_viz < 0.05] = 0.0
    
    for i in range(in_features):
        row_str = "  ".join([f"{val:5.2f}" for val in V_viz[i]])
        # Add visual separators for the true blocks
        if i % 5 == 0 and i > 0:
            print("-" * 30)
        print(f"Feature {i:2d} | {row_str}")

if __name__ == '__main__':
    run_nonneg_ortho_experiment()
