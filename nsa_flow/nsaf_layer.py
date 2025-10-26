import torch
import torch.nn as nn
from .core import nsa_flow, nsa_flow_retract_auto, defect_fast

class NSAFlowLayer(nn.Module):
    def __init__(self, w=0.5, retraction="soft_polar", max_iter=50, tol=1e-5,
                 apply_nonneg=True, optimizer="fast", initial_learning_rate="default",
                 simplified=False, project_full_gradient=False, device=None):
        super().__init__()
        self.w = w
        self.retraction = retraction
        self.max_iter = max_iter  # Reduced for NN efficiency
        self.tol = tol
        self.apply_nonneg = apply_nonneg
        self.optimizer = optimizer
        self.initial_learning_rate = initial_learning_rate
        self.simplified = simplified
        self.project_full_gradient = project_full_gradient
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, X0, Y0=None, seed=42):
        # X0: input tensor, e.g., [batch_size, p, k]
        # Assume batched; if not, unsqueeze(0)
        if X0.dim() == 2:
            X0 = X0.unsqueeze(0)
        
        batch_size, p, k = X0.shape
        outputs = []
        
        for i in range(batch_size):
            # Run nsa_flow per batch item (parallelize if needed)
            result = nsa_flow(
                Y0=Y0[i] if Y0 is not None else X0[i],  # Use X0 as initial if no Y0
                X0=X0[i],
                w=self.w,
                retraction=self.retraction,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=False,
                seed=seed,
                apply_nonneg=self.apply_nonneg,
                optimizer=self.optimizer,
                initial_learning_rate=self.initial_learning_rate,
                simplified=self.simplified,
                project_full_gradient=self.project_full_gradient,
                device=self.device
            )
            outputs.append(result["Y"])
        
        return torch.stack(outputs)  # Output: [batch_size, p, k]

# Example usage in a simple NN
class SimpleNet(nn.Module):
    def __init__(self, input_dim, k):
        super().__init__()
        self.linear = nn.Linear(input_dim, p)  # Assume p derived from input
        self.nsa_layer = NSAFlowLayer(w=0.7, max_iter=20, apply_nonneg=True)

    def forward(self, x):
        features = self.linear(x)  # Shape: [batch_size, p]
        features = features.unsqueeze(-1)  # Make [batch_size, p, k=1] for simplicity
        orthogonalized = self.nsa_layer(features)
        return orthogonalized.squeeze(-1)  # Or further process


