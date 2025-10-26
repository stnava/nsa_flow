import torch
import torch.nn as nn
from nsa_flow import nsa_flow, nsa_flow_retract_auto, defect_fast, NSAFlowLayer
# Assuming you have the full definitions from previous context:
# - get_torch_optimizer
# - invariant_orthogonality_defect
# - defect_fast
# - inv_sqrt_sym_adaptive
# - nsa_flow_retract_auto
# - nsa_flow
# - NSAFlowLayer (as defined in the previous response)

# Paste those definitions here if not already in your script.
# For brevity, assuming they are available.

# Simple test script
def test_nsa_flow_layer(batch_size=2, p=20, k=5, apply_nonneg=False, max_iter=100, verbose=True):
    # Create dummy data: batch of matrices to approximate
    X0 = torch.randn(batch_size, p, k)
    if apply_nonneg:
        X0 = torch.abs(X0)  # Make non-negative for testing

    # Instantiate the layer
    layer = NSAFlowLayer(
        w=0.5,
        retraction="soft_polar",
        max_iter=max_iter,
        tol=1e-5,
        apply_nonneg=apply_nonneg,
        optimizer="adam",  # Or "fast"
        initial_learning_rate=0.01,  # Adjust as needed
        simplified=False,
        project_full_gradient=False
    )

    # Forward pass
    Y = layer(X0)

    # Verify shapes
    assert Y.shape == (batch_size, p, k), f"Unexpected shape: {Y.shape}"

    # Check non-negativity if enabled
    if apply_nonneg:
        assert torch.all(Y >= 0), "Non-negativity not enforced"

    # Compute orthogonality defect for each batch item
    defects = []
    for i in range(batch_size):
        defect = defect_fast(Y[i])
        defects.append(defect.item())
        if verbose:
            print(f"Batch {i}: Orthogonality defect = {defect.item():.6e}")

    # Fidelity: mean squared error to X0
    mse = torch.mean((Y - X0) ** 2).item()
    if verbose:
        print(f"Mean MSE to target: {mse:.6e}")
        print(f"Average defect: {sum(defects)/batch_size:.6e}")

    return Y

# Run the test
if __name__ == "__main__":
    print("Testing without non-negativity:")
    test_nsa_flow_layer(apply_nonneg=False, max_iter=200)

    print("\nTesting with non-negativity:")
    test_nsa_flow_layer(apply_nonneg=True, max_iter=200)

