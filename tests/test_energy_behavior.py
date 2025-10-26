import torch
import matplotlib.pyplot as plt
import nsa_flow  # make sure your module is importable
from nsa_flow import nsa_flow,  nsa_flow_retract_auto, invariant_orthogonality_defect, defect_fast, nsa_flow_autograd
torch.manual_seed(42)
Y = torch.randn(50, 10)
X0 = torch.randn(Y.shape[0], Y.shape[1])
retraction='soft_polar'
###################
result = nsa_flow(
        Y,
        w=0.8, 
        retraction='soft_polar',
        optimizer='adam',
        max_iter=50,
        record_every=1,
        tol=1e-8,
        initial_learning_rate=1e-2,
        verbose=True,
    )
invariant_orthogonality_defect(Y)
invariant_orthogonality_defect(result['Y'])

def nsa_energy(V, w=0.5, retraction='soft_polar',apply_nonneg=False):
    Vp = nsa_flow_retract_auto(V, w_retract=w, retraction_type=retraction)
    if apply_nonneg:
        Vp = torch.clamp(Vp, min=0.0)
    fid = 0.5 * torch.sum((Vp - X0) ** 2)
    orth_term = 0.25 * w * defect_fast(Vp) if w > 0 else 0.0
    print( fid )
    print( orth_term )
    return fid + orth_term

nsa_energy(Y0)
E0 = nsa_energy(Y)
Yp = Y.clone().detach().requires_grad_(True)
E1 = nsa_energy(Yp)
print(E0.item(), E1.item())


result = nsa_flow_autograd(
        Y0,
        X0,
        w=0.5, 
        retraction='soft_polar',
        max_iter=50,
        record_every=1,
        tol=1e-8,
        initial_learning_rate=1e-4,
        optimizer="LARS",  # or "LARS"
        verbose=True,
    )


def test_energy_behavior():
    # Random seed for reproducibility
    torch.manual_seed(42)

    # Random initialization
    Y0 = torch.randn(12, 5)
Y0 = torch.randn(12, 5)
    # Run the algorithm
    result = nsa_flow.nsa_flow_autograd(
        Y0,
        max_iter=50,
        record_every=1,
        tol=1e-8,
        initial_learning_rate=1e-4,
        optimizer="adam",  # or "LARS"
        verbose=True,
    )

    # Extract energy trace
    energies = [t["total_energy"] for t in result["traces"]]

    # Print summary
    print("\nEnergy trace:")
    for i, e in enumerate(energies):
        print(f"Iter {i+1:3d}: {e:.6f}")

    # Compute monotonicity diagnostic
    diffs = torch.diff(torch.tensor(energies))
    num_increases = (diffs > 0).sum().item()
    print(f"\nNumber of increases: {num_increases} / {len(energies)-1}")
    print(f"Initial energy: {energies[0]:.6f}")
    print(f"Final energy:   {energies[-1]:.6f}")

    # Plot energy curve
    plt.figure(figsize=(6, 4))
    plt.plot(energies, marker="o", label="Total energy")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("NSA-Flow Energy Trend")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_energy_behavior()