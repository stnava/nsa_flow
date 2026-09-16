import torch

def invariant_orthogonality_defect(V):
    """Scale-invariant orthogonality defect. Handles batched [B, p, k] or [p, k]."""
    if V.ndim == 3:
        # V: [B, p, k]
        norms2 = torch.sum(V**2, dim=(1, 2))
        S = V.mT @ V
        diagS = torch.diagonal(S, dim1=-2, dim2=-1)
        off_f2 = torch.sum(S**2, dim=(1, 2)) - torch.sum(diagS**2, dim=1)
        defects = off_f2 / (norms2**2).clamp_min(1e-12)
        return torch.mean(torch.where(norms2 <= 1e-12, torch.zeros_like(defects), defects))

    # Original 2D case
    norm2 = torch.sum(V ** 2)
    if norm2 <= 1e-12:
        return torch.tensor(0.0, device=V.device, dtype=V.dtype)
    S = V.T @ V
    diagS = torch.diag(S)
    off_f2 = torch.sum(S * S) - torch.sum(diagS ** 2)
    return off_f2 / (norm2 ** 2).clamp_min(1e-24)

def defect_fast(V):
    return invariant_orthogonality_defect(V)

def fidelity_basic(Y, X):
    """||Y - X||²"""
    return 0.5 * torch.sum((Y - X) ** 2)

def fidelity_scaled(Y, X):
    """||Y - X||² / ||X||². Handles batched [B, p, k] or [p, k]."""
    if Y.ndim == 3:
        num = torch.sum((Y - X)**2, dim=(1, 2))
        den = torch.sum(X**2, dim=(1, 2)).clamp_min(1e-12)
        return 0.5 * torch.mean(num / den)
    denom = torch.sum(X ** 2).clamp_min(1e-12)
    return 0.5 * torch.sum((Y - X) ** 2) / denom

def fidelity_symmetric(Y, X):
    """||Y - X||² / (||X||² + ||Y||²). Handles batched [B, p, k] or [p, k]."""
    if Y.ndim == 3:
        num = torch.sum((Y - X)**2, dim=(1, 2))
        den = 0.5 * (torch.sum(X**2, dim=(1, 2)) + torch.sum(Y**2, dim=(1, 2))).clamp_min(1e-12)
        return 0.5 * torch.mean(num / den)
    denom = 0.5 * (torch.sum(X ** 2) + torch.sum(Y ** 2)).clamp_min(1e-12)
    return 0.5 * torch.sum((Y - X) ** 2) / denom

def energy_fidelity(M, Xc, w):
    """Smooth fidelity energy used for autograd (no prox)."""
    n = Xc.shape[0]
    return -0.5 * w * torch.sum((Xc @ M) ** 2) / n

def compute_energy(
    Y, X0, w=0.5,
    fidelity_type="scale_invariant",
    orth_type="scale_invariant",
    fid_eta=1.0,
    c_orth=1.0,
    track_grad=True,
    return_dict=False,
):
    """
    Centralized energy computation for NSA-Flow.
    Combines fidelity and orthogonality losses into a total energy value.
    """
    w = float(w)
    fid_eta = float(fid_eta)
    c_orth = float(c_orth)

    # --- Fidelity term ---
    if fidelity_type == "basic":
        fidelity = fidelity_basic(Y, X0) * fid_eta
    elif fidelity_type == "scale_invariant":
        fidelity = fidelity_scaled(Y, X0) * fid_eta
    elif fidelity_type == "symmetric":
        fidelity = fidelity_symmetric(Y, X0) * fid_eta
    else:
        raise ValueError(f"Unknown fidelity_type: {fidelity_type}")

    # --- Orthogonality term ---
    if orth_type == "basic":
        if Y.ndim == 3:
            B, p, k = Y.shape
            I = torch.eye(k, device=Y.device, dtype=Y.dtype).unsqueeze(0)
            orth = c_orth * torch.mean((Y.mT @ Y - I) ** 2)
        else:
            I = torch.eye(Y.shape[1], device=Y.device, dtype=Y.dtype)
            orth = c_orth * torch.mean((Y.T @ Y - I) ** 2)
    elif orth_type == "scale_invariant":
        orth = c_orth * defect_fast(Y)
    else:
        raise ValueError(f"Unknown orth_type: {orth_type}")

    # --- Total energy ---
    total = fidelity * (1 - w) + orth * w

    if not track_grad:
        fidelity = fidelity.detach()
        orth = orth.detach()
        total = total.detach()

    if return_dict:
        return {
            "fidelity": fidelity,
            "orthogonality": orth,
            "total": total,
        }

    return total
