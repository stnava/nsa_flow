import torch

def _inv_sqrt_eig(A: torch.Tensor, eps: float = 1e-8, verbose: bool = False) -> torch.Tensor:
    """
    Compute (A + eps I)^{-1/2} using symmetric eigendecomposition.
    A must be symmetric (k x k).
    """
    w, V = torch.linalg.eigh(A)
    w_reg = w.clamp_min(eps)
    inv_sqrt_w = (w_reg**-0.5)
    if verbose:
        small = (w_reg <= eps).sum().item()
        print(f"_inv_sqrt_eig: min_eig={w_reg.min().item():.3e}, #clamped={small}")
    return (V * inv_sqrt_w.unsqueeze(-2)) @ V.transpose(-2, -1)

def _inv_sqrt_newton_schulz(A: torch.Tensor, eps: float = 1e-8,
                            ns_iter: int = 10, verbose: bool = False) -> torch.Tensor:
    """
    Newton-Schulz iteration for matrix inverse square root of (A + eps I).
    This expects A to be positive definite and reasonably well-conditioned after eps.
    Works best when ||I - A|| < 1 (so we scale).
    """
    k = A.shape[-1]
    I = torch.eye(k, device=A.device, dtype=A.dtype)
    # Ensure support for batched I
    if A.ndim == 3:
        I = I.unsqueeze(0).expand(A.shape[0], -1, -1)
        
    A_reg = A + eps * I

    # compute spectral norm for scaling (convergence requires ||I - A/norm|| < 1)
    normA = torch.linalg.norm(A_reg, ord=2, dim=(-2, -1), keepdim=True)
    normA = torch.where(normA == 0, torch.ones_like(normA), normA)
    Y = A_reg / normA
    Z = I.clone()

    if verbose:
        print(f"_inv_sqrt_newton_schulz: normA={normA.mean().item():.3e}, ns_iter={ns_iter}")

    for _ in range(ns_iter):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Z / torch.sqrt(normA)

def _inv_sqrt_diag(A: torch.Tensor, eps: float = 1e-8, verbose: bool = False) -> torch.Tensor:
    """
    Diagonal approximation: invert sqrt of diagonal entries only.
    """
    if A.ndim == 3:
        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        diag_reg = diag.clamp_min(eps)
        inv_sqrt = (diag_reg**-0.5)
        if verbose:
            print(f"_inv_sqrt_diag: min_diag={diag_reg.min().item():.3e}")
        return torch.diagonal_scatter(torch.zeros_like(A), inv_sqrt, dim1=-2, dim2=-1)
    else:
        diag = torch.diagonal(A, 0)
        diag_reg = diag.clamp_min(eps)
        inv_sqrt = (diag_reg**-0.5)
        if verbose:
            print(f"_inv_sqrt_diag: min_diag={diag_reg.min().item():.3e}")
        return torch.diag(inv_sqrt)

def inv_sqrt_sym_adaptive(YtY: torch.Tensor,
                          epsilon: float = 1e-8,
                          method: str = "eig",
                          ns_iter: int = 4,
                          eig_thresh: int = 128,
                          verbose: bool = False) -> torch.Tensor:
    """
    Adaptive inverse-square-root for symmetric positive-definite matrix YtY.
    method: "eig" | "ns" | "diag" | "auto"
    """
    k = YtY.shape[-1]
    if method == "auto":
        if k <= eig_thresh:
            method_use = "eig"
        else:
            method_use = "ns"
    else:
        method_use = method

    if method_use == "eig":
        return _inv_sqrt_eig(YtY, eps=epsilon, verbose=verbose)
    elif method_use == "ns":
        try:
            T = _inv_sqrt_newton_schulz(YtY, eps=epsilon, ns_iter=ns_iter, verbose=verbose)
            if torch.isfinite(T).all():
                return T
            else:
                if verbose:
                    print("Newton-Schulz produced non-finite entries; falling back to eig.")
                return _inv_sqrt_eig(YtY, eps=epsilon, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"Newton-Schulz error: {e}; falling back to eig.")
            return _inv_sqrt_eig(YtY, eps=epsilon, verbose=verbose)
    elif method_use == "diag":
        return _inv_sqrt_diag(YtY, eps=epsilon, verbose=verbose)
    else:
        raise ValueError(f"Unknown inv sqrt method: {method}")

def nsa_flow_retract_newton_schulz(Y: torch.Tensor, ns_iter: int = 5, eps: float = 1e-8) -> torch.Tensor:
    """
    Polar retraction using direct Newton-Schulz iteration on the matrix itself.
    """
    is_batched = (Y.ndim == 3)
    if not is_batched:
        Y_input = Y.unsqueeze(0)
    else:
        Y_input = Y

    B, p, k = Y_input.shape
    # Normalize by Frobenius norm to guarantee spectral norm <= 1.0 < sqrt(3)
    norm = torch.norm(Y_input, p='fro', dim=(1, 2), keepdim=True).clamp_min(eps)
    X = Y_input / norm

    I = torch.eye(k, device=Y.device, dtype=Y.dtype).unsqueeze(0).expand(B, -1, -1)

    for _ in range(ns_iter):
        XtX = X.mT @ X
        X = 0.5 * X @ (3.0 * I - XtX)

    if not is_batched:
        X = X.squeeze(0)
    return X

def nsa_flow_retract_cayley(Y: torch.Tensor, ns_iter: int = 5, eps: float = 1e-8) -> torch.Tensor:
    """
    Cayley transform-based retraction on the Stiefel manifold using QR-orthogonal base.
    """
    is_batched = (Y.ndim == 3)
    if not is_batched:
        Y_input = Y.unsqueeze(0)
    else:
        Y_input = Y

    B, p, k = Y_input.shape
    
    # We obtain the base point X via QR decomposition (which is on the Stiefel manifold)
    X, _ = torch.linalg.qr(Y_input)
    
    # Construct the skew-symmetric matrix W = Y X^T - X Y^T via Sherman-Morrison-Woodbury
    # U = [Y, X], V = [X, -Y]
    U = torch.cat([Y_input, X], dim=-1) # [B, p, 2k]
    V = torch.cat([X, -Y_input], dim=-1) # [B, p, 2k]
    
    # We solve (I - 0.5 * V^T U)^{-1} V^T X
    # VtU is of shape [B, 2k, 2k]
    VtU = V.mT @ U
    
    I_2k = torch.eye(2 * k, device=Y.device, dtype=Y.dtype).unsqueeze(0).expand(B, -1, -1)
    
    # Solve system: (I_2k - 0.5 * VtU) Z = VtX
    VtX = V.mT @ X # [B, 2k, k]
    lhs = I_2k - 0.5 * VtU
    
    # Differentiable solve
    Z = torch.linalg.solve(lhs, VtX) # [B, 2k, k]
    
    # Y_new = X + U Z
    Y_new = X + U @ Z
    
    if not is_batched:
        Y_new = Y_new.squeeze(0)
    return Y_new

def nsa_flow_retract_auto(
    Y: torch.Tensor,
    w_retract: torch.Tensor | float = 1.0,
    retraction_type: str = "soft_polar",
    eps_rf: float = 1e-6,
    max_condition: float = 1e4,
    verbose: bool = False,
    ns_iter: int = 5,
) -> torch.Tensor:
    """
    Differentiable retraction for NSA-Flow.
    """
    if not torch.is_tensor(w_retract):
        if float(w_retract) == 0.0:
            return Y.clone()
        w_retract_t = torch.tensor(float(w_retract), dtype=Y.dtype, device=Y.device)
    else:
        if (w_retract == 0.0).all():
            return Y.clone()
        w_retract_t = w_retract.to(dtype=Y.dtype, device=Y.device)

    if Y.ndim == 3:
        normY = torch.norm(Y, dim=(1, 2), keepdim=True)
    else:
        normY = torch.norm(Y)

    if (normY < 1e-12).any() or not torch.isfinite(normY).all():
        return Y.clone()

    try:
        if retraction_type in ["none"]:
            return Y.clone()

        if retraction_type in ["soft_polar", "polar"]:
            orig_dtype = Y.dtype
            Y_work = Y.to(torch.float64)
            U, S, Vh = torch.linalg.svd(Y_work, full_matrices=False)

            S = S.clamp_min(eps_rf)
            if max_condition is not None and max_condition > 0:
                s_max = S.max(dim=-1, keepdim=True).values
                s_floor = s_max / max_condition
                S = torch.where(S < s_floor, s_floor.expand_as(S), S)

            Y_polar = (U @ Vh).to(orig_dtype)

            if retraction_type == "soft_polar":
                Y_re = (1.0 - w_retract_t) * Y + w_retract_t * Y_polar
            else:
                Y_re = Y_polar

            if Y.ndim == 3:
                scale_factor = normY / torch.norm(Y_re, dim=(1, 2), keepdim=True).clamp_min(1e-12)
            else:
                scale_factor = normY / torch.norm(Y_re).clamp_min(1e-12)
            Y_re = Y_re * scale_factor

        elif retraction_type in ["newton_schulz", "soft_newton_schulz", "ns", "soft_ns"]:
            Y_polar = nsa_flow_retract_newton_schulz(Y, ns_iter=ns_iter, eps=eps_rf)
            if retraction_type in ["soft_newton_schulz", "soft_ns"]:
                Y_re = (1.0 - w_retract_t) * Y + w_retract_t * Y_polar
            else:
                Y_re = Y_polar
                
            if Y.ndim == 3:
                scale_factor = normY / torch.norm(Y_re, dim=(1, 2), keepdim=True).clamp_min(1e-12)
            else:
                scale_factor = normY / torch.norm(Y_re).clamp_min(1e-12)
            Y_re = Y_re * scale_factor

        elif retraction_type in ["cayley", "soft_cayley"]:
            Y_polar = nsa_flow_retract_cayley(Y, ns_iter=ns_iter, eps=eps_rf)
            if retraction_type == "soft_cayley":
                Y_re = (1.0 - w_retract_t) * Y + w_retract_t * Y_polar
            else:
                Y_re = Y_polar
                
            if Y.ndim == 3:
                scale_factor = normY / torch.norm(Y_re, dim=(1, 2), keepdim=True).clamp_min(1e-12)
            else:
                scale_factor = normY / torch.norm(Y_re).clamp_min(1e-12)
            Y_re = Y_re * scale_factor

        elif retraction_type == "normalize":
            norms = torch.norm(Y, dim=-2, keepdim=True).clamp_min(eps_rf)
            Y_re = Y / norms
            if Y.ndim == 3:
                scale_factor = normY / torch.norm(Y_re, dim=(1, 2), keepdim=True).clamp_min(1e-12)
            else:
                scale_factor = normY / torch.norm(Y_re).clamp_min(1e-12)
            Y_re = Y_re * scale_factor

        else:
            raise ValueError(f"Unknown retraction_type: {retraction_type}")

        if not torch.isfinite(Y_re).all():
            if verbose:
                print("⚠️ nsa_flow_retract_auto: non-finite values detected; reverting to input.")
            Y_re = Y.clone()

        return Y_re.to(dtype=Y.dtype, device=Y.device)

    except RuntimeError as e:
        if verbose:
            print(f"⚠️ Retraction fallback due to: {e}")
        return Y.clone()
