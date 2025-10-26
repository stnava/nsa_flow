import torch
import time
from typing import Optional

def invariant_orthogonality_defect(V):
        norm2 = torch.sum(V ** 2)
        if norm2 <= 1e-12:
            return torch.tensor(0.0, device=V.device)
        S = V.T @ V
        diagS = torch.diag(S)
        off_f2 = torch.sum(S * S) - torch.sum(diagS ** 2)
        return off_f2 / (norm2 ** 2)

def _inv_sqrt_eig(A: torch.Tensor, eps: float = 1e-8, verbose: bool = False) -> torch.Tensor:
    """
    Compute (A + eps I)^{-1/2} using symmetric eigendecomposition.
    A must be symmetric (k x k).
    """
    # eigh returns eigenvalues in ascending order
    w, V = torch.linalg.eigh(A)
    # regularize eigenvalues
    w_reg = w.clamp_min(eps)
    inv_sqrt_w = (w_reg**-0.5)
    if verbose:
        small = (w_reg <= eps).sum().item()
        print(f"_inv_sqrt_eig: min_eig={w_reg.min().item():.3e}, #clamped={small}")
    return (V * inv_sqrt_w.unsqueeze(0)) @ V.transpose(-2, -1)


def _inv_sqrt_newton_schulz(A: torch.Tensor, eps: float = 1e-8,
                            ns_iter: int = 10, verbose: bool = False) -> torch.Tensor:
    """
    Newton-Schulz iteration for matrix inverse square root of (A + eps I).
    This expects A to be positive definite and reasonably well-conditioned after eps.
    Works best when ||I - A|| < 1 (so we scale).
    """
    k = A.shape[-1]
    I = torch.eye(k, device=A.device, dtype=A.dtype)
    A_reg = A + eps * I

    # compute norm for scaling
    normA = torch.linalg.norm(A_reg)
    if normA == 0:
        return I  # fallback
    Y = A_reg / normA
    Z = I.clone()

    if verbose:
        print(f"_inv_sqrt_newton_schulz: normA={normA.item():.3e}, ns_iter={ns_iter}")

    for i in range(ns_iter):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    # Z approximates A^{-1/2} * sqrt(normA)
    return Z / torch.sqrt(normA)


def _inv_sqrt_diag(A: torch.Tensor, eps: float = 1e-8, verbose: bool = False) -> torch.Tensor:
    """
    Diagonal approximation: invert sqrt of diagonal entries only.
    Useful when matrix is approximately diagonal or to cheaply approximate.
    """
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
     - "eig": eigendecomposition
     - "ns": Newton-Schulz
     - "diag": diagonal fallback
     - "auto": choose based on dimension heuristics (k <= eig_thresh => eig, else ns)
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
        # try Newton-Schulz; if it fails (e.g. nan/inf), fallback to eig
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


def nsa_flow_retract_auto(Y_cand: torch.Tensor,
                          w_retract: float = 1.0,
                          retraction_type: str = "soft_polar",
                          eps_rf: float = 1e-8,
                          inv_method: str = "auto",
                          ns_iter: int = 4,
                          eig_thresh: int = 128,
                          diag_thresh: int = 8192,
                          verbose: bool = False) -> torch.Tensor:
    """
    Retraction onto Stiefel manifold (matches R nsa_flow_retract_auto behaviour).
    Y_cand: (p, k) torch tensor
    retraction_type: "polar" | "soft_polar" | "none"
    w_retract: blend weight in [0,1] where result = (1-w)*Y + w * retraction_operator(Y)
    inv_method: "eig" | "ns" | "diag" | "auto"
    ns_iter, eig_thresh, diag_thresh: heuristics from R version
    """
    if not torch.is_tensor(Y_cand):
        raise TypeError("Y_cand must be a torch.Tensor")
    if retraction_type is None or retraction_type == "none" or w_retract <= 0.0:
        return Y_cand.clone()

    p, k = Y_cand.shape
    device = Y_cand.device
    dtype = Y_cand.dtype

    normY = torch.norm(Y_cand)
    if normY < 1e-12:
        # trivial zero input -> just return copy (no retraction meaningful)
        return Y_cand.clone()

    retraction_type = retraction_type.lower()
    if retraction_type == "polar":
        # economy SVD
        U, S, Vh = torch.linalg.svd(Y_cand, full_matrices=False)
        Q = U @ Vh
        # preserve Frobenius norm optionally
        cur_norm = torch.norm(Q)
        if cur_norm > 1e-12:
            Q = Q / cur_norm * normY
        return Q.to(device=device, dtype=dtype).clone()

    if retraction_type == "soft_polar":
        # Decide whether to use additive SVD fallback (R's heuristic)
        use_additive_svd_fallback = False
        if (k > diag_thresh) and (p < k):
            use_additive_svd_fallback = True

        # (R code had an override that forced fallback; default behaviour should be heuristic-driven)
        if not use_additive_svd_fallback:
            # compute small k x k Gram matrix Y^T Y
            # crossprod in R is t(Y) %*% Y
            YtY = Y_cand.T @ Y_cand
            # compute inverse sqrt of (YtY + eps I)
            I_k = torch.eye(k, device=device, dtype=dtype)
            YtY = YtY + eps_rf * I_k

            T = inv_sqrt_sym_adaptive(YtY, epsilon=eps_rf, method=inv_method,
                                      ns_iter=ns_iter, eig_thresh=eig_thresh, verbose=verbose)

            # multiplicative soft-polar: T_w = (1-w) I + w * T
            T_w = (1.0 - w_retract) * I_k + w_retract * T
            Ytilde = Y_cand @ T_w

            # preserve Frobenius norm
            cur_norm = torch.norm(Ytilde)
            if cur_norm > 1e-12:
                Ytilde = Ytilde / cur_norm * normY

            return Ytilde.clone()
        else:
            # fallback: economy SVD additive blend
            U, S, Vh = torch.linalg.svd(Y_cand, full_matrices=False)
            Q = U @ Vh
            Ytilde = (1.0 - w_retract) * Y_cand + w_retract * Q
            cur_norm = torch.norm(Ytilde)
            if cur_norm > 1e-12:
                Ytilde = Ytilde / cur_norm * normY
            return Ytilde.clone()

    raise ValueError(f"unsupported retraction_type: {retraction_type}")



def energy_fidelity(M, Xc, w):
    """Smooth fidelity energy used for autograd (no prox)."""
    n = Xc.shape[0]
    # negative sign to match original formulation (we minimize energy = -fid + prox)
    return -0.5 * w * torch.sum((Xc @ M) ** 2) / n




def nsa_flow(Y0, X0=None, w=0.5,
             retraction="soft_polar",
             max_iter=500, tol=1e-5, verbose=False, seed=42,
             apply_nonneg=True, optimizer="fast",
             initial_learning_rate="default",
             record_every=1, window_size=5, c1_armijo=1e-6,
             simplified=False,
             device=None):
    """
    NSA-Flow optimization (PyTorch version)

    Parameters
    ----------
    Y0 : torch.Tensor
        Initial matrix (p x k).
    X0 : torch.Tensor or None
        Target matrix for fidelity.
    w : float
        Weight for orthogonality penalty (0..1).
    retraction : str
        Retraction method ('soft_polar', 'polar', 'none').
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress if True.
    seed : int
        Random seed.
    apply_nonneg : bool
        Enforce nonnegativity after each retraction if True.
    optimizer : str
        Only 'fast' is supported currently (simple gradient step).
    initial_learning_rate : float or str
        Learning rate or 'default' for auto-estimate.
    record_every : int
        Iteration interval for recording traces.
    window_size : int
        Energy stability window for convergence check.
    simplified : bool
        Use simplified orthogonality objective.
    device : torch.device or str or None
        Device to run on.

    Returns
    -------
    dict with keys:
        'Y', 'traces', 'final_iter', 'best_total_energy', 'best_Y_iteration'
    """

    torch.manual_seed(seed)
    if device is None:
        device = Y0.device

    Y = Y0.clone().detach().to(device)
    p, k = Y.shape

    if X0 is None:
        if apply_nonneg:
            X0 = torch.clamp(Y, min=0)
        else:
            X0 = Y.clone()
        perturb_scale = torch.norm(Y) / (Y.numel() ** 0.5) * 0.05
        Y = Y + perturb_scale * torch.randn_like(Y)
        if verbose:
            print("Added perturbation to Y0.")
    else:
        X0 = X0.to(device)
        if apply_nonneg:
            X0 = torch.clamp(X0, min=0)
        assert X0.shape == Y.shape, "X0 must match Y0 shape."

    # --- Defect and gradients ---
    def defect_fast(V):
        norm2 = torch.sum(V ** 2)
        if norm2 <= 1e-12:
            return torch.tensor(0.0, device=V.device)
        S = V.T @ V
        diagS = torch.diag(S)
        off_f2 = torch.sum(S * S) - torch.sum(diagS ** 2)
        return off_f2 / (norm2 ** 2)

    def compute_ortho_terms(Y, c_orth=1.0):
        norm2 = torch.sum(Y ** 2)
        if norm2 <= 1e-12 or c_orth <= 0:
            grad_orth = torch.zeros_like(Y)
            return grad_orth, torch.tensor(0.0, device=Y.device), norm2

        S = Y.T @ Y
        diagS = torch.diag(S)
        off_f2 = torch.sum(S * S) - torch.sum(diagS ** 2)
        defect = off_f2 / (norm2 ** 2)

        if simplified:
            grad_orth = -2 * c_orth * (Y @ (S - torch.diag(torch.diag(S))))
        else:
            Y_S = Y @ S
            Y_diag_scale = Y * diagS
            term1 = (Y_S - Y_diag_scale) / (norm2 ** 2)
            term2 = (defect / norm2) * Y
            grad_orth = c_orth * (term1 - term2)
        return grad_orth, defect, norm2

    def symm(A):
        return 0.5 * (A + A.T)

    # --- Scaling constants ---
    g0 = 0.5 * torch.sum((Y - X0) ** 2) / (p * k)
    g0 = torch.clamp(g0, min=1e-8)
    d0 = torch.clamp(defect_fast(Y), min=1e-8)
    fid_eta = (1 - w) / (g0 * p * k)
    c_orth = 4 * w / d0

    # --- Learning rate ---
    if initial_learning_rate == "default":
        lr = 1e-3 if apply_nonneg else 1.0
    else:
        lr = float(initial_learning_rate)

    traces = []
    recent_energies = []
    t0 = time.time()

    best_Y = Y.clone()
    best_total_energy = float("inf")
    best_Y_iteration = 0

    # --- Initial gradient for relative norm ---
    grad_fid_init = -fid_eta * (Y - X0)
    grad_orth_init, _, _ = compute_ortho_terms(Y, c_orth)
    sym_term_orth_init = symm(Y.T @ grad_orth_init)
    rgrad_orth_init = grad_orth_init - Y @ sym_term_orth_init
    rgrad_init = grad_fid_init + rgrad_orth_init
    init_grad_norm = torch.norm(rgrad_init) + 1e-8

    # --- Energy ---
    def nsa_energy(Vp):
        Vp = nsa_flow_retract_auto(Vp, w, retraction)
        if apply_nonneg:
            Vp = torch.clamp(Vp, min=0)
        e = 0.5 * fid_eta * torch.sum((Vp - X0) ** 2)
        norm2_V = torch.sum(Vp ** 2)
        if c_orth > 0 and norm2_V > 0:
            defect = defect_fast(Vp)
            e = e + 0.25 * c_orth * defect
        return e

    # --- Main loop ---
    for it in range(1, max_iter + 1):
        grad_fid = -fid_eta * (Y - X0)
        grad_orth, defect_val, _ = compute_ortho_terms(Y, c_orth)
        if c_orth > 0:
            sym_term_orth = symm(Y.T @ grad_orth)
            rgrad_orth = grad_orth - Y @ sym_term_orth
        else:
            rgrad_orth = grad_orth
        rgrad = grad_fid + rgrad_orth

        if torch.isnan(rgrad).any() or torch.isinf(rgrad).any():
            if verbose:
                print("NaN or Inf in gradient; stopping.")
            break

        # Simple gradient step (we can add optimizer class later)
        Y_new = Y - lr * rgrad
        Y_new = nsa_flow_retract_auto(Y_new, w, retraction)
        if apply_nonneg:
            Y_new = torch.clamp(Y_new, min=0)

        current_energy = nsa_energy(Y_new)

        # --- Backtracking line search ---
        alpha = 1.0
        count = 0
        max_bt = 20
        while current_energy > best_total_energy and count < max_bt:
            alpha *= 0.5
            Y_try = best_Y + alpha * (Y_new - best_Y)
            current_energy = nsa_energy(Y_try)
            count += 1
            if current_energy < best_total_energy:
                Y_new = nsa_flow_retract_auto(Y_try, w, retraction)
                break

        if count == max_bt:
            if verbose:
                print("Backtracking failed; reverting to best Y.")
            Y_new = best_Y.clone()

        if count > 2:
            lr *= 0.95
        elif count == 0 and it % 5 == 0:
            lr *= 1.01

        Y = Y_new

        fidelity = 0.5 * fid_eta * torch.sum((Y - X0) ** 2)
        orthogonality = defect_fast(Y)
        total_energy = fidelity + 0.25 * c_orth * orthogonality
        dt = time.time() - t0

        if total_energy < best_total_energy:
            best_total_energy = total_energy
            best_Y = Y.clone()
            best_Y_iteration = it

        recent_energies.append(total_energy.item())
        if len(recent_energies) > window_size:
            recent_energies.pop(0)

        if it % record_every == 0:
            traces.append({
                "iter": it,
                "time": dt,
                "fidelity": fidelity.item(),
                "orthogonality": orthogonality.item(),
                "total_energy": total_energy.item()
            })

        if verbose:
            print(f"[Iter {it:3d}] Total: {total_energy.item():.6e} | "
                  f"Fid: {fidelity.item():.6e} | Orth: {orthogonality.item():.6e}")

        # --- Convergence ---
        grad_norm = torch.norm(rgrad)
        rel_grad_norm = grad_norm / init_grad_norm
        if rel_grad_norm < tol:
            if verbose:
                print(f"Converged at iter {it} (grad norm < {tol:.2e})")
            break
        if len(recent_energies) == window_size:
            e_max, e_min = max(recent_energies), min(recent_energies)
            e_avg = sum(recent_energies) / len(recent_energies)
            rel_var = (e_max - e_min) / (abs(e_avg) + 1e-12)
            if rel_var < tol:
                if verbose:
                    print(f"Converged at iter {it} (energy stable < {tol:.2e})")
                break

    return {
        "Y": best_Y,
        "traces": traces,
        "final_iter": len(traces),
        "best_total_energy": best_total_energy.item(),
        "best_Y_iteration": best_Y_iteration,
        "target": X0
    }

