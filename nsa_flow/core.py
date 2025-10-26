import torch

def invariant_orthogonality_defect(Y):
    # Normalize columns to unit-norm then compute deviation from identity
    col_norms = torch.norm(Y, dim=0)
    safe = torch.clamp(col_norms, min=1e-12)
    Yn = Y / safe.unsqueeze(0)   # (p x k) with each column unit L2-norm
    YtY = Yn.T @ Yn
    I = torch.eye(YtY.shape[0], dtype=YtY.dtype, device=YtY.device)
    return torch.norm(YtY - I) / YtY.shape[0]

def inv_sqrt_sym_adaptive(YtY, eps=1e-8, ns_iter=1, method="diag"):
    """Compute (YtY + eps*I)^(-1/2) adaptively."""
    k = YtY.shape[0]
    # enforce symmetry
    YtY = 0.5 * (YtY + YtY.T)
    if method == "diag":
        d = torch.diag(YtY)
        # clamp to avoid small negative numerical noise
        d = torch.clamp(d, min=eps)
        d_inv_sqrt = torch.rsqrt(d + eps)
        return torch.diag(d_inv_sqrt)
    else:
        eigvals, eigvecs = torch.linalg.eigh(YtY + eps * torch.eye(k, device=YtY.device))
        eigvals_inv_sqrt = torch.rsqrt(torch.clamp(eigvals, min=eps))
        return eigvecs @ torch.diag(eigvals_inv_sqrt) @ eigvecs.T


import torch

def nsa_flow_retract_auto(Y_cand, w_retract=1.0,
                          retraction_type="soft_polar",
                          eps_rf=1e-8, verbose=False):
    """
    Weighted polar blend retraction.
    - returns (1-w)*Y + w*Q where Q = polar(Y) = U V^T.
    - preserves Frobenius norm via re-scaling (optional/preserving).
    """
    p, k = Y_cand.shape
    normY = torch.norm(Y_cand)
    if normY < 1e-12 or retraction_type == "none" or w_retract <= 0.0:
        return Y_cand.clone()

    # compute polar factor Q via thin SVD (stable)
    U, _, Vh = torch.linalg.svd(Y_cand, full_matrices=False)
    Q = U @ Vh  # shape (p, k) with Q^T Q = I_k (up to numeric error)

    # weighted polar blend in ambient space
    Y_blend = (1.0 - w_retract) * Y_cand + w_retract * Q

    # preserve Frobenius norm of original Y (optional; matches previous R behaviour)
    cur_norm = torch.norm(Y_blend)
    if cur_norm > 1e-12:
        Y_blend = Y_blend / cur_norm * normY

    return Y_blend


def soft_threshold(M, thresh):
    """Elementwise soft-thresholding (proximal for L1)."""
    # M: tensor
    return torch.sign(M) * torch.clamp(torch.abs(M) - thresh, min=0.0)


def energy_fidelity(M, Xc, w_pca):
    """Smooth fidelity energy used for autograd (no prox)."""
    n = Xc.shape[0]
    # negative sign to match original formulation (we minimize energy = -fid + prox)
    return -0.5 * w_pca * torch.sum((Xc @ M) ** 2) / n


def energy_total(M, Xc, w_pca, lambda_):
    """Full energy used for Armijo/backtracking (includes prox)."""
    return energy_fidelity(M, Xc, w_pca) + lambda_ * torch.sum(torch.abs(M))


def nsa_flow_py(Y, Xc, w_pca=1.0, lambda_=0.01, lr=1e-2, max_iter=100, retraction_type="soft_polar",
                armijo_beta=0.5, armijo_c=1e-4, tol=1e-6, verbose=False, max_grad_norm=1e6):
    """
    NSA-Flow with corrected Riemannian Armijo backtracking and proximal L1 handling.
    - Uses fidelity term for gradients (autograd).
    - Uses full energy (fidelity + L1 prox) for Armijo/backtracking checks.
    - Applies proximal soft-threshold AFTER retraction (matching R code).
    """
    # ensure tensors are float
    Y = Y.clone().detach().to(dtype=torch.get_default_dtype())
    Xc = Xc.clone().detach().to(dtype=torch.get_default_dtype())

    # make Y a leaf with grad
    Y.requires_grad_(True)

    p, k = Y.shape
    n = Xc.shape[0]

    # compute E_total_old for Armijo baseline
    Y_detached = Y.detach()
    E_total_old = energy_total(Y_detached, Xc, w_pca, lambda_).item()

    for t in range(max_iter):
        # zero accumulated gradients
        if Y.grad is not None:
            Y.grad.zero_()

        # compute smooth fidelity energy and gradient
        E_fid = energy_fidelity(Y, Xc, w_pca)
        E_fid.backward()
        grad = Y.grad.detach().clone()   # gradient w.r.t Y for smooth part
        # gradient clipping to avoid explosion
        grad_norm = torch.norm(grad)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at iteration {t}")
        if grad_norm > max_grad_norm:
            grad = grad * (max_grad_norm / grad_norm)
            grad_norm = torch.norm(grad)

        # search direction (Riemannian projection would be applied if orth term present)
        Y_dir = -grad

        # Backtracking Armijo: use total energy (with prox) for comparisons.
        step = lr
        E_old_total = energy_total(Y.detach(), Xc, w_pca, lambda_).item()
        # directional derivative (inner product grad * search dir), should be negative
        dir_deriv = torch.sum(grad * Y_dir).item()   # = -||grad||^2 (<=0)

        # backtracking loop
        bt_count = 0
        max_bt = 50
        accepted = False
        while bt_count < max_bt:
            # candidate: gradient step (explicit), then retraction, then proximal
            with torch.no_grad():
                Y_cand = Y + step * Y_dir
            Y_cand = nsa_flow_retract_auto(Y_cand, w_retract=1.0, retraction_type=retraction_type)
            # proximal soft-threshold with threshold = step * lambda_
            thresh = step * lambda_
            Y_cand_prox = soft_threshold(Y_cand, thresh)

            # compute total energy for Armijo test
            E_new_total = energy_total(Y_cand_prox, Xc, w_pca, lambda_).item()

            # Armijo condition: E_new <= E_old + c * step * dir_deriv
            rhs = E_old_total + armijo_c * step * dir_deriv
            if not (torch.isfinite(torch.tensor(E_new_total))):
                # shrink step and retry
                step *= armijo_beta
                bt_count += 1
                continue

            if E_new_total <= rhs:
                accepted = True
                break

            # otherwise shrink
            step *= armijo_beta
            bt_count += 1

        # Apply accepted candidate (or best fallback) to Y
        with torch.no_grad():
            if accepted:
                Y[:] = Y_cand_prox
            else:
                # if never accepted, apply the best candidate computed (safe fallback)
                Y[:] = Y_cand_prox

        # clear gradient for next iter
        if Y.grad is not None:
            Y.grad.zero_()

        # diagnostics
        E_now_total = energy_total(Y.detach(), Xc, w_pca, lambda_).item()
        if verbose and (t % 10 == 0 or t == 0):
            print(f"iter {t:3d} | E_total={E_now_total:.6e} | grad_norm={grad_norm:.3e} | step={step:.3e} | bt={bt_count}")

        # convergence
        if abs(E_old_total - E_now_total) < tol:
            if verbose:
                print(f"Converged at iter {t} (ΔE < {tol})")
            break

    return {"Y": Y.detach(), "energy": E_now_total, "iter": t + 1}
