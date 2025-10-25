import torch

def inv_sqrt_sym_adaptive(YtY, eps=1e-8, ns_iter=1, method="diag"):
    """Compute (YtY + eps*I)^(-1/2) adaptively."""
    k = YtY.shape[0]
    YtY = 0.5 * (YtY + YtY.T)
    if method == "diag":
        d = torch.diag(YtY)
        d_inv_sqrt = torch.rsqrt(d + eps)
        return torch.diag(d_inv_sqrt)
    else:
        eigvals, eigvecs = torch.linalg.eigh(YtY + eps * torch.eye(k, device=YtY.device))
        eigvals_inv_sqrt = torch.rsqrt(torch.clamp(eigvals, min=eps))
        return eigvecs @ torch.diag(eigvals_inv_sqrt) @ eigvecs.T


def nsa_flow_retract_auto(Y_cand, w_retract=1.0, retraction_type="soft_polar",
                          eps_rf=1e-8, inv_method="diag", ns_iter=1,
                          eig_thresh=128, diag_thresh=8192, verbose=False):
    """PyTorch version of nsa_flow_retract_auto."""
    p, k = Y_cand.shape
    normY = torch.norm(Y_cand)

    if retraction_type == "none":
        return Y_cand

    if retraction_type == "polar":
        U, _, Vt = torch.linalg.svd(Y_cand, full_matrices=False)
        Ytilde = U @ Vt
        cur_norm = torch.norm(Ytilde)
        if cur_norm > 1e-12:
            Ytilde = Ytilde / cur_norm * normY
        return Ytilde

    if retraction_type == "soft_polar":
        use_svd_fallback = k > diag_thresh and p < k
        if not use_svd_fallback:
            YtY = Y_cand.T @ Y_cand
            T = inv_sqrt_sym_adaptive(YtY, eps=eps_rf, method=inv_method, ns_iter=ns_iter)
            T_w = (1 - w_retract) * torch.eye(k, device=Y_cand.device) + w_retract * T
            Ytilde = Y_cand @ T_w
        else:
            U, _, Vt = torch.linalg.svd(Y_cand, full_matrices=False)
            Q = U @ Vt
            Ytilde = (1 - w_retract) * Y_cand + w_retract * Q

        cur_norm = torch.norm(Ytilde)
        if cur_norm > 1e-12:
            Ytilde = Ytilde / cur_norm * normY
        return Ytilde

    raise ValueError("Unsupported retraction_type in nsa_flow_retract_auto()")


def nsa_flow_py(Y, Xc, w_pca=1.0, lambda_=0.01, lr=1e-2, max_iter=100, retraction_type="soft_polar",
                armijo_beta=0.5, armijo_c=1e-4, tol=1e-6, verbose=False):
    """NSA-Flow with Riemannian Armijo line search and adaptive retraction."""
    Y = Y.clone().requires_grad_(True)
    p, k = Y.shape
    n = Xc.shape[0]

    def energy(M):
        fid = -0.5 * w_pca * torch.sum((Xc @ M) ** 2) / n
        prox = lambda_ * torch.sum(torch.abs(M))
        return fid + prox

    E_prev = energy(Y).item()

    for t in range(max_iter):
        E_prev = energy(Y)
        E_prev.backward()

        grad = Y.grad
        Y_dir = -grad

        # Riemannian Armijo: evaluate along retracted path
        step = lr
        while True:
            Y_cand = Y + step * Y_dir
            Y_cand = nsa_flow_retract_auto(Y_cand, retraction_type=retraction_type)
            E_new = energy(Y_cand)
            rhs = E_prev + armijo_c * step * torch.sum(grad * Y_dir)
            if E_new <= rhs:
                break
            step *= armijo_beta
            if step < 1e-12:
                break

        with torch.no_grad():
            Y[:] = nsa_flow_retract_auto(Y + step * Y_dir, retraction_type=retraction_type)
        Y.grad.zero_()

        E_now = energy(Y).item()
        if verbose and t % 10 == 0:
            print(f"iter {t}: energy={E_now:.6f} step={step:.3e}")
        if abs(E_prev - E_now) < tol:
            break

    return {"Y": Y.detach(), "energy": E_now, "iter": t + 1}
