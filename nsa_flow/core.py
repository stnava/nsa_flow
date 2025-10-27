import os
import copy
import math
import time
import torch
import torch_optimizer
from typing import Optional
import warnings
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import json
import hashlib
from pathlib import Path
import numpy as np

def traces_to_dataframe(traces):
    """
    Convert list of trace dictionaries into a clean pandas DataFrame.
    """
    clean = []
    for t in traces:
        clean.append({
            "iter": t["iter"],
            "time": float(t["time"]),
            "fidelity": float(t["fidelity"].detach().cpu().item() if isinstance(t["fidelity"], torch.Tensor) else t["fidelity"]),
            "orthogonality": float(t["orthogonality"]),
            "total_energy": float(t["total_energy"].detach().cpu().item() if isinstance(t["total_energy"], torch.Tensor) else t["total_energy"]),
        })
    return pd.DataFrame(clean)



def plot_nsa_trace(trace_df, retraction="soft_polar", figsize=(8,5)):
    fig, ax1 = plt.subplots(figsize=figsize)

    color_fid = "#1f78b4"      # Fidelity (blue)
    color_orth = "#33a02c"     # Orthogonality (green)

    # Compute ratio for scaling the secondary y-axis
    max_fid = trace_df["fidelity"].max()
    max_orth = trace_df["orthogonality"].max()
    ratio = max_fid / max_orth if max_orth > 0 else 1.0

    # Left axis: fidelity
    ax1.plot(trace_df["iter"], trace_df["fidelity"], color=color_fid, label="Fidelity", linewidth=2)
    ax1.scatter(trace_df["iter"], trace_df["fidelity"], color=color_fid, s=20, alpha=0.7)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fidelity Energy", color=color_fid)
    ax1.tick_params(axis='y', labelcolor=color_fid)

    # Right axis: orthogonality (scaled)
    ax2 = ax1.twinx()
    ax2.plot(trace_df["iter"], trace_df["orthogonality"], color=color_orth, label="Orthogonality", linewidth=2)
    ax2.scatter(trace_df["iter"], trace_df["orthogonality"], color=color_orth, s=20, alpha=0.7)
    ax2.set_ylabel("Orthogonality Defect", color=color_orth)
    ax2.tick_params(axis='y', labelcolor=color_orth)

    # Title and grid
    plt.title(f"NSA-Flow Optimization Trace: {retraction}\nFidelity and Orthogonality (Dual Scales)", fontsize=13, weight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()

def apply_nonnegativity(Y, mode="softplus"):
    """
    Apply nonnegativity transformation to a tensor.

    Parameters
    ----------
    Y : torch.Tensor
        Input tensor.
    mode : str or bool
        Nonnegativity mode:
        - 'none' or False : no constraint
        - 'softplus'      : smooth differentiable mapping (soft nonnegativity)
        - True or 'hard'  : hard projection via clamping

    Returns
    -------
    torch.Tensor
        Transformed tensor according to the specified mode.
    """
    if mode in [False, "none", None]:
        return Y
    elif mode == "softplus":
        return F.softplus(Y)
    elif mode in [True, "hard", "Hard"]:
        return torch.clamp(Y, min=0.0)
    else:
        raise ValueError(f"Invalid apply_nonneg mode: {mode}. "
                         "Use 'none', 'softplus', or True/'hard'.")


def get_torch_optimizer(opt_name: str = None, params=None, lr: float = 1e-3, return_list: bool = False, **kwargs):
    """
    Returns a PyTorch optimizer instance based on the provided name,
    or lists all available optimizers if return_list=True.

    Supports optimizers from both torch.optim and (optionally) torch_optimizer.

    Args:
        opt_name (str, optional): Name of the optimizer (case-insensitive).
        params: Model parameters to optimize.
        lr (float): Learning rate.
        return_list (bool): If True, return a list of available optimizer names.
        **kwargs: Additional arguments to pass to the optimizer.

    Returns:
        optimizer (torch.optim.Optimizer) or list[str]:
            - If return_list=False (default): an optimizer instance.
            - If return_list=True: sorted list of available optimizer names.
    """
    # --- Base PyTorch optimizers ---
    optimizers = {
        "adam": lambda p, lr: torch.optim.Adam(p, lr=lr, **kwargs),
        "adamw": lambda p, lr: torch.optim.AdamW(p, lr=lr, **kwargs),
        "sgd": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9, **kwargs),
        "sgd_nesterov": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9, nesterov=True, **kwargs),
        "rmsprop": lambda p, lr: torch.optim.RMSprop(p, lr=lr, **kwargs),
        "adagrad": lambda p, lr: torch.optim.Adagrad(p, lr=lr, **kwargs),
        "lbfgs": lambda p, lr: torch.optim.LBFGS(p, lr=lr, max_iter=10, **kwargs),
        "adamax": lambda p, lr: torch.optim.Adamax(p, lr=lr, **kwargs),
        "asgd": lambda p, lr: torch.optim.ASGD(p, lr=lr, **kwargs),
        "nadam": lambda p, lr: torch.optim.NAdam(p, lr=lr, **kwargs),
        "radam": lambda p, lr: torch.optim.RAdam(p, lr=lr, **kwargs),
        "rprop": lambda p, lr: torch.optim.Rprop(p, lr=lr, **kwargs),
    }

    # --- Optionally extend with torch_optimizer if available ---
    try:
        import torch_optimizer as optimx

        extra_opts = {
            "lars": lambda p, lr: optimx.LARS(p, lr=lr, **kwargs),
            "yogi": lambda p, lr: optimx.Yogi(p, lr=lr, **kwargs),
            "adabound": lambda p, lr: optimx.AdaBound(p, lr=lr, **kwargs),
            "qhadam": lambda p, lr: optimx.QHAdam(p, lr=lr, **kwargs),
            "pid": lambda p, lr: optimx.PID(p, lr=lr, **kwargs),
            "qhm": lambda p, lr: optimx.QHM(p, lr=lr, **kwargs),
            "adamp": lambda p, lr: optimx.AdamP(p, lr=lr, **kwargs),
            "sgdp": lambda p, lr: optimx.SGDP(p, lr=lr, **kwargs),
            "padam": lambda p, lr: optimx.PAdam(p, lr=lr, **kwargs),
            "lookahead": lambda p, lr: optimx.Lookahead(torch.optim.Adam(p, lr=lr, **kwargs)),
        }
        optimizers.update(extra_opts)

    except ImportError:
        warnings.warn("torch_optimizer not installed. Skipping extra optimizers.", UserWarning)

    # --- Return the list of available optimizers if requested ---
    if return_list:
        return sorted(optimizers.keys())

    # --- Otherwise, construct the requested optimizer ---
    if opt_name is None:
        raise ValueError("You must specify `opt_name` unless `return_list=True`.")

    name = opt_name.lower()
    if name not in optimizers:
        raise ValueError(
            f"Unsupported optimizer: '{opt_name}'. "
            f"Supported options: {sorted(optimizers.keys())}"
        )

    return optimizers[name](params, lr)



def estimate_learning_rate(
    objective_fn,
    params_init,
    optimizer_class,
    lr_min=1e-6,
    lr_max=1e2,
    num_steps=150,
    strategy="exponential",
    smoothing=0.9,
    stop_factor=10,
    device=None,
    plot=True,
):
    """
    Empirically estimates an optimal learning rate by exploring the loss landscape
    across a wide LR range (default 1e-6 → 1e2).

    Strategies:
        - 'exponential' : LR increases exponentially each step (Leslie Smith-style)
        - 'linear'      : LR increases linearly
        - 'random'      : Samples LRs log-uniformly from [lr_min, lr_max]
        - 'adaptive'    : Explores adaptively, slowing down where loss decreases fastest

    Args:
        objective_fn (callable): Function taking params (Tensor) -> scalar loss (Tensor).
        params_init (torch.Tensor): Initial parameters, requires_grad=True.
        optimizer_class (callable): e.g. lambda p: torch.optim.Adam(p, lr=1e-3)
        lr_min (float): Minimum learning rate.
        lr_max (float): Maximum learning rate.
        num_steps (int): Number of test steps.
        strategy (str): One of {'exponential', 'linear', 'random', 'adaptive'}.
        smoothing (float): EWMA factor for smoothed loss (for 'exponential' mode).
        stop_factor (float): Stop if loss > stop_factor × best_loss so far.
        device (torch.device or str): Target device.
        plot (bool): Whether to plot LR vs loss (log-log).

    Returns:
        dict with:
            'lr_curve' : list of (lr, loss)
            'best_lr'  : float (suggested LR)
            'losses'   : list of float
            'strategy' : strategy used
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Prepare ---
    params = params_init.clone().detach().to(device).requires_grad_(True)
    optimizer = optimizer_class([params])

    # --- Generate LR schedule ---
    if strategy == "exponential":
        lrs = torch.logspace(math.log10(lr_min), math.log10(lr_max), num_steps)
    elif strategy == "linear":
        lrs = torch.linspace(lr_min, lr_max, num_steps)
    elif strategy == "random":
        lrs = torch.pow(10, torch.rand(num_steps) * (math.log10(lr_max) - math.log10(lr_min)) + math.log10(lr_min))
    elif strategy == "adaptive":
        # Start exponential, adapt step size dynamically
        lrs = [lr_min]
        growth = (lr_max / lr_min) ** (1.0 / num_steps)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from exponential, linear, random, adaptive.")

    losses = []
    smoothed = None
    best_loss = float("inf")
    best_lr = lr_min
    prev_loss = None

    for i in range(num_steps):
        if strategy == "adaptive" and i > 0:
            lrs.append(min(lrs[-1] * growth, lr_max))
        lr = lrs[i] if strategy != "adaptive" else lrs[-1]

        for g in optimizer.param_groups:
            g["lr"] = lr.item() if torch.is_tensor(lr) else lr

        optimizer.zero_grad(set_to_none=True)
        loss = objective_fn(params)

        if not torch.isfinite(loss):
            print(f"Non-finite loss at step {i}, stopping.")
            break

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if smoothed is None:
            smoothed = loss_val
        else:
            smoothed = smoothing * smoothed + (1 - smoothing) * loss_val

        losses.append(smoothed)

        if smoothed < best_loss:
            best_loss = smoothed
            best_lr = lr.item() if torch.is_tensor(lr) else lr

        if prev_loss is not None and smoothed > stop_factor * best_loss:
            print(f"Stopping early: loss exploded by >{stop_factor}× at step {i}.")
            break
        prev_loss = smoothed

    if not torch.is_tensor(lrs):
        lrs = torch.tensor(lrs[:len(losses)], dtype=torch.float64)
    else:
        lrs = lrs[:len(losses)].detach().clone().to(dtype=torch.float64)

    # --- Plot ---
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(lrs.cpu(), losses, marker='o', alpha=0.8)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Smoothed Loss")
        plt.title(f"Learning Rate Finder ({strategy.capitalize()} Sweep)")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.axvline(best_lr, color="red", linestyle="--", label=f"Suggested LR = {best_lr:.2e}")
        plt.legend()
        plt.show()

    return {
        "lr_curve": list(zip(lrs.cpu().numpy(), losses)),
        "best_lr": best_lr,
        "losses": losses,
        "strategy": strategy,
    }


def estimate_learning_rate_for_nsa_flow(
    Y0,
    X0=None,
    w=0.5,
    retraction="soft_polar",
    optimizer="Adam",
    fid_eta=None,
    c_orth=None,
    apply_nonneg=True,
    strategy="armijo",
    max_steps=20,
    plot=False,
    verbose=False,
    device=None,
):
    """
    Unified learning-rate estimator for NSA-Flow.
    All strategies are scale-invariant and normalize energy terms.
    """

    import torch
    import numpy as np

    torch.set_default_dtype(torch.float64)
    device = device or Y0.device
    Y0 = Y0.clone().detach().to(device)
    if X0 is None:
        X0 = torch.clamp(Y0, min=0) if apply_nonneg else Y0.clone()
    X0 = X0.to(device)

    p, k = Y0.shape
    # --- default scalings if missing ---
    if fid_eta is None or c_orth is None:
        g0 = 0.5 * torch.sum((Y0 - X0) ** 2) / (p * k)
        g0 = torch.clamp(g0, min=1e-8)
        d0 = torch.clamp(defect_fast(Y0), min=1e-8)
        if fid_eta is None:
            fid_eta = (1 - w) / (g0 * p * k)
        if c_orth is None:
            c_orth = 4 * w / d0

    # --- energy function (normalized) ---
    def energy_fn(Y):
        Yr = nsa_flow_retract_auto(Y, w, retraction)
        Yr = apply_nonnegativity(Yr, apply_nonneg)
        e_norm = 0.5 * (torch.sum(X0**2) + torch.sum(Yr**2))
        fidelity = 0.5 * fid_eta * torch.sum((Yr - X0) ** 2) / (e_norm + 1e-12)
        orth = 0.25 * c_orth * defect_fast(Yr)
        return (fidelity + orth)

    # --- compute reference gradient direction ---
    Y_ref = Y0.clone().detach().requires_grad_(True)
    f_ref = energy_fn(Y_ref)
    f_ref.backward()
    grad = Y_ref.grad.detach()
    grad_norm = torch.norm(grad) + 1e-12
    grad_dir = grad / grad_norm
    f0 = f_ref.item()

    # candidate learning rates to probe
    lr_candidates = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0    ])
    lr_vals, losses = [], []

    # --- unified probe loop ---
    for lr in lr_candidates:
        with torch.no_grad():
            Y_try = Y_ref - lr * grad_dir
        f_new = energy_fn(Y_try).item()
        lr_vals.append(lr)
        losses.append(f_new)

    losses = np.array(losses)
    if np.any(np.isnan(losses)) or np.any(np.isinf(losses)):
        losses = np.nan_to_num(losses, nan=np.inf)

    # --- strategy logic ---
    strategy = strategy.lower()
    if strategy.startswith("armijo"):
        c = 1e-4 if "aggressive" not in strategy else 5e-3
        good_lrs = [
            lr for lr, f_new in zip(lr_vals, losses)
            if f_new <= f0 - c * lr * grad_norm**2
        ]
        best_lr = float(np.quantile(good_lrs, 0.95)) if good_lrs else float(lr_candidates[0])

    elif strategy == "grid":
        best_lr = float(lr_candidates[np.argmin(losses)])

    elif strategy == "adaptive":
        diffs = np.diff(losses)
        if np.all(diffs >= 0):
            best_lr = float(lr_candidates[0])
        elif np.all(diffs <= 0):
            best_lr = float(lr_candidates[-1])
        else:
            idx = np.where(diffs > 0)[0]
            best_lr = float(lr_candidates[max(idx[0] - 1, 0)]) if len(idx) else float(lr_candidates[-1])

    elif strategy == "exploratory":
        rel_change = (losses[0] - losses) / (abs(f0) + 1e-12)
        threshold = 0.05
        good = lr_candidates[rel_change > threshold]
        best_lr = float(np.quantile(good, 0.95)) if len(good) else float(lr_candidates[0])

    else:
        raise ValueError(f"Unknown LR strategy: {strategy}")

    if verbose:
        print(f"[LR-Est] {strategy:<20} best_lr={best_lr:.2e}")

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(lr_vals, losses, "o-", label=strategy)
        plt.axvline(best_lr, color="r", linestyle="--")
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Normalized energy")
        plt.title(f"NSA-Flow LR Search ({strategy})")
        plt.legend()
        plt.show()

    return {
        "lr_candidates": lr_vals,
        "losses": losses,
        "best_lr": best_lr,
        "strategy": strategy,
    }


def invariant_orthogonality_defect(V):
        norm2 = torch.sum(V ** 2)
        if norm2 <= 1e-12:
            return torch.tensor(0.0, device=V.device)
        S = V.T @ V
        diagS = torch.diag(S)
        off_f2 = torch.sum(S * S) - torch.sum(diagS ** 2)
        return off_f2 / (norm2 ** 2)

def defect_fast(V):
        return invariant_orthogonality_defect(V)

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




def nsa_flow_old(Y0, X0=None, w=0.5,
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

    # --- optimizer setup ---
    if optimizer == "fast":
        opt = None  # use manual gradient step
    else:
        Y.requires_grad_(True)
        opt = get_torch_optimizer(optimizer, [Y], lr)

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

    zz = len(traces)
    traces = traces_to_dataframe(traces)
    return {
        "Y": best_Y,
        "traces": traces,
        "final_iter": zz,
        "best_total_energy": best_total_energy.item(),
        "best_Y_iteration": best_Y_iteration,
        "target": X0
    }

def nsa_flow(Y0, X0=None, w=0.5,
             retraction="soft_polar",
             max_iter=500, tol=1e-5, verbose=False, seed=42,
             apply_nonneg=True, optimizer="fast",
             initial_learning_rate="default",
             record_every=1, window_size=5, c1_armijo=1e-6,
             simplified=False,
             project_full_gradient=False,
             device=None, precision='float64'):
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
        Optimizer to use ('fast' for simple gradient step, or PyTorch optimizers like 'Adam').
    initial_learning_rate : float or str
        Learning rate or 'default' for auto-estimate.
    record_every : int
        Iteration interval for recording traces.
    window_size : int
        Energy stability window for convergence check.
    simplified : bool
        Use simplified orthogonality objective.
    project_full_gradient : bool
        If True, project the full gradient (fidelity + orthogonality) onto the tangent space.
        If False (default), only project the orthogonality gradient, keeping fidelity as Euclidean.
    device : torch.device or str or None
        Device to run on.
    precision : str
        Floating point precision ('float32', 'float64'). Default: 'float64' for stability.

    Returns
    -------
    dict with keys:
        'Y', 'traces', 'final_iter', 'best_total_energy', 'best_Y_iteration'
    """

    if precision == 'float32':
        dtype = torch.float32
    elif precision == 'float64':
        dtype = torch.float64
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 'float32' or 'float64'.")
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
            grad_orth = -2 * c_orth * (Y @ (S - torch.diag(diagS)))
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
    if project_full_gradient:
        full_grad_init = grad_fid_init + grad_orth_init
        sym_term_init = symm(Y.T @ full_grad_init)
        rgrad_init = full_grad_init - Y @ sym_term_init
    else:
        rgrad_fid_init = grad_fid_init
        if c_orth > 0:
            sym_term_orth_init = symm(Y.T @ grad_orth_init)
            rgrad_orth_init = grad_orth_init - Y @ sym_term_orth_init
        else:
            rgrad_orth_init = grad_orth_init
        rgrad_init = rgrad_fid_init + rgrad_orth_init
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

    # --- optimizer setup ---
    if optimizer == "fast":
        opt = None  # use manual gradient step
    else:
        Y.requires_grad_(True)
        opt = get_torch_optimizer(optimizer, [Y], lr)

    # --- Main loop ---
    for it in range(1, max_iter + 1):
        grad_fid = -fid_eta * (Y - X0)
        grad_orth, defect_val, _ = compute_ortho_terms(Y, c_orth)
        if project_full_gradient:
            full_grad = grad_fid + grad_orth
            sym_term = symm(Y.T @ full_grad)
            rgrad = full_grad - Y @ sym_term
        else:
            rgrad_fid = grad_fid
            if c_orth > 0:
                sym_term_orth = symm(Y.T @ grad_orth)
                rgrad_orth = grad_orth - Y @ sym_term_orth
            else:
                rgrad_orth = grad_orth
            rgrad = rgrad_fid + rgrad_orth

        if torch.isnan(rgrad).any() or torch.isinf(rgrad).any():
            if verbose:
                print("NaN or Inf in gradient; stopping.")
            break

        # Compute proposed update before retraction
        if optimizer == "fast":
            Y_proposed = Y - lr * rgrad  # Reverted to original sign convention
        else:
            Y_old = Y.clone().detach()
            opt.zero_grad()
            Y.grad = rgrad  # Set to positive gradient for optimizer to subtract
            opt.step()
            Y_proposed = Y.clone().detach()
            Y.data = Y_old.data  # Reset Y to old value (optimizer state remains updated)

        Y_new = nsa_flow_retract_auto(Y_proposed, w, retraction)

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

        # Apply nonnegativity after backtracking (to ensure consistency)
        if apply_nonneg:
            Y_new = torch.clamp(Y_new, min=0)

        if count > 2:
            if optimizer == "fast":
                lr *= 0.95
            else:
                opt.param_groups[0]['lr'] *= 0.95
        elif count == 0 and it % 5 == 0:
            if optimizer == "fast":
                lr *= 1.01
            else:
                opt.param_groups[0]['lr'] *= 1.01

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

    zz = len(traces)
    traces = traces_to_dataframe(traces)
    return {
        "Y": best_Y,
        "traces": traces,
        "final_iter": zz,
        "best_total_energy": best_total_energy.item(),
        "best_Y_iteration": best_Y_iteration,
        "target": X0
    }



def nsa_flow_autograd(
    Y0, X0=None, w=0.5,
    retraction="soft_polar",
    max_iter=500, tol=1e-6, verbose=False, seed=42,
    apply_nonneg=True, optimizer="Adam",
    initial_learning_rate=None, 
    lr_strategy="auto",
    record_every=1, window_size=5,
    simplified=False, project_full_gradient=False,
    device=None, precision="float64"
):
    """
    Autograd-compatible NSA-Flow (scale-invariant version).
    """

    if precision == "float32":
        dtype = torch.float32
    elif precision == "float64":
        dtype = torch.float64
    else:
        raise ValueError("precision must be 'float32' or 'float64'")

    torch.manual_seed(seed)
    if device is None:
        device = Y0.device

    # --- Initialize and scale normalize ---
    Y = Y0.clone().detach().to(device).to(dtype)
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
        X0 = X0.to(device).to(dtype)
        if apply_nonneg:
            X0 = torch.clamp(X0, min=0)
        assert X0.shape == Y.shape, "X0 must match Y0 shape."

    # --- Normalize for scale invariance ---
    scale_ref = torch.sqrt(torch.sum(X0 ** 2) / X0.numel()).item() + 1e-12
    X0 = X0 / scale_ref
    Y = Y / scale_ref

    # --- Energy scaling ---
    g0 = 0.5 * torch.sum((Y - X0) ** 2) / (p * k)
    g0 = torch.clamp(g0, min=1e-8)
    d0 = torch.clamp(defect_fast(Y), min=1e-8)
    fid_eta = (1 - w) / (g0 * p * k)
    c_orth = 4 * w / d0

    # --- Optimization variable ---
    Z = torch.nn.Parameter(Y.clone().detach().requires_grad_(True))

    # --- Optional LR auto-tuning ---
    if initial_learning_rate is None and isinstance(lr_strategy, str) and lr_strategy.lower() in ["auto", "exponential", "linear", "random", "adaptive"]:
        if verbose:
            print(f"[NSA-Flow] Estimating learning rate ({lr_strategy}) ...")
        lr_result = estimate_learning_rate_for_nsa_flow(
            Y0=torch.randn_like(X0),
            X0=X0,
            w=w,
            retraction=retraction,
            optimizer=optimizer,
            fid_eta=fid_eta,
            c_orth=c_orth,
            apply_nonneg=apply_nonneg,
            strategy="armijo_aggressive" if lr_strategy == "auto" else lr_strategy,
            plot=False,
            verbose=verbose,
        )
        lr = lr_result["best_lr"]
        if verbose:
            print(f"[NSA-Flow] Selected learning rate: {lr:.2e}")
        opt = get_torch_optimizer(optimizer, [Z], lr=lr)
    else:
        opt = get_torch_optimizer(optimizer, [Z], lr=initial_learning_rate or 1e-3)

    traces = []
    recent_energies = []
    t0 = time.time()
    best_Y = None
    best_energy = float("inf")
    best_iter = 0

    for it in range(1, max_iter + 1):
        opt.zero_grad()

        # --- Retraction and nonnegativity ---
        Y_retracted = nsa_flow_retract_auto(Z, w, retraction)
        Y_retracted = apply_nonnegativity(Y_retracted, apply_nonneg)

        # --- Energy computation (scale-invariant) ---
        e_norm = 0.5 * (torch.sum(X0 ** 2).item() + torch.sum(Y_retracted ** 2).item())
        fidelity = (0.5 * fid_eta * torch.sum((Y_retracted - X0) ** 2)) / (e_norm + 1e-12)
        orth_term = 0.25 * c_orth * defect_fast(Y_retracted)
        total_energy = fidelity + orth_term

        total_energy.backward()
        opt.step()

        # --- Evaluation and tracking ---
        with torch.no_grad():
            Y_eval = nsa_flow_retract_auto(Z, w, retraction)
            Y_eval = apply_nonnegativity(Y_eval, apply_nonneg)
            e_norm = 0.5 * (torch.sum(X0 ** 2).item() + torch.sum(Y_eval ** 2).item())
            fidelity = (0.5 * fid_eta * torch.sum((Y_eval - X0) ** 2).item()) / (e_norm + 1e-12)
            orth_val = defect_fast(Y_eval).item()
            total_val = fidelity + 0.25 * c_orth * orth_val

            if total_val < best_energy:
                best_energy = total_val
                best_Y = Y_eval.clone()
                best_iter = it

            if it % record_every == 0:
                traces.append({
                    "iter": it,
                    "time": time.time() - t0,
                    "fidelity": fidelity,
                    "orthogonality": orth_val,
                    "total_energy": total_val
                })

            recent_energies.append(total_val)
            if len(recent_energies) > window_size:
                recent_energies.pop(0)

            if verbose and (it % record_every == 0 or it < 10):
                print(f"[Iter {it:3d}] Total={total_val:.6e} | Fid={fidelity:.6e} | Orth={orth_val:.6e}")

            # --- Convergence criterion ---
            if len(recent_energies) == window_size:
                e_max, e_min = max(recent_energies), min(recent_energies)
                e_avg = sum(recent_energies) / len(recent_energies)
                rel_var = (e_max - e_min) / (abs(e_avg) + 1e-12)
                if rel_var < tol:
                    if verbose:
                        print(f"Converged at iter {it} (energy stable < {tol:.2e})")
                    break

    traces = traces_to_dataframe(traces)
    return {
        "Y": best_Y * scale_ref,  # rescale back to original magnitude
        "traces": traces,
        "final_iter": best_iter,
        "best_total_energy": best_energy,
        "best_Y_iteration": best_iter,
        "target": X0 * scale_ref,
    }



def test_scale_invariance():
    torch.manual_seed(0)
    p, k = 30, 5
    X = torch.randn(p, k)
    Y = X + 0.1 * torch.randn(p, k)

    print("Testing scale invariance...")
    lr1 = estimate_learning_rate_for_nsa_flow(Y, X, w=0.5)["best_lr"]
    lr2 = estimate_learning_rate_for_nsa_flow(10 * Y, 10 * X, w=0.5)["best_lr"]

    print(f"Learning rate (original): {lr1:.3e}")
    print(f"Learning rate (scaled ×10): {lr2:.3e}")
    print(f"Δlog10(lr) = {abs(np.log10(lr1) - np.log10(lr2)):.3e}")

    assert abs(np.log10(lr1) - np.log10(lr2)) < 0.1, "Learning rate should be approximately scale invariant"

    res1 = nsa_flow_autograd(Y, X, max_iter=50, verbose=False)
    res2 = nsa_flow_autograd(10 * Y, 10 * X, max_iter=50, verbose=False)

    diff = torch.norm(res1["Y"] / torch.norm(res1["Y"]) - res2["Y"] / torch.norm(res2["Y"]))
    print(f"Normalized output difference: {diff:.3e}")
    assert diff < 0.05, "NSA-Flow output should be invariant up to scale"

    print("✅ Scale invariance test passed successfully!")


def return_lr_estimation_strategies():
    """
    Return a list of supported learning rate estimation strategies
    for NSA-Flow's estimate_learning_rate_for_nsa_flow function.

    Returns:
        List[str]: Supported strategies.
    """
    strategies = [
        "armijo",             # conservative Armijo backtracking
        "armijo_aggressive",  # more aggressive Armijo search
        "grid",               # fixed grid search
        "adaptive",           # adaptive exponential growth
        "exploratory",        # broad exploratory search
        "exponential",        # log-spaced steps
        "linear",             # linearly spaced steps
        "random"              # random log-space sampling
    ]
    return strategies

def test_lr_strategies():
    torch.manual_seed(0)
    X = torch.randn(30, 5)
    Y = X + 0.1 * torch.randn(30, 5)

    for strat in return_lr_estimation_strategies():
        res = estimate_learning_rate_for_nsa_flow(Y, X, strategy=strat, plot=False)
        print(f"{strat:>18} → best_lr={res['best_lr']:.2e}")    

def test_autograd_scale_invariance( lr_strategy="auto" ):
    """
    Test: NSA-Flow autograd version should be invariant to scaling of X and Y.

    This runs NSA-Flow with a scaled input and verifies that the invariant
    orthogonality defect and energy evolution are stable. Also visualizes
    the learning trace for manual inspection.
    """
    import torch
    import matplotlib.pyplot as plt
    from nsa_flow import (
        nsa_flow_autograd,
        invariant_orthogonality_defect,
        plot_nsa_trace,
    )

    torch.manual_seed(42)
    Y = torch.randn(50, 10)
    X0 = torch.randn_like(Y)
    retraction = "soft_polar"
    optimizer = "lars"

    print("[TEST] Running NSA-Flow autograd with scaled inputs (×1e4)...")

    result = nsa_flow_autograd(
        Y * 1e4,            # scaled input
        X0=X0 * 1e4,        # ensure target matches scale
        w=0.5,
        retraction=retraction,
        optimizer=optimizer,
        max_iter=500,
        record_every=1,
        tol=1e-8,
        initial_learning_rate=None,
        lr_strategy=lr_strategy,
        apply_nonneg=True,
        verbose=True,
    )

    Y_final = result["Y"]
    inv_def_init = invariant_orthogonality_defect(Y)
    inv_def_final = invariant_orthogonality_defect(Y_final)

    print(f"[TEST] Invariant orthogonality defect (init):  {inv_def_init:.3e}")
    print(f"[TEST] Invariant orthogonality defect (final): {inv_def_final:.3e}")

    # --- optional sanity check ---
    ratio = inv_def_final / (inv_def_init + 1e-12)
    print(f"[TEST] Relative change in invariant defect: {ratio:.3f}")
    assert torch.isfinite(torch.tensor(inv_def_final)), "NaNs in result!"
    assert ratio < 10, "Invariant defect exploded — possible scale issue!"

    # --- plot learning trace ---
    plot_nsa_trace(result["traces"])

    print("[TEST] ✓ Scale invariance check complete.")
    return result