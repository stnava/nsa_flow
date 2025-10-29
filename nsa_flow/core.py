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
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def invariant_orthogonality_defect(V):
    """Scale-invariant orthogonality defect."""
    norm2 = torch.sum(V ** 2)
    if norm2 <= 1e-12:
        return torch.tensor(0.0, device=V.device, dtype=V.dtype)
    S = V.T @ V
    diagS = torch.diag(S)
    off_f2 = torch.sum(S * S) - torch.sum(diagS ** 2)
    return off_f2 / (norm2 ** 2)


def defect_fast(V):
    return invariant_orthogonality_defect(V)

# ----------------------------
# Fidelity term functions
# ----------------------------
def fidelity_basic(Y, X):
    """||Y - X||²"""
    return 0.5 * torch.sum((Y - X) ** 2)

def fidelity_scaled(Y, X):
    """||Y - X||² / ||X||²"""
    denom = torch.sum(X ** 2).clamp_min(1e-12)
    return 0.5 * torch.sum((Y - X) ** 2) / denom

def fidelity_symmetric(Y, X):
    """||Y - X||² / (||X||² + ||Y||²)"""
    denom = 0.5 * (torch.sum(X ** 2) + torch.sum(Y ** 2)).clamp_min(1e-12)
    return 0.5 * torch.sum((Y - X) ** 2) / denom


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



def compute_energy(
    Y, X0, w=0.5,
    fidelity_type="scale_invariant",   # can be "basic", "scale_invariant", or "symmetric"
    orth_type="scale_invariant",       # can be "basic" or "scale_invariant"
    fid_eta=1.0,
    c_orth=1.0,
    track_grad=True,
    return_dict=False,
):
    """
    Centralized energy computation for NSA-Flow.
    """

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
        orth = 0.25 * c_orth * torch.norm(Y.T @ Y - torch.eye(Y.shape[1], device=Y.device))**2
    elif orth_type == "scale_invariant":
        orth = 0.25 * c_orth * defect_fast(Y) 
    else:
        raise ValueError(f"Unknown orth_type: {orth_type}")

    total = fidelity + orth

    if not track_grad:
        fidelity = fidelity.detach().item()
        orth = orth.detach().item()
        total = total.detach().item()

    if return_dict:
        return {
            "fidelity": fidelity,
            "orthogonality": orth,
            "total": total,
        }

    return total

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
#        "lbfgs": lambda p, lr: torch.optim.LBFGS(p, lr=lr, max_iter=10, **kwargs),
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

def estimate_learning_rate_for_nsa_flow(
    Y0,
    X0,
    w=0.5,
    retraction="soft_polar",
    fid_eta=None,
    c_orth=None,
    apply_nonneg=True,
    strategy="armijo",
    aggression=0.5,
    verbose=False,
    plot=False,
    fidelity_type="scale_invariant",
    orth_type="scale_invariant",
):
    """
    Estimate a suitable learning rate for NSA-Flow optimization with 10 strategy modes
    and aggression-based scaling. Robust to NaN/Inf energies.
    """

    import torch, numpy as np, warnings

    def _to_float(x):
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.numel() == 1:
                return float(x.item())
            return float(x.mean().item())
        try:
            return float(x)
        except Exception:
            return np.nan

    # ---------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    Y_ref = Y0.clone().detach().requires_grad_(True)
    X0 = X0.clone().detach()

    fid_eta = 1.0 if fid_eta is None else fid_eta
    c_orth = 1.0 if c_orth is None else c_orth
    aggression = float(np.clip(aggression, 0.0, 1.0))

    # --- Define energy function safely ---
    def safe_energy(Y):
        try:
            Yr = nsa_flow_retract_auto(Y, w, retraction)
            Yr = apply_nonnegativity(Yr, apply_nonneg)
            e = compute_energy(
                Yr,
                X0,
                w=w,
                fid_eta=fid_eta,
                c_orth=c_orth,
                fidelity_type=fidelity_type,
                orth_type=orth_type,
                track_grad=False,
            )
            e_val = _to_float(e)
            if not np.isfinite(e_val):
                return np.nan
            return e_val
        except Exception as ex:
            if verbose:
                warnings.warn(f"safe_energy failed: {ex}")
            return np.nan

    # ---------------------------------------------------------------------
    # Baseline energy and gradient
    # ---------------------------------------------------------------------
    f_ref = safe_energy(Y_ref)
    if np.isnan(f_ref):
        raise ValueError("Initial energy computation failed (NaN).")

    f_ref_tensor = torch.tensor(f_ref, dtype=Y_ref.dtype, requires_grad=True)
    f_ref_tensor.backward(retain_graph=False)
    grad_ref = (
        Y_ref.grad.detach().clone()
        if Y_ref.grad is not None
        else torch.zeros_like(Y_ref)
    )
    grad_norm = torch.norm(grad_ref).item() + 1e-12

    # ---------------------------------------------------------------------
    # Candidate LR generation
    # ---------------------------------------------------------------------
    strategies = [
        "armijo", "armijo_aggressive", "exponential", "linear", "entropy",
        "random", "adaptive", "momentum_boost", "poly_decay", "grid", "bayes"
    ]
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Initial LR sets per strategy (base sweep)
    if strategy == "grid":
        lr_candidates = np.logspace(-6, 2, 30)

    elif strategy == "entropy":
        base = np.logspace(-6, 2, 50)
        temp = 1e-3 + 5 * aggression
        logits = np.log(base + 1e-12) / (temp + 1e-8)
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        lr_mean = np.sum(base * probs)
        lr_candidates = lr_mean * np.logspace(-1, 1, 25)

    elif strategy == "armijo":
        lr_candidates = np.logspace(-4, 0, 20) * (1 + 3 * aggression)

    elif strategy == "armijo_aggressive":
        lr_candidates = np.logspace(-2, 2, 20) * (1 + 5 * aggression)

    elif strategy == "exponential":
        base = np.logspace(-6, 0, 25)
        lr_candidates = base ** (1 - 0.5 * aggression) * (1 + aggression)

    elif strategy == "linear":
        start = 1e-5
        stop = 1.0 * (1 + 4 * aggression)
        lr_candidates = np.linspace(start, stop, 25)

    elif strategy == "adaptive":
        base = np.logspace(-6, -2, 25)
        scale = np.clip(1.0 / (1.0 + grad_norm), 1e-4, 1.0)
        lr_candidates = base * (1 + 10 * aggression) * scale

    elif strategy == "momentum_boost":
        base = np.logspace(-5, -1, 25)
        boost = np.linspace(1.0, 1.0 + 9.0 * aggression, 25)
        lr_candidates = base * boost

    elif strategy == "poly_decay":
        base = np.linspace(1e-3, 1.0, 25)
        power = 1.0 + 2.5 * aggression
        lr_candidates = (base ** power) * (0.5 + aggression)
        lr_candidates = np.clip(lr_candidates, 1e-5, None)

    elif strategy == "random":
        rng = np.random.default_rng(42)
        lr_candidates = rng.uniform(1e-5, 1.0 * (1 + 4 * aggression), 25)

    elif strategy == "bayes":
        base = np.logspace(-3, 2, 25)
        prior = np.linspace(0.1, 1.0, 25)
        lr_candidates = base * (prior ** (1 - aggression)) * (1 + aggression)

    # ---------------------------------------------------------------------
    # Aggression-based fine tuning of range
    # ---------------------------------------------------------------------
    def scale_range(base, lo_exp=-6, hi_exp=2):
        lo = lo_exp + aggression * (hi_exp - lo_exp) * 0.7
        hi = hi_exp + aggression * (hi_exp - lo_exp) * 0.3
        return np.logspace(lo, hi, len(base))

    if strategy in ["armijo", "armijo_aggressive"]:
        lr_candidates = scale_range(lr_candidates, -4, 1 if strategy == "armijo" else 2)
    elif strategy == "grid":
        lr_candidates = scale_range(lr_candidates, -6, 1 + 2 * aggression)
    elif strategy == "exponential":
        exp_range = np.linspace(-6 + 2 * aggression, aggression, len(lr_candidates))
        lr_candidates = 10 ** exp_range
    elif strategy == "linear":
        max_lr = 0.1 + aggression * 10.0
        lr_candidates = np.linspace(1e-5, max_lr, len(lr_candidates))
    elif strategy == "random":
        rng = np.random.default_rng(42)
        lr_candidates = rng.random(len(lr_candidates)) ** (1.0 - aggression)
        lr_candidates *= 10 ** (2 * aggression - 1)
    elif strategy == "adaptive":
        base_scale = np.clip(grad_norm / (1e-3 + abs(f_ref)), 1e-4, 1e4)
        scale_factor = (base_scale ** (0.5 + aggression)) * (10 ** (2 * aggression))
        lr_candidates = np.logspace(-6, 0, len(lr_candidates)) * scale_factor
        lr_candidates = np.clip(lr_candidates, 1e-8, 1e3)
    elif strategy == "momentum_boost":
        base_lr = np.linspace(1e-5, 1e-4, len(lr_candidates))
        lr_candidates = base_lr * (1.0 + aggression * np.linspace(0.0, 10.0, len(lr_candidates)))
    elif strategy == "entropy":
        ent = np.log1p(grad_norm) / (np.log1p(abs(f_ref)) + 1e-12)
        ent_scale = float(np.clip(ent, 0.1, 5.0))
        base_lrs = np.logspace(-5, 0, len(lr_candidates))
        scale_factor = (1 + aggression * ent_scale) ** (1 + aggression * 2.0)
        lr_candidates = np.maximum.accumulate(base_lrs * scale_factor)
    elif strategy == "poly_decay":
        lr_candidates = (
            np.linspace(0.0, 1.0, len(lr_candidates)) ** (1.0 - aggression)
        ) * (10 ** (2 * aggression))
    elif strategy == "bayes":
        rng = np.random.default_rng(7)
        priors = np.exp(-np.linspace(0, 5, len(lr_candidates)))
        weights = priors ** (1.0 - aggression) + rng.normal(0, 0.05, len(lr_candidates))
        weights = np.clip(weights, 1e-6, None)
        weights /= np.sum(weights)
        lr_candidates = np.cumsum(weights) * (10 ** (1 + 2 * aggression))

    # --- Final safety clamp ---
    lr_candidates = np.clip(lr_candidates, 1e-8, 1e3)
    lr_candidates = np.unique(np.sort(lr_candidates))

    # ---------------------------------------------------------------------
    # Evaluate energies
    # ---------------------------------------------------------------------
    losses, rel_changes = [], []
    for lr in lr_candidates:
        Y_try = Y_ref - lr * grad_ref
        f_new = safe_energy(Y_try)
        losses.append(np.nan if not np.isfinite(f_new) else f_new)
        rel_changes.append((f_ref - f_new) / (abs(f_ref) + 1e-12))

    losses = np.array(losses)
    rel_changes = np.array(rel_changes)
    valid_mask = np.isfinite(losses)

    if not np.any(valid_mask):
        if verbose:
            print("⚠️ All candidate losses invalid — defaulting to conservative LR.")
        return {
            "best_lr": 1e-3,
            "best_energy": np.nan,
            "strategy": strategy,
            "aggression": aggression,
        }

    lr_candidates, losses, rel_changes = (
        lr_candidates[valid_mask],
        losses[valid_mask],
        rel_changes[valid_mask],
    )

    # ---------------------------------------------------------------------
    # Strategy-specific best LR selection
    # ---------------------------------------------------------------------
    best_lr = None
    if strategy in ["armijo", "armijo_aggressive"]:
        c = (1e-4 + 5e-3 * aggression) if strategy == "armijo_aggressive" else (1e-4 + 1e-3 * aggression)
        good = [
            lr for lr, f_new in zip(lr_candidates, losses)
            if np.isfinite(f_new) and f_new <= f_ref - c * lr * grad_norm**2
        ]
        best_lr = float(np.quantile(good, 0.9)) if good else float(lr_candidates[np.argmin(losses)])

    elif strategy == "entropy":
        safe_losses = np.where(np.isfinite(losses), losses, np.nanmean(losses))
        exp_vals = np.exp(-safe_losses / (np.nanstd(safe_losses) + 1e-8))
        exp_vals[np.isnan(exp_vals)] = 0
        probs = exp_vals / (np.sum(exp_vals) + 1e-12)
        # --- Normalize probabilities safely before sampling ---
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)
        s = probs.sum()
        if not np.isfinite(s) or s <= 0.0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / s
        idx = np.random.choice(len(lr_candidates), p=probs)
        best_lr = float(lr_candidates[idx])

    elif strategy == "bayes":
        exp_improve = (f_ref - losses) / (abs(f_ref) + 1e-12)
        grad_loss = np.gradient(losses)
        denom = np.abs(grad_loss) + 1e-8
        scores = exp_improve / denom
        scores[~np.isfinite(scores)] = -np.inf
        best_lr = float(lr_candidates[np.argmax(scores)])
    else:
        best_lr = float(lr_candidates[np.argmin(losses)])

    # ---------------------------------------------------------------------
    # Reporting & plotting
    # ---------------------------------------------------------------------
    if verbose:
        print(f"[LR-Est] {strategy:<16} | agg={aggression:.2f} | best_lr={best_lr:.3e} | minE={np.nanmin(losses):.3e}")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(lr_candidates, losses, "o-", label=f"{strategy} (agg={aggression:.2f})")
        plt.axvline(best_lr, color="r", linestyle="--", alpha=0.6)
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Energy")
        plt.legend()
        plt.title(f"LR Strategy: {strategy}")
        plt.tight_layout()
        plt.show()

    return {
        "best_lr": best_lr,
        "best_energy": float(np.nanmin(losses)),
        "strategy": strategy,
        "aggression": aggression,
        "lr_candidates": lr_candidates,
        "losses": losses,
        "rel_changes": rel_changes,
    }


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



def energy_fidelity(M, Xc, w):
    """Smooth fidelity energy used for autograd (no prox)."""
    n = Xc.shape[0]
    # negative sign to match original formulation (we minimize energy = -fid + prox)
    return -0.5 * w * torch.sum((Xc @ M) ** 2) / n

def nsa_flow_retract_auto(
    Y: torch.Tensor,
    w_retract: torch.Tensor | float = 1.0,
    retraction_type: str = "soft_polar",
    eps_rf: float = 1e-6,
    max_condition: float = 1e4,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Differentiable retraction for NSA-Flow layers.
    Removes X0 dependency, stabilizes SVD gradients, and preserves centering.

    Parameters
    ----------
    Y : torch.Tensor
        Input tensor of shape (p, k)
    w_retract : float or torch.Tensor
        Blending weight (tensor-safe, differentiable)
    retraction_type : str
        One of ["soft_polar", "polar", "normalize"]
    eps_rf : float
        Epsilon for numerical stability (added to singular values)
    max_condition : float
        Singular value clipping threshold
    verbose : bool
        Print diagnostics

    Returns
    -------
    torch.Tensor : Differentiable retraction result, same shape as Y.
    """

    # Ensure type/device consistency
    if not torch.is_tensor(w_retract):
        w_retract = torch.tensor(w_retract, dtype=Y.dtype, device=Y.device)
    else:
        w_retract = w_retract.to(dtype=Y.dtype, device=Y.device)

    # Handle degenerate case
    normY = torch.norm(Y)
    if normY < 1e-12 or not torch.isfinite(normY):
        return Y.clone()

    # Center for numerical stability
    Y_mean = Y.mean(dim=0, keepdim=True)
    Y_centered = Y - Y_mean

    try:
        if retraction_type in ["soft_polar", "polar"]:
            # Differentiable SVD
            U, S, Vh = torch.linalg.svd(Y_centered, full_matrices=False)

            # Clamp singular values
            S_clamped = torch.clamp(S, min=eps_rf, max=max_condition)

            # Polar orthogonalization
            Y_polar = U @ Vh

            if retraction_type == "soft_polar":
                Y_re_centered = (1.0 - w_retract) * Y_centered + w_retract * Y_polar
            else:
                Y_re_centered = Y_polar

            # Preserve scale
            scale_factor = normY / torch.norm(Y_re_centered).clamp_min(1e-12)
            Y_re_centered = Y_re_centered * scale_factor

        elif retraction_type == "normalize":
            norms = torch.norm(Y_centered, dim=0, keepdim=True).clamp_min(eps_rf)
            Y_re_centered = Y_centered / norms

        else:
            raise ValueError(f"Unknown retraction_type: {retraction_type}")

        # De-center to original mean space
        Y_re = Y_re_centered + Y_mean

        # Safety check
        if not torch.isfinite(Y_re).all():
            if verbose:
                print("⚠️ nsa_flow_retract_auto: non-finite values detected; reverting to input.")
            Y_re = Y.clone()

        return Y_re.to(dtype=Y.dtype, device=Y.device)

    except RuntimeError as e:
        if verbose:
            print(f"⚠️ Retraction fallback due to: {e}")
        return Y.clone()



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
        Optimizer to use ('fast' for simple gradient step, or PyTorch optimizers like 'asgd').
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
    apply_nonneg=True, optimizer="asgd",
    initial_learning_rate=None, 
    lr_strategy="bayes",
    fidelity_type="scale_invariant",
    orth_type="scale_invariant",
    aggression=0.5,
    record_every=1, window_size=5,
    device=None, precision="float64"
):
    """
    Autograd-compatible NSA-Flow (modular energy version).
    Allows user-specified fidelity_type and orth_type.

    fidelity_type ∈ {"basic", "scale_invariant", "symmetric"}
    orth_type ∈ {"basic", "scale_invariant"}
    """

    import torch, time

    # --- Precision and reproducibility ---
    if precision == "float32":
        dtype = torch.float32
    elif precision == "float64":
        dtype = torch.float64
    else:
        raise ValueError("precision must be 'float32' or 'float64'")

    torch.manual_seed(seed)
    if device is None:
        device = Y0.device

    # --- Initialization ---
    Y = Y0.clone().detach().to(device).to(dtype)
    p, k = Y.shape

    if X0 is None:
        X0 = torch.clamp(Y, min=0) if apply_nonneg else Y.clone()
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

    # --- Default scaling constants ---
    g0 = 0.5 * torch.sum((Y - X0) ** 2) / (p * k)
    g0 = torch.clamp(g0, min=1e-8)
    d0 = torch.clamp(defect_fast(Y), min=1e-8)
    fid_eta = (1 - w) / (g0 * p * k)
    c_orth = 4 * w / d0

    # --- Optimization variable ---
    Z = torch.nn.Parameter(Y.clone().detach().requires_grad_(True))

    # --- Learning rate strategy ---
    if initial_learning_rate is None and isinstance(lr_strategy, str) and \
       lr_strategy.lower() in get_lr_estimation_strategies():
        if verbose:
            print(f"[NSA-Flow] Estimating learning rate ({lr_strategy}) ...")
        lr_result = estimate_learning_rate_for_nsa_flow(
            Y0=torch.randn_like(X0),
            X0=X0,
            w=w,
            retraction=retraction,
            fid_eta=fid_eta,
            c_orth=c_orth,
            apply_nonneg=apply_nonneg,
            strategy="bayes" if lr_strategy == "auto" else lr_strategy,
            aggression=aggression,
            plot=False,
            verbose=verbose,
        )
        lr = lr_result["best_lr"]
        if verbose:
            print(f"[NSA-Flow] Selected learning rate: {lr:.2e}")
    else:
        lr = initial_learning_rate or 1e-3

    opt = get_torch_optimizer(optimizer, [Z], lr=lr)

    # --- Tracking and monitoring ---
    traces, recent_energies = [], []
    t0 = time.time()
    best_Y, best_energy, best_iter = None, float("inf"), 0

    # --- Main optimization loop ---
    for it in range(1, max_iter + 1):
        opt.zero_grad()

        # --- Retraction + optional nonnegativity ---
        Y_retracted = nsa_flow_retract_auto(Z, w, retraction)
        Y_retracted = apply_nonnegativity(Y_retracted, apply_nonneg)

        # --- Compute energy (autograd-safe) ---
        total_energy = compute_energy(
            Y_retracted,
            X0,
            w=w,
            fidelity_type=fidelity_type,
            orth_type=orth_type,
            fid_eta=fid_eta,
            c_orth=c_orth,
            track_grad=True,
        )

        total_energy.backward()
        opt.step()

        # --- Evaluate & store progress ---
        with torch.no_grad():
            Y_eval = nsa_flow_retract_auto(Z, w, retraction)
            Y_eval = apply_nonnegativity(Y_eval, apply_nonneg)
            E = compute_energy(
                Y_eval,
                X0,
                w=w,
                fidelity_type=fidelity_type,
                orth_type=orth_type,
                fid_eta=fid_eta,
                c_orth=c_orth,
                track_grad=False,
                return_dict=True
            )

            total_val = E["total"]
            fidelity_val = E["fidelity"]
            orth_val = E["orthogonality"]

            if total_val < best_energy:
                best_energy = total_val
                best_Y = Y_eval.clone()
                best_iter = it

            if it % record_every == 0:
                traces.append({
                    "iter": it,
                    "time": time.time() - t0,
                    "fidelity": fidelity_val,
                    "orthogonality": orth_val,
                    "total_energy": total_val
                })

            recent_energies.append(total_val)
            if len(recent_energies) > window_size:
                recent_energies.pop(0)

            if verbose and (it % record_every == 0 or it < 10):
                print(f"[Iter {it:3d}] Total={total_val:.6e} | Fid={fidelity_val:.6e} | Orth={orth_val:.6e}")

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
        "Y": best_Y * scale_ref,  # rescale to original magnitude
        "traces": traces,
        "final_iter": best_iter,
        "best_total_energy": best_energy,
        "best_Y_iteration": best_iter,
        "target": X0 * scale_ref,
        "settings": {
            "fidelity_type": fidelity_type,
            "orth_type": orth_type,
            "retraction": retraction,
            "optimizer": optimizer,
            "learning_rate": lr,
        }
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


def get_lr_estimation_strategies():
    """
    Return a list of supported learning rate estimation strategies
    for NSA-Flow's estimate_learning_rate_for_nsa_flow function.

    Returns:
        List[str]: Supported strategies.
    """
    strategies = [
        'armijo', 'armijo_aggressive', 'exponential', 
         'linear', 'adaptive', 'poly_decay', 'momentum_boost', # needs fixing
        'random', 'entropy', 'bayes'
    ]    
    return strategies

def test_lr_strategies():
    torch.manual_seed(0)
    X = torch.randn(30, 5)
    Y = X + 0.1 * torch.randn(30, 5)

    for strat in get_lr_estimation_strategies():
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


def test_estimate_learning_rate_for_nsa_flow(
    fidelity_type="scale_invariant",
    orth_type="scale_invariant",
    plot=True,
):
    """
    Test learning rate estimation with differentiable energy computation.
    Ensures autograd compatibility.
    """

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)

    # --- Setup ---
    Y_ref = torch.randn(40, 10, requires_grad=True)
    X0 = torch.randn_like(Y_ref)

    # --- Energy function (autograd-safe) ---
    def energy_fn(Y):
        Yr = nsa_flow_retract_auto(Y, 0.5, "soft_polar")
        Yr = apply_nonnegativity(Yr, True)
        # compute_energy must not use .item() internally
        total_energy = compute_energy(
            Yr, X0,
            w=0.5,
            fidelity_type=fidelity_type,
            orth_type=orth_type,
            track_grad=True   # ensure autograd-safe version
        )
        return total_energy  # return tensor, not .item()

    # --- Compute base energy and gradient ---
    f0 = energy_fn(Y_ref)
    assert torch.is_tensor(f0), "Energy function must return a tensor."
    assert f0.requires_grad, "Energy must retain grad tracking."

    f0.backward()  # Should work fine now
    grad = Y_ref.grad.detach()
    grad_norm = grad.norm().item()
    print(f"Initial energy={f0.item():.4e}, grad_norm={grad_norm:.4e}")

    # --- Run LR estimation across strategies ---
    strategies = get_lr_estimation_strategies()
    results = {}

    for s in strategies:
        res = estimate_learning_rate_for_nsa_flow(
            Y0=Y_ref.detach(),
            X0=X0,
            strategy=s,
            plot=False,
            verbose=True,
        )
        results[s] = res["best_lr"]
        print(f"{s:<20} → best_lr={res['best_lr']:.2e}")

    # --- Optional visualization ---
    if plot:
        plt.figure(figsize=(6, 4))
        plt.bar(results.keys(), results.values())
        plt.xticks(rotation=30)
        plt.ylabel("Estimated best learning rate")
        plt.title("Learning Rate Strategies Comparison")
        plt.tight_layout()
        plt.show()

    return results


def test_estimate_learning_rate_for_nsa_flow_modular(plot=False):
    import torch
    torch.manual_seed(0)
    Y = torch.randn(40, 8)
    X = torch.randn_like(Y)

    res = estimate_learning_rate_for_nsa_flow(
        Y, X,
        w=0.4,
        retraction="soft_polar",
        fidelity_type="symmetric",
        orth_type="scale_invariant",
        strategy="armijo_aggressive",
        plot=plot,
        verbose=True
    )

    assert "best_lr" in res
    assert res["best_lr"] > 0, "Learning rate should be positive."
    assert np.isfinite(res["losses"]).all(), "Losses contain NaN or Inf."
    print(f"✅ Passed LR modular test — best_lr={res['best_lr']:.3e}")
    return res



def test_retraction_across_w(plot=True):
    import torch, numpy as np, matplotlib.pyplot as plt
    torch.manual_seed(0)
    Y = torch.randn(30, 5)
    ws = torch.linspace(0.0, 1.0, steps=11)
    norms = []
    orth = []

    for w in ws:
        Yr = nsa_flow_retract_auto(Y, w_retract=w, retraction_type="soft_polar")
        norms.append(torch.norm(Yr).item())
        orth.append(defect_fast(Yr).item())

    norms = np.array(norms)
    orth = np.array(orth)

    if plot:
        fig, ax1 = plt.subplots(figsize=(7, 4))

        color1 = "tab:blue"
        ax1.set_xlabel("w_retract")
        ax1.set_ylabel("‖Y_retracted‖ (scale)", color=color1)
        ax1.plot(ws, norms, "-o", color=color1, label="‖Y_retracted‖")
        ax1.tick_params(axis="y", labelcolor=color1)

        # Create second axis for orthogonality defect
        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("Orthogonality defect", color=color2)
        ax2.plot(ws, orth, "-s", color=color2, label="Orth defect")
        ax2.tick_params(axis="y", labelcolor=color2)

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")

        plt.title("Retraction effects on scale vs orthogonality")
        plt.tight_layout()
        plt.show()

    assert np.all(np.isfinite(norms)) and np.all(np.isfinite(orth)), "NaN or Inf detected."
    print("✅ Retraction visualization and sanity check passed.")

def test_estimate_learning_rate_for_nsa_flow_modular(plot=False):
    import torch
    torch.manual_seed(0)
    Y = torch.randn(40, 8)
    X = torch.randn_like(Y)

    res = estimate_learning_rate_for_nsa_flow(
        Y, X,
        w=0.4,
        retraction="soft_polar",
        fidelity_type="symmetric",
        orth_type="scale_invariant",
        strategy="armijo_aggressive",
        plot=plot,
        verbose=True
    )

    assert "best_lr" in res
    assert res["best_lr"] > 0, "Learning rate should be positive."
    assert np.isfinite(res["losses"]).all(), "Losses contain NaN or Inf."
    print(f"✅ Passed LR modular test — best_lr={res['best_lr']:.3e}")
    return res


def test_nsa_flow_autograd_modular(plot=True):
    import torch
    torch.manual_seed(1)
    Y = torch.randn(50, 10)
    X = torch.randn_like(Y)

    result = nsa_flow_autograd(
        Y0=Y,
        X0=X,
        w=0.5,
        retraction="soft_polar",
        fidelity_type="scale_invariant",
        orth_type="scale_invariant",
        max_iter=100,
        lr_strategy="auto",
        verbose=True,
    )
    nrgratio = result['traces']['total_energy'].iloc[result['best_Y_iteration']-1] / result['traces']['total_energy'].iloc[0]
    Y_final = result["Y"]
    assert torch.isfinite(Y_final).all(), "Non-finite values detected in Y."
    assert nrgratio < 1.0, "Energy did not decrease sufficiently."
    print(f"✅ Passed NSA-Flow autograd modular test at iter {result['final_iter']}")

    if plot:
        plot_nsa_trace(result["traces"])
    return result



def test_aggression_effect_with_convergence():
    """
    Demonstrate that increasing aggression raises the learning rate
    and accelerates (but destabilizes) convergence.
    """
    import torch, numpy as np, matplotlib.pyplot as plt

    torch.manual_seed(0)
    np.random.seed(0)
    Y0 = torch.randn(40, 5, dtype=torch.float64)
    X0 = torch.randn(40, 5, dtype=torch.float64)

    aggression_levels = [0.0, 0.5, 1.0]
    results = []

    for agg in aggression_levels:
        res = estimate_learning_rate_for_nsa_flow(
            Y0, X0, w=0.5, retraction="soft_polar",
            aggression=agg, verbose=False, plot=False
        )
        # One gradient descent step
        Y_try = Y0 - res["best_lr"] * torch.randn_like(Y0)
        f_new = torch.norm(Y_try - X0).item()
        results.append((agg, res["best_lr"], res["best_energy"], f_new))

    print("\n=== Aggression Effect ===")
    for agg, lr, e0, f1 in results:
        print(f"agg={agg:.1f} → best_lr={lr:.2e}, energy={e0:.3e}, after-step={f1:.3e}")

    # Visualization
    plt.figure(figsize=(7, 5))
    for agg, lr, e0, f1 in results:
        plt.scatter(agg, lr, label=f"agg={agg}", s=80)
    plt.xlabel("Aggression")
    plt.ylabel("Selected Learning Rate (log10)")
    plt.yscale("log")
    plt.title("Aggression vs Selected Learning Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()



def test_lr_aggression_monotonicity(verbose=True, plot=False):
    """
    Test that learning rate estimates increase monotonically with aggression level
    for each available strategy in NSA-Flow.
    """

    torch.manual_seed(0)
    np.random.seed(0)

    # Minimal example data
    Y = torch.randn(30, 5)
    X = torch.randn(30, 5)

    strategies = get_lr_estimation_strategies(  )
    strategies.remove("entropy")  # temporarily exclude due to instability
    aggressions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    results = {}

    print("=" * 80)
    print("🧪 Testing learning rate monotonicity vs. aggression")
    print("=" * 80)

    for strategy in strategies:
        lr_vals = []
        for agg in aggressions:
            try:
                res = estimate_learning_rate_for_nsa_flow(
                    Y,
                    X,
                    w=0.5,
                    retraction="soft_polar",
                    strategy=strategy,
                    aggression=float(agg),
                    verbose=False,
                    plot=False,
                )
                lr_vals.append(res["best_lr"])
            except Exception as e:
                lr_vals.append(np.nan)
                print(f"{strategy:<18} agg={agg:.2f} ⚠️ {str(e)}")

        lr_vals = np.array(lr_vals)
        results[strategy] = lr_vals

        # --- Monotonicity test ---
        diffs = np.diff(np.log10(lr_vals + 1e-12))  # check log-scale differences
        monotonic = np.all(diffs >= -1e-3)  # small tolerance

        if verbose:
            print(f"{strategy:<18} LRs: {[f'{v:.2e}' for v in lr_vals]} | {'✅' if monotonic else '❌ Non-monotonic'}")

        assert np.any(np.isfinite(lr_vals)), f"{strategy}: all NaN learning rates"
        assert monotonic, f"{strategy}: learning rate did not increase monotonically with aggression"

    # --- Optional visualization ---
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for strat, vals in results.items():
            if not np.all(np.isnan(vals)):
                plt.plot(aggressions, vals, "-o", label=strat)
        plt.xlabel("Aggression Level")
        plt.ylabel("Estimated Learning Rate")
        plt.title("Monotonicity of LR vs. Aggression Across Strategies")
        plt.yscale("log")
        plt.grid(True, ls="--", alpha=0.6)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    print("✅ All strategies tested for monotonicity.")
    return results



def run_single_experiment(
    size=[50,10],
    w=0.5,
    strategy='auto',
    optimizer_name='lars',
    aggression=0.5,
    fidelity_type="scale_invariant",
    orth_type="scale_invariant",
    device="cpu",
    seed=42,
    verbose=False,
):
    """
    Run a single NSA-Flow optimization with given configuration and aggression level.
    Returns summary dict of results including fidelity and orth energies.
    """
    import torch, numpy as np, time
    torch.manual_seed(seed + int(aggression * 1000))
    np.random.seed(seed + int(aggression * 1000))

    p, k = size
    X0 = torch.randn(p, k, device=device)
    Y0 = torch.randn_like(X0)

    try:
        t0 = time.time()
        res = nsa_flow_autograd(
            Y0,
            X0=X0,
            w=w,
            retraction="soft_polar",
            optimizer=optimizer_name,
            lr_strategy=strategy,
            aggression=aggression,
            max_iter=200,
            tol=1e-5,
            verbose=verbose,
            apply_nonneg=True,
            seed=seed,
            record_every=10,
            precision="float32",
        )
        elapsed = time.time() - t0

        Y_final = res.get("Y", None)
        if Y_final is None and "Y_final" in res:
            Y_final = res["Y_final"]
        if Y_final is not None:
            energy_dict = compute_energy(
                Y_final,
                X0,
                w=w,
                fidelity_type=fidelity_type,
                orth_type=orth_type,
                fid_eta=1.0,
                c_orth=1.0,
                track_grad=False,
                return_dict=True,
            )
            total_energy = energy_dict["total"]
            fid_energy = energy_dict["fidelity"]
            orth_energy = energy_dict["orthogonality"]
        else:
            total_energy = fid_energy = orth_energy = np.nan

        best_iter = int(res.get("best_Y_iteration", np.nan))
        success = True

    except Exception as e:
        total_energy = fid_energy = orth_energy = np.nan
        best_iter = np.nan
        elapsed = np.nan
        success = False
        print(f"⚠️ Failure: {strategy} / {optimizer_name} (agg={aggression:.2f}) — {e}")

    return {
        "size": f"{p}x{k}",
        "w": w,
        "strategy": strategy,
        "optimizer": optimizer_name,
        "aggression": aggression,
        "fidelity_type": fidelity_type,
        "orth_type": orth_type,
        "final_energy": total_energy,
        "total_energy": total_energy,
        "fidelity_energy": fid_energy,
        "orth_energy": orth_energy,
        "best_iter": best_iter,
        "elapsed_sec": elapsed,
        "success": success,
    }

def evaluate(seed: int = 42, fast: bool = False, verbose: bool = True):
    """
    Comprehensive benchmark of NSA-Flow configurations.

    Tests combinations of:
    - learning rate estimation strategies
    - optimizers
    - aggression levels
    - weighting parameter w
    - matrix sizes
    - fidelity and orthogonality energy types

    Returns
    -------
    pd.DataFrame
        Results of all experiments with metrics and rankings.
    """
    import time
    import torch
    import numpy as np
    import pandas as pd
    import nsa_flow

    # ------------------------------------------------------------------
    # Setup reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Experimental configuration
    # ------------------------------------------------------------------
    strategies = nsa_flow.get_lr_estimation_strategies()
    optimizers = nsa_flow.get_torch_optimizer(return_list=True)
    optimizers = [opt for opt in optimizers if opt.lower() not in ["test", "none"]]
    aggressions = [0.0, 0.25, 0.5, 0.75, 1.0]
    ws = [0.01, 0.1, 0.25, 0.5, 0.9, 0.99]
    matrix_sizes = [(20, 5), (50, 10), (100, 20), (200, 40), (40, 200)]

    if fast == 1:
        optimizers = ["lars",  "asgd" ]
        matrix_sizes = [(50, 20), (20, 50)]# , (100,400)]
        ws = [0.05, 0.25, 0.5, 0.7, 0.9]
        aggressions = [0.1,0.25, 0.5, 0.75]
        strategies = ["random", "armijo_aggressive", "armijo", "bayes"]
    elif fast == 2:
        optimizers = ["lars" ]
        matrix_sizes = [(50, 10)]
        ws = [ 0.5, 0.9]
        aggressions = [ 0.5, 0.75 ]
        strategies = ["armijo", "bayes"]


    fidelity_types = ["basic", "scale_invariant", "symmetric"]
    orth_types = ["basic", "scale_invariant"]

    all_results = []

    # ✅ include w in total job count
    total_jobs = (
        len(strategies)
        * len(optimizers)
        * len(aggressions)
        * len(ws)
        * len(matrix_sizes)
        * len(fidelity_types)
        * len(orth_types)
    )

    print(f"🔍 Evaluating {total_jobs} configurations...\n")

    job = 0
    t0 = time.time()

    # ------------------------------------------------------------------
    # Main experiment loop
    # ------------------------------------------------------------------
    for size in matrix_sizes:
        for strategy in strategies:
            for optimizer_name in optimizers:
                for agg in aggressions:
                    for fid_type in fidelity_types:
                        for orth_type in orth_types:
                            for w in ws:
                                job += 1
                                if verbose:
                                    print(
                                        f"[{job:5d}/{total_jobs}] size={size}, strat={strategy}, "
                                        f"opt={optimizer_name}, agg={agg:.2f}, "
                                        f"fid={fid_type}, orth={orth_type}, w={w}"
                                    )

                                try:
                                    # ✅ Pass w into run_single_experiment
                                    res = run_single_experiment(
                                        size=size,
                                        w=w,
                                        strategy=strategy,
                                        optimizer_name=optimizer_name,
                                        aggression=agg,
                                        fidelity_type=fid_type,
                                        orth_type=orth_type,
                                        seed=seed,
                                    )

                                    # ✅ Ensure energies are computed if missing
                                    if "fidelity_energy" not in res or "orth_energy" not in res:
                                        Y_final = res.get("Y_final", None)
                                        X0 = res.get("X0", None)
                                        if Y_final is not None and X0 is not None:
                                            e_total, e_fid, e_orth = nsa_flow.compute_energy_components(
                                                Y_final,
                                                X0,
                                                w=w,
                                                fid_eta=res.get("fid_eta", 1.0),
                                                c_orth=res.get("c_orth", 1.0),
                                                fidelity_type=fid_type,
                                                orth_type=orth_type,
                                            )
                                            res["fidelity_energy"] = e_fid
                                            res["orth_energy"] = e_orth
                                            res["total_energy"] = e_total

                                    # ✅ Record metadata cleanly
                                    res.update(
                                        dict(
                                            size=size,
                                            w=w,
                                            strategy=strategy,
                                            optimizer=optimizer_name,
                                            aggression=agg,
                                            fidelity_type=fid_type,
                                            orth_type=orth_type,
                                            seed=seed,
                                        )
                                    )
                                    all_results.append(res)

                                except Exception as e:
                                    if verbose:
                                        print(
                                            f"⚠️ Failure: {strategy}/{optimizer_name} "
                                            f"(agg={agg}, w={w}) — {e}"
                                        )
                                    all_results.append(
                                        dict(
                                            size=size,
                                            w=w,
                                            strategy=strategy,
                                            optimizer=optimizer_name,
                                            aggression=agg,
                                            fidelity_type=fid_type,
                                            orth_type=orth_type,
                                            error=str(e),
                                        )
                                    )

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_results)

    # ✅ Compute ranking metrics safely
    if "total_energy" in df.columns:
        df["rank_total"] = df["total_energy"].rank(method="dense", ascending=True)
    if "fidelity_energy" in df.columns:
        df["rank_fid"] = df["fidelity_energy"].rank(method="dense", ascending=True)
    if "orth_energy" in df.columns:
        df["rank_orth"] = df["orth_energy"].rank(method="dense", ascending=True)

    elapsed_total = time.time() - t0
    print(f"\n✅ Completed all {job} experiments in {elapsed_total/60:.2f} minutes.")

    # ------------------------------------------------------------------
    # Summary ranking
    # ------------------------------------------------------------------
    if "rank_total" in df.columns:
        summary = (
            df.groupby(["strategy", "optimizer", "fidelity_type", "orth_type", "w"])
            .agg(
                mean_total_energy=("total_energy", "mean"),
                mean_rank=("rank_total", "mean"),
                std_total_energy=("total_energy", "std"),
                n=("total_energy", "count"),
            )
            .sort_values("mean_rank")
        )
        print("\n🏆 Top Performing Configurations:\n")
        print(summary.head(10))

    return df


def plot_evaluation_summary(df, save_dir=None, timestamp=None, show_plots=True):
    """
    Generate and optionally save detailed summary visualizations for NSA-Flow evaluation results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame output from `evaluate()`, expected to include:
        ['strategy', 'optimizer', 'aggression', 'total_energy', 'w', 'error']
    save_dir : str, optional
        Directory where plots and summary CSVs will be saved.
    timestamp : str, optional
        Timestamp to append to saved filenames.
    show_plots : bool, default=True
        Whether to display plots interactively.
    """
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    def _robust_clip(df, cols, lower=1, upper=99):
        """Clip dataframe numeric columns to percentile range to reduce outlier influence."""
        df_clipped = df.copy()
        for col in cols:
            if col in df.columns:
                lo, hi = np.percentile(df[col].dropna(), [lower, upper])
                df_clipped[col] = df[col].clip(lo, hi)
        return df_clipped
    # ------------------------------------------------------------------
    # Filter successful runs
    # ------------------------------------------------------------------
    if "error" in df.columns:
        df_plot = df[df["error"].isna()].copy()
    else:
        df_plot = df.copy()

    df_plot = df_plot[df_plot["total_energy"].notna() & np.isfinite(df_plot["total_energy"])]
    df_plot = _robust_clip(df_plot, ["total_energy", "fidelity_energy", "orth_energy"])
    # ------------------------------------------------------------------
    # Summary table: Top configurations
    # ------------------------------------------------------------------
    summary = (
        df_plot.groupby(["strategy", "optimizer", "w"])
        .agg(
            mean_total=("total_energy", "mean"),
            std_total=("total_energy", "std"),
            n=("total_energy", "count"),
        )
        .reset_index()
        .sort_values("mean_total", ascending=True)
    )
    print("\n🏆 Top 10 Configurations by Mean Total Energy:\n")
    print(summary.head(10).to_string(index=False, float_format="%.4e"))

    if save_dir:
        summary_csv = f"{save_dir}/summary_ranking_{timestamp or ''}.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"💾 Saved ranking summary: {summary_csv}")

    # ------------------------------------------------------------------
    # Heatmap 1: Strategy × Aggression
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    pivot_strat = (
        df_plot.groupby(["strategy", "aggression"])["total_energy"].mean().unstack()
    )
    sns.heatmap(pivot_strat, annot=True, fmt=".2e", cmap="viridis", cbar_kws={"label": "Mean Total Energy"})
    plt.title("NSA-Flow: Strategy vs Aggression (Mean Total Energy)")
    plt.ylabel("Strategy")
    plt.xlabel("Aggression Level")
    plt.tight_layout()
    if save_dir:
        fname = f"{save_dir}/heatmap_strategy_vs_aggression_{timestamp or ''}.png"
        plt.savefig(fname, dpi=300)
        print(f"💾 Saved: {fname}")
    if show_plots:
        plt.show()
    plt.close()

    # ------------------------------------------------------------------
    # Heatmap 2: Optimizer × Aggression
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    pivot_opt = (
        df_plot.groupby(["optimizer", "aggression"])["total_energy"].mean().unstack()
    )
    sns.heatmap(pivot_opt, annot=True, fmt=".2e", cmap="magma", cbar_kws={"label": "Mean Total Energy"})
    plt.title("NSA-Flow: Optimizer vs Aggression (Mean Total Energy)")
    plt.ylabel("Optimizer")
    plt.xlabel("Aggression Level")
    plt.tight_layout()
    if save_dir:
        fname = f"{save_dir}/heatmap_optimizer_vs_aggression_{timestamp or ''}.png"
        plt.savefig(fname, dpi=300)
        print(f"💾 Saved: {fname}")
    if show_plots:
        plt.show()
    plt.close()

    # ------------------------------------------------------------------
    # Heatmap 3: Strategy × Weight (w)
    # ------------------------------------------------------------------
    if "w" in df_plot.columns:
        plt.figure(figsize=(10, 6))
        pivot_w = df_plot.groupby(["strategy", "w"])["total_energy"].mean().unstack()
        sns.heatmap(pivot_w, annot=True, fmt=".2e", cmap="plasma", cbar_kws={"label": "Mean Total Energy"})
        plt.title("NSA-Flow: Strategy vs Weight (w)")
        plt.ylabel("Strategy")
        plt.xlabel("Weight (w)")
        plt.tight_layout()
        if save_dir:
            fname = f"{save_dir}/heatmap_strategy_vs_w_{timestamp or ''}.png"
            plt.savefig(fname, dpi=300)
            print(f"💾 Saved: {fname}")
        if show_plots:
            plt.show()
        plt.close()

        # ------------------------------------------------------------------
        # Faceted plot: Energy vs Aggression (by Optimizer × Strategy)
        # ------------------------------------------------------------------
        g = sns.FacetGrid(df_plot, col="optimizer", row="strategy", margin_titles=True, sharey=False)
        g.map_dataframe(sns.lineplot, x="aggression", y="total_energy", hue="w", marker="o")
        g.add_legend(title="Weight (w)")
        g.set_axis_labels("Aggression", "Total Energy")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("NSA-Flow: Total Energy vs Aggression (Faceted by Optimizer × Strategy)")

        # ⭐ Highlight best overall config
        best_row = df_plot.loc[df_plot["total_energy"].idxmin()]
        best_opt, best_strat = best_row["optimizer"], best_row["strategy"]
        best_agg, best_w, best_E = best_row["aggression"], best_row["w"], best_row["total_energy"]

        # Find the right facet
        for (opt, strat), ax in g.axes_dict.items():
            if opt == best_opt and strat == best_strat:
                ax.scatter(best_agg, best_E, s=150, color="gold", edgecolor="black", marker="*", zorder=10, label="⭐ Best")
                ax.legend()

        # Save and show
        if save_dir:
            fname = f"{save_dir}/facet_energy_vs_aggression_{timestamp or ''}_highlighted.png"
            g.savefig(fname, dpi=300)
            print(f"💾 Saved with highlight: {fname}")

        if show_plots:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # Line plot: Energy vs Aggression (Top Strategies)
    # ------------------------------------------------------------------
    mean_energy = df_plot.groupby(["strategy", "aggression"])["total_energy"].mean().reset_index()
    top_strats = mean_energy.groupby("strategy")["total_energy"].mean().nsmallest(5).index

    plt.figure(figsize=(8, 5))
    for strat in top_strats:
        subset = mean_energy[mean_energy["strategy"] == strat]
        plt.plot(subset["aggression"], subset["total_energy"], "-o", label=strat)
    plt.xlabel("Aggression Level")
    plt.ylabel("Mean Total Energy")
    plt.title("Top Strategies: Energy vs Aggression")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        fname = f"{save_dir}/lineplot_top_strategies_{timestamp or ''}.png"
        plt.savefig(fname, dpi=300)
        print(f"💾 Saved: {fname}")
    if show_plots:
        plt.show()
    plt.close()

    # ------------------------------------------------------------------
    # Line plot: Energy vs Weight (Top Optimizers)
    # ------------------------------------------------------------------
    if "w" in df_plot.columns:
        mean_w = df_plot.groupby(["optimizer", "w"])["total_energy"].mean().reset_index()
        top_opts = mean_w.groupby("optimizer")["total_energy"].mean().nsmallest(5).index
        plt.figure(figsize=(8, 5))
        for opt in top_opts:
            subset = mean_w[mean_w["optimizer"] == opt]
            plt.plot(subset["w"], subset["total_energy"], "-o", label=opt)
        plt.xlabel("Weight (w)")
        plt.ylabel("Mean Total Energy")
        plt.title("Top Optimizers: Energy vs Weight")
        plt.legend()
        plt.tight_layout()
        if save_dir:
            fname = f"{save_dir}/lineplot_top_optimizers_vs_w_{timestamp or ''}.png"
            plt.savefig(fname, dpi=300)
            print(f"💾 Saved: {fname}")
        if show_plots:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # Save cleaned data summary
    # ------------------------------------------------------------------
    if save_dir:
        clean_csv = f"{save_dir}/cleaned_results_{timestamp or ''}.csv"
        df_plot.to_csv(clean_csv, index=False)
        print(f"💾 Saved cleaned result data: {clean_csv}")

    print("✅ All summary plots and rankings generated successfully.")


### deep learning stuff below 

######################################################################
# NSA-Flow Lightweight Test Suite — Full Drop-In Replacement
######################################################################
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
import torch.distributions as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


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


######################################################################
# NSA Flow Layer (Computational Retraction)
######################################################################
class NSAFlowLayer(nn.Module):
    """Differentiable orthogonality regularizer via gradient–retraction flow."""

    def __init__(self, w_retract=0.5, w_learnable=True,
                 retraction_type="soft_polar", tradeoff_alpha=0.9, tradeoff_learnable=True,
                 nn_constraint="softplus", **kwargs):
        super().__init__()
        self.w_retract = nn.Parameter(torch.tensor(float(w_retract)), requires_grad=w_learnable)
        self.tradeoff_alpha = nn.Parameter(torch.tensor(float(tradeoff_alpha)), requires_grad=tradeoff_learnable)
        self.retraction_type = retraction_type
        self.nn_constraint = nn_constraint
        # Filter out 'out_dim' if present to avoid passing to retract_auto
        kwargs.pop('out_dim', None)
        self.kwargs = kwargs

    def forward(self, Y, X=None):
        from .core import nsa_flow_retract_auto, defect_fast  # local import to avoid circular refs
        if not Y.requires_grad:
            Y = Y.requires_grad_(True)

        try:
            alpha = torch.sigmoid(self.tradeoff_alpha)
            f_weight = alpha
            o_weight = 1.0 - alpha
            E_fid = fidelity_scaled(Y, X) * f_weight
            E_orth = invariant_orthogonality_defect(Y) * o_weight
            E = E_fid + E_orth
            grad_Y = torch.autograd.grad(E, Y, create_graph=self.training)[0]
            Y_step = Y - grad_Y
            Y_re = nsa_flow_retract_auto(
                Y_step,
                w_retract=self.w_retract.to(Y),
                retraction_type=self.retraction_type,
                **self.kwargs,
            )
            Y_re = apply_nonnegativity(Y_re, mode=self.nn_constraint)
            if not torch.isfinite(Y_re).all():
                Y_re = defect_fast(Y)
            return Y_re, {"E_fid": E_fid.detach().item(), "E_orth": E_orth.detach().item()}
        except Exception as e:
            print(f"⚠️ NSAFlowLayer fallback: {e}")
            Y_out = Y.detach()  # Return detached Y to preserve shape instead of potential scalar
            return Y_out, {"E_fid": float("nan"), "E_orth": float("nan")}


######################################################################
# NSA Flow Learnable Retraction Layer
######################################################################
class NSAFlowLearnableRetractionLayer(nn.Module):
    """Differentiable orthogonality regularizer with learnable retraction transform."""

    def __init__(self, w_retract=0.5, w_learnable=True,
                 retraction_type="soft_polar", tradeoff_alpha=0.9, tradeoff_learnable=True,
                 nn_constraint="softplus", out_dim=5, **kwargs):
        super().__init__()
        self.w_retract = nn.Parameter(torch.tensor(float(w_retract)), requires_grad=w_learnable)
        self.tradeoff_alpha = nn.Parameter(torch.tensor(float(tradeoff_alpha)), requires_grad=tradeoff_learnable)
        self.retraction_type = retraction_type
        self.nn_constraint = nn_constraint
        self.kwargs = kwargs

        # Learnable transform: a small MLP to approximate the retraction
        self.transform_net = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )  # Residual-like

    def forward(self, Y, X=None):
        from .core import defect_fast  # local import to avoid circular refs
        if not Y.requires_grad:
            Y = Y.requires_grad_(True)

        Y_re = Y + self.transform_net(Y)  # Residual connection for stability
        Y_re = apply_nonnegativity(Y_re, mode=self.nn_constraint)
        if not torch.isfinite(Y_re).all():
            Y_re = defect_fast(Y)
        # Compute info for monitoring/aux loss
        with torch.enable_grad():
            alpha = torch.sigmoid(self.tradeoff_alpha)
            f_weight = alpha
            o_weight = 1.0 - alpha
            E_fid = fidelity_scaled(Y_re, X) * f_weight
            E_orth = invariant_orthogonality_defect(Y_re) * o_weight
        return Y_re, {"E_fid": E_fid.detach().item(), "E_orth": E_orth.detach().item()}


######################################################################
# Tiny Regression MLP
######################################################################
class SmallRegressor(nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=5, use_flow=False, flow_class=NSAFlowLayer, flow_kwargs=None):
        super().__init__()
        self.use_flow = use_flow
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(in_dim, hid_dim*2),
            nn.ReLU(),
            nn.Linear(hid_dim*2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )
        self.flow = None
        if use_flow:
            flow_kwargs = flow_kwargs or {}
            flow_kwargs['out_dim'] = out_dim  # Pass out_dim for potential use
            self.flow = flow_class(**(flow_kwargs))

    def forward(self, x):
        z = self.net(x)
        if not self.use_flow:
            return z, None
        z_flow, info = self.flow(z, X=z.detach())
        return z_flow, info


######################################################################
# Evaluation Helper
######################################################################
def eval_model(model, loader, loss_fn, is_flow=False):
    model.eval()
    total_loss = 0.0
    Ef_hist, Eo_hist = [], []

    if is_flow:
        with torch.enable_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred, info = model(xb)
                total_loss += loss_fn(y_pred, yb).item() * xb.size(0)
                if info:
                    Ef_hist.append(info.get("E_fid", float("nan")))
                    Eo_hist.append(info.get("E_orth", float("nan")))
    else:
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred, _ = model(xb)
                total_loss += loss_fn(y_pred, yb).item() * xb.size(0)
    return total_loss / len(loader.dataset), Ef_hist, Eo_hist


######################################################################
# Prediction Collector
######################################################################
def get_predictions(model, X, batch_size=64, is_flow=False):
    model.eval()
    preds = []
    context = torch.enable_grad() if is_flow else torch.no_grad()
    with context:
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            y_pred, _ = model(xb)
            preds.append(y_pred.detach().cpu())
    return torch.cat(preds).numpy()


######################################################################
# PCA Regression Data + Deep Comparison
######################################################################
def test_nsa_flow_layer_lite(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
#    torch.set_default_dtype(torch.float64)
    print("🧠 Training models...")
    # Synthetic PCA regression data with high multivariate correlation
    n_samples, n_features, n_targets = 512, 100, 3
    mean = torch.zeros(n_features, device=device)
    corr = 0.5  # High correlation
    R = torch.full((n_features, n_features), corr, device=device)
    R.diagonal().fill_(1.0)
    X = dist.MultivariateNormal(mean, R).sample((n_samples,)).to(device)
    pca = PCA(n_components=n_targets)
    Y = torch.from_numpy(pca.fit_transform(X.cpu().numpy())).to(device)#.float()

    # Train/Test split
    idx = torch.randperm(n_samples)
    tr, te = idx[:400], idx[400:]
    Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]

    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xte, Yte), batch_size=64)

    # Common flow kwargs
    flow_kwargs = dict(w_retract=0.5, retraction_type="soft_polar", tradeoff_alpha=0.909, tradeoff_learnable=True, nn_constraint="none")

    # Models
    base_model = SmallRegressor(n_features, hid_dim=64, out_dim=n_targets, use_flow=False).to(device)
    flow_comp_model = SmallRegressor(
        n_features, hid_dim=64, out_dim=n_targets, use_flow=True, flow_class=NSAFlowLayer, flow_kwargs=flow_kwargs
    ).to(device)
    flow_learn_model = SmallRegressor(
        n_features, hid_dim=64, out_dim=n_targets, use_flow=True, flow_class=NSAFlowLearnableRetractionLayer, flow_kwargs=flow_kwargs
    ).to(device)

    loss_fn = nn.MSELoss()
    opt_base = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    opt_flow_comp = torch.optim.Adam(flow_comp_model.parameters(), lr=1e-3)
    opt_flow_learn = torch.optim.Adam(flow_learn_model.parameters(), lr=1e-3)

    base_hist = []
    flow_comp_hist, flow_comp_Efid_hist, flow_comp_Eorth_hist = [], [], []
    flow_learn_hist, flow_learn_Efid_hist, flow_learn_Eorth_hist = [], [], []

    for epoch in range(1, 51):
        base_model.train()
        flow_comp_model.train()
        flow_learn_model.train()
        base_loss_epoch = flow_comp_loss_epoch = flow_learn_loss_epoch = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # baseline
            opt_base.zero_grad()
            y_pred, _ = base_model(xb)
            loss_b = loss_fn(y_pred, yb)
            loss_b.backward()
            opt_base.step()
            base_loss_epoch += loss_b.item() * xb.size(0)

            # flow comp model
            opt_flow_comp.zero_grad()
            y_pred_fc, info_fc = flow_comp_model(xb)
            loss_fc = loss_fn(y_pred_fc, yb)
            loss_fc.backward()
            opt_flow_comp.step()
            flow_comp_loss_epoch += loss_fc.item() * xb.size(0)
            if info_fc:
                flow_comp_Efid_hist.append(info_fc["E_fid"])
                flow_comp_Eorth_hist.append(info_fc["E_orth"])

            # flow learn model
            opt_flow_learn.zero_grad()
            y_pred_fl, info_fl = flow_learn_model(xb)
            loss_fl = loss_fn(y_pred_fl, yb)
            # Add auxiliary loss to train the transform_net to minimize E
            E_fid = torch.tensor(info_fl["E_fid"], device=device)
            E_orth = torch.tensor(info_fl["E_orth"], device=device)
            aux_loss = E_fid + E_orth
            loss_fl += 0.1 * aux_loss  # Weight auxiliary loss
            loss_fl.backward()
            opt_flow_learn.step()
            flow_learn_loss_epoch += loss_fl.item() * xb.size(0)
            if info_fl:
                flow_learn_Efid_hist.append(info_fl["E_fid"])
                flow_learn_Eorth_hist.append(info_fl["E_orth"])

        # Evaluate
        tb = eval_model(base_model, test_loader, loss_fn, is_flow=False)[0]
        tfc, Efc_hist, Eoc_hist = eval_model(flow_comp_model, test_loader, loss_fn, is_flow=True)
        tfl, Efl_hist, Eol_hist = eval_model(flow_learn_model, test_loader, loss_fn, is_flow=True)
        base_hist.append(tb)
        flow_comp_hist.append(tfc)
        flow_learn_hist.append(tfl)

        print(f"Epoch {epoch:04d} | "
              f"Base train/test {base_loss_epoch/len(train_loader.dataset):.4e}/{tb:.4e} | "
              f"FlowComp train/test {flow_comp_loss_epoch/len(train_loader.dataset):.4e}/{tfc:.4e} | "
              f"FlowLearn train/test {flow_learn_loss_epoch/len(train_loader.dataset):.4e}/{tfl:.4e} | "
              f"w_comp={flow_comp_model.flow.w_retract.item():.4f} "
              f"alpha_comp={torch.sigmoid(flow_comp_model.flow.tradeoff_alpha).item():.4f} | "
              f"w_learn={flow_learn_model.flow.w_retract.item():.4f} "
              f"alpha_learn={torch.sigmoid(flow_learn_model.flow.tradeoff_alpha).item():.4f}")

    ##################################################################
    # Deep Reporting
    ##################################################################
    Zb = get_predictions(base_model, Xte, is_flow=False)
    Zfc = get_predictions(flow_comp_model, Xte, is_flow=True)
    Zfl = get_predictions(flow_learn_model, Xte, is_flow=True)
    Yt = Yte.cpu().numpy()

    from .core import invariant_orthogonality_defect

    # Additional metrics for deeper comparison
    with torch.no_grad():
        Zb_t = torch.from_numpy(Zb).to(device)
        Zfc_t = torch.from_numpy(Zfc).to(device)
        Zfl_t = torch.from_numpy(Zfl).to(device)
        ortho_def_base = invariant_orthogonality_defect(Zb_t).item()
        ortho_def_comp = invariant_orthogonality_defect(Zfc_t).item()
        ortho_def_learn = invariant_orthogonality_defect(Zfl_t).item()

    # Procrustes alignment of learned subspaces to ground truth PCA
    Rb, _ = orthogonal_procrustes(Zb, Yt)
    Rfc, _ = orthogonal_procrustes(Zfc, Yt)
    Rfl, _ = orthogonal_procrustes(Zfl, Yt)
    Zb_aligned = Zb @ Rb
    Zfc_aligned = Zfc @ Rfc
    Zfl_aligned = Zfl @ Rfl
    procrustes_err_base = np.linalg.norm(Zb_aligned - Yt) / np.linalg.norm(Yt)
    procrustes_err_comp = np.linalg.norm(Zfc_aligned - Yt) / np.linalg.norm(Yt)
    procrustes_err_learn = np.linalg.norm(Zfl_aligned - Yt) / np.linalg.norm(Yt)

    # Singular value trajectories
    sv_b = np.linalg.svd(Zb, compute_uv=False)
    sv_fc = np.linalg.svd(Zfc, compute_uv=False)
    sv_fl = np.linalg.svd(Zfl, compute_uv=False)

    # Correlations between embedding dimensions
    corr_base = np.corrcoef(Zb.T)
    corr_comp = np.corrcoef(Zfc.T)
    corr_learn = np.corrcoef(Zfl.T)
    off_diag_avg_base = np.mean(np.abs(corr_base - np.eye(corr_base.shape[0])))
    off_diag_avg_comp = np.mean(np.abs(corr_comp - np.eye(corr_comp.shape[0])))
    off_diag_avg_learn = np.mean(np.abs(corr_learn - np.eye(corr_learn.shape[0])))

    # Histograms of fidelity and orth energies (combined for flow models)
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    axs[0, 0].plot(base_hist, label="Base test MSE")
    axs[0, 0].plot(flow_comp_hist, label="FlowComp test MSE")
    axs[0, 0].plot(flow_learn_hist, label="FlowLearn test MSE")
    axs[0, 0].set_title("Test MSE Trajectories")
    axs[0, 0].legend()

    axs[0, 1].plot(sv_b, "o-", label="Base SVs")
    axs[0, 1].plot(sv_fc, "x-", label="FlowComp SVs")
    axs[0, 1].plot(sv_fl, "s-", label="FlowLearn SVs")
    axs[0, 1].set_title("Singular Values")
    axs[0, 1].legend()

    axs[1, 0].hist(flow_comp_Efid_hist, bins=20, alpha=0.5, label="Comp Fidelity")
    axs[1, 0].hist(flow_comp_Eorth_hist, bins=20, alpha=0.5, label="Comp Orth")
    axs[1, 0].hist(flow_learn_Efid_hist, bins=20, alpha=0.5, label="Learn Fidelity")
    axs[1, 0].hist(flow_learn_Eorth_hist, bins=20, alpha=0.5, label="Learn Orth")
    axs[1, 0].set_title("Per-Batch Energy Distributions")
    axs[1, 0].legend()

    axs[1, 1].bar(["Base", "Comp", "Learn"], [procrustes_err_base, procrustes_err_comp, procrustes_err_learn])
    axs[1, 1].set_title("Procrustes Alignment Error (↓ better)")

    axs[0, 2].bar(["Base", "Comp", "Learn"], [ortho_def_base, ortho_def_comp, ortho_def_learn])
    axs[0, 2].set_title("Orthogonality Defect (↓ better)")

    axs[1, 2].bar(["Base", "Comp", "Learn"], [off_diag_avg_base, off_diag_avg_comp, off_diag_avg_learn])
    axs[1, 2].set_title("Avg Off-Diag Corr (↓ better)")

    plt.tight_layout()
    plt.show()

    # Summary Table
    print("\n📊 Summary ------------------------------------------------------")
    print(f"Final Test Loss — Base: {base_hist[-1]:.6f} | Comp: {flow_comp_hist[-1]:.6f} | Learn: {flow_learn_hist[-1]:.6f}")
    print(f"Procrustes Error — Base: {procrustes_err_base:.6f} | Comp: {procrustes_err_comp:.6f} | Learn: {procrustes_err_learn:.6f}")
    print(f"Orthogonality Defect — Base: {ortho_def_base:.6f} | Comp: {ortho_def_comp:.6f} | Learn: {ortho_def_learn:.6f}")
    print(f"Final w_retract — Comp: {flow_comp_model.flow.w_retract.item():.6f} | Learn: {flow_learn_model.flow.w_retract.item():.6f}")
    print(f"Final tradeoff_alpha — Comp: {torch.sigmoid(flow_comp_model.flow.tradeoff_alpha).item():.6f} | Learn: {torch.sigmoid(flow_learn_model.flow.tradeoff_alpha).item():.6f}")
    print(f"Average Off-Diagonal Correlation (Embeddings) — Base: {off_diag_avg_base:.6f} | Comp: {off_diag_avg_comp:.6f} | Learn: {off_diag_avg_learn:.6f}")
    print(f"Base Correlation Matrix:\n{corr_base}")
    print(f"Comp Correlation Matrix:\n{corr_comp}")
    print(f"Learn Correlation Matrix:\n{corr_learn}")
    print(f"Median Fidelity Energy — Comp: {np.nanmedian(flow_comp_Efid_hist):.4e} | Learn: {np.nanmedian(flow_learn_Efid_hist):.4e}")
    print(f"Median Orthogonality Energy — Comp: {np.nanmedian(flow_comp_Eorth_hist):.4e} | Learn: {np.nanmedian(flow_learn_Eorth_hist):.4e}")
    print("---------------------------------------------------------------\n")

    return dict(
        base_test_hist=base_hist,
        flow_comp_test_hist=flow_comp_hist,
        flow_learn_test_hist=flow_learn_hist,
        flow_comp_Efid_hist=flow_comp_Efid_hist,
        flow_comp_Eorth_hist=flow_comp_Eorth_hist,
        flow_learn_Efid_hist=flow_learn_Efid_hist,
        flow_learn_Eorth_hist=flow_learn_Eorth_hist,
        procrustes_base=procrustes_err_base,
        procrustes_comp=procrustes_err_comp,
        procrustes_learn=procrustes_err_learn,
    )
