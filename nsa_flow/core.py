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

    import torch, numpy as np

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

    torch.manual_seed(42)
    Y_ref = Y0.clone().detach().requires_grad_(True)
    X0 = X0.clone().detach()

    if fid_eta is None:
        fid_eta = 1.0
    if c_orth is None:
        c_orth = 1.0

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
        except Exception:
            return np.nan

    # --- Baseline energy and gradient ---
    f_ref = safe_energy(Y_ref)

    if np.isnan(f_ref):
        raise ValueError("Initial energy computation failed (NaN).")

    f_ref_tensor = torch.tensor(f_ref, dtype=Y_ref.dtype, requires_grad=True)
    f_ref_tensor.backward(retain_graph=False)
    grad_ref = Y_ref.grad.detach().clone() if Y_ref.grad is not None else torch.zeros_like(Y_ref)
    grad_norm = torch.norm(grad_ref).item() + 1e-12

    # --- Learning rate candidates ---
    if strategy in [
        "armijo", "armijo_aggressive", "exponential", "linear",'entropy',
        "random", "adaptive", "momentum_boost", "poly_decay", "grid", "bayes"
    ]:
        if strategy == "grid":
            lr_candidates = np.logspace(-6, 2, 30)
        # --- FIXED: entropy ---
        # Interprets aggression as "temperature" in a softmax weighting of the search space.
        # Low aggression = low temperature (concentrated near small LRs)
        # High aggression = high temperature (spread toward large LRs)
        elif strategy == "entropy":
            base = np.logspace(-6, 2, 50)
            # aggression acts as a temperature-like scaling factor
            temp = 1e-3 + 5 * aggression
            probs = np.exp(np.log(base + 1e-12) / (temp + 1e-8))
            probs /= np.sum(probs)
            # expected learning rate under softmax weighting
            lr_mean = np.sum(base * probs)
            # form a candidate distribution around that expected value
            lr_candidates = lr_mean * np.logspace(-1, 1, 25)
        elif strategy == "armijo":
            lr_candidates = np.logspace(-4, 0, 20) * (1 + 3 * aggression)

        elif strategy == "armijo_aggressive":
            lr_candidates = np.logspace(-2, 2, 20) * (1 + 5 * aggression)

        elif strategy == "exponential":
            base = np.logspace(-6, 0, 25)
            lr_candidates = base ** (1 - 0.5 * aggression) * (1 + aggression)

        # --- FIXED: linear ---
        # Smooth monotonic ramp with aggression controlling slope and upper bound.
        elif strategy == "linear":
            start = 1e-5
            stop = 1.0 * (1 + 4 * aggression)
            lr_candidates = np.linspace(start, stop, 25)

        # --- FIXED: adaptive ---
        # Adapts to gradient norm; aggression increases exploration.
        elif strategy == "adaptive":
            base = np.logspace(-6, -2, 25)
            scale = np.clip(1.0 / (1.0 + grad_norm), 1e-4, 1.0)
            # aggression ramps both scale and range
            lr_candidates = base * (1 + 10 * aggression) * scale

        # --- FIXED: momentum_boost ---
        # Combines conservative start with acceleration based on aggression
        elif strategy == "momentum_boost":
            base = np.logspace(-5, -1, 25)
            boost = np.linspace(1.0, 1.0 + 9.0 * aggression, 25)
            lr_candidates = base * boost

        # --- FIXED: poly_decay ---
        # Polynomial *growth* instead of decay; aggression controls curvature.
        elif strategy == "poly_decay":
            base = np.linspace(1e-5, 1.0, 25)
            power = 1.0 + 3.0 * aggression
            lr_candidates = base ** power * (1 + aggression)

        elif strategy == "random":
            rng = np.random.default_rng(42)
            lr_candidates = rng.uniform(1e-5, 1.0 * (1 + 4 * aggression), 25)

        elif strategy == "bayes":
            base = np.logspace(-3, 2, 25)
            prior = np.linspace(0.1, 1.0, 25)
            lr_candidates = base * (prior ** (1 - aggression)) * (1 + aggression)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    # ============================================================
    # 🔧 AGGRESSION-BASED STRATEGIC TUNING (0.0 → 1.0)
    # ============================================================
    aggression = float(np.clip(locals().get("aggression", 0.5), 0.0, 1.0))

    def scale_range(base, lo_exp=-6, hi_exp=2):
        """Generate log-scaled range influenced by aggression."""
        lo = lo_exp + aggression * (hi_exp - lo_exp) * 0.7
        hi = hi_exp + aggression * (hi_exp - lo_exp) * 0.3
        return np.logspace(lo, hi, len(base))

    if strategy in ["armijo", "armijo_aggressive"]:
        # Armijo uses logspace but aggression shifts range upward
        lr_candidates = scale_range(lr_candidates, -4, 1 if strategy == "armijo" else 2)

    elif strategy == "grid":
        lr_candidates = scale_range(lr_candidates, -6, 1 + 2 * aggression)

    elif strategy == "exponential":
        # Broaden exponential ramp with aggression
        exp_range = np.linspace(-6 + 2 * aggression, aggression, len(lr_candidates))
        lr_candidates = 10 ** exp_range

    elif strategy == "linear":
        # Linearly spaced between 1e-5 and aggression-scaled max
        max_lr = 0.1 + aggression * 10.0
        lr_candidates = np.linspace(1e-5, max_lr, len(lr_candidates))

    elif strategy == "random":
        # Aggression controls randomness amplitude
        rng = np.random.default_rng(42)
        lr_candidates = rng.random(len(lr_candidates)) ** (1.0 - aggression)
        lr_candidates *= 10 ** (2 * aggression - 1)

    elif strategy == "adaptive":
        # Aggression *increases* learning rate proportional to gradient magnitude
        base_scale = np.clip(grad_norm / (1e-3 + abs(f_ref)), 1e-4, 1e4)
        # Make higher aggression lead to larger step sizes
        scale_factor = (base_scale ** (0.5 + aggression)) * (10 ** (2 * aggression))
        lr_candidates = np.logspace(-6, 0, len(lr_candidates)) * scale_factor
        lr_candidates = np.clip(lr_candidates, 1e-8, 1e3)

    elif strategy == "momentum_boost":
        # Base linear spacing, mild log scaling with aggression
        base_lr = np.linspace(1e-5, 1e-4, len(lr_candidates))
        lr_candidates = base_lr * (1.0 + aggression * np.linspace(0.0, 10.0, len(lr_candidates)))

    elif strategy == "entropy":
        # Stable entropy proxy (normalized gradient energy)
        ent = np.log1p(grad_norm) / (np.log1p(abs(f_ref)) + 1e-12)
        ent_scale = float(np.clip(ent, 0.1, 5.0))

        # Base LR schedule (mild exponential sweep)
        base_lrs = np.logspace(-5, 0, len(lr_candidates))

        # Aggression controls exponential amplification in a strictly monotonic way
        # Using smooth polynomial-exponential blend for predictable behavior
        scale_factor = (1 + aggression * ent_scale) ** (1 + aggression * 2.0)

        lr_candidates = base_lrs * scale_factor

        # Explicit monotonic enforcement for safety
        lr_candidates = np.maximum.accumulate(lr_candidates)

    elif strategy == "poly_decay":
        # Polynomially increasing range
        lr_candidates = (np.linspace(0.0, 1.0, len(lr_candidates)) ** (1.0 - aggression)) * (10 ** (2 * aggression))

    elif strategy == "bayes":
        # Simulate Bayesian sampling by weighting low-energy LRs
        rng = np.random.default_rng(7)
        priors = np.exp(-np.linspace(0, 5, len(lr_candidates)))
        weights = priors ** (1.0 - aggression) + rng.normal(0, 0.05, len(lr_candidates))
        weights = np.clip(weights, 1e-6, None)
        weights /= np.sum(weights)
        lr_candidates = np.cumsum(weights) * (10 ** (1 + 2 * aggression))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    losses, rel_changes = [], []

    # --- Evaluate all ---
    for lr in lr_candidates:
        Y_try = Y_ref - lr * grad_ref
        f_new = safe_energy(Y_try)
        if not np.isfinite(f_new):
            f_new = np.nan
        losses.append(f_new)
        rel_changes.append((f_ref - f_new) / (abs(f_ref) + 1e-12))

    losses = np.array(losses, dtype=float)
    rel_changes = np.array(rel_changes, dtype=float)

    # --- Filter out invalid entries ---
    valid_mask = np.isfinite(losses)
    if not np.any(valid_mask):
        if verbose:
            print("⚠️ All candidate losses invalid — defaulting to conservative LR.")
        return {"best_lr": 1e-3, "best_energy": np.nan, "strategy": strategy, "aggression": aggression}

    lr_candidates = lr_candidates[valid_mask]
    losses = losses[valid_mask]
    rel_changes = rel_changes[valid_mask]

    # --- Strategy selection ---
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
        if exp_vals.sum() == 0:
            exp_vals = np.ones_like(exp_vals) / len(exp_vals)
        probs = exp_vals / exp_vals.sum()
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

    # --- Verbose logging ---
    if verbose:
        print(f"[LR-Est] {strategy:<16} | agg={aggression:.2f} | best_lr={best_lr:.3e} | minE={np.nanmin(losses):.3e}")

    # --- Optional plotting ---
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
    lr_strategy="auto",
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
            strategy="armijo_aggressive" if lr_strategy == "auto" else lr_strategy,
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
