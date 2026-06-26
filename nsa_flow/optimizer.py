import warnings
import numpy as np
import torch

from .retraction import nsa_flow_retract_auto
from .utils import apply_nonnegativity
from .energy import compute_energy

def get_torch_optimizer(opt_name: str = None, params=None, lr: float = 1e-3, return_list: bool = False, **kwargs):
    """
    Returns a PyTorch optimizer instance based on the provided name,
    or lists all available optimizers if return_list=True.
    """
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
        pass

    if return_list:
        return sorted(optimizers.keys())

    if opt_name is None:
        raise ValueError("You must specify `opt_name` unless `return_list=True`.")

    name = opt_name.lower()
    if name not in optimizers:
        raise ValueError(
            f"Unsupported optimizer: '{opt_name}'. "
            f"Supported options: {sorted(optimizers.keys())}"
        )

    return optimizers[name](params, lr)

def get_lr_estimation_strategies():
    return [
        "armijo", "armijo_aggressive", "exponential", "linear", "entropy",
        "random", "adaptive", "momentum_boost", "poly_decay", "grid", "bayes"
    ]

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
    np.random.seed(42)

    Y_ref = Y0.clone().detach().requires_grad_(True)
    X0 = X0.clone().detach()

    fid_eta = 1.0 if fid_eta is None else fid_eta
    c_orth = 1.0 if c_orth is None else c_orth
    aggression = float(np.clip(aggression, 0.0, 1.0))

    def safe_energy(Y):
        try:
            with torch.no_grad():
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

    try:
        Yr_ref = nsa_flow_retract_auto(Y_ref, w, retraction)
        Yr_ref = apply_nonnegativity(Yr_ref, apply_nonneg)
        f_ref_tensor = compute_energy(
            Yr_ref,
            X0,
            w=w,
            fid_eta=fid_eta,
            c_orth=c_orth,
            fidelity_type=fidelity_type,
            orth_type=orth_type,
            track_grad=True,
        )
        f_ref = _to_float(f_ref_tensor)
    except Exception as ex:
        if verbose:
            warnings.warn(f"Initial energy computation failed: {ex}")
        raise ValueError("Initial energy computation failed (NaN).")

    if not np.isfinite(f_ref):
        raise ValueError("Initial energy computation failed (NaN).")

    grad_ref = torch.autograd.grad(
        f_ref_tensor, Y_ref, create_graph=False, retain_graph=False
    )[0].detach().clone()
    grad_norm = torch.norm(grad_ref).item() + 1e-12

    strategies = get_lr_estimation_strategies()
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}")

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

    lr_candidates = np.clip(lr_candidates, 1e-8, 1e3)
    lr_candidates = np.unique(np.sort(lr_candidates))

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
