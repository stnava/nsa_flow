import time
import numpy as np
import torch
import pandas as pd

from .utils import safe_to_tensor, apply_nonnegativity, traces_to_dataframe
from .energy import compute_energy, defect_fast
from .retraction import nsa_flow_retract_auto
from .optimizer import get_torch_optimizer, get_lr_estimation_strategies, estimate_learning_rate_for_nsa_flow

def nsa_flow(*args, **kwargs):
    return nsa_flow_orth(*args, **kwargs)

def nsa_flow_autograd(*args, **kwargs):
    return nsa_flow_orth(*args, **kwargs)

def nsa_flow_orth(
    Y0, X0=None, w=0.5,
    retraction="soft_polar",
    max_iter=500, tol=1e-4, verbose=False, seed=42,
    apply_nonneg='hard', optimizer="asgd",
    initial_learning_rate=None, 
    lr_strategy="bayes",
    fidelity_type="scale_invariant",
    orth_type="scale_invariant",
    aggression=0.5,
    record_every=1, 
    window_size=10,
    warmup_iters = 0,
    device=None, 
    precision="float64",
    lr_scheduler=False,
    lr_scheduler_patience=10,
    lr_scheduler_factor=0.5,
):
    """
    Autograd-compatible NSA-Flow (modular energy version).
    Allows user-specified fidelity_type and orth_type.
    """
    if precision == "float32":
        dtype = torch.float32
    elif precision == "float64":
        dtype = torch.float64
    else:
        raise ValueError("precision must be 'float32' or 'float64'")

    Y0 = safe_to_tensor(Y0, dtype=torch.float64, name="Y0")
    if X0 is not None:
        X0 = safe_to_tensor(X0, dtype=torch.float64, name="X0")

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

    # --- Default scaling constants (with safety clamps) ---
    g0 = 0.5 * torch.sum((Y - X0) ** 2) / (p * k)
    g0 = torch.clamp(g0, min=1e-8)
    d0 = torch.clamp(defect_fast(Y), min=1e-8)
    fid_eta = min(float((1 - w) / (g0 * p * k)), 1e6)
    c_orth = min(float(4 * w / d0), 1e6)

    # --- Optimization variable ---
    Z = torch.nn.Parameter(Y.clone().detach().requires_grad_(True))
    lr = 1.0
    w_use = w
    if warmup_iters > 0:
        w_use = w
        Z = torch.nn.Parameter(Y.clone().detach().requires_grad_(True))
        opt = get_torch_optimizer(optimizer, [Z], lr=lr)

        #  WARMUP PHASE: Estimate relative scaling between fidelity & orth terms
        fid_eta = 1.0
        c_orth = 1.0
        for _round in range(1):
            warmup_fids, warmup_orths = [], []
            if warmup_iters > 0:
                if verbose:
                    print(f"\n=== Warmup ({warmup_iters} iters) to estimate weights ===")
                    print(f" init fid_eta={fid_eta:.3e}, c_orth={c_orth:.3e}, lr={lr:.3e}")

                for it in range(1, warmup_iters + 1):
                    opt.zero_grad()
                    Y_retracted = nsa_flow_retract_auto(Z, w_use, retraction)
                    Y_retracted = apply_nonnegativity(Y_retracted, apply_nonneg)
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

                    # Evaluate & store progress
                    with torch.no_grad():
                        Y_eval = nsa_flow_retract_auto(Z, w_use, retraction)
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

                    warmup_fids.append(E["fidelity"])
                    warmup_orths.append(E["orthogonality"])
                    if verbose and (it % max(1, warmup_iters // 5) == 0 or it == warmup_iters):
                        print(f"[Warmup {it:3d}] Fid={E['fidelity']:.3e} Orth={E['orthogonality']:.3e}")

                mean_fid = float(np.mean(warmup_fids) + 1e-12)
                mean_orth = float(np.mean(warmup_orths) + 1e-12)

                if mean_fid <= 0 or mean_orth <= 0:
                    if verbose:
                        print("[Warmup] non-positive mean encountered, skipping reweight round.")
                    continue

                fid_eta = 5e-4 / mean_fid
                c_orth = 1.0 / mean_orth
                if verbose:
                    print(f"\n[Reweighting learned]")
                    print(f" mean_fid={mean_fid:.3e}, mean_orth={mean_orth:.3e}")
                    print(f" → new fid_eta={fid_eta:.3e}, c_orth={c_orth:.3e}")
                    print(f"=== Restarting optimization with new weights ===\n")

                Z = torch.nn.Parameter(Y.clone().detach().requires_grad_(True))
                opt = get_torch_optimizer(optimizer, [Z], lr=lr)

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

    Z = torch.nn.Parameter(Y.clone().detach().requires_grad_(True))
    opt = get_torch_optimizer(optimizer, [Z], lr=lr)

    # --- LR Scheduler ---
    scheduler = None
    if lr_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            opt, 
            mode='min', 
            factor=lr_scheduler_factor, 
            patience=lr_scheduler_patience, 
            threshold=1e-6
        )

    # --- Tracking and monitoring ---
    traces, recent_iters, recent_totals, recent_energies = [], [], [], []
    t0 = time.time()
    best_Y, best_energy, best_iter = None, float("inf"), 0

    # --- Main optimization loop ---
    for it in range(1, max_iter + 1):
        opt.zero_grad()

        # --- Retraction + optional nonnegativity ---
        Y_retracted = nsa_flow_retract_auto(Z, w_use, retraction)
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
        # --- Backward pass with gradient clipping ---
        total_energy.backward()
        torch.nn.utils.clip_grad_norm_([Z], max_norm=10.0)
        opt.step()

        # --- NaN/Inf detection ---
        if not torch.isfinite(total_energy):
            if verbose:
                print(f"Non-finite energy at iter {it}; reverting to best.")
            if best_Y is not None:
                Z.data.copy_(best_Y / scale_ref)
            break

        # --- Evaluate & store progress ---
        with torch.no_grad():
            Y_eval = nsa_flow_retract_auto(Z, w_use, retraction)
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

            total_val = float(E["total"].item())
            fidelity_val = float(E["fidelity"].item())
            orth_val = float(E["orthogonality"].item())

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

            # Step the scheduler if enabled
            if scheduler is not None:
                scheduler.step(total_val)

            # --- Convergence criterion ---
            recent_iters.append(it)
            recent_totals.append(total_val)
            if len(recent_iters) > window_size:
                recent_iters.pop(0)
                recent_totals.pop(0)

            if len(recent_iters) >= (window_size-1):
                xs = np.array(recent_iters, dtype=float)
                ys = np.array(recent_totals, dtype=float)
                xm = xs.mean()
                ym = ys.mean()
                denom = np.sum((xs - xm) ** 2)
                if denom > 0:
                    slope = float(np.sum((xs - xm) * (ys - ym)) / denom)
                else:
                    slope = 0.0

                if verbose and (it % max(1, record_every) == 0):
                    print(f"[Iter {it:4d}] Total={total_val:.6e} | Fid={fidelity_val:.6e} | Orth={orth_val:.6e} | slope={slope:.3e}")

                if slope > -tol:
                    if verbose:
                        print(f"Converged at iter {it} (slope {slope:.3e} > -{tol:.3e}).")
                    break

    traces = traces_to_dataframe(traces)
    return {
        "Y": best_Y * scale_ref,
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
