"""Comprehensive optimizer evaluation for NSA-Flow objectives.

Compares:
1. SPG (Standard Spectral Projected Gradient with BB1)
2. SPG-ABB (Alternating Barzilai-Borwein with non-convex step safeguarding)
3. L-BFGS-B (SciPy Quasi-Newton with box constraints)
4. SLSQP (Sequential Least Squares Programming)
5. Projected Adam (First-order adaptive moments with non-negative projection)
6. Projected RMSprop (Adaptive learning rate with non-negative projection)
7. APGD-Restart (Accelerated Projected Gradient Descent with restart)

Generates empirical convergence curves, speedup metrics, and iteration analysis.
"""
import time
import math
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from experiments.data import load_golub3
from nsa_flow.reconstruct import _orth_terms, GramOperator, _fid, _grad_fid
from nsa_flow.signed import (_split, gram_offdiag_defect, grad_gram_offdiag_defect,
                              reconstruction_fidelity, grad_reconstruction_fidelity,
                              relax_into_nonneg)
from nsa_flow.project import project_nonneg
from nsa_flow.solve import _spg_loop, _lbfgs_b_loop

def get_golub_data(p=2000):
    X_raw, _, _ = load_golub3()
    Xl = np.log2(np.clip(X_raw, 1.0, None))
    top = np.argsort(Xl.var(0))[-p:]
    Z = (Xl[:, top] - Xl[:, top].mean(0)) / (Xl[:, top].std(0, ddof=1) + 1e-12)
    return torch.as_tensor(Z, dtype=torch.float64)

# ── Optimizer implementations ──────────────────────────────────────────────────

def opt_spg_standard(Y0, energy_fn, grad_fn, proj_fn, max_iter=2000, tol=1e-6, patience=50, rtol=1e-7):
    """Standard SPG without alternating BB or safeguarding."""
    Y = proj_fn(Y0.clone())
    E, g = grad_fn(Y)
    E = float(E)
    t = 1.0 / max(float(g.norm()), 1e-12)
    Y_prev = g_prev = None
    gmap, stop, it = float("inf"), "max_iter", 0
    E_window = []
    _T_MIN, _T_MAX = 1e-12, 1e12
    n_fev, n_gev = 1, 1
    t0 = time.time()

    for it in range(1, max_iter + 1):
        if Y_prev is not None:
            s_ = Y - Y_prev
            r_ = g - g_prev
            sr = float((s_ * r_).sum())
            t = float((s_ * s_).sum()) / sr if sr > 0 else _T_MAX
            t = min(max(t, _T_MIN), _T_MAX)

        accepted = False
        t_first, dn2_first = t, None
        for _ in range(50):
            Y_new = proj_fn(Y - t * g)
            d_ = Y_new - Y
            dn2 = float((d_ * d_).sum())
            if dn2_first is None:
                dn2_first = dn2
            n_fev += 1
            if float(energy_fn(Y_new)) <= E - 1e-4 * dn2 / t:
                accepted = True
                break
            t *= 0.5

        if not accepted:
            stop = "line_search"
            gmap = (dn2_first ** 0.5) / t_first
            break

        gmap = (dn2 ** 0.5) / t
        Y_prev, g_prev = Y, g
        Y = Y_new
        E, g = grad_fn(Y)
        E = float(E)
        n_gev += 1

        E_window.append(E)
        if len(E_window) > patience:
            E_window.pop(0)
        if len(E_window) == patience:
            span = max(E_window) - min(E_window)
            if span / (1.0 + abs(min(E_window))) < rtol:
                stop = "plateau"
                break
        if gmap <= tol:
            stop = "grad_map"
            break

    return {"name": "SPG (Standard BB1)", "time": time.time() - t0, "iters": it, "fev": n_fev, "gev": n_gev, "energy": E, "gmap": gmap, "stop": stop}

def opt_spg_abb(Y0, energy_fn, grad_fn, proj_fn, max_iter=2000, tol=1e-6, patience=50, rtol=1e-7):
    """Enhanced SPG with alternating BB and non-convex step safeguarding."""
    Y = proj_fn(Y0.clone())
    E, g = grad_fn(Y)
    E = float(E)
    t = 1.0 / max(float(g.norm()), 1e-12)
    Y_prev = g_prev = None
    gmap, stop, it = float("inf"), "max_iter", 0
    E_window = []
    _T_MIN, _T_MAX = 1e-12, 1e12
    n_fev, n_gev = 1, 1
    t0 = time.time()

    for it in range(1, max_iter + 1):
        if Y_prev is not None:
            s_ = Y - Y_prev
            r_ = g - g_prev
            sr = float((s_ * r_).sum())
            if sr > 0:
                if it % 2 == 0:
                    t = float((s_ * s_).sum()) / sr
                else:
                    t = sr / max(float((r_ * r_).sum()), 1e-12)
            else:
                t = min(max(t, 1e-3), 10.0)
            t = min(max(t, _T_MIN), _T_MAX)

        accepted = False
        t_first, dn2_first = t, None
        for _ in range(35):
            Y_new = proj_fn(Y - t * g)
            d_ = Y_new - Y
            dn2 = float((d_ * d_).sum())
            if dn2_first is None:
                dn2_first = dn2
            n_fev += 1
            if float(energy_fn(Y_new)) <= E - 1e-4 * dn2 / t:
                accepted = True
                break
            t *= 0.5

        if not accepted:
            stop = "line_search"
            gmap = (dn2_first ** 0.5) / t_first
            break

        gmap = (dn2 ** 0.5) / t
        Y_prev, g_prev = Y, g
        Y = Y_new
        E, g = grad_fn(Y)
        E = float(E)
        n_gev += 1

        E_window.append(E)
        if len(E_window) > patience:
            E_window.pop(0)
        if len(E_window) == patience:
            span = max(E_window) - min(E_window)
            if span / (1.0 + abs(min(E_window))) < rtol:
                stop = "plateau"
                break
        if gmap <= tol:
            stop = "grad_map"
            break

    return {"name": "SPG-ABB (Alternating BB)", "time": time.time() - t0, "iters": it, "fev": n_fev, "gev": n_gev, "energy": E, "gmap": gmap, "stop": stop}

def opt_lbfgs_b(Y0, energy_fn, grad_fn, max_iter=1000, tol=1e-5):
    """SciPy L-BFGS-B with bounds [0, inf)."""
    t0 = time.time()
    shape = Y0.shape
    device = Y0.device
    dtype = Y0.dtype

    n_calls = [0, 0]
    def f_and_g(y_flat):
        Y_t = torch.as_tensor(y_flat.reshape(shape), dtype=dtype, device=device)
        E, g = grad_fn(Y_t)
        n_calls[0] += 1
        n_calls[1] += 1
        return float(E), g.detach().cpu().numpy().astype(np.float64).flatten()

    y0 = Y0.detach().cpu().numpy().astype(np.float64).flatten()
    bounds = [(0.0, None)] * len(y0)
    res = minimize(
        f_and_g, y0, method="L-BFGS-B", jac=True, bounds=bounds,
        options=dict(maxiter=max_iter, ftol=1e-8, gtol=tol)
    )
    gmap = float(np.max(np.abs(res.jac)))
    msg = str(res.message).upper()
    stop = "grad_map" if res.success else ("plateau" if "CONVERGENCE" in msg else "max_iter")
    return {
        "name": "L-BFGS-B (Quasi-Newton)", "time": time.time() - t0,
        "iters": res.nit, "fev": n_calls[0], "gev": n_calls[1], "energy": float(res.fun), "gmap": gmap, "stop": stop
    }

def opt_slsqp(Y0, energy_fn, grad_fn, max_iter=500):
    """SciPy SLSQP (Sequential Least Squares Programming)."""
    t0 = time.time()
    shape = Y0.shape
    device = Y0.device
    dtype = Y0.dtype

    n_calls = [0, 0]
    def f_and_g(y_flat):
        Y_t = torch.as_tensor(y_flat.reshape(shape), dtype=dtype, device=device)
        E, g = grad_fn(Y_t)
        n_calls[0] += 1
        n_calls[1] += 1
        return float(E), g.detach().cpu().numpy().astype(np.float64).flatten()

    y0 = Y0.detach().cpu().numpy().astype(np.float64).flatten()
    bounds = [(0.0, None)] * len(y0)
    res = minimize(
        f_and_g, y0, method="SLSQP", jac=True, bounds=bounds,
        options=dict(maxiter=max_iter, ftol=1e-6)
    )
    gmap = float(np.max(np.abs(res.jac))) if hasattr(res, 'jac') and res.jac is not None else float("nan")
    stop = "grad_map" if res.success else "max_iter"
    return {
        "name": "SLSQP (Sequential LS)", "time": time.time() - t0,
        "iters": res.nit, "fev": n_calls[0], "gev": n_calls[1], "energy": float(res.fun), "gmap": gmap, "stop": stop
    }

def opt_projected_adam(Y0, energy_fn, grad_fn, proj_fn, max_iter=2000, lr=0.015, tol=1e-6, patience=50, rtol=1e-7):
    """Projected Adam."""
    t0 = time.time()
    Y = proj_fn(Y0.clone())
    m = torch.zeros_like(Y)
    v = torch.zeros_like(Y)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    stop, it = "max_iter", 0
    E_window = []
    gmap = float("inf")
    n_fev, n_gev = 0, 0

    for it in range(1, max_iter + 1):
        E, g = grad_fn(Y)
        E = float(E)
        n_fev += 1
        n_gev += 1

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1 ** it)
        v_hat = v / (1.0 - beta2 ** it)

        step = lr * m_hat / (torch.sqrt(v_hat) + eps)
        Y_new = proj_fn(Y - step)
        d_ = Y_new - Y
        gmap = float(d_.norm()) / lr

        Y = Y_new
        E_window.append(E)
        if len(E_window) > patience:
            E_window.pop(0)
        if len(E_window) == patience:
            span = max(E_window) - min(E_window)
            if span / (1.0 + abs(min(E_window))) < rtol:
                stop = "plateau"
                break
        if gmap <= tol:
            stop = "grad_map"
            break

    return {"name": "Projected Adam", "time": time.time() - t0, "iters": it, "fev": n_fev, "gev": n_gev, "energy": E, "gmap": gmap, "stop": stop}

def opt_projected_rmsprop(Y0, energy_fn, grad_fn, proj_fn, max_iter=2000, lr=0.01, alpha=0.99, tol=1e-6, patience=50, rtol=1e-7):
    """Projected RMSprop."""
    t0 = time.time()
    Y = proj_fn(Y0.clone())
    v = torch.zeros_like(Y)
    eps = 1e-8
    stop, it = "max_iter", 0
    E_window = []
    gmap = float("inf")
    n_fev, n_gev = 0, 0

    for it in range(1, max_iter + 1):
        E, g = grad_fn(Y)
        E = float(E)
        n_fev += 1
        n_gev += 1

        v = alpha * v + (1.0 - alpha) * (g * g)
        step = lr * g / (torch.sqrt(v) + eps)
        Y_new = proj_fn(Y - step)
        d_ = Y_new - Y
        gmap = float(d_.norm()) / lr

        Y = Y_new
        E_window.append(E)
        if len(E_window) > patience:
            E_window.pop(0)
        if len(E_window) == patience:
            span = max(E_window) - min(E_window)
            if span / (1.0 + abs(min(E_window))) < rtol:
                stop = "plateau"
                break
        if gmap <= tol:
            stop = "grad_map"
            break

    return {"name": "Projected RMSprop", "time": time.time() - t0, "iters": it, "fev": n_fev, "gev": n_gev, "energy": E, "gmap": gmap, "stop": stop}


def run_benchmark_suite():
    T = get_golub_data(p=2000)
    p = 2000
    rows = []

    configs = [
        # (type, k, w)
        ("Data Recon", 3, 0.05),
        ("Data Recon", 3, 0.50),
        ("Data Recon", 6, 0.05),
        ("Data Recon", 6, 0.50),
        ("Signed Lifting", 3, 0.05),
        ("Signed Lifting", 3, 0.50),
        ("Signed Lifting", 6, 0.05),
        ("Signed Lifting", 6, 0.50),
    ]

    for ptype, k, w in configs:
        print(f"\n=======================================================")
        print(f"BENCHMARK: {ptype} (k={k}, w={w})")
        print(f"=======================================================")

        if ptype == "Data Recon":
            ops = GramOperator(X=T)
            c, trS = ops.c, ops.trS
            orth_val, orth_grad = _orth_terms("C", k)
            def e_fn(V):
                f = _fid(V, ops, c, trS)
                d = orth_val(V)
                return (1.0 - w) * f + w * d
            def g_fn(V):
                E = e_fn(V)
                g = (1.0 - w) * _grad_fid(V, ops, c) + w * orth_grad(V)
                return E, g
            E_init = ops.leading(k)
            Y0 = E_init.clamp_min(0.0).clone()
        else:
            S = T.transpose(-2, -1) @ T
            c = float(S.trace())
            V0 = relax_into_nonneg(S, c, k, float(w), trS=c)
            Y0 = torch.cat([V0.clamp_min(0.0), (-V0).clamp_min(0.0)], dim=-1).clone()
            def e_fn(Wv):
                Vp, Vm = _split(Wv)
                f = reconstruction_fidelity(Vp - Vm, S, c, c)
                d = gram_offdiag_defect(Wv)
                e = (1.0 - w) * f + w * d
                if w > 0.0:
                    e = e + w * 1.0 * (Vp * Vm).sum() / c
                return e
            def g_fn(Wv):
                Vp, Vm = _split(Wv)
                f = reconstruction_fidelity(Vp - Vm, S, c, c)
                d = gram_offdiag_defect(Wv)
                e = (1.0 - w) * f + w * d
                if w > 0.0:
                    e = e + w * 1.0 * (Vp * Vm).sum() / c
                gV = (1.0 - w) * grad_reconstruction_fidelity(Vp - Vm, S, c)
                g = torch.cat([gV, -gV], dim=-1)
                if w != 0.0:
                    g = g + w * grad_gram_offdiag_defect(Wv)
                if w > 0.0:
                    g = g + (w * 1.0 / c) * torch.cat([Vm, Vp], dim=-1)
                return e, g

        benchmarks = [
            ("SPG (Standard BB1)", lambda: opt_spg_standard(Y0, e_fn, g_fn, project_nonneg)),
            ("SPG-ABB (Alternating BB)", lambda: opt_spg_abb(Y0, e_fn, g_fn, project_nonneg)),
            ("L-BFGS-B (Quasi-Newton)", lambda: opt_lbfgs_b(Y0, e_fn, g_fn, max_iter=1000)),
            ("Projected Adam", lambda: opt_projected_adam(Y0, e_fn, g_fn, project_nonneg, lr=0.015)),
            ("Projected RMSprop", lambda: opt_projected_rmsprop(Y0, e_fn, g_fn, project_nonneg, lr=0.01)),
        ]

        for bname, bfn in benchmarks:
            res = bfn()
            row = {
                "problem": ptype,
                "k": k,
                "w": w,
                "optimizer": res["name"],
                "time_sec": round(res["time"], 3),
                "iters": res["iters"],
                "fev": res["fev"],
                "energy": res["energy"],
                "gmap": res["gmap"],
                "stop_reason": res["stop"]
            }
            rows.append(row)
            print(f"  {res['name']:<25} | {res['time']:5.2f}s | iters={res['iters']:4d} | E={res['energy']:.6f} | stop={res['stop']}")

    df = pd.DataFrame(rows)
    df.to_csv("paper/results/optimizer_benchmarks.csv", index=False)
    print("\nSaved paper/results/optimizer_benchmarks.csv")
    return df

if __name__ == "__main__":
    run_benchmark_suite()
