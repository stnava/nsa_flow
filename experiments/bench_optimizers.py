"""Benchmark optimizers for NSA-Flow objectives.

Tests:
1. SPG (current: monotone Armijo + BB1)
2. SPG-ABB (Alternating BB: BB1/BB2 with non-monotone line search)
3. APGD / FISTA (Nesterov accelerated projected gradient with adaptive restart)
4. L-BFGS-B (Scipy bounded quasi-Newton)
5. Projected Adam (Adaptive learning rate + non-negative projection)

Evaluated on Golub dataset (p=2000, n=72) at k=6 for data and signed objectives.
"""
import time
import math
import numpy as np
import torch
from scipy.optimize import minimize
from experiments.data import load_golub3
from nsa_flow.reconstruct import _orth_terms, GramOperator, _fid, _grad_fid
from nsa_flow.project import project_nonneg

def get_golub_data(p=2000):
    X_raw, _, _ = load_golub3()
    Xl = np.log2(np.clip(X_raw, 1.0, None))
    top = np.argsort(Xl.var(0))[-p:]
    Z = (Xl[:,top] - Xl[:,top].mean(0)) / (Xl[:,top].std(0,ddof=1)+1e-12)
    return torch.as_tensor(Z, dtype=torch.float64)

# ── 1. SPG (Current implementation) ───────────────────────────────────────────
def run_spg(Y0, energy_fn, grad_fn, proj_fn, max_iter=2000, tol=1e-6, patience=50, rtol=1e-6):
    from nsa_flow.solve import _spg_loop
    t0 = time.time()
    Y, E, it, stop, gmap = _spg_loop(
        Y0.clone(), proj_fn, energy_fn, grad_fn,
        max_iter=max_iter, tol=tol, sigma=1e-4, patience=patience, rtol=rtol
    )
    return {"name": "SPG (Current)", "time": time.time() - t0, "iters": it, "energy": E, "gmap": gmap, "stop": stop}

# ── 2. SPG-ABB (Alternating BB + Safeguarded BB step) ─────────────────────────
def run_spg_abb(Y0, energy_fn, grad_fn, proj_fn, max_iter=2000, tol=1e-6, patience=50, rtol=1e-6):
    t0 = time.time()
    Y = proj_fn(Y0.clone())
    E, g = grad_fn(Y)
    E = float(E)
    t = 1.0 / max(float(g.norm()), 1e-12)
    Y_prev = g_prev = None
    gmap, stop, it = float("inf"), "max_iter", 0
    E_window = []
    _T_MIN, _T_MAX = 1e-12, 1e4

    for it in range(1, max_iter + 1):
        if Y_prev is not None:
            s_ = Y - Y_prev
            r_ = g - g_prev
            sr = float((s_ * r_).sum())
            if sr > 0:
                # Alternating BB: alternate between BB1 and BB2
                if it % 2 == 0:
                    t = float((s_ * s_).sum()) / sr        # BB1
                else:
                    t = sr / float((r_ * r_).sum())        # BB2
            else:
                # Safeguard: when sr <= 0 (non-convex descent direction),
                # do NOT jump to 1e12! Keep previous t or modest step.
                t = min(max(t, 1e-3), 10.0)
            t = min(max(t, _T_MIN), _T_MAX)

        accepted = False
        t_first, dn2_first = t, None
        for _ in range(30):
            Y_new = proj_fn(Y - t * g)
            d_ = Y_new - Y
            dn2 = float((d_ * d_).sum())
            if dn2_first is None:
                dn2_first = dn2
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

    return {"name": "SPG-ABB (Alternating BB)", "time": time.time() - t0, "iters": it, "energy": E, "gmap": gmap, "stop": stop}

# ── 3. APGD / FISTA with Adaptive Restart ─────────────────────────────────────
def run_apgd(Y0, energy_fn, grad_fn, proj_fn, max_iter=2000, tol=1e-6, patience=50, rtol=1e-6):
    t0 = time.time()
    Y = proj_fn(Y0.clone())
    X = Y.clone()
    E, g = grad_fn(X)
    E = float(E)
    t = 1.0 / max(float(g.norm()), 1e-12)
    gmap, stop, it = float("inf"), "max_iter", 0
    E_window = []
    theta = 1.0

    for it in range(1, max_iter + 1):
        # Line search on step size t from X
        accepted = False
        t_cur = t * 1.2  # slight growth
        for _ in range(25):
            Y_new = proj_fn(X - t_cur * g)
            d_ = Y_new - X
            dn2 = float((d_ * d_).sum())
            if float(energy_fn(Y_new)) <= E - 1e-4 * dn2 / t_cur:
                accepted = True
                t = t_cur
                break
            t_cur *= 0.5

        if not accepted:
            stop = "line_search"
            break

        gmap = (dn2 ** 0.5) / t

        # Adaptive restart: if momentum points uphill, restart to 0
        diff = Y_new - Y
        if float((diff * g).sum()) > 0:
            theta = 1.0
            X = Y_new.clone()
        else:
            theta_next = (1.0 + math.sqrt(1.0 + 4.0 * theta * theta)) / 2.0
            beta = (theta - 1.0) / theta_next
            X = Y_new + beta * diff
            theta = theta_next

        Y = Y_new
        E, g = grad_fn(X)
        E = float(E)

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

    return {"name": "APGD (FISTA + Restart)", "time": time.time() - t0, "iters": it, "energy": E, "gmap": gmap, "stop": stop}

# ── 4. Projected Adam ─────────────────────────────────────────────────────────
def run_projected_adam(Y0, energy_fn, grad_fn, proj_fn, max_iter=2000, lr=0.01, tol=1e-6, patience=50, rtol=1e-6):
    t0 = time.time()
    Y = proj_fn(Y0.clone())
    m = torch.zeros_like(Y)
    v = torch.zeros_like(Y)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    stop, it = "max_iter", 0
    E_window = []
    gmap = float("inf")

    for it in range(1, max_iter + 1):
        E, g = grad_fn(Y)
        E = float(E)

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

    return {"name": f"Projected Adam (lr={lr})", "time": time.time() - t0, "iters": it, "energy": E, "gmap": gmap, "stop": stop}

# ── 5. SciPy L-BFGS-B ─────────────────────────────────────────────────────────
def run_lbfgs_b(Y0, energy_fn, grad_fn, max_iter=500):
    t0 = time.time()
    shape = Y0.shape
    def f_and_g(y_flat):
        Y_t = torch.as_tensor(y_flat.reshape(shape), dtype=Y0.dtype)
        E, g = grad_fn(Y_t)
        return float(E), g.numpy().astype(np.float64).flatten()

    bounds = [(0.0, None)] * Y0.numel()
    res = minimize(
        f_and_g, Y0.numpy().flatten(),
        method="L-BFGS-B", jac=True, bounds=bounds,
        options=dict(maxiter=max_iter, ftol=1e-7, gtol=1e-5)
    )
    return {
        "name": "L-BFGS-B (SciPy)", "time": time.time() - t0,
        "iters": res.nit, "energy": float(res.fun), "gmap": float(np.max(np.abs(res.jac))),
        "stop": "success" if res.success else "max_iter"
    }

def benchmark_problem(name, Y0, energy_fn, grad_fn, proj_fn):
    print(f"\n=======================================================")
    print(f"BENCHMARK: {name}")
    print(f"=======================================================")
    optimizers = [
        lambda: run_spg(Y0, energy_fn, grad_fn, proj_fn),
        lambda: run_spg_abb(Y0, energy_fn, grad_fn, proj_fn),
        lambda: run_apgd(Y0, energy_fn, grad_fn, proj_fn),
        lambda: run_projected_adam(Y0, energy_fn, grad_fn, proj_fn, lr=0.02),
        lambda: run_lbfgs_b(Y0, energy_fn, grad_fn, max_iter=300),
    ]
    results = []
    for opt in optimizers:
        res = opt()
        results.append(res)
        print(f"{res['name']:<25} | Time: {res['time']:6.2f}s | Iters: {res['iters']:5d} | Energy: {res['energy']:.6e} | Stop: {res['stop']}")
    return results

if __name__ == "__main__":
    T = get_golub_data(p=2000)
    k = 6
    p = 2000
    ops = GramOperator(X=T)
    c = ops.c
    trS = ops.trS

    for w in [0.05, 0.5]:
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
        V0 = E_init.clamp_min(0.0).clone()
        benchmark_problem(f"Golub Data-Reconstruction (k={k}, p={p}, w={w})", V0, e_fn, g_fn, project_nonneg)

    # ── Benchmark Signed Objective ─────────────────────────────────────────────
    from nsa_flow.signed import _split, gram_offdiag_defect, grad_gram_offdiag_defect, reconstruction_fidelity, grad_reconstruction_fidelity, relax_into_nonneg
    S = T.transpose(-2, -1) @ T
    c = float(S.trace())
    for w in [0.05, 0.5]:
        V0 = relax_into_nonneg(S, c, k, float(w), trS=c)
        W0 = torch.cat([V0.clamp_min(0.0), (-V0).clamp_min(0.0)], dim=-1).clone()

        def s_e_fn(Wv):
            Vp, Vm = _split(Wv)
            f = reconstruction_fidelity(Vp - Vm, S, c, c)
            d = gram_offdiag_defect(Wv)
            e = (1.0 - w) * f + w * d
            if w > 0.0:
                e = e + w * 1.0 * (Vp * Vm).sum() / c
            return e

        def s_g_fn(Wv):
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

        benchmark_problem(f"Golub Signed-Lifting (k={k}, p={p}, w={w})", W0, s_e_fn, s_g_fn, project_nonneg)
