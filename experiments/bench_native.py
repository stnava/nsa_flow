import time
import torch
from nsa_flow.reconstruct import nsa_flow_data
import nsa_flow.optim as optim_mod
import nsa_flow.lbfgsb as lbfgsb_mod

shapes = [
    ("ADNI cortical", 300, 66, 5),
    ("Tall small", 500, 200, 10),
    ("Tall medium", 1000, 300, 10),
    ("Wide genomics", 200, 1000, 10),
]

results = []
for name, n, p, k in shapes:
    torch.manual_seed(42)
    X = torch.randn(n, p, dtype=torch.float64).abs()

    # 1. Native
    _ = nsa_flow_data(X, k=k, init="clamp", w=0.5, optimizer="lbfgsb", tol=1e-9, max_iter=200)
    t0 = time.perf_counter()
    reps = 5
    iters_nat = 0
    for _ in range(reps):
        r_nat = nsa_flow_data(X, k=k, init="clamp", w=0.5, optimizer="lbfgsb", tol=1e-9, max_iter=200)
        iters_nat += r_nat.iters
    t_nat = (time.perf_counter() - t0) / reps
    us_nat = (t_nat / (iters_nat / reps)) * 1e6

    # 2. Python fallback
    orig_lbf = lbfgsb_mod._native
    lbfgsb_mod._native = None
    try:
        _ = nsa_flow_data(X, k=k, init="clamp", w=0.5, optimizer="lbfgsb", tol=1e-9, max_iter=200)
        t0 = time.perf_counter()
        iters_py = 0
        for _ in range(reps):
            r_py = nsa_flow_data(X, k=k, init="clamp", w=0.5, optimizer="lbfgsb", tol=1e-9, max_iter=200)
            iters_py += r_py.iters
        t_py = (time.perf_counter() - t0) / reps
        us_py = (t_py / (iters_py / reps)) * 1e6
    finally:
        lbfgsb_mod._native = orig_lbf

    speedup = t_py / t_nat
    e_diff = abs(r_nat.energy - r_py.energy)
    results.append({
        "name": name, "shape": f"({n}, {p}, k={k})",
        "t_nat_ms": t_nat * 1e3, "us_nat": us_nat,
        "t_py_ms": t_py * 1e3, "us_py": us_py,
        "speedup": speedup, "energy_diff": e_diff
    })

print(f"{'Problem':<15} | {'Shape':<17} | {'Native (ms)':<11} | {'us/it':<7} | {'Python (ms)':<11} | {'us/it':<7} | {'Speedup':<8} | {'E diff'}")
print("-" * 100)
for r in results:
    print(f"{r['name']:<15} | {r['shape']:<17} | {r['t_nat_ms']:11.2f} | {r['us_nat']:7.1f} | {r['t_py_ms']:11.2f} | {r['us_py']:7.1f} | {r['speedup']:7.2f}x | {r['energy_diff']:.2e}")
