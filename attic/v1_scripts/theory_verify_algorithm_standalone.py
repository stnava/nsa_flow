"""Verification of the algorithmic claims: SPG descent, KKT stationarity,
the w=0 and w=1 limits, continuation, and scale boundedness."""
import itertools, math, time
import torch

from nsa_flow import nsa_flow, energy, stiefel_defect, effective_rank, project_nonneg

torch.set_default_dtype(torch.float64)
OK = []
def chk(name, cond, detail=""):
    OK.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))

torch.manual_seed(0)
p, k = 80, 8
X0 = torch.rand(p, k)

# --------------------------------------------------------- A1 monotone descent
print("\nA1  energy decreases monotonically at every accepted step")
for w in [0.1, 0.5, 0.9]:
    r = nsa_flow(X0, w=w, max_iter=3000, keep_trace=True)
    e = [t["energy"] for t in r.trace]
    rises = [(i, e[i+1] - e[i]) for i in range(len(e)-1) if e[i+1] > e[i] + 1e-15]
    chk(f"w={w} monotone over {len(e)} steps", not rises,
        f"E: {e[0]:.6e} -> {e[-1]:.6e}, worst rise={max([d for _,d in rises], default=0.0):.2e}")

# ------------------------------------------------------- A2 KKT / stationarity
print("\nA2  KKT at termination: grad=0 where Y>0, grad>=0 where Y=0")
from nsa_flow import grad_energy
for w in [0.1, 0.5, 0.9]:
    r = nsa_flow(X0, w=w, max_iter=20000, tol=1e-12)
    Y = r.Y; g = grad_energy(Y, r.target, w=w, denom=r.target.pow(2).sum())
    free, act = Y > 1e-13, Y <= 1e-13
    gmax = g[free].abs().max().item() if free.any() else 0.0
    gmin = g[act].min().item() if act.any() else 0.0
    scale = g.abs().max().item()
    chk(f"w={w} stationary", gmax <= 1e-7 * max(scale, 1.0) and gmin >= -1e-9,
        f"max|g| on free set={gmax:.3e}  min g on active set={gmin:.3e}  (converged={r.converged}, iters={r.iters})")

# ----------------------------------------------------------------- A3 w=0 limit
print("\nA3  w=0 limit is exactly max(0, X0)")
Xs = torch.randn(p, k)                       # signed target so the clamp bites
r0 = nsa_flow(Xs, w=0.0, max_iter=5000, tol=1e-14)
chk("w=0 recovers the projection", torch.allclose(r0.Y, Xs.clamp_min(0), atol=1e-9),
    f"max dev={(r0.Y - Xs.clamp_min(0)).abs().max().item():.3e}")
chk("w=0 fidelity equals that of max(0,X0)",
    abs(r0.fidelity - (Xs.clamp_min(0) - Xs).pow(2).sum().item()/Xs.pow(2).sum().item()) < 1e-12)

# --------------------------------------------------------------- A4 w->1 limit
print("\nA4  w->1 drives Y to disjoint supports (a hard clustering of features)")
for w in [0.9, 0.99, 1.0]:
    r = nsa_flow(X0, w=w, max_iter=40000, tol=1e-13, continuation=8)
    Y = r.Y
    supp = [set((Y[:, i] > 1e-7 * Y.abs().max()).nonzero().flatten().tolist()) for i in range(k)]
    overlaps = sum(len(supp[i] & supp[j]) for i, j in itertools.combinations(range(k), 2))
    cover = len(set().union(*supp))
    chk(f"w={w} disjoint supports", overlaps == 0,
        f"pairwise overlap={overlaps}, features covered={cover}/{p}, D={r.raw_defect:.3e}, ER={r.effective_rank:.3f}")

# --------------------------------------------------- A5 continuation vs cold start
print("\nA5  continuation in w reaches lower energy than a cold start (graduated non-convexity)")
torch.manual_seed(7)
wins = 0; trials = 12; gaps = []
for s in range(trials):
    Xr = torch.rand(60, 6) * (torch.rand(60, 6) < 0.5)      # sparse, multi-modal
    Xr = Xr + 1e-3
    cold = nsa_flow(Xr, w=0.85, max_iter=8000, tol=1e-12)
    warm = nsa_flow(Xr, w=0.85, max_iter=8000, tol=1e-12, continuation=12)
    gaps.append(cold.energy - warm.energy)
    if warm.energy <= cold.energy + 1e-12: wins += 1
chk(f"continuation no worse on {wins}/{trials} problems", wins >= trials - 1,
    f"mean energy gain={sum(gaps)/len(gaps):+.3e}, best={max(gaps):+.3e}")

# ------------------------------------------------------- A6 scale boundedness
print("\nA6  ||Y|| stays bounded away from 0 (D cannot shrink scale; F pins it)")
for w in [0.5, 0.9, 0.99]:
    r = nsa_flow(X0, w=w, max_iter=5000, keep_trace=True)
    ratio = r.Y.norm().item() / X0.norm().item()
    chk(f"w={w} ||Y||/||X0|| in (0.1, 10)", 0.1 < ratio < 10.0, f"ratio={ratio:.4f}")

# --------------------------------------------- A7 no factorisation in the loop
print("\nA7  inner loop is factorisation-free: cost scales as O(p k^2)")
import numpy as np
def timeit(pp, kk, iters=300):
    Xt = torch.rand(pp, kk)
    t0 = time.perf_counter(); nsa_flow(Xt, w=0.5, max_iter=iters, tol=0.0); return time.perf_counter() - t0
base = timeit(400, 10)
d_p  = timeit(800, 10)            # 2x p  -> expect ~2x
d_k  = timeit(400, 20)            # 2x k  -> expect ~4x
chk("doubling p roughly doubles cost", 1.4 < d_p/base < 3.2, f"ratio={d_p/base:.2f} (expect ~2)")
chk("doubling k roughly quadruples cost", 2.0 < d_k/base < 7.0, f"ratio={d_k/base:.2f} (expect ~4)")

print(f"\n{'='*66}\n{sum(OK)}/{len(OK)} algorithmic checks passed\n{'='*66}")
raise SystemExit(0 if all(OK) else 1)
