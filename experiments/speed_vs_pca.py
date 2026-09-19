r"""How much slower than PCA is NSA-Flow, and where does the time actually go?

PCA is the right yardstick: it solves the ``w = 0``, unconstrained, sign-free
version of the same problem with one direct factorisation.  NSA-Flow is
iterative and constrained, so it cannot be *as* fast -- but "same order of
magnitude" is a reasonable bar for a library meant to sit in a normal ML
pipeline, and it is a bar that can be measured rather than asserted.

What this measures, and why it is set up this way
-------------------------------------------------
**Seeded.**  An earlier ad-hoc version of this benchmark drew a fresh random
``X`` for every run.  On nominally identical problems SPG's gradient count read
924, 1365 and 3362 across three runs, because they were three different
problems.  Every configuration here is seeded, and the same ``X`` is handed to
PCA and to every optimiser.

**Repeated, best-of.**  Wall clock on a laptop is noisy; we report the minimum
over ``reps`` runs after a warm-up, which is the standard way to estimate a
floor rather than a mean polluted by scheduling.

**Synchronised.**  On MPS and CUDA the work is queued asynchronously, so timing
without a device synchronise measures the enqueue, not the compute.

**Broken down by cost centre.**  ``us_per_eval`` is the wall time divided by
(gradient + energy evaluations).  It is the number that says whether the loop is
bound by arithmetic or by overhead: at ``n=500, p=200, k=10`` one gradient is
about 2 MFLOP, which is ~10 us of arithmetic; anything far above that is kernel
launch and Python dispatch.  ``n_energy / n_grad`` says how much of the cost is
the line search, which is where the first-order methods lose -- backtracking
Armijo on a Barzilai-Borwein step spends 3 to 5 energy evaluations per gradient,
while FISTA's Lipschitz backtracking spends about 1.15.

Run
---
``python experiments/speed_vs_pca.py [--devices cpu,mps] [--quick]``

Writes ``paper/results/speed_vs_pca.csv``.
"""
import argparse
import time
import warnings
from pathlib import Path

import pandas as pd
import torch

from nsa_flow.optim import optimizer_names
from nsa_flow.reconstruct import nsa_flow_data

RESULTS = Path(__file__).resolve().parent.parent / "paper" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

#: ``(n, p, k)``.  Spans both cost regimes of the data solver and both sides of
#: the ``p > n`` matrix-free crossover.
SHAPES = [
    (500, 200, 10),      # tall, small
    (2000, 500, 10),     # tall, medium
    (500, 2000, 10),     # wide
    (200, 5000, 10),     # very wide: the genomics regime
    (500, 200, 50),      # many components
]
QUICK_SHAPES = SHAPES[:3]


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _best_of(fn, device, reps):
    fn()                                              # warm up caches / kernels
    _sync(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return min(times)


def pca_reference(X, k):
    """The yardstick: a full SVD, which is what ``sklearn`` PCA does for these
    shapes, returning the same ``[p, k]`` loading matrix NSA-Flow produces."""
    Xc = X - X.mean(0)
    return torch.linalg.svd(Xc, full_matrices=False)[2][:k].T


def run(devices=("cpu",), shapes=SHAPES, reps=3, seed=0, budget=20000,
        optimizers=None):
    # the live optimizers only; scipy_lbfgsb is a host-side reference and
    # torch_lbfgs is deprecated -- both belong in optimizer_study, not here
    optimizers = tuple(optimizers or ("lbfgsb", "fista", "spg", "pqn"))
    rows = []
    for device in devices:
        dtypes = (torch.float32,) if device != "cpu" else (torch.float32, torch.float64)
        for dtype in dtypes:
            for (n, p, k) in shapes:
                gen = torch.Generator().manual_seed(seed)
                X = torch.rand(n, p, generator=gen, dtype=torch.float64)
                X = X.to(dtype=dtype, device=device)

                try:
                    t_pca = _best_of(lambda: pca_reference(X, k), device, reps)
                except Exception as exc:
                    print(f"  PCA failed on {device}/{dtype}: {exc!r}")
                    continue
                rows.append(dict(device=device, dtype=str(dtype).split(".")[-1],
                                 n=n, p=p, k=k, method="PCA (SVD)",
                                 seconds=t_pca, vs_pca=1.0))

                for opt in optimizers:
                    if opt == "lbfgsb" and device != "cpu":
                        continue                       # host-side by construction
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            fit = lambda: nsa_flow_data(          # noqa: E731
                                X, k=k, w=0.5, optimizer=opt, max_iter=budget)
                            t = _best_of(fit, device, reps=1)
                            r = fit()
                        except Exception as exc:
                            print(f"  {opt} failed on {device}/{dtype} "
                                  f"{n}x{p}: {exc!r}")
                            continue
                    n_eval = max(r.n_grad + r.n_energy, 1)
                    rows.append(dict(
                        device=device, dtype=str(dtype).split(".")[-1],
                        n=n, p=p, k=k, method=opt, seconds=t, vs_pca=t / t_pca,
                        n_grad=r.n_grad, n_energy=r.n_energy,
                        ls_ratio=r.n_energy / max(r.n_grad, 1),
                        us_per_eval=t / n_eval * 1e6,
                        energy=r.energy, grad_map=r.grad_map,
                        certificate=r.certificate, converged=r.converged))
                    print(f"{device:4s}/{str(dtype).split('.')[-1]:7s} "
                          f"{n:5d}x{p:<5d} k={k:<3d} {opt:12s} "
                          f"{t*1e3:9.1f}ms  {t/t_pca:7.1f}x PCA  "
                          f"n_grad={r.n_grad:5d} ls={r.n_energy/max(r.n_grad,1):4.1f} "
                          f"{r.certificate}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--devices", default="cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    devices = []
    for d in args.devices.split(","):
        d = d.strip()
        if d == "mps" and not torch.backends.mps.is_available():
            continue
        if d == "cuda" and not torch.cuda.is_available():
            continue
        devices.append(d)

    df = run(devices=tuple(devices),
             shapes=QUICK_SHAPES if args.quick else SHAPES, reps=args.reps)
    df.to_csv(RESULTS / "speed_vs_pca.csv", index=False)

    print("\n" + "=" * 90)
    piv = (df[df.method != "PCA (SVD)"]
           .pivot_table(index=["device", "dtype", "method"], values="vs_pca",
                        aggfunc=["median", "max"]))
    print(piv.to_string(float_format=lambda v: f"{v:.1f}"))
    print("=" * 90)
    print(f"wrote {RESULTS/'speed_vs_pca.csv'}")


if __name__ == "__main__":
    main()
