r"""Which optimiser should NSA-Flow use?  Decided by measurement, not by assertion.

Motivation
----------
The default optimiser was changed to ``torch_lbfgs`` on the strength of a wall-clock
speedup, without a matched comparison of *solution quality* or a common
stationarity certificate.  Both were missing, and both matter more than speed:

* the data and signed objectives are **quartic and non-convex**, so different
  optimisers reach different stationary points -- the choice changes the answer,
  not only the time taken;
* the certificate each loop reported was a different function, so "converged"
  was not comparable between them.

Both are fixed (see :mod:`nsa_flow.diagnostics` and :mod:`nsa_flow.optim`).  This
module supplies the evidence that should have accompanied the change.

What is measured
----------------
For each problem instance -- mode x shape x ``w`` x dtype x device -- every
optimiser is run from the *same* initialisation to the *same* tolerance under
the *same* budget, and we record:

``seconds``, ``n_grad``, ``n_energy``
    Cost.  ``n_grad`` is the portable currency; ``seconds`` is what the user
    feels and is the only one that sees device transfers and kernel launches.
``grad_map``, ``stop_reason``, ``certificate``
    Did it actually converge, under the one shared definition.
``energy_excess``
    ``E - min(E over all optimisers on this instance)``.  **The quality metric.**
    A method that is twice as fast and lands 2e-3 higher on a quartic has not
    won; it has found a worse basin.
``t_to_1e-6``, ``t_to_1e-9``
    Seconds to first reach that certificate, read off the trace.  Separates
    "fast to a rough answer" from "fast to a good one".
``support_jaccard``, ``subspace_cos``
    Agreement of the returned basis with the best-energy solution on the same
    instance.  The interpretability claims are about supports, so a basis that
    matches the best energy but not its support is a different answer.

Run
---
``python experiments/optimizer_study.py [--quick] [--devices cpu,mps]``

Writes ``paper/results/optimizer_study.csv`` (one row per run) and
``paper/results/optimizer_study_summary.csv`` (the recommendation table).
"""
import argparse
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nsa_flow.optim import OPTIMIZERS
from nsa_flow.reconstruct import nsa_flow_data
from nsa_flow.signed import nsa_flow_signed
from nsa_flow.solve import _nsa_flow_anchored

RESULTS = Path(__file__).resolve().parent.parent / "paper" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

#: ``label -> (n, p, k)``.  Chosen to straddle the two cost regimes the data
#: solver switches between: ``p < n`` (form ``S = X'X``, ``O(p^2 k)``) and
#: ``p > n`` (matrix-free, ``O(n p k)``).
SHAPES = {
    "small":  (200, 40, 5),
    "medium": (400, 400, 10),
    "wide":   (120, 4000, 10),      # p >> n, the genomics regime
}
QUICK_SHAPES = ("small", "medium")
WS = (0.1, 0.5, 0.9)
QUICK_WS = (0.5,)
MODES = ("anchored", "data", "signed")


#: Two problem families, because they ask different questions.
#:
#: ``planted``
#:     Low-rank disjoint-support structure plus noise -- the regime the method
#:     is designed for, where a good optimiser should recover the same basis.
#: ``random``
#:     Unstructured Gaussian.  There is no planted answer, the quartic landscape
#:     is genuinely multi-modal, and this is where optimisers separate: on one
#:     60x30 instance SPG and the quasi-Newton methods differ by 2e-3 in energy,
#:     which is four orders larger than the tolerance anyone would set.  A study
#:     run only on ``planted`` would conclude, wrongly, that the choice is
#:     purely a speed question.
FAMILIES = ("planted", "random")


def make_problem(mode, shape, seed, dtype, device, family="planted"):
    """A reproducible instance from the named family."""
    n, p, k = SHAPES[shape]
    gen = torch.Generator().manual_seed(seed)
    if family == "random":
        X = torch.randn(n, p, generator=gen, dtype=torch.float64)
    else:
        V = torch.zeros(p, k, dtype=torch.float64)
        per = max(1, p // k)
        for j in range(k):                               # planted partition
            V[j * per:(j + 1) * per, j] = torch.rand(
                min(per, p - j * per), generator=gen, dtype=torch.float64) + 0.5
        Z = torch.rand(n, k, generator=gen, dtype=torch.float64) * 5.0
        X = Z @ V.T + 0.3 * torch.randn(n, p, generator=gen, dtype=torch.float64)
    X = (X - X.mean(0))                                  # centred: signed
    X = X.to(dtype=dtype, device=device)
    if mode == "anchored":
        # the anchored form takes a [p, k] target, not the data
        target = torch.linalg.svd(X, full_matrices=False)[2][:k].T.contiguous()
        return dict(target=target, k=k)
    if mode == "data":
        Xp = (X - X.min()).clamp_min(0.0)                 # non-negative
        return dict(X=Xp, k=k)
    return dict(X=X, k=k)


def run_one(mode, prob, optimizer, w, budget, tol):
    """One fit, traced.  Returns ``(result, trace)`` or ``(None, reason)``."""
    kw = dict(w=w, max_iter=budget, tol=tol, optimizer=optimizer,
              keep_trace=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        if mode == "anchored":
            r = _nsa_flow_anchored(prob["target"], **kw)
        elif mode == "data":
            r = nsa_flow_data(prob["X"], k=prob["k"], init="clamp", **kw)
        else:
            r = nsa_flow_signed(prob["X"], k=prob["k"], init="auto", **kw)
        wall = time.time() - t0
    return r, wall


def _time_to(trace, target):
    """Seconds to first reach ``grad_map <= target``; NaN if never."""
    for row in trace or ():
        if row.get("grad_map", float("inf")) <= target:
            return float(row["seconds"])
    return float("nan")


def _support(Y, rel=1e-8):
    return (Y.abs() > rel * Y.abs().max()).cpu().numpy()


def _agreement(Y, Y_ref):
    """Support Jaccard and mean matched |cos| against a reference basis."""
    A, B = _support(Y), _support(Y_ref)
    inter, union = (A & B).sum(), (A | B).sum()
    jac = float(inter / union) if union else 1.0
    An = Y / Y.norm(dim=0, keepdim=True).clamp_min(1e-300)
    Bn = Y_ref / Y_ref.norm(dim=0, keepdim=True).clamp_min(1e-300)
    Cm = (An.T @ Bn).abs().double().cpu().numpy()
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-Cm)
        cos = float(Cm[r, c].mean())
    except ImportError:                                            # pragma: no cover
        cos = float(Cm.max(axis=0).mean())
    return jac, cos


#: ``torch_lbfgs`` is excluded from the default grid.  It is not a close call
#: and it is not cheap to measure: across every configuration run so far it
#: failed to certify convergence even once, landed 1e-9 to 5e-8 above the best
#: energy, and cost 7.8 s where the others cost 0.01-0.02 s.  Carrying it
#: through the full grid spends most of the study's wall time re-confirming a
#: settled result.  Pass ``include_slow=True`` to put it back.
_EXCLUDED = ("torch_lbfgs",)


def study(devices=("cpu",), quick=False, budget=20000, tol=1e-9, seeds=(0, 1),
          include_slow=False):
    shapes = QUICK_SHAPES if quick else tuple(SHAPES)
    ws = QUICK_WS if quick else WS
    if quick:
        seeds = seeds[:1]
    opts = tuple(o for o in OPTIMIZERS
                 if include_slow or o not in _EXCLUDED)
    rows = []

    for device in devices:
        dtypes = (torch.float32,) if device == "mps" else (torch.float64,)
        for mode, shape, w, dtype, family, seed in itertools.product(
                MODES, shapes, ws, dtypes, FAMILIES, seeds):
            prob = make_problem(mode, shape, seed, dtype, device, family)
            tol_d = tol if dtype == torch.float64 else 1e-6
            instance = []
            for opt in opts:
                # scipy is host-side float64; running it from a device tensor
                # measures the transfer, not the method
                if opt == "lbfgsb" and device != "cpu":
                    continue
                try:
                    r, wall = run_one(mode, prob, opt, w, budget, tol_d)
                except Exception as exc:                          # pragma: no cover
                    rows.append(dict(device=device, mode=mode, shape=shape, w=w,
                                     dtype=str(dtype).split(".")[-1],
                                     family=family, seed=seed,
                                     optimizer=opt, error=repr(exc)[:200]))
                    continue
                instance.append((opt, r, wall))

            if not instance:
                continue
            best_E = min(r.energy for _, r, _ in instance)
            Y_ref = min(instance, key=lambda t: t[1].energy)[1].Y

            for opt, r, wall in instance:
                jac, cos = _agreement(r.Y, Y_ref)
                rows.append(dict(
                    device=device, mode=mode, shape=shape, w=w,
                    dtype=str(dtype).split(".")[-1], family=family, seed=seed,
                    optimizer=opt,
                    seconds=wall, n_grad=r.n_grad, n_energy=r.n_energy,
                    iters=r.iters, energy=r.energy, energy_excess=r.energy - best_E,
                    grad_map=r.grad_map, stop_reason=r.stop_reason,
                    certificate=r.certificate, converged=r.converged,
                    fidelity=r.fidelity, defect_D=r["defect_D"],
                    sparsity=r["sparsity"], support_overlap=r["support_overlap"],
                    t_to_1e_6=_time_to(r.trace, 1e-6),
                    t_to_1e_9=_time_to(r.trace, 1e-9),
                    support_jaccard=jac, subspace_cos=cos,
                    error=None))
            print(f"[{device}/{str(dtype).split('.')[-1]}] {mode:9s} {shape:7s} "
                  f"{family:8s} s{seed} w={w:<4} -> " + "  ".join(
                      f"{o}:{r.seconds:.2f}s/{r.energy - best_E:+.1e}"
                      for o, r, _ in instance))
    return pd.DataFrame(rows)


def summarise(df):
    """Recommendation table: per mode and device, who wins and by how much."""
    ok = df[df["error"].isna()].copy()
    # rank within each instance by (quality first, then cost)
    keys = ["device", "mode", "shape", "w", "dtype", "family", "seed"]
    ok["rank_energy"] = ok.groupby(keys)["energy"].rank(method="min")
    ok["rank_time"] = ok.groupby(keys)["seconds"].rank(method="min")
    ok["found_best"] = ok["energy_excess"] <= 1e-9 * (1.0 + ok["energy"].abs())
    g = ok.groupby(["device", "mode", "optimizer"])
    out = g.agg(
        n=("seconds", "size"),
        median_s=("seconds", "median"),
        median_n_grad=("n_grad", "median"),
        frac_certified=("converged", "mean"),
        frac_stationary=("certificate", lambda s: float((s == "stationary").mean())),
        frac_found_best=("found_best", "mean"),
        median_energy_excess=("energy_excess", "median"),
        max_energy_excess=("energy_excess", "max"),
        median_support_jaccard=("support_jaccard", "median"),
        mean_rank_energy=("rank_energy", "mean"),
        mean_rank_time=("rank_time", "mean"),
    ).reset_index()
    return out.sort_values(["device", "mode", "mean_rank_energy", "median_s"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small grid, one dtype")
    ap.add_argument("--devices", default="cpu",
                    help="comma-separated: cpu,mps,cuda")
    ap.add_argument("--budget", type=int, default=20000,
                    help="max gradient evaluations per fit")
    ap.add_argument("--include-slow", action="store_true",
                    help="also measure torch_lbfgs (see _EXCLUDED)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    devices = []
    for d in args.devices.split(","):
        d = d.strip()
        if d == "mps" and not torch.backends.mps.is_available():
            print("skipping mps: unavailable")
            continue
        if d == "cuda" and not torch.cuda.is_available():
            print("skipping cuda: unavailable")
            continue
        devices.append(d)

    df = study(devices=tuple(devices), quick=args.quick, budget=args.budget,
               seeds=tuple(range(args.seeds)), include_slow=args.include_slow)
    out = Path(args.out)
    df.to_csv(out / "optimizer_study.csv", index=False)
    s = summarise(df)
    s.to_csv(out / "optimizer_study_summary.csv", index=False)
    print("\n" + "=" * 100)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("=" * 100)
    print(f"\nwrote {out/'optimizer_study.csv'} and {out/'optimizer_study_summary.csv'}")


if __name__ == "__main__":
    main()
