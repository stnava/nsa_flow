"""E6 -- cost and numerical robustness, v1 against the corrected implementation.

v1 evaluated two float64 SVDs per iteration (one inside the autograd graph, one
for the progress trace).  The corrected inner loop forms one Gram product and
two [p,k]x[k,k] matrix products, so it is factorisation-free and O(p k^2).
"""
import time
import warnings

import numpy as np
import pandas as pd
import torch

from nsa_flow import nsa_flow, polar_factor, stiefel_defect_normalised
from nsa_flow.energy import value_and_grad

from . import v1

warnings.filterwarnings("ignore")
F64 = torch.float64


def per_iteration_cost():
    """Isolated cost of one value-and-gradient evaluation."""
    rows = []
    for p, k in [(62, 5), (400, 10), (800, 10), (1600, 10), (400, 20), (400, 40),
                 (2000, 20), (7129, 5), (7129, 20)]:
        Y = torch.rand(p, k, dtype=F64)
        X0 = torch.rand(p, k, dtype=F64)
        d = X0.pow(2).sum()
        value_and_grad(Y, X0, 0.5, d)
        t0 = time.perf_counter()
        for _ in range(300):
            value_and_grad(Y, X0, 0.5, d)
        us = (time.perf_counter() - t0) / 300 * 1e6
        rows.append(dict(p=p, k=k, us_per_iter=us, mflop=4 * p * k * k / 1e6))
    return pd.DataFrame(rows)


def end_to_end():
    """v1 against v2 on matched problems: iterations, wall clock, and what we get."""
    rows = []
    torch.manual_seed(0)
    for p, k in [(100, 10), (500, 10), (2000, 20), (7129, 5)]:
        X0 = torch.rand(p, k, dtype=F64)
        for w in [0.5, 0.9]:
            t0 = time.perf_counter()
            r = nsa_flow(X0, w=w, max_iter=20000, tol=1e-10)
            t2 = time.perf_counter() - t0
            rows.append(dict(version="v2", p=p, k=k, w=w, seconds=t2, iters=r.iters,
                             defect=r.raw_defect, fidelity=r.fidelity,
                             eff_rank=r.effective_rank, converged=r.converged))
            try:
                t0 = time.perf_counter()
                rv = v1.nsa_flow_orth(X0 + 0.0, X0, w=w, max_iter=500,
                                      initial_learning_rate=1e-3, apply_nonneg="hard",
                                      precision="float64", optimizer="adam")
                t1 = time.perf_counter() - t0
                Yv = rv["Y"]
                rows.append(dict(
                    version="v1", p=p, k=k, w=w, seconds=t1,
                    iters=int(rv["final_iter"]),
                    defect=float(stiefel_defect_normalised(Yv)),
                    fidelity=float((Yv - rv["target"]).pow(2).sum()
                                   / rv["target"].pow(2).sum()),
                    eff_rank=np.nan, converged=np.nan))
            except Exception as e:
                rows.append(dict(version="v1", p=p, k=k, w=w, seconds=np.nan,
                                 iters=-1, error=str(e)[:60]))
    return pd.DataFrame(rows)


def default_learning_rate_pathology():
    """v1's default lr_strategy='bayes' chose the step by rewarding flat regions."""
    rows = []
    torch.manual_seed(42)
    np.random.seed(42)
    for p, k in [(100, 20), (500, 10)]:
        X0 = torch.rand(p, k, dtype=F64)
        for strat in ["bayes", "armijo", "grid", "random"]:
            try:
                res = v1.estimate_learning_rate_for_nsa_flow(
                    torch.randn_like(X0), X0, w=0.5, strategy=strat, aggression=0.5)
                rows.append(dict(strategy=strat, p=p, k=k, best_lr=res["best_lr"]))
            except Exception as e:
                rows.append(dict(strategy=strat, p=p, k=k, best_lr=np.nan,
                                 error=str(e)[:60]))
    return pd.DataFrame(rows)


def svd_gradient_failure():
    """Differentiating the SVD fails at repeated singular values; the polar
    factor's own derivative does not.  nn.init.orthogonal_ lands exactly there."""
    rows = []
    for p, k in [(30, 5), (64, 16), (128, 32)]:
        G = torch.randn(p, k, dtype=F64)
        for label, Y in [("generic (distinct sigma)", torch.randn(p, k, dtype=F64)),
                         ("orthogonal (sigma all 1)",
                          torch.linalg.qr(torch.randn(p, k, dtype=F64))[0])]:
            out = {}
            for method, fn in [("svd_autograd", lambda A: torch.linalg.svd(
                                    A, full_matrices=False)[0]
                                    @ torch.linalg.svd(A, full_matrices=False)[2]),
                               ("polar_sylvester", polar_factor)]:
                A = Y.clone().requires_grad_(True)
                try:
                    (fn(A) * G).sum().backward()
                    out[method] = (bool(torch.isfinite(A.grad).all()),
                                   int(torch.isnan(A.grad).sum()))
                except Exception:
                    out[method] = (False, -1)
            rows.append(dict(p=p, k=k, case=label,
                             svd_finite=out["svd_autograd"][0],
                             svd_nans=out["svd_autograd"][1],
                             polar_finite=out["polar_sylvester"][0],
                             polar_nans=out["polar_sylvester"][1]))
    return pd.DataFrame(rows)


def uniqueness(n_restarts=24):
    """Random restarts reach the same optimum for w < 1: no restart lottery."""
    rows = []
    from .data import planted_partition
    for w in [0.5, 0.9, 0.99]:
        for seed in range(4):
            X, V, _ = planted_partition(p=60, k=6, noise=0.4, seed=seed)
            from sklearn.decomposition import PCA
            L0 = PCA(n_components=6, random_state=0).fit(X).components_.T
            T = torch.as_tensor(L0, dtype=F64)
            es = []
            for s in range(n_restarts):
                g = torch.Generator().manual_seed(s)
                es.append(nsa_flow(T, w=w, init=torch.rand(60, 6, generator=g, dtype=F64),
                                   max_iter=30000, tol=1e-12).energy)
            cold = nsa_flow(T, w=w, max_iter=30000, tol=1e-12).energy
            es = np.array(es)
            rows.append(dict(w=w, seed=seed, n_restarts=n_restarts,
                             e_min=es.min(), e_max=es.max(),
                             rel_spread=(es.max() - es.min()) / max(abs(es.min()), 1e-300),
                             cold=cold))
    return pd.DataFrame(rows)


def run():
    return dict(per_iteration=per_iteration_cost(), end_to_end=end_to_end(),
                lr_pathology=default_learning_rate_pathology(),
                svd_gradient=svd_gradient_failure(), uniqueness=uniqueness())
