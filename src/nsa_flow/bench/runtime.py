"""Profiles, parallel execution and cost estimation.

Profiles trade breadth for time: ``smoke`` (seconds, synthetic), ``quick`` (minutes), ``fast`` (standard splits with a
smaller inner forest), ``standard`` (the reported numbers), ``full`` (more splits).  ``parallel_map`` runs items in loky workers; functions must be
importable by module name (workers cannot unpickle functions defined in ``__main__``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    n_outer: int            # outer splits (prediction) / draws (module recovery); datasets may cap it
    n_inner: int            # inner folds
    rf_trees: int                   # forest size for the outer (reported) fits
    max_method_minutes: float = 30.0
    rf_trees_inner: int = 0         # forest size while selecting hyper-parameters in the inner CV (0 = same as rf_trees)


PROFILES = {
    "smoke": Profile("smoke", 2, 2, 20, rf_trees_inner=10),
    "quick": Profile("quick", 5, 3, 50, rf_trees_inner=25),
    "standard": Profile("standard", 20, 5, 100),                       # full-forest selection: the reference fidelity
    "fast": Profile("fast", 20, 5, 100, rf_trees_inner=25),           # 25-tree selection: ~2-4x faster, attenuates small effects
    "full": Profile("full", 40, 5, 200, rf_trees_inner=50),
}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]


_POOL = None
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def parallel_map(fn, items, jobs=1, verbose=0):
    """``[fn(*item) for item in items]`` across workers; serial when ``jobs == 1``."""
    if jobs == 1:
        return [fn(*it) for it in items]
    import os
    for v in THREAD_ENV:                                   # workers inherit: no 5 x 12 OpenMP oversubscription
        os.environ.setdefault(v, "1")
    from joblib import Parallel, delayed
    return Parallel(n_jobs=jobs, verbose=verbose)(delayed(fn)(*it) for it in items)


def probe_fit_seconds(method, rep, hp, seed=0, ctx=None) -> float:
    """Wall time of one fit: the data-driven cost estimate used before a run (no guessed constants)."""
    t0 = time.time()
    method.fit(rep, hp, seed, ctx or {})
    return time.time() - t0


def limit_threads(n=1):
    """Worker setup: one torch thread per worker, and no per-fit warning spam (the solver certificate is
    recorded in every NSA row, so non-convergence is data, not console noise)."""
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from threadpoolctl import threadpool_limits   # BLAS + OpenMP pools (sklearn depends on it)
        global _POOL
        _POOL = threadpool_limits(limits=n)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
