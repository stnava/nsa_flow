"""Speed protocol: best-of-3 fit time of NSA methods against PCA on seeded synthetic shapes.

Seeded and repeated: the same ``X`` goes to every method; the minimum over repeats is reported (a floor,
not a mean polluted by scheduling).  Rows use metric ``seconds`` with the shape in ``variant``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import methods as _methods  # noqa: F401  (populates METHODS in workers)
from ..registry import METHODS, protocol

SHAPES = {"smoke": [(100, 30, 3)], "quick": [(300, 66, 5), (1000, 500, 5)],
          "standard": [(300, 66, 5), (1000, 500, 5), (1500, 1000, 5), (300, 2000, 5)],
          "full": [(300, 66, 5), (1000, 500, 5), (1500, 1000, 5), (300, 2000, 5), (3000, 2000, 8)]}


@protocol("speed")
def run(ds, method_names, profile, jobs=1, seed0=0, reps=3, **_):
    rows = []
    for (n, p, k) in SHAPES.get(profile.name, SHAPES["standard"]):
        X = np.random.default_rng(seed0).standard_normal((n, p))
        rep = {"c": X - X.mean(0), "pos": np.clip(X - X.min(0), 0, None)}
        for name in method_names:
            m = METHODS[name]
            hp = {**m.module_cells[0], "k": k}
            best = np.inf
            for _ in range(reps):
                t0 = time.time()
                m.fit(rep, hp, 0, {})
                best = min(best, time.time() - t0)
            rows.append(dict(dataset=f"speed(n={n},p={p},k={k})", protocol="speed", split=0, method=name,
                             variant="best_of_%d" % reps, model="", metric="seconds", value=best))
    return pd.DataFrame(rows)
