"""Module-recovery protocol: do feature modules match known groups, with unlabelled background present?

Features are the labelled genes/words plus unlabelled background (``groups == -1``); methods never see
labels.  Over random sample draws (``Profile.n_outer`` of them), per method and variant:
  ari          ARI of labelled features' dominant component vs group, k = number of groups
  stability    split-half ARI of assignments over all fitted features, same k
  recall2k     k = 2 x groups: fraction of same-group pairs placed together
  separation2k k = 2 x groups: fraction of different-group pairs placed apart
  contiguity   ordered features only: fraction of adjacent features in the same module
Datasets without ``groups`` (Tecator) get stability and contiguity at ``k = 5``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import methods as _methods  # noqa: F401  (populates METHODS in workers)
from ..metrics import contiguity, dominant_assignment, pair_scores
from ..registry import METHODS, protocol
from ..runtime import limit_threads, parallel_map

DEFAULT_K = 5


def variants(method):
    """(label, hp) per module variant; NSA variants are labelled with their w."""
    return [((f"{method.name} w={c['w']}" if "w" in c else method.name), c) for c in method.module_cells]


def _draw(ds, r, n_samples, method_names, primary_only=False):
    from sklearn.metrics import adjusted_rand_score as ari_score
    limit_threads(1)
    rng = np.random.default_rng(r)
    idx = rng.choice(ds.n, min(n_samples, ds.n), replace=False)
    rep, _ = ds.prep(idx, idx[:1])
    h = rng.permutation(len(idx))
    half_a, half_b = idx[h[: len(idx) // 2]], idx[h[len(idx) // 2:]]
    rep_a, rep_b = ds.prep(half_a, half_a[:1])[0], ds.prep(half_b, half_b[:1])[0]
    ctx = {"spherical": ds.kmeans_spherical}
    labelled = None if ds.groups is None else np.where(ds.groups >= 0)[0]
    k = len(set(ds.groups[labelled])) if labelled is not None else DEFAULT_K
    rows = []
    for name in method_names:
        m = METHODS[name]
        for label, cell in (variants(m)[:1] if primary_only else variants(m)):
            hp = {**cell, "k": k}
            V, cert = m.fit(rep, hp, 0, ctx)
            lab = dominant_assignment(V)
            base = dict(dataset=ds.name, protocol="module_recovery", split=r, method=label, variant="draw",
                        model="", cert_converged=cert.get("converged"), cert_defect=cert.get("defect_D"))
            va, vb = m.fit(rep_a, hp, 0, ctx)[0], m.fit(rep_b, hp, 0, ctx)[0]
            vals = {"stability": ari_score(dominant_assignment(va), dominant_assignment(vb))}
            if labelled is not None:
                vals["ari"] = ari_score(ds.groups[labelled], lab[labelled])
                lab2 = dominant_assignment(m.fit(rep, {**cell, "k": 2 * k}, 0, ctx)[0])
                vals["recall2k"], vals["separation2k"] = pair_scores(lab2[labelled], ds.groups[labelled])
            if ds.ordered:
                vals["contiguity"] = contiguity(V)
            rows += [dict(base, metric=mt, value=float(v)) for mt, v in vals.items()]
    return rows


@protocol("module_recovery")
def run(ds, method_names, profile, jobs=1, seed0=0, n_samples=None, n_outer=None, primary_only=False, **_):
    use, skipped = [], []
    for name in method_names:
        ok, why = METHODS[name].applies(ds)
        (use if ok else skipped).append(name if ok else (name, why))
    n_samples = n_samples or ds.draw_n or ds.n // 2
    items = [(ds, seed0 + r, n_samples, use, primary_only) for r in range(n_outer or profile.n_outer)]
    df = pd.DataFrame([r for part in parallel_map(_draw, items, jobs) for r in part])
    for name, why in skipped:
        df = pd.concat([df, pd.DataFrame([dict(dataset=ds.name, protocol="module_recovery", method=name,
                                               skipped_reason=why)])], ignore_index=True)
    return df
