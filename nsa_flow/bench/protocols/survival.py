"""Survival protocol: survey-weighted forest on [covariates, basis scores], scored by Harrell's C and AUC.

Same bases as the prediction protocol but at fixed ``k = 5`` and the method's default ``w`` (no inner tuning, so
every method gets the same budget).  The forest is fitted with the survey weights (normalised), the predicted death
probability is the risk score, and the exact Harrell C-index is computed against ``(time, event)`` on the held-out
rows.  Splits: stratified repeated k-fold on the event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import methods as _methods  # noqa: F401  (populates METHODS in workers)
from ..metrics import harrell_c
from ..registry import METHODS, protocol
from ..runtime import limit_threads, parallel_map
from .nested_cv import outer_splits


def _split(ds, split, tr, te, method_names, profile, k):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    limit_threads(1)
    seed = 1000 + split
    rep_tr, rep_te = ds.prep(tr, te)
    ctx = {"spherical": ds.kmeans_spherical}
    cm, cs = ds.covariates[tr].mean(0), ds.covariates[tr].std(0) + 1e-12
    Ctr, Cte = (ds.covariates[tr] - cm) / cs, (ds.covariates[te] - cm) / cs
    sw = ds.weights[tr] / ds.weights[tr].mean()
    rows = []
    for name in method_names:
        m = METHODS[name]
        hp = {**m.module_cells[0], "k": k if m.family != "control" or "Ceiling" not in name else 0}
        V, cert = m.fit(rep_tr, hp, seed, ctx)
        Str, Ste = rep_tr[m.key] @ V, rep_te[m.key] @ V
        mu, sd = Str.mean(0), Str.std(0) + 1e-12
        P1, P2 = np.column_stack([Ctr, (Str - mu) / sd]), np.column_stack([Cte, (Ste - mu) / sd])
        rf = RandomForestClassifier(profile.rf_trees, min_samples_leaf=5, random_state=seed, n_jobs=1)
        risk = rf.fit(P1, ds.event[tr], sample_weight=sw).predict_proba(P2)[:, 1]
        base = dict(dataset=ds.name, protocol="survival", split=split, method=name, variant="fixed", model="forest",
                    hp=str(hp), cert_converged=cert.get("converged"), cert_defect=cert.get("defect_D"))
        rows.append(dict(base, metric="cindex", value=harrell_c(ds.time[te], ds.event[te], risk)))
        rows.append(dict(base, metric="auc", value=float(roc_auc_score(ds.event[te], risk))))
    return rows


@protocol("survival")
def run(ds, method_names, profile, jobs=1, seed0=0, n_outer=None, max_n=None, k=5, **_):
    if ds.time is None or ds.event is None or ds.weights is None or ds.covariates is None:
        raise ValueError(f"{ds.name} needs time, event, weights and covariates for the survival protocol")
    if max_n:
        ds = ds.subsample(max_n, seed=seed0)
    items = [(ds, s, tr, te, method_names, profile, k) for s, tr, te in outer_splits(ds, n_outer or profile.n_outer, seed0)]
    return pd.DataFrame([r for part in parallel_map(_split, items, jobs) for r in part])
