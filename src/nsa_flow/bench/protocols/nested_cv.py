"""Predictive protocol: nested cross-validation of every method's hyper-parameters.

Each method (PCA included) picks its own hyper-parameters on the outer-training portion only, by inner
CV, separately for each downstream model (linear / forest), and is scored once on the outer test portion.
NSA methods also get a fixed ``w=0.5, k=5`` row on the same splits so the cost of selecting blind is
visible.  Everything -- centring, feature selection, standardisation, basis, model -- is refitted inside
each training portion.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .. import methods as _methods  # noqa: F401  (populates METHODS in workers)
from ..metrics import sparsity, support_overlap
from ..registry import METHODS, protocol
from ..runtime import limit_threads, parallel_map

MODELS = ("linear", "forest")
FIXED = {"w": 0.5, "k": 5}


# ------------------------------------------------------------------ splits and scoring
def outer_splits(ds, n_splits, seed0=0):
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, train_test_split
    idx = np.arange(ds.n)
    if ds.outer == "holdout":
        for s in range(n_splits):
            tr, te = train_test_split(idx, test_size=0.2, random_state=seed0 + s)
            yield s, tr, te
        return
    reps = int(np.ceil(n_splits / 5))
    if ds.outer == "kfold":
        it = RepeatedKFold(n_splits=5, n_repeats=reps, random_state=seed0).split(idx)
    else:
        it = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=seed0).split(idx, ds.y)
    for s, (tr, te) in enumerate(it):
        if s >= n_splits:
            break
        yield s, tr, te


def inner_splits(ds, tr, n_inner, seed):
    from sklearn.model_selection import KFold, StratifiedKFold
    if ds.task == "classification":
        it = StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(tr, ds.y[tr])
    else:
        it = KFold(n_inner, shuffle=True, random_state=seed).split(tr)
    for a, b in it:
        yield tr[a], tr[b]


def score_models(ds, V, key, rep_tr, rep_te, tr, te, seed, trees, rotate=False):
    """{model: score}: R^2 (regression), balanced accuracy or AUC (classification) for linear and forest.
    ``rotate=True`` is the axis control: the same scores rotated onto their principal axes (a linear model is invariant to
    it, a forest is not), separating "better subspace" from "axis-aligned features a tree can split on"."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
    Ztr, Zte = rep_tr[key] @ V, rep_te[key] @ V
    if rotate and Ztr.shape[1] > 1:                  # control: rotate the scores onto their principal axes (training rows)
        from sklearn.decomposition import PCA
        pc = PCA(n_components=Ztr.shape[1]).fit(Ztr)
        Ztr, Zte = pc.transform(Ztr), pc.transform(Zte)
    Ctr = Cte = None
    if ds.covariates is not None:
        Ctr, Cte = ds.covariates[tr], ds.covariates[te]
    if ds.standardize_scores:                       # z-score scores and covariates on the training rows
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-12
        Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
        if Ctr is not None:
            cm, cs = Ctr.mean(0), Ctr.std(0) + 1e-12
            Ctr, Cte = (Ctr - cm) / cs, (Cte - cm) / cs
    if Ctr is not None:
        Ztr, Zte = np.column_stack([Ctr, Ztr]), np.column_stack([Cte, Zte])
    if ds.task == "regression":
        lin, rf, metric = LinearRegression(), RandomForestRegressor(trees, random_state=seed, n_jobs=1), r2_score
        pred = lambda m: m.predict(Zte)
    elif ds.metric == "auc":
        lin = LogisticRegression(C=1.0, max_iter=1000)
        rf = RandomForestClassifier(trees, min_samples_leaf=5, random_state=seed, n_jobs=1)
        metric, pred = roc_auc_score, (lambda m: m.predict_proba(Zte)[:, 1])
    else:
        lin = LogisticRegression(C=1.0, max_iter=500)
        rf, metric = RandomForestClassifier(trees, random_state=seed, n_jobs=1), balanced_accuracy_score
        pred = lambda m: m.predict(Zte)
    return {"linear": float(metric(ds.y[te], pred(lin.fit(Ztr, ds.y[tr])))),
            "forest": float(metric(ds.y[te], pred(rf.fit(Ztr, ds.y[tr]))))}


def metric_name(ds):
    return "r2" if ds.task == "regression" else (ds.metric or "balanced_accuracy")


# ------------------------------------------------------------------ selection
def select_hp(ds, method, grid, tr, seed, profile):
    folds = list(inner_splits(ds, tr, profile.n_inner, seed))
    prepped = [ds.prep(a, b) for a, b in folds]
    ctx = {"spherical": ds.kmeans_spherical}
    inner = {m: np.zeros(len(grid)) for m in MODELS}
    for gi, hp in enumerate(grid):
        acc = {m: [] for m in MODELS}
        for (a, b), (rtr, rva) in zip(folds, prepped):
            V, _ = method.fit(rtr, hp, seed, ctx)
            sc = score_models(ds, V, method.key, rtr, rva, a, b, seed, profile.rf_trees_inner or profile.rf_trees)
            for m in MODELS:
                acc[m].append(sc[m])
        for m in MODELS:
            inner[m][gi] = np.mean(acc[m])
    best = {m: int(np.argmax(inner[m])) for m in MODELS}
    return {m: grid[best[m]] for m in MODELS}, {m: float(inner[m][best[m]]) for m in MODELS}


def _outer_split(ds, split, tr, te, method_names, profile, seed0):
    import os
    _t0 = time.time()
    try:
        return _outer_split_impl(ds, split, tr, te, method_names, profile, seed0)
    finally:
        if os.environ.get("NSA_BENCH_TRACE"):
            with open(os.environ["NSA_BENCH_TRACE"], "a") as f:
                f.write(f"{ds.name} split={split} pid={os.getpid()} {time.time() - _t0:.1f}s\n")


def _outer_split_impl(ds, split, tr, te, method_names, profile, seed0):
    limit_threads(1)
    seed = 1000 + split
    rep_tr, rep_te = ds.prep(tr, te)
    ctx = {"spherical": ds.kmeans_spherical}
    fixed_k = FIXED["k"] if profile.name != "smoke" else 3
    rows, cache = [], {}
    metric = metric_name(ds)

    def outer_fit(method, hp):
        key = (method.name, tuple(sorted(hp.items())))
        if key not in cache:
            t0 = time.time()
            V, cert = method.fit(rep_tr, hp, seed, ctx)
            sc = score_models(ds, V, method.key, rep_tr, rep_te, tr, te, seed, profile.rf_trees)
            cache[key] = (sc, sparsity(V), support_overlap(V), cert, time.time() - t0, V)
        return cache[key]

    def emit(label, method, hp, inner_best, sel_s, model, variant):
        sc, sp, ov, cert, fit_s, _ = outer_fit(method, hp)
        rows.append(dict(dataset=ds.name, protocol="nested_cv", split=split, method=label, variant=variant,
                         model=model, metric=metric, value=sc[model], inner_value=inner_best,
                         hp=json.dumps(hp), sparsity=sp, overlap=ov, seconds=sel_s + fit_s,
                         cert_converged=cert.get("converged"), cert_defect=cert.get("defect_D")))

    for name in method_names:
        method = METHODS[name]
        grid = method.grid(profile, ds.p)
        t0 = time.time()
        chosen, inner_best = select_hp(ds, method, grid, tr, seed, profile)
        sel_s = time.time() - t0
        for m in MODELS:
            emit(name, method, chosen[m], inner_best[m], sel_s, m, "tuned")
        # score collinearity of the forest-selected basis (training rows): the diagnostic behind the Tecator result
        fit_f = outer_fit(method, chosen["forest"])
        V, cert_f = fit_f[5], fit_f[3]
        S = rep_tr[method.key] @ V
        C = np.nan_to_num(np.corrcoef(S.T)) if S.shape[1] > 1 else np.zeros((1, 1))
        rows.append(dict(dataset=ds.name, protocol="nested_cv", split=split, method=name, variant="tuned", model="",
                         metric="score_collinearity", cert_converged=cert_f.get("converged"),
                         cert_defect=cert_f.get("defect_D"),
                         value=float(np.abs(C[~np.eye(len(C), dtype=bool)]).mean()) if len(C) > 1 else 0.0))
        if method.family == "nsa" and S.shape[1] > 1:
            for m in MODELS:                                       # axis control: same scores, PCA-rotated
                hp = chosen[m]
                fit_m = outer_fit(method, hp)
                Vm, cert_m = fit_m[5], fit_m[3]
                sc = score_models(ds, Vm, method.key, rep_tr, rep_te, tr, te, seed, profile.rf_trees, rotate=True)
                rows.append(dict(dataset=ds.name, protocol="nested_cv", split=split, method=f"{name} (scores PCA-rotated)",
                                 variant="control", model=m, metric=metric, value=sc[m], hp=json.dumps(hp),
                                 cert_converged=cert_m.get("converged"), cert_defect=cert_m.get("defect_D")))
        if method.family == "nsa" and name != "NSA nonneg":
            hp = {"w": FIXED["w"], "k": fixed_k}
            for m in MODELS:
                emit(f"{name} (fixed w=.5,k={fixed_k})", method, hp, np.nan, 0.0, m, "fixed")
    return rows


@protocol("nested_cv")
def run(ds, method_names, profile, jobs=1, seed0=0, n_outer=None, n_inner=None, max_n=None, **_):
    from dataclasses import replace
    if n_inner:
        profile = replace(profile, n_inner=n_inner)
    if max_n:                                    # cost cap for very large datasets (documented in the entry params)
        ds = ds.subsample(max_n, seed=seed0)
    if ds.y is None or ds.task is None:
        raise ValueError(f"{ds.name} has no outcome; nested_cv needs y")
    skipped, use = [], []
    for name in method_names:
        ok, why = METHODS[name].applies(ds)
        (use if ok else skipped).append(name if ok else (name, why))
    items = [(ds, s, tr, te, use, profile, seed0) for s, tr, te in outer_splits(ds, n_outer or profile.n_outer, seed0)]
    rows = [r for part in parallel_map(_outer_split, items, jobs) for r in part]
    df = pd.DataFrame(rows)
    for name, why in skipped:
        df = pd.concat([df, pd.DataFrame([dict(dataset=ds.name, protocol="nested_cv", method=name,
                                               skipped_reason=why)])], ignore_index=True)
    return df
