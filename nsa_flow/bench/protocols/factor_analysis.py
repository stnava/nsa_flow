"""Factor-analysis protocol: interpretability, recovery, fit, validity and stability on held-out respondents.

Repeated 80/20 splits of the respondents.  Every method is fitted on the training rows only (correlation matrix
built from them); the number of factors is the dataset's known number.  Each method gets a ``fixed`` row at its
pre-registered default (``w = 0.5`` for NSA.FA) and, if it has a grid, a ``tuned`` row whose hyper-parameter is
chosen by inner CV on the training rows with an unsupervised criterion (held-out reproduced-correlation error).

Metrics (per split, per method):
  hofmann, cross_loadings, sparsity, ortho_defect    interpretability of the (pattern) loadings
  factor_corr                                        mean |off-diagonal| factor correlation (0 for orthogonal methods)
  ari, tucker                                        recovery of the known item->scale map / true loadings
  recon_err, heywood                                 test off-diagonal reproduced-correlation error of L Phi L'; items
                                                     with model-implied communality > 1
  cfa_rmsea, cfa_cfi, cfa_tli, cfa_srmr, cfa_bic     ML-CFA of the method's own simple structure on the test covariance
                                                     (only when the test split has >= 5 respondents per item; it is
                                                     identical across methods whenever their assignments agree)
  score_validity                                     mean |r| between held-out regression factor scores and the
                                                     matched keyed scale totals
  stability_ari, stability_tucker                    agreement of two fits on disjoint halves of the training rows
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .. import methods as _methods  # noqa: F401  (registry side effects)
from ..fa import methods as _fa_methods  # noqa: F401  (registers FA_METHODS)
from ..fa.cfa import fit_cfa
from ..fa.paf import corr
from ..metrics import (cross_loadings, dominant_assignment, hofmann_complexity, orthogonality_defect,
                       reproduced_correlation_error, tucker_congruence)
from ..registry import FA_METHODS, protocol
from ..runtime import limit_threads, parallel_map


MIN_CFA_RATIO = 5


def _splits(n, n_splits, seed0=0):
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    for s in range(n_splits):
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed0 + s)
        yield s, tr, te


def _keys(ds):
    return ds.meta.get("keys")


def select_hp(ds, method, tr, seed, profile, k):
    """Inner CV over the training rows; criterion = held-out reproduced-correlation error."""
    from sklearn.model_selection import KFold
    grid = method.grid(profile)
    if len(grid) <= 1:
        return (grid[0] if grid else method.default), np.nan
    folds = list(KFold(profile.n_inner, shuffle=True, random_state=seed).split(tr))
    errs = []
    for hp in grid:
        e = []
        for a, b in folds:
            res = method.fit(ds.X[tr[a]], k, hp, seed, keys=_keys(ds))
            e.append(reproduced_correlation_error(corr(ds.X[tr[b]]), res.loadings, res.phi))
        errs.append(np.mean(e))
    best = int(np.argmin(errs))
    return grid[best], float(errs[best])


def _scale_totals(ds, Z):
    """Keyed scale totals from standardised items: one column per known scale."""
    keys = ds.meta.get("keys", np.ones(ds.p))
    groups = ds.groups
    return np.column_stack([(Z[:, groups == g] * keys[groups == g]).sum(1) for g in sorted(set(groups[groups >= 0]))])


def _score_validity(ds, L, phi, tr, te):
    from scipy.optimize import linear_sum_assignment
    mu, sd = ds.X[tr].mean(0), ds.X[tr].std(0, ddof=1) + 1e-12
    Zte = (ds.X[te] - mu) / sd
    R = corr(ds.X[tr])
    P = np.eye(L.shape[1]) if phi is None else phi
    F = Zte @ np.linalg.solve(R + 1e-8 * np.eye(len(R)), L @ P)           # regression (Thurstone) scores W = R^-1 L Phi
    T = _scale_totals(ds, Zte)
    C = np.abs(np.corrcoef(F.T, T.T)[: F.shape[1], F.shape[1]:])
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())


def _cfa_metrics(ds, L, Rte_cov, n_te):
    lab = dominant_assignment(L, rel=1e-6)
    keep = lab >= 0
    if keep.sum() < 4:
        return {}
    fit = fit_cfa(Rte_cov[np.ix_(keep, keep)], n_te, lab[keep])
    if fit is None or not np.isfinite(fit.rmsea):
        return {}
    return {"cfa_rmsea": fit.rmsea, "cfa_cfi": fit.cfi, "cfa_tli": fit.tli, "cfa_srmr": fit.srmr, "cfa_bic": fit.bic}


def _evaluate(ds, res, tr, te, k, seed, method, hp):
    from sklearn.metrics import adjusted_rand_score
    L = res.loadings
    Rte = corr(ds.X[te])
    vals = {"hofmann": float(hofmann_complexity(L).mean()), "cross_loadings": float(cross_loadings(L)),
            "sparsity": float((np.abs(L) < 0.1).mean()), "ortho_defect": orthogonality_defect(L),
            "recon_err": reproduced_correlation_error(Rte, L, res.phi),
            "heywood": float((np.diag(res.implied()) > 1.0).sum()),
            "factor_corr": 0.0 if res.phi is None else float(np.abs(res.phi[~np.eye(len(res.phi), dtype=bool)]).mean())}
    if ds.groups is not None:
        lab = dominant_assignment(L)
        lab_idx = ds.groups >= 0
        vals["ari"] = float(adjusted_rand_score(ds.groups[lab_idx], lab[lab_idx]))
        vals["score_validity"] = _score_validity(ds, L, res.phi, tr, te)
    if "true_loadings" in ds.meta:
        vals["tucker"] = tucker_congruence(L, ds.meta["true_loadings"])[0]
    if len(te) >= MIN_CFA_RATIO * ds.p:       # CFA on a test covariance needs at least ~5 respondents per item
        vals.update(_cfa_metrics(ds, L, np.cov(ds.X[te].T), len(te)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr)
    ha, hb = perm[: len(tr) // 2], perm[len(tr) // 2:]
    La = method.fit(ds.X[ha], k, hp, seed, keys=_keys(ds)).loadings
    Lb = method.fit(ds.X[hb], k, hp, seed, keys=_keys(ds)).loadings
    vals["stability_ari"] = float(adjusted_rand_score(dominant_assignment(La), dominant_assignment(Lb)))
    vals["stability_tucker"] = tucker_congruence(La, Lb)[0]
    return vals


def _one_split(ds, split, tr, te, method_names, profile):
    limit_threads(1)
    k = int(ds.meta["n_factors"])
    seed = 1000 + split
    rows = []
    for name in method_names:
        m = FA_METHODS[name]
        variants = [("fixed", name, m.default, np.nan)]
        if m.grid(profile) and len(m.grid(profile)) > 1:
            hp_t, inner = select_hp(ds, m, tr, seed, profile, k)
            variants.append(("tuned", f"{name} (tuned)", hp_t, inner))
        for variant, label, hp, inner in variants:
            t0 = time.time()
            res = m.fit(ds.X[tr], k, hp, seed, keys=_keys(ds))
            vals = _evaluate(ds, res, tr, te, k, seed, m, hp)
            cert = res.cert or {}
            for metric, v in vals.items():
                rows.append(dict(dataset=ds.name, protocol="factor_analysis", split=split, method=label,
                                 variant=variant, model="", metric=metric, value=float(v), inner_value=inner,
                                 hp=json.dumps(hp), seconds=time.time() - t0,
                                 cert_converged=cert.get("converged"), cert_defect=cert.get("defect_D")))
    return rows


@protocol("factor_analysis")
def run(ds, method_names, profile, jobs=1, seed0=0, n_outer=None, **_):
    if ds.meta.get("n_factors") is None:
        raise ValueError(f"{ds.name} has no known number of factors (meta['n_factors'])")
    items = [(ds, s, tr, te, method_names, profile) for s, tr, te in _splits(ds.n, n_outer or profile.n_outer, seed0)]
    return pd.DataFrame([r for part in parallel_map(_one_split, items, jobs) for r in part])
