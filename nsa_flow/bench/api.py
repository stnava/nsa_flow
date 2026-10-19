"""Standalone API: benchmark NSA-Flow on your own arrays with the same protocols and schema as the suites."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import protocols  # noqa: F401  (registers protocols)
from .datasets.base import Dataset
from .methods import DEFAULT_MODULES, DEFAULT_PREDICT
from .metrics import long_summary
from .registry import PROTOCOLS
from .runtime import get_profile


@dataclass
class Evaluation:
    rows: pd.DataFrame
    summary: pd.DataFrame

    def report(self) -> str:
        from .report import result_tables
        return result_tables(self.summary)


def evaluate(X, y=None, groups=None, *, task=None, covariates=None, ordered=False, standardize=True,
             methods=None, profile="quick", jobs=1, name="user", seed=0) -> Evaluation:
    """Benchmark NSA-Flow and baselines on ``X`` (samples x features).

    ``y`` given -> nested-CV prediction (task inferred: <= 20 distinct integer values means classification).
    ``groups`` given (one id per feature, -1 for unlabelled) -> module recovery against that truth.
    ``ordered=True`` adds contiguity for features with a natural order.  Both may be given.
    """
    X = np.asarray(X, float)
    if y is None and groups is None and not ordered:
        raise ValueError("give y (prediction) and/or groups/ordered (module recovery)")
    if y is not None:
        y = np.asarray(y)
        task = task or ("classification" if np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) <= 20
                        else "regression")
    ds = Dataset(name, X, y=y, covariates=covariates, groups=None if groups is None else np.asarray(groups),
                 ordered=ordered, task=task, standardize=standardize,
                 outer="stratified_kfold" if task == "classification" else "holdout")
    prof = get_profile(profile)
    frames = []
    if y is not None:
        frames.append(PROTOCOLS["nested_cv"](ds, methods or DEFAULT_PREDICT, prof, jobs=jobs, seed0=seed))
    if groups is not None or ordered:
        frames.append(PROTOCOLS["module_recovery"](ds, methods or DEFAULT_MODULES, prof, jobs=jobs, seed0=seed))
    rows = pd.concat(frames, ignore_index=True)
    valid = rows[rows.value.notna()]
    return Evaluation(rows, long_summary(valid, ["PCA", "KMeans features", "NMF"]))


def evaluate_fa(X, n_factors, groups=None, *, keys=None, true_loadings=None, methods=None, profile="quick",
                jobs=1, name="user", seed=0) -> Evaluation:
    """Factor-analysis benchmark on your own items (samples x items) with ``n_factors`` factors.

    ``groups`` (item -> scale id, -1 unlabelled) enables recovery (ARI) and score validity; ``keys`` (+1/-1 per item)
    reverse-keys items when scale totals are formed; ``true_loadings`` (items x factors) enables Tucker congruence.
    Same protocol, metrics and schema as the ``factor_analysis`` suites.
    """
    from .fa.methods import DEFAULT_FA
    X = np.asarray(X, float)
    meta = {"n_factors": int(n_factors), "keys": np.ones(X.shape[1]) if keys is None else np.asarray(keys, float)}
    if true_loadings is not None:
        meta["true_loadings"] = np.asarray(true_loadings, float)
    ds = Dataset(name, X, groups=None if groups is None else np.asarray(groups), standardize=True, meta=meta)
    rows = PROTOCOLS["factor_analysis"](ds, methods or DEFAULT_FA, get_profile(profile), jobs=jobs, seed0=seed)
    valid = rows[rows.value.notna()]
    return Evaluation(rows, long_summary(valid, ["PAF varimax", "PAF"]))
