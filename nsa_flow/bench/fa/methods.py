"""Factor-analysis methods for the ``factor_analysis`` protocol.

``fit(X_train, k, hp, seed) -> FAResult`` takes raw items (samples x items); each method builds the correlation
matrix itself.  ``default`` is the pre-registered setting (``w = 0.5`` for the NSA methods); ``grid`` is what the
tuned variant searches (by held-out reproduced-correlation error, an unsupervised criterion).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..registry import FA_METHODS
from .nsa_fa import nsa_fa, nsa_fa_post
from .paf import FAResult, corr, ml_varimax, paf, paf_varimax, pca_varimax, threshold_loadings, varimax


@dataclass
class FAMethod:
    name: str
    fit: Callable
    default: dict = field(default_factory=dict)
    grids: dict = field(default_factory=dict)          # profile name -> list of hp dicts (tuned variant)
    family: str = "baseline"

    def grid(self, profile):
        return self.grids.get(profile.name, self.grids.get("standard", []))


def _fit_paf(X, k, hp, seed):
    return paf(corr(X), k)


def _fit_paf_varimax(X, k, hp, seed):
    return paf_varimax(corr(X), k)


def _fit_nsa_fa(X, k, hp, seed):
    return nsa_fa(corr(X), k, w=hp["w"])


def _fit_nsa_fa_post(X, k, hp, seed):
    return nsa_fa_post(corr(X), k, w=hp["w"])


def _fit_ml(X, k, hp, seed):
    return ml_varimax(X, k, seed)


def _fit_pca(X, k, hp, seed):
    return pca_varimax(corr(X), k)


def _fit_thresholded(X, k, hp, seed):
    res = paf_varimax(corr(X), k)
    L = threshold_loadings(res.loadings, hp["tau"])
    return FAResult(L, (L ** 2).sum(1), res.iterations, res.converged, res.cert)


_W = {"smoke": [{"w": 0.5}], "quick": [{"w": 0.25}, {"w": 0.5}, {"w": 0.75}],
      "standard": [{"w": 0.1}, {"w": 0.25}, {"w": 0.5}, {"w": 0.75}, {"w": 0.9}]}
_TAU = {"smoke": [{"tau": 0.3}], "quick": [{"tau": 0.2}, {"tau": 0.3}, {"tau": 0.4}],
        "standard": [{"tau": 0.2}, {"tau": 0.3}, {"tau": 0.4}]}


def _register():
    add = FA_METHODS.add
    add("PAF", FAMethod("PAF", _fit_paf))
    add("PAF varimax", FAMethod("PAF varimax", _fit_paf_varimax))
    add("NSA.FA", FAMethod("NSA.FA", _fit_nsa_fa, {"w": 0.5}, _W, "nsa"))
    add("NSA.FA post", FAMethod("NSA.FA post", _fit_nsa_fa_post, {"w": 0.5}, _W, "nsa"))
    add("ML varimax", FAMethod("ML varimax", _fit_ml))
    add("PCA varimax", FAMethod("PCA varimax", _fit_pca))
    add("Thresholded PAF", FAMethod("Thresholded PAF", _fit_thresholded, {"tau": 0.3}, _TAU))


_register()
DEFAULT_FA = ["PAF varimax", "NSA.FA", "NSA.FA post", "ML varimax", "PCA varimax", "Thresholded PAF"]
