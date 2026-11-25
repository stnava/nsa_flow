"""Factor-analysis methods for the ``factor_analysis`` protocol.

``fit(X_train, k, hp, seed, keys=None) -> FAResult`` takes raw items (samples x items); each method builds the
correlation matrix itself.  ``keys`` are optional +1/-1 reverse-keying signs (used only by the keyed non-negative
method).  ``default`` is the pre-registered setting (``w = 0.5`` for NSA); ``grid`` is what the tuned variant searches
(by held-out reproduced-correlation error, an unsupervised criterion).

Reference points: ``PAF varimax`` (orthogonal), ``PAF promax`` and ``PAF oblimin`` (oblique; they preserve the PAF fit
exactly and are the right baselines for any simple-structure method).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..registry import FA_METHODS
from .nsa_fa import nsa_fa, nsa_rotation
from .paf import (FAResult, _result, corr, ml_varimax, paf, paf_oblimin, paf_promax, paf_varimax, pca_varimax,
                  threshold_loadings)


@dataclass
class FAMethod:
    name: str
    fit: Callable
    default: dict = field(default_factory=dict)
    grids: dict = field(default_factory=dict)          # profile name -> list of hp dicts (tuned variant)
    family: str = "baseline"

    def grid(self, profile):
        return self.grids.get(profile.name, self.grids.get("standard", []))


def _fit_paf(X, k, hp, seed, keys=None):
    return paf(corr(X), k)


def _fit_paf_varimax(X, k, hp, seed, keys=None):
    return paf_varimax(corr(X), k)


def _fit_promax(X, k, hp, seed, keys=None):
    return paf_promax(corr(X), k)


def _fit_oblimin(X, k, hp, seed, keys=None):
    return paf_oblimin(corr(X), k)


def _fit_nsa_shrink(X, k, hp, seed, keys=None):
    return nsa_fa(corr(X), k, w=hp["w"])


def _fit_nsa_rotation(X, k, hp, seed, keys=None):
    return nsa_rotation(corr(X), k, w=hp["w"])


def _fit_nsa_keyed(X, k, hp, seed, keys=None):
    if keys is None:
        raise ValueError("NSA keyed non-negative needs item keys")
    return nsa_rotation(corr(X), k, w=hp["w"], keys=keys, nonneg=True)


def _fit_ml(X, k, hp, seed, keys=None):
    return ml_varimax(X, k, seed)


def _fit_pca(X, k, hp, seed, keys=None):
    return pca_varimax(corr(X), k)


def _fit_thresholded(X, k, hp, seed, keys=None):
    res = paf_promax(corr(X), k)                       # threshold the promax pattern, keep its factor correlations
    return _result(threshold_loadings(res.loadings, hp["tau"]), res.phi, res, res.cert)


_W = {"smoke": [{"w": 0.5}], "quick": [{"w": 0.25}, {"w": 0.5}, {"w": 0.9}],
      "standard": [{"w": 0.25}, {"w": 0.5}, {"w": 0.75}, {"w": 0.9}]}
_TAU = {"smoke": [{"tau": 0.3}], "quick": [{"tau": 0.2}, {"tau": 0.3}, {"tau": 0.4}],
        "standard": [{"tau": 0.2}, {"tau": 0.3}, {"tau": 0.4}]}


def _register():
    add = FA_METHODS.add
    add("PAF", FAMethod("PAF", _fit_paf))
    add("PAF varimax", FAMethod("PAF varimax", _fit_paf_varimax))
    add("PAF promax", FAMethod("PAF promax", _fit_promax))
    add("PAF oblimin", FAMethod("PAF oblimin", _fit_oblimin))
    add("NSA rotation", FAMethod("NSA rotation", _fit_nsa_rotation, {"w": 0.5}, {}, "nsa"))
    add("NSA keyed non-negative", FAMethod("NSA keyed non-negative", _fit_nsa_keyed, {"w": 0.5}, _W, "nsa"))
    add("NSA.FA shrink", FAMethod("NSA.FA shrink", _fit_nsa_shrink, {"w": 0.5}, {}, "nsa"))
    add("ML varimax", FAMethod("ML varimax", _fit_ml))
    add("PCA varimax", FAMethod("PCA varimax", _fit_pca))
    add("Thresholded promax", FAMethod("Thresholded promax", _fit_thresholded, {"tau": 0.3}, _TAU))


_register()
DEFAULT_FA = ["PAF varimax", "PAF promax", "PAF oblimin", "NSA rotation", "NSA keyed non-negative", "NSA.FA shrink",
              "ML varimax", "PCA varimax", "Thresholded promax"]
