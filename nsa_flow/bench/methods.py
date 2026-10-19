"""Every method the benchmark compares, with its tuning grid, defined once.

``fit(rep, hp, seed, ctx) -> (V, cert)`` returns a ``p x k`` basis and a certificate dict (empty for
non-NSA methods; NSA methods report ``converged`` and ``defect_D`` from the solver).  ``rep`` holds the
training representations from ``Dataset.prep`` (``c`` centred/standardised, ``pos`` shifted non-negative);
``hp`` always contains ``k``; ``ctx`` carries dataset conventions (``spherical`` k-means).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .registry import METHODS


def _grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]


@dataclass
class Method:
    name: str
    fit: Callable
    key: str                                   # representation used to project: "c" or "pos"
    grids: dict                                # profile name -> list of hp dicts (each contains k)
    module_cells: list = field(default_factory=lambda: [{}])   # extra hp for module recovery (k set by protocol)
    family: str = "baseline"                   # "nsa" | "baseline"
    max_p: int | None = None                   # skipped (visibly) above this many features: cost, not principle
    note: str = ""

    def grid(self, profile, p=None):
        g = self.grids.get(profile.name, self.grids["standard"])
        if self.family == "nsa" and p is not None and p > 500 and profile.name == "standard":
            # NSA costs ~2 s/fit at p=2000: coarse w grid on high-p data (recorded in the manifest)
            g = [h for h in g if h.get("w") in (0.25, 0.5, 0.75)]
        return g

    def applies(self, ds) -> tuple[bool, str]:
        if self.max_p is not None and ds.p > self.max_p:
            return False, f"{self.name} skipped: p={ds.p} > {self.max_p} (cost; {self.note})"
        return True, ""


def _nsa(rep, hp, seed, ctx, mode, consolidate=False):
    from nsa_flow import nsa_flow
    if mode == "signed":
        r = nsa_flow(rep["c"], k=hp["k"], mode="signed", w=hp["w"], consolidate=consolidate)
    else:
        r = nsa_flow(rep["pos"], k=hp["k"], mode="data", w=hp["w"])
    return r.Y.detach().cpu().numpy(), {"converged": bool(r["converged"]), "defect_D": float(r["defect_D"])}


def _pca(rep, hp, seed, ctx):
    from sklearn.decomposition import PCA
    return PCA(n_components=hp["k"], random_state=seed).fit(rep["c"]).components_.T, {}


def _thresholded_pca(rep, hp, seed, ctx):
    V = _pca(rep, hp, seed, ctx)[0].copy()
    keep = max(1, int(round(hp["frac"] * V.shape[0])))
    for j in range(V.shape[1]):
        cut = np.sort(np.abs(V[:, j]))[-keep]
        V[np.abs(V[:, j]) < cut, j] = 0.0
        V[:, j] /= np.linalg.norm(V[:, j])
    return V, {}


def _sparse_pca(rep, hp, seed, ctx):
    from sklearn.decomposition import SparsePCA
    return SparsePCA(n_components=hp["k"], alpha=hp["alpha"], random_state=seed,
                     max_iter=100).fit(rep["c"]).components_.T, {}


def _nmf(rep, hp, seed, ctx):
    from sklearn.decomposition import NMF
    return NMF(n_components=hp["k"], init="nndsvd", random_state=seed, max_iter=500,
               tol=1e-4).fit(rep["pos"]).components_.T, {}


def _kmeans(rep, hp, seed, ctx):
    from sklearn.cluster import KMeans
    W = rep["c"].T
    if ctx.get("spherical"):
        W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    lab = KMeans(hp["k"], n_init=3, random_state=seed).fit_predict(W)
    V = np.zeros((rep["c"].shape[1], hp["k"]))
    V[np.arange(V.shape[0]), lab] = 1.0 / np.sqrt(np.bincount(lab, minlength=hp["k"]))[lab]
    return V, {}


def _ceiling(rep, hp, seed, ctx):
    """No reduction: V = identity, so the model sees every feature (the ceiling arm)."""
    return np.eye(rep["c"].shape[1]), {}


def _nsa_nonneg_shuffled(rep, hp, seed, ctx):
    """Control: the fitted non-negative basis with its feature assignment permuted (density, column norms and the
    whole value multiset preserved; only WHICH feature each weight belongs to is destroyed)."""
    V, cert = _nsa(rep, hp, seed, ctx, "data")
    return V[np.random.default_rng(seed).permutation(V.shape[0])], cert


def _nsa_grids():
    return {"smoke": _grid(w=[0.5], k=[3]),
            "quick": _grid(w=[0.25, 0.5, 0.75], k=[3, 5]),
            "standard": _grid(w=[0.1, 0.25, 0.5, 0.75, 0.9], k=[3, 5, 8])}


def _simple_grids(**extra):
    return {"smoke": _grid(k=[3], **{a: b[:1] for a, b in extra.items()}),
            "quick": _grid(k=[3, 5], **{a: b[:2] for a, b in extra.items()}),
            "standard": _grid(k=[3, 5, 8], **extra)}


def _register():
    reg = METHODS.add
    reg("PCA", Method("PCA", _pca, "c", _simple_grids()))
    reg("NSA signed", Method("NSA signed", lambda *a: _nsa(*a, "signed"), "c", _nsa_grids(),
                             [{"w": 0.5}, {"w": 0.25}], family="nsa"))
    reg("NSA consolidated", Method("NSA consolidated", lambda *a: _nsa(*a, "signed", True), "c", _nsa_grids(),
                                   [{"w": 0.5}, {"w": 0.25}], family="nsa"))
    reg("NSA nonneg", Method("NSA nonneg", lambda *a: _nsa(*a, "data"), "pos", _nsa_grids(),
                             [{"w": 0.5}], family="nsa", note="mode='data' on non-negatively shifted data"))
    reg("Thresholded PCA", Method("Thresholded PCA", _thresholded_pca, "c",
                                  _simple_grids(frac=[0.1, 0.2, 0.4]), [{"frac": 0.2}]))
    reg("SparsePCA", Method("SparsePCA", _sparse_pca, "c", _simple_grids(alpha=[1.0, 4.0]), [{"alpha": 1.0}],
                            max_p=500, note="sklearn SparsePCA is ~18 s/fit at p=2000"))
    reg("NMF", Method("NMF", _nmf, "pos", _simple_grids()))
    reg("KMeans features", Method("KMeans features", _kmeans, "c", _simple_grids()))
    one = {"smoke": [{"k": 0}], "quick": [{"k": 0}], "standard": [{"k": 0}]}
    reg("Ceiling (all features)", Method("Ceiling (all features)", _ceiling, "c", one, family="control",
                                          note="V = I; upper reference for any reduction"))
    reg("NSA nonneg (shuffled support)", Method("NSA nonneg (shuffled support)", _nsa_nonneg_shuffled, "pos",
                                                _nsa_grids(), [{"w": 0.5}], family="control",
                                                note="fitted non-negative basis, feature assignment permuted"))


_register()

DEFAULT_PREDICT = ["PCA", "NSA signed", "NSA consolidated", "Thresholded PCA", "SparsePCA", "NMF",
                   "KMeans features"]
DEFAULT_MODULES = ["NSA consolidated", "NSA nonneg", "KMeans features", "NMF", "PCA"]
