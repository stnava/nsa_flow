"""Declarative suites.  Each entry PRE-REGISTERS its primary contrast and pass rule before any run.

``Primary(metric, method, ref, model)``: the contrast ``method - ref`` on ``metric`` (paired over
splits/draws).  ``expect``:
  "effect"  verdict PASS if the 95% interval for the difference excludes 0 in the stated direction
  "null"    control: with ``ref`` set, PASS if the interval for the difference includes 0; with
            ``ref=None`` the method's own mean must be within ``tol`` of 0 (e.g. stability on noise)
  "info"    reported, no verdict
The manifest stores the whole suite, so the primary is never chosen after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .methods import DEFAULT_MODULES, DEFAULT_PREDICT
from .registry import SUITES


@dataclass
class Primary:
    metric: str
    method: str
    ref: Optional[str]
    model: str = ""
    direction: str = "gt"
    tol: float = 0.05


@dataclass
class Entry:
    dataset: str
    protocol: str
    methods: list
    primary: Optional[Primary] = None
    expect: str = "info"
    params: dict = field(default_factory=dict)


@dataclass
class Suite:
    name: str
    description: str
    entries: list


NSA = "NSA consolidated"
FIND = "NSA consolidated w=0.5"
MOD = DEFAULT_MODULES


def _pred(ds, metric, model, expect="effect", methods=None, **params):
    return Entry(ds, "nested_cv", methods or DEFAULT_PREDICT, Primary(metric, NSA, "PCA", model), expect, params)


def _mod(ds, expect="effect", metric="ari", ref="KMeans features"):
    return Entry(ds, "module_recovery", MOD, Primary(metric, FIND, ref), expect)


SUITES.add("smoke", Suite("smoke", "seconds, synthetic only; guards the harness and the solver", [
    _pred("planted", "r2", "forest", "info", ["PCA", "NSA consolidated", "KMeans features"]),
    Entry("planted", "module_recovery", ["NSA consolidated", "NSA nonneg", "KMeans features", "PCA"],
          Primary("ari", FIND, "KMeans features"), "info"),
    Entry("noise", "module_recovery", ["NSA consolidated", "KMeans features", "PCA"],
          Primary("stability", FIND, None), "null"),
    Entry("noise", "speed", ["PCA", "NSA consolidated", "NSA nonneg"]),
]))

SUITES.add("standard", Suite("standard", "reported benchmarks: prediction, module recovery, null controls, speed", [
    # prediction (primary: NSA consolidated vs tuned PCA, forest for regression, linear for classification)
    _pred("adni", "r2", "forest"),
    _pred("golub", "balanced_accuracy", "linear", n_outer=10, n_inner=3),   # NSA at p=2000 is ~2 s/fit: 10 splits, 3 inner folds (n=72)
    _pred("tecator", "r2", "forest"),
    # module recovery against known groups (primary: ARI of pre-specified NSA consolidated w=0.5 vs k-means)
    _mod("planted"),
    _mod("newsgroups"),
    _mod("metabric"),
    _mod("pbmc3k"),
    Entry("tecator", "module_recovery", MOD, Primary("contiguity", FIND, "KMeans features"), "info"),
    # null controls: structure must not be invented; selection must not add optimism
    _pred("adni@null_y", "r2", "forest", "null", ["PCA", "NSA consolidated", "KMeans features"]),
    Entry("noise", "module_recovery", ["NSA consolidated", "KMeans features", "PCA"],
          Primary("stability", FIND, None), "null"),
    Entry("metabric@shuffled", "module_recovery", ["NSA consolidated", "KMeans features", "PCA"],
          Primary("stability", FIND, None), "null"),
    Entry("speed", "speed", ["PCA", "NSA consolidated", "NSA nonneg"]),
]))
SUITES["paper"] = SUITES["standard"]
