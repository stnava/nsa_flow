"""Declarative suites.  Each entry PRE-REGISTERS its primary contrast and pass rule before any run.

``Primary(metric, method, ref, model)``: the contrast ``method - ref`` on ``metric`` (paired over
splits/draws).  ``expect``:
  "effect"  verdict PASS if the 95% interval for the difference excludes 0 in the stated direction
  "null"    control: with ``ref`` set, PASS if the interval for the difference includes 0; with
            ``ref=None`` the method's own mean must be within ``tol`` of 0 (e.g. stability on noise)
  "info"    reported, no verdict
``guards`` add non-inferiority conditions on other metrics of the same method-vs-reference contrast: for a
lower-is-better metric ("lt") the upper CI bound of the difference must be below ``tol``; for higher-is-better
("gt") the lower bound must be above ``-tol``.  A PASS whose guard fails is reported as FAIL (guard).
The manifest stores the whole suite, so the primary is never chosen after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .fa.methods import DEFAULT_FA
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
    guards: tuple = ()          # non-inferiority guards: (metric, "lt"|"gt", tol); PASS needs every guard to hold


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


def _fa(ds):
    """Interpretability gain of NSA.FA (w=0.5) over PAF+varimax, guarded by non-inferior fit.

    The fit guard is the held-out reproduced-correlation error (+0.02 allowed).  An earlier draft guarded on CFA
    RMSEA/CFI, which was vacuous: the CFA is fitted on each method's own dominant-item assignment, so it is
    identical across methods whenever the assignments agree and cannot register loading shrinkage.  The CFA
    indices are still reported, and kept as secondary guards."""
    return Entry(ds, "factor_analysis", DEFAULT_FA,
                 Primary("hofmann", "NSA.FA", "PAF varimax", direction="lt",
                         guards=(("recon_err", "lt", 0.02), ("cfa_rmsea", "lt", 0.01),
                                 ("cfa_cfi", "gt", 0.02))), "effect")


def _fa_null(ds):
    return Entry(ds, "factor_analysis", ["PAF varimax", "NSA.FA"], Primary("ari", "NSA.FA", None, tol=0.1), "null")


NHANES_METHODS = ["PCA", "NMF", "NSA nonneg", "NSA nonneg (shuffled support)", "Ceiling (all features)"]
# NHANES is n~48k: cap the rows (stratified) and the splits so each entry stays a minutes-scale job
NHANES_CAP = dict(max_n=20000, n_outer=10, n_inner=3)   # a first 6,000-row pass gave +0.0065 [0.0003, 0.0127] raw


def _nhanes(ds="nhanes"):
    """Pre-registered: does the non-negative NSA basis beat PCA on AUC (forest) with covariates in every model?
    The shuffled-support control and the all-features ceiling are reported alongside, without verdicts."""
    return Entry(ds, "nested_cv", NHANES_METHODS, Primary("auc", "NSA nonneg", "PCA", "forest"), "effect",
                 dict(NHANES_CAP))


SUITES.add("smoke", Suite("smoke", "seconds, synthetic only; guards the harness and the solver", [
    _pred("planted", "r2", "forest", "info", ["PCA", "NSA consolidated", "KMeans features"]),
    Entry("planted", "module_recovery", ["NSA consolidated", "NSA nonneg", "KMeans features", "PCA"],
          Primary("ari", FIND, "KMeans features"), "info"),
    Entry("noise", "module_recovery", ["NSA consolidated", "KMeans features", "PCA"],
          Primary("stability", FIND, None), "null"),
    Entry("noise", "speed", ["PCA", "NSA consolidated", "NSA nonneg"]),
    Entry("fa_sim_n300", "factor_analysis", ["PAF varimax", "NSA.FA"],
          Primary("hofmann", "NSA.FA", "PAF varimax", direction="lt"), "info"),
]))

SUITES.add("fa", Suite("fa", "factor analysis: interpretability vs fit, recovery, stability; with null controls", [
    _fa("bfi"), _fa("holzinger"), _fa("fa_sim_n100"), _fa("fa_sim_n300"), _fa("fa_sim_n1000"),
    _fa_null("fa_null"), _fa_null("bfi@shuffled"),
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
    _fa("bfi"), _fa("holzinger"), _fa("fa_sim_n100"), _fa("fa_sim_n300"), _fa("fa_sim_n1000"),
    _fa_null("fa_null"), _fa_null("bfi@shuffled"),
    _nhanes("nhanes"), _nhanes("nhanes@density"),
    Entry("nhanes@density", "survival", NHANES_METHODS, Primary("cindex", "NSA nonneg", "PCA", "forest"), "effect",
          dict(max_n=20000, n_outer=10)),
]))
SUITES["paper"] = SUITES["standard"]

SUITES.add("nhanes", Suite("nhanes", "NHANES dietary -> mortality: AUC, energy-adjusted variants, survival C-index", [
    _nhanes("nhanes"), _nhanes("nhanes@density"), _nhanes("nhanes@residual"),
    Entry("nhanes@density", "survival", NHANES_METHODS, Primary("cindex", "NSA nonneg", "PCA", "forest"), "effect",
          dict(max_n=20000, n_outer=10)),
]))
