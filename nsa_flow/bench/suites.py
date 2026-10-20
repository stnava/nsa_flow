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
FIND = "NSA consolidated w=0.5"                       # signed consolidated (kept as a secondary row)
FIND_NN = "NSA nonneg (multistart) w=0.5"             # module-recovery primary: the non-negative data mode, 5 starts
MOD = DEFAULT_MODULES


def _pred(ds, metric, model, expect="effect", methods=None, **params):
    return Entry(ds, "nested_cv", methods or DEFAULT_PREDICT, Primary(metric, NSA, "PCA", model), expect, params)


def _mod(ds, expect="effect", metric="ari", ref="KMeans features", methods=None, **params):
    """Module recovery.  Corrected primary (after a diagnostic study): the NON-NEGATIVE mode with multistart against
    k-means with the same number of starts (``KMeans features``, n_init=5).  The earlier primary, signed consolidated NSA,
    finds variance-ranked contrasts and is not a clustering method; it stays in the tables as a secondary row."""
    return Entry(ds, "module_recovery", methods or MOD, Primary(metric, FIND_NN, ref), expect, params)


def _fa(ds):
    """Does NSA used as a fit-preserving ROTATION give simpler loadings than promax (the right baseline: oblique, same fit)?

    The fit guard is the held-out reproduced-correlation error (NSA rotation preserves the PAF fit by construction, so a
    +0.01 allowance is generous).  Earlier drafts compared a loading-shrinking NSA.FA with varimax and guarded on CFA
    indices; shrinking is dominated by oblique rotation and the CFA guard was vacuous (it is fitted on each method's own
    item assignment and so is identical across methods whenever assignments agree), see docs/BENCHMARKING.md.
    The keyed non-negative variant is reported in the tables as a simplicity/fit trade-off row, without a verdict."""
    return Entry(ds, "factor_analysis", DEFAULT_FA,
                 Primary("hofmann", "NSA rotation", "PAF promax", direction="lt",
                         guards=(("recon_err", "lt", 0.01),)), "effect")


def _fa_null(ds):
    return Entry(ds, "factor_analysis", ["PAF promax", "NSA rotation"], Primary("ari", "NSA rotation", None, tol=0.1), "null")


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
    Entry("planted", "module_recovery", ["NSA nonneg (multistart)", "NSA nonneg", "KMeans features", "PCA"],
          Primary("ari", FIND_NN, "KMeans features"), "info", dict(k_factors=(1, 2))),
    Entry("noise", "module_recovery", ["NSA nonneg (multistart)", "KMeans features", "PCA"],
          Primary("stability", FIND_NN, None), "null", dict(k_factors=(1,))),
    Entry("noise", "speed", ["PCA", "NSA consolidated", "NSA nonneg"]),
    Entry("fa_sim_n300", "factor_analysis", ["PAF promax", "NSA rotation"],
          Primary("hofmann", "NSA rotation", "PAF promax", direction="lt"), "info"),
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
    _mod("newsgroups", n_outer=10),                 # multistart NSA is expensive; 10 draws suffice for these effect sizes
    _mod("metabric", n_outer=10),                 # multistart NSA is expensive; 10 draws suffice for these effect sizes
    _mod("pbmc3k", n_outer=10),                 # multistart NSA is expensive; 10 draws suffice for these effect sizes
    Entry("tecator", "module_recovery", MOD, Primary("contiguity", FIND_NN, "KMeans features"), "info"),
    # null controls: structure must not be invented; selection must not add optimism
    _pred("adni@null_y", "r2", "forest", "null", ["PCA", "NSA consolidated", "KMeans features"]),
    Entry("noise", "module_recovery", ["NSA nonneg (multistart)", "KMeans features", "PCA"],
          Primary("stability", FIND_NN, None), "null", dict(k_factors=(1,))),
    Entry("metabric@shuffled", "module_recovery", ["NSA nonneg (multistart)", "KMeans features", "PCA"],
          Primary("stability", FIND_NN, None), "null", dict(primary_only=True, n_outer=10, k_factors=(1,))),
    Entry("speed", "speed", ["PCA", "NSA consolidated", "NSA nonneg"]),
    _fa("bfi"), _fa("holzinger"), _fa("fa_sim_n100"), _fa("fa_sim_n300"), _fa("fa_sim_n1000"),
    _fa_null("fa_null"), _fa_null("bfi@shuffled"),
    _nhanes("nhanes"), _nhanes("nhanes@density"),
    Entry("nhanes@density", "survival", NHANES_METHODS, Primary("cindex", "NSA nonneg", "PCA", "forest"), "effect",
          dict(max_n=20000, n_outer=10)),
]))
SUITES["paper"] = SUITES["standard"]
SUITES["standard-full"] = SUITES["standard"]

# standard-fast: the daily-driver subset, target <= 10 minutes with 5 workers.  Same pre-registered primaries as
# `standard`; cheaper settings: three prediction methods, primary module variant only, 10 draws on the large
# expression sets, NHANES at 6,000 rows.  Left to standard-full: golub, tecator, newsgroups, metabric@shuffled, speed,
# NHANES at 20,000 rows and the residual variant.
_FAST_MODULE_METHODS = ["NSA nonneg (multistart)", "NSA nonneg", "KMeans features", "NMF"]
_FAST_MODULE = dict(primary_only=True, n_outer=6, k_factors=(1, 2, 4))
_FAST_PRED_METHODS = ["PCA", "NSA consolidated", "KMeans features"]
_FAST_NHANES = dict(max_n=6000, n_outer=10, n_inner=3)
SUITES.add("standard-fast", Suite("standard-fast", "daily-driver subset of standard (~10 min with 5 workers)", [
    _pred("adni", "r2", "forest", methods=_FAST_PRED_METHODS),
    _mod("planted"),
    _mod("metabric", methods=_FAST_MODULE_METHODS, **_FAST_MODULE),
    _mod("pbmc3k", methods=_FAST_MODULE_METHODS, **_FAST_MODULE),
    _pred("adni@null_y", "r2", "forest", "null", _FAST_PRED_METHODS),
    Entry("noise", "module_recovery", ["NSA nonneg (multistart)", "KMeans features", "PCA"],
          Primary("stability", FIND_NN, None), "null", dict(k_factors=(1,))),
    _fa("bfi"), _fa("holzinger"), _fa("fa_sim_n100"), _fa("fa_sim_n300"), _fa("fa_sim_n1000"),
    _fa_null("fa_null"), _fa_null("bfi@shuffled"),
    Entry("nhanes", "nested_cv", NHANES_METHODS, Primary("auc", "NSA nonneg", "PCA", "forest"), "effect",
          dict(_FAST_NHANES)),
    Entry("nhanes@density", "nested_cv", NHANES_METHODS, Primary("auc", "NSA nonneg", "PCA", "forest"), "effect",
          dict(_FAST_NHANES)),
    Entry("nhanes@density", "survival", NHANES_METHODS, Primary("cindex", "NSA nonneg", "PCA", "forest"), "effect",
          dict(max_n=6000, n_outer=10)),
]))

_AB_NSA = ["NSA consolidated", "NSA consolidated (legacy init)", "NSA nonneg", "NSA nonneg (legacy init)"]
_AB_MOD = ["KMeans features"] + _AB_NSA


def _ab_pred(ds, metric, nsa, legacy, methods, **params):
    """Paired A/B of the library's default initialiser against the legacy one on identical splits (info only)."""
    return Entry(ds, "nested_cv", methods, Primary(metric, legacy, nsa, "forest"), "info", params)


def _ab_mod(ds):
    return Entry(ds, "module_recovery", _AB_MOD, Primary("ari", "NSA nonneg (legacy init) w=0.5", "NSA nonneg w=0.5"),
                 "info", dict(primary_only=True, n_outer=10))


SUITES.add("init_ab", Suite("init_ab", "A/B of the default initialiser vs the legacy one (info only; minutes)", [
    _ab_pred("adni", "r2", "NSA consolidated", "NSA consolidated (legacy init)",
             ["PCA", "NSA consolidated", "NSA consolidated (legacy init)"]),
    _ab_pred("nhanes", "auc", "NSA nonneg", "NSA nonneg (legacy init)", ["PCA", "NSA nonneg", "NSA nonneg (legacy init)"],
             **NHANES_CAP),
    _ab_pred("nhanes@density", "auc", "NSA nonneg", "NSA nonneg (legacy init)",
             ["PCA", "NSA nonneg", "NSA nonneg (legacy init)"], **NHANES_CAP),
    _ab_mod("pbmc3k"), _ab_mod("metabric"), _ab_mod("newsgroups"),
]))

SUITES.add("nhanes", Suite("nhanes", "NHANES dietary -> mortality: AUC, energy-adjusted variants, survival C-index", [
    _nhanes("nhanes"), _nhanes("nhanes@density"), _nhanes("nhanes@residual"),
    Entry("nhanes@density", "survival", NHANES_METHODS, Primary("cindex", "NSA nonneg", "PCA", "forest"), "effect",
          dict(max_n=20000, n_outer=10)),
]))
