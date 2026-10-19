"""Dataset container, spec and registry.

A ``Dataset`` is everything a protocol needs: the feature matrix, an optional outcome and covariates,
optional feature-group truth (for module recovery), and the preprocessing/splitting conventions that go
with it.  Derived null controls (``name@null_y``, ``name@shuffled``) are built from any dataset.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

from ..registry import DATASETS


class DatasetUnavailable(RuntimeError):
    """The data is not present (or not allowed to be fetched); the message says where it looked."""


@dataclass
class Dataset:
    name: str
    X: np.ndarray                                   # n x p, rows are samples
    y: Optional[np.ndarray] = None                  # outcome (regression: float, classification: int)
    covariates: Optional[np.ndarray] = None         # n x q, appended to the scores for prediction
    groups: Optional[np.ndarray] = None             # length-p truth: group id per feature, -1 = background
    ordered: bool = False                           # features have a natural order (wavelengths)
    task: Optional[str] = None                      # "regression" | "classification" | None
    outer: str = "holdout"                          # "holdout" | "kfold" | "stratified_kfold"
    standardize: bool = True                        # standardise features on the training portion
    top_variance: Optional[int] = None              # keep the top-variance features (on training only)
    draw_n: Optional[int] = None                    # default sample size for module-recovery draws
    kmeans_spherical: bool = False                  # L2-normalise feature vectors before k-means
    weights: Optional[np.ndarray] = None            # survey weights (survival protocol fits with them)
    time: Optional[np.ndarray] = None               # follow-up time (survival protocol)
    event: Optional[np.ndarray] = None              # event indicator (survival protocol)
    metric: Optional[str] = None                    # override the default metric: "auc" for binary outcomes
    standardize_scores: bool = False                # z-score basis scores and covariates before the model
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def p(self):
        return self.X.shape[1]

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(str(self.X.shape).encode())
        h.update(np.ascontiguousarray(self.X[:: max(1, self.n // 50)]).tobytes()[:1 << 20])
        if self.y is not None:
            h.update(np.ascontiguousarray(self.y).tobytes())
        return h.hexdigest()[:16]

    # ---------------------------------------------------------------- preprocessing
    def prep(self, tr, te):
        """Representations of the train/test rows fitted on ``tr`` only: ``c`` (centred, optionally
        standardised, optionally top-variance) and ``pos`` (shifted by the training minimum, clipped >= 0)."""
        Xtr, Xte = self.X[tr], self.X[te]
        if self.top_variance and self.top_variance < self.p:
            keep = np.argsort(Xtr.var(0))[-self.top_variance:]
            Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        mu = Xtr.mean(0)
        sd = Xtr.std(0) + 1e-12 if self.standardize else 1.0
        mn = Xtr.min(0)
        return ({"c": (Xtr - mu) / sd, "pos": np.clip(Xtr - mn, 0, None)},
                {"c": (Xte - mu) / sd, "pos": np.clip(Xte - mn, 0, None)})

    # ---------------------------------------------------------------- derived null controls
    def derive(self, kind: str, seed: int = 0) -> "Dataset":
        rng = np.random.default_rng(seed)
        if kind == "null_y":
            if self.y is None:
                raise ValueError(f"{self.name} has no outcome to permute")
            return replace(self, name=f"{self.name}@null_y", y=rng.permutation(self.y),
                           meta={**self.meta, "null": "outcome permuted across samples"})
        if kind == "shuffled":
            Xs = np.column_stack([rng.permutation(self.X[:, j]) for j in range(self.p)])
            return replace(self, name=f"{self.name}@shuffled", X=Xs, y=None if self.y is None else self.y,
                           meta={**self.meta, "null": "each feature permuted independently across samples"})
        if kind in ("density", "residual"):
            return self._energy_adjusted(kind)
        raise ValueError(f"unknown derivation {kind!r}")

    def _energy_adjusted(self, kind):
        """Energy adjustment for compositional intake data (needs ``meta['energy_index']``).

        ``density``: divide every nutrient by energy per 1000 kcal.  ``residual``: log-log OLS residual of each
        nutrient on energy.  The energy column is dropped.  Both are fitted on all rows using the features only
        (two parameters per nutrient, no outcome), a deliberate simplification stated here.
        """
        j = self.meta.get("energy_index")
        if j is None:
            raise ValueError(f"{self.name} has no energy column (meta['energy_index'])")
        keep = [c for c in range(self.p) if c != j]
        e = np.clip(self.X[:, j], 1.0, None)
        A = self.X[:, keep]
        if kind == "density":
            X2 = A / e[:, None] * 1000.0
        else:
            D = np.c_[np.ones(len(e)), np.log(e)]
            beta = np.linalg.lstsq(D, np.log1p(A), rcond=None)[0]
            X2 = np.log1p(A) - D @ beta
        feats = self.meta.get("features")
        meta = {**self.meta, "adjustment": kind, "energy_index": None,
                "features": None if feats is None else [f for c, f in enumerate(feats) if c != j]}
        return replace(self, name=f"{self.name}@{kind}", X=X2, meta=meta)

    def subsample(self, n, seed=0):
        """Stratified (by outcome when classification) subsample of ``n`` rows, carrying every per-row array."""
        if n >= self.n:
            return self
        rng = np.random.default_rng(seed)
        if self.task == "classification" and self.y is not None:
            idx = np.sort(np.concatenate([
                rng.choice(np.where(self.y == c)[0], max(1, int(round(n * (self.y == c).mean()))), replace=False)
                for c in np.unique(self.y)]))
        else:
            idx = np.sort(rng.choice(self.n, n, replace=False))
        pick = lambda a: None if a is None else a[idx]
        return replace(self, X=self.X[idx], y=pick(self.y), covariates=pick(self.covariates), weights=pick(self.weights),
                       time=pick(self.time), event=pick(self.event), meta={**self.meta, "n_full": self.n})


@dataclass
class DatasetSpec:
    name: str
    load: Callable[..., Dataset]
    private: bool = False
    tier: str = "public"                            # "public" | "restricted" (data-use agreement) | "local" (cache only)
    source: str = ""
    description: str = ""
    fetch: Optional[Callable[[], None]] = None      # optional: download into the data home (explicit only)


def dataset(name, *, private=False, tier=None, source="", description="", fetch=None):
    tier = tier or ("local" if private else "public")

    def deco(fn):
        DATASETS.add(name, DatasetSpec(name, fn, private, tier, source, description, fetch))
        return fn
    return deco


def get_dataset(name: str, **kw) -> Dataset:
    """Resolve ``name`` or ``name@null_y`` / ``name@shuffled``."""
    base, _, kind = name.partition("@")
    ds = DATASETS.get_or_raise(base).load(**kw)
    return ds.derive(kind) if kind else ds


def availability(name: str) -> tuple[bool, str]:
    """(available, reason) without keeping the data."""
    try:
        get_dataset(name)
        return True, "ok"
    except DatasetUnavailable as e:
        return False, str(e)
    except Exception as e:  # a loader bug must not hide as 'unavailable'
        return False, f"loader error: {type(e).__name__}: {e}"


def tier_of(name: str) -> str:
    """Tier of ``name`` (derived ``name@null`` datasets inherit it)."""
    base = name.partition("@")[0]
    return DATASETS[base].tier if base in DATASETS else "public"
