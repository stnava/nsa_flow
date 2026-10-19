"""Metrics and summaries shared by every protocol (the single source; ``experiments.common`` re-exports)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ interpretability
def sparsity(L, rel=1e-3):
    """Fraction of loading entries below ``rel`` times the column maximum."""
    L = np.abs(L)
    return float((L <= rel * L.max(axis=0, keepdims=True)).mean())


def support_overlap(L, rel=1e-3):
    """Mean number of components each active feature loads on, minus one (0 = disjoint supports)."""
    L = np.abs(L)
    per_feature = (L > rel * L.max(axis=0, keepdims=True)).sum(axis=1)
    live = per_feature > 0
    return float(per_feature[live].mean() - 1.0) if live.any() else 0.0


# ------------------------------------------------------------------ module recovery
def dominant_assignment(V, rel=1e-3):
    """Component each feature loads on most strongly; -1 where the whole row is ~0."""
    A = np.abs(np.asarray(V))
    lab = A.argmax(1)
    lab[A.max(1) <= rel * A.max()] = -1
    return lab


def module_ari(V, truth):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(truth, dominant_assignment(V)))


def module_nmi(V, truth):
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(truth, dominant_assignment(V)))


def pair_scores(lab, truth):
    """(recall, separation): fraction of same-group pairs placed together / different-group pairs apart."""
    same = diff = rec = sep = 0
    for i, j in combinations(range(len(truth)), 2):
        if truth[i] == truth[j]:
            same += 1
            rec += lab[i] == lab[j]
        else:
            diff += 1
            sep += lab[i] != lab[j]
    return rec / max(same, 1), sep / max(diff, 1)


def contiguity(V, rel=1e-3):
    """Fraction of adjacent (ordered) feature pairs, both assigned, in the same module."""
    lab = dominant_assignment(V, rel)
    ok = (lab[:-1] >= 0) & (lab[1:] >= 0)
    return float((lab[:-1][ok] == lab[1:][ok]).mean()) if ok.any() else float("nan")


def split_half_ari(fit, X, n_rep=10, seed=0):
    """Mean ARI between feature assignments learned on two disjoint halves of the rows; chance ~ 0."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        perm = rng.permutation(len(X))
        a, b = perm[: len(X) // 2], perm[len(X) // 2:]
        out.append(adjusted_rand_score(dominant_assignment(fit(X[a])), dominant_assignment(fit(X[b]))))
    return float(np.mean(out)), float(np.std(out, ddof=1) / np.sqrt(n_rep))


# ------------------------------------------------------------------ paired summaries
def paired_summary(df, metrics, refs, unit="draw"):
    """Mean +/- se per method and paired t differences (95% CI) against each reference method.

    ``df`` is wide-by-method after pivoting on ``unit`` (one row per method per unit).
    """
    from scipy import stats
    out = []
    for metric in metrics:
        if metric not in df:
            continue
        piv = df.pivot(index=unit, columns="method", values=metric)
        n = len(piv)
        for m in piv.columns:
            row = dict(metric=metric, method=m, mean=piv[m].mean(), se=piv[m].std(ddof=1) / np.sqrt(n))
            for ref in refs:
                if ref not in piv:
                    continue
                d = (piv[m] - piv[ref]).dropna()
                tag = ref.split()[0]
                if m == ref or len(d) < 3:
                    lo = hi = np.nan
                else:
                    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                                              scale=d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
                row.update({f"d_vs_{tag}": 0.0 if m == ref else d.mean(),
                            f"ci_lo_{tag}": lo, f"ci_hi_{tag}": hi,
                            f"wins_vs_{tag}": np.nan if m == ref else int((d > 0).sum())})
            out.append(row)
    return pd.DataFrame(out)


def long_summary(rows: pd.DataFrame, refs) -> pd.DataFrame:
    """Summarise the tidy results schema: paired differences against ``refs`` per (dataset, protocol,
    metric, model).  The pairing unit is ``split`` (outer fold / draw)."""
    keys = ["dataset", "protocol", "metric", "model"]
    parts = []
    for key, g in rows.groupby(keys, dropna=False):
        wide = g.rename(columns={"value": "v"})
        s = paired_summary(wide, ["v"], [r for r in refs if r in set(g.method)], unit="split")
        s = s.drop(columns="metric", errors="ignore")
        for k, v in zip(keys, key):
            s[k] = v
        parts.append(s)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out[keys + [c for c in out.columns if c not in keys]]


# ------------------------------------------------------------------ factor-analysis metrics
def hofmann_complexity(L):
    """Per-item Hofmann complexity (sum l^2)^2 / sum l^4: 1 = loads on one factor, k = spread evenly."""
    A = np.abs(np.asarray(L, float))
    return (A ** 2).sum(1) ** 2 / np.maximum((A ** 4).sum(1), 1e-300)


def cross_loadings(L, thr=0.3):
    """Number of items with |loading| > thr on more than one factor."""
    return int(((np.abs(np.asarray(L)) > thr).sum(1) > 1).sum())


def orthogonality_defect(L):
    """Scale-invariant defect of the column Gram: ||G - I/k||_F^2 / (1 - 1/k), G = L'L / tr(L'L)."""
    L = np.asarray(L, float)
    k = L.shape[1]
    if k < 2:
        return 0.0
    G = L.T @ L
    G = G / max(np.trace(G), 1e-300)
    return float(((G - np.eye(k) / k) ** 2).sum() / (1 - 1 / k))


def tucker_congruence(L, target):
    """Mean Tucker congruence between the columns of ``L`` and ``target`` under the best column matching
    (sign-free), plus the matching (cols of L, cols of target)."""
    from scipy.optimize import linear_sum_assignment
    A = np.asarray(L, float)
    B = np.asarray(target, float)
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    C = np.abs(An.T @ Bn)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean()), (r, c)


def reproduced_correlation_error(R, L, uniqueness=None):
    """Relative Frobenius error of the off-diagonal of R against L L' (diagonal ignored: communalities are free)."""
    R = np.asarray(R, float)
    M = np.asarray(L, float) @ np.asarray(L, float).T
    off = ~np.eye(len(R), dtype=bool)
    return float(np.linalg.norm((R - M)[off]) / max(np.linalg.norm(R[off]), 1e-300))


# ------------------------------------------------------------------ survival
def harrell_c(time, event, risk):
    """Harrell's concordance index (exact): among comparable pairs (i has an event before j's time) the fraction
    where i has the higher risk; ties in risk count 1/2."""
    t, e, r = np.asarray(time, float), np.asarray(event).astype(bool), np.asarray(risk, float)
    num = den = 0.0
    for i in np.where(e)[0]:
        later = t > t[i]
        n_later = int(later.sum())
        if n_later:
            den += n_later
            num += float((r[i] > r[later]).sum()) + 0.5 * float((r[i] == r[later]).sum())
    return num / den if den else float("nan")
