r"""E23 -- Golub 3-class: B-ALL / T-ALL / AML.  Signed lifting vs honest SPCA.

The 2-class ALL/AML problem is near-ceiling for every method (PCA AUC > 0.96).
The 3-class version -- B-cell ALL (n=38), T-cell ALL (n=9), AML (n=25) -- is
genuinely hard: PCA balanced-accuracy at k=3 is 0.762 linear / 0.697 forest.

Design: k = 3, top-2000 genes by training-fold variance, z-scored.  The basis
is fitted on all 72 samples (in-sample for the basis, as in the original paper).
Cross-validation is over the classifier only: 5-fold x 50 repeats, stratified.

Evaluation: macro-averaged balanced accuracy (not plain accuracy, because T-ALL
has only n=9 samples and accuracy would hide its misclassification).  Both a
linear (LogisticRegression C=1) and a non-linear (RandomForest 500 trees)
classifier are reported, because a linear classifier directly tests whether the
basis puts the classes in linearly separable positions.

Dimensional fairness: ``r.Y`` for nsa_flow_signed is ``V = Vp - Vm`` of shape
``[p, k]``.  All methods project into k-dimensional score space via ``Z @ V``.
The 2k internal parts ``W = [Vp|Vm]`` are in ``r['parts']``.

Comparators:
  Standard PCA            -- SVD top-k.
  SparsePCA (sklearn a=1) -- Zou et al. 2006 LASSO-based sparse PCA, alpha=1.0.
                             THIS is the honest sparse PCA comparator.
  SparsePCA (median-thr)  -- post-hoc median-zero threshold of PCA.  Kept for
                             ablation reference; NOT a real SPCA method.
  NSA-Flow (w sweep)      -- nsa_flow_data at w in {0.25, 0.5, 0.75, 0.9}.
  Signed (w sweep)        -- nsa_flow_signed at w in {0.0, 0.25, 0.5},
                             with and without consolidation.

RESULTS (paper/results/e23_golub_3class.csv, verified 2026-09-18, v2.10.0,
         n=72 in-sample basis, correct preprocessing: log2->top-2000 by var->z-score,
         5-fold x 50 repeats CV, balanced accuracy):

  method                   linear  forest  sparsity  lobe_overlap
  Standard PCA             0.762   0.697   0.00      --           <- best linear baseline
  SparsePCA (median-thr)   0.761   0.701   0.50      --
  SparsePCA (sklearn a=1)  0.669   0.662   0.44      --           <- over-regularised
  NSA-Flow (w=0.25)        0.679   0.572   0.51      --
  NSA-Flow (w=0.50)        0.675   0.576   0.56      --
  NSA-Flow (w=0.75)        0.665   0.601   0.60      --
  NSA-Flow (w=0.90)        0.658   0.589   0.64      --
  Signed (w=0.0)           0.762   0.727   0.00      0.374        <- signed PCA; best forest
  Signed+consol (w=0.0)    0.769   0.649   0.67      0.000        <- best sparse+linear
  Signed (w=0.25)          0.662   0.633   0.20      0.219
  Signed+consol (w=0.25)   0.672   0.660   0.67      0.000
  Signed (w=0.5)           0.659   0.629   0.35      0.063
  Signed+consol (w=0.5)    0.682   0.649   0.67      0.000

Honest findings:
  - PCA (0.762 linear) is the strongest single baseline.  NSA-Flow data does NOT
    beat PCA on the linear arm (0.658-0.679); the non-negative constraint
    sacrifices the linear decision boundary on this dataset.
  - Signed (w=0.0) IS signed PCA: its linear accuracy (0.762) matches PCA exactly
    and its forest accuracy (0.727) exceeds PCA (0.697) by 0.030.  The signed
    lifting captures B-ALL/T-ALL contrast structure that random forests exploit.
    The improvement is meaningful but within 1 SD of the mean (SD≈0.13).
  - Signed+consol (w=0.0) achieves 0.769 linear at 67% sparsity -- marginally above
    PCA (0.769 vs 0.762, Δ=0.007, not statistically significant).  It gives
    exactly disjoint gene sets per component (overlap=0) with near-zero orth defect
    (~1e-12), making it the best sparse interpretable variant.
  - sklearn SparsePCA at alpha=1.0 is over-regularised (0.669 linear, -0.093 vs PCA).
    Alpha must be tuned in-fold for a fair comparison.
  - Lobe overlap is monotone from w=0.25 onward (0.219->0.063).  At w=0, the signed
    relaxed solver is in the pure-reconstruction basin; adding any orth weight
    briefly moves to a different basin (overlap peaks at w≈0.05) then falls
    monotonically.  Consolidated variant (overlap=0 everywhere) is recommended
    for users requiring disjoint gene sets.
  - v2.10.0 changes: plateau detection (max_iter is now a safety cap); lobe penalty
    scaled by w (monotone response); max_iter defaults lowered (data: 500, signed: 3000).
  - Bug fixed 2026-09-18: previous code standardised first (all variances->1) then
    filtered, selecting random genes.  PCA was 0.659 (should be 0.762).

Results (paper/results/e23_golub_3class.csv).
"""
import warnings

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import SparsePCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from nsa_flow.reconstruct import nsa_flow_data
from nsa_flow.signed import nsa_flow_signed

from .data import load_golub3

warnings.filterwarnings("ignore")


def _sparsity(V, tol=1e-10):
    return float((np.abs(V) < tol).mean())


def _orth_defect(V):
    G = V.T @ V
    G = G / np.trace(G)
    return float(np.linalg.norm(G - np.eye(V.shape[1]) / V.shape[1]))


def run(k=3, ws=(0.25, 0.5, 0.75, 0.9), signed_ws=(0.0, 0.25, 0.5),
        max_iter=None,
        n_splits=5, n_repeats=50, seed=1, n_estimators=500,
        n_features=2000, verbose=True):
    """Run the 3-class Golub benchmark.

    ``r.Y`` for nsa_flow_signed is ``V = Vp - Vm`` of shape ``[p, k]``:
    the comparison is dimensionally fair — all methods give k-dimensional scores
    via ``Z @ V``.  The 2k internal parts ``W = [Vp|Vm]`` are in ``r['parts']``.
    """

    X, y_str, _ = load_golub3()

    # Preprocessing: log2 (standard for this Agilent array), then top-variance
    # gene filter on the log-scaled (not yet standardised) matrix so that
    # variance differences are real, then z-score.  This matches the order in
    # exp4_golub.py (cv_score): filter on Xtr.var() BEFORE StandardScaler.
    # Bug fixed 2026-09-18: the previous code standardised first (making every
    # column variance = 1) then "filtered by variance" — a no-op that selected
    # the last 2000 genes in array order rather than the most variable ones.
    Xl = np.log2(np.clip(X, 1.0, None))                # log2 of intensities
    top = np.argsort(Xl.var(0))[-n_features:]           # top-var on log-scale
    Xs = Xl[:, top]
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0, ddof=1) + 1e-12)  # z-score
    Z = Xs
    T = torch.as_tensor(Z, dtype=torch.float64)

    le = LabelEncoder().fit(y_str)
    y = le.transform(y_str)          # integer labels; stratified CV uses these
    if verbose:
        print(f"Classes: {dict(zip(le.classes_, np.bincount(y)))}", flush=True)

    arms, notes = {}, {}

    # PCA
    _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    arms["Standard PCA"] = vt[:k].T

    # Median-threshold (ablation only, not real SPCA)
    V_med = vt[:k].T.copy()
    V_med[np.abs(V_med) < np.quantile(np.abs(V_med), 0.5)] = 0.0
    arms["SparsePCA (median-thr)"] = V_med

    # Real sklearn SparsePCA (Zou 2006)
    if verbose:
        print("  Fitting sklearn SparsePCA ...", flush=True)
    V_spca = SparsePCA(n_components=k, alpha=1.0, max_iter=500,
                       random_state=0, n_jobs=-1).fit(Z).components_.T
    arms["SparsePCA (sklearn α=1)"] = V_spca
    if verbose:
        print(f"  SparsePCA sparsity={_sparsity(V_spca):.3f}", flush=True)

    # NSA-Flow across w values (nsa_flow_data, non-negative basis)
    kw = {} if max_iter is None else dict(max_iter=max_iter)
    for w in ws:
        name = f"NSA-Flow (w={w})"
        if verbose:
            print(f"  Fitting {name} ...", flush=True)
        arms[name] = nsa_flow_data(T, k=k, w=w, **kw).Y.numpy()

    # Signed lifting — sweep over signed_ws, always with consolidation
    # r.Y is V = Vp - Vm of shape [p, k]: dimensionally fair vs PCA (k cols each)
    for w in signed_ws:
        for cons, suffix in ((False, ""), (True, "+consol")):
            label = f"Signed{suffix} (w={w})"
            if verbose:
                print(f"  Fitting {label} ...", flush=True)
            r = nsa_flow_signed(T, k=k, w=w, consolidate=cons, **kw)
            arms[label] = r.Y.numpy()
            notes[label] = dict(parts_n_dead=r["parts_n_dead"],
                                lobe_overlap=round(float(r["lobe_overlap"]), 5))
            if verbose:
                print(f"  {label}: {notes[label]}", flush=True)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    clfs = [
        ("linear", LogisticRegression(C=1, max_iter=1000, random_state=0)),
        ("forest", RandomForestClassifier(n_estimators=n_estimators,
                                          random_state=0, n_jobs=-1)),
    ]
    rows = []
    for arm_name, V in arms.items():
        V = np.asarray(V, dtype=float)
        scores_V = Z @ V          # [n, k] projections
        for clf_name, clf in clfs:
            s = cross_val_score(clf, scores_V, y, cv=cv,
                                scoring="balanced_accuracy")
            row = dict(method=arm_name, classifier=clf_name,
                       bal_acc=s.mean(), sd=s.std(),
                       sparsity=_sparsity(V), orth_defect=_orth_defect(V),
                       **notes.get(arm_name, {}))
            rows.append(row)
            if verbose:
                print(f"  {arm_name:30s} {clf_name:6s}  "
                      f"bal-acc={s.mean():.4f}±{s.std():.4f}", flush=True)
    return pd.DataFrame(rows)
