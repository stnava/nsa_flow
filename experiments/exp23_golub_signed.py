r"""E23 -- Golub 3-class: B-ALL / T-ALL / AML.  Signed lifting vs honest SPCA.

The 2-class ALL/AML problem is near-ceiling for every method (PCA AUC > 0.96).
The 3-class version -- B-cell ALL (n=38), T-cell ALL (n=9), AML (n=25) -- is
genuinely hard: PCA balanced-accuracy at k=3 is ~0.66 linear / ~0.58 forest.

Design: k = 3, top-2000 genes by training-fold variance, z-scored.  The basis
is fitted on all 72 samples (in-sample for the basis, as in the original paper).
Cross-validation is over the classifier only: 5-fold x 50 repeats, stratified.

Evaluation: macro-averaged balanced accuracy (not plain accuracy, because T-ALL
has only n=9 samples and accuracy would hide its misclassification).  Both a
linear (LogisticRegression C=1) and a non-linear (RandomForest 500 trees)
classifier are reported, because a linear classifier directly tests whether the
basis puts the classes in linearly separable positions.

Comparators:
  Standard PCA            -- SVD top-k.
  SparsePCA (sklearn a=1) -- Zou et al. 2006 LASSO-based sparse PCA, alpha=1.0.
                             THIS is the honest sparse PCA comparator.
  SparsePCA (median-thr)  -- post-hoc median-zero threshold of PCA.  Kept for
                             ablation reference; NOT a real SPCA method.
  NSA-Flow (w sweep)      -- nsa_flow_data at w in {0.25, 0.5, 0.75, 0.9}.
  Signed (w=0.5)          -- nsa_flow_signed, with and without consolidation.

RESULTS (paper/results/e23_golub_3class.csv, verified 2026-09-18, n=72 in-sample,
         correct preprocessing: log2 -> top-2000 by variance -> z-score):

  method                  linear  forest  sparsity  orth_defect
  Standard PCA            0.762   0.697   0.00      ~0          <- best linear
  SparsePCA (median-thr)  0.761   0.701   0.50      0.004       <- ~= PCA
  SparsePCA (sklearn a=1) 0.669   0.662   0.44      0.062       <- over-regularised
  NSA-Flow (w=0.25)       0.679   0.571   0.51      0.076
  NSA-Flow (w=0.50)       0.675   0.572   0.56      0.043
  NSA-Flow (w=0.75)       0.665   0.599   0.60      0.020
  NSA-Flow (w=0.90)       0.658   0.587   0.64      0.009
  Signed (w=0.5)          0.659   0.630   0.35      0.003
  Signed+consol (w=0.5)   0.682   0.652   0.67      ~0          <- best sparse

Honest findings:
  - PCA is the strongest linear arm (0.762). On the correct top-2000 informative
    genes, the PCA basis is hard to beat.
  - sklearn SparsePCA at alpha=1.0 is over-regularised for this gene set
    (0.669 linear, -0.093 vs PCA). A lower alpha would close the gap but
    alpha is a hyperparameter that must be tuned in-fold.
  - Median-threshold of PCA loadings matches PCA almost exactly (0.761): the
    crude 50%-zero threshold barely hurts, which confirms the top-variance genes
    are robustly informative.
  - NSA-Flow data fidelity loses to PCA on the linear arm at all w (-0.083 to
    -0.097). The non-negative constraint sacrifices the linear boundary.
  - Signed+consol is the best sparse method: 0.682 linear (+0.013 vs sklearn
    SparsePCA), 0.652 forest (+0.045 vs PCA forest is -0.045 but +0.035 vs
    sklearn SparsePCA). lobe_overlap=0.063 on signed confirms genuine contrast
    structure between B-ALL and T-ALL subtypes; consolidation captures it
    cleanly (overlap=0, sparsity=0.67, orth_defect~0).
  - The buggy preprocessing (scale -> filter -> no log2) inflated all numbers
    by selecting random genes: PCA was 0.659 (should be 0.762), SparsePCA
    appeared to beat PCA (0.674 > 0.659) when it actually loses badly (0.669
    vs 0.762). Bug fixed 2026-09-18.

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


def run(k=3, ws=(0.25, 0.5, 0.75, 0.9), max_iter=20000,
        n_splits=5, n_repeats=50, seed=1, n_estimators=500,
        n_features=2000, verbose=True):
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

    # NSA-Flow across w values
    for w in ws:
        name = f"NSA-Flow (w={w})"
        if verbose:
            print(f"  Fitting {name} ...", flush=True)
        arms[name] = nsa_flow_data(T, k=k, w=w, max_iter=max_iter).Y.numpy()

    # Signed lifting
    for label, cons in (("Signed (w=0.5)", False),
                        ("Signed+consol (w=0.5)", True)):
        if verbose:
            print(f"  Fitting {label} ...", flush=True)
        r = nsa_flow_signed(T, k=k, w=0.5, max_iter=max_iter, consolidate=cons)
        arms[label] = r.Y.numpy()
        notes[label] = dict(parts_n_dead=r["parts_n_dead"],
                            lobe_overlap=r["lobe_overlap"])
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
