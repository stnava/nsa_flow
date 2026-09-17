r"""E21 -- second cohort, chosen for being confound-light rather than large.

The PPMI attempt (``exp20``) failed not on sample size but on confounding: its
targets are proxies for cohort membership, and cohort carries age, medication,
duration, site and protocol.  These two UCI sets have the opposite property.
They are single-source, fully tabulated, have no acquisition covariates to
adjust for, and their outcomes are not proxies for anything else in the table.
That makes the comparison between bases interpretable, which is the only thing
that matters for the question being asked.

They also probe a regime the imaging data cannot: ``p = 13`` and ``p = 10``, where
``p/k`` is small and near-disjoint supports are most restrictive.  Earlier sweeps
found the largest losses for the constrained bases at small ``p`` (wine ``p = 13``,
ADNI volumes ``p = 31``), so this is a deliberately unfavourable test.

Sources are the loaders used in the SiMLR paper scripts.

Result (k = 3, w = 0.5, 5-fold x 6 repeats, paired t on folds; see
``paper/results/e21_uci_replication.csv``).  It splits, and it splits on ``p``
and on outcome type rather than on anything about the solver.

Heart (n = 297, p = 13, AUC).  PCA is best and the signed lifting is slightly
but consistently worse: -0.0068 under logistic regression (t = -3.4, p = 0.002)
and -0.0124 under a forest (t = -2.2, p = 0.035); consolidation costs a further
factor of roughly two, -0.0166 and -0.0180.  The data-anchored and subspace
bases tie PCA to within 0.0005 and are not distinguishable from it.  With p = 13
and k = 3 a disjoint partition gives each part about four features, so the
consolidated basis is close to a hard feature partition and the cost of that is
exactly what shows up here.

Diabetes (n = 442, p = 10, R^2).  The signed lifting is best: +0.0134 linear
(t = 3.5, p = 0.001) and +0.0444 forest (t = 4.1, p = 0.0003).  Subspace wins
under the linear model (+0.0182, t = 10.1), the data-anchored basis gains under
both.  Consolidation is neutral here (+0.005 and -0.004, both n.s.).

Taken with the ADNI results this is the shape to expect: the non-negative bases
are competitive-to-better on regression targets, and on small-p classification
they pay a modest, reproducible price for the constraint.  Consolidation is a
sparsity knob, not an accuracy one -- it helped CDRSB, is neutral on diabetes,
and hurts on heart, and the deciding variable is how much room ``p / k`` leaves
for disjoint supports.  Neither set supports a claim that the method dominates
PCA, and neither is confounded in a way that would hide such a claim.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from .rmd_support import _basis

warnings.filterwarnings("ignore")
MODES = ["pca", "signed", "signed_consolidated", "subspace", "data"]
HEART_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
             "heart-disease/processed.cleveland.data")
HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


def load_heart():
    """Cleveland heart disease: 13 features, binary outcome."""
    df = (pd.read_csv(HEART_URL, names=HEART_COLS)
          .replace("?", np.nan).dropna().apply(pd.to_numeric))
    y = (df["num"].to_numpy(int) > 0).astype(int)
    return df.drop(columns="num").to_numpy(float), y


def run(k=3, w=0.5, n_splits=5, n_repeats=6, seed=0):
    rows = []

    # --- heart disease: classification, no covariates to adjust -----------
    try:
        X, y = load_heart()
    except Exception as exc:                          # offline: skip, do not fake
        print(f"  heart unavailable ({type(exc).__name__}); skipped")
        X = None
    if X is not None:
        fold = {(m, md): [] for m in ("logistic", "forest") for md in MODES}
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        for tr, te in cv.split(X, y):
            sc = StandardScaler().fit(X[tr])
            A, B = sc.transform(X[tr]), sc.transform(X[te])
            V = {md: _basis(A, k, w, md) for md in MODES}
            for m, mk in (("logistic", lambda: LogisticRegression(max_iter=3000)),
                          ("forest", lambda: RandomForestClassifier(
                              n_estimators=250, random_state=0, n_jobs=-1))):
                for md in MODES:
                    pr = mk().fit(A @ V[md], y[tr]).predict_proba(B @ V[md])[:, 1]
                    fold[(m, md)].append(roc_auc_score(y[te], pr))
        for m in ("logistic", "forest"):
            pca = np.array(fold[(m, "pca")])
            for md in MODES:
                v = np.array(fold[(m, md)])
                t, p = (np.nan, np.nan) if md == "pca" else stats.ttest_rel(v, pca)
                rows.append(dict(dataset="heart", metric="AUC", model=m, basis=md,
                                 n=len(y), p_feat=X.shape[1], folds=len(v),
                                 score=v.mean(), d_vs_pca=v.mean() - pca.mean(),
                                 t_vs_pca=t, p_vs_pca=p))

    # --- diabetes: regression on the sklearn copy -------------------------
    d = load_diabetes()
    Xr, yr = d.data, d.target
    fold = {(m, md): [] for m in ("linear", "forest") for md in MODES}
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for tr, te in cv.split(Xr):
        sc = StandardScaler().fit(Xr[tr])
        A, B = sc.transform(Xr[tr]), sc.transform(Xr[te])
        sst = float(((yr[te] - yr[tr].mean()) ** 2).sum())
        V = {md: _basis(A, k, w, md) for md in MODES}
        for m, mk in (("linear", lambda: LinearRegression()),
                      ("forest", lambda: RandomForestRegressor(
                          n_estimators=250, random_state=0, n_jobs=-1))):
            for md in MODES:
                pr = mk().fit(A @ V[md], yr[tr]).predict(B @ V[md])
                fold[(m, md)].append(1.0 - float(((yr[te] - pr) ** 2).sum()) / sst)
    for m in ("linear", "forest"):
        pca = np.array(fold[(m, "pca")])
        for md in MODES:
            v = np.array(fold[(m, md)])
            t, p = (np.nan, np.nan) if md == "pca" else stats.ttest_rel(v, pca)
            rows.append(dict(dataset="diabetes", metric="R2", model=m, basis=md,
                             n=len(yr), p_feat=Xr.shape[1], folds=len(v),
                             score=v.mean(), d_vs_pca=v.mean() - pca.mean(),
                             t_vs_pca=t, p_vs_pca=p))
    return pd.DataFrame(rows)
