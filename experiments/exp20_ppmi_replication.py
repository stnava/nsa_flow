r"""E20 -- PPMI, and why the result it produced must not be interpreted.

RETRACTED AS AN EVALUATION.  This ran with age and education as the only
covariates.  PPMI carries confounds that dominate both targets and that this
design leaves unmodelled, so the comparison between bases is uninterpretable in
either direction.  It is kept as a record of the attempt and of what a usable
version would have to adjust for.

For SAA status the objection is fatal rather than a matter of degree.  The
cohort label alone predicts SAA at AUC 0.920 in this extract -- 996 of 1055
PD-sporadic subjects are positive and 262 of 279 controls are negative -- while
the imaging-based AUC reported here was 0.60.  The target is therefore largely a
proxy for cohort membership, and cohort correlates with age, disease duration,
medication, site and acquisition protocol.  What the run measured is how
efficiently each basis recovers cohort through those confounds.  A usable test
would score SAA WITHIN a diagnosis stratum, which removes the collinearity by
construction (see ``ppmi.SAA_TASKS``).

For UPDRS-I, medication dose (``LEDD``) and disease duration (``duration_yrs``)
are first-order: treated patients score lower.  Neither was adjusted.

Available in the clinical table and not used here: ``commonSex`` (2476 values --
an earlier note in this file claiming it was absent was wrong, the column having
been mis-detected by applying ``pd.to_numeric`` to strings), ``imaging_protocol``
(two levels), ``brainVolume``, ``duration_yrs``, ``LEDD``, ``joinedDX``.

The numbers this produced, recorded so the retraction is checkable rather than
asserted: on SAA every basis was worse than PCA under both models (signed
-0.040, consolidated -0.057, subspace -0.011, data -0.040 by AUC under logistic
regression, all p < 0.001); on UPDRS-I under a random forest every basis beat
PCA (consolidated +0.017, p = 0.018) while under a linear model no basis moved
\(\Delta R^2\) off 0.0005, imaging adding nothing linearly to age and education.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from .ppmi import load_ppmi_continuous
from .rmd_support import _basis

warnings.filterwarnings("ignore")
MODES = ["pca", "signed", "signed_consolidated", "subspace", "data"]


def _covars(df):
    age = pd.to_numeric(df.age_BL, errors="coerce").to_numpy(float)
    edu = pd.to_numeric(df.educ, errors="coerce").to_numpy(float)
    return np.column_stack([np.ones(len(df)), age, edu])


def run(modality="T1w", k=5, w=0.5, n_splits=5, n_repeats=4, seed=0):
    X, df, _ = load_ppmi_continuous(modality)
    C = _covars(df)
    rows = []

    # ---- SAA status: binary, scored by AUC -------------------------------
    lab = df.AsynStatus.astype(str)
    m = lab.isin(["Positive", "Negative"]).to_numpy() & np.isfinite(C).all(1)
    Xs, Cs, ys = X[m], C[m], (lab[m] == "Positive").to_numpy(int)
    fold = {(mo, md): [] for mo in ("logistic", "forest") for md in MODES}
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    for tr, te in cv.split(Xs, ys):
        sc = StandardScaler().fit(Xs[tr])
        A, B = sc.transform(Xs[tr]), sc.transform(Xs[te])
        V = {md: _basis(A, k, w, md) for md in MODES}
        for mo, mk in (("logistic", lambda: LogisticRegression(max_iter=3000)),
                       ("forest", lambda: RandomForestClassifier(
                           n_estimators=250, random_state=0, n_jobs=-1))):
            for md in MODES:
                Z = np.column_stack([Cs[tr][:, 1:], A @ V[md]])
                Zt = np.column_stack([Cs[te][:, 1:], B @ V[md]])
                pr = mk().fit(Z, ys[tr]).predict_proba(Zt)[:, 1]
                fold[(mo, md)].append(roc_auc_score(ys[te], pr))
    for mo in ("logistic", "forest"):
        pca = np.array(fold[(mo, "pca")])
        for md in MODES:
            v = np.array(fold[(mo, md)])
            t, pv = (np.nan, np.nan) if md == "pca" else stats.ttest_rel(v, pca)
            rows.append(dict(outcome="SAA status", metric="AUC", model=mo,
                             basis=md, n=int(m.sum()), folds=len(v),
                             score=v.mean(), d_vs_pca=v.mean() - pca.mean(),
                             t_vs_pca=t, p_vs_pca=pv))

    # ---- UPDRS-I: continuous, scored by out-of-sample dR2 ----------------
    y = pd.to_numeric(df.updrs1_score, errors="coerce").to_numpy(float)
    m = np.isfinite(y) & np.isfinite(C).all(1)
    Xr, Cr, yr = X[m], C[m], y[m]
    fold = {(mo, md): [] for mo in ("linear", "forest")
            for md in MODES + ["cov"]}
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for tr, te in cv.split(Xr):
        mu = Xr[tr].mean(0)
        A, B = Xr[tr] - mu, Xr[te] - mu
        sst = float(((yr[te] - yr[tr].mean()) ** 2).sum())
        if sst <= 0:
            continue
        V = {md: _basis(A, k, w, md) for md in MODES}
        for mo, mk in (("linear", lambda: LinearRegression()),
                       ("forest", lambda: RandomForestRegressor(
                           n_estimators=250, random_state=0, n_jobs=-1))):
            def r2(Ztr, Zte):
                pr = mk().fit(Ztr, yr[tr]).predict(Zte)
                return 1.0 - float(((yr[te] - pr) ** 2).sum()) / sst
            fold[(mo, "cov")].append(r2(Cr[tr], Cr[te]))
            for md in MODES:
                fold[(mo, md)].append(
                    r2(np.column_stack([Cr[tr], A @ V[md]]),
                       np.column_stack([Cr[te], B @ V[md]])))
    for mo in ("linear", "forest"):
        base = np.array(fold[(mo, "cov")])
        pca = np.array(fold[(mo, "pca")])
        for md in MODES:
            v = np.array(fold[(mo, md)])
            t, pv = (np.nan, np.nan) if md == "pca" else stats.ttest_rel(v, pca)
            rows.append(dict(outcome="UPDRS-I", metric="dR2", model=mo,
                             basis=md, n=int(m.sum()), folds=len(v),
                             score=v.mean() - base.mean(),
                             d_vs_pca=v.mean() - pca.mean(),
                             t_vs_pca=t, p_vs_pca=pv))
    return pd.DataFrame(rows)
