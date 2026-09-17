r"""E22 -- PPMI, done properly: stratified targets and modelled confounds.

``exp20`` had to be retracted because it used age and education as its only
covariates and scored SAA over the whole cohort, where the label is close to a
restatement of diagnosis.  This is the same cohort with both defects repaired.

Stratify, twice.  Scoring SAA within the prodromal group is necessary but not
sufficient: inside that group genetic subtype almost determines the label --
sporadic prodromals are 63% positive (190/301), GBA carriers 7.2% (12/166),
LRRK2 carriers 7.0% (12/171) -- so stopping at diagnosis would repeat exp20's
error one level down.  The primary target is therefore SAA within SPORADIC
prodromals (111 negative / 190 positive, n = 299 after intersecting imaging),
where no diagnostic or genetic variable stands in for the label.  The mixed
strata are also run, with genotype as a covariate, as a secondary.

Adjust.  Every model carries age, sex, education, imaging protocol and brain
volume; mixed-genotype strata add genotype; PD strata add levodopa-equivalent
dose and disease duration, the two confounds that act directly on symptom
scores.  Complete cases are taken on the union of imaging, outcome and
confounds, so every comparison is paired on the same rows.

Two designs are run, and the pair is the point:

``joint``      -- fit on ``[C | X V]``.  Asks what the basis adds to a model
                  that already has the confounds.  A basis can still win here
                  by encoding a confound more efficiently than the raw column
                  does, e.g. by capturing head size better than ``brainVolume``.
``residual``   -- regress ``X`` on ``C`` within the training fold, fit the basis
                  on the residual, then fit on ``[C | X_res V]``, applying the
                  training-fold regression to the test fold.  Nothing the basis
                  sees is linearly predictable from the confounds, so a gain
                  here is imaging structure and not confound re-encoding.

The confound-only model is fit on the same folds and reported, so "imaging adds
this much" is a number rather than an assumption.  Everything is compared
against PCA on identical folds with a paired t-test.
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

from . import ppmi
from .rmd_support import _basis

warnings.filterwarnings("ignore")
MODES = ["pca", "signed", "signed_consolidated", "subspace", "data"]
DESIGNS = ["joint", "residual"]

TASKS = [
    # (label, stratum, outcome, confounds, binary?)  -- primaries first
    ("SAA in sporadic prodromal", "ProdromalSporadic", "AsynStatus", None, True),
    ("UPDRS-I in sporadic PD", "PDSporadic", "updrs1_score",
     ppmi.CONFOUNDS_PD, False),
    ("SAA in prodromal, geno-adj", "Prodromal", "AsynStatus",
     ppmi.CONFOUNDS_GEN, True),
    ("UPDRS-I in PD, geno-adj", "PD", "updrs1_score",
     ppmi.CONFOUNDS_PD_GEN, False),
    ("UPDRS-I in prodromal, geno-adj", "Prodromal", "updrs1_score",
     ppmi.CONFOUNDS_GEN, False),
]


def _fit_score(mk, Atr, ytr, Ate, yte, binary, sst=None):
    m = mk().fit(Atr, ytr)
    if binary:
        return roc_auc_score(yte, m.predict_proba(Ate)[:, 1])
    return 1.0 - float(((yte - m.predict(Ate)) ** 2).sum()) / sst


def run(modality="T1w", k=5, w=0.5, n_splits=5, n_repeats=4, seed=0, max_p=None,
        verbose=True):
    rows = []
    for label, stratum, outcome, conf, binary in TASKS:
        X, y, C, cols, cnames = ppmi.load_ppmi_modeled(
            modality, stratum, outcome, conf, max_p=max_p)
        models = (("logistic", lambda: LogisticRegression(max_iter=4000)),
                  ("forest", lambda: RandomForestClassifier(
                      n_estimators=300, random_state=0, n_jobs=-1))) if binary else \
                 (("linear", lambda: LinearRegression()),
                  ("forest", lambda: RandomForestRegressor(
                      n_estimators=300, random_state=0, n_jobs=-1)))
        cv = (RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                      random_state=seed) if binary else
              RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                            random_state=seed))
        split = cv.split(X, y) if binary else cv.split(X)

        acc = {(m, d, md): [] for m, _ in models for d in DESIGNS for md in MODES}
        cov = {m: [] for m, _ in models}
        for tr, te in split:
            sx, sc = StandardScaler().fit(X[tr]), StandardScaler().fit(C[tr])
            A, B = sx.transform(X[tr]), sx.transform(X[te])
            Ctr, Cte = sc.transform(C[tr]), sc.transform(C[te])
            sst = None if binary else float(((y[te] - y[tr].mean()) ** 2).sum())

            # residual design: confound regression estimated on the train fold only
            beta = np.linalg.lstsq(np.c_[np.ones(len(tr)), Ctr], A, rcond=None)[0]
            Ares = A - np.c_[np.ones(len(tr)), Ctr] @ beta
            Bres = B - np.c_[np.ones(len(te)), Cte] @ beta
            feats = {"joint": (A, B), "residual": (Ares, Bres)}

            V = {(d, md): _basis(feats[d][0], k, w, md)
                 for d in DESIGNS for md in MODES}
            for m, mk in models:
                cov[m].append(_fit_score(mk, Ctr, y[tr], Cte, y[te], binary, sst))
                for d in DESIGNS:
                    P, Q = feats[d]
                    for md in MODES:
                        acc[(m, d, md)].append(_fit_score(
                            mk, np.c_[Ctr, P @ V[(d, md)]], y[tr],
                            np.c_[Cte, Q @ V[(d, md)]], y[te], binary, sst))

        for m, _ in models:
            cv_only = np.array(cov[m])
            for d in DESIGNS:
                pca = np.array(acc[(m, d, "pca")])
                for md in MODES:
                    v = np.array(acc[(m, d, md)])
                    tc, pc = stats.ttest_rel(v, cv_only)
                    tp, pp = ((np.nan, np.nan) if md == "pca"
                              else stats.ttest_rel(v, pca))
                    rows.append(dict(
                        task=label, stratum=stratum, outcome=outcome,
                        metric="AUC" if binary else "R2", model=m, design=d,
                        basis=md, n=len(y), p_feat=X.shape[1],
                        n_conf=C.shape[1], folds=len(v),
                        score=v.mean(), conf_only=cv_only.mean(),
                        d_vs_conf=v.mean() - cv_only.mean(),
                        t_vs_conf=tc, p_vs_conf=pc,
                        d_vs_pca=v.mean() - pca.mean(),
                        t_vs_pca=tp, p_vs_pca=pp))
        if verbose:
            print(f"  {label}: n={len(y)} p={X.shape[1]} conf={C.shape[1]} "
                  f"({', '.join(cnames)})", flush=True)
    return pd.DataFrame(rows)
