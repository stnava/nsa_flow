"""Upper bound on PPMI T1w signal: use ALL p features, no reduction.

Any k=5 basis can only extract a subspace of what the full feature set carries,
so this is the ceiling for every method in exp22.  If the ceiling is at the
confound-only level, a basis comparison in this cohort cannot be informative
regardless of which basis wins.
"""
import warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from experiments import ppmi
from experiments.exp22_ppmi_modeled import TASKS
warnings.filterwarnings("ignore")

rows = []
for label, stratum, outcome, conf, binary in TASKS:
    X, y, C, cols, _ = ppmi.load_ppmi_modeled("T1w", stratum, outcome, conf)
    cv = (RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0) if binary
          else RepeatedKFold(n_splits=5, n_repeats=4, random_state=0))
    models = (("linear", lambda: LogisticRegression(max_iter=5000)),
              ("forest", lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))) if binary else \
             (("linear", lambda: RidgeCV(alphas=np.logspace(-2, 4, 25))),
              ("forest", lambda: RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)))
    acc = {(m, v): [] for m, _ in models for v in ("conf", "full")}
    for tr, te in (cv.split(X, y) if binary else cv.split(X)):
        sx, sc = StandardScaler().fit(X[tr]), StandardScaler().fit(C[tr])
        A, B, Ctr, Cte = sx.transform(X[tr]), sx.transform(X[te]), sc.transform(C[tr]), sc.transform(C[te])
        sst = None if binary else float(((y[te] - y[tr].mean()) ** 2).sum())
        for m, mk in models:
            for v, (P, Q) in (("conf", (Ctr, Cte)),
                              ("full", (np.c_[Ctr, A], np.c_[Cte, B]))):
                f = mk().fit(P, y[tr])
                s = (roc_auc_score(y[te], f.predict_proba(Q)[:, 1]) if binary
                     else 1.0 - float(((y[te] - f.predict(Q)) ** 2).sum()) / sst)
                acc[(m, v)].append(s)
    for m, _ in models:
        c, f = np.array(acc[(m, "conf")]), np.array(acc[(m, "full")])
        t, pv = stats.ttest_rel(f, c)
        rows.append(dict(task=label, metric="AUC" if binary else "R2", model=m,
                         n=len(y), p=X.shape[1], conf_only=c.mean(),
                         full_p=f.mean(), ceiling_gain=f.mean() - c.mean(),
                         t=t, p_val=pv))
    print(f"  {label} done", flush=True)
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(d.round(4).to_string(index=False))
d.to_csv("paper/results/e22_ceiling.csv", index=False)
print("SAVED")
