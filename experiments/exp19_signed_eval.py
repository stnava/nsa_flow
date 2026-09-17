r"""E19 -- the signed lifting evaluated across every ADNI cognitive outcome.

Paired: per-fold \(R^2\) is retained for each basis on the same folds, so the
comparison against PCA is a paired test rather than a difference of independent
means.  Both a linear model and a random forest are run, because a linear model
on projected scores is exactly invariant to reparametrising the basis and can
therefore only see the span.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold

from .rmd_support import COG_VARS, _basis, _covar_design, load_adni_thickness

warnings.filterwarnings("ignore")
MODES = ["pca", "signed", "signed_consolidated", "subspace", "data"]


def run(k=5, w=0.5, n_splits=5, n_repeats=4, seed=0, models=("linear", "forest")):
    """``w = 0.5`` is the settled default; see ``summarise`` and the module
    docstring of ``nsa_flow.signed`` for why 0.75 is materially worse."""
    X, df, _ = load_adni_thickness()
    C = _covar_design(df)
    mk = {"linear": lambda: LinearRegression(),
          "forest": lambda: RandomForestRegressor(n_estimators=200,
                                                  random_state=0, n_jobs=-1)}
    rows = []
    for cog in [c for c in COG_VARS if c in df.columns]:
        y = pd.to_numeric(df[cog], errors="coerce").to_numpy(float)
        keep = np.isfinite(y) & np.isfinite(C).all(1)
        Xk, Ck, yk = X[keep], C[keep], y[keep]
        fold = {(m, md): [] for m in models for md in MODES + ["cov"]}
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                           random_state=seed)
        for tr, te in cv.split(Xk):
            mu = Xk[tr].mean(0)
            Xtr, Xte = Xk[tr] - mu, Xk[te] - mu
            sst = float(((yk[te] - yk[tr].mean()) ** 2).sum())
            if sst <= 0:
                continue
            V = {md: _basis(Xtr, k, w, md) for md in MODES}
            for m in models:
                def r2(Ztr, Zte):
                    pr = mk[m]().fit(Ztr, yk[tr]).predict(Zte)
                    return 1.0 - float(((yk[te] - pr) ** 2).sum()) / sst
                fold[(m, "cov")].append(r2(Ck[tr], Ck[te]))
                for md in MODES:
                    fold[(m, md)].append(
                        r2(np.column_stack([Ck[tr], Xtr @ V[md]]),
                           np.column_stack([Ck[te], Xte @ V[md]])))
        for m in models:
            base = np.array(fold[(m, "cov")])
            pca = np.array(fold[(m, "pca")])
            for md in MODES:
                v = np.array(fold[(m, md)])
                if md == "pca":
                    t, pv = np.nan, np.nan
                else:
                    t, pv = stats.ttest_rel(v, pca)
                rows.append(dict(cog=cog, model=m, basis=md, n=int(keep.sum()),
                                 folds=len(v), r2=v.mean(),
                                 dR2_vs_cov=v.mean() - base.mean(),
                                 d_vs_pca=v.mean() - pca.mean(),
                                 t_vs_pca=t, p_vs_pca=pv))
        print(f"  {cog} done", flush=True)
    return pd.DataFrame(rows)


def summarise(d):
    """Wins over PCA per basis and model, with a sign test across outcomes."""
    out = []
    for (m, md), g in d[d.basis != "pca"].groupby(["model", "basis"]):
        wins = int((g.d_vs_pca > 0).sum())
        sig = int(((g.p_vs_pca < 0.05) & (g.d_vs_pca > 0)).sum())
        out.append(dict(model=m, basis=md, outcomes=len(g), better_than_pca=wins,
                        significant=sig, mean_d=g.d_vs_pca.mean(),
                        median_d=g.d_vs_pca.median(),
                        sign_test_p=stats.binomtest(wins, len(g)).pvalue))
    return pd.DataFrame(out).sort_values(["model", "mean_d"], ascending=[True, False])
