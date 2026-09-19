"""ADNI cortical thickness -> CDRSB, over REPEATED held-out splits.

Why this exists.  The paper's primary imaging result was a single 80/20 split
(``rapid_primary_benchmarks.benchmark_adni_cdrsb``, ``random_state=42``) that
reported consolidated-signed NSA-Flow at forest R^2 = 0.463 against PCA at
0.281.  Re-run with a solver that actually converges (defect_D 0.006 -> 0.0001)
the same split gives 0.309: the +0.18 was the unconverged solution, not the
method.  At n ~ 300 a single split has R^2 noise of the same order as any
effect being claimed, so neither the old number nor the new one is evidence.
This script estimates the split-to-split distribution instead.

Design: ``n_splits`` random 80/20 splits; every step (centering, shift for the
non-negative mode, basis, regressor) fitted on the training portion only;
paired differences against PCA on the same splits, with a 95% interval from the
paired t distribution.  Same k, w and models as the single-split version, so the
only change is the number of splits.
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from nsa_flow import nsa_flow
from experiments.rmd_support import _covar_design, load_adni_thickness

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent.parent / "paper" / "results"

METHODS = [
    ("Covariates only", None, {}),
    ("PCA", "pca", {}),
    ("NSA data w=0.5", "data", {"w": 0.5}),
    ("NSA signed w=0.5", "signed", {"w": 0.5}),
    ("NSA consolidated w=0.5", "signed", {"w": 0.5, "consolidate": True}),
    ("NSA anchored w=0.5", "anchored", {"w": 0.5}),
]


def one_split(X, C, y, k, seed):
    Xtr, Xte, Ctr, Cte, ytr, yte = train_test_split(X, C, y, test_size=0.2,
                                                    random_state=seed)
    mu = Xtr.mean(0)
    Xtr_c, Xte_c = Xtr - mu, Xte - mu
    mn = Xtr.min(0)
    Xtr_p, Xte_p = np.clip(Xtr - mn, 0, None), np.clip(Xte - mn, 0, None)
    out = {}
    for name, kind, kw in METHODS:
        t0 = time.time()
        if kind is None:
            Ztr = Zte = None
        elif kind == "pca":
            m = PCA(n_components=k, random_state=seed).fit(Xtr_c)
            Ztr, Zte = m.transform(Xtr_c), m.transform(Xte_c)
        elif kind == "data":
            V = nsa_flow(Xtr_p, k=k, mode="data", **kw).Y.numpy()
            Ztr, Zte = Xtr_p @ V, Xte_p @ V
        elif kind == "signed":
            V = nsa_flow(Xtr_c, k=k, mode="signed", **kw).Y.numpy()
            Ztr, Zte = Xtr_c @ V, Xte_c @ V
        else:
            L = PCA(n_components=k, random_state=seed).fit(Xtr_c).components_.T
            V = nsa_flow(L, **kw).Y.numpy()
            Ztr, Zte = Xtr_c @ V, Xte_c @ V
        Ftr = Ctr if Ztr is None else np.column_stack([Ctr, Ztr])
        Fte = Cte if Zte is None else np.column_stack([Cte, Zte])
        lr = LinearRegression().fit(Ftr, ytr)
        rf = RandomForestRegressor(n_estimators=150, random_state=seed).fit(Ftr, ytr)
        out[name] = dict(r2_linear=r2_score(yte, lr.predict(Fte)),
                         r2_forest=r2_score(yte, rf.predict(Fte)),
                         seconds=time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    X, df, _ = load_adni_thickness()
    C = _covar_design(df)
    y = pd.to_numeric(df["CDRSB"], errors="coerce").to_numpy(float)
    keep = np.isfinite(y) & np.isfinite(C).all(1)
    X, C, y = X[keep], C[keep], y[keep]
    print(f"n={len(y)} p={X.shape[1]} k={args.k} splits={args.n_splits}")

    rows = []
    for s in range(args.n_splits):
        res = one_split(X, C, y, args.k, seed=s)
        for name, r in res.items():
            rows.append(dict(split=s, method=name, **r))
        print(f"  split {s:2d}: " + "  ".join(
            f"{n.split()[0] if n.startswith('NSA') else n[:3]}"
            f"{'-' + n.split()[1] if n.startswith('NSA') else ''}:{r['r2_forest']:.3f}"
            for n, r in res.items()))
    df_ = pd.DataFrame(rows)
    df_.to_csv(RESULTS / "adni_cdrsb_repeated.csv", index=False)

    # paired against PCA on the same splits
    piv = {m: df_.pivot(index="split", columns="method", values=m) for m in ("r2_linear", "r2_forest")}
    summary = []
    for name, _, _ in METHODS:
        row = {"method": name}
        for m in ("r2_linear", "r2_forest"):
            v = piv[m][name]
            d = v - piv[m]["PCA"]
            ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d))) \
                if name != "PCA" else (0.0, 0.0)
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_sd"] = v.std(ddof=1)
            row[f"d{m}_vs_PCA"] = d.mean()
            row[f"d{m}_ci_lo"], row[f"d{m}_ci_hi"] = ci
        summary.append(row)
    S = pd.DataFrame(summary)
    S.to_csv(RESULTS / "adni_cdrsb_repeated_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
