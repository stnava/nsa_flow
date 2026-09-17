"""Pressure test of the NHANES dietary -> mortality result.

The headline was: nsa_raw beats centred PCA by +0.0039 AUC (forest, p < 1e-4)
and NMF by +0.0034, with a shuffled-support null at +0.0096 over covariates
against PCA's +0.0139.  Five ways that could be spurious, each tested here.

A  ENERGY ADJUSTMENT -- the one that matters.  Nutrient intakes are
   compositional: total energy drives all 77 columns, so a shared positive
   offset is built into the data.  Today established that this method is
   sensitive to exactly that structure.  If the advantage is an energy effect it
   dies under nutrient densities (per 1000 kcal) or the residual method.
B  k -- an advantage only at k = 5 would be a fluke of rank.
C  w -- likewise for the one dial.
D  SURVEY WEIGHTS + SURVIVAL -- NHANES is a weighted multi-stage sample and the
   outcome is censored time-to-event, not a coin flip.  Weighted fitting and a
   Harrell C-index against (time, event) both applied.
E  SAMPLE SIZE -- does the gap shrink with n (real effect) or appear only at
   full n (over-reading of a tiny difference)?
"""
import os, sys, time, warnings
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
from nsa_flow import nsa_flow_data, nsa_flow_signed

P = "/Users/stnava/Downloads/cleaned_nhanes_21743372/"
CACHE = os.path.expanduser("~/.cache/nsa_flow_nhanes.npz")
OUT = "paper/results/e29_nhanes_harden.csv"
F64 = torch.float64
t0 = time.time()

try:
    z = np.load(CACHE)
    X, y, C, E, WT, TT = (z["X"], z["y"], z["C"], z["E"], z["WT"], z["TT"])
except Exception:
    ID = ["Unnamed: 0", "SEQN", "SEQN_new", "SDDSRVYR"]
    di = pd.read_csv(P + "dietary_clean.csv", low_memory=False)
    num = di.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ID if c in di], errors="ignore")
    cols = list(num.notna().mean().pipe(lambda c: c[c >= 0.9]).index)
    cols = [c for c in cols if c not in {"RIDAGEYR", "RIAGENDR", "survey_day"}
            and not c.startswith("VNDRXS")]
    keepc = ["SEQN", "WTDRD1"] + cols
    D = di[[c for c in keepc if c in di.columns]].dropna().groupby("SEQN").mean().reset_index()
    mo = pd.read_csv(P + "mortality_clean.csv", low_memory=False)[
        ["SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_INT"]].drop_duplicates("SEQN")
    de = pd.read_csv(P + "demographics_clean.csv", low_memory=False)[
        ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "DMDEDUC2",
         "SDDSRVYR"]].drop_duplicates("SEQN")
    M = D.merge(mo, on="SEQN").merge(de, on="SEQN").query("ELIGSTAT == 1")
    cov = pd.get_dummies(
        M[["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "SDDSRVYR"]]
        .assign(**{c: M[c].astype("category") for c in
                   ["RIAGENDR", "RIDRETH1", "DMDEDUC2", "SDDSRVYR"]}),
        drop_first=True, dummy_na=False).astype(float)
    ok = (M[cols].notna().all(axis=1) & cov.notna().all(axis=1)
          & M.MORTSTAT.notna() & M.PERMTH_INT.notna() & M.WTDRD1.notna())
    X = M.loc[ok, cols].to_numpy(float)
    y = M.loc[ok, "MORTSTAT"].to_numpy(int)
    C = cov[ok.to_numpy()].to_numpy(float)
    E = M.loc[ok, "DRXTKCAL"].to_numpy(float)
    WT = M.loc[ok, "WTDRD1"].to_numpy(float)
    TT = M.loc[ok, "PERMTH_INT"].to_numpy(float)
    np.savez(CACHE, X=X, y=y, C=C, E=E, WT=WT, TT=TT,
             cols=np.array(cols, dtype=object))
    del di, num, D, mo, de, M, cov
print(f"X={X.shape} deaths={int(y.sum())} energy median={np.median(E):.0f} kcal "
      f"[{time.time()-t0:.0f}s]", flush=True)
EIDX = None      # index of the energy column, excluded under adjustment
z = np.load(CACHE, allow_pickle=True); COLS = list(z["cols"])
EIDX = COLS.index("DRXTKCAL")


def prep(A, mode):
    """Energy adjustment.  'density' divides by kcal; 'residual' regresses out."""
    if mode == "raw":
        return A
    keep = [j for j in range(A.shape[1]) if j != EIDX]
    e = np.clip(A[:, EIDX], 1.0, None)
    if mode == "density":
        return A[:, keep] / e[:, None] * 1000.0
    R = A[:, keep].copy()                          # residual method, log-log OLS
    le = np.log(e)
    Dm = np.c_[np.ones(len(le)), le]
    beta = np.linalg.lstsq(Dm, np.log1p(R), rcond=None)[0]
    return np.log1p(R) - Dm @ beta


def basis(A, arm, k, w, rs):
    if arm == "pca":
        return PCA(n_components=k).fit(A).components_.T
    if arm == "nmf":
        return NMF(n_components=k, init="nndsvd", max_iter=250,
                   random_state=0).fit(A - A.min() if A.min() < 0 else A).components_.T
    T = torch.as_tensor(A - A.min() if A.min() < 0 else A, dtype=F64)
    if arm == "nsa_raw":
        return nsa_flow_data(T, k=k, w=w).Y.numpy()
    if arm == "consolidated":
        return nsa_flow_signed(T, k=k, w=w, consolidate=True).Y.numpy()
    V = nsa_flow_data(T, k=k, w=w).Y.numpy()
    return V[rs.permutation(V.shape[0])]           # NULL, matched density


def cindex(risk, tt, ev, rs, npairs=200000):
    """Harrell C on sampled comparable pairs."""
    n = len(tt); i = rs.randint(0, n, npairs); j = rs.randint(0, n, npairs)
    ok = ((ev[i] == 1) & (tt[i] < tt[j])) | ((ev[j] == 1) & (tt[j] < tt[i]))
    i, j = i[ok], j[ok]
    first = np.where(tt[i] < tt[j], i, j); second = np.where(tt[i] < tt[j], j, i)
    conc = (risk[first] > risk[second]).sum() + 0.5 * (risk[first] == risk[second]).sum()
    return float(conc / len(first))


def run(mode="raw", k=5, w=0.5, arms=("pca", "nmf", "nsa_raw", "consolidated", "NULL"),
        n_splits=5, n_repeats=2, weighted=False, sub=None, seed=0, surv=False):
    idx = np.arange(len(X))
    if sub is not None and sub < len(X):
        idx = np.random.RandomState(7).choice(len(X), sub, replace=False)
    Xs, ys, Cs, Ws, Ts = X[idx], y[idx], C[idx], WT[idx], TT[idx]
    A_all = prep(Xs, mode)
    acc = {a: [] for a in list(arms) + ["cov"]}
    cix = {a: [] for a in arms}
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    for f, (tr, te) in enumerate(cv.split(A_all, ys)):
        rs = np.random.RandomState(900 + f)
        sc = StandardScaler().fit(Cs[tr]); Ctr, Cte = sc.transform(Cs[tr]), sc.transform(Cs[te])
        sw = Ws[tr] / Ws[tr].mean() if weighted else None
        mk = lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                            random_state=0, n_jobs=-1)
        m = mk().fit(Ctr, ys[tr], sample_weight=sw)
        acc["cov"].append(roc_auc_score(ys[te], m.predict_proba(Cte)[:, 1]))
        for a in arms:
            V = basis(A_all[tr], a, k, w, rs)
            Str, Ste = A_all[tr] @ V, A_all[te] @ V
            ss = StandardScaler().fit(Str)
            P1, P2 = np.c_[Ctr, ss.transform(Str)], np.c_[Cte, ss.transform(Ste)]
            m = mk().fit(P1, ys[tr], sample_weight=sw)
            pr = m.predict_proba(P2)[:, 1]
            acc[a].append(roc_auc_score(ys[te], pr))
            if surv:
                cix[a].append(cindex(pr, Ts[te], ys[te], rs))
    out = {}
    ref = np.array(acc["pca"])
    for a in list(arms) + ["cov"]:
        v = np.array(acc[a])
        t, p = ((np.nan, np.nan) if a == "pca" else stats.ttest_rel(v, ref))
        out[a] = dict(auc=float(v.mean()), sd=float(v.std()),
                      d_vs_pca=float(v.mean() - ref.mean()), p_vs_pca=float(p),
                      cindex=float(np.mean(cix[a])) if surv and a in cix and cix[a] else None)
    return out


res = {}
STAGES = [
    ("A energy=raw",        dict(mode="raw")),
    ("A energy=density",    dict(mode="density")),
    ("A energy=residual",   dict(mode="residual")),
    ("B k=3",               dict(mode="density", k=3)),
    ("B k=8",               dict(mode="density", k=8)),
    ("B k=12",              dict(mode="density", k=12)),
    ("C w=0.25",            dict(mode="density", w=0.25)),
    ("C w=0.75",            dict(mode="density", w=0.75)),
    ("D weighted+surv",     dict(mode="density", weighted=True, surv=True)),
    ("D seed=1",            dict(mode="density", seed=1)),
    ("E n=8000",            dict(mode="density", sub=8000)),
    ("E n=20000",           dict(mode="density", sub=20000)),
]
for name, kw in STAGES:
    res[name] = run(**kw)
    r = res[name]
    line = "  ".join(f"{a}={r[a]['auc']:.4f}" for a in ("cov", "pca", "nsa_raw"))
    print(f"  {name:20s} {line}  nsa-pca={r['nsa_raw']['d_vs_pca']:+.4f} "
          f"p={r['nsa_raw']['p_vs_pca']:.2g}  [{time.time()-t0:.0f}s]", flush=True)
rows = [dict(stage=st, arm=a, **v) for st, d in res.items() for a, v in d.items()]
pd.DataFrame(rows).to_csv(OUT, index=False)
print("SAVED", OUT, f"{time.time()-t0:.0f}s")
