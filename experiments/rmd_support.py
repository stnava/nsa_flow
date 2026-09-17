r"""Experiment functions for ``paper/nsa_flow_v2.Rmd``.

Each returns a tidy ``DataFrame`` so the document is narrative plus tables and
plots, with the computation here where it can be tested and reused.  Sizes are
chosen so the whole document knits in a few minutes.
"""
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
F64 = torch.float64


# ---------------------------------------------------------------- 1. properties
def properties_table():
    """Does each functional recognise the structures the method is built around?

    ``D`` is orthoNORMality, ``d`` is v1's off-diagonal-only defect, ``C`` is
    orthogonality.  The rows are the cases that separate them.
    """
    from nsa_flow import stiefel_defect, angle_defect
    torch.manual_seed(0)
    p, k = 20, 5
    Q = torch.linalg.qr(torch.randn(p, k, dtype=F64))[0]

    def d_off(V):
        G = V.T @ V
        G = G / G.trace()
        return float((G - torch.diag(torch.diagonal(G))).pow(2).sum())

    cases = {}
    cases["orthonormal"] = Q
    cases["orthogonal, unequal norms"] = Q @ torch.diag(
        torch.tensor([1.0, 1, 1, 1, 8], dtype=F64))
    V = torch.zeros(p, k, dtype=F64)
    for i in range(k):                       # disjoint blocks, unequal scales
        V[i * 4:(i + 1) * 4, i] = torch.rand(4, dtype=F64) * (i + 1) * 5
    cases["disjoint nonneg supports"] = V
    Z = torch.zeros(p, k, dtype=F64)
    Z[:, 0] = torch.rand(p, dtype=F64)
    cases["rank collapse (1 live col)"] = Z
    cases["all columns collinear"] = (torch.rand(p, 1, dtype=F64)
                                      @ torch.ones(1, k, dtype=F64))
    rows = []
    for name, M in cases.items():
        rows.append(dict(case=name, D=float(stiefel_defect(M)), d_v1=d_off(M),
                         C=float(angle_defect(M))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. abs trap
def abs_trap_table(X):
    """Four ways to get a non-negative basis, scored by reconstruction error.

    ``abs()`` is not even the projection onto the non-negative cone: for an entry
    ``-3`` it moves distance 6 where the projection moves 3.  And no one-shot map
    can work, because a signed component encodes a contrast a single
    non-negative column cannot represent.
    """
    from nsa_flow import nsa_flow_data
    from nsa_flow.reconstruct import reconstruction_fidelity
    T = torch.as_tensor(np.asarray(X), dtype=F64)
    S, c = T.T @ T, T.pow(2).sum()
    k = 5
    _, _, Vh = torch.linalg.svd(T, full_matrices=False)
    E = Vh[:k].T

    def err(V):
        return float(reconstruction_fidelity(V, S, c)) ** 0.5

    rows = [
        dict(basis="signed PCA (unconstrained optimum)", nonneg=False, err=err(E)),
        dict(basis="abs(PCA)", nonneg=True, err=err(E.abs())),
        dict(basis="clamp(PCA) -- the true projection", nonneg=True,
             err=err(E.clamp_min(0.0))),
        dict(basis="fitted to the data (w=0)", nonneg=True,
             err=err(nsa_flow_data(T, k=k, w=0.0).Y)),
    ]
    out = pd.DataFrame(rows)
    out["cost_vs_signed"] = out.err / out.err.iloc[0]
    return out


# --------------------------------------------------------------------- 3. dial
def w_dial_table(X, ws=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99), k=5):
    """One parameter, monotone in the quantity it is supposed to control."""
    from nsa_flow import nsa_flow_data, angle_defect
    from nsa_flow.reconstruct import reconstruction_fidelity
    T = torch.as_tensor(np.asarray(X), dtype=F64)
    S, c = T.T @ T, T.pow(2).sum()
    rows = []
    for w in ws:
        r = nsa_flow_data(T, k=k, w=w)
        V = r.Y.numpy()
        sup = np.abs(V) > 1e-10
        live = sup.any(1)
        rows.append(dict(
            w=w, recon_err=float(reconstruction_fidelity(r.Y, S, c)) ** 0.5,
            C=float(angle_defect(r.Y)), sparsity=float((V == 0).mean()),
            overlap=float((sup.sum(1)[live] - 1).mean()) if live.any() else 0.0,
            iters=int(r.iters), converged=bool(r.converged)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 4. homotopy
def homotopy_table(X, k=5, w=0.5):
    """Path resolution, not iteration count, selects the local minimum."""
    from nsa_flow import nsa_flow_data
    from nsa_flow.reconstruct import GramOperator, relax_into_nonneg, _fid
    T = torch.as_tensor(np.asarray(X), dtype=F64)
    ops = GramOperator(S=T.T @ T)
    c = ops.c
    schedules = {
        "1 stage  (mu = 0 only)": [0.0],
        "3 stages": [0.0, 1.0, 1e2],
        "9 stages (default)": [0.0] + [10.0 ** e for e in range(-3, 5)],
    }
    rows = []
    for name, mus in schedules.items():
        for it in (150, 3000):
            t0 = time.time()
            V = relax_into_nonneg(ops, c, k, w, mus=mus, max_iter=it)
            rows.append(dict(schedule=name, iters_per_stage=it,
                             energy=float(_fid(V.clamp_min(0), ops, c)),
                             seconds=time.time() - t0))
    return pd.DataFrame(rows)


# -------------------------------------------------------------- 5. scaling
def scaling_table(n=200, ps=(500, 2000, 8000), k=5, max_iter=60):
    """Matrix-free against forming the Gram matrix."""
    from nsa_flow import nsa_flow_data
    rows = []
    for p in ps:
        torch.manual_seed(0)
        X = torch.rand(n, p, dtype=F64)
        entry = dict(p=p, gram_GB=p * p * 8 / 1e9)
        for mf in (False, True):
            t0 = time.time()
            nsa_flow_data(X, k=k, w=0.5, matrix_free=mf, max_iter=max_iter,
                          init="clamp")
            entry["matrix_free_s" if mf else "gram_s"] = time.time() - t0
        entry["speedup"] = entry["gram_s"] / entry["matrix_free_s"]
        rows.append(entry)
    return pd.DataFrame(rows)


# ------------------------------------------------------- 6. real data frontier
def adni_frontier(n_repeats=2, k=5, ws=(0.0, 0.5, 0.9), alphas=(1.0, 4.0)):
    """Accuracy against interpretability on ADNI, the comparison that matters.

    Every fit -- scaling, covariate adjustment, basis, classifier -- happens
    inside the training fold.  The competitor is sparse PCA, not PCA: sparse PCA
    is the method that already buys interpretability nearly for free.
    """
    from sklearn.base import BaseEstimator, TransformerMixin
    from .common import NMFLoadings, PCALoadings, SparsePCALoadings, cv_score
    from .data import load_adni
    from nsa_flow import nsa_flow_data

    class DataAnchored(BaseEstimator, TransformerMixin):
        def __init__(self, n_components=5, w=0.5):
            self.n_components, self.w = n_components, w

        def fit(self, X, y=None):
            self.components_ = nsa_flow_data(
                torch.as_tensor(np.asarray(X), dtype=F64),
                k=self.n_components, w=self.w).Y.numpy()
            return self

        def transform(self, X):
            return np.asarray(X) @ self.components_

    X, meta, _ = load_adni("right")
    cov = np.column_stack([meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])
    tasks = {"CN vs DEM": ("CN", "DEM"), "CN vs MCI": ("CN", "MCI"),
             "MCI vs DEM": ("MCI", "DEM")}
    specs = [("PCA", lambda: PCALoadings(k), "PCA")]
    specs += [(f"SparsePCA (a={a})", (lambda a=a: SparsePCALoadings(k, alpha=a)),
               "SparsePCA") for a in alphas]
    specs += [("NMF", lambda: NMFLoadings(k), "NMF")]
    specs += [(f"NSA (w={w})", (lambda w=w: DataAnchored(k, w)), "NSA")
              for w in ws]
    rows = []
    for task, (a, b) in tasks.items():
        m = meta.DX.isin([a, b]).to_numpy()
        Xt, yt, ct = X[m], (meta.DX[m] == b).to_numpy(int), cov[m]
        for name, loader, family in specs:
            s = cv_score(Xt, yt, loader, n_components=k, n_splits=5,
                         n_repeats=n_repeats, seed=0, covariates=ct)
            s.update(method=name, family=family, task=task)
            rows.append(s)
    return pd.DataFrame(rows)[["task", "method", "family", "auc", "auc_sd",
                               "overlap", "sparsity", "defect"]]


# ---------------------------------------------------------- 7. reproducibility
def reproducibility_table(n_splits=5, k=5, w=0.5):
    """Do independent subsamples return the same basis?

    PCA's components 2-5 sit in a near-degenerate block on this data (see the
    eigengaps in the document), so Davis-Kahan says their orientation is
    noise-dominated.  A disjointness constraint selects a representative.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from .common import SparsePCALoadings
    from .data import load_adni
    from nsa_flow import nsa_flow_data

    X, meta, _ = load_adni("right")
    cov = np.column_stack([np.ones(len(X)), meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])

    def matched(A, B):
        A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
        Cm = np.abs(A.T @ B)
        r, c = linear_sum_assignment(-Cm)
        return float(Cm[r, c].mean())

    fits = {"PCA": [], "SparsePCA (a=4)": [], "NSA (w=%.2g)" % w: []}
    for tr, _ in KFold(n_splits, shuffle=True, random_state=0).split(X):
        beta, *_ = np.linalg.lstsq(cov[tr], X[tr], rcond=None)
        Xa = StandardScaler().fit_transform(X[tr] - cov[tr] @ beta)
        fits["PCA"].append(PCA(k, random_state=0).fit(Xa).components_.T)
        fits["SparsePCA (a=4)"].append(
            SparsePCALoadings(k, alpha=4.0).fit(Xa).components_)
        fits["NSA (w=%.2g)" % w].append(
            nsa_flow_data(torch.as_tensor(Xa, dtype=F64), k=k, w=w).Y.numpy())
    rows = []
    for name, Ls in fits.items():
        vals = [matched(Ls[i], Ls[j]) for i in range(len(Ls))
                for j in range(i + 1, len(Ls))]
        rows.append(dict(method=name, n_pairs=len(vals),
                         matched_cos=float(np.mean(vals)),
                         sd=float(np.std(vals))))
    return pd.DataFrame(rows)


def adni_spectrum(k=8):
    """Leading eigenvalues and consecutive gap ratios of the analysis matrix."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from .data import load_adni
    X, meta, _ = load_adni("right")
    cov = np.column_stack([np.ones(len(X)), meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])
    beta, *_ = np.linalg.lstsq(cov, X, rcond=None)
    Xa = StandardScaler().fit_transform(X - cov @ beta)
    ev = PCA(k, random_state=0).fit(Xa).explained_variance_
    return pd.DataFrame(dict(component=np.arange(1, k + 1), eigenvalue=ev,
                             gap_ratio=np.r_[ev[:-1] / ev[1:], np.nan]))


# ------------------------------------------------- 8. fidelity-mode comparison
def adni_fidelity_modes(n_repeats=3, k=5, ws=(0.25, 0.5, 0.9)):
    r"""The three ways to ask for a non-negative basis near PCA's, on ADNI.

    ``anchor``    -- entrywise \|Y - X0\|^2, X0 the signed PCA loadings.  The
                     negative entries are unreachable, so this degenerates
                     toward max(0, X0); reported for comparison.
    ``subspace``  -- sign-blind \|(I-P)Y\|^2/\|Y\|^2, anchoring to range(X0).
                     Refinement, done without an unreachable target.
    ``data``      -- \|X - XVV'\|^2, no target matrix at all.
    """
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.decomposition import PCA
    from .common import PCALoadings, SparsePCALoadings, cv_score
    from .data import load_adni
    from nsa_flow import nsa_flow, nsa_flow_data

    class Refine(BaseEstimator, TransformerMixin):
        """Refine the (signed) PCA loadings under a chosen fidelity."""

        def __init__(self, n_components=5, w=0.5, fidelity="subspace"):
            self.n_components, self.w, self.fidelity = n_components, w, fidelity

        def fit(self, X, y=None):
            L = PCA(self.n_components, svd_solver="randomized",
                    random_state=0).fit(np.asarray(X)).components_.T
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = nsa_flow(torch.as_tensor(L, dtype=F64), w=self.w,
                             fidelity=self.fidelity, max_iter=4000)
            self.components_ = r.Y.numpy()
            self.mode_ = r["fidelity_mode"]
            self.clamp_distance_ = r["clamp_distance"]
            return self

        def transform(self, X):
            return np.asarray(X) @ self.components_

    class Data(BaseEstimator, TransformerMixin):
        def __init__(self, n_components=5, w=0.5):
            self.n_components, self.w = n_components, w

        def fit(self, X, y=None):
            self.components_ = nsa_flow_data(
                torch.as_tensor(np.asarray(X), dtype=F64),
                k=self.n_components, w=self.w).Y.numpy()
            return self

        def transform(self, X):
            return np.asarray(X) @ self.components_

    X, meta, _ = load_adni("right")
    cov = np.column_stack([meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])
    tasks = {"CN vs DEM": ("CN", "DEM"), "CN vs MCI": ("CN", "MCI"),
             "MCI vs DEM": ("MCI", "DEM")}
    specs = [("PCA", lambda: PCALoadings(k), "PCA", np.nan),
             ("SparsePCA (a=4)", lambda: SparsePCALoadings(k, alpha=4.0),
              "SparsePCA", np.nan)]
    for w in ws:
        specs.append((f"anchor (w={w})", (lambda w=w: Refine(k, w, "anchor")),
                      "anchor", w))
        specs.append((f"subspace (w={w})", (lambda w=w: Refine(k, w, "subspace")),
                      "subspace", w))
        specs.append((f"data (w={w})", (lambda w=w: Data(k, w)), "data", w))
    rows = []
    for task, (a, b) in tasks.items():
        m = meta.DX.isin([a, b]).to_numpy()
        Xt, yt, ct = X[m], (meta.DX[m] == b).to_numpy(int), cov[m]
        for name, loader, family, w in specs:
            s = cv_score(Xt, yt, loader, n_components=k, n_splits=5,
                         n_repeats=n_repeats, seed=0, covariates=ct)
            s.update(method=name, family=family, w=w, task=task)
            rows.append(s)
            print(f"{task:11s} {name:18s} auc={s['auc']:.4f} "
                  f"overlap={s['overlap']:.2f} sparsity={s['sparsity']:.3f}",
                  flush=True)
    return pd.DataFrame(rows)[["task", "method", "family", "w", "auc", "auc_sd",
                               "overlap", "sparsity", "defect"]]


# ----------------------------------------------- 9. ADNI cortical thickness
THK = Path(os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/code/multidisorder/data/"
    "ppmiadni_filtered.csv"))

COG_VARS = ["CDRSB", "ADAS13", "ADASQ4", "MMSE", "FAQ", "mPACCdigit",
            "EcogPtTotal", "EcogSPTotal", "LDELTOTAL"]
COG_COVARS = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4"]


def load_adni_thickness(path=None):
    """Baseline ADNI cortical thickness (bilateral averages) plus covariates."""
    p = Path(os.path.expanduser(path)) if path else THK
    if not p.exists():
        raise FileNotFoundError(f"ADNI thickness table not found: {p}")
    df = pd.read_csv(p, low_memory=False)
    df = df[(df.studyName == "ADNI") & (df.yearsbl == 0)]
    regions = [c for c in df.columns
               if "T1Hier_thk_" in c and "LRAVG" in c
               and not any(x in c for x in ("Asym", "reference", "adjusted"))]
    X = df[regions].apply(pd.to_numeric, errors="coerce")
    ok = X.notna().all(axis=1)
    X, df = X[ok], df[ok]
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), df.reset_index(drop=True), list(X.columns)


def _basis(Xc, k, w, mode):
    """One of the three ways to obtain a non-negative basis."""
    from sklearn.decomposition import PCA
    from nsa_flow import nsa_flow, nsa_flow_data
    T = torch.as_tensor(np.ascontiguousarray(Xc), dtype=F64)
    if mode == "pca":
        L = PCA(k, random_state=0).fit(Xc).components_.T
        return L / np.linalg.norm(L, axis=0, keepdims=True)
    if mode == "data":
        return nsa_flow_data(T, k=k, w=w).Y.numpy()
    if mode in ("signed", "signed_consolidated"):
        # V = V+ - V-, the lifting; .Y is the signed p x k basis
        from nsa_flow import nsa_flow_signed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nsa_flow_signed(
                T, k=k, w=w, init="split",
                consolidate=(mode == "signed_consolidated")).Y.numpy()
    L = PCA(k, random_state=0).fit(Xc).components_.T
    L = L / np.linalg.norm(L, axis=0, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return nsa_flow(torch.as_tensor(L, dtype=F64), w=w, fidelity=mode,
                        max_iter=4000).Y.numpy()


def _nested_F(y, Z0, Z1):
    """p-value of the nested F-test comparing design Z0 against Z1 (Z1 adds columns)."""
    from scipy.stats import f as fdist
    def rss(Z):
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return float(((y - Z @ beta) ** 2).sum())
    r0, r1 = rss(Z0), rss(Z1)
    q = Z1.shape[1] - Z0.shape[1]
    dfe = len(y) - Z1.shape[1]
    if dfe <= 0 or r1 <= 0:
        return np.nan
    F = ((r0 - r1) / q) / (r1 / dfe)
    return float(fdist.sf(F, q, dfe))


def cognitive_insample(k=5, ws=(0.25, 0.5, 0.75, 0.9),
                       modes=("anchor", "subspace", "data")):
    """The paper's original design: in-sample nested F-test, reported as log p.

    Answers "do these network scores add explanatory power beyond the
    covariates, and does NSA add more than PCA".  It is NOT held-out prediction;
    ``cognitive_cv`` is.
    """
    X, df, _ = load_adni_thickness()
    Xc = X - X.mean(0)
    cogs = [c for c in COG_VARS if c in df.columns]
    C = _covar_design(df)
    rows = []
    for mode in modes:
        for w in ws:
            V = _basis(Xc, k, w, mode)
            if np.any(V.var(0) == 0):
                continue
            S_nsa, S_pca = X @ V, X @ _basis(Xc, k, w, "pca")
            for cog in cogs:
                y = pd.to_numeric(df[cog], errors="coerce").to_numpy(float)
                m = np.isfinite(y) & np.isfinite(C).all(1)
                for nm, S in (("nsa", S_nsa), ("pca", S_pca)):
                    mm = m & np.isfinite(S).all(1)
                    p = _nested_F(y[mm], C[mm], np.column_stack([C[mm], S[mm]]))
                    rows.append(dict(mode=mode, w=w, cog=cog, method=nm,
                                     log_p=np.log(max(p, 1e-300))))
    return pd.DataFrame(rows)


def _covar_design(df):
    age = pd.to_numeric(df.AGE, errors="coerce").to_numpy(float)
    sex = (df.PTGENDER.astype(str).str.upper().str.startswith("M")).to_numpy(float)
    edu = pd.to_numeric(df.PTEDUCAT, errors="coerce").to_numpy(float)
    apo = pd.to_numeric(df.APOE4, errors="coerce").to_numpy(float)
    return np.column_stack([np.ones(len(df)), age, sex, edu, apo])


def cognitive_cv(k=5, ws=(0.5, 0.9), modes=("anchor", "subspace", "data"),
                 n_splits=5, n_repeats=4, seed=0, model="linear"):
    r"""Held-out prediction of cognitive scores --- what the section title claims.

    Everything is fitted inside the training fold: the centring, the basis
    (PCA and NSA alike), and the regression.  We report out-of-sample
    \(\Delta R^2\), the gain from adding the network scores to the covariate
    model, so the comparison is against a real baseline rather than against
    nothing.  ``model="rf"`` uses a random forest, matching the machinery of the
    diagnosis analysis.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import RepeatedKFold

    X, df, _ = load_adni_thickness()
    cogs = [c for c in COG_VARS if c in df.columns]
    C = _covar_design(df)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    def fit_pred(Ztr, ytr, Zte):
        if model == "rf":
            m = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
        else:
            m = LinearRegression()
        return m.fit(Ztr, ytr).predict(Zte)

    rows = []
    for cog in cogs:
        y = pd.to_numeric(df[cog], errors="coerce").to_numpy(float)
        keep = np.isfinite(y) & np.isfinite(C).all(1)
        Xk, Ck, yk = X[keep], C[keep], y[keep]
        for mode in modes:
            for w in ws:
                r2_base, r2_pca, r2_nsa = [], [], []
                for tr, te in cv.split(Xk):
                    mu = Xk[tr].mean(0)
                    Xtr, Xte = Xk[tr] - mu, Xk[te] - mu
                    try:
                        Vn = _basis(Xtr, k, w, mode)
                        Vp = _basis(Xtr, k, w, "pca")
                    except Exception:
                        continue
                    sst = float(((yk[te] - yk[tr].mean()) ** 2).sum())
                    if sst <= 0:
                        continue
                    def r2(Ztr, Zte):
                        pr = fit_pred(Ztr, yk[tr], Zte)
                        return 1.0 - float(((yk[te] - pr) ** 2).sum()) / sst
                    r2_base.append(r2(Ck[tr], Ck[te]))
                    r2_pca.append(r2(np.column_stack([Ck[tr], Xtr @ Vp]),
                                     np.column_stack([Ck[te], Xte @ Vp])))
                    r2_nsa.append(r2(np.column_stack([Ck[tr], Xtr @ Vn]),
                                     np.column_stack([Ck[te], Xte @ Vn])))
                if not r2_base:
                    continue
                rows.append(dict(
                    cog=cog, mode=mode, w=w, model=model, n=int(keep.sum()),
                    r2_covariates=np.mean(r2_base),
                    r2_pca=np.mean(r2_pca), r2_nsa=np.mean(r2_nsa),
                    dR2_pca=np.mean(r2_pca) - np.mean(r2_base),
                    dR2_nsa=np.mean(r2_nsa) - np.mean(r2_base),
                    nsa_minus_pca=np.mean(r2_nsa) - np.mean(r2_pca),
                    sd_nsa_minus_pca=np.std(np.array(r2_nsa) - np.array(r2_pca)),
                    n_folds=len(r2_base)))
    return pd.DataFrame(rows)
