r"""E12 -- basis reproducibility, the criterion PCA structurally cannot win.

After covariate adjustment and standardisation the ADNI spectrum is

    eigenvalues   15.649  1.612  1.382  1.234  0.996 ...
    gap ratios       9.71  1.167  1.120  1.238  1.209

Component 1 is well separated; components 2-5 sit in a near-degenerate block.
By Davis-Kahan the error of an individual eigenvector scales as the inverse
eigengap, so PCA's components 2-5 have essentially arbitrary orientation within
that block and should rotate or permute between resamples.  A disjointness
constraint selects a particular representative of the block, so it should be far
more reproducible.  This measures that rather than asserting it.

For every pair of training folds we Hungarian-match the two loading matrices on
|cos| and report the matched agreement, plus support Jaccard -- "do we name the
same regions" -- which is the quantity a neuroimaging reader cares about.
"""
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from .common import NSAPCA, SparsePCALoadings
from .data import load_adni
from nsa_flow.reconstruct import nsa_flow_data

warnings.filterwarnings("ignore")
F64 = torch.float64
K = 5


def _unit(L):
    return L / (np.linalg.norm(L, axis=0, keepdims=True) + 1e-12)


def _match(A, B):
    """Hungarian match on |cos|; returns matched |cos| and support Jaccard."""
    A, B = _unit(A), _unit(B)
    C = np.abs(A.T @ B)
    r, c = linear_sum_assignment(-C)
    cos = C[r, c]
    sa, sb = np.abs(A) > 1e-8, np.abs(B) > 1e-8
    jac = []
    for i, j in zip(r, c):
        u = (sa[:, i] | sb[:, j]).sum()
        jac.append(((sa[:, i] & sb[:, j]).sum() / u) if u else 1.0)
    return cos, np.array(jac)


def _fit(kind, Xa):
    if kind == "PCA":
        return PCA(K, random_state=0).fit(Xa).components_.T
    if kind.startswith("SparsePCA"):
        return SparsePCALoadings(K, alpha=4.0).fit(Xa).components_
    if kind.startswith("anchored"):
        return NSAPCA(K, 0.9).fit(Xa).components_
    if kind.startswith("relax"):
        return nsa_flow_data(torch.as_tensor(Xa, dtype=F64), k=K, w=0.9,
                             init="relax").Y.numpy()
    raise ValueError(kind)


def run(task=("CN", "DEM"), n_repeats=4, n_splits=5, seed=0):
    X, meta, regions = load_adni("right")
    cov = np.column_stack([meta.AGE.to_numpy(float),
                           (meta.SEX.astype(str) == "M").to_numpy(float)])
    m = meta.DX.isin(list(task)).to_numpy()
    Xt, yt, ct = X[m], (meta.DX[m] == task[1]).to_numpy(int), cov[m]

    methods = ["PCA", "SparsePCA (a=4)", "anchored (w=0.9)", "relax (w=0.9)"]
    fits = {k: [] for k in methods}
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    for tr, _ in cv.split(Xt, yt):
        C = np.column_stack([np.ones(len(tr)), ct[tr]])
        beta, *_ = np.linalg.lstsq(C, Xt[tr], rcond=None)
        Xa = StandardScaler().fit_transform(Xt[tr] - C @ beta)
        for k in methods:
            fits[k].append(_fit(k, Xa))
        print(f"  fitted fold {len(fits['PCA'])}", flush=True)

    rows = []
    for k, Ls in fits.items():
        cos_all, jac_all, per_comp = [], [], []
        for i in range(len(Ls)):
            for j in range(i + 1, len(Ls)):
                cos, jac = _match(Ls[i], Ls[j])
                cos_all.append(cos.mean()); jac_all.append(jac.mean())
                per_comp.append(np.sort(cos)[::-1])
        pc = np.array(per_comp).mean(0)
        rows.append(dict(method=k, n_pairs=len(cos_all),
                         cos=np.mean(cos_all), cos_sd=np.std(cos_all),
                         jaccard=np.mean(jac_all),
                         **{f"cos_rank{i+1}": pc[i] for i in range(K)}))
    return pd.DataFrame(rows)
