"""Shared pipeline and evaluation for the paper's experiments.

Every estimator is fitted strictly inside the training fold -- standardisation,
covariate adjustment, component extraction and NSA refinement alike.  The
components are unsupervised, but they are fitted on data whose scaling would
otherwise carry information from the held-out samples, so fitting them on the
full matrix inflates every downstream score.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, PCA, SparsePCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nsa_flow import nsa_flow, stiefel_defect_normalised

F64 = torch.float64


# --------------------------------------------------------------- loading matrices
class NSAPCA(BaseEstimator, TransformerMixin):
    """PCA loadings refined by NSA-Flow: the "NSA-PCA" pipeline.

    ``w = 0`` reduces to non-negative PCA loadings (``max(0, .)``), so the whole
    family is indexed by one interpretable parameter.
    """

    def __init__(self, n_components=5, w=0.5, max_iter=5000, tol=1e-10):
        self.n_components = n_components
        self.w = w
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        pca = PCA(n_components=self.n_components, svd_solver="randomized",
                  random_state=0).fit(X)
        # PCA loadings are sign-ambiguous; NSA seeks a non-negative basis, so the
        # magnitude of the loading is the meaningful quantity.
        L = np.abs(pca.components_.T)
        res = nsa_flow(torch.as_tensor(L, dtype=F64), w=self.w,
                       max_iter=self.max_iter, tol=self.tol)
        self.components_ = res.Y.numpy()
        self.defect_ = res.defect
        self.effective_rank_ = res.effective_rank
        self.fidelity_ = res.fidelity
        self.pca_defect_ = float(stiefel_defect_normalised(torch.as_tensor(L, dtype=F64)))
        return self

    def transform(self, X):
        return X @ self.components_


class PCALoadings(BaseEstimator, TransformerMixin):
    """Plain PCA, as a baseline expressed in the same loading-matrix form."""

    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y=None):
        pca = PCA(n_components=self.n_components, svd_solver="randomized",
                  random_state=0).fit(X)
        self.components_ = pca.components_.T
        return self

    def transform(self, X):
        return X @ self.components_


class SparsePCALoadings(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5, alpha=1.0):
        self.n_components = n_components
        self.alpha = alpha

    def fit(self, X, y=None):
        m = SparsePCA(n_components=self.n_components, alpha=self.alpha,
                      random_state=0, max_iter=200).fit(X)
        self.components_ = m.components_.T
        return self

    def transform(self, X):
        return X @ self.components_


class NMFLoadings(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y=None):
        self._shift = X.min()
        m = NMF(n_components=self.n_components, init="nndsvd", random_state=0,
                max_iter=1000, tol=1e-5).fit(X - self._shift)
        self.components_ = m.components_.T
        return self

    def transform(self, X):
        return (X - self._shift) @ self.components_


# ------------------------------------------------------------------ interpretability
def sparsity(L, rel=1e-3):
    """Fraction of loading entries below ``rel`` times the column maximum."""
    L = np.abs(L)
    thresh = rel * L.max(axis=0, keepdims=True)
    return float((L <= thresh).mean())


def support_overlap(L, rel=1e-3):
    """Mean number of components each active feature loads on, minus one.

    Zero means every feature belongs to exactly one component: a clean, readable
    factor. Larger values mean features are shared across components.
    """
    L = np.abs(L)
    active = L > rel * L.max(axis=0, keepdims=True)
    per_feature = active.sum(axis=1)
    live = per_feature > 0
    return float(per_feature[live].mean() - 1.0) if live.any() else 0.0


def matched_cosine(L, V_true):
    """Mean |cosine| between recovered and true columns under the best matching."""
    A = L / (np.linalg.norm(L, axis=0, keepdims=True) + 1e-300)
    B = V_true / (np.linalg.norm(V_true, axis=0, keepdims=True) + 1e-300)
    C = np.abs(A.T @ B)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())


def support_f1(L, V_true, rel=1e-3):
    """F1 of recovered vs true support, under the best column matching."""
    A = np.abs(L) > rel * np.abs(L).max(axis=0, keepdims=True)
    B = V_true > 0
    An = L / (np.linalg.norm(L, axis=0, keepdims=True) + 1e-300)
    Bn = V_true / (np.linalg.norm(V_true, axis=0, keepdims=True) + 1e-300)
    r, c = linear_sum_assignment(-np.abs(An.T @ Bn))
    f1s = []
    for i, j in zip(r, c):
        tp = (A[:, i] & B[:, j]).sum()
        prec = tp / max(A[:, i].sum(), 1)
        rec = tp / max(B[:, j].sum(), 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


# ------------------------------------------------------------------- CV evaluation
def cv_score(X, y, loader, n_components=5, n_splits=5, n_repeats=4, seed=0,
             covariates=None, C=1.0, n_features=None):
    """Repeated stratified CV AUC and accuracy for a loading-matrix method.

    ``loader`` is a callable returning a fresh unfitted transformer.  Everything
    is fitted on the training fold only.  ``covariates`` (``[n, c]``) are
    regressed out of ``X`` using coefficients estimated on the training fold.

    ``n_features``, if given, keeps that many columns of highest variance.  The
    selection is unsupervised and computed on the training fold only, so it
    carries no information about the held-out labels; it is applied to every
    method identically.  It exists because sparse PCA's cost grows superlinearly
    in the number of features (112 s at p=7129 against 1.4 s at p=2000 for this
    data), which would otherwise make the comparison intractable rather than
    unfair.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    binary = len(np.unique(y)) == 2
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs, accs, defects, sparsities, overlaps = [], [], [], [], []

    for tr, te in cv.split(X, y):
        Xtr, Xte = X[tr], X[te]
        if n_features is not None and n_features < Xtr.shape[1]:
            keep = np.argsort(Xtr.var(axis=0))[::-1][:n_features]
            Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        if covariates is not None:
            Ctr = np.column_stack([np.ones(len(tr)), covariates[tr]])
            Cte = np.column_stack([np.ones(len(te)), covariates[te]])
            beta, *_ = np.linalg.lstsq(Ctr, Xtr, rcond=None)
            Xtr = Xtr - Ctr @ beta
            Xte = Xte - Cte @ beta

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("load", loader()),
            ("clf", LogisticRegression(max_iter=5000, C=C)),
        ])
        pipe.fit(Xtr, y[tr])
        accs.append(pipe.score(Xte, y[te]))
        if binary:
            aucs.append(roc_auc_score(y[te], pipe.predict_proba(Xte)[:, 1]))
        else:
            aucs.append(roc_auc_score(y[te], pipe.predict_proba(Xte),
                                      multi_class="ovr", average="macro"))
        L = pipe.named_steps["load"].components_
        defects.append(float(stiefel_defect_normalised(torch.as_tensor(L, dtype=F64))))
        sparsities.append(sparsity(L))
        overlaps.append(support_overlap(L))

    return dict(
        auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
        acc=float(np.mean(accs)), acc_sd=float(np.std(accs)),
        defect=float(np.mean(defects)), sparsity=float(np.mean(sparsities)),
        overlap=float(np.mean(overlaps)), n_fits=len(aucs),
    )
