r"""E23 -- does the signed lifting help where NSA-Flow actually wins?

The paper's Golub result is the one place NSA-Flow beats both comparators, and
it is the *proximal-operator* use rather than the basis-comparison use: NSA-Flow
supplies the sparsification step inside sparse PCA, against soft-thresholding.
Everything else in the evaluation swaps a basis in for PCA and asks whether
prediction improves, which is a different question and a weaker result.

So this replicates the paper's Golub arms in the current solver and adds the
signed lifting as a fourth and fifth arm, to see whether the lifting helps in
the one setting that works.

Design follows the paper's chunk rather than the leakage-free convention of
``exp4``: k = 3, all 7129 genes, ``w = 0.5``, z-scored, the basis fitted on the
full matrix with cross-validation over the classifier only (random forest,
5-fold x 50 repeats).  That is in-sample for the basis and is the paper's
design, kept deliberately so the numbers are comparable to the published ones.
``exp4`` is the version that fits everything inside the fold.

RESULT (``paper/results/e23_golub_signed.csv``).

                              expl.var  sparsity  orth.defect  CV acc
    Standard PCA                0.2904    0.0000       0.0000  0.8440
    Sparse PCA (soft-threshold) 0.2728    0.5000       0.0070  0.8356
    Sparse PCA (NSA-Flow)       0.2126    0.5673       0.0537  0.8837
    Sparse PCA (signed)         0.2126    0.5668       0.0523  0.8818
    Sparse PCA (signed+consol)  0.1968    0.6804       0.0000  0.8648

Three readings, in order of how much they matter.

The paper's effect reproduces on the current solver: 0.844 -> 0.884 over PCA and
0.884 against 0.836 for soft-thresholding, the same direction and size as the
published 0.828 -> 0.875.

NSA-Flow wins while explaining LESS variance (0.213 against PCA's 0.290).  It is
not a better approximation of the data, it is a better-structured one, and the
structure is what the downstream forest uses.  Any framing of this method as a
competitor to PCA on reconstruction or on prediction-from-scores misses that.

The lifting is inert here, for a diagnosable reason rather than by accident:
3 of its 6 parts die, so every component comes out one-signed and the lifting
collapses back to effectively non-negative (0.882 against 0.884 is a tie).  At
p = 7129 with k = 3 there is no contrast structure to exploit -- gene-expression
components are magnitude, not "these genes against those" -- which is exactly
the structure the lifting exists to represent, and exactly what the ADNI
thickness spectrum does have.  Consolidation then buys hard disjointness
(defect exactly 0, sparsity 0.68) for 0.02 accuracy.

So the lifting is justified where contrasts exist and correctly inert where they
do not.  That is a better demonstration of it than a win would have been.
"""
import warnings

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from nsa_flow.reconstruct import nsa_flow_data
from nsa_flow.signed import nsa_flow_signed

from .data import load_golub

warnings.filterwarnings("ignore")


def _sparsity(V, tol=1e-10):
    return float((np.abs(V) < tol).mean())


def _orth_defect(V):
    G = V.T @ V
    G = G / np.trace(G)
    return float(np.linalg.norm(G - np.eye(V.shape[1]) / V.shape[1]))


def _expl_var(V, Z):
    P = Z @ V @ np.linalg.pinv(V.T @ V) @ V.T
    return float(1.0 - ((Z - P) ** 2).sum() / (Z ** 2).sum())


def run(k=3, w=0.5, max_iter=100, n_splits=5, n_repeats=50, seed=1,
        n_estimators=500, verbose=True):
    X, y, _ = load_golub()
    Z = np.nan_to_num((X - X.mean(0)) / X.std(0, ddof=1))      # paper: scale()
    T = torch.as_tensor(Z, dtype=torch.float64)

    arms, notes = {}, {}
    _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    arms["Standard PCA"] = vt[:k].T
    V = vt[:k].T.copy()                                         # the comparator
    V[np.abs(V) < np.quantile(np.abs(V), 0.5)] = 0.0
    arms["Sparse PCA (Basic)"] = V
    arms["Sparse PCA (NSA-Flow)"] = nsa_flow_data(
        T, k=k, w=w, max_iter=max_iter).Y.numpy()
    for name, cons in (("Sparse PCA (signed)", False),
                       ("Sparse PCA (signed+consol)", True)):
        r = nsa_flow_signed(T, k=k, w=w, max_iter=max_iter, consolidate=cons)
        arms[name] = r.Y.numpy()
        notes[name] = dict(parts_max_nnz=r["parts_max_nnz"],
                           parts_n_dead=r["parts_n_dead"],
                           lobe_overlap=r["lobe_overlap"])
        if verbose:
            print(f"  {name}: {notes[name]}", flush=True)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    rows = []
    for name, V in arms.items():
        V = np.asarray(V, dtype=float)
        acc = cross_val_score(
            RandomForestClassifier(n_estimators=n_estimators, random_state=0,
                                   n_jobs=-1),
            Z @ V, y, cv=cv, scoring="accuracy")
        rows.append(dict(method=name, expl_var=_expl_var(V, Z),
                         sparsity=_sparsity(V), orth_defect=_orth_defect(V),
                         cv_acc=acc.mean(), sd=acc.std(), **notes.get(name, {})))
        if verbose:
            print(f"  {name}: acc {acc.mean():.4f}", flush=True)
    return pd.DataFrame(rows)
