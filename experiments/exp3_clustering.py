"""E3 -- the w=1 limit is a hard clustering of features.

Non-negativity plus orthogonality forces disjoint supports, so the w=1 problem
is a feature partition.  This measures how closely the NSA solution agrees with
k-means on the features, and how the agreement degrades as w is relaxed.
"""
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from nsa_flow import nsa_flow
from .data import planted_partition

warnings.filterwarnings("ignore")
F64 = torch.float64
WS = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def _assign(L, rel=1e-6):
    """Assign each feature to its dominant component (-1 if inactive)."""
    L = np.abs(L)
    lab = L.argmax(axis=1)
    lab[L.max(axis=1) <= rel * L.max()] = -1
    return lab


def run():
    rows = []
    for seed in range(10):
        for noise in [0.2, 0.4, 0.8]:
            X, V, _ = planted_partition(p=60, k=6, n=300, noise=noise, seed=seed)
            truth = V.argmax(axis=1)
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)

            km = KMeans(n_clusters=6, n_init=20, random_state=0).fit(Xs.T)
            rows.append(dict(method="k-means on features", w=np.nan, seed=seed,
                             noise=noise, ari_truth=adjusted_rand_score(truth, km.labels_),
                             ari_kmeans=1.0, n_empty=0))

            L0 = np.abs(PCA(n_components=6, random_state=0).fit(Xs).components_.T)
            for w in WS:
                r = nsa_flow(torch.as_tensor(L0, dtype=F64), w=w,
                             max_iter=40000, tol=1e-12)
                lab = _assign(r.Y.numpy())
                rows.append(dict(
                    method=f"NSA (w={w})", w=w, seed=seed, noise=noise,
                    ari_truth=adjusted_rand_score(truth, lab),
                    ari_kmeans=adjusted_rand_score(km.labels_, lab),
                    n_empty=int(6 - len(set(lab[lab >= 0]))),
                    defect=r.raw_defect, eff_rank=r.effective_rank))
    return pd.DataFrame(rows)


def summarise(df):
    return (df.groupby(["method", "w"], dropna=False)
              .agg(ari_truth=("ari_truth", "mean"), ari_truth_sd=("ari_truth", "std"),
                   ari_kmeans=("ari_kmeans", "mean"), n=("ari_truth", "size"))
              .reset_index().sort_values("w", na_position="first"))
