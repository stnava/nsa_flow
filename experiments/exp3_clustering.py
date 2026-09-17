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
from sklearn.metrics import adjusted_rand_score

from nsa_flow import nsa_flow
from .common import NMFLoadings, PCALoadings, SparsePCALoadings
from .data import planted_partition

warnings.filterwarnings("ignore")
F64 = torch.float64
WS = [0.5, 0.75, 0.9, 0.95, 0.99]
BASES = {"PCA": lambda k: PCALoadings(k),
         "NMF": lambda k: NMFLoadings(k),
         "SparsePCA": lambda k: SparsePCALoadings(k, alpha=1.0)}


def _assign(L, rel=1e-6):
    """Assign each feature to its dominant component (-1 if inactive)."""
    L = np.abs(L)
    lab = L.argmax(axis=1)
    lab[L.max(axis=1) <= rel * L.max()] = -1
    return lab


def run():
    """Compare the induced feature partition with truth and with k-means.

    As in E1, NSA refines a supplied basis, so the partition it induces depends
    on that basis; we therefore report all three rather than only PCA, whose
    loadings are the weakest starting point.
    """
    rows = []
    k = 6
    for seed in range(10):
        for noise in [0.2, 0.4, 0.8]:
            X, V, _ = planted_partition(p=60, k=k, n=300, noise=noise, seed=seed)
            truth = V.argmax(axis=1)
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)

            km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs.T)
            rows.append(dict(method="k-means on features", base="--", w=np.nan,
                             seed=seed, noise=noise,
                             ari_truth=adjusted_rand_score(truth, km.labels_),
                             ari_kmeans=1.0))

            for base, mk in BASES.items():
                L0 = mk(k).fit(Xs).components_
                rows.append(dict(method=f"{base} (unrefined)", base=base, w=-1.0,
                                 seed=seed, noise=noise,
                                 ari_truth=adjusted_rand_score(truth, _assign(L0)),
                                 ari_kmeans=adjusted_rand_score(km.labels_, _assign(L0))))
                T = torch.as_tensor(L0, dtype=F64)
                for w in WS:
                    r = nsa_flow(T, w=w, max_iter=40000, tol=1e-12)
                    lab = _assign(r.Y.numpy())
                    rows.append(dict(
                        method=f"{base}+NSA (w={w})", base=base, w=w,
                        seed=seed, noise=noise,
                        ari_truth=adjusted_rand_score(truth, lab),
                        ari_kmeans=adjusted_rand_score(km.labels_, lab),
                        n_empty=int(k - len(set(lab[lab >= 0]))),
                        defect=r.raw_defect, eff_rank=r.effective_rank))
    return pd.DataFrame(rows)


def summarise(df):
    return (df.groupby(["method", "base", "w"], dropna=False)
              .agg(ari_truth=("ari_truth", "mean"), ari_truth_sd=("ari_truth", "std"),
                   ari_kmeans=("ari_kmeans", "mean"), n=("ari_truth", "size"))
              .reset_index().sort_values(["base", "w"], na_position="first"))


def best_table(df):
    """k-means, each unrefined base, and each base's best refinement."""
    rows = [{"method": "k-means on features",
             "ARI vs truth": df[df.method.str.startswith("k-means")].ari_truth.mean(),
             "ARI vs $k$-means": 1.0}]
    for base in BASES:
        b = df[df.method == f"{base} (unrefined)"]
        rows.append({"method": f"{base} (unrefined)",
                     "ARI vs truth": b.ari_truth.mean(),
                     "ARI vs $k$-means": b.ari_kmeans.mean()})
        r = df[(df.base == base) & (df.w > 0)]
        g = r.groupby("w")[["ari_truth", "ari_kmeans"]].mean()
        w = g.ari_truth.idxmax()
        rows.append({"method": f"{base}+NSA ($w={w}$)",
                     "ARI vs truth": g.loc[w, "ari_truth"],
                     "ARI vs $k$-means": g.loc[w, "ari_kmeans"]})
    return pd.DataFrame(rows)
