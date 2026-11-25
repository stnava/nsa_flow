"""Synthetic datasets: a planted partition (positive control) and a pure-noise matrix."""
import numpy as np

from .base import Dataset, dataset


def planted_partition(p=60, k=6, n=400, noise=0.3, seed=0, overlap=0):
    """Data with a known non-negative, disjoint-support basis ``V`` (the ``w -> 1`` solution class)."""
    rng = np.random.default_rng(seed)
    V = np.zeros((p, k))
    per = p // k
    for j in range(k):
        V[j * per:(j + 1) * per, j] = 1.0
        if overlap:
            lo = (j + 1) % k * per
            V[lo:lo + overlap, j] = 0.7
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    Z = rng.random((n, k)) * 5.0
    return Z @ V.T + noise * rng.standard_normal((n, p)), V, Z


@dataset("planted", source="synthetic", description="planted partition: 5 disjoint groups of 12 features; "
                                                    "outcome is a smooth function of the latent module scores")
def load_planted(n=300, p=60, k=5, noise=1.0, seed=0):
    X, V, Z = planted_partition(p=p, k=k, n=n, noise=noise, seed=seed)
    rng = np.random.default_rng(seed + 1)
    y = Z[:, 0] + 0.5 * Z[:, 1] ** 2 / 5 + 0.5 * rng.standard_normal(n)
    groups = np.repeat(np.arange(k), p // k)
    return Dataset("planted", X, y=y, groups=groups, task="regression", outer="holdout",
                   draw_n=n // 2, meta={"source": "synthetic"})


@dataset("noise", source="synthetic", description="i.i.d. Gaussian features and outcome; no structure at all")
def load_noise(n=300, p=60, seed=0):
    rng = np.random.default_rng(seed)
    return Dataset("noise", rng.standard_normal((n, p)), y=rng.standard_normal(n), task="regression",
                   outer="holdout", draw_n=n // 2, meta={"source": "synthetic", "null": "no structure"})
