"""Module-recovery metrics: exact on constructed input, near 1 on a planted partition, ~0 on noise."""
import numpy as np

from nsa_flow.bench.metrics import dominant_assignment, module_ari, module_nmi, split_half_ari
from nsa_flow.bench.datasets.synthetic import planted_partition


def test_dominant_assignment_and_perfect_ari():
    V = np.zeros((6, 2)); V[:3, 0] = 1; V[3:, 1] = 2
    truth = np.array([0, 0, 0, 1, 1, 1])
    assert dominant_assignment(V).tolist() == [0, 0, 0, 1, 1, 1]
    assert module_ari(V, truth) == 1.0 and module_nmi(V, truth) == 1.0
    assert module_ari(V[:, ::-1], truth) == 1.0  # label permutation invariant


def test_zero_rows_are_unassigned():
    V = np.zeros((4, 2)); V[0, 0] = 1; V[1, 1] = 1
    assert dominant_assignment(V).tolist() == [0, 1, -1, -1]


def test_nsa_recovers_planted_partition():
    from nsa_flow import nsa_flow
    X, _, _ = planted_partition(p=60, k=5, n=300, noise=1.0, seed=0)
    V = nsa_flow(X - X.mean(0), k=5, mode="signed", w=0.5, consolidate=True).Y.detach().numpy()
    assert module_ari(V, np.repeat(np.arange(5), 12)) > 0.9


def test_split_half_ari_near_zero_on_noise():
    from sklearn.decomposition import PCA
    X = np.random.default_rng(0).standard_normal((300, 40))
    m, se = split_half_ari(lambda A: PCA(4, random_state=0).fit(A).components_.T, X, n_rep=6)
    assert abs(m) < 0.1
