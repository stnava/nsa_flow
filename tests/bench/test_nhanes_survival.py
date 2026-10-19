"""NHANES-style machinery: AUC metric, derived energy adjustments, subsampling, controls, survival protocol."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from nsa_flow.bench import protocols  # noqa: F401,E402
from nsa_flow.bench.datasets import Dataset, DatasetUnavailable, get_dataset  # noqa: E402
from nsa_flow.bench.metrics import harrell_c  # noqa: E402
from nsa_flow.bench.protocols import nested_cv  # noqa: E402
from nsa_flow.bench.registry import METHODS, PROTOCOLS  # noqa: E402
from nsa_flow.bench.runtime import get_profile  # noqa: E402


def _toy_intake(n=600, p=8, seed=0):
    """Non-negative 'nutrient' matrix with an energy column, covariates, a binary outcome and survival times."""
    rng = np.random.default_rng(seed)
    energy = rng.uniform(1200, 3200, n)
    comp = rng.dirichlet(np.ones(p - 1), n)
    X = np.column_stack([energy, comp * energy[:, None] / 100.0])
    risk = (comp[:, 0] - comp[:, 1]) * 3 + 0.5 * rng.standard_normal(n)
    event = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)
    time = rng.exponential(60 / (1 + np.exp(risk)), n) + 1
    C = rng.standard_normal((n, 3))
    return Dataset("toy_intake", X, y=event, covariates=C, task="classification", outer="stratified_kfold",
                   standardize=False, metric="auc", standardize_scores=True, weights=rng.uniform(0.5, 2.0, n),
                   time=time, event=event, meta={"energy_index": 0, "features": [f"f{i}" for i in range(p)]})


def test_harrell_c_matches_brute_force_and_extremes():
    rng = np.random.default_rng(1)
    t, e, r = rng.exponential(10, 120), rng.random(120) < 0.6, rng.standard_normal(120)
    num = den = 0.0
    for i in range(120):
        for j in range(120):
            if e[i] and t[i] < t[j]:
                den += 1
                num += (r[i] > r[j]) + 0.5 * (r[i] == r[j])
    assert harrell_c(t, e, r) == pytest.approx(num / den)
    assert harrell_c(t, e, -t) == pytest.approx(1.0)                 # earlier time, higher risk
    assert harrell_c(t, e, t) == pytest.approx(0.0)


def test_energy_adjustments_and_subsample():
    ds = _toy_intake()
    d = ds.derive("density")
    e = np.clip(ds.X[:, 0], 1, None)
    assert d.X.shape[1] == ds.p - 1 and np.allclose(d.X * e[:, None] / 1000.0, ds.X[:, 1:])
    r = ds.derive("residual")
    assert abs(np.corrcoef(r.X[:, 0], np.log(e))[0, 1]) < 1e-8      # energy regressed out
    assert d.meta["features"] == [f"f{i}" for i in range(1, ds.p)] and d.name == "toy_intake@density"
    s = ds.subsample(200, seed=3)
    assert s.n == 200 and s.weights.shape == (200,) and s.time.shape == (200,) and s.covariates.shape[0] == 200
    assert abs(s.y.mean() - ds.y.mean()) < 0.03                      # stratified: event rate preserved


def test_auc_metric_and_standardised_scores():
    from sklearn.metrics import roc_auc_score
    ds = _toy_intake()
    tr, te = np.arange(0, 450), np.arange(450, 600)
    rep_tr, rep_te = ds.prep(tr, te)
    V = np.eye(ds.p)
    sc = nested_cv.score_models(ds, V, "c", rep_tr, rep_te, tr, te, 0, 30)
    assert nested_cv.metric_name(ds) == "auc" and all(0.0 <= v <= 1.0 for v in sc.values())
    assert sc["linear"] > 0.55                                       # the toy signal is learnable


def test_controls_ceiling_identity_and_shuffled_support():
    ds = _toy_intake()
    rep, _ = ds.prep(np.arange(400), np.arange(400, 600))
    V0, _ = METHODS["Ceiling (all features)"].fit(rep, {"k": 0}, 0, {})
    assert np.array_equal(V0, np.eye(ds.p))
    m, c = METHODS["NSA nonneg"], METHODS["NSA nonneg (shuffled support)"]
    hp = {"w": 0.5, "k": 3}
    V, _ = m.fit(rep, hp, 0, {})
    Vs, _ = c.fit(rep, hp, 0, {})
    assert np.allclose(np.sort(Vs.ravel()), np.sort(V.ravel()))      # value multiset preserved
    assert not np.allclose(Vs, V)                                    # feature assignment destroyed


def test_survival_protocol_rows():
    ds = _toy_intake()
    df = PROTOCOLS["survival"](ds, ["PCA", "NSA nonneg", "Ceiling (all features)"], get_profile("smoke"), n_outer=2)
    assert set(df.metric) == {"cindex", "auc"} and set(df.protocol) == {"survival"} and df.value.notna().all()
    assert df[df.metric == "cindex"].value.between(0.3, 1.0).all()
    with pytest.raises(ValueError):
        PROTOCOLS["survival"](Dataset("x", ds.X, y=ds.y, task="classification"), ["PCA"], get_profile("smoke"))


def test_nhanes_loader_when_available():
    try:
        ds = get_dataset("nhanes")
    except DatasetUnavailable:
        pytest.skip("NHANES cleaned release not available")
    assert ds.p == 77 and ds.metric == "auc" and ds.weights is not None and ds.time is not None
    assert not any(f in ds.meta["features"] for f in ("RIDAGEYR", "RIAGENDR", "survey_day"))
    assert ds.X.min() >= 0 and 0.1 < ds.y.mean() < 0.2
