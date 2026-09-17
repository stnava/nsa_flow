"""Tests for the functions that compute the manuscript's tables.

These are experiment drivers rather than library code, but the paper quotes
their output, so a silent change in shape or sign would put a wrong number in
print.  The real-data functions are skipped when their inputs are absent.
"""
import numpy as np
import pytest
import torch

from experiments import rmd_support as R

F64 = torch.float64


@pytest.fixture(scope="module")
def synthetic():
    """A matrix with a planted non-negative near-disjoint basis."""
    rng = np.random.default_rng(0)
    p, k, n = 40, 5, 150
    V = np.zeros((p, k))
    for i in range(k):
        V[i * 8:(i + 1) * 8, i] = rng.random(8) + 0.5
    U = rng.random((n, k))
    return U @ V.T + 0.05 * rng.standard_normal((n, p))


def test_properties_table_separates_the_three_functionals():
    t = R.properties_table().set_index("case")
    # orthogonal with unequal norms: only D objects
    row = t.loc["orthogonal, unequal norms"]
    assert row.D > 0.5 and row.d_v1 < 1e-20 and row.C < 1e-20
    # a disjoint non-negative basis is the target structure; D still charges it
    row = t.loc["disjoint nonneg supports"]
    assert row.D > 0.05 and row.C < 1e-20
    # rank collapse: only v1's functional rewards it
    row = t.loc["rank collapse (1 live col)"]
    assert row.d_v1 < 1e-20 and row.C > 0.1 and row.D > 0.5
    assert t.loc["all columns collinear"].C == pytest.approx(1.0, abs=1e-9)


def test_abs_trap_orders_the_four_bases_correctly(synthetic):
    t = R.abs_trap_table(synthetic).set_index("basis")
    signed = t.loc["signed PCA (unconstrained optimum)", "err"]
    fitted = t.loc["fitted to the data (w=0)", "err"]
    absolute = t.loc["abs(PCA)", "err"]
    clamped = t.loc["clamp(PCA) -- the true projection", "err"]
    # signed is optimal; fitting is close to it; both maps are worse than fitting
    assert signed <= fitted < clamped
    assert fitted < absolute
    # and abs() is worse than the actual projection, which is the whole point
    assert clamped < absolute
    assert t.loc["signed PCA (unconstrained optimum)", "cost_vs_signed"] == \
        pytest.approx(1.0)


def test_w_dial_is_monotone_in_what_it_controls(synthetic):
    t = R.w_dial_table(synthetic, ws=(0.0, 0.5, 0.9), k=5)
    assert t.C.is_monotonic_decreasing                 # more w, less defect
    assert t.recon_err.is_monotonic_increasing         # paid for in fidelity
    assert t.overlap.iloc[-1] <= t.overlap.iloc[0]
    assert t.converged.all()


def test_homotopy_table_shows_path_beats_iterations(synthetic):
    t = R.homotopy_table(synthetic, k=5, w=0.5)
    by = t.groupby("schedule").energy.min()
    nine = [k for k in by.index if k.startswith("9")][0]
    one = [k for k in by.index if k.startswith("1 ")][0]
    assert by[nine] < by[one]
    # and within a schedule, a 20x iteration budget changes little
    coarse = t[t.schedule == one].sort_values("iters_per_stage").energy.to_numpy()
    assert abs(coarse[1] - coarse[0]) < 0.5 * abs(by[one] - by[nine])


def test_scaling_table_reports_both_routes():
    t = R.scaling_table(n=60, ps=(200, 600), k=4, max_iter=20)
    assert (t.gram_GB > 0).all()
    assert (t.matrix_free_s > 0).all() and (t.gram_s > 0).all()
    assert set(t.p) == {200, 600}


def test_nested_F_recovers_a_known_signal():
    rng = np.random.default_rng(1)
    n = 300
    Z0 = np.column_stack([np.ones(n), rng.standard_normal(n)])
    extra = rng.standard_normal((n, 2))
    y_null = Z0 @ [1.0, 2.0] + rng.standard_normal(n)
    y_sig = y_null + 3.0 * extra[:, 0]
    p_null = R._nested_F(y_null, Z0, np.column_stack([Z0, extra]))
    p_sig = R._nested_F(y_sig, Z0, np.column_stack([Z0, extra]))
    assert p_sig < 1e-10 < p_null


# --------------------------------------------------------- real-data functions
def _have(path):
    try:
        return path.exists()
    except Exception:
        return False


@pytest.mark.skipif(not _have(R.THK), reason="ADNI thickness table not present")
def test_thickness_loader_shape_and_columns():
    X, df, cols = R.load_adni_thickness()
    assert X.ndim == 2 and X.shape[0] > 100 and X.shape[1] > 20
    assert np.isfinite(X).all()
    assert all("T1Hier_thk_" in c and "LRAVG" in c for c in cols)
    assert not any(x in c for c in cols for x in ("Asym", "reference"))
    assert [c for c in R.COG_VARS if c in df.columns]


@pytest.mark.skipif(not _have(R.THK), reason="ADNI thickness table not present")
def test_covariate_design_is_full_rank_on_its_complete_cases():
    """APOE4 is missing for a substantial minority, so the design has NaN rows.

    Every consumer must drop them, which makes the analysis sample smaller than
    the imaging sample; that is worth asserting rather than discovering.
    """
    _, df, _ = R.load_adni_thickness()
    C = R._covar_design(df)
    assert C.shape[1] == 5
    ok = np.isfinite(C).all(1)
    assert 0 < (~ok).sum() < 0.3 * len(C)      # some missing, not most
    assert np.linalg.matrix_rank(C[ok]) == 5


@pytest.mark.skipif(not _have(R.THK), reason="ADNI thickness table not present")
def test_cognitive_analyses_report_the_sample_they_actually_used():
    """The analysis n is smaller than the imaging n; it must not be implied."""
    X, df, _ = R.load_adni_thickness()
    t = R.cognitive_cv(k=3, ws=(0.5,), modes=("subspace",), n_splits=3,
                       n_repeats=1, model="linear")
    assert (t.n < len(X)).all()
    assert (t.n > 0.7 * len(X)).all()


@pytest.mark.skipif(not _have(R.THK), reason="ADNI thickness table not present")
def test_cognitive_cv_is_out_of_sample_and_self_consistent():
    t = R.cognitive_cv(k=3, ws=(0.5,), modes=("subspace",), n_splits=3,
                       n_repeats=1, model="linear")
    assert len(t) > 0
    # the reported gains must equal the differences they are derived from
    assert np.allclose(t.dR2_nsa, t.r2_nsa - t.r2_covariates)
    assert np.allclose(t.nsa_minus_pca, t.r2_nsa - t.r2_pca)
    # out-of-sample R^2 can be negative, but not above 1
    assert (t.r2_nsa <= 1.0).all() and (t.r2_pca <= 1.0).all()
    assert (t.n_folds == 3).all()
