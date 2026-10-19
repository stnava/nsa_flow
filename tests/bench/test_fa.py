"""Factor-analysis benchmark: own PAF/varimax/CFA validated against R, metrics, protocol, guards, nulls."""
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from nsa_flow.bench import protocols, suites  # noqa: F401,E402
from nsa_flow.bench.datasets import DatasetUnavailable, get_dataset  # noqa: E402
from nsa_flow.bench.datasets.psychometric import simulate_factor_model  # noqa: E402
from nsa_flow.bench.fa.cfa import fit_cfa  # noqa: E402
from nsa_flow.bench.fa.nsa_fa import nsa_fa, nsa_fa_post  # noqa: E402
from nsa_flow.bench.fa.paf import corr, paf, paf_varimax, varimax  # noqa: E402
from nsa_flow.bench.metrics import (cross_loadings, hofmann_complexity, orthogonality_defect,  # noqa: E402
                                    reproduced_correlation_error, tucker_congruence)
from nsa_flow.bench.registry import PROTOCOLS  # noqa: E402
from nsa_flow.bench.runtime import get_profile  # noqa: E402


def _r_has(*pkgs):
    if shutil.which("Rscript") is None:
        return False
    code = "cat(all(sapply(c(%s), requireNamespace, quietly=TRUE)))" % ",".join(f'"{p}"' for p in pkgs)
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True)
    return out.stdout.strip().endswith("TRUE")


# ------------------------------------------------------------------ metrics
def test_hofmann_cross_loadings_defect_tucker():
    simple = np.array([[0.8, 0.0], [0.7, 0.0], [0.0, 0.9], [0.0, 0.6]])
    flat = np.full((4, 2), 0.5)
    assert hofmann_complexity(simple) == pytest.approx(1.0)
    assert hofmann_complexity(flat) == pytest.approx(2.0)
    assert cross_loadings(simple) == 0 and cross_loadings(flat) == 4
    assert orthogonality_defect(np.array([[1.0, 0], [0, 1.0]])) == pytest.approx(0.0)
    assert orthogonality_defect(np.ones((3, 2))) > 0.9
    t, _ = tucker_congruence(simple, simple[:, ::-1] * np.array([1, -1]))     # permuted and sign-flipped
    assert t == pytest.approx(1.0)


def test_varimax_is_orthogonal_and_preserves_communality():
    L = paf(corr(simulate_factor_model(500, seed=1)[0]), 5).loadings
    Lr, T = varimax(L)
    assert np.allclose(T.T @ T, np.eye(5), atol=1e-8)
    assert np.allclose((L ** 2).sum(1), (Lr ** 2).sum(1), atol=1e-8)


# ------------------------------------------------------------------ recovery and the NSA.FA pipeline
def test_paf_recovers_simulated_loadings():
    X, L, _ = simulate_factor_model(3000, seed=2)
    Lh = paf_varimax(corr(X), 5).loadings
    assert tucker_congruence(Lh, L)[0] > 0.95


def test_nsa_fa_runs_certifies_and_w0_matches_paf_varimax():
    X, L, _ = simulate_factor_model(1000, seed=3)
    R = corr(X)
    r = nsa_fa(R, 5, w=0.5)
    assert r.converged and r.cert["converged"] and np.isfinite(r.loadings).all()
    base = paf_varimax(R, 5).loadings
    r0 = nsa_fa(R, 5, w=0.0)
    assert tucker_congruence(r0.loadings, base)[0] > 0.995                    # no regularisation = PAF + varimax
    assert nsa_fa_post(R, 5, w=0.5).loadings.shape == (25, 5)
    assert orthogonality_defect(r.loadings) < orthogonality_defect(base)       # the intended effect of w > 0


# ------------------------------------------------------------------ validation against R
@pytest.mark.skipif(not _r_has("psych", "lavaan"), reason="needs Rscript with psych and lavaan")
def test_paf_varimax_and_cfa_match_r(tmp_path):
    X, _, groups = simulate_factor_model(600, seed=4)
    p = X.shape[1]
    names = [f"i{j}" for j in range(p)]
    pd.DataFrame(X, columns=names).to_csv(tmp_path / "d.csv", index=False)
    model = "\\n".join(f"f{g} =~ " + " + ".join(names[i] for i in range(p) if groups[i] == g) for g in range(5))
    script = f'''
suppressMessages({{library(psych); library(lavaan)}})
d <- read.csv("{tmp_path}/d.csv"); R <- cor(d)
f <- fa(R, nfactors=5, fm="pa", rotate="none", max.iter=50, min.err=1e-3)
write.csv(unclass(f$loadings), "{tmp_path}/paf.csv", row.names=FALSE)
write.csv(unclass(varimax(unclass(f$loadings))$loadings), "{tmp_path}/vm.csv", row.names=FALSE)
fit <- cfa("{model}", sample.cov=cov(d), sample.nobs=nrow(d), std.lv=TRUE)
write.csv(as.data.frame(as.list(fitMeasures(fit, c("chisq","df","rmsea","cfi","tli","srmr","bic")))),
          "{tmp_path}/cfa.csv", row.names=FALSE)
'''
    subprocess.run(["Rscript", "-e", script], check=True, capture_output=True)
    ref = pd.read_csv(tmp_path / "paf.csv").to_numpy()
    L = paf(corr(X), 5).loadings
    assert tucker_congruence(L, ref)[0] > 0.9999
    assert np.abs(varimax(ref)[0] - pd.read_csv(tmp_path / "vm.csv").to_numpy()).max() < 1e-6
    fit = fit_cfa(np.cov(X.T), len(X), groups)
    rc = pd.read_csv(tmp_path / "cfa.csv").iloc[0]
    for a in ("chisq", "rmsea", "cfi", "tli", "srmr"):
        assert getattr(fit, a) == pytest.approx(rc[a], abs=1e-3), a
    assert fit.df == rc["df"] and fit.bic == pytest.approx(rc["bic"], abs=0.1)


def test_holzinger_swineford_cfa_golden_values():
    try:
        ds = get_dataset("holzinger")
    except DatasetUnavailable:
        pytest.skip("holzinger not exported")
    fit = fit_cfa(np.cov(ds.X.T), ds.n, ds.groups)
    assert fit.chisq == pytest.approx(85.306, abs=0.01) and fit.df == 24                # published lavaan values
    assert fit.cfi == pytest.approx(0.931, abs=0.001) and fit.rmsea == pytest.approx(0.092, abs=0.001)


# ------------------------------------------------------------------ protocol
def test_fa_protocol_schema_no_leakage_and_nulls(monkeypatch):
    from nsa_flow.bench.fa import methods as fam
    ds = get_dataset("fa_sim_n300")
    seen = []
    m = fam.FA_METHODS["PAF varimax"]
    orig = m.fit
    monkeypatch.setattr(m, "fit", lambda X, k, hp, seed: (seen.append(X.copy()), orig(X, k, hp, seed))[1])
    P = get_profile("smoke")
    df = PROTOCOLS["factor_analysis"](ds, ["PAF varimax", "NSA.FA"], P, n_outer=2)
    assert {"hofmann", "ari", "tucker", "recon_err", "score_validity", "stability_ari"} <= set(df.metric)
    assert "cfa_rmsea" not in set(df.metric)               # test split (60) < 5 respondents per item (25): CFA skipped
    big = PROTOCOLS["factor_analysis"](get_dataset("fa_sim_n1000"), ["PAF varimax"], P, n_outer=1)
    assert {"cfa_rmsea", "cfa_cfi", "cfa_tli", "cfa_srmr", "cfa_bic"} <= set(big.metric)
    assert set(df.protocol) == {"factor_analysis"} and df.value.notna().all()
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(ds.n), test_size=0.2, random_state=0)
    train_rows = {tuple(r) for r in ds.X[tr]}
    first = seen[:3]                                     # fit on train, then the two stability halves
    assert all({tuple(r) for r in X} <= train_rows for X in first)
    null = PROTOCOLS["factor_analysis"](get_dataset("fa_null"), ["NSA.FA"], get_profile("quick"), n_outer=4)
    assert abs(null[null.metric == "ari"].value.mean()) < 0.2                            # no structure, no recovery


def test_guard_logic_marks_failed_non_inferiority():
    from nsa_flow.bench.runner import evaluate_verdicts
    from nsa_flow.bench.suites import Entry, Primary, Suite
    prim = Primary("hofmann", "A", "B", direction="lt", guards=(("cfa_rmsea", "lt", 0.01),))

    def row(metric, d, lo, hi):
        return dict(dataset="d", protocol="factor_analysis", metric=metric, model="", method="A", mean=0.0, se=0.0,
                    d_vs_B=d, ci_lo_B=lo, ci_hi_B=hi)
    suite = Suite("t", "t", [Entry("d", "factor_analysis", ["A", "B"], prim, "effect")])
    good = pd.DataFrame([row("hofmann", -0.2, -0.3, -0.1), row("cfa_rmsea", 0.002, -0.001, 0.005)])
    bad = pd.DataFrame([row("hofmann", -0.2, -0.3, -0.1), row("cfa_rmsea", 0.03, 0.02, 0.04)])
    assert evaluate_verdicts(suite, good)[0]["status"] == "PASS"
    v = evaluate_verdicts(suite, bad)[0]
    assert v["status"] == "FAIL (guard)" and not v["guards"][0]["ok"]
    unknown = pd.DataFrame([row("hofmann", -0.2, -0.3, -0.1)])                           # guard metric not evaluable
    v2 = evaluate_verdicts(suite, unknown)[0]
    assert v2["status"] == "PASS" and v2["guards"][0]["ok"] is None


def test_evaluate_fa_standalone_and_suite_registered():
    from nsa_flow.bench import api
    from nsa_flow.bench.registry import SUITES
    X, L, groups = simulate_factor_model(400, seed=5)
    ev = api.evaluate_fa(X, 5, groups=groups, true_loadings=L, methods=["PAF varimax", "NSA.FA"], profile="smoke")
    assert "tucker" in set(ev.rows.metric) and "NSA.FA" in ev.report()
    assert {"fa"} <= set(SUITES) and any(e.protocol == "factor_analysis" for e in SUITES["standard"].entries)
