"""Redesign after the diagnostic study: oblique baselines, NSA as a fit-preserving rotation, multistart nonneg NSA,
pair-accuracy ceilings, and the axis-control for prediction."""
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from nsa_flow.bench import methods as bench_methods, protocols, suites  # noqa: F401,E402
from nsa_flow.bench.datasets import get_dataset  # noqa: E402
from nsa_flow.bench.datasets.psychometric import simulate_factor_model  # noqa: E402
from nsa_flow.bench.fa.nsa_fa import nsa_rotation  # noqa: E402
from nsa_flow.bench.fa.paf import corr, oblimin, paf, paf_varimax, promax  # noqa: E402
from nsa_flow.bench.metrics import hofmann_complexity, reproduced_correlation_error, tucker_congruence  # noqa: E402
from nsa_flow.bench.registry import METHODS, PROTOCOLS, SUITES  # noqa: E402
from nsa_flow.bench.runtime import get_profile  # noqa: E402


def _r_has(*pkgs):
    if shutil.which("Rscript") is None:
        return False
    code = "cat(all(sapply(c(%s), requireNamespace, quietly=TRUE)))" % ",".join(f'"{p}"' for p in pkgs)
    return subprocess.run(["Rscript", "-e", code], capture_output=True, text=True).stdout.strip().endswith("TRUE")


# ------------------------------------------------------------------ oblique baselines
@pytest.mark.skipif(not _r_has("psych", "GPArotation"), reason="needs Rscript with psych and GPArotation")
def test_promax_and_oblimin_match_r(tmp_path):
    X, _, _ = simulate_factor_model(800, seed=7, phi=0.3)
    L0 = paf(corr(X), 5).loadings
    pd.DataFrame(L0).to_csv(tmp_path / "L.csv", index=False)
    script = f'''
suppressMessages(library(GPArotation)); L <- as.matrix(read.csv("{tmp_path}/L.csv"))
p <- promax(L); write.csv(unclass(p$loadings), "{tmp_path}/pm.csv", row.names=FALSE)
ui <- solve(p$rotmat); write.csv(ui %*% t(ui), "{tmp_path}/pm_phi.csv", row.names=FALSE)
o <- GPArotation::oblimin(L, gam=0, normalize=FALSE, eps=1e-5)
write.csv(unclass(o$loadings), "{tmp_path}/ob.csv", row.names=FALSE); write.csv(o$Phi, "{tmp_path}/ob_phi.csv", row.names=FALSE)
'''
    subprocess.run(["Rscript", "-e", script], check=True, capture_output=True)
    Lp, Pp = promax(L0)
    assert np.abs(Lp - pd.read_csv(tmp_path / "pm.csv").to_numpy()).max() < 1e-6
    assert np.abs(Pp - pd.read_csv(tmp_path / "pm_phi.csv").to_numpy()).max() < 1e-6
    Lo, Po = oblimin(L0)
    assert tucker_congruence(Lo, pd.read_csv(tmp_path / "ob.csv").to_numpy())[0] > 0.9999
    assert np.abs(np.sort(np.abs(Po[np.triu_indices(5, 1)])) - np.sort(np.abs(pd.read_csv(tmp_path / "ob_phi.csv").to_numpy()[np.triu_indices(5, 1)]))).max() < 1e-3


def test_oblique_rotations_preserve_paf_fit_and_simplify():
    X, L, _ = simulate_factor_model(600, seed=8, phi=0.4, cross=0.3, p_cross=0.5)
    R = corr(X)
    base = paf(R, 5)
    e0 = reproduced_correlation_error(R, base.loadings)
    for name in ("PAF promax", "PAF oblimin"):
        r = METHODS_FA()[name].fit(X, 5, {}, 0)
        assert r.phi is not None and abs(reproduced_correlation_error(R, r.loadings, r.phi) - e0) < 1e-6
    vm = paf_varimax(R, 5).loadings
    assert hofmann_complexity(METHODS_FA()["PAF promax"].fit(X, 5, {}, 0).loadings).mean() < hofmann_complexity(vm).mean()
    assert tucker_congruence(METHODS_FA()["PAF promax"].fit(X, 5, {}, 0).loadings, L)[0] > tucker_congruence(vm, L)[0]


def METHODS_FA():
    from nsa_flow.bench.registry import FA_METHODS
    from nsa_flow.bench.fa import methods  # noqa: F401
    return FA_METHODS


# ------------------------------------------------------------------ NSA rotation
def test_nsa_rotation_preserves_fit_and_keyed_variant_is_nonnegative():
    X, L, _ = simulate_factor_model(500, seed=9)
    R = corr(X)
    base = paf(R, 5)
    rot = nsa_rotation(R, 5, w=0.5)
    assert rot.cert["converged"] and rot.phi.shape == (5, 5) and np.allclose(np.diag(rot.phi), 1.0)
    assert abs(reproduced_correlation_error(R, rot.loadings, rot.phi) - reproduced_correlation_error(R, base.loadings)) < 1e-6
    assert hofmann_complexity(rot.loadings).mean() <= hofmann_complexity(paf_varimax(R, 5).loadings).mean() + 1e-6
    keys = np.ones(X.shape[1])
    keys[[2, 7]] = -1.0
    Xk = X * keys                                                            # reverse-keyed items
    kr = nsa_rotation(corr(Xk), 5, w=0.5, keys=keys, nonneg=True)
    assert (np.diag(keys) @ kr.loadings).min() > -1e-6                       # loadings are non-negative in keyed space
    with pytest.raises(ValueError):
        nsa_rotation(R, 5, nonneg=True)


# ------------------------------------------------------------------ module recovery
def test_multistart_keeps_the_lowest_energy_and_respects_starts(monkeypatch):
    import torch
    calls = []

    class Fake(dict):
        pass

    def fake_result(rep, hp, mode, consolidate=False, init=None, pos_key="pos"):
        energies = {"varimax": 3.0, "pca_clamp": 1.0, "nmf": 2.0, "random": 0.5}
        calls.append(init)
        r = Fake(energy=energies[init], converged=True, defect_D=0.0)
        r.Y = torch.full((4, 2), energies[init])
        return r
    monkeypatch.setattr(bench_methods, "_nsa_result", fake_result)
    V, _ = bench_methods._nsa_multistart({}, {"k": 2, "w": 0.5}, 0, {})
    assert calls == list(bench_methods.MULTISTART_INITS) and np.allclose(V, 0.5)              # lowest energy wins
    calls.clear()
    V, _ = bench_methods._nsa_multistart({}, {"k": 2, "w": 0.5, "starts": 2}, 0, {})
    assert calls == ["varimax", "pca_clamp"] and np.allclose(V, 1.0)


def test_kmeans_restart_budgets(monkeypatch):
    import sklearn.cluster as skc
    seen = []
    real = skc.KMeans

    monkeypatch.setattr(skc, "KMeans", lambda *a, **k: (seen.append(k.get("n_init")), real(*a, **k))[1])
    ds = get_dataset("planted")
    rep, _ = ds.prep(np.arange(200), np.arange(200, 300))
    METHODS["KMeans features"].fit(rep, {"k": 5}, 0, {})
    METHODS["KMeans features (n_init=30)"].fit(rep, {"k": 5}, 0, {})
    assert seen == [4, 30]


def test_module_protocol_pair_accuracy_grid_and_ceiling():
    ds = get_dataset("planted")
    P = get_profile("smoke")
    df = PROTOCOLS["module_recovery"](ds, ["NSA nonneg (multistart)", "KMeans features"], P, n_outer=2, k_factors=(1, 2, 4))
    assert {"pba_x1", "pba_x2", "pba_x4", "recall2k", "separation2k", "ari", "stability"} <= set(df.metric)
    ceil = df[df.method.str.startswith("Ceiling")]
    assert set(ceil.metric) == {"ari", "pba_x1"} and ceil[ceil.metric == "ari"].value.mean() > 0.5
    one = PROTOCOLS["module_recovery"](ds, ["KMeans features"], P, n_outer=1, k_factors=(1,))
    assert "pba_x1" in set(one.metric) and "pba_x2" not in set(one.metric)


def test_standardise_then_shift_representation():
    ds = get_dataset("planted")
    tr, te = np.arange(200), np.arange(200, 300)
    rtr, rte = ds.prep(tr, te)
    assert rtr["cpos"].min() >= 0 and rte["cpos"].min() >= 0
    assert np.allclose(rtr["cpos"], np.clip(rtr["c"] - rtr["c"].min(0), 0, None))


# ------------------------------------------------------------------ prediction axis control
def test_rotated_score_control_and_collinearity_rows():
    ds = get_dataset("planted")
    df = PROTOCOLS["nested_cv"](ds, ["PCA", "NSA consolidated"], get_profile("smoke"), n_outer=2)
    lin = df[(df.model == "linear") & (df.metric == "r2")]
    a = lin[lin.method == "NSA consolidated"].sort_values("split").value.to_numpy()
    b = lin[lin.method == "NSA consolidated (scores PCA-rotated)"].sort_values("split").value.to_numpy()
    assert np.allclose(a, b, atol=1e-6)                                      # a linear model is rotation invariant
    col = df[df.metric == "score_collinearity"]
    assert set(col.method) == {"PCA", "NSA consolidated"} and col[col.method == "PCA"].value.max() < 1e-6
    assert (col.value >= 0).all() and col.cert_converged.notna().sum() >= 1
    assert not any("rotated" in m for m in df.method if m.startswith("PCA"))   # control only for NSA methods


# ------------------------------------------------------------------ suites
def test_suites_carry_the_corrected_primaries():
    full = SUITES["standard-full"]
    mods = [e for e in full.entries if e.protocol == "module_recovery" and e.dataset in ("metabric", "pbmc3k", "newsgroups")]
    assert mods and all(e.primary.method == "NSA nonneg (multistart) w=0.5" and e.primary.ref == "KMeans features" for e in mods)
    fas = [e for e in full.entries if e.protocol == "factor_analysis" and e.expect == "effect"]
    assert fas and all(e.primary.method == "NSA rotation" and e.primary.ref == "PAF promax" and
                       e.primary.guards[0][:2] == ("recon_err", "lt") for e in fas)
    assert all("NSA nonneg (multistart)" in e.methods and "KMeans features (n_init=30)" in e.methods for e in mods)


def test_cost_estimate_runs_for_every_default_fa_method():
    """Regression: the estimator once called FA methods without item keys and crashed on the keyed variant."""
    from nsa_flow.bench.runner import estimate_minutes
    from nsa_flow.bench.suites import Entry, Primary
    from nsa_flow.bench.fa.methods import DEFAULT_FA
    e = Entry("fa_sim_n100", "factor_analysis", DEFAULT_FA, Primary("hofmann", "NSA rotation", "PAF promax", direction="lt"))
    assert estimate_minutes(e, get_dataset("fa_sim_n100"), get_profile("smoke"), 1) >= 0.0
