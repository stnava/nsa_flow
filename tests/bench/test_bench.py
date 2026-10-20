"""nsa_flow.bench: registry, datasets, protocols (leakage, determinism, schema), suites, reports, CLI, API."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from nsa_flow.bench import cli, protocols, suites  # noqa: F401,E402
from nsa_flow.bench.datasets import Dataset, DatasetUnavailable, get_dataset  # noqa: E402
from nsa_flow.bench.registry import DATASETS, METHODS, PROTOCOLS, SUITES  # noqa: E402
from nsa_flow.bench.results import COLUMNS  # noqa: E402
from nsa_flow.bench.runner import run_suite  # noqa: E402
from nsa_flow.bench.runtime import get_profile  # noqa: E402

SMOKE = get_profile("smoke")


# ------------------------------------------------------------------ registry and datasets
def test_registries_populated():
    assert {"planted", "noise", "adni", "golub", "tecator", "newsgroups", "metabric", "pbmc3k"} <= set(DATASETS)
    assert {"PCA", "NSA signed", "NSA consolidated", "NSA nonneg", "KMeans features", "NMF"} <= set(METHODS)
    assert {"nested_cv", "module_recovery", "speed"} <= set(PROTOCOLS)
    assert {"smoke", "standard", "paper"} <= set(SUITES)


def test_unknown_names_say_what_is_known():
    with pytest.raises(KeyError, match="known"):
        DATASETS.get_or_raise("nope")


def test_null_derivations():
    ds = get_dataset("planted", noise=0.3)
    y0 = get_dataset("planted@null_y", noise=0.3)
    assert sorted(y0.y) == pytest.approx(sorted(ds.y)) and not np.allclose(y0.y, ds.y)
    sh = get_dataset("planted@shuffled", noise=0.3)
    assert np.allclose(np.sort(sh.X, 0), np.sort(ds.X, 0))                       # marginals kept
    off = lambda A: np.abs(np.corrcoef(A.T) - np.eye(A.shape[1])).mean()
    assert off(sh.X) < 0.5 * off(ds.X)                                            # correlations destroyed


def test_prep_fits_on_training_rows_only():
    ds = get_dataset("planted")
    tr, te = np.arange(0, 200), np.arange(200, 300)
    rtr, rte = ds.prep(tr, te)
    assert np.allclose(rtr["c"].mean(0), 0, atol=1e-8)
    assert not np.allclose(rte["c"].mean(0), 0, atol=1e-3)                        # test not re-centred
    assert rtr["pos"].min() >= 0 and rte["pos"].min() >= 0


def test_unavailable_dataset_is_skipped_with_reason(monkeypatch):
    from nsa_flow.bench.datasets.base import DatasetSpec
    from nsa_flow.bench.suites import Entry, Suite
    def boom():
        raise DatasetUnavailable("not here: searched [/nowhere]")
    monkeypatch.setitem(DATASETS, "ghost", DatasetSpec("ghost", boom))
    suite = Suite("t", "t", [Entry("ghost", "nested_cv", ["PCA"])])
    rows, _, _, _ = run_suite(suite, SMOKE, save=False, log=lambda *_: None)
    assert rows.skipped_reason.iloc[0].startswith("not here")


# ------------------------------------------------------------------ protocols
def _toy_regression(seed=0, n=90, p=12):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    return Dataset("toy", X, y=X[:, :3].sum(1) + 0.1 * rng.standard_normal(n), task="regression")


def test_selection_never_touches_outer_test(monkeypatch):
    from nsa_flow.bench.protocols import nested_cv
    ds = _toy_regression()
    seen = set()
    orig = Dataset.prep

    def spy(self, tr, te):
        seen.update(np.asarray(tr).tolist()); seen.update(np.asarray(te).tolist())
        return orig(self, tr, te)
    monkeypatch.setattr(Dataset, "prep", spy)
    tr = np.arange(0, 60)
    nested_cv.select_hp(ds, METHODS["PCA"], METHODS["PCA"].grid(SMOKE), tr, 1, SMOKE)
    assert seen <= set(tr.tolist())


def test_nested_cv_rows_schema_and_determinism():
    ds = get_dataset("planted")
    a = PROTOCOLS["nested_cv"](ds, ["PCA", "NSA consolidated"], SMOKE)
    b = PROTOCOLS["nested_cv"](ds, ["PCA", "NSA consolidated"], SMOKE)
    assert {"dataset", "protocol", "split", "method", "variant", "model", "metric", "value"} <= set(a.columns)
    assert set(a.columns) <= set(COLUMNS)
    assert set(a.model) == {"linear", "forest", ""} and set(a.variant) == {"tuned", "fixed", "control"}   # "" = collinearity rows
    pd.testing.assert_frame_equal(a.drop(columns="seconds"), b.drop(columns="seconds"))
    assert a[a.method.str.startswith("NSA")].cert_converged.notna().all()           # certificate recorded


def test_module_recovery_planted_and_noise():
    rows = PROTOCOLS["module_recovery"](get_dataset("planted"), ["NSA consolidated", "KMeans features"],
                                        get_profile("quick"))
    ari = rows[rows.metric == "ari"].groupby("method").value.mean()
    assert ari["NSA consolidated w=0.5"] > 0.8
    noise = PROTOCOLS["module_recovery"](get_dataset("noise"), ["NSA consolidated"], get_profile("quick"))
    assert abs(noise[noise.metric == "stability"].value.mean()) < 0.1


def test_expensive_method_skipped_visibly():
    big = Dataset("big", np.random.default_rng(0).standard_normal((60, 600)), y=np.arange(60.0), task="regression")
    df = PROTOCOLS["nested_cv"](big, ["PCA", "SparsePCA"], SMOKE)
    assert "SparsePCA skipped" in df.skipped_reason.dropna().iloc[0]
    assert "SparsePCA" not in set(df.method.dropna()[df.value.notna()])


# ------------------------------------------------------------------ suites, reports, cli, api
@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    rows, summary, verdicts, run_dir = run_suite(SUITES["smoke"], "smoke", out=out, log=lambda *_: None)
    return rows, summary, verdicts, run_dir, out


def test_smoke_suite_end_to_end(smoke_run):
    rows, summary, verdicts, run_dir, _ = smoke_run
    assert (run_dir / "rows.csv").exists() and (run_dir / "manifest.json").exists()
    man = json.loads((run_dir / "manifest.json").read_text())
    assert man["suite"]["name"] == "smoke" and man["versions"]["nsa_flow"] and "entries" in man
    assert [v for v in verdicts if v["expect"] == "null"][0]["status"] == "PASS"   # no structure invented in noise
    assert set(rows[rows.protocol == "speed"].metric) == {"seconds"}


def test_report_and_marker_injection(smoke_run, tmp_path):
    from nsa_flow.bench.report import inject, render_markdown
    *_, run_dir, out = smoke_run
    md = render_markdown(run_dir)
    assert "Pre-registered primary contrasts" in md and "Solver health" in md
    f = tmp_path / "R.md"
    f.write_text("a\n<!-- bench:t:start -->\nold\n<!-- bench:t:end -->\nb\n")
    inject(f, "t", "NEW")
    assert "NEW" in f.read_text() and "old" not in f.read_text()
    with pytest.raises(ValueError):
        inject(f, "missing", "x")


def test_budget_skips_visibly():
    suite = SUITES["smoke"]
    rows, _, verdicts, _ = run_suite(suite, "quick", save=False, budget_min=-1.0, only=["planted"],
                                     log=lambda *_: None)
    assert (rows.method == "*").all() and rows.skipped_reason.str.contains("budget").all()
    assert all(v["status"] == "SKIPPED" for v in verdicts)


def test_cli_list_and_fetch_gate(capsys, monkeypatch):
    assert cli.main(["list"]) == 0
    assert "datasets:" in capsys.readouterr().out
    called = []
    monkeypatch.setattr(DATASETS["pbmc3k"], "fetch", lambda: called.append(1))
    assert cli.main(["fetch", "pbmc3k"]) == 1 and not called                        # no --download, no download
    assert cli.main(["fetch", "pbmc3k", "--download"]) == 0 and called


def test_check_passes_and_catches_a_perturbed_solver(monkeypatch):
    assert cli.main(["check"]) == 0
    m = METHODS["NSA consolidated"]
    orig = m.fit
    def perturbed(rep, hp, seed, ctx):
        V, cert = orig(rep, hp, seed, ctx)
        return V[np.random.default_rng(0).permutation(V.shape[0])], cert            # scramble feature identity
    monkeypatch.setattr(m, "fit", perturbed)
    assert cli.main(["check"]) == 1


def test_evaluate_standalone_api():
    from nsa_flow.bench import evaluate
    from nsa_flow.bench.datasets.synthetic import planted_partition
    X, V, Z = planted_partition(p=40, k=4, n=200, noise=1.0, seed=3)
    y = Z[:, 0] + 0.3 * np.random.default_rng(0).standard_normal(200)
    ev = evaluate(X, y=y, groups=np.repeat(np.arange(4), 10), methods=["PCA", "NSA consolidated", "KMeans features"],
                  profile="smoke")
    assert {"nested_cv", "module_recovery"} == set(ev.rows.protocol.dropna())
    assert "NSA consolidated" in ev.report()
    with pytest.raises(ValueError):
        evaluate(X)


def test_import_nsa_flow_stays_light():
    code = "import nsa_flow, sys; assert 'sklearn' not in sys.modules and 'pandas' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0
    code = "import nsa_flow.bench, sys; assert 'sklearn' not in sys.modules and 'pandas' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0
