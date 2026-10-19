"""Speed infrastructure: per-entry persistence, resume keyed on inputs and core hash, cheap inner selection, fast suite."""
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from nsa_flow.bench import protocols, runner, suites  # noqa: F401,E402
from nsa_flow.bench.datasets import get_dataset  # noqa: E402
from nsa_flow.bench.protocols import module_recovery, nested_cv  # noqa: E402
from nsa_flow.bench.registry import METHODS, PROTOCOLS, SUITES  # noqa: E402
from nsa_flow.bench.results import core_hash  # noqa: E402
from nsa_flow.bench.runtime import PROFILES, get_profile  # noqa: E402


def _quiet(*_):
    pass


def test_entries_saved_per_entry_with_progress_and_core_hash(tmp_path):
    rows, _, _, d = runner.run_suite(SUITES["smoke"], "smoke", out=tmp_path, log=_quiet)
    n = len(SUITES["smoke"].entries)
    assert len(list((d / "entries").glob("entry_*.csv"))) == n and len(list((d / "entries").glob("entry_*.json"))) == n
    assert len((d / "progress.log").read_text().strip().splitlines()) == n
    man = json.loads((d / "manifest.json").read_text())
    assert len(man["core_hash"]) == 12 and man["tree_dirty"] in (True, False, None)
    assert core_hash()["core_hash"] == man["core_hash"]
    assert not (tmp_path / "smoke" / "running.txt").exists()                  # finished run is not 'running'


def test_resume_reuses_finished_entries_and_recomputes_on_changed_inputs(tmp_path, monkeypatch):
    _, _, _, d = runner.run_suite(SUITES["smoke"], "smoke", out=tmp_path, log=_quiet)
    calls = []
    orig = dict(PROTOCOLS)
    for name, fn in orig.items():
        monkeypatch.setitem(PROTOCOLS, name, lambda *a, _n=name, _f=fn, **k: (calls.append(_n), _f(*a, **k))[1])
    rows, _, _, d2 = runner.run_suite(SUITES["smoke"], "smoke", out=tmp_path, log=_quiet, resume=d)
    assert calls == [] and d2 == d                                              # everything came from disk
    assert len(rows) > 0
    monkeypatch.setattr(runner, "core_hash", lambda: {"core_hash": "changedcore0", "tree_dirty": False})
    runner.run_suite(SUITES["smoke"], "smoke", out=tmp_path, log=_quiet, resume=d)
    assert len(calls) == len(SUITES["smoke"].entries)                           # a changed core is a different experiment
    calls.clear()
    monkeypatch.undo()
    for name, fn in orig.items():
        monkeypatch.setitem(PROTOCOLS, name, lambda *a, _n=name, _f=fn, **k: (calls.append(_n), _f(*a, **k))[1])
    runner.run_suite(SUITES["smoke"], "quick", out=tmp_path, log=_quiet, resume=d, only=["planted"])
    assert calls                                                                # a different profile is recomputed


def test_inner_selection_uses_the_smaller_forest(monkeypatch):
    p = get_profile("fast")
    assert 0 < p.rf_trees_inner < p.rf_trees and all(0 < q.rf_trees_inner <= q.rf_trees for q in PROFILES.values()
                                                     if q.name not in ("standard",))
    assert get_profile("standard").rf_trees_inner in (0, get_profile("standard").rf_trees)   # reference = full forest
    ds = get_dataset("planted")
    seen = []
    orig = nested_cv.score_models
    monkeypatch.setattr(nested_cv, "score_models",
                        lambda *a, **k: (seen.append(a[-1]), orig(*a, **k))[1])
    tr = np.arange(0, 200)
    nested_cv.select_hp(ds, METHODS["PCA"], METHODS["PCA"].grid(get_profile("smoke")), tr, 1, p)
    assert seen and set(seen) == {p.rf_trees_inner}


def test_primary_only_module_variants():
    ds = get_dataset("planted")
    P = get_profile("smoke")
    full = PROTOCOLS["module_recovery"](ds, ["NSA consolidated"], P)
    one = PROTOCOLS["module_recovery"](ds, ["NSA consolidated"], P, primary_only=True)
    assert set(full.method) == {"NSA consolidated w=0.5", "NSA consolidated w=0.25"}
    assert set(one.method) == {"NSA consolidated w=0.5"}


def test_standard_fast_is_a_cheaper_subset_with_the_same_primaries():
    fast, full = SUITES["standard-fast"], SUITES["standard-full"]
    assert full is SUITES["standard"] and len(fast.entries) < len(full.entries)
    full_primary = {(e.dataset, e.protocol): e.primary for e in full.entries}
    for e in fast.entries:
        if e.primary is not None:
            assert (e.dataset, e.protocol) in full_primary
            fp = full_primary[(e.dataset, e.protocol)]
            assert (e.primary.metric, e.primary.method, e.primary.ref) == (fp.metric, fp.method, fp.ref)
    excluded = {"golub", "tecator", "newsgroups", "metabric@shuffled", "speed"}
    assert not excluded & {e.dataset for e in fast.entries}
    assert all(e.params.get("primary_only") for e in fast.entries if e.dataset in ("metabric", "pbmc3k"))
