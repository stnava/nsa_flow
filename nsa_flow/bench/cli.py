"""``nsa-flow-bench``: list, fetch, run, report, check."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASELINE = Path(__file__).with_name("baseline_smoke.json")


def _imports():
    from . import datasets, methods, protocols, suites  # noqa: F401  (import registers everything)


def cmd_list(a):
    _imports()
    from .datasets import availability
    from .registry import DATASETS, METHODS, PROTOCOLS, SUITES
    print("datasets:")
    for n, s in sorted(DATASETS.items()):
        ok, why = availability(n)
        print(f"  {n:12s} {'ok      ' if ok else 'MISSING '}[{s.tier}] {s.description}"
              + ("" if ok else f"\n               -> {why}"))
    print("methods:   " + ", ".join(sorted(METHODS)))
    print("protocols: " + ", ".join(sorted(PROTOCOLS)))
    print("suites:    " + ", ".join(sorted(SUITES)))


def cmd_fetch(a):
    _imports()
    from .registry import DATASETS
    spec = DATASETS.get_or_raise(a.dataset)
    if spec.fetch is None:
        print(f"{a.dataset}: nothing to fetch ({spec.source})")
        return 0
    if not a.download:
        print(f"{a.dataset}: would download from {spec.source}; re-run with --download to proceed")
        return 1
    spec.fetch()
    print(f"{a.dataset}: fetched")
    return 0


def cmd_run(a):
    _imports()
    from .registry import SUITES
    from .runner import run_suite
    suite = SUITES.get_or_raise(a.suite)
    rows, summary, verdicts, run_dir = run_suite(
        suite, a.profile, a.jobs, a.out, only=a.only, budget_min=a.budget_min, public_only=a.public_only,
        command=" ".join(["nsa-flow-bench"] + sys.argv[1:]))
    from .report import verdict_table
    print("\n" + verdict_table(verdicts))
    return 0


def cmd_report(a):
    from .report import inject, render_markdown
    md = render_markdown(a.run, a.out, brief=a.brief)
    if a.inject:
        path, name = a.inject.split(":")
        inject(path, name, md)
        print(f"injected into {path} (bench:{name})")
    else:
        print(md)
    return 0


def cmd_check(a):
    """Run the smoke suite and compare the primary estimates to the stored baseline."""
    _imports()
    from .registry import SUITES
    from .runner import run_suite
    rows, summary, verdicts, _ = run_suite(SUITES["smoke"], "smoke", 1, "bench_results", save=False, log=lambda *_: None)
    cur = {f"{v['dataset']}|{v['protocol']}|{v['metric']}|{v['method']}": round(v["estimate"], 6) for v in verdicts}
    if a.update:
        BASELINE.write_text(json.dumps(cur, indent=2))
        print(f"baseline written: {BASELINE}")
        return 0
    base = json.loads(BASELINE.read_text())
    bad = {k: (base.get(k), cur.get(k)) for k in set(base) | set(cur)
           if base.get(k) is None or cur.get(k) is None or abs(base[k] - cur[k]) > a.tol}
    if bad:
        print("CHECK FAILED (baseline vs current):")
        for k, (b, c) in sorted(bad.items()):
            print(f"  {k}: {b} -> {c}")
        return 1
    print(f"check ok ({len(cur)} primary estimates within {a.tol})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nsa-flow-bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("fetch"); p.add_argument("dataset"); p.add_argument("--download", action="store_true")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("run")
    p.add_argument("--suite", default="standard"); p.add_argument("--profile", default="standard")
    p.add_argument("--jobs", type=int, default=1); p.add_argument("--out", default="bench_results")
    p.add_argument("--only", nargs="*"); p.add_argument("--budget-min", type=float, default=None)
    p.add_argument("--public-only", action="store_true", help="skip restricted/local datasets")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("report"); p.add_argument("run", help="run directory or suite name (latest)")
    p.add_argument("--out", default="bench_results"); p.add_argument("--inject", help="FILE:NAME marker block")
    p.add_argument("--brief", action="store_true", help="verdicts + solver health only")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("check"); p.add_argument("--update", action="store_true")
    p.add_argument("--tol", type=float, default=1e-3); p.set_defaults(fn=cmd_check)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
