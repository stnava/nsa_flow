"""Run storage: append-only run directories with a tidy ``rows.csv`` and a ``manifest.json``."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

COLUMNS = ["suite", "dataset", "protocol", "split", "method", "variant", "model", "metric", "value",
           "inner_value", "hp", "sparsity", "overlap", "seconds", "cert_converged", "cert_defect",
           "skipped_reason"]


def git_sha() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def versions() -> dict:
    import numpy
    out = {"python": platform.python_version(), "numpy": numpy.__version__}
    for mod in ("torch", "sklearn", "scipy", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    import nsa_flow
    out["nsa_flow"] = getattr(nsa_flow, "__version__", "unknown")
    return out


def manifest(suite, profile, jobs, entries_info, command) -> dict:
    return dict(suite=asdict(suite), profile=asdict(profile), jobs=jobs, versions=versions(), git=git_sha(),
                host=platform.platform(), started=time.strftime("%Y-%m-%dT%H:%M:%S"), command=command,
                entries=entries_info)


def save_run(out_dir, suite_name, rows: pd.DataFrame, man: dict, summary=None, verdicts=None) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    d = Path(out_dir) / suite_name / run_id
    d.mkdir(parents=True, exist_ok=True)
    cols = [c for c in COLUMNS if c in rows.columns] + [c for c in rows.columns if c not in COLUMNS]
    rows[cols].to_csv(d / "rows.csv", index=False)
    if summary is not None:
        summary.to_csv(d / "summary.csv", index=False)
    man = {**man, "ended": time.strftime("%Y-%m-%dT%H:%M:%S"), "verdicts": verdicts or []}
    (d / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
    (Path(out_dir) / suite_name / "latest.txt").write_text(run_id)
    return d


def resolve_run(path_or_suite, out_dir="bench_results") -> Path:
    """A run directory, or ``suite`` (-> its latest run under ``out_dir``)."""
    p = Path(path_or_suite)
    if (p / "rows.csv").exists():
        return p
    s = Path(out_dir) / str(path_or_suite)
    if (s / "latest.txt").exists():
        return s / (s / "latest.txt").read_text().strip()
    raise FileNotFoundError(f"no run at {p} and no latest run for suite {path_or_suite!r} under {out_dir}")


def load_run(path):
    d = Path(path)
    rows = pd.read_csv(d / "rows.csv")
    man = json.loads((d / "manifest.json").read_text())
    summ = pd.read_csv(d / "summary.csv") if (d / "summary.csv").exists() else pd.DataFrame()
    return rows, man, summ
