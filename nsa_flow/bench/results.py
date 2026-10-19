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
    return dict(suite=asdict(suite), profile=asdict(profile), jobs=jobs, versions=versions(), git=git_sha(), **core_hash(),
                host=platform.platform(), started=time.strftime("%Y-%m-%dT%H:%M:%S"), command=command,
                entries=entries_info)


def core_hash() -> dict:
    """Fingerprint of the library under test: sha1 over the core ``nsa_flow/*.py`` sources (``bench/`` excluded) and
    whether the working tree has uncommitted changes there.  Results are only comparable across runs with the same
    core hash; a changed core is a different experiment (the loky/branch-switch lesson)."""
    import hashlib
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha1()
    for f in sorted(root.rglob("*.py")):
        if "bench" in f.relative_to(root).parts[:1] or "__pycache__" in f.parts:
            continue
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    try:
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "."], capture_output=True,
                                    text=True, timeout=10).stdout.strip())
    except Exception:
        dirty = None
    return {"core_hash": h.hexdigest()[:12], "tree_dirty": dirty}


def start_run(out_dir, suite_name) -> Path:
    """Create the run directory up front so every finished entry is saved immediately."""
    d = Path(out_dir) / suite_name / time.strftime("%Y%m%d-%H%M%S")
    (d / "entries").mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / suite_name / "running.txt").write_text(d.name)
    return d


def entry_key(entry, profile_name, core) -> str:
    """Identity of an entry's inputs: the pre-registered entry, the profile and the core hash."""
    import hashlib
    return hashlib.sha1(json.dumps([asdict(entry), profile_name, core], sort_keys=True, default=str)
                        .encode()).hexdigest()[:16]


def save_entry(d, idx, key, rows: pd.DataFrame, info: dict):
    rows.to_csv(Path(d) / "entries" / f"entry_{idx:02d}.csv", index=False)
    (Path(d) / "entries" / f"entry_{idx:02d}.json").write_text(json.dumps({**info, "key": key}, default=str))
    log_progress(d, f"entry {idx:02d} {info.get('dataset')} / {info.get('protocol')} {info.get('status')} "
                    f"{info.get('seconds', '')}s")


def load_entry(d, idx, key):
    """The saved (rows, info) of entry ``idx`` if it was saved with the same key, else None."""
    j = Path(d) / "entries" / f"entry_{idx:02d}.json"
    c = Path(d) / "entries" / f"entry_{idx:02d}.csv"
    if not (j.exists() and c.exists()):
        return None
    info = json.loads(j.read_text())
    return (pd.read_csv(c), info) if info.get("key") == key else None


def log_progress(d, line):
    with open(Path(d) / "progress.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {line}\n")


def finalize_run(d, suite_name, rows: pd.DataFrame, man: dict, summary=None, verdicts=None) -> Path:
    d = Path(d)
    cols = [c for c in COLUMNS if c in rows.columns] + [c for c in rows.columns if c not in COLUMNS]
    rows[cols].to_csv(d / "rows.csv", index=False)
    if summary is not None:
        summary.to_csv(d / "summary.csv", index=False)
    man = {**man, "ended": time.strftime("%Y-%m-%dT%H:%M:%S"), "verdicts": verdicts or []}
    (d / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
    (d.parent / "latest.txt").write_text(d.name)
    (d.parent / "running.txt").unlink(missing_ok=True)
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
