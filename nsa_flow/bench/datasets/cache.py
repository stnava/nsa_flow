"""Data locations and the explicit-download gate.

Search order for a cached file: ``$NSA_BENCH_DATA``, ``~/.cache/nsa_flow_bench``, and the repository's
``experiments/cache`` when running from a checkout.  Nothing is downloaded unless the caller passes
``allow_download=True`` (the CLI's ``fetch --download``); a missing dataset raises ``DatasetUnavailable``
with the places that were searched.
"""
from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .base import DatasetUnavailable


def data_home() -> Path:
    p = Path(os.environ.get("NSA_BENCH_DATA", Path.home() / ".cache" / "nsa_flow_bench"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def search_paths() -> list[Path]:
    paths = []
    if os.environ.get("NSA_BENCH_DATA"):
        paths.append(Path(os.environ["NSA_BENCH_DATA"]))
    paths.append(Path.home() / ".cache" / "nsa_flow_bench")
    repo_cache = Path(__file__).resolve().parents[3] / "experiments" / "cache"
    if repo_cache.exists():
        paths.append(repo_cache)
    return paths


def find(name: str) -> Path | None:
    for d in search_paths():
        if (d / name).exists():
            return d / name
    return None


def require(name: str, hint: str = "") -> Path:
    p = find(name)
    if p is None:
        raise DatasetUnavailable(f"{name} not found in {[str(d) for d in search_paths()]}. {hint}".strip())
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    """Download ``url`` to ``dest`` and verify the checksum when one is given."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    got = sha256(dest)
    if expected_sha256 and got != expected_sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}: expected {expected_sha256}, got {got}")
    return dest


def safe_extract(tar_path: Path, out_dir: Path):
    """Extract a tarball into an empty directory using the data filter (no path traversal, no links)."""
    with tarfile.open(tar_path) as t:
        t.extractall(out_dir, filter="data")


def scratch_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="nsa_bench_dl_"))
