"""Psychometric item data with known item->scale structure, plus factor-model simulators.

``bfi``        psych::bfi, 25 personality items (A, C, E, N, O scales, 5 items each), complete cases (n=2436).
               Reverse-keyed items (A1, C4, C5, E1, E2, O2, O5) are recorded in ``meta["keys"]`` and used only
               when scale totals are formed.
``holzinger``  lavaan::HolzingerSwineford1939, 9 mental tests, 3 factors (visual, textual, speed), n=301; the
               canonical CFA reference.
``fa_sim_n*``  simulated factor models (5 factors x 5 items, simple structure plus a few cross-loadings, factor
               correlation 0.2) with the true loadings in ``meta["true_loadings"]``; n = 100, 300, 1000.
``fa_null``    i.i.d. Gaussian items: no factor structure at all (the groups are an arbitrary assignment).

bfi and holzinger are exported once from the R packages psych and lavaan (``nsa-flow-bench fetch bfi --download``
runs the export; it needs Rscript with both packages) and cached as CSV in the data home.
"""
import subprocess

import numpy as np

from .base import Dataset, dataset
from .cache import data_home, require

BFI_REVERSE = {"A1", "C4", "C5", "E1", "E2", "O2", "O5"}
_R_EXPORT = r'''
suppressMessages({library(psych); library(lavaan)})
out <- commandArgs(TRUE)[1]
data(bfi); b <- bfi[, 1:25]; b <- b[complete.cases(b), ]
write.csv(b, file.path(out, "bfi.csv"), row.names = FALSE)
write.csv(HolzingerSwineford1939[, paste0("x", 1:9)], file.path(out, "holzinger.csv"), row.names = FALSE)
'''


def fetch():
    """Export bfi and holzinger from the installed R packages into the data home (explicit; no network)."""
    home = data_home()
    script = home / "_export_psychometric.R"
    script.write_text(_R_EXPORT)
    subprocess.run(["Rscript", str(script), str(home)], check=True, capture_output=True)


HINT = "run `nsa-flow-bench fetch bfi --download` (exports from R packages psych and lavaan; needs Rscript)"


@dataset("bfi", fetch=fetch, source="psych::bfi (R package psych, GPL), exported to CSV",
         description="Big Five Inventory: 25 items, 5 scales (A, C, E, N, O), n=2436 complete cases")
def load_bfi():
    import pandas as pd
    df = pd.read_csv(require("bfi.csv", HINT))
    groups = np.repeat(np.arange(5), 5)
    keys = np.array([-1.0 if c in BFI_REVERSE else 1.0 for c in df.columns])
    return Dataset("bfi", df.to_numpy(float), groups=groups, standardize=True,
                   meta={"items": list(df.columns), "keys": keys, "scales": list("ACENO"), "n_factors": 5})


@dataset("holzinger", fetch=fetch, source="lavaan::HolzingerSwineford1939 (R package lavaan), exported to CSV",
         description="Holzinger-Swineford: 9 mental tests, 3 factors (visual, textual, speed), n=301")
def load_holzinger():
    import pandas as pd
    df = pd.read_csv(require("holzinger.csv", HINT))
    return Dataset("holzinger", df.to_numpy(float), groups=np.repeat(np.arange(3), 3), standardize=True,
                   meta={"items": list(df.columns), "keys": np.ones(9), "scales": ["visual", "textual", "speed"],
                         "n_factors": 3})


def simulate_factor_model(n, p_per=5, k=5, phi=0.2, cross=0.15, p_cross=0.25, seed=0):
    """Items = F L' + unique noise with unit item variance; returns (X, loadings L, groups)."""
    rng = np.random.default_rng(seed)
    p = p_per * k
    L = np.zeros((p, k))
    groups = np.repeat(np.arange(k), p_per)
    for i in range(p):
        L[i, groups[i]] = rng.uniform(0.5, 0.8)
        if rng.random() < p_cross:
            other = rng.choice([j for j in range(k) if j != groups[i]])
            L[i, other] = rng.choice([-1, 1]) * rng.uniform(0.5 * cross, cross)
    Phi = np.full((k, k), phi) + (1 - phi) * np.eye(k)
    F = rng.multivariate_normal(np.zeros(k), Phi, size=n)
    common = F @ L.T
    uniq = np.sqrt(np.clip(1.0 - np.diag(L @ Phi @ L.T), 0.05, None))
    return common + rng.standard_normal((n, p)) * uniq, L, groups


def _make_sim(n):
    @dataset(f"fa_sim_n{n}", source="synthetic",
             description=f"simulated factor model, 5 factors x 5 items, known loadings, n={n}")
    def load(seed=0):
        X, L, groups = simulate_factor_model(n, seed=seed)
        return Dataset(f"fa_sim_n{n}", X, groups=groups, standardize=True,
                       meta={"keys": np.ones(X.shape[1]), "true_loadings": L, "n_factors": 5, "source": "synthetic"})
    return load


for _n in (100, 300, 1000):
    _make_sim(_n)


@dataset("fa_null", source="synthetic", description="i.i.d. Gaussian items (n=300, 25 items): no factor structure")
def load_fa_null(seed=0):
    X = np.random.default_rng(seed).standard_normal((300, 25))
    return Dataset("fa_null", X, groups=np.repeat(np.arange(5), 5), standardize=True,
                   meta={"keys": np.ones(25), "n_factors": 5, "null": "no structure", "source": "synthetic"})
