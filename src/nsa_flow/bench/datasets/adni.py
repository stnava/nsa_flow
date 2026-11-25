"""ADNI baseline cortical thickness -> CDRSB (private; load from a configured path).

Set ``NSA_BENCH_ADNI_THICKNESS`` to the filtered ADNI/PPMI thickness CSV.  Rows: ADNI baseline scans;
features: bilateral-average thickness columns (``T1Hier_thk_*LRAVG``); covariates: intercept, age, sex,
education, APOE4.  Features are centred, not standardised (as in the paper's ADNI analysis).
"""
import os
from pathlib import Path

import numpy as np

from .base import Dataset, DatasetUnavailable, dataset

DEFAULT = ("~/Library/Mobile Documents/com~apple~CloudDocs/code/multidisorder/data/ppmiadni_filtered.csv")


@dataset("adni", private=True, tier="restricted", source="ADNI (private; set NSA_BENCH_ADNI_THICKNESS)",
         description="cortical thickness (bilateral averages) -> CDRSB, covariates age/sex/education/APOE4")
def load_adni():
    import pandas as pd
    p = Path(os.path.expanduser(os.environ.get("NSA_BENCH_ADNI_THICKNESS", DEFAULT)))
    if not p.exists():
        raise DatasetUnavailable(f"ADNI thickness table not found at {p}; set NSA_BENCH_ADNI_THICKNESS")
    df = pd.read_csv(p, low_memory=False)
    df = df[(df.studyName == "ADNI") & (df.yearsbl == 0)]
    regions = [c for c in df.columns if "T1Hier_thk_" in c and "LRAVG" in c
               and not any(x in c for x in ("Asym", "reference", "adjusted"))]
    X = df[regions].apply(pd.to_numeric, errors="coerce")
    ok = X.notna().all(axis=1)
    X, df = X[ok], df[ok].reset_index(drop=True)
    X = X.loc[:, X.std() > 0]
    C = np.column_stack([
        np.ones(len(df)),
        pd.to_numeric(df.AGE, errors="coerce").to_numpy(float),
        df.PTGENDER.astype(str).str.upper().str.startswith("M").to_numpy(float),
        pd.to_numeric(df.PTEDUCAT, errors="coerce").to_numpy(float),
        pd.to_numeric(df.APOE4, errors="coerce").to_numpy(float)])
    y = pd.to_numeric(df["CDRSB"], errors="coerce").to_numpy(float)
    keep = np.isfinite(y) & np.isfinite(C).all(1)
    return Dataset("adni", X.to_numpy(float)[keep], y=y[keep], covariates=C[keep], task="regression",
                   outer="holdout", standardize=False,
                   meta={"features": list(X.columns), "outcome": "CDRSB"})
