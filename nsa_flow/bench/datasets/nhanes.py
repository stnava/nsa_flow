"""NHANES dietary nutrients -> all-cause mortality (public data; a cleaned release is read from a local folder).

Source: CDC NHANES (https://wwwn.cdc.gov/nchs/nhanes/Default.aspx) with NCHS linked mortality.  The benchmark reads a
cleaned release (``dietary_clean.csv``, ``mortality_clean.csv``, ``demographics_clean.csv``) from the folder in
``NSA_BENCH_NHANES_DIR`` (default ``~/Downloads/cleaned_nhanes_21743372``); the raw CDC files are public but are not
fetched or rebuilt here, so reproducing from scratch means obtaining the same cleaned release.  The first load builds
``nhanes_bench.npz`` in the data home; a legacy ``~/.cache/nsa_flow_nhanes.npz`` is used when the folder is absent.

Construction: dietary recall days are averaged per SEQN; the 77 numeric nutrient columns that are at least 90%
complete, after dropping the age/sex/survey-day columns and the derived ``VNDRXS*`` duplicates (age inside the "diet"
block would leak the strongest mortality predictor); complete cases only, nothing imputed.  Covariates in every
model: age, sex, race/ethnicity, poverty-income ratio, education, survey cycle.  Outcome: all-cause death among
mortality-eligible participants; survival time ``PERMTH_INT``; survey weight ``WTDRD1``.

Data are intrinsically non-negative (true zeros), so the non-negative NSA mode needs no shift.  Basis methods see the
raw nutrients (centred, not standardised); scores and covariates are z-scored before the downstream model.
Energy-adjusted variants: ``nhanes@density`` (per 1000 kcal) and ``nhanes@residual`` (log-log residual).
"""
import os
from pathlib import Path

import numpy as np

from .base import Dataset, DatasetUnavailable, dataset
from .cache import data_home

DEFAULT_DIR = "~/Downloads/cleaned_nhanes_21743372"
LEGACY = Path.home() / ".cache" / "nsa_flow_nhanes.npz"
EXCLUDE = {"RIDAGEYR", "RIAGENDR", "survey_day"}
ID = ["Unnamed: 0", "SEQN", "SEQN_new", "SDDSRVYR"]


def _build(folder: Path):
    import pandas as pd
    di = pd.read_csv(folder / "dietary_clean.csv", low_memory=False)
    num = di.select_dtypes(include=[np.number]).drop(columns=[c for c in ID if c in di], errors="ignore")
    cols = [c for c in num.columns[num.notna().mean() >= 0.9] if c not in EXCLUDE and not c.startswith("VNDRXS")]
    D = di[[c for c in ["SEQN", "WTDRD1"] + cols if c in di.columns]].dropna().groupby("SEQN").mean().reset_index()
    mo = pd.read_csv(folder / "mortality_clean.csv", low_memory=False)[
        ["SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_INT"]].drop_duplicates("SEQN")
    cov_cols = ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "SDDSRVYR"]
    de = pd.read_csv(folder / "demographics_clean.csv", low_memory=False)[["SEQN"] + cov_cols].drop_duplicates("SEQN")
    M = D.merge(mo, on="SEQN").merge(de, on="SEQN").query("ELIGSTAT == 1")
    cats = ["RIAGENDR", "RIDRETH1", "DMDEDUC2", "SDDSRVYR"]
    cov = pd.get_dummies(M[cov_cols].assign(**{c: M[c].astype("category") for c in cats}),
                         drop_first=True, dummy_na=False).astype(float)
    ok = (M[cols].notna().all(axis=1) & cov.notna().all(axis=1) & M.MORTSTAT.notna() & M.PERMTH_INT.notna()
          & M.WTDRD1.notna())
    return dict(X=M.loc[ok, cols].to_numpy(float), y=M.loc[ok, "MORTSTAT"].to_numpy(int),
                C=cov[ok.to_numpy()].to_numpy(float), E=M.loc[ok, "DRXTKCAL"].to_numpy(float),
                WT=M.loc[ok, "WTDRD1"].to_numpy(float), TT=M.loc[ok, "PERMTH_INT"].to_numpy(float),
                cols=np.array(cols, dtype=object))


def _arrays():
    cache = data_home() / "nhanes_bench.npz"
    if cache.exists():
        return np.load(cache, allow_pickle=True)
    folder = Path(os.path.expanduser(os.environ.get("NSA_BENCH_NHANES_DIR", DEFAULT_DIR)))
    if folder.exists():
        arrays = _build(folder)
        np.savez(cache, **arrays)
        return np.load(cache, allow_pickle=True)
    if LEGACY.exists():
        return np.load(LEGACY, allow_pickle=True)
    raise DatasetUnavailable(f"NHANES cleaned release not found at {folder} (set NSA_BENCH_NHANES_DIR); "
                             f"no cache at {cache} or {LEGACY}")


@dataset("nhanes", source="CDC NHANES + NCHS linked mortality (cleaned release read from a local folder)",
         description="77 dietary nutrients -> all-cause mortality (n~49k, 15% deaths); covariates age/sex/race/PIR/"
                     "education/cycle; survey weights and survival time carried")
def load_nhanes():
    z = _arrays()
    cols = [str(c) for c in z["cols"]]
    return Dataset("nhanes", z["X"], y=z["y"].astype(int), covariates=z["C"], task="classification",
                   outer="stratified_kfold", standardize=False, metric="auc", standardize_scores=True,
                   weights=z["WT"], time=z["TT"], event=z["y"].astype(int),
                   meta={"features": cols, "energy_index": cols.index("DRXTKCAL"), "outcome": "all-cause mortality",
                         "source": "CDC NHANES"})
