r"""PPMI: a harder biomedical benchmark than ADNI regional volumes.

Parkinson's diagnosis from imaging-derived phenotypes is substantially harder
than Alzheimer's from grey-matter volumes -- PD atrophy is subtle and the
discriminative signal is not dominated by global size -- and the IDP table is
multimodal (528 rsfMRI, 372 T1 hierarchical, 338 DTI, 62 T1w features), so it
also spans several p/n regimes within one cohort.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

IDPS = Path(os.path.expanduser("~/Downloads/ppmi_idps_trim_v1.4.0.csv"))
CLIN = Path(os.path.expanduser("~/Downloads/ppmitrim0_SR_first_extended.csv"))

MODALITIES = {"rsfMRI": "rsfMRI", "T1Hier": "T1Hier", "DTI": "DTI", "T1w": "T1w"}
TASKS = {
    "PD vs CN": (["PDSporadic", "PDLRRK2", "PDGBA"], ["CN"]),
    "Prodromal vs CN": (["ProdromalSporadic", "ProdromalLRRK2", "ProdromalGBA"], ["CN"]),
    "PD vs Prodromal": (["PDSporadic", "PDLRRK2", "PDGBA"],
                        ["ProdromalSporadic", "ProdromalLRRK2", "ProdromalGBA"]),
}


def load_ppmi(modality="T1Hier", task="PD vs CN", min_complete=0.9, max_p=None):
    """Return ``(X, y, meta, cols)`` for one modality and one binary contrast.

    One row per subject (earliest visit).  Columns are kept when at least
    ``min_complete`` of subjects have a finite, non-constant value; rows are then
    restricted to complete cases, so nothing is imputed -- imputation would
    invent covariance among exactly the variables being decorrelated.
    """
    for f in (IDPS, CLIN):
        if not f.exists():
            raise FileNotFoundError(f)
    pref = MODALITIES[modality]
    head = pd.read_csv(IDPS, nrows=0).columns.tolist()
    cols = [c for c in head if c.startswith(pref)]
    if not cols:
        raise ValueError(f"no columns with prefix {pref!r}")
    keep = ["commonID", "age_BL", "commonSex", "yearsbl"] + cols
    keep = [c for c in keep if c in head]
    idp = pd.read_csv(IDPS, usecols=keep, low_memory=False)
    idp = idp.sort_values("yearsbl").drop_duplicates("commonID") if "yearsbl" in idp else \
        idp.drop_duplicates("commonID")

    clin = pd.read_csv(CLIN, usecols=["subjectID", "joinedDX", "age_BL", "commonSex"],
                       low_memory=False).dropna(subset=["joinedDX"])
    clin = clin.drop_duplicates("subjectID").rename(columns={"subjectID": "commonID"})
    # the two tables type commonID differently (str vs int64); normalise to str
    for t in (idp, clin):
        t["commonID"] = t["commonID"].astype(str).str.strip()
    df = idp.merge(clin[["commonID", "joinedDX"]], on="commonID", how="inner")

    pos, neg = TASKS[task]
    df = df[df.joinedDX.isin(pos + neg)].reset_index(drop=True)

    F = df[cols].apply(pd.to_numeric, errors="coerce")
    ok = (F.notna().mean() >= min_complete) & (F.std(numeric_only=True) > 0)
    cols = [c for c in cols if ok.get(c, False)]
    F = F[cols]
    row_ok = F.notna().all(axis=1)
    F, df = F[row_ok], df[row_ok]
    if max_p is not None and len(cols) > max_p:          # unsupervised, for speed
        cols = list(F.var().sort_values(ascending=False).index[:max_p])
        F = F[cols]
    y = df.joinedDX.isin(pos).to_numpy(int)
    meta = df[["commonID", "joinedDX"]].copy()
    if "age_BL" in df:
        meta["age"] = pd.to_numeric(df["age_BL"], errors="coerce")
    if "commonSex" in df:
        meta["sex"] = df["commonSex"].astype(str)
    return F.to_numpy(float), y, meta, cols
