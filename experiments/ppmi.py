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

# Clinical-diagnosis contrasts.  Retained for reference only: PD-vs-CN from
# structural MRI is close to hopeless (PCA reaches AUC 0.551), because the label
# is a clinical judgement rather than a biological state.
DX_TASKS = {
    "PD vs CN": (["PDSporadic", "PDLRRK2", "PDGBA"], ["CN"]),
    "Prodromal vs CN": (["ProdromalSporadic", "ProdromalLRRK2", "ProdromalGBA"], ["CN"]),
    "PD vs Prodromal": (["PDSporadic", "PDLRRK2", "PDGBA"],
                        ["ProdromalSporadic", "ProdromalLRRK2", "ProdromalGBA"]),
}

# The real target: alpha-synuclein seed amplification assay status, a
# biologically defined label (1479 Positive / 997 Negative).  Stratifying by
# diagnosis removes the clinical confound entirely -- within the prodromal group
# SAA status is exactly the question clinical DX cannot answer.
SAA_TASKS = {
    "SAA+ vs SAA-": (None, None),
    "SAA within Prodromal": ("Prodromal", None),
    "SAA within PD": ("PD", None),
    "SAA within CN+Prodromal": ("CN|Prodromal", None),
}
TASKS = DX_TASKS


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

    cl_cols = ["subjectID", "joinedDX", "AsynStatus", "age_BL", "commonSex"]
    clin = pd.read_csv(CLIN, usecols=cl_cols, low_memory=False)
    clin = clin.drop_duplicates("subjectID").rename(columns={"subjectID": "commonID"})
    # the two tables type commonID differently (str vs int64); normalise to str
    for t in (idp, clin):
        t["commonID"] = t["commonID"].astype(str).str.strip()
    df = idp.merge(clin[["commonID", "joinedDX", "AsynStatus"]], on="commonID",
                   how="inner")

    if task in SAA_TASKS:                       # biologically defined label
        df = df.dropna(subset=["AsynStatus"])
        stratum = SAA_TASKS[task][0]
        if stratum is not None:
            df = df[df.joinedDX.fillna("").str.contains(stratum, regex=True)]
        df = df.reset_index(drop=True)
    else:
        df = df.dropna(subset=["joinedDX"])
        pos, neg = DX_TASKS[task]
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
    if task in SAA_TASKS:
        y = (df.AsynStatus == "Positive").to_numpy(int)
    else:
        y = df.joinedDX.isin(pos).to_numpy(int)
    meta = df[["commonID", "joinedDX", "AsynStatus"]].copy()
    if "age_BL" in df:
        meta["age"] = pd.to_numeric(df["age_BL"], errors="coerce")
    if "commonSex" in df:
        meta["sex"] = df["commonSex"].astype(str)
    return F.to_numpy(float), y, meta, cols


# ---------------------------------------------------------------------------
# Continuous outcomes, for the same design used on ADNI cortical thickness.
# This is a second cohort rather than a second analysis of the first: different
# disease, different scanner protocol, different outcome scales.  T1w has p = 66,
# the same feature count as the ADNI thickness table, which makes it a close
# replication rather than a loose analogy.
# ---------------------------------------------------------------------------
# SAA status is the biomarker-defined label; UPDRS-I is non-motor
# experiences of daily living, a continuous score.
PPMI_OUTCOMES = ["AsynStatus", "updrs1_score"]
PPMI_COVARS = ["age_BL", "educ"]        # sex is absent from this extract


def load_ppmi_continuous(modality="T1w", min_complete=0.9, max_p=None):
    """Return ``(X, df, cols)`` with the IDPs and the continuous clinical scores.

    Unlike ``load_ppmi`` this does not select a binary contrast; it returns every
    subject with complete imaging so that each outcome can define its own
    complete-case sample, as in the ADNI analysis.
    """
    pref = MODALITIES[modality]
    head = pd.read_csv(IDPS, nrows=0).columns.tolist()
    cols = [c for c in head if c.startswith(pref)]
    keep = [c for c in ["commonID", "age_BL", "yearsbl"] + cols if c in head]
    idp = pd.read_csv(IDPS, usecols=keep, low_memory=False)
    idp = (idp.sort_values("yearsbl").drop_duplicates("commonID")
           if "yearsbl" in idp else idp.drop_duplicates("commonID"))

    want = ["subjectID"] + PPMI_OUTCOMES + PPMI_COVARS
    clin = pd.read_csv(CLIN, low_memory=False)
    want = [c for c in want if c in clin.columns]
    clin = clin[want].drop_duplicates("subjectID").rename(
        columns={"subjectID": "commonID"})
    # both tables carry age_BL; keep the clinical one and drop the imaging copy
    # so the merge does not produce age_BL_x / age_BL_y
    if "age_BL" in clin.columns and "age_BL" in idp.columns:
        idp = idp.drop(columns=["age_BL"])
    for t in (idp, clin):
        t["commonID"] = t["commonID"].astype(str).str.strip()
    df = idp.merge(clin, on="commonID", how="inner")

    F = df[cols].apply(pd.to_numeric, errors="coerce")
    ok = (F.notna().mean() >= min_complete) & (F.std(numeric_only=True) > 0)
    cols = [c for c in cols if ok.get(c, False)]
    F = F[cols]
    row_ok = F.notna().all(axis=1)
    F, df = F[row_ok], df[row_ok].reset_index(drop=True)
    if max_p is not None and len(cols) > max_p:
        cols = list(F.var().sort_values(ascending=False).index[:max_p])
        F = F[cols]
    return F.to_numpy(float), df, cols


# ---------------------------------------------------------------------------
# Confound-modelled design.  The first PPMI attempt (exp20) used age and
# education only, which is not enough: see exp20's docstring for why the result
# it produced had to be retracted.  Two things have to change.
#
# Stratify.  SAA status over the whole cohort is largely a restatement of
# diagnosis (cohort label alone predicts it at AUC 0.920), so it must be scored
# WITHIN a stratum.  Prodromal is the stratum that matters and the only one with
# usable balance: 606 negative / 276 positive.  Within PD the label is 91%
# positive and within CN it is 6%, so neither carries much.
#
# Adjust.  The extract has five subject-level confounds that apply everywhere --
# age, sex, education, imaging protocol (two levels) and brain volume -- plus
# two that apply only to PD: levodopa-equivalent dose, which lowers symptom
# scores, and disease duration.  ``duration_yrs`` is undefined outside PD (1405
# of 1423 values), so any model using it is a within-PD model by construction.
# Diagnosis is not the only label the target can hide behind.  Within the
# prodromal group, genetic subtype almost determines SAA status: sporadic
# prodromals are 63% positive (190/301) while GBA and LRRK2 carriers are 7%
# (12/166 and 12/171).  Stratifying on diagnosis and stopping there would repeat
# exp20's error one level down, so ``genotype`` is a confound wherever a stratum
# mixes subtypes, and the primary SAA target is the sporadic prodromal stratum,
# where the label is balanced (111 negative / 190 positive) and no genetic or
# diagnostic variable stands in for it.
CONFOUNDS_ALL = ["age_BL", "commonSex", "educ", "imaging_protocol", "brainVolume"]
CONFOUNDS_GEN = CONFOUNDS_ALL + ["genotype"]
CONFOUNDS_PD = CONFOUNDS_ALL + ["LEDD", "duration_yrs"]
CONFOUNDS_PD_GEN = CONFOUNDS_PD + ["genotype"]

STRATA = {                      # name -> regex matched against ``joinedDX``
    "Prodromal": r"^Prodromal",
    "ProdromalSporadic": r"^ProdromalSporadic$",
    "PD": r"^PD",
    "PDSporadic": r"^PDSporadic$",
    "CN": r"^CN$",
    "CN+Prodromal": r"^(CN|Prodromal)",
    "all": None,
}


def load_ppmi_modeled(modality="T1w", stratum="Prodromal", outcome="AsynStatus",
                      confounds=None, min_complete=0.9, max_p=None):
    """Return ``(X, y, C, cols, cnames)`` for one stratum, outcome and confound set.

    ``X`` is the imaging matrix, ``y`` the outcome (0/1 for ``AsynStatus``, float
    otherwise), and ``C`` the confound design with categoricals dummy-coded and
    the intercept left out.  Complete cases only, on the union of imaging,
    outcome and confounds, so the confound-adjusted and unadjusted models are
    fit on exactly the same rows -- otherwise adding a covariate would change
    the sample and the comparison would not be paired.
    """
    confounds = list(CONFOUNDS_ALL if confounds is None else confounds)
    pref = MODALITIES[modality]
    head = pd.read_csv(IDPS, nrows=0).columns.tolist()
    cols = [c for c in head if c.startswith(pref)]
    keep = [c for c in ["commonID", "yearsbl"] + cols if c in head]
    idp = pd.read_csv(IDPS, usecols=keep, low_memory=False)
    idp = (idp.sort_values("yearsbl").drop_duplicates("commonID")
           if "yearsbl" in idp else idp.drop_duplicates("commonID"))

    clin = pd.read_csv(CLIN, low_memory=False)
    want = ["subjectID", "joinedDX", outcome] + confounds
    clin = (clin[[c for c in dict.fromkeys(want) if c in clin.columns]]
            .drop_duplicates("subjectID").rename(columns={"subjectID": "commonID"}))
    if "genotype" in confounds:               # derived, not a column in the file
        clin["genotype"] = (clin.joinedDX.fillna("NA")
                            .str.extract(r"(Sporadic|GBA|LRRK2|SNCA|PRKN)",
                                         expand=False).fillna("none"))
    missing = [c for c in confounds + [outcome] if c not in clin.columns]
    if missing:
        raise ValueError(f"not in the clinical extract: {missing}")
    for t in (idp, clin):
        t["commonID"] = t["commonID"].astype(str).str.strip()
    df = idp.merge(clin, on="commonID", how="inner")

    pat = STRATA[stratum] if stratum in STRATA else stratum
    if pat is not None:
        df = df[df.joinedDX.fillna("NA").str.match(pat)]
    df = df.reset_index(drop=True)

    F = df[cols].apply(pd.to_numeric, errors="coerce")
    ok = (F.notna().mean() >= min_complete) & (F.std(numeric_only=True) > 0)
    cols = [c for c in cols if ok.get(c, False)]
    F = F[cols]

    # dummy-code before dropping rows so a level lost to the stratum disappears
    des = df[confounds].copy()
    for c in confounds:
        if des[c].dtype == object or des[c].nunique(dropna=True) <= 2:
            des[c] = des[c].astype("category")
    des = pd.get_dummies(des, drop_first=True, dummy_na=False).astype(float)
    des = des.loc[:, des.std() > 0]

    yy = df[outcome]
    yy = (yy == "Positive").astype(float).where(yy.notna()) \
        if outcome == "AsynStatus" else pd.to_numeric(yy, errors="coerce")
    row_ok = F.notna().all(axis=1) & des.notna().all(axis=1) & yy.notna()
    F, des, yy = F[row_ok], des[row_ok], yy[row_ok]
    if max_p is not None and len(cols) > max_p:
        cols = list(F.var().sort_values(ascending=False).index[:max_p])
        F = F[cols]
    y = yy.to_numpy(int) if outcome == "AsynStatus" else yy.to_numpy(float)
    return (F.to_numpy(float), y, des.to_numpy(float), cols,
            list(des.columns))
