"""Dataset loaders for the paper's real-data experiments.

Both datasets are loaded from the user's machine; neither is redistributed.

Golub leukemia
    Exported once from the Bioconductor package ``golubEsets`` (installed in the
    local R library) into ``experiments/cache/``.  72 samples x 7129 genes,
    47 ALL / 25 AML.

ADNI
    DKT regional grey-matter volumes from an ANTs single-subject-template run,
    joined to ADNIMERGE2 on the numeric roster ID parsed out of the image ID.
    816 subjects x 62 regions with diagnosis and demographics.  Note these are
    *volumes*, not cortical thickness.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

ADNI_VOLUMES = Path(os.path.expanduser("~/Downloads/adniGrayMatterVolumesDktANTsSST.csv"))
ADNI_MASTER = Path(os.path.expanduser("~/Downloads/ADNIMERGE2/ADNI_Master_Full_Prefix.csv"))

_GOLUB_R = r'''
suppressMessages({library(golubEsets); library(Biobase)})
data(Golub_Merge)
write.csv(t(exprs(Golub_Merge)), "%s", row.names=FALSE)
write.csv(data.frame(y=pData(Golub_Merge)$ALL.AML), "%s", row.names=FALSE)
'''


def load_golub():
    """Return ``(X, y, gene_names)`` with ``X`` [72, 7129] and ``y`` in {0=ALL, 1=AML}."""
    xf, yf = CACHE / "golub_X.csv", CACHE / "golub_y.csv"
    if not (xf.exists() and yf.exists()):
        script = CACHE / "_golub_export.R"
        script.write_text(_GOLUB_R % (xf, yf))
        subprocess.run(["Rscript", str(script)], check=True, capture_output=True)
    X = pd.read_csv(xf)
    y = (pd.read_csv(yf)["y"].astype(str) == "AML").astype(int).to_numpy()
    return X.to_numpy(dtype=np.float64), y, list(X.columns)


def load_adni(hemisphere="right", min_complete=0.95):
    """Return ``(X, meta, region_names)``: DKT grey-matter volumes and metadata.

    One scan per subject (the first), restricted to subjects with a diagnosis.

    ``hemisphere``
        ``"right"`` (default) uses the 31 right-hemisphere regions for every
        subject.  ``"both"`` uses all 62 regions on complete cases only.
        ``"left"`` is offered for completeness and is not recommended.

    Why the default is one hemisphere: in this derived table the left-hemisphere
    columns suffer systematic label dropout -- 24 of 31 left regions contain
    zeros, up to 53% of subjects for left fusiform, affecting 65% of subjects
    overall -- while only 6 of 31 right regions contain any zero at all, and
    those in 0.2% of subjects.  Where a left value *is* present it agrees with
    its right counterpart to a mean absolute log-ratio of 0.09, so the defect is
    missing labels rather than wrong magnitudes.  A zero here is a failed
    segmentation, not a measurement; imputing it would invent covariance in
    precisely the variables the method is asked to decorrelate.
    """
    for f in (ADNI_VOLUMES, ADNI_MASTER):
        if not f.exists():
            raise FileNotFoundError(f"ADNI input not found: {f}")

    vol = pd.read_csv(ADNI_VOLUMES)
    vol["RID"] = vol["ID"].str.extract(r"_S_(\d+)$")[0].astype(int)
    # The volumes table is longitudinal: 2932 scans over 816 subjects, up to 12
    # each.  Select the baseline scan explicitly as the smallest ADNI image UID,
    # which within a subject increases with acquisition date.  The file happens
    # to arrive sorted, so a plain drop_duplicates picks the same rows today,
    # but that is a property of the delivery and not of the data.
    vol["_iuid"] = vol["image id"].str.extract(r"^I(\d+)$")[0].astype(int)
    vol = vol.loc[vol.groupby("RID")["_iuid"].idxmin()].drop(columns="_iuid")

    master = pd.read_csv(
        ADNI_MASTER, usecols=["RID", "AGE", "SEX", "EDUC", "APOE", "DX"], low_memory=False
    )
    master = master.dropna(subset=["DX"]).drop_duplicates("RID")
    df = vol.merge(master, on="RID", how="inner")

    allr = [c for c in vol.columns if c.startswith("volume ")]
    if hemisphere == "both":
        regions = allr
    elif hemisphere in ("left", "right"):
        regions = [c for c in allr if f" {hemisphere} " in c]
    else:
        raise ValueError("hemisphere must be 'right', 'left' or 'both'")

    complete = (df[regions] > 0).mean()
    regions = [r for r in regions if complete[r] >= min_complete]
    n_before = len(df)
    df = df[(df[regions] > 0).all(axis=1)].reset_index(drop=True)

    meta = df[["RID", "AGE", "SEX", "EDUC", "APOE", "DX"]].copy()
    meta.attrs.update(hemisphere=hemisphere, n_dropped_subjects=n_before - len(df),
                      n_regions=len(regions))
    return df[regions].to_numpy(dtype=np.float64), meta, regions


def planted_partition(p=60, k=6, n=400, noise=0.3, seed=0, overlap=0):
    """Synthetic data with a known non-negative, disjoint-support basis.

    ``V_true`` assigns each of ``p`` features to one of ``k`` components, so the
    ground truth is exactly the ``w -> 1`` solution class.  ``overlap > 0`` bleeds
    that many features into a neighbouring component, breaking exact disjointness.
    """
    rng = np.random.default_rng(seed)
    V = np.zeros((p, k))
    per = p // k
    for j in range(k):
        V[j * per:(j + 1) * per, j] = 1.0
        if overlap:
            lo = (j + 1) % k * per
            V[lo:lo + overlap, j] = 0.7
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    Z = rng.random((n, k)) * 5.0
    X = Z @ V.T + noise * rng.standard_normal((n, p))
    return X, V, Z
