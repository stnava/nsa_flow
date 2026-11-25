"""METABRIC log2 expression: PAM50 functional groups among ~1000 high-variance background genes.

Reads ``metabric_expr.npz`` (arrays ``X`` genes x tumours, ``gene_syms``) from a data location.  The four
group lists (proliferation, luminal/ER, basal, HER2) are the benchmark authors' annotation of the published
PAM50 panel (Parker et al., J Clin Oncol 2009), NOT data shipped with the cache; BLVRA, MDM2, MMP11 and
TMEM45B have no clean group and are treated as background.  ORC6L is ORC6 in current symbols.  Review the
lists before citing results.
"""
import numpy as np

from .base import Dataset, dataset
from .cache import require

GROUPS = {
    "proliferation": "BIRC5 CCNB1 CDC20 CDC6 CENPF CEP55 EXO1 KIF2C MELK MKI67 MYBL2 NDC80 "
                     "NUF2 ORC6 PTTG1 RRM2 TYMS UBE2C UBE2T ANLN CCNE1".split(),
    "luminal": "ESR1 PGR FOXA1 BCL2 MLPH NAT1 SLC39A6 GPR160 MAPT BAG1 CXXC5".split(),
    "basal": "CDH3 EGFR FOXC1 KRT14 KRT17 KRT5 ACTR3B PHGDH MIA SFRP1".split(),
    "her2": "ERBB2 GRB7 FGFR4".split(),
}
UNGROUPED = "BLVRA MDM2 MMP11 TMEM45B".split()


@dataset("metabric", private=True, tier="local", source="METABRIC expression (local cache metabric_expr.npz)",
         description="45 PAM50 genes in 4 functional groups + 1000 high-variance background genes, 1980 tumours")
def load_metabric(n_background=1000):
    z = np.load(require("metabric_expr.npz"), allow_pickle=True)
    X, syms = z["X"].astype(float), list(z["gene_syms"])
    idx = {g: i for i, g in enumerate(syms)}
    lab = {idx[g]: gi for gi, ms in enumerate(GROUPS.values()) for g in ms if g in idx}
    excl = set(lab) | {idx[g] for g in UNGROUPED if g in idx}
    bg = [i for i in np.argsort(-X.var(1)) if i not in excl][:n_background]
    cols = list(lab) + bg
    groups = np.array([lab[c] for c in lab] + [-1] * len(bg))
    return Dataset("metabric", X[cols].T, groups=groups, standardize=True, draw_n=1500,
                   meta={"annotation": "authors' functional grouping of PAM50", "groups": list(GROUPS)})
