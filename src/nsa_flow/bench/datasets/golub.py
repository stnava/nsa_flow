"""Golub leukemia, 3 classes (B-ALL / T-ALL / AML), 7129 genes exported from Bioconductor golubEsets.

Expects ``golub_X.csv`` and ``golub_y3.csv`` in a data location (the repository's ``experiments/cache``
has them; ``experiments/data.py`` documents the one-off R export).  The 2000 highest-variance genes are
selected on the training portion of each split, after log2 with a floor of 1.
"""
import numpy as np

from .base import Dataset, dataset
from .cache import require

HINT = "export once with experiments/data.py:load_golub3() (needs R + golubEsets)"


@dataset("golub", source="Bioconductor golubEsets (cached CSV export)",
         description="Golub 3-class (B-ALL n=38, T-ALL n=9, AML n=25), p=7129, top-2000 variance genes per split")
def load_golub():
    import pandas as pd
    X = pd.read_csv(require("golub_X.csv", HINT)).to_numpy(float)
    labels = pd.read_csv(require("golub_y3.csv", HINT))["y"].to_numpy()
    classes = sorted(set(labels))
    y = np.array([classes.index(v) for v in labels])
    return Dataset("golub", np.log2(np.clip(X, 1.0, None)), y=y, task="classification",
                   outer="stratified_kfold", standardize=True, top_variance=2000,
                   meta={"classes": classes})
