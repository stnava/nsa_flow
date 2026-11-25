"""Tecator NIR (OpenML 505): 100 absorbance wavelengths -> fat %.  Features are in spectral order."""
from pathlib import Path

from .base import Dataset, DatasetUnavailable, dataset


def _cached():
    return (Path.home() / "scikit_learn_data/openml/openml.org/api/v1/json/data/505.gz").exists()


def fetch():
    load_tecator(allow_download=True)


@dataset("tecator", source="OpenML 505 via scikit-learn (cached in ~/scikit_learn_data)", fetch=fetch,
         description="Tecator NIR spectra (n=240, p=100, ordered wavelengths) -> fat %")
def load_tecator(allow_download=False):
    from sklearn.datasets import fetch_openml
    if not (_cached() or allow_download):
        raise DatasetUnavailable("Tecator (OpenML 505) not cached in ~/scikit_learn_data; "
                                 "run `nsa-flow-bench fetch tecator --download`")
    t = fetch_openml(data_id=505, as_frame=True, parser="auto")
    cols = [c for c in t.data.columns if "absorbance" in c]
    return Dataset("tecator", t.data[cols].to_numpy(float), y=t.target.to_numpy(float), ordered=True,
                   task="regression", outer="kfold", standardize=True, draw_n=192,
                   meta={"wavelengths": cols})

