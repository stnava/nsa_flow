"""20 Newsgroups TF-IDF words -> six super-categories (module recovery).

Features: the 2000 most frequent TF-IDF words (sublinear tf, English stop words removed, min_df=20).
Truth: the super-category (comp / rec / sci / forsale / politics / religion) holding most of a word's TF-IDF
mass over the full training set, kept only for words with >= 60% of their mass in one category; other words
are unlabelled background (-1).  Labels define the reference and are never shown to a method.
"""
from pathlib import Path

import numpy as np

from .base import Dataset, DatasetUnavailable, dataset

CATS = ["comp", "rec", "sci", "misc.forsale", "politics", "religion"]


def _category(name):
    if name.startswith("talk.politics"):
        return "politics"
    if name.startswith(("alt.atheism", "soc.religion", "talk.religion")):
        return "religion"
    return next(c for c in CATS if name.startswith(c))


def fetch():
    load_newsgroups(allow_download=True)


@dataset("newsgroups", fetch=fetch, source="scikit-learn fetch_20newsgroups (cached in ~/scikit_learn_data)",
         description="20 Newsgroups: 2000 TF-IDF words, truth = 6 super-categories by majority mass (>= 60%)")
def load_newsgroups(allow_download=False, p=2000, min_share=0.6):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not (allow_download or (Path.home() / "scikit_learn_data/20news-bydate_py3.pkz").exists()):
        raise DatasetUnavailable("20 Newsgroups not cached in ~/scikit_learn_data; "
                                 "run `nsa-flow-bench fetch newsgroups --download`")
    d = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    X = TfidfVectorizer(max_features=p, min_df=20, stop_words="english",
                        sublinear_tf=True).fit_transform(d.data).tocsr()
    sup = np.array([CATS.index(_category(d.target_names[t])) for t in d.target])
    M = np.column_stack([np.asarray(X[sup == c].sum(0)).ravel() for c in range(len(CATS))])
    share = M / M.sum(1, keepdims=True)
    groups = np.where(share.max(1) >= min_share, share.argmax(1), -1)
    return Dataset("newsgroups", X.toarray().astype(np.float32), y=sup, groups=groups, standardize=False,
                   kmeans_spherical=True, draw_n=2000, task="classification", outer="stratified_kfold",
                   meta={"annotation": "super-category by majority TF-IDF mass", "categories": CATS})
