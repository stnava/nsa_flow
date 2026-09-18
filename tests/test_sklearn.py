"""Tests for the scikit-learn compatible NSAFlow estimator."""
import numpy as np
import pytest
import torch

from nsa_flow import NSAFlow


def test_sklearn_fit_transform_nonneg():
    """NSAFlow works as a transformer on non-negative data."""
    np.random.seed(42)
    X = np.random.rand(50, 15)
    model = NSAFlow(n_components=4, w=0.5)
    scores = model.fit_transform(X)
    assert scores.shape == (50, 4)
    assert model.components_.shape == (4, 15)
    assert (model.components_ >= -1e-12).all()
    assert model.result_.converged


def test_sklearn_fit_transform_signed():
    """NSAFlow works as a transformer on signed data."""
    np.random.seed(43)
    X = np.random.randn(50, 15)
    model = NSAFlow(n_components=4, w=0.5)
    scores = model.fit_transform(X)
    assert scores.shape == (50, 4)
    assert model.components_.shape == (4, 15)
    assert (model.components_ < 0).any() and (model.components_ > 0).any()
    assert model.result_.converged


def test_sklearn_consolidate():
    """NSAFlow with consolidate=True achieves zero lobe overlap."""
    np.random.seed(44)
    X = np.random.randn(50, 15)
    model = NSAFlow(n_components=4, w=0.5, consolidate=True)
    scores = model.fit_transform(X)
    assert scores.shape == (50, 4)
    assert model.result_["lobe_overlap"] == 0.0


def test_sklearn_pipeline_compatible():
    """NSAFlow can be used inside a scikit-learn Pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    np.random.seed(45)
    X = np.random.randn(60, 12)
    y = np.random.choice([0, 1], size=60)

    pipe = Pipeline([
        ("nsa", NSAFlow(n_components=3, w=0.5)),
        ("clf", LogisticRegression())
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == 60


def test_sklearn_lbfgs_optimizer():
    """NSAFlow works with optimizer='lbfgs'."""
    np.random.seed(46)
    X = np.random.rand(40, 10)
    model = NSAFlow(n_components=3, w=0.5, optimizer="lbfgs")
    scores = model.fit_transform(X)
    assert scores.shape == (40, 3)
    assert model.result_.converged
