"""Scikit-Learn compatible estimator for NSA-Flow."""
import numpy as np
import torch
from .solve import nsa_flow

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    _Base = (BaseEstimator, TransformerMixin)
except ImportError:
    _Base = (object,)


class NSAFlow(*_Base):
    """Scikit-Learn compatible transformer for NSA-Flow representation learning.

    Fits an interpretable non-negative or signed contrast basis on data matrix ``X``
    and projects new or existing samples into the sparse component subspace via
    linear projection: ``scores = X @ V``.

    Parameters
    ----------
    n_components : int, default 6
        Number of components (k) to extract.
    w : float, default 0.5
        Trade-off weight in [0, 1]. w=0 maximizes reconstruction fidelity,
        w=1 maximizes orthogonality (disjoint supports).
    mode : {"auto", "data", "signed", "anchored"}, default "auto"
        Execution mode. 'auto' inspects data sign distribution:
        - If data is strictly non-negative -> fits non-negative basis V >= 0.
        - If data has negative entries -> fits signed contrast basis V = V+ - V-.
    consolidate : bool, default False
        Guarantees exact disjoint supports (zero lobe overlap) in signed mode.
    optimizer : {"spg", "lbfgs"}, default "spg"
        Optimization algorithm:
        - "spg": Accelerated Spectral Projected Gradient (pure PyTorch).
        - "lbfgs": Limited-memory BFGS with bound constraints (SciPy).
    init : str or Tensor, default None
        Initialization strategy ('clamp', 'relax', 'nmf', 'random') or tensor.
    max_iter : int, optional
        Maximum iterations cap.
    tol : float, optional
        Stationarity tolerance.
    **kwargs :
        Additional arguments forwarded to `nsa_flow`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal components / loadings (transposed basis, V.T), adhering
        to the standard scikit-learn PCA/NMF convention.
    result_ : NSAResult
        The full result object containing energies, defects, and convergence diagnostics.
    """

    def __init__(self, n_components=6, w=0.5, mode="auto", consolidate=False,
                 optimizer="spg", init=None, max_iter=None, tol=None, **kwargs):
        self.n_components = n_components
        self.w = w
        self.mode = mode
        self.consolidate = consolidate
        self.optimizer = optimizer
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """Fit NSA-Flow basis on data matrix X.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for scikit-learn API compatibility.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        res = nsa_flow(
            X, k=self.n_components, w=self.w, mode=self.mode,
            consolidate=self.consolidate, optimizer=self.optimizer,
            init=self.init, max_iter=self.max_iter, tol=self.tol,
            **self.kwargs
        )
        self.result_ = res
        self.components_ = res.Y.detach().cpu().numpy().T  # [k, p]
        return self

    def transform(self, X):
        """Project X into the fitted NSA-Flow subspace.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projected component scores.
        """
        if not hasattr(self, "components_"):
            raise ValueError("NSAFlow instance is not fitted yet. Call 'fit' before 'transform'.")
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        return X_arr @ self.components_.T  # [n, k]

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : Ignored

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projected component scores.
        """
        return self.fit(X, y).transform(X)
