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
    optimizer : str, optional
        Optimization algorithm; see :func:`nsa_flow.optim.optimizer_names`.
        ``None`` (default) uses :data:`nsa_flow.solve.DEFAULT_OPTIMIZER`.

        This estimator deliberately holds **no defaults of its own** for
        ``optimizer``, ``max_iter`` or ``tol``.  It used to
        (``optimizer="torch_lbfgs", max_iter=150, tol=1e-5``), and those drifted
        away from the library's, so the same data fitted through ``NSAFlow``
        and through ``nsa_flow`` gave different answers with different
        convergence claims.  ``None`` means "whatever the library decided",
        which is the only value that cannot drift.
    init : str or Tensor, default "auto"
        Initialization strategy ('clamp', 'relax', 'nmf', 'random') or tensor.
    max_iter : int, optional
        Cap on GRADIENT EVALUATIONS (see :mod:`nsa_flow.optim`).  ``None`` uses
        the library default.
    tol : float, optional
        Stationarity tolerance on the shared gradient-mapping certificate.
        ``None`` uses the working-precision default.
    center : bool, default True
        Subtract the training column means before fitting and in ``transform``
        (stored as ``mean_``).  Applied only to signed fits -- ``mode="signed"``,
        or ``mode="auto"`` on data with negative entries -- so it never turns a
        non-negative matrix signed and changes the mode it dispatches to.
    **kwargs :
        Additional arguments forwarded to `nsa_flow`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal components / loadings (transposed basis, V.T), adhering
        to the standard scikit-learn PCA/NMF convention.
    result_ : NSAResult
        The full result object containing energies, defects, and convergence diagnostics.
    converged_ : bool
        Whether the fit produced a certified solution.
    certificate_ : {"stationary", "numerical_floor", "none"}
        Which claim the fit supports; see :mod:`nsa_flow.diagnostics`.
    grad_map_ : float
        The shared scale-invariant stationarity certificate at the solution.
    n_iter_ : int
        Accepted steps taken.
    """

    def __init__(self, n_components=6, w=0.5, mode="auto", consolidate=False,
                 optimizer=None, init="auto", max_iter=None, tol=None,
                 center=True, **kwargs):
        self.center = center
        self.n_components = n_components
        self.w = w
        self.mode = mode
        self.consolidate = consolidate
        self.optimizer = optimizer
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.kwargs = kwargs
        # sklearn's clone() reconstructs an estimator from get_params(), which
        # is derived from the __init__ signature and therefore cannot see
        # **kwargs.  Surfacing them as real attributes keeps cross_val_score and
        # GridSearchCV from silently dropping them.
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_params(self, deep=True):
        """``BaseEstimator.get_params`` plus the ``**kwargs`` passthrough."""
        try:
            params = super().get_params(deep=deep)
        except AttributeError:                       # sklearn not installed
            params = {k: getattr(self, k) for k in
                      ("n_components", "w", "mode", "consolidate",
                       "optimizer", "init", "max_iter", "tol", "center")}
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        for key in list(params):
            if key in self.kwargs:
                self.kwargs[key] = params.pop(key)
                setattr(self, key, self.kwargs[key])
        if params:
            try:
                super().set_params(**params)
            except AttributeError:                   # sklearn not installed
                for key, val in params.items():
                    setattr(self, key, val)
        return self

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
        X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
        # Centre once here and store the mean, so transform() applies the same
        # shift.  Every benchmark in experiments/ was doing this by hand, which
        # is the most likely thing a user forgets.  Off for non-negative modes
        # where the sign of the data is the point.
        # Centering must never change which mode the data selects: under
        # mode="auto" non-negative data means the non-negative (data) mode, and
        # centering it would turn it signed.  So centre only when the fit is
        # signed by request or the data already is.
        signed_fit = (self.mode in ("signed", "contrast")
                      or (self.mode == "auto" and float(X_arr.min()) < 0.0))
        self.mean_ = (X_arr.mean(axis=0) if (self.center and signed_fit)
                      else np.zeros(X_arr.shape[1]))
        res = nsa_flow(
            X_arr - self.mean_, k=self.n_components, w=self.w, mode=self.mode,
            consolidate=self.consolidate, optimizer=self.optimizer,
            init=self.init, max_iter=self.max_iter, tol=self.tol,
            **self.kwargs
        )
        self.result_ = res
        self.components_ = res.Y.detach().cpu().numpy().T  # [k, p]
        # The canonical diagnostics, under the shared definitions, so a caller
        # never has to know which functional this mode happened to optimise.
        self.converged_ = bool(res["converged"])
        self.certificate_ = res["certificate"]
        self.grad_map_ = float(res["grad_map"])
        self.n_iter_ = int(res["iters"])
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
        X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
        return (X_arr - self.mean_) @ self.components_.T  # [n, k]

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
