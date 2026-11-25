"""``nsa_flow.bench``: the official benchmarking package for NSA-Flow.

Datasets, methods, protocols, metrics, pre-registered suites, results and reports in one place, usable
standalone (``evaluate(X, y, groups)``) and integrated (``nsa-flow-bench run --suite standard``).
Needs the ``[bench]`` extra (numpy, pandas, scikit-learn, scipy, joblib); ``import nsa_flow`` does not.
"""
_LAZY = {"evaluate": "api", "run_suite": "runner", "get_dataset": "datasets", "Dataset": "datasets",
         "METHODS": "registry", "DATASETS": "registry", "SUITES": "registry", "PROTOCOLS": "registry"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        for m in ("datasets", "methods", "protocols", "suites"):
            importlib.import_module(f"{__name__}.{m}")
        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(name)


__all__ = list(_LAZY)
