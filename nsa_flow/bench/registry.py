"""Registries for datasets, methods, protocols and suites.

Everything the benchmark can run is registered here under a string name, so the CLI, the suites
and the standalone API all resolve names through one place.
"""
from __future__ import annotations

from typing import Callable


class Registry(dict):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def add(self, name, obj):
        if name in self:
            raise KeyError(f"{self.kind} {name!r} is already registered")
        self[name] = obj
        return obj

    def get_or_raise(self, name):
        if name not in self:
            raise KeyError(f"unknown {self.kind} {name!r}; known: {sorted(self)}")
        return self[name]


DATASETS = Registry("dataset")      # name -> DatasetSpec
METHODS = Registry("method")        # name -> Method
PROTOCOLS = Registry("protocol")    # name -> callable(spec, dataset, methods, profile, jobs) -> DataFrame
SUITES = Registry("suite")          # name -> Suite


def protocol(name: str):
    def deco(fn: Callable):
        PROTOCOLS.add(name, fn)
        return fn
    return deco
