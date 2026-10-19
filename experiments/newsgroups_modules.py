"""DEPRECATED shim: this benchmark now lives in `nsa_flow.bench`.

Equivalent:  nsa-flow-bench run --suite standard --only newsgroups --jobs 4
(see `nsa-flow-bench list`, README \"Benchmarking\").  Arguments to this script are ignored.
"""
import sys
import warnings

warnings.warn("experiments/newsgroups_modules.py is deprecated; use nsa-flow-bench run --suite standard --only newsgroups",
              DeprecationWarning, stacklevel=1)

if __name__ == "__main__":
    from nsa_flow.bench.cli import main
    sys.exit(main(["run", "--suite", "standard", "--only", "newsgroups", "--jobs", "4"]))
