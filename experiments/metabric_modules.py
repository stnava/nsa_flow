"""DEPRECATED shim: this benchmark now lives in `nsa_flow.bench`.

Equivalent:  nsa-flow-bench run --suite standard --only metabric --jobs 4
(see `nsa-flow-bench list`, README \"Benchmarking\").  Arguments to this script are ignored.
"""
import sys
import warnings

warnings.warn("experiments/metabric_modules.py is deprecated; use nsa-flow-bench run --suite standard --only metabric",
              DeprecationWarning, stacklevel=1)

if __name__ == "__main__":
    from nsa_flow.bench.cli import main
    sys.exit(main(["run", "--suite", "standard", "--only", "metabric", "--jobs", "4"]))
