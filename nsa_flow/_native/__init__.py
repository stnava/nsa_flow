import os
import pathlib

if os.environ.get("NSA_FLOW_DISABLE_NATIVE", "0") in ("1", "true", "True"):
    raise ImportError("nsa_flow native extension disabled via NSA_FLOW_DISABLE_NATIVE")

try:
    from ._lbfgsb_cpu import *
except ImportError:
    import torch
    from torch.utils.cpp_extension import load

    src = pathlib.Path(__file__).parent / "lbfgsb_cpu.cpp"
    if not src.exists():
        raise ImportError(f"Cannot find native kernel source at {src}")

    extra_cflags = ["-O3"]
    if os.uname().sysname == "Darwin":
        extra_cflags += ["-stdlib=libc++", "-mmacosx-version-min=10.15"]

    _mod = load(
        name="_lbfgsb_cpu",
        sources=[str(src)],
        extra_cflags=extra_cflags,
        verbose=False,
    )
    for _k, _v in _mod.__dict__.items():
        if not _k.startswith("_"):
            globals()[_k] = _v
