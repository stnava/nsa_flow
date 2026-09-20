import os
import sys
from setuptools import setup

try:
    from torch.utils.cpp_extension import CppExtension, BuildExtension

    extra_compile_args = ["-O3"]
    if sys.platform == "darwin":
        extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.15"]

    ext_modules = [
        CppExtension(
            name="nsa_flow._native._lbfgsb_cpu",
            sources=["nsa_flow/_native/lbfgsb_cpu.cpp"],
            extra_compile_args=extra_compile_args,
            optional=True,
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
except Exception:
    ext_modules = []
    cmdclass = {}

if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )
