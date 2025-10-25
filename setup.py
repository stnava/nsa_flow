from setuptools import setup, find_packages

setup(
    name="nsa_flow",
    version="0.1",
    packages=find_packages(),
    install_requires=["torch>=2.0"],
    description="Nonnegative Stiefel Approximation (NSA-Flow) in PyTorch",
    author="Brian B. Avants",
)

