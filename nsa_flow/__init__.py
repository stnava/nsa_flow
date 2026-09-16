"""NSA-Flow: Non-negative Stiefel-Approximating Flow.

A single calibration-free energy

    E_w(Y) = (1 - w) ||Y - X0||_F^2 / ||X0||_F^2  +  w Dtilde(Y),   Y >= 0

where ``Dtilde = D / (1 - 1/k)`` and ``D(Y) = ||Y'Y||_F^2 / ||Y||_F^4 - 1/k`` is
the squared distance from the normalised Gram matrix of ``Y`` to isotropy.
``D`` vanishes exactly on the scaled Stiefel manifold, equals
``k Var(lambda) = 1/EffectiveRank - 1/k``, and penalises rank collapse with the
floor ``D >= 1/r - 1/k`` at rank ``r``.

``w`` is a genuine convex weight: ``w = 0`` gives ``max(0, X0)`` and ``w = 1``
gives orthogonal columns, which under non-negativity means disjoint supports --
a hard clustering of the ``p`` features.  The solver is spectral projected
gradient; ``continuation`` traces the path in ``w``.
"""
from .energy import (
    gram,
    stiefel_defect,
    stiefel_defect_normalised,
    grad_stiefel_defect,
    fidelity,
    grad_fidelity,
    energy,
    grad_energy,
    effective_rank,
    defect_floor,
    procrustes_rotation,
    aligned_target,
)
from .project import project_nonneg, project_scaled_stiefel, polar_factor
from .solve import nsa_flow, NSAResult
from .layers import NSAFlowLinear, NSAFlowConv2d, NSAFlowLayer

__version__ = "2.1.0"

__all__ = [
    # energy
    "gram", "stiefel_defect", "stiefel_defect_normalised", "grad_stiefel_defect",
    "fidelity", "grad_fidelity", "energy", "grad_energy",
    "effective_rank", "defect_floor",
    "procrustes_rotation", "aligned_target",
    # projections
    "project_nonneg", "project_scaled_stiefel", "polar_factor",
    # solver
    "nsa_flow", "NSAResult",
    # layers
    "NSAFlowLinear", "NSAFlowConv2d", "NSAFlowLayer",
    "__version__",
]
