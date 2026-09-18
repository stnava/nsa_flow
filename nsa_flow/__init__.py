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
from .angle import (angle_defect, cosine_matrix, grad_angle_defect,
                    gram_offdiag_defect, grad_gram_offdiag_defect)
from .project import project_nonneg, project_scaled_stiefel, polar_factor
from .solve import nsa_flow, NSAResult
from .reconstruct import (GramOperator, nsa_flow_data, reconstruction_fidelity,
                          grad_reconstruction_fidelity, relax_into_nonneg)
from .signed import consolidate_supports, nsa_flow_signed, part_sparsity
from .subspace import (SubspaceAnchor, negative_mass, subspace_fidelity,
                       grad_subspace_fidelity)
from .layers import NSAFlowLinear, NSAFlowConv2d, NSAFlowLayer
from .sklearn import NSAFlow

__version__ = "2.14.0"

__all__ = [
    # energy
    "gram", "stiefel_defect", "stiefel_defect_normalised", "grad_stiefel_defect",
    "fidelity", "grad_fidelity", "energy", "grad_energy",
    "effective_rank", "defect_floor",
    "procrustes_rotation", "aligned_target",
    # orthogonality (as against orthonormality)
    "angle_defect", "cosine_matrix", "grad_angle_defect",
    "gram_offdiag_defect", "grad_gram_offdiag_defect",
    # projections
    "project_nonneg", "project_scaled_stiefel", "polar_factor",
    # solver
    "nsa_flow", "NSAResult",
    "nsa_flow_data", "reconstruction_fidelity", "grad_reconstruction_fidelity",
    "relax_into_nonneg", "GramOperator",
    # signed lifting: V = V+ - V-, both lobes sparse
    "consolidate_supports", "part_sparsity",
    # sign-blind fidelity, for signed targets
    "SubspaceAnchor", "negative_mass", "subspace_fidelity",
    "grad_subspace_fidelity",
    # experimental; see nsa_flow/signed.py -- prefer nsa_flow_data
    "nsa_flow_signed",
    # layers
    "NSAFlowLinear", "NSAFlowConv2d", "NSAFlowLayer",
    # scikit-learn estimator
    "NSAFlow",
    "__version__",
]
