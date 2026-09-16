"""Verbatim v1 (nsa_flow 1.5.0) modules, extracted from git 1ab26a6.

Present only so the paper can measure the v1 formulation against the corrected
one.  Not part of the installed package and not maintained.
"""
from .energy import compute_energy, defect_fast, invariant_orthogonality_defect
from .retraction import nsa_flow_retract_auto
from .optimizer import estimate_learning_rate_for_nsa_flow, get_torch_optimizer
from .flow import nsa_flow_orth

__all__ = ["compute_energy", "defect_fast", "invariant_orthogonality_defect",
           "nsa_flow_retract_auto", "estimate_learning_rate_for_nsa_flow",
           "get_torch_optimizer", "nsa_flow_orth"]
