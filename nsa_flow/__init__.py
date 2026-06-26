from .utils import (
    safe_to_tensor,
    apply_nonnegativity,
    traces_to_dataframe,
    plot_nsa_trace,
    run_single_experiment,
    evaluate,
    plot_evaluation_summary,
)
from .energy import (
    invariant_orthogonality_defect,
    defect_fast,
    fidelity_basic,
    fidelity_scaled,
    fidelity_symmetric,
    compute_energy,
    energy_fidelity,
)
from .retraction import (
    inv_sqrt_sym_adaptive,
    nsa_flow_retract_newton_schulz,
    nsa_flow_retract_cayley,
    nsa_flow_retract_auto,
)
from .optimizer import (
    get_torch_optimizer,
    get_lr_estimation_strategies,
    estimate_learning_rate_for_nsa_flow,
)
from .flow import (
    nsa_flow,
    nsa_flow_autograd,
    nsa_flow_orth,
)
from .layers import (
    SimpleMLP,
    NSAFlowLayer,
    NSAFlowLinear,
    NSAFlowConv2d,
)
from . import legacy

__all__ = [
    # utils
    "safe_to_tensor",
    "apply_nonnegativity",
    "traces_to_dataframe",
    "plot_nsa_trace",
    "run_single_experiment",
    "evaluate",
    "plot_evaluation_summary",
    # energy
    "invariant_orthogonality_defect",
    "defect_fast",
    "fidelity_basic",
    "fidelity_scaled",
    "fidelity_symmetric",
    "compute_energy",
    "energy_fidelity",
    # retraction
    "inv_sqrt_sym_adaptive",
    "nsa_flow_retract_newton_schulz",
    "nsa_flow_retract_cayley",
    "nsa_flow_retract_auto",
    # optimizer
    "get_torch_optimizer",
    "get_lr_estimation_strategies",
    "estimate_learning_rate_for_nsa_flow",
    # flow
    "nsa_flow",
    "nsa_flow_autograd",
    "nsa_flow_orth",
    # layers
    "SimpleMLP",
    "NSAFlowLayer",
    "NSAFlowLinear",
    "NSAFlowConv2d",
    # legacy
    "legacy",
]
