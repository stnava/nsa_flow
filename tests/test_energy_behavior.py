import torch
import matplotlib.pyplot as plt
import nsa_flow  # make sure your module is importable
from nsa_flow import nsa_flow,  nsa_flow_retract_auto, invariant_orthogonality_defect, defect_fast, nsa_flow_autograd, plot_nsa_trace
torch.manual_seed(42)
Y = torch.randn(50, 10)
X0 = torch.randn(Y.shape[0], Y.shape[1])
retraction='soft_polar'
o='lars'
###################
result = nsa_flow_autograd(
        Y,
        w=0.5, 
        retraction=retraction,
        optimizer=o,
        max_iter=50,
        record_every=1,
        tol=1e-8,
        initial_learning_rate=None,
        lr_strategy="auto",
        apply_nonneg=True,
        verbose=True,
    )
invariant_orthogonality_defect(Y)
invariant_orthogonality_defect(result['Y'])
plot_nsa_trace( result['traces']   )
