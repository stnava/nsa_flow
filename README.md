# 🧠 NSA-Flow: Non-negative Stiefel Approximating Flow

**NSA-Flow** is a general-purpose optimization framework for interpretable representation learning.  
It unifies sparse matrix factorization, orthogonalization, and manifold constraints into a single, differentiable algorithm that operates near the Stiefel manifold.

---

## ✨ Overview

Interpretable representation learning remains a core challenge in high-dimensional domains such as neuroimaging, genomics, and text analysis.  
**NSA-Flow** provides a smooth geometric mechanism for balancing **reconstruction fidelity** and **column-wise decorrelation**, producing sparse, stable, and interpretable representations.

NSA-Flow enforces structured sparsity via a single tunable weight parameter, combining:
- Continuous orthogonality control via manifold retraction (e.g., soft-polar, polar)
- Non-negativity via proximal updates
- Adaptive gradient scaling and learning-rate control

---

## 🧩 Key Features

- ⚙️ **Continuous flow near the Stiefel manifold**
- 🧮 **Non-negative and orthogonal constraints**
- 🧠 **Interpretable latent representations**
- 🚀 **Compatible with PyTorch optimization routines**
- 🧬 **Validated on neuroimaging and genomics datasets**

---

## 📦 Installation

Install from PyPI (once published):

```bash
pip install nsa_flow

Or install the latest development version directly from GitHub:

pip install git+https://github.com/stnava/nsa_flow.git


⸻

🧰 Dependencies
	•	Python ≥ 3.9
	•	PyTorch ≥ 2.0
	•	NumPy ≥ 1.23
	•	Matplotlib (for optional visualization)

⸻

🚀 Quick Start

import torch
import nsa_flow
from nsa_flow import nsa_flow, invariant_orthogonality_defect

torch.manual_seed(42)

# Random initialization
Y = torch.randn(50, 10)
X0 = torch.randn_like(Y)

# Run NSA-Flow optimization
result = nsa_flow(
    Y,
    X0=X0,
    w=0.8,
    retraction="soft_polar",
    optimizer="sgdp",
    max_iter=50,
    record_every=1,
    tol=1e-8,
    initial_learning_rate=1e-2,
    verbose=True,
)

print("Initial orthogonality defect:", invariant_orthogonality_defect(Y))
print("Final orthogonality defect:", invariant_orthogonality_defect(result["Y"]))


⸻

📖 Documentation

NSA-Flow exposes a small set of high-level functions:

Function	Description
nsa_flow()	Main optimization loop balancing fidelity and orthogonality
nsa_flow_retract_auto()	Retraction operator enforcing manifold constraints
invariant_orthogonality_defect()	Computes orthogonality defect measure
defect_fast()	Fast approximate defect metric
nsa_flow_autograd()	Autograd-compatible variant for joint optimization
get_torch_optimizer()	Returns a configured PyTorch optimizer


⸻

🧪 Validation

NSA-Flow has been validated in:
	•	Golub leukemia gene expression dataset
	•	Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset

NSA-Flow constraints maintain or improve performance while simplifying latent representations and improving interpretability.

⸻

🧑‍💻 Citation

If you use NSA-Flow in research, please cite:

Stnava et al. (2025). NSA-Flow: Non-negative Stiefel Approximating Flow for Interpretable Representation Learning.

⸻

⚖️ License

MIT License © 2025 

⸻

📫 Contact

For issues, feature requests, or contributions, open an issue on
GitHub.

---



