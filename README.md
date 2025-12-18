# 🧠 NSA-Flow: Non-negative Stiefel Approximating Flow

**NSA-Flow** is a general-purpose optimization framework for interpretable representation learning.  
It unifies sparse matrix factorization, orthogonalization, and manifold constraints into a single, differentiable algorithm that operates near the Stiefel manifold.

![The NSA-flow framework](docs/nsaflow_info.png)

Documentation of functions [here](https://htmlpreview.github.io/?https://raw.githubusercontent.com/stnava/nsa_flow/main/docs/nsa_flow.html)


[Download the Project Slides](./docs/project-Tuning_Interpretability_with_Orthogonal_Flow.pdf)


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

```
⸻

🧰 Dependencies
	•	Python ≥ 3.9
	•	PyTorch ≥ 2.0
	•	NumPy ≥ 1.23
	•	Matplotlib (for optional visualization)

⸻

🚀 Quick Start


```python
import torch
import nsa_flow
torch.manual_seed(42)
# Random initialization
Y = torch.randn(120, 200)+1
print("Initial orthogonality defect:", nsa_flow.invariant_orthogonality_defect(Y))
# Run NSA-Flow optimization
result = nsa_flow.nsa_flow_orth(
    Y,
    w=0.5,
    retraction="soft_polar",
    optimizer="asgd",
    max_iter=5000,
    record_every=1,
    tol=1e-8,
    initial_learning_rate=None,
    lr_strategy='bayes',
    warmup_iters=10,
    verbose=False,
)
nsa_flow.plot_nsa_trace( result['traces'] )
print("Final orthogonality defect:", nsa_flow.invariant_orthogonality_defect(result["Y"]))
```

⸻

📖 Documentation

NSA-Flow exposes a small set of high-level functions:

Function	Description

- nsa_flow()	Main optimization loop balancing fidelity and orthogonality

- nsa_flow_retract_auto()	Retraction operator enforcing manifold constraints

- invariant_orthogonality_defect()	Computes orthogonality defect measure

- defect_fast()	Fast approximate defect metric

- nsa_flow_autograd()	Autograd-compatible variant for joint optimization

- get_torch_optimizer()	Returns a configured PyTorch optimizer


⸻

🧪 Validation

NSA-Flow has been validated in:

	•	Golub leukemia gene expression dataset

	•	Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset

NSA-Flow constraints maintain or improve performance while simplifying latent representations and improving interpretability.

There is also a layer that can be included (potentially) in deep learning tools.  See `tests/test_nsaf_layer.py`. 
This has not been used tested.
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



