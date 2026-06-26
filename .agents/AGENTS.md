# nsa_flow Project Rules & Development Guidelines

This file outlines workspace-specific rules and algorithmic guidance for developing, testing, and optimizing the `nsa_flow` library.

## 1. Testing and Execution Guidelines
* **PYTHONPATH Configuration**: Always run tests and scripts using the local project directory as python path to avoid importing cached or global system installations of `nsa_flow`:
  ```bash
  PYTHONPATH=. pytest
  ```
* **BypassSandbox Requirement**: When running `pytest` or other system/interpreter-dependent tools, execution within the standard sandbox will fail due to access boundaries on system Python modules. Run commands with `BypassSandbox: true` to succeed.

## 2. Algorithmic Guidance (Retractions)
* **Newton-Schulz Iterations (`ns_iter`)**:
  * Newton-Schulz-based polar retraction (`soft_newton_schulz`) is SVD-free and avoids gradient collapses (`NaN`s) on GPU workloads.
  * To guarantee SVD-level mathematical exactness (under high orthogonality weights $w \ge 0.5$), Newton-Schulz **must be run with `ns_iter=10`**. Running with `ns_iter=5` converges poorly, while $\ge 50$ iterations add redundant overhead with zero added accuracy.
* **Cayley Retractions**:
  * Cayley-based retractions (`soft_cayley`) provide superior energy minimization profiles under balanced weights and are extremely fast for skinny matrices ($k \ll p$).
* **Library Defaults**:
  * Keep SVD-based `soft_polar` as the default retraction throughout the package to ensure out-of-the-box mathematical exactness under general usage.
