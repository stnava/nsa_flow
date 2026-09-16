# nsa_flow project rules

## Running things
* Always run with the local project on the path, so a globally installed
  `nsa_flow` is not picked up instead:
  ```bash
  PYTHONPATH=. pytest
  ```
* `make test` (133 assertions), `make theory` (the 70-assertion property
  battery), `make experiments`, `make paper`.
* Experiments write to `paper/results/` and `paper/figs/`. Both are generated;
  never hand-edit them.

## The invariant that matters
`tests/test_theory.py` is the executable statement of the paper's propositions.
Every claim the paper makes about the energy is asserted there. If you change
`nsa_flow/energy.py`, the battery is the thing that says whether the theory still
holds. Do not weaken an assertion to make a change pass — if a property has to
go, the paper has to change too.

## Formulation (do not re-derive it wrong)
* The orthogonality term is `D(Y) = ||G - I/k||_F^2` with `G = Y'Y/tr(Y'Y)`.
  Compute it in that form, not as `||Y'Y||_F^2/||Y||_F^4 - 1/k` — the latter
  cancels catastrophically and returns small negatives near the optimum.
* `D` splits into a decorrelation term plus a norm-balance term. The
  decorrelation term alone is the old v1 "defect"; it is blind to conditioning,
  not `O(k)`-invariant, and minimised by rank-deficient matrices. Do not
  reintroduce it.
* Both terms of `E_w` are dimensionless and `O(1)` **by construction**. If you
  find yourself adding a constant estimated from the data to balance them, that
  is the v1 bug returning — the calibration test in the battery covers every
  `fidelity`/`orth` option pair for exactly this reason.
* `<grad D, Y> = 0` identically, so `D` cannot change `||Y||`. Never add a
  renormalisation step "to preserve scale"; scale is the fidelity term's job.

## Numerics
* No SVD, eigendecomposition or QR belongs in the solver's inner loop. The loop
  is one Gram product plus two `[p,k] x [k,k]` products.
* Never differentiate `torch.linalg.svd` to get a polar factor. Its backward
  divides by `sigma_i^2 - sigma_j^2` and returns NaN for *every* entry at
  repeated singular values, which is exactly what `nn.init.orthogonal_`
  produces. Use `nsa_flow.polar_factor`, which carries the Sylvester derivative.
* Report `stop_reason` and `grad_map` rather than asserting convergence. A run
  that hit `max_iter` must not claim to have converged.

## Experiments
* Fit everything inside the CV fold — scaling, feature selection, covariate
  adjustment, components. Golub has n=72; fitting components on the full matrix
  leaks through the scaling and is why such numbers do not replicate.
* Sparse PCA cost grows superlinearly in `p` (112 s at p=7129, 1.4 s at
  p=2000), which is why the Golub experiment applies an in-fold top-variance
  gene filter.
* ADNI: use the right hemisphere. The left-hemisphere columns of
  `adniGrayMatterVolumesDktANTsSST.csv` have systematic label dropout (65% of
  subjects affected). This is a defect in that derived table, not in the method.
* NSA-Flow does **not** beat PCA on predictive accuracy on either real dataset.
  Do not let a refactor quietly reintroduce that claim.

## History
`attic/` holds the v1 package, tests and scripts, kept for reference and for the
paper's ablation. `experiments/v1/` is a verbatim copy of the v1 modules used to
generate the comparison tables; it is not maintained and not installed.
