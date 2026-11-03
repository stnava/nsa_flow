# What next — decision memo (updated 2026-11-03, v3.6.0)

## Done since the 2026-09-19 memo
- **On-device GPU execution & D2H scalar-sync elimination (2026-11-03, v3.6.0):** exposed
  `device=None` and `dtype=None` across `nsa_flow`, `nsa_flow_data`, `nsa_flow_signed`, and
  `_nsa_flow_anchored` (`coerce_tensor` in `nsa_flow/diagnostics.py`, including MPS `float64`
  coercion and `float16`/`bfloat16` pure-PyTorch dispatch); eliminated intermediate host copies
  and per-step D2H scalar syncs (`torch.topk` replacing `torch.nonzero` in `_cauchy_point`,
  batched curvature/Wolfe/certificate checks in `lbfgsb.py`, branchless BB & `finfo.tiny` guards
  in `optim.py`, de-synced `_smooth_descent` in `reconstruct.py`, fused `jacobi_eigh` sweep norms
  and accelerator subspace iteration for `p > 512` Gram matrices in `linalg.py`); added 47 device
  and transfer-free tests in `tests/test_device.py` (508 total assertions passing) and multi-scale
  CPU vs MPS benchmark (`experiments/bench_device_performance.py`).
- **Manuscript audit, sync & Rmd deprecation (2026-11-01, v3.5.2):** reconciled `paper/nsaflow.tex`
  with `v3.5.1` code and `nsa_flow.bench` findings (removed Tecator Conclusion contradiction,
  replaced ADNI RF explanation with PCA-rotated axis-alignment control, aligned Eq. 4 with
  monotone Armijo `_spg`, updated matched-multistart module-recovery and oblique FA rotation
  metrics, qualified un-nested fixed-$k$ Prostate and 9-outcome cognitive battery claims, synced
  461-test count plus Orthogonal Varimax / Kaiser escape / `simlr_sparseness` documentation),
  deprecated legacy `paper/nsa_flow.Rmd` (`rmd-paper` in `Makefile`), and rebuilt `paper/nsaflow.pdf`.
- **Quantile-sparseness tie-breaking & Hadamard saddle escape (2026-11-01, v3.5.1):** fixed
  rank-1 collapse on tied/Hadamard bases in `simlr_sparseness` and `_resolve_column_sign_gauge`
  (`nsa_flow/project.py`), added Kaiser 2-column Givens saddle escape in `varimax_rotation`
  (`nsa_flow/linalg.py`), switched post-solve support selection in `solve.py`, `reconstruct.py`,
  and `signed.py` to `alg="hard"` (preventing `float32` soft-shrunk false stationarity), and
  added 5 regression tests in `tests/test_solver.py`.
- **A. Native kernel (CPU):** fused C++ L-BFGS-B iteration shipped v3.2.0–3.3.2 (6–11×,
  energy parity < 1e-12, `NSA_FLOW_DISABLE_NATIVE=1` falls back). Plan: `docs/NATIVE_KERNEL_PLAN.md`.
  Not done: a fused MPS/CUDA path (dispatch-bound, ~3 ms/gradient on MPS).
- **B. Nested CV of `w`, `k`:** `experiments/nested_cv.py` (`make nested`), equally tuned
  baselines, tests in `tests/test_nested_cv_no_leak.py`. Result (README §Evaluation): the ADNI
  CDRSB forest effect survives (+0.147 [0.078, 0.216]); the Golub gain does not (+0.03, CI
  includes 0). Paper abstract/results/limitations updated to say so.
- Makefile: `make nested`, `make speed`.

- **Ground-truth module benchmarks (2026-10-19):** null controls, Tecator contiguity,
  20 Newsgroups, METABRIC (`experiments/null_controls.py`, `tecator_contiguity.py`,
  `newsgroups_modules.py`, `metabric_modules.py`). NSA recovers planted structure and is not
  fooled by noise, but only ties k-means/NMF on real data (README §Evaluation).

## Open
0. **Benchmarks now live in `nsa_flow.bench`** (`nsa-flow-bench`, `make bench`; docs/BENCHMARKING.md). Standard run
   findings: ADNI forest gain survives nested CV; Golub does not; Tecator loses under forests; signed consolidated NSA
   loses to k-means on module recovery, the non-negative mode (`mode="data"`) is competitive. Follow-ups: (a) a
   non-negative *consolidated* mode; (b) choose `w` by stability for module tasks; (c) real pathway truth (MSigDB
   hallmark) instead of hand-typed lists; (d) port rapid_primary_benchmarks, benchmark_new_public_data,
   speed_vs_pca, adni_cdrsb_repeated, then shim them (PPMI is excluded: not public, not reproducible); (e) Golub point estimate differs
   from the earlier standalone script (+0.016 vs +0.032, both CIs include 0): cause not identified.
1. **Claims not yet nested-CV'd:** Prostate (0.824 vs 0.805), the 9-outcome cognitive battery,
   and trinary-diagnosis AUC (now explicitly qualified as fixed-$k$ single-setting benchmarks in
   `paper/nsaflow.tex`; Tecator's nested-CV linear parity and forest loss are now reconciled
   throughout `paper/nsaflow.tex`). Each remaining claim needs the nested-CV harness or should be cut.
2. **Replication:** ADNI is one restricted-access cohort. Replicate on public data only (PPMI is excluded: not public, not
   reproducible); NHANES and the public benchmarks in `nsa_flow.bench` are the replication tier.
3. **True orthogonal NMF** baseline (multiplicative updates) — current NMF baseline is plain NMF;
   k-means features covers the disjoint extreme.
4. Golub at full grid / more splits if the NSA grid is wanted at parity with ADNI (~20 min).
5. float32 `tol`/`patience`; MPS Jacobi eigensolver speed at n>200 (only if an MPS use case exists).
6. Remove `torch_lbfgs` at v4.0.
