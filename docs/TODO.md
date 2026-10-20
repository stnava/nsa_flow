# What next — decision memo (updated 2026-10-19, v3.5.0)

## Done since the 2026-09-19 memo
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
1. **Claims not yet nested-CV'd:** Prostate (0.824 vs 0.805), Tecator (README shows NSA *worse*
   than PCA: forest R² 0.699 vs 0.913, which contradicts the paper's "spectroscopy win"), the 9-outcome
   cognitive battery, trinary-diagnosis AUC. Each needs the same harness or the claim should be cut.
2. **Replication:** ADNI is one restricted-access cohort. Replicate on public data only (PPMI is excluded: not public, not
   reproducible); NHANES and the public benchmarks in `nsa_flow.bench` are the replication tier.
3. **True orthogonal NMF** baseline (multiplicative updates) — current NMF baseline is plain NMF;
   k-means features covers the disjoint extreme.
4. Golub at full grid / more splits if the NSA grid is wanted at parity with ADNI (~20 min).
5. float32 `tol`/`patience`; MPS Jacobi eigensolver speed at n>200 (only if an MPS use case exists).
6. Remove `torch_lbfgs` at v4.0.
