# NSA-Flow: Non-negative Stiefel-Approximating Flow

A calibration-free relaxation between reconstruction and feature clustering.

```python
import torch
from nsa_flow import nsa_flow

X0 = torch.rand(120, 8)          # a loading matrix to refine, e.g. |PCA loadings|
r  = nsa_flow(X0, w=0.7)         # w is a genuine convex weight in [0, 1]

print(r)
# NSAResult(w=0.7, iters=61, energy=..., fidelity=..., defect=...,
#           eff_rank=7.41, stop=grad_map, |Gmap|=4.2e-13)
print(r.Y.shape)                 # (120, 8), non-negative, near-orthogonal columns
```

## What it computes

Minimise, over `Y >= 0`,

```
E_w(Y) = (1 - w) * ||Y - X0||_F^2 / ||X0||_F^2   +   w * D(Y) / (1 - 1/k)
```

where, with `G = Y'Y / tr(Y'Y)` the trace-normalised Gram matrix,

```
D(Y) = || G - I/k ||_F^2 = ||Y'Y||_F^2 / ||Y||_F^4 - 1/k
```

Both terms are dimensionless and `O(1)`, so **`w` needs no calibration**: no
normalising constants estimated from the data, no warm-up phase, no dependence on
`p`, `k` or the scale of `X0`. A reported `w` means the same thing on every
problem.

### Why this `D`

`D` is the squared distance from the normalised Gram matrix to isotropy. It

- vanishes **exactly** on `R_{>0} * St(p, k)` — orthogonal columns of *equal* norm;
- equals `k * Var(eigenvalues of G)` = `1/EffectiveRank - 1/k`, so minimising it
  drives the effective rank to `k`;
- is invariant under `Y -> cY`, `Y -> UY` and `Y -> YV` for orthogonal `U, V` —
  the same group that preserves the constraint being relaxed;
- charges at least `1/r - 1/k` for rank `r < k`, so rank collapse is penalised;
- is bounded: `0 <= D <= 1 - 1/k`;
- satisfies `<grad D, Y> = 0`, so it **cannot change `||Y||`** — scale is pinned by
  the fidelity term alone, and no renormalisation step is needed.

It splits as

```
D  =  sum_{i != j} G_ij^2   +   sum_i (G_ii - 1/k)^2
      \_______________/         \___________________/
        decorrelation              norm balance
```

The first term alone is the "invariant orthogonality defect" used in v1 and
elsewhere. Dropping the second is what made that functional blind to
conditioning, basis-dependent, and minimised by rank-deficient matrices.

### What `w` does

`w = 0` returns `max(0, X0)`. As `w -> 1`, `D -> 0`, and since `Y >= 0` with
`Y'Y` diagonal forces **pairwise disjoint column supports**, the limit is a hard
clustering of the `p` features into `k` groups — the feasible set of orthogonal
NMF, equivalent to k-means. In between, the overlap is bounded:

```
max_{i != j} <y_i, y_j>  <=  sqrt(D) * ||Y||_F^2
```

so "approximately disjoint factors" is a claim with a number attached.

## Solver

Pure-PyTorch L-BFGS-B with optional native C++ kernel acceleration (see below); every accepted step is certified
by one shared gradient-mapping norm, `result.grad_map`, and `result.stop_reason`
/ `result.certificate` say exactly what was and was not established — never a
silent claim of convergence.

The inner loop forms one Gram product and two `[p,k] x [k,k]` products: `O(p k^2)`,
with **no SVD, eigendecomposition or QR**. Typical convergence is 15–300
deterministic iterations.

### Native C++ Kernel Acceleration (v3.2.6)

When compiled, `nsa_flow._native._lbfgsb_cpu` executes fused iterations in C++ on CPU,
eliminating Python interpreter and ATen dispatch overhead while achieving bit-exact numerical parity
(`< 1e-12` energy difference) with pure PyTorch.

- **Speedup:** 5–12× speedup on standard problem shapes:
  - Tall small data (`500 x 50, k=5`): 11.8× speedup (7.3 ms vs 86.6 ms; 49.8 µs/iter vs 601.7 µs/iter).
  - ADNI cortical signed (`64 x 10, k=10`): 11.1× speedup (9.1 ms vs 101.1 ms; 61.4 µs/iter vs 683.1 µs/iter).
  - Tall medium data (`1000 x 100, k=10`): 5.0× speedup (20.9 ms vs 104.9 ms; 142.4 µs/iter vs 723.7 µs/iter).
  - Wide genomics data (`50 x 2000, k=5`): 1.8× speedup (185.5 ms vs 334.3 ms; 1.25 ms/iter vs 2.26 ms/iter).
- **Direct CBLAS & Zero Inner Allocations:** Cauchy point, subspace minimization, compact quasi-Newton $M$ inversion, and line search operate on preallocated workspace buffers and direct CBLAS/LAPACK calls with stack solvers for small-to-moderate dimensions.
- **v3.2.6 Optimizations:**
  - *CBLAS SYRK:* Rank-k symmetric updates replace general matrix products for $A = X_V^T X_V$, $B = V^T V$, and $BW = W^T W$, cutting FLOPs by 50% for quadratic terms.
  - *Zero-Copy Free Subspaces:* Direct contiguous pointer indexing for $W_{\text{free}}$ when all coordinates are free ($n_{\text{free}} == n$).
  - *Lazy Line-Search Copies:* Defers workspace buffer copying until trial 1 fails Wolfe conditions, eliminating overhead on >85% of iterations.
  - *Noise-Floor Stall Gating:* Anchors stall classification to `max(stall_slack * tol, sqrt(eps))` to certified plateaus at the float precision floor without false certification on far points.
- **Pure-Python fallback:** To force pure-Python execution, set `NSA_FLOW_DISABLE_NATIVE=1`. All reference instances match to $< 10^{-12}$.

Empirically `E_w` has a unique optimum for `w < 1` — 24 random restarts agree to
machine precision on every problem family tested — so there are no restarts,
schedules or step-size heuristics to tune.

## Benchmarking (`nsa_flow.bench`)

One package owns datasets, methods, protocols, metrics, pre-registered suites, results and reports.
Install the extra and use it from the CLI or from Python:

```bash
pip install -e ".[bench]"
nsa-flow-bench list                       # datasets (ok / MISSING + why), methods, protocols, suites
nsa-flow-bench fetch pbmc3k --download    # public data downloads only on explicit request, checksummed
nsa-flow-bench run --suite smoke --profile smoke      # seconds, synthetic only
nsa-flow-bench run --suite standard --jobs 4          # the reported benchmarks -> bench_results/
nsa-flow-bench report standard                        # markdown from the stored rows
nsa-flow-bench check                                  # smoke suite vs stored baseline (solver regression guard)
```

```python
from nsa_flow.bench import evaluate            # standalone: your own arrays, same protocols and schema
ev = evaluate(X, y=y, groups=feature_group_ids, profile="quick")   # y -> nested CV; groups -> module recovery
print(ev.report())
```

- **Protocols.** `nested_cv` (every method, PCA included, tunes its own hyper-parameters on the training portion
  by inner CV; a fixed `w=0.5, k=5` row is reported beside it), `module_recovery` (feature modules against known
  groups with unlabelled background present: ARI, split-half stability, pair recall/separation, contiguity),
  `speed`. Null controls are the same protocols on derived data (`adni@null_y`, `metabric@shuffled`, `noise`).
- **Pre-registration.** Each suite entry fixes its primary contrast (method, reference, metric, model) and pass rule
  before running (`nsa_flow/bench/suites.py`); the run manifest stores the suite, the profile, versions, git sha and
  dataset fingerprints. The report prints primary and secondary results separately; verdicts are PASS / FAIL /
  WORSE / INFO / SKIPPED.
- **Solver certificate.** Every NSA fit records `converged` and `defect_D` in its row; the report prints convergence
  and the worst defect per method.
- **Data tiers.** Each dataset is `public`, `restricted` (ADNI, data-use agreement) or `local` (METABRIC cache);
  `--public-only` runs public data only, and missing data is skipped with a stated reason. PPMI is excluded (not public,
  not reproducible). Full guide: `docs/BENCHMARKING.md`. Hand-typed gene/marker lists are the authors' annotation and are flagged as such in the report.
- **Cost.** A time estimate from measured single-fit probes is printed before each entry; `--budget-min` skips entries
  over budget visibly, and methods that are too slow for a dataset (sklearn `SparsePCA` above p=500) are skipped with
  the reason in the results.
- **Extending.** A new dataset is a loader decorated with `@dataset(...)` returning a `Dataset`; a new method is a
  `Method(fit, grids, ...)` registered in `methods.py`; both show up in the CLI and suites automatically.

### Latest standard run
<!-- bench:standard:start -->
# nsa_flow.bench report: standard
profile `standard` · nsa_flow 3.3.4 (git cfedf20) · torch 2.13.0 · sklearn 1.9.0 · 2026-10-19T11:58:13

## Pre-registered primary contrasts
| dataset | tier | protocol | contrast (metric, model) | estimate [95% CI] | expect | status |
|---|---|---|---|---|---|---|
| adni | restricted | nested_cv | NSA consolidated − PCA (r2, forest) | 0.147 [0.078, 0.216] | effect | PASS |
| golub | public | nested_cv | NSA consolidated − PCA (balanced_accuracy, linear) | 0.016 [-0.076, 0.107] | effect | FAIL |
| tecator | public | nested_cv | NSA consolidated − PCA (r2, forest) | -0.242 [-0.293, -0.192] | effect | WORSE |
| planted | public | module_recovery | NSA consolidated w=0.5 − KMeans features (ari) | 0.377 [0.319, 0.436] | effect | PASS |
| newsgroups | public | module_recovery | NSA consolidated w=0.5 − KMeans features (ari) | -0.111 [-0.176, -0.046] | effect | WORSE |
| metabric | local | module_recovery | NSA consolidated w=0.5 − KMeans features (ari) | -0.160 [-0.213, -0.107] | effect | WORSE |
| pbmc3k | public | module_recovery | NSA consolidated w=0.5 − KMeans features (ari) | -0.163 [-0.205, -0.122] | effect | WORSE |
| tecator | public | module_recovery | NSA consolidated w=0.5 − KMeans features (contiguity) | 0.000 [-0.002, 0.002] | info | INFO |
| adni@null_y | restricted | nested_cv | NSA consolidated − PCA (r2, forest) | 0.032 [-0.028, 0.091] | null | PASS |
| noise | public | module_recovery | NSA consolidated w=0.5 (mean vs 0) (stability) | -0.002 [-0.012, 0.008] | null | PASS |
| metabric@shuffled | local | module_recovery | NSA consolidated w=0.5 (mean vs 0) (stability) | 0.000 [-0.000, 0.001] | null | PASS |

## Skipped (explicit)
- golub / nested_cv / SparsePCA: SparsePCA skipped: p=7129 > 500 (cost; sklearn SparsePCA is ~18 s/fit at p=2000)

Solver health:

| method | NSA fits | converged | max defect_D |
|---|---|---|---|
| NSA consolidated | 140 | 100% | 0.02 |
| NSA consolidated (fixed w=.5,k=5) | 140 | 100% | 1.8e-12 |
| NSA consolidated w=0.25 | 460 | 100% | 4.5e-07 |
| NSA consolidated w=0.5 | 460 | 100% | 1.8e-06 |
| NSA nonneg w=0.5 | 360 | 100% | 0.00082 |
| NSA signed | 100 | 97% | 0.02 |
| NSA signed (fixed w=.5,k=5) | 100 | 100% | 7e-05 |

<!-- bench:standard:end -->

## Evaluation results (v3.0.0, certified solver; superseded by the generated block above where they overlap)

All numbers below were produced by the pure-PyTorch L-BFGS-B default with the
shared certificate, on bases fitted **inside** every training fold or split.
Earlier releases fitted the Golub basis on all 72 samples and reported the
ADNI result from one 80/20 split of an unconverged solve; both are corrected
here and the differences are stated.

**Golub leukemia, 3 classes** (B-ALL / T-ALL / AML; p=2000, k=3; 5-fold
stratified CV, balanced accuracy; `experiments/rapid_primary_benchmarks.py`)

| basis | linear | forest | fit time (5 folds) |
|---|---|---|---|
| PCA | 0.698 | 0.660 | — |
| NSA signed, w=0 | 0.698 | 0.660 | 2.9 s |
| NSA signed, w=0.5 | **0.748** | **0.681** | 13.6 s |
| NSA signed + consolidate, w=0.5 | 0.735 | 0.639 | 14.1 s |

At w=0 the signed lifting is PCA and now returns it exactly (the previous
solver reported `defect_D = 0.10` there). The old transductive numbers were
0.06–0.07 higher across the board; that was leakage.

**ADNI cortical thickness → CDRSB** (n≈300, p=66, k=5; **20 repeated 80/20
splits**, ΔR² against PCA on the same splits, 95% paired-t interval;
`experiments/adni_cdrsb_repeated.py`)

| basis (w=0.5) | ΔR² forest [95% CI] | ΔR² linear [95% CI] |
|---|---|---|
| NSA signed + consolidate | **+0.123 [0.065, 0.182]** | +0.023 [0.003, 0.043] |
| NSA signed | +0.094 [0.042, 0.146] | +0.023 [0.008, 0.039] |
| NSA data | +0.092 [0.016, 0.169] | −0.021 [−0.043, 0.001] |
| NSA anchored | +0.065 [0.008, 0.122] | +0.002 [−0.002, 0.007] |

Split-to-split SD of R² is 0.24–0.31, so the single-split value in earlier
releases (+0.18 forest, from a solve at `defect_D = 0.006`) was not evidence
either way. The effect survives with a converged solver: smaller, and now
with an interval.

**What the standard run shows** (`make bench`; full tables via `nsa-flow-bench report standard`; the block above is
generated, the sentences below summarise it and can lag it):
- *Prediction.* The ADNI CDRSB forest gain over tuned PCA survives nested CV of `w` and `k` (+0.147 [0.078, 0.216];
  fixed `w=0.5, k=5`: +0.155) and is not produced by selection (permuted outcome: +0.032 [−0.028, 0.091]). Golub
  is not distinguishable from tuned PCA (+0.016 [−0.076, 0.107], linear). On Tecator there is no linear gain and a clear
  forest loss (−0.242 [−0.293, −0.192]).
- *Module recovery.* NSA recovers a planted partition exactly and invents nothing from noise or column-shuffled data.
  The signed consolidated mode (the pre-specified primary) is worse than k-means on 20 Newsgroups, METABRIC and PBMC3k
  (ARI −0.11 to −0.16, intervals exclude 0). The non-negative mode (`mode="data"`) is competitive: more stable than
  k-means on newsgroups and PBMC3k, higher pair recall there, more stable than NMF on all three; it is worse than
  k-means on METABRIC stability/recall and on PBMC3k ARI. It is sensitive to input scaling (shifted raw data here;
  standardised-then-shifted gave PBMC3k ARI 0.64 earlier).
- *Cost.* NSA is far slower than PCA (e.g. n=1500, p=1000, k=5: 4.4 s signed, 10.8 s non-negative, 0.009 s PCA).
- Group lists and marker lists are the authors' annotation; review before citing.

**Public data** (`experiments/benchmark_new_public_data.py`, 5-fold; defect
column is `defect_D` for every method, comparable to PCA)

| dataset | PCA | NSA signed w=0 | NSA signed w=0.5 | NSA consolidate w=0.5 |
|---|---|---|---|---|
| Sonar (AUC) | 0.819 | 0.819 | 0.810 | 0.812 |
| Prostate (AUC) | 0.896 | 0.896 | 0.890 | 0.884 |
| Tecator (forest R²) | 0.913 | — | 0.699 | 0.662 |

Signed w=0 equals PCA exactly on both classification sets; the previous solver
reported `defect_D` of 0.04 (Sonar) and 0.50 (Prostate) at that setting. Fits
are 10–20× faster than in 2.15 (Prostate: 12.6 s → 1.3 s per fit).

**Speed against PCA** (`top_k_eigenvectors`, exact, on-device; float64 CPU,
planted structure, w=0.5, default tolerance; v3.1.0, measured on an otherwise idle machine)

| shape | mode | PCA | NSA-Flow | gradients | ms / gradient |
|---|---|---|---|---|---|
| 300×66, k=5 (ADNI-like) | signed | 0.4 ms | 93 ms | 159 | 0.58 |
| 300×66, k=5 | data | 0.4 ms | 93 ms | 195 | 0.48 |
| 57×2000, k=3 (Golub-like) | signed | 0.4 ms | 681 ms | 329 | 2.1 |
| 57×2000, k=3 | data | 0.4 ms | 506 ms | 371 | 1.4 |
| 500×200, k=10 | signed | 2.6 ms | 480 ms | 416 | 1.2 |

v3.1.0 is ~3× faster than v3.0.0 on every row (ADNI-like 262 → 93 ms,
Golub-like signed 2080 → 681 ms). Two changes did it, neither of them
`torch.compile`: the L-BFGS-B subspace step is branchless (no per-iteration
host syncs), and the signed lifting now uses the matrix-free `X'(XV)` route
when p > n instead of forming the p×p Gram — the data mode already did.
`torch.compile` on the pure-tensor pieces was measured at +5–9% and is
available via `nsa_flow.lbfgsb.set_compile(True)` but off by default.

NSA-Flow is an iterative constrained method; PCA is one factorisation. A fit
on the paper's imaging shape is a tenth of a second. The objective is ~10% of
each iteration; the rest is L-BFGS-B's active-set bookkeeping in Python
(~0.5–2 ms/iteration of tensor dispatch), which is also why the same code on
MPS runs at ~3 ms/gradient with zero host↔device copies. It is **not** within
an order of magnitude of PCA on small problems and this README does not claim
it is. A compiled (C++/Metal) iteration is the remaining lever.

## Torch layers

Two routes, both sound:

```python
from nsa_flow import NSAFlowLinear, NSAFlowConv2d

# (preferred) penalty: a standard layer plus a regulariser
layer = NSAFlowLinear(256, 32, w=0.0)
loss = task_loss + 0.3 * layer.defect()        # defect() is O(p k^2), exact gradient

# parameterisation: the effective weight is driven toward the feasible set
layer = NSAFlowLinear(256, 32, w=0.5)                     # Stiefel blend, exact at w=1
layer = NSAFlowLinear(256, 32, w=0.5, nonneg="hard")      # non-negative AND near-orthogonal
layer.project_()                                          # exact anchored prox, in place, no autograd
```

**Unified continuous blend and certified solver semantics (v3.3.2)**.
Across all layers (`NSAFlowLinear`, `NSAFlowConv2d`, `NSAFlowLayer`), `operator="auto"` is the default:
- **Signed (`nonneg=None`)**: uses the fast differentiable blend $W_{\text{eff}} = (1 - w) W + w\,\text{project\_scaled\_stiefel}(W)$.
- **Non-negative (`nonneg="hard"` / `"softplus"`, default `operator="auto"` $\to$ `"solver"`)**: Forward pass computes the exact anchored prox via `nsa_flow(mode="anchored", fidelity="anchor")` — identical to `layer.project_()` and linear SiMLR. With `nonneg="softplus"`, $W_{\text{base}} = \text{softplus}(W)$ is pre-applied and used in the Straight-Through Estimator (STE: $W_{\text{base}} + (W_{\text{eff}} - W_{\text{base}}).\text{detach}()$) so backward gradients scale smoothly by $\sigma(W)$. Inference is cached in `eval()`.
- **Explicit unrolled blend (`operator="blend"`)**: uses $W_{\text{eff}} = (1 - w) W_0 + w P(W_0)$ where $P(W_0)$ is the in-orthant defect flow run for `n_steps` iterations (default 8) independently of $w$, compiling a static graph without Dynamo recompilation overhead across $w$ sweeps.

`NSAFlowLayer` applies the same map per sample to a `[B, p, k]` batch.

**Cost** (66×5, one CPU thread; from `experiments`-style timing):

```
nn.Linear fwd 10 us
nonneg=None     w=0.5: train fwd   61 us (was 58) | eval fwd 9 us
nonneg=hard     w=0.5: train fwd  377 us (was 539) | eval fwd 9 us
nonneg=hard     w=1.0: train fwd  731 us (was 719) | eval fwd 9 us
nonneg=softplus w=1.0: train fwd  649 us (was 842) | eval fwd 9 us
operator=solver w=0.5: train fwd ~500 us          | eval fwd 9 us
MPS blend w=0.5 831 us (was 1227) | MPS hard w=1 5383 us (was 6372)
```

In `eval()` the effective weight is cached until the parameter changes, so inference costs the same as `nn.Linear` (9 µs). `compile=True` on any layer compiles the defect flow as one graph (`fullgraph=True`): measured 732 → 154 µs for the `hard, w=1` training forward, 393 → 66 µs for the flow alone.

## Documentation (ANTsPy Style)

Standard Sphinx documentation is built using `sphinx_rtd_theme`:
```bash
make docs        # builds HTML docs in docs/_build/html with 0 warnings
make docs-clean  # removes build cache
```
The documentation includes thematic guides for layers, solvers, energy functionals, data-anchored reconstruction, scikit-learn estimators, and diagnostic certificates, plus recursive API references.


## API

| Class / Function | Purpose |
|---|---|
| `nsa_flow(data_or_target, k=..., w=...)` | **Unified high-level entry point**; auto-dispatches based on data signs and dimensions |
| `NSAFlow(n_components=..., w=...)` | **Scikit-learn compatible estimator** with `fit`, `transform`, `fit_transform` |
| `nsa_flow_data(X, k, w, ...)` | fit non-negative basis to data; matrix-free when `p > n` |
| `nsa_flow_signed(X, k, w, consolidate=True)` | `V = V⁺ − V⁻`, signed contrast lifting with disjoint lobes |
| `relax_into_nonneg(...)` | continuation in `μ` into the non-negative cone |
| `stiefel_defect(Y)` | `D(Y)`, orthoNORMality |
| `angle_defect(Y, diagonal=)` | `C(Y)`, orthogonality at any column norms |
| `subspace_fidelity(Y, X0)` | sign-blind distance to `range(X0)` |
| `negative_mass(X0)` | how much of a target is unreachable under `Y ≥ 0` |
| `part_sparsity(W)` / `consolidate_supports(W)` | per-lobe support diagnostics and rounding |
| `effective_rank(Y)` | `k/(kD+1)`, in `[1,k]` |
| `project_nonneg` / `project_scaled_stiefel` / `polar_factor` | projections |
| `NSAFlowLinear` / `NSAFlowConv2d` / `NSAFlowLayer` | torch layers |

### High-Level Unified Interface

`nsa_flow` serves as **the unified wrapper** across all problem modes:

```python
from nsa_flow import nsa_flow, NSAFlow

# 1. Non-negative data -> fits non-negative basis V >= 0 (auto data mode)
r_data = nsa_flow(X_positive, k=5, w=0.5)

# 2. Signed or centered data -> signed contrast lifting V = V+ - V- (auto signed mode)
r_signed = nsa_flow(X_centered, k=5, w=0.5, consolidate=True)

# 3. Target loadings matrix -> anchored flow refinement
r_anc = nsa_flow(PCA_loadings, w=0.5)

# 4. Scikit-learn Pipeline Integration
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("nsa", NSAFlow(n_components=5, w=0.5, consolidate=True)),
    ("clf", LogisticRegression())
])
pipe.fit(X_train, y_train)
```

### Optimizers & Performance

All solvers default to `optimizer="lbfgsb"`: L-BFGS-B (Byrd, Lu, Nocedal & Zhu 1995). On CPU, a native C++ extension (`nsa_flow._native._lbfgsb_cpu`, built via Accelerate/BLAS with zero heap allocation in the inner loop) accelerates the solver by **6–11×** (ADNI cortical thickness: **47.7 µs/iter** vs 501.3 µs/iter in pure PyTorch; 2–4× on larger shapes) while maintaining bit-exact energy parity (< 1e-12). Setting `NSA_FLOW_DISABLE_NATIVE=1` cleanly switches back to the pure PyTorch reference solver. The implementation runs unmodified on CPU, CUDA and MPS in float32 or float64 with **no host↔device transfers** (the top-*k* eigenvector initialisation is computed on-device too; see `nsa_flow/linalg.py`). It was chosen by measurement (`experiments/optimizer_study.py`): it matches SciPy's Fortran L-BFGS-B energy to twelve figures on every problem tried, finds a strictly lower minimum on the hardest, and is the only pure-torch method that certified convergence on every configuration. `fista`, `spg` and `pqn` remain available; `torch_lbfgs` is deprecated (it never certified convergence and froze the support at its initialisation).

Every result carries the same diagnostics under fixed definitions — `grad_map` is one scale-invariant certificate for every optimiser, `defect_D`/`defect_Cg`/`defect_C` are all evaluated on the returned basis, and `converged` is set only when a certificate is earned (`result.certificate` is `"stationary"` or `"numerical_floor"`). `max_iter` caps gradient evaluations. The default `tol` is 1e-6 (float64) / 1e-4 (float32), which returns the same support as 1e-9 for 30–50% less work. Stall classification reliably terminates at the float floor across both float32 and float64 without spinning in noise.

`nsa_flow_signed` writes each component as a contrast of two non-negative
parts, which restores a signed basis's representational capacity: at `w = 0` it
reproduces signed PCA's reconstruction to the digit. The relaxation alone leaves
each part concentrated but not sparse, so pass `consolidate=True` to round to
exactly disjoint supports and re-solve with the support fixed. On ADNI cortical
thickness at `w = 0.5` that takes the largest part from 39 of 66 features to 18,
leaves `V⁺` and `V⁻` each about 10% dense with no component lost, and *improves*
held-out prediction — the small tail was noise.

Leave `w` at its default. Across nine ADNI cognitive outcomes the lifting is the
best of the four variants at `w = 0.5` (mean ΔR² over PCA +0.011 linear, +0.044
forest) and worse than PCA on 0 of 9 at `w = 0.75`. Since a linear model sees
only the span, that collapse is a loss of span quality: pushing `w` up rounds the
contrasts toward a partition that no longer spans what the data needs. `w = 0.5`
is also the best setting for sparsity, so there is nothing to trade off.

For `k > p`, orthonormal columns are impossible and `inf D = 1/p - 1/k > 0`;
this is reported rather than hidden behind a silently row-orthonormal answer.

## Install, test, reproduce

```bash
make install       # editable install with experiment + test extras
make test          # 403 assertions
make theory        # just the property battery (79 assertions)
make experiments   # regenerate paper/results/ and paper/figs/
make paper         # build paper/nsaflow.pdf
```

`tests/test_theory.py` states every proposition in the paper executably — the
bounds, the zero set, the invariances, the spectral identity, the decomposition,
the collapse floor, the gradient identities, the disjoint-support equivalence,
the overlap bound, and term calibration for every option pair. If a claim in the
paper is weakened, one of those fails.

## Migrating from 1.x

The 1.x API is gone; `nsa_flow(target, w=...)` replaces
`nsa_flow_orth(Y0, X0=..., ...)` and the retraction, optimiser and
learning-rate-strategy modules are removed. The appendix of the paper lists the
substantive changes and why each was made; the 1.x code and tests are preserved
under `attic/` for reference.

## Citation

```
Avants, B. NSA-Flow: Non-negative Stiefel-Approximating Flow --- a
calibration-free relaxation between reconstruction and feature clustering.
```

MIT licensed.
