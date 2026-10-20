# nsa_flow.bench: benchmarking guide

`nsa_flow.bench` is the official benchmarking sub-package of NSA-Flow. It owns datasets, methods, protocols,
metrics, pre-registered suites, results and reports, and works standalone (on your own arrays) or integrated
(`make bench`, the paper, the README).

**Scope.** This package contains benchmark code only. It calls the public `nsa_flow` API and never modifies core
solver modules.

## Quick start
```bash
pip install -e ".[bench]"            # numpy, pandas, scikit-learn, scipy, joblib (import nsa_flow needs none of them)
nsa-flow-bench list                  # datasets (ok / MISSING + why, tier), methods, protocols, suites
nsa-flow-bench run --suite smoke --profile smoke        # seconds, synthetic only
nsa-flow-bench run --suite standard-fast --profile fast --jobs 5   # daily driver: ~5 min with 5 workers (make bench)
nsa-flow-bench run --suite standard-full --jobs 5       # everything, ~40 min, full-forest selection (make bench-full); `standard` is the same suite
nsa-flow-bench run --suite standard-fast --profile fast --resume   # reuse finished entries of an interrupted run
nsa-flow-bench run --suite standard-fast --public-only  # public datasets only
nsa-flow-bench report standard [--brief]                # markdown from the latest stored run
nsa-flow-bench report standard --brief --inject README.md:standard   # refresh a generated README block
nsa-flow-bench check [--update]                         # smoke suite vs stored baseline
nsa-flow-bench fetch pbmc3k --download                  # public data only downloads on explicit request
```
Standalone, on your own data:
```python
from nsa_flow.bench import evaluate
ev = evaluate(X, y=y, groups=feature_group_ids, profile="quick")   # y -> nested CV; groups -> module recovery
print(ev.report()); ev.rows; ev.summary
```

## Layout
```
nsa_flow/bench/
  registry.py     DATASETS / METHODS / PROTOCOLS / SUITES
  datasets/       Dataset + DatasetSpec (tier), cache/download gate, one module per dataset, derived variants
                  (nulls, energy adjustment)
  methods.py      every method and grid, once; NSA methods return a solver certificate with each fit
  protocols/      nested_cv, module_recovery, factor_analysis, survival, speed
  fa/             PAF, varimax, ML-CFA, NSA.FA (own implementations, validated against R psych/lavaan)
  metrics.py      ARI, stability, pair recall/separation, contiguity, sparsity, paired summaries
  suites.py       pre-registered primary contrasts + pass rules (smoke, standard = paper)
  runner.py       availability, measured cost estimate, run, summary, verdicts
  results.py      rows.csv + manifest.json (suite, profile, versions, git sha, fingerprints)
  report.py       markdown from stored rows; README/paper marker injection
  api.py          evaluate(X, y, groups, ...) for arbitrary user data
  cli.py          nsa-flow-bench {list, fetch, run, report, check}
  baseline_smoke.json   stored primary estimates for `check`
tests/bench/      registry, leakage, determinism, schema, suites, reports, CLI, API, light-import
```

## Speed
Measured bottlenecks and what was done about them (bench code only):
- **Downstream models, not NSA, dominate nested CV**: the linear model plus the forest take 84% (ADNI) to 97% (NHANES) of
  the time; basis fits are 3-16% (Tecator is the exception: NSA solves dominate). Hyper-parameters are therefore selected
  with a smaller forest in the `fast` profile (`Profile.rf_trees_inner = 25`; reported outer fits keep 100; the `standard`
  profile selects with the full forest and is the reference fidelity). Agreement check
  on ADNI/Tecator: the same hyper-parameters in 67-83% of selections, outer scores within -0.004 +/- 0.008 of full-forest
  selection, selection 2.2-3.7x faster on ADNI (1.1x on Tecator).
- **Every finished entry is saved at once** (`<run>/entries/entry_NN.csv|json`, `progress.log`); `--resume` reuses entries whose
  inputs (the pre-registered entry, the profile and the core hash) are unchanged, so a stopped run loses nothing.
- **`standard-fast`** keeps the same pre-registered primaries as `standard-full` with cheaper settings: three prediction
  methods, primary module variant only, 10 draws on the large expression sets, NHANES at 6,000 rows; golub, tecator,
  newsgroups, `metabric@shuffled`, speed, NHANES at 20,000 rows and the residual variant stay in the full suite.
  Wall time with 5 workers: about 4.5 minutes (adni 25 s, metabric 19 s, pbmc3k 44 s, adni@null_y 23 s, the seven factor-analysis
  entries 42 s, NHANES 41 + 42 + 8 s) against about 40 minutes for the full suite. At 6,000 rows NHANES has too little power to see
  effects of 0.003-0.007 AUC; use `standard-full` for that claim. **Cheap selection also attenuates small effects**: on NHANES raw
  (20,000 rows) the legacy-init gain over PCA is +0.0074 [0.0049, 0.0100] with 100-tree inner selection and +0.0039 [0.0003, 0.0074]
  with 25 trees, so `fast` is for development and guarding, not for small-effect claims.
- Every manifest records a **core hash** (sha1 of the library sources outside `bench/`) and whether the tree was dirty; runs
  with different core hashes are different experiments, and `--resume` will not mix them.

## Why a package
Benchmarks used to be ~50 scripts under `experiments/`, each with its own splits, baselines, grids, summaries and
CSV schema. That produced real errors: PCA tuned in one script and fixed in another gave a spurious Golub gain;
METABRIC was first scored on a toy 45-gene design; slow methods were discovered by running them; numbers were copied
by hand into the README and the paper. The package fixes the structure, not just the code.

## Rules
1. **Pre-registration.** The primary contrast and pass rule are fixed in the suite before the run; the manifest
   stores the whole suite, profile, versions, git sha and dataset fingerprints.
2. **Equal tuning budget.** Every method, PCA included, tunes its hyper-parameters on the training portion by the
   same inner CV; a fixed-setting (`w=0.5, k=5`) row is reported alongside.
3. **One schema, one summariser.** Every protocol emits the columns in `results.COLUMNS`; paired t intervals, win
   counts and sparsity come from `metrics.long_summary` only.
4. **Cost-aware.** A measured single-fit probe estimates each entry before it runs; `--budget-min` skips entries
   over budget and methods too slow for a dataset (e.g. sklearn `SparsePCA` above p=500) are skipped, both with the
   reason stored in the results. The probe covers basis fits only, so it undercounts model-fitting time.
5. **Nulls are derived data.** `name@null_y` (outcome permuted) and `name@shuffled` (each feature permuted) run
   through the same protocols with `expect="null"` verdicts; `noise` is an unstructured dataset.
6. **Solver certificate.** Every NSA fit records `converged` and `defect_D`; the report prints convergence and the
   worst defect per method.

## Data tiers and policy
Every dataset has a tier: `public` (openly obtainable; synthetic data counts), `restricted` (data-use agreement,
e.g. ADNI) or `local` (cache only, e.g. the METABRIC npz). Verdict tables show the tier and `--public-only` runs
public datasets only. Non-public data is never required to reproduce the public tier. PPMI is deliberately not
registered (not public, not reproducible).

Data locations are searched in this order: `$NSA_BENCH_DATA`, `~/.cache/nsa_flow_bench`, and the repository's
`experiments/cache` when run from a checkout. Nothing is downloaded unless `fetch --download` is used; a missing
dataset is skipped with the places that were searched. ADNI reads `NSA_BENCH_ADNI_THICKNESS`. Gene and marker lists
(METABRIC PAM50 groups, PBMC3k markers) are the authors' annotation and are flagged as such in every report.

## Protocols
- `nested_cv`: outer splits per dataset convention (holdout, k-fold, stratified k-fold); everything (centring,
  feature selection, standardisation, basis, model) is fitted inside each training portion; selection is separate
  for each downstream model (linear / forest); regression reports R², classification balanced accuracy.
  **Axis control:** every NSA method also gets a `(scores PCA-rotated)` row (the same scores rotated onto their principal
  axes; a linear model is invariant to it, a forest is not) and a `score_collinearity` metric (mean |off-diagonal correlation|
  of the training scores). They separate "better subspace" from "axis-aligned features a tree can split on": on ADNI the
  forest gain over PCA (+0.122) disappears under the rotation (-0.009); on Tecator collinear NSA/band scores (0.98) give the same
  forest R² as the raw wavelengths (0.69-0.72) and rotating them restores PCA's 0.926.
- `module_recovery`: feature modules vs known groups with unlabelled background present. Per random sample draw:
  `ari` (k = #groups), `stability` (split-half assignment ARI), `recall2k`/`separation2k` and `pba_x{m}` (pair balanced
  accuracy (recall + separation)/2 at k = m × #groups for m in 1, 2, 4, 8: fairer than ARI when background must also be
  partitioned), `contiguity` (ordered features), and an oracle **ceiling** row (k-means on the labelled features alone).
  **Primary (corrected after a diagnostic study):** the non-negative data mode with multistart (`NSA nonneg (multistart)`:
  lowest energy of 4 initialisers, the first being the library default) against `KMeans features` with the same number of
  starts (`n_init=4`). Signed consolidated NSA is a secondary row: it finds variance-ranked contrasts and is not a clustering
  method (METABRIC ARI 0.23, PBMC3k 0.74, the worst of all methods). Also reported: `KMeans features (n_init=30)`,
  `NSA nonneg (standardised)` (standardise-then-shift; the default `NSA nonneg` shifts raw-scale data), NMF, PCA.
  Datasets without groups get stability and contiguity.
- `factor_analysis`: repeated 80/20 splits of respondents; every method is fitted on the training rows (correlation
  built from them) with the dataset's known number of factors. Each method has a `fixed` row at its pre-registered
  default (`w = 0.5` for NSA.FA) and, if it has a grid, a `tuned` row (inner CV, criterion = held-out
  reproduced-correlation error, which is unsupervised). Metrics: interpretability (`hofmann`, `cross_loadings`,
  `sparsity`, `ortho_defect`), recovery (`ari` vs the known item->scale map, `tucker` vs true loadings on simulated
  data), fit (`recon_err` off-diagonal reproduced-correlation error on the test split, `heywood`, ML-CFA
  `cfa_rmsea/cfi/tli/srmr/bic` of each method's own simple structure on the test covariance, computed only when
  the test split has at least 5 respondents per item), `score_validity` (held-out regression factor scores vs keyed
  scale totals) and `stability_ari/tucker` (two fits on disjoint halves). CFA indices are identical across methods
  whenever their item assignments agree, so they cannot register loading shrinkage; `recon_err` is the fit guard.
- `survival`: survey-weighted forest on [covariates, basis scores] at fixed `k = 5` and the default `w`, scored by the
  exact Harrell C-index and AUC on held-out rows (stratified repeated k-fold on the event).
- `speed`: best-of-3 fit time on seeded synthetic shapes.

## Factor analysis
Methods (`FA_METHODS`): `PAF varimax`, **`PAF promax`** and **`PAF oblimin`** (oblique; they preserve the PAF fit exactly and are
the right baselines for any simple-structure method; validated against R `stats::promax` / `GPArotation` when available),
`NSA rotation`, `NSA keyed non-negative`, `NSA.FA shrink` (legacy), `ML varimax`, `PCA varimax`, `Thresholded promax`.
Written for this benchmark from the public `nsa_flow()` API; PAF and varimax follow `psych::fa(fm="pa")` and
`stats::varimax`, the CFA follows lavaan (`std.lv`); tests compare all of them with R when Rscript, psych, lavaan and
GPArotation are present, plus golden Holzinger-Swineford values (chi-square 85.306, df 24, CFI 0.931, RMSEA 0.092).
- **NSA rotation** (default `w = 0.5`) is NSA as a fit-preserving rotation: PAF, then a simple-structure basis of the SAME
  subspace (`fidelity="subspace"`), then factor correlations by least squares. The model-implied correlation, hence the fit,
  is PAF's by construction. In exploratory runs it did not depend on `w` (subspace fidelity leaves `w` no trade-off to make).
- **NSA keyed non-negative**: items are reverse-keyed (`meta["keys"]`) and a non-negative anchored subspace is requested:
  independent-cluster structure, trading some reproduced-correlation fit for simplicity (the deleted cross-loadings).
- **NSA.FA shrink** (the first design: loadings shrunk toward orthogonal columns at every PAF iteration) is dominated by
  oblique rotation (higher fit error, no simpler structure) and is kept only as a secondary row.
Datasets (public): `bfi`, `holzinger`, `fa_sim_n100/300/1000` (5 factors x 5 items, factor correlation 0.2, known loadings),
`fa_null`, `bfi@shuffled`; `bfi` and `holzinger` are exported once from R with `nsa-flow-bench fetch bfi --download`.
Metrics per held-out split: interpretability (`hofmann`, `cross_loadings`, `sparsity`, `ortho_defect`, `factor_corr`),
recovery (`ari`, `tucker`), fit (`recon_err` of L Phi L', `heywood`, ML-CFA `cfa_*` when the test split has at least 5
respondents per item), `score_validity` (regression scores W = R^-1 L Phi) and `stability_*`. CFA indices are identical across
methods whenever item assignments agree, so `recon_err` is the fit guard.
**Pre-registered primary:** `NSA rotation - PAF promax` on Hofmann complexity (lower is better) with the guard `recon_err` +0.01
(a fit-preserving method should not lose fit); a PASS whose guard fails is `FAIL (guard)`, a guard that cannot be evaluated is
`n/a`. The keyed non-negative row is reported as a simplicity/fit trade-off, without a verdict. Null controls require NSA
rotation's recovery ARI on unstructured items to be within 0.1 of zero. History: the first design compared the shrinking
NSA.FA with varimax; an oblique baseline and a fit-preserving use of the prox were the missing pieces.

## NHANES
`nhanes` (CDC NHANES + NCHS linked mortality, public provenance; a cleaned release is read from
`NSA_BENCH_NHANES_DIR`, default `~/Downloads/cleaned_nhanes_21743372`; the raw CDC files are not fetched or rebuilt):
77 dietary nutrients (age, sex and survey-day columns removed so age cannot leak), covariates in every model, outcome
all-cause mortality, n = 47,742 with survey weights and follow-up time. Data are intrinsically non-negative, so
`NSA nonneg` needs no shift. `nhanes@density` (per 1000 kcal) and `nhanes@residual` (log-log energy residual) test whether
an advantage is an energy effect. Controls reported without verdicts: `Ceiling (all features)` (V = I) and
`NSA nonneg (shuffled support)` (fitted basis with its feature assignment permuted). Cost control: the entries use
`max_n = 20000` stratified rows, 10 outer splits and 3 inner folds; a first pass at 6,000 rows is kept in the notes
below. The energy adjustments are fitted on all rows from the features only (two parameters per nutrient).

## Verdicts
`effect`: PASS if the 95% interval of (method − reference) excludes 0 in the stated direction, WORSE if it excludes
0 the other way, FAIL otherwise; optional non-inferiority guards on other metrics must also hold (`FAIL (guard)`). `null`: PASS if the interval includes 0 (or, with no reference, the method's own
mean is within `tol` of 0). `info`: no verdict. SKIPPED: dataset unavailable, not public (with `--public-only`),
or over budget.

## Extending
- **Dataset.** A function decorated with `@dataset(name, tier=..., source=..., fetch=...)` returning a `Dataset`
  (`X`, optional `y`, `covariates`, `groups`, `ordered`, `task`, `outer`, `standardize`, `top_variance`, `draw_n`).
  Raise `DatasetUnavailable` with a clear reason when the data is absent.
- **Method.** A `Method(name, fit, key, grids, ...)` registered in `methods.py`; `fit(rep, hp, seed, ctx)` returns
  `(V, cert)` where `V` is `p x k`. It then appears in the CLI, `evaluate`, and any suite that lists it.
- **Suite entry.** `Entry(dataset, protocol, methods, Primary(metric, method, ref, model), expect)` in `suites.py`.

## Regression guard
Run long jobs from a clean checkout or `git worktree` pinned to a commit: a shared working tree whose branch or core sources change
mid-run (another session, a merge) silently mixes code versions. `nsa-flow-bench check` runs the smoke suite and compares its primary estimates with `baseline_smoke.json`
(tolerance 1e-3). Use `--update` only after an intended change.

## Troubleshooting
- A quiet log from a backgrounded run is not a hung run: redirected stdout is block-buffered. The runner flushes its
  own lines; for per-split progress set `NSA_BENCH_TRACE=/path/trace.txt` to append one line per outer split.
- Workers cannot unpickle functions defined in `__main__`; protocol workers live in importable modules.
- Worker threads: each worker uses one torch/BLAS/OpenMP thread (`runtime.limit_threads`).

## Standard run (reference, 2026-10-19, post-varimax-init core 115f30a, full-forest selection; 20 splits/draws, Golub and NHANES 10)
| entry | primary contrast | estimate [95% CI] | status |
|---|---|---|---|
| adni (restricted) | NSA consolidated − PCA, forest R² | +0.155 [0.069, 0.242] | PASS |
| golub | NSA consolidated − PCA, linear bal. acc. | +0.049 [−0.015, 0.113] | FAIL |
| tecator | NSA consolidated − PCA, forest R² | −0.237 [−0.286, −0.188] | WORSE |
| planted | ARI, NSA consolidated w=0.5 − k-means | +0.377 | PASS |
| newsgroups / metabric (local) / pbmc3k | same | −0.076 / −0.166 / −0.207 | FAIL / WORSE / WORSE |
| adni@null_y, noise, metabric@shuffled | null controls | within tolerance | PASS ×3 |
| bfi, holzinger, fa_sim n=100/300/1000 | NSA.FA − PAF varimax, Hofmann complexity | −0.097, −0.031, −0.127, −0.084, −0.068 | FAIL (guard): reproduced-correlation error worsens |
| fa_null, bfi@shuffled | null controls | ARI ≈ 0.03 | PASS ×2 |
| nhanes / nhanes@density (AUC), nhanes@density survival (C-index) | NSA nonneg − PCA, forest | +0.001 / +0.003 / +0.007 | FAIL / FAIL / PASS |

Initialiser A/B on NHANES raw (same data and splits, 100-tree selection): legacy `init="pca_clamp"` +0.0074 [0.0049, 0.0100];
default (varimax) +0.0012 [−0.0020, 0.0045]. The non-negative mode (`mode="data"`) is competitive with k-means on module
recovery and more stable than NMF; it is sensitive to input scaling and to the initialiser. See the README for the interpretation.

## Not yet ported
`rapid_primary_benchmarks.py` (HTML report, diabetes, synthetic recovery), `benchmark_new_public_data.py` (Sonar,
Prostate sweeps), `speed_vs_pca.py` (the package's `speed` protocol is simpler) and `adni_cdrsb_repeated.py`
(superseded by the `fixed` rows in `nested_cv`). They still work and are not shims because the package does not yet
reproduce their numbers. NHANES and the factor-analysis protocol are built (above).
