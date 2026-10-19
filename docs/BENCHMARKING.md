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
nsa-flow-bench run --suite standard --jobs 4            # the reported benchmarks (~25 min with 5 workers)
nsa-flow-bench run --suite standard --public-only       # public datasets only
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
- `module_recovery`: feature modules vs known groups with unlabelled background present. Per random sample draw:
  `ari` (k = #groups, the usual primary), `stability` (split-half assignment ARI), `recall2k`/`separation2k`
  (k = 2 × #groups pair scores), `contiguity` (ordered features). Datasets without groups get stability and
  contiguity.
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
Methods (`FA_METHODS`): `PAF`, `PAF varimax` (the reference), `NSA.FA`, `NSA.FA post`, `ML varimax`, `PCA varimax`,
`Thresholded PAF`. NSA.FA is written for this benchmark from the public `nsa_flow()` API: each PAF iteration takes the
leading eigen-loadings of R with the current communalities on the diagonal, refines them with the signed anchored mode
(`nsa_flow(L, w, mode="anchored", nonneg=False)`), rotates (varimax) and updates communalities with damping
(`eta_h = 0.5`; it stops when the reproduced-correlation energy and the communalities settle, and reports the solver's own certificate). `w = 0.5` is the default and
the pre-registered setting; `NSA.FA post` regularises once after ordinary PAF. PAF and varimax follow `psych::fa(fm="pa")`
and `stats::varimax`; the CFA follows lavaan (`std.lv`); all three are tested against R when Rscript, psych and lavaan are
present (`tests/bench/test_fa.py`), plus golden Holzinger-Swineford values (chi-square 85.306, df 24, CFI 0.931,
RMSEA 0.092).
Datasets (public): `bfi` (psych, 25 items, 5 scales, reverse keys applied only to scale totals), `holzinger`
(lavaan, 9 tests, 3 factors), `fa_sim_n100/300/1000` (5 factors x 5 items, known loadings), `fa_null` (i.i.d. items) and
`bfi@shuffled`. `bfi` and `holzinger` are exported once from R with `nsa-flow-bench fetch bfi --download` (no network; needs
Rscript with psych and lavaan).
Pre-registered primary: `NSA.FA - PAF varimax` on Hofmann complexity (lower is better) with non-inferiority guards on fit
(`recon_err` +0.02, `cfa_rmsea` +0.01, `cfa_cfi` -0.02). The guards are listed in the verdict row; a PASS whose guard fails
is reported `FAIL (guard)`; a guard that cannot be evaluated (CFA skipped on a small test split) is `n/a` and does not
decide. History: an earlier draft guarded only on CFA RMSEA/CFI, which is vacuous (see above); `recon_err` was added as the
fit guard because it measures what the guard was meant to measure. Null controls require NSA.FA's recovery ARI on
unstructured items to be within 0.1 of zero.

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
`nsa-flow-bench check` runs the smoke suite and compares its primary estimates with `baseline_smoke.json`
(tolerance 1e-3). Use `--update` only after an intended change.

## Troubleshooting
- A quiet log from a backgrounded run is not a hung run: redirected stdout is block-buffered. The runner flushes its
  own lines; for per-split progress set `NSA_BENCH_TRACE=/path/trace.txt` to append one line per outer split.
- Workers cannot unpickle functions defined in `__main__`; protocol workers live in importable modules.
- Worker threads: each worker uses one torch/BLAS/OpenMP thread (`runtime.limit_threads`).

## Standard run (reference, 2026-10-19, 20 splits/draws; Golub 10)
| entry | primary contrast | estimate [95% CI] | status |
|---|---|---|---|
| adni (restricted) | NSA consolidated − PCA, forest R² | +0.147 [0.078, 0.216] | PASS |
| golub | NSA consolidated − PCA, linear bal. acc. | +0.016 [−0.076, 0.107] | FAIL |
| tecator | NSA consolidated − PCA, forest R² | −0.242 [−0.293, −0.192] | WORSE |
| planted / newsgroups / metabric (local) / pbmc3k | ARI, NSA consolidated w=0.5 − k-means | +0.377 / −0.111 / −0.160 / −0.163 | PASS / WORSE ×3 |
| adni@null_y, noise, metabric@shuffled | null controls | within tolerance | PASS ×3 |

The non-negative mode (`mode="data"`) is competitive with k-means on module recovery and more stable than NMF; it is
sensitive to input scaling. See the README for the interpretation.

## Not yet ported
`rapid_primary_benchmarks.py` (HTML report, diabetes, synthetic recovery), `benchmark_new_public_data.py` (Sonar,
Prostate sweeps), `speed_vs_pca.py` (the package's `speed` protocol is simpler) and `adni_cdrsb_repeated.py`
(superseded by the `fixed` rows in `nested_cv`). They still work and are not shims because the package does not yet
reproduce their numbers. NHANES and the factor-analysis protocol are built (above).
