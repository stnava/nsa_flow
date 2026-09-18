# Hand-off, 2026-09-17

State of the repo, what changed today, which numbers you can trust, and what I
got wrong. Written to be checkable: every claim carries the measurement.

## Repo state

* Version **2.8.0** (`nsa_flow/__init__.py`, `pyproject.toml`).
* **301 tests pass** (`python -m pytest -p no:warnings`). Use `-p no:warnings`:
  counting `.` characters in the log is unreliable because warning prose and
  file paths contain periods — that is how I mis-reported "210" and "324"
  earlier. 301 is the authentic figure from pytest's own summary.
* **7 commits unpushed** (`2dd7592..f1c7d92`). Nothing has been pushed today.
* Working tree clean except your untracked `paper/figure4_*` files, which I
  left alone throughout (I did once stage them by accident with `git add -A`
  and amended them back out — check them if anything looks off).

## The headline: a solver bug that invalidated part of the evidence

`nsa_flow_signed` at shipped defaults **exited after one iteration**, reporting
`stop_reason="line_search"`, `grad_map=inf`, `converged=True`, and returned its
own initialisation. It reproduced on every dataset and rank tried, and all 16
signed tests passed regardless.

Three distinct faults, fixed in `c2d5e46`, `d868f85`, `f1c7d92`:

1. **`C` is discontinuous at a zero column.** It depends only on column
   *directions*, so a column of norm `1e-20` contributes the same O(1) cosines
   as one of norm 1. An all-zero lobe is therefore a spurious local minimum no
   descent step can leave, and `eps`-smoothing cannot fix it (the discontinuity
   is in the direction limit, not the magnitude — I tried).
   `init="relax"` seeded `V⁻ ≡ 0`, landing exactly there.
2. **The certificate was a sentinel.** `gmap` was initialised to `inf` and
   assigned only after an accepted step, so a first-iteration failure reported
   `inf` as though measured, and `converged = stop != "max_iter"` called it
   success. Same pattern in `solve.py`, `signed.py`, `reconstruct.py`.
3. **The consolidate re-solve reported stale diagnostics** — `fidelity`,
   `defect` and `stop_reason` described the pre-consolidation solve while
   `energy` and `Y` were post.

### The fix

* **`orth="Cg"`, a new smooth defect, now the signed default.** Mass-weighting
  `C`'s cosines by each column's energy share cancels the `|vᵢ||vⱼ|` and
  collapses to `‖offdiag(V'V)‖²_F / tr(V'V)²` — no per-column normalisation,
  smooth everywhere, and exactly the *decorrelation half of `D`* (orthogonality
  without norm balance, which is what `C` was reaching for). Closed-form
  gradient verified against autograd to **1.5e-17** even with a column at
  `1e-18`. In `nsa_flow/angle.py` as `gram_offdiag_defect`; also available to
  `nsa_flow_data` via `orth="Cg"`.
* **`init="relax"` now seeds both lobes** (`V⁻ = (-V0).clamp_min(0)`).
* **Default `max_iter` for signed raised to 20000.**
* Honest reporting in all three loops: certificate measured on failure,
  `converged` requires it finite, `RuntimeWarning` when it stops far from
  stationarity.

Result across 18 dataset/k/w combinations (ADNI centred and raw, METABRIC;
k = 2, 5; w = 0.25, 0.5, 0.9):

| configuration | reach `gmap` < 1e-09 | dead lobes |
|---|---|---|
| where it started | 7 of 18 (6 exited after 1 iteration) | 1–3 in 11 cases |
| after the smooth defect | 12 of 18 | **0 in all 18** |
| **+ raised cap (current)** | **18 of 18** | **0 in all 18** |

69 s for the whole grid.

### Two step-rule changes tried and rejected on measurement

Recorded in `signed.py`'s docstring so they are not retried.

* **5× Barzilai–Borwein step: worse everywhere.** METABRIC k=2 w=0.25
  converges at 4118 iterations unboosted and hits the cap at 2×/5×/10×; ADNI
  centred k=5 w=0.5 goes 312 → ~1600 iterations. Armijo backtracks the
  inflation away and the secant property is lost.
* **Non-monotone GLL line search (memory 10), which BMR actually specify:**
  fixed 2 of 4 hard cases *before* the cap was raised, but afterwards needs
  **15% more iterations** (87533 vs 75805) at equal wall time. Reverted;
  monotonicity is worth keeping and `tests/test_optimizer.py` asserts it.

### Why the tests missed it, and what now guards it

`iters` never asserted, `grad_map` never referenced, `converged is True` never
asserted for signed, the one `stop_reason` check whitelisted *every* possible
value, 12 of 13 solver calls passed `init="split"` explicitly, the one
quantitative test ran at `w=0` (the single value where `relax` did converge),
and the fixture was **centred** — the one regime where the stall does not show.
The right guards already existed for `nsa_flow` (`test_solver.py:22-27`,
`:245-256`) and `nsa_flow_data` (`:298-306`) and had never been generalised.

**`tests/test_optimizer.py`** (new, 75 tests) now holds every entry point to one
contract through a single table — `iters > 1`, finite certificate, energy
strictly decreasing from the initialisation, monotone energy, `stop_reason`
consistent with `grad_map`, honest reporting at the cap — across centred, raw
non-negative and wide (p > n) regimes. **Adding a solver to `SOLVERS` is all
that is needed to hold it to the same contract.** Do that for any new solver.

## Which numbers you can trust

`_basis` in `experiments/rmd_support.py` passes `init="split"` on centred data,
which converged and which the fix did not touch. So:

| study | status |
|---|---|
| **ADNI CDRSB, signed +0.098 R² (t=4.2)** | **VALID** — split on centred, converged (1407 iters) |
| **ADNI 9 cognitive outcomes, w-effect table** | **VALID**, same reason |
| **UCI heart / diabetes** | **VALID** (z-scored ⇒ centred) |
| **NHANES consolidated** | **RE-MEASURED, unchanged**: +0.0035 vs +0.0036, t=6.49, p=1.4e-05 |
| **Golub signed / consolidated** | **RE-MEASURED and a conclusion REVERSED** (below) |
| **METABRIC k=2 signed / consolidated** | **RE-MEASURED**, slightly worse, conclusion unchanged |

### Golub: my "correctly inert" conclusion was wrong

`cc0e34b` concluded the lifting is inert on Golub because gene expression has
no contrast structure (3 of 6 lobes died). That was the trapped solver. Fixed:

| arm | old | **corrected** |
|---|---|---|
| Sparse PCA (signed) | 0.8818, 3/6 lobes dead | **0.8891, 0/6 dead** |
| Sparse PCA (signed+consol) | 0.8648 | 0.8659 |
| Sparse PCA (NSA-Flow, `data`) | 0.8837 | unchanged |
| Standard PCA | 0.8440 | unchanged |
| soft-threshold | 0.8356 | unchanged |

**The lifting is now the best arm on Golub** (0.8891 > 0.8837 > 0.8440), and
`exp23_golub_signed.py`'s docstring still says the opposite. **Fix that file.**

**Caveat: the Golub and METABRIC re-runs above were measured before the
`max_iter` raise in `f1c7d92`.** Re-run both at current defaults before quoting
them. That is the single most urgent open item.

## What I got wrong today (so you don't inherit it)

* **"Every signed/consolidated number in this project is invalidated"** —
  overstated, in the alarming direction. Only the `relax`-path studies were.
* **Test counts 210 and 324** — artifacts of counting `.` in logs. Real: 301.
* **"The lifting is correctly inert on Golub"** — artifact of the trapped solver.
* **"`init="split"` works"** — only on *centred* data. It stalls at 2 iterations
  on non-negative uncentred input, with reconstruction up to 12× worse.
* **The offset story.** Three interventions (offset column in the basis,
  row-centring, fitted rank-one) and none beat plain centring on ADNI. My
  simulator built the global level as an exact uniform `g·1ᵀ`, which was too
  kind to my own fix — real cortical thinning has slope CV 0.767 across regions
  with 4 of 66 loading negatively, so row-centring removes only the uniform
  part. The offset line is closed; see the exchange, not a file.
* **My contamination metric was circular** for OLS-based removal: removing the
  fit on `g` zeroes correlation with `g` by construction.
* **`6ac39bf`'s docstring** described the stall as a *property* with a
  sparsity-versus-contrast trade-off. Both were artifacts of comparing a
  converged solve against a stalled one. Superseded by `4e2b22e`, then by
  `c2d5e46`, which has the correct account.

## Open items, roughly in priority order

1. **Re-run Golub and METABRIC k=2 at current defaults** (post-`f1c7d92`) and
   correct `experiments/exp23_golub_signed.py`'s docstring, which still claims
   the lifting is inert.
2. **The z-scoring scope condition in the paper.** The ADNI cognitive claim
   (+0.043) **reverses to −0.033 under standardisation**, significant in both
   directions (4 outcomes tested: `+0.0379` centre-only vs `−0.0325` z-scored,
   forest). The number is correct for its preprocessing but the paper does not
   say the preprocessing is load-bearing. You deferred this; it is the one
   remaining correctness item in the manuscript.
3. **The paper has no section on the signed lifting**, though it is now a
   default-carrying feature and `C`'s `diagonal=False` variant existed only for
   it. §2 currently references the code module because there is no section.
4. **pysimlr: `ortho` and `nsaflow` are the same arm** — bit-identical on all 8
   seeds, one `elif` branch whose guard admits both at any positive weight. Any
   benchmark comparing those strings compares an arm with itself. Real defect,
   other repo, you had me leave it.
5. **Five duplicated SPG loops** (`solve.py:78-135`, `signed.py` ×2,
   `reconstruct.py` ×2). This is why one bug existed in three places.
   `_solve_fixed_w` is the natural core but its `vg` signature is hardwired and
   the consolidate loop needs a masked projection.
6. **`align`** is implemented, undocumented, and its justification (`D`'s
   right-`O(k)` invariance) does not hold under `C`. I recommend deleting it.
7. `orth="Cg"` is not offered on the anchored `nsa_flow`, only on `signed` and
   `nsa_flow_data`.

## Where things are

* **Standalone study write-up:** `NHANES_DIETARY_MORTALITY.md` (committed,
  `c14846c`) — the pressure-tested dietary→mortality result, including what did
  *not* survive.
* **Experiments:** `experiments/exp29_nhanes.py` (12-configuration pressure
  test), `exp23_golub_signed.py` (docstring needs fixing), `exp21_uci_*`,
  `exp22_ppmi_modeled.py` (confound-modelled, null), `exp20_ppmi_*` (retracted
  in its own docstring).
* **`paper/results/` is gitignored** — all CSVs there are local artifacts.
* **Data paths:** ADNI and PPMI CSVs in `~/Downloads/`; NHANES at
  `~/Downloads/cleaned_nhanes_21743372/`; METABRIC downloaded to a session
  scratchpad (gone) — refetch from
  `media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/`
  (689 MB expression, public, no credentials).
* **NHANES gotcha:** the ≥90%-complete dietary column set contains
  `RIDAGEYR`, `RIAGENDR` and `survey_day`. Age inside the "diet" block while
  also a covariate is a leak; `exp29` drops them plus the `VNDRXS*` derived
  duplicates (87 → 77 columns). A merge collision is the only reason this
  surfaced — with unique names it would have silently produced a large, very
  significant, wrong result.

## The standing methodological lesson

Four datasets in, prediction benchmarks could not resolve basis differences:
ADNI (preprocessing-reversible), PPMI (no signal — ceiling +0.067 AUC), Golub
(saturated at 0.98), NHANES (real but clinically trivial — the whole diet signal
is +0.038 AUC and the contest was over 0.004 of it). Where a difference *was*
measurable, a shuffled-support null at matched density sat within ~0.004 of PCA
— the same order as the effect. **Calibrate a null band before interpreting any
gap, and measure the ceiling before comparing arms inside it.** METABRIC is the
one dataset where dimension reduction demonstrably does real work: all 2000
genes is *worse* than clinical alone (−0.034) while PCA-20 is better (+0.029),
because `p > n`.
