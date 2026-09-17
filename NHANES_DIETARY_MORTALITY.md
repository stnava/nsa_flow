# NHANES dietary patterns and all-cause mortality

Standalone record of a fresh, well-powered test of NSA-Flow, and of what
survived pressure-testing.  Nothing here is in the paper.  Written to be
checkable rather than persuasive: every claim carries the number that supports
it and the configurations under which it fails.

Code: `experiments/exp29_nhanes.py` runs the pressure test (twelve
configurations, `STAGES` at the bottom of the module) and writes
`paper/results/e29_nhanes_harden.csv`.  The primary 15-fold run and the
split-half stability run that preceded it wrote
`paper/results/e28_nhanes_mortality.csv` and `e27_nhanes_stability.csv`; note
that `paper/results/` is gitignored, so those are local artifacts and the module
is the reproducible entry point.  Data path is hard-coded to
`~/Downloads/cleaned_nhanes_21743372/`.

## Why this dataset

Every predictive result in this project before now came from ADNI cortical
thickness: `n = 372`, one cohort, and a conclusion whose *sign* depends on
whether features are z-scored (`+0.038` centred, `−0.033` standardised, both
significant).  PPMI could not adjudicate anything, because the total imaging
signal available there is a few hundredths (ceiling `+0.067` AUC for SAA,
`R² ≈ 0` for UPDRS-I).  Golub saturates at AUC 0.98 where every basis lands
within 0.009.

NHANES dietary is a different regime on every axis that mattered:

| | ADNI thickness | NHANES dietary |
|---|---|---|
| n | 372 | **49,283** primary / 47,742 pressure test |
| events | ~300 per score | **7,569 deaths** (15.4%) / 6,995, median 117 mo |
| p | 66 | 77 nutrients |
| non-negativity | imposed by a preprocessing choice | **intrinsic**, `min = 0.00`, true zeros |

Intrinsic non-negativity is the point.  It removes the preprocessing degree of
freedom that made the ADNI conclusion reversible, and it puts the data in the
cone the method assumes rather than forcing it there.

Source: `~/Downloads/cleaned_nhanes_21743372/` (a cleaned NHANES release).
Dietary recall days are averaged to one row per `SEQN`; 77 nutrient columns at
≥90% completeness; complete cases only, nothing imputed.  The pressure test
additionally requires `WTDRD1` and `PERMTH_INT` to be present for the weighted
and survival stages, which is why its n is 47,742 against the primary 49,283 —
part of the magnitude difference between the two runs.

### A contamination found and removed

The ≥90%-complete column set also contains `RIDAGEYR`, `RIAGENDR` and
`survey_day`.  Age inside the "diet" block while it is also a covariate would
leak the strongest mortality predictor into the thing being evaluated.  All
three are dropped, as are the `VNDRXS*` columns, which are derived duplicates of
their `DRXT*` counterparts (`VNDRXSMFAT` of `DRXTMFAT`) and so collinear rather
than informative.  87 columns → 77.

An earlier stability run in this session was executed *before* this was caught
and its matrix therefore contained age and sex; its numbers are superseded by
`e27` as regenerated here.

## Design

Covariates in **every** model: age, sex, race/ethnicity, poverty-income ratio,
education, survey cycle.  So every number below is what diet adds on top of
demography, not a marginal association.

Two safeguards, both adopted because their absence wrecked earlier work in this
project:

* **A ceiling arm** — all 77 nutrients, no reduction.  A basis comparison
  conducted inside a gap that has no signal is uninterpretable; not measuring
  the ceiling first is what made the PPMI attempt worthless.
* **A shuffled-support null** — the fitted basis with its feature assignment
  permuted, preserving density, column norms and the whole value multiset, and
  destroying only *which* nutrient each weight belongs to.  This calibrates the
  noise floor before any difference is read.

`k = 5`, `w = 0.5` unless swept.  Neither was tuned on this outcome, so no
selection bias — and equally, neither is optimised.

## Primary result

Random forest, 15 folds (5 × 3), AUC:

| arm | AUC | Δ vs covariates | Δ vs PCA | p |
|---|---:|---:|---:|---:|
| covariates only | 0.8013 | — | — | — |
| **ceiling, all 77** | **0.8391** | +0.0379 | +0.0239 | — |
| `nsa_raw` | 0.8191 | +0.0178 | **+0.0039** | <1e-4 |
| `nsa_consolidated` | 0.8188 | +0.0176 | +0.0036 | <1e-4 |
| NMF | 0.8157 | +0.0145 | +0.0005 | 0.36 |
| centred PCA | 0.8152 | +0.0139 | 0 | — |
| shuffled-support null | 0.8109 | +0.0096 | −0.0043 | — |

Read against the ceiling: the five-component arms capture 47% of the available
diet signal (`nsa_raw`) against PCA's 37%.

## Pressure test

Twelve configurations, 10 folds each (`experiments/exp29_nhanes.py`, stage
list at the bottom of the module).  `nsa_raw − PCA` in every one:

| configuration | Δ vs PCA | p |
|---|---:|---:|
| energy: raw | +0.0012 | 0.007 |
| **energy: density (per 1000 kcal)** | **+0.0026** | **0.003** |
| energy: residual (log-log) | +0.0013 | **0.053** |
| k = 3 | +0.0023 | 0.010 |
| k = 8 | +0.0049 | 3e-5 |
| k = 12 | +0.0044 | 8e-5 |
| w = 0.25 | +0.0022 | 3e-4 |
| w = 0.75 | +0.0022 | 0.005 |
| survey-weighted + survival | +0.0038 | 0.002 |
| different fold seed | +0.0019 | 0.016 |
| n = 20,000 | +0.0020 | **0.097** |
| n = 8,000 | +0.0069 | 0.006 |

Survival concordance (Harrell C against `PERMTH_INT`, survey-weighted fit),
which is the correct analysis for a censored outcome rather than the binary:

| arm | C-index | Δ vs PCA | p |
|---|---:|---:|---:|
| `nsa_consolidated` | **0.7435** | +0.0042 | 3e-4 |
| `nsa_raw` | 0.7424 | +0.0031 | 0.002 |
| NMF | 0.7417 | — | — |
| centred PCA | 0.7393 | 0 | — |
| null | 0.7376 | −0.0032 | 0.19 |

## What survives, and what does not

**Survives.**  The sign: `nsa_raw` is above centred PCA in **12 of 12**
configurations, across energy adjustment, `k` from 3 to 12, `w` at 0.25 and
0.75, survey weighting, a second fold seed, three sample sizes, and a proper
survival concordance.  Consolidation — an exactly disjoint partition, one
nutrient per component, 20% density — is **free**: it matches `nsa_raw` on AUC
everywhere and is the *best* arm on the survival C-index.  That is the first
setting in this project where hard disjointness costs nothing, and it is the
`p/k` story resolving favourably (`p = 77`, `k = 5`; at `p = 13` on UCI heart it
cost 0.018 AUC).

**Does not survive: the claim that NSA-Flow beats NMF.**  In the unadjusted
primary, NMF was `+0.0005` (n.s.) against `nsa_raw`'s `+0.0039`.  Under energy
adjustment they are `+0.0021` and `+0.0026` — indistinguishable.  The honest
statement is that *non-negative bases* beat centred PCA here, not that this
particular one beats the established alternative.

**Does not survive: the magnitude.**  It is 3× smaller under a slightly
different sample and fold count (`+0.0039` → `+0.0012`), borderline under the
residual energy method (`p = 0.053`), and not significant at `n = 20,000`
(`p = 0.097`).  The effect is small and its size is not stable across
reasonable analysis choices; only its direction is.

**The null puts that in perspective.**  The shuffled-support control sits
`−0.0021` to `−0.0050` from PCA across stages while `nsa_raw` sits `+0.0012` to
`+0.0069` above it.  So NSA's margin over PCA is the *same order* as the null's
own displacement from PCA.  What distinguishes them is consistency of sign — the
null is below PCA in 9 of 12 stages, `nsa_raw` above in 12 of 12 — not the size
of the gap.  Anyone quoting `+0.004 AUC` without the null band is overstating
this.

## Caveats

* One dataset, one outcome.  All-cause mortality over ~10 years.
* `k` and `w` were not selected by nested cross-validation, here or anywhere in
  this project.
* Nutrients are compositional and mutually collinear; three energy adjustments
  were tried and they disagree in strength, which is itself a finding about how
  sensitive this comparison is to the treatment of the shared offset.
* NHANES is a weighted multi-stage sample.  `WTDRD1` is used in stage D only;
  cycle is adjusted everywhere.  Proper variance estimation for a complex survey
  design (strata, PSUs) is not done, so the p-values are cross-validation
  p-values, not survey-corrected inference.
* Cross-validation folds are not independent samples, so repeated-fold
  p-values overstate significance in the usual way.  The 12-configuration sign
  consistency is the more trustworthy evidence than any single p-value.
* The interpretability claim — that five disjoint nutrient groups correspond to
  recognisable dietary patterns — is **not** evaluated.  It needs a nutrition
  reader, not a cross-validation loop.

## Bottom line

On a cohort 128× larger than the one the paper relies on, with intrinsic
non-negativity and a confound-adjusted, null-calibrated design, NSA-Flow gives a
small but directionally robust improvement over centred PCA (`+0.002` to
`+0.004` AUC, sign holding in 12 of 12 configurations), and delivers an exactly
disjoint one-nutrient-per-component partition at no predictive cost — the best
arm on survival concordance.  It does not beat NMF once energy is adjusted.
