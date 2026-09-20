# What next — decision memo (2026-09-19, v3.1.0)

State: `main` at 27a8dd6, 366 tests pass. Pure-PyTorch L-BFGS-B is the universal
default; one scale-invariant certificate across every optimiser; no host↔device
transfers; benchmarks re-run in-fold on the certified solver (README §Evaluation).
Nothing below is started.

Ordered by recommendation. Each item states what is known, what it would take,
what it should buy, and the risk.

---

## A. Compiled L-BFGS-B iteration — the only thing that changes the speed story

**Precise implementation plan: `docs/NATIVE_KERNEL_PLAN.md`.** Diagnosis is now
exact: 314 aten ops/iteration at 2.2 µs; init, tol, threads, memory ruled out.

**Where the time is.** Profiled ADNI-shaped fit: the objective is ~10% of each
iteration; the other 90% is ~50 small tensor ops of L-BFGS-B bookkeeping
dispatched from Python at 0.5–2 ms/iteration. `torch.compile` on the compilable
pieces measured +5–9% (`nsa_flow.lbfgsb.set_compile`, off by default): the
Cauchy walk's early exit and the memory-length changes are Python control flow,
so the graph breaks exactly where the cost is. A two-phase (active-set-then-face)
variant halved the algebra per iteration and did not move the clock — confirming
dispatch, not arithmetic, is the cost.

**What it takes.** One iteration (Cauchy walk + subspace solve + compact rebuild +
Wolfe step) as a C++ torch extension, with a Metal path for MPS. The algorithm is
finished and validated — energies match SciPy's Fortran to 12 significant figures
on 17/18 configurations and beat it on the 18th — so this is a translation with a
bit-exact reference to test against. SciPy's Fortran is a direct model. Keep the
Python path as fallback so `pip install` stays pure-Python.

**Expected.** SciPy's Fortran is ~4× faster than ours on identical iterations; a
native kernel should land there on CPU and remove most of MPS's ~3 ms/gradient.
ADNI-shaped fit ~25 ms (now 106), Golub-shaped ~200 ms (now 720). Still not PCA's
0.4 ms — nothing iterative will be — but a legitimate tens-of-milliseconds story.

**Effort / risk.** 2–4 days. Adds a build step (C++ toolchain; Metal for MPS):
ship wheels or keep the fallback. Medium risk, fully testable.

---

## B. Nested-CV selection of `w` and `k` — the only thing that changes the evidence story

**The hole.** `signed.py`'s own docstring: "`w` and `k` were not selected by nested
cross-validation anywhere." Every table fixes `w=0.5, k=5` by convention. The
ADNI result (+0.12 R² over PCA under a forest, 20 splits, CI [0.07, 0.18]) is real
*at that setting*; the reviewer's first question is whether `w=0.5` was chosen
because it worked. The in-paper w-sweeps *are* that selection, done in-sample.

**What it takes.** Inner CV over `w ∈ {0.1, 0.25, 0.5, 0.75, 0.9} × k ∈ {3, 5, 8}`
inside each outer split, select on outer-training-fold validation score, score once
on the outer test fold. `experiments/adni_cdrsb_repeated.py` already has the outer
loop. ≈ 20 splits × 15 configs × 5 inner folds ≈ 1500 fits at ~0.1 s — minutes.
Same for Golub via `rapid_primary_benchmarks.py`.

**Expected.** Either +0.12 survives with `w` chosen blind (publishable as-is), or it
shrinks and the paper says so with the honest number. Both outcomes are worth more
than the current one.

**Effort / risk.** One day. No technical risk; the risk is to the result, which is
the point.

---

## C. Jacobi eigensolver speed on MPS at medium `n`

**The gap.** MPS has no native `eigh` (and `svd` silently falls back to CPU), so
`top_k_eigenvectors` uses the on-device Jacobi solver for the `min(n,p)`-sided
Gram. Side 66 (ADNI) is negligible; side 200 is ~2 s in CPU float64, so on MPS the
*initialisation* can cost more than the solve for n≈200–500. The loop is (n−1)
rounds × ~8 sweeps of ~10 small ops.

**Options.** (i) One gather/scatter pair per round instead of row/column slicing —
~3× on the constant. (ii) Lower the dense threshold on MPS to ~64 and route larger
sides through subspace iteration, which is exact only with an eigengap — fine for
an initialiser, not for a claimed optimum; must warn. (iii) Both.

**Effort / risk.** Half a day. Low risk; affects MPS with `min(n,p) > ~64` only.
Worth it only if MPS at that scale is a target.

---

## D. Small items to fold into whichever of A–C goes first

- **Remove `torch_lbfgs`** at the v4.0 boundary. Deprecated, never certified on any
  configuration, froze the support at its initialisation. 30 min.
- **float32 tolerance / patience.** Default `tol=1e-4` in float32 relies on the
  plateau detector; a float32 signed fit took 3231 gradients to reach the floor
  where float64 took ~300. Probably `patience` tuned for float64. 1 hour to
  measure and set.
- **Wire `experiments/speed_vs_pca.py` into the Makefile** and emit its CSV; it is
  seeded and correct but not part of any target. 15 min.
- **The paper** (`paper/nsa_flow.Rmd`, `paper/nsaflow.tex`) still carries the
  pre-3.0 ADNI and Golub numbers and the single-split framing. B's result is what
  to write in. Parked by decision.

---

## Recommendation

**B, then A.** B is a day, has no downside, and determines what the method is
worth before days are spent making it fast. If B holds, A is worth doing properly
— a fast, certified, honestly evaluated solver is a tool people adopt. If B shrinks
the effect, A is still worth doing but the framing changes, and that is better
known first. C only with an MPS use case at n>200; D rides along.

Against doing A first: it is the most visible and the most fun, and it is the one
whose value depends on what B finds.

---

## Reference: what the numbers currently are

| | value | source |
|---|---|---|
| ADNI CDRSB, consolidated vs PCA, forest ΔR² | +0.123 [0.065, 0.182], 20 splits | `experiments/adni_cdrsb_repeated.py` |
| Golub 3-class, signed w=0.5 vs PCA, linear | 0.748 vs 0.698, in-fold | `experiments/rapid_primary_benchmarks.py` |
| Signed w=0 vs PCA | identical (by construction; old solver: `defect_D` up to 0.50) | public-data benchmarks |
| ADNI-shaped fit | 106 ms, 159 gradients | README speed table |
| Golub-shaped signed fit | 720 ms, 329 gradients | README speed table |
| torch vs SciPy L-BFGS-B energy | match to 12 figures 17/18, lower 1/18 | `experiments/optimizer_study.py` |
