# Plan: a fused native L-BFGS-B iteration for NSA-Flow

Written 2026-09-19 for whoever implements it (human or LLM). Self-contained:
read this, `nsa_flow/lbfgsb.py`, `nsa_flow/optim.py`, `nsa_flow/reconstruct.py`
and `nsa_flow/signed.py`, and nothing else is required.

## 0. Why, in numbers (do not skip — this is the acceptance test)

nsa_flow 3.1.3 on the paper's imaging shape (300×66, k=5, float64, CPU, one thread):

| quantity | measured |
|---|---|
| wall time per fit | 93–108 ms |
| iterations (L-BFGS-B steps) | 157–195 |
| **aten ops issued per iteration** | **314** |
| mean cost per aten op | 2.2 µs |
| arithmetic per iteration (FLOP-bound estimate) | ~10 µs |
| PCA on the same matrix (`top_k_eigenvectors`) | 0.4 ms |

The loop is **dispatch-bound**: 314 tiny tensor ops on 20-dimensional vectors
per iteration, none individually slow. Verified *not* to be the cause: thread
contention (threads=1 gives −15%), L-BFGS memory size (5–40: ±10%), tolerance
(default 1e-6 is required on unstructured data), initialization (all inits
converge in 118–371 gradients to the identical energy). `torch.compile` gives
+5–9% because the Cauchy walk's early exit is data-dependent control flow.

Per-iteration split (ADNI shape): Cauchy walk 0.16 ms, subspace step 0.10,
compact-form rebuild + 20×20 solve 0.05, objective ≈ 0.19, Wolfe/bookkeeping
the rest.

**Target:** one iteration ≈ 10–20 µs → ADNI fit ≈ 2–4 ms (5–10× PCA), Golub
shape (57×2000, k=3) ≈ 10 ms. Anything that does not reach ≤ 30 µs/iteration
on the ADNI shape has not met the goal; measure before reporting.

## 1. What already exists and must not change

- **The algorithm is finished and validated.** `nsa_flow/lbfgsb.py::lbfgsb_minimize`
  is L-BFGS-B (Byrd–Lu–Nocedal–Zhu 1995): generalized Cauchy point over the
  compact representation `B = θI − W M Wᵀ`, `W = [Y, θS]`, `M = [[−D, Lᵀ],[L, θSᵀS]]⁻¹`;
  direct-primal subspace minimization; strong-Wolfe line search with
  cubic interpolation on the feasible segment; projected backtracking fallback.
  It matches SciPy's Fortran to 12 significant figures on 17/18 configurations and
  is strictly lower on the 18th. **The native kernel is a translation of this
  file. It is the bit-exact reference. Do not redesign the algorithm.**
- **The contract** (`nsa_flow/optim.py`): every optimizer takes
  `(Y0, energy_fn, grad_fn, proj, *, max_iter, tol, ...)`, `max_iter` caps
  gradient evaluations, the certificate is
  `gradient_mapping(Y, g, proj) = ‖P(Y − ‖Y‖²g) − Y‖/‖Y‖`
  (`nsa_flow/diagnostics.py`), stop reasons are `grad_map | plateau |
  line_search | max_iter`, and `converged` is set only by `make_result`. The
  native path must return exactly the same `Report` fields
  (`Y, energy, iters, stop, grad_map, n_grad, n_energy, energy_start,
  grad_map_start, seconds`).
- **Pure-Python fallback stays.** `pip install nsa_flow` must keep working with
  no compiler. The native path is selected only when the extension imported
  successfully; `nsa_flow.optim.OPTIMIZERS["lbfgsb"]` dispatches.
- **Tests that must keep passing unchanged:** `tests/test_contract.py`
  (every optimizer × every mode: converges, certifies, respects the cap, same
  metric definitions), `tests/test_optimizer.py`, and the whole suite (372).

## 2. Scope, in phases — measure after each

### Phase 1 — CPU kernel for the L-BFGS-B iteration (objective stays in Python)

**Deliverable:** `nsa_flow/_native/lbfgsb_cpu.cpp` built with
`torch.utils.cpp_extension` (or `setuptools` `CppExtension`), exposing one
function:

```cpp
// One L-BFGS-B "step" given the current gradient: computes the search
// direction from the compact representation and returns it, together with
// the feasibility limit a_max along it. The caller runs the line search.
std::tuple<at::Tensor /*d*/, double /*a_max*/, at::Tensor /*fixed mask*/>
lbfgsb_direction(const at::Tensor& x, const at::Tensor& g,
                 const at::Tensor& lo, const at::Tensor& hi /* may be empty */,
                 const at::Tensor& S, const at::Tensor& Y, double theta,
                 const at::Tensor& M, int64_t max_breakpoints);

// Rebuild M = [[-D, L'], [L, th S'S]]^{-1} from the correction pairs.
at::Tensor lbfgsb_build_M(const at::Tensor& S, const at::Tensor& Y, double theta);
```

Everything inside `_cauchy_point` and `_subspace_min_tensor` becomes plain
C++ loops over `double*` (or `float*`) — the breakpoint walk is a *sequential*
loop with early exit and is trivial in C++; the vectorised cumsum version in
Python exists only because Python loops were the cost. `_build_K` is a 2m×2m
dense solve: use `at::linalg_solve` on the small matrix or write LU by hand.

**Wire-up:** in `lbfgsb.py`, if `nsa_flow._native` imports, `_cauchy_point` +
`_subspace_min` are replaced by one call to `lbfgsb_direction`, and
`_CompactLBFGS._rebuild` by `lbfgsb_build_M`. The Wolfe search, memory
update, plateau/stall logic and certificate stay in Python for this phase.

**Expected gain:** those three pieces are ~0.31 of the 0.69 ms; with the
objective and Wolfe still in Python you should see ~2–2.5×. **This is not the
goal yet.** It is the checkpoint that proves the translation is bit-exact.

**Acceptance for Phase 1:**
- `tests/test_native_parity.py` (new): on the 6 reference instances from
  `experiments/optimizer_study.make_problem` (`anchored/data/signed ×
  planted/random`, small, seed 0) and on 20 random `(x, g, S, Y)` states,
  the native `d`, `a_max`, `fixed` and `M` equal the Python versions to
  `atol=1e-12` (float64). Final energies of full fits equal to 1e-12.
- Whole suite green with the extension present *and* with it absent
  (`NSA_FLOW_DISABLE_NATIVE=1` env var to force the fallback; add it).

### Phase 2 — the whole iteration in one call (objective included)

The callback boundary (Python → C++ → Python per evaluation) caps Phase 1 at
~3×. To reach the target, the *objective and gradient* must be evaluated
inside the kernel so a full iteration — direction, Wolfe search with its
several energy/gradient evaluations, memory update, certificate — is **one**
call.

The three objectives are small closed forms; translate them from the Python
exactly:

| mode | value & gradient | source |
|---|---|---|
| data | `F = (trS − 2 tr A + ⟨A, Bᵀ⟩)/c`, `A = VᵀSV` (or `(XV)ᵀ(XV)`), `B = VᵀV`; `∇F = (2/c)(−2SV + SV·B + V·A)`; plus `w·D̃` | `reconstruct.py::_fid_and_grad`, `energy.py::grad_stiefel_defect` |
| signed | data objective on `V = V₊ − V₋` with `[V₊ V₋] ≥ 0`, orth term `Cg` on the `[p, 2k]` parts, lobe penalty `w·lobe·⟨V₊,V₋⟩/c` | `signed.py::parts_energy / parts_grad`, `angle.py::gram_offdiag_defect` |
| anchored | `(1−w)‖Y−X0‖²/‖X0‖² + w·D̃(Y)` (fidelity `anchor`); subspace fidelity is a separate closed form (`subspace.py`) | `energy.py::value_and_grad`, `subspace.py` |

Matrix-free vs Gram route follows the same `p > n` rule as `GramOperator`.

**Interface:** `lbfgsb_solve(x0, problem_spec, lo, hi, mask, max_grad, tol,
memory, sigma, patience, rtol, stall_slack) -> (x, f, n_grad, n_fun, iters,
stop_code, grad_map, energy_start, grad_map_start)` where `problem_spec`
carries mode, `X` or `S`, `c`, `w`, `orth`, `lobe`, `X0`/`denom` for anchored.
Return the same stop classification as `lbfgsb_minimize` (grad_map / plateau
via `_gmap_stalled` + precision-floor test / line_search / max_iter) — copy
the rules, do not re-derive them; `tests/test_contract.py` pins them.

**Acceptance for Phase 2:** parity to 1e-12 on final energy, `n_grad`, and
`stop` for the 6 reference instances and the contract grid; suite green both
ways; **ADNI shape ≤ 30 µs/iteration, whole fit ≤ 5 ms** (report the number).

### Phase 3 — packaging

- `pyproject.toml`: build the extension optionally (`setuptools` with
  `ext_modules`, `optional=True`); on failure the wheel is pure-Python.
- CI matrix: macOS arm64, Linux x86_64; float32 and float64; with and without
  the extension.
- Prebuilt wheels for those two platforms; sdist falls back.
- Document `NSA_FLOW_DISABLE_NATIVE=1` in the README "Solver" section.

### Phase 4 (optional) — Metal / CUDA

Same kernel in MSL (and CUDA). MPS today: zero host↔device copies but ~3 ms
per gradient from per-kernel latency (~40 launches per gradient), so the
device path benefits even more. Do this only after Phase 2 has hit its number
on CPU; the CPU result is what the paper needs.

## 3. Estimates

|  | human | an LLM in one session |
|---|---|---|
| Phase 1 | 2 days | 2–3 h |
| Phase 2 | 1–2 days | 1–2 h |
| Phase 3 | 1–2 days | half a day → open-ended (toolchain round trips) |
| Phase 4 | 2–3 days | ~1 day |
| tests + benchmark refresh | 0.5 day | 1 h |

The uncertain item is Phase 3, not the kernel. Ship Phase 1+2 behind the
fallback first; wheels can follow.

## 4. Pitfalls already hit in this codebase (save yourself the time)

1. **`clamp_min(1e-300)` on a float32 tensor is 0.0.** Every floor must be in
   the tensor's own dtype (`std::numeric_limits<T>::min()`). This was a
   ZeroDivisionError in the Cauchy walk (`_tiny()` in `lbfgsb.py`).
2. **First step must be scale-free.** With empty memory the direction is
   `−g/‖g‖`, not `−g` — the energy is dimensionless so `‖g‖ ∼ 1/‖x‖`; an unscaled
   first step stalled at `|Gmap| = 0.28` after 4 evaluations and was certified.
3. **A stall is certified only within `1e3·tol` of stationarity**
   (`STALL_SLACK`). Never certify a line-search failure on energy evidence
   alone.
4. **A non-finite trial energy is a rejected step**, never accepted; a
   non-finite start is refused before any iteration (3.1.3).
5. **The curvature condition matters.** Plain backtracking Armijo gave 3.7× the
   iterations of strong Wolfe on the same problem because the `(s, y)` pairs
   were uninformative. Keep the Wolfe search exactly as in `_strong_wolfe`.
6. **`s·y > eps·‖y‖²` before pushing a pair**, else `B` loses positive
   definiteness on the non-convex objectives and you get ascent directions;
   on an ascent direction reset the memory and retry with scaled steepest
   descent once before declaring a stall (see `lbfgsb_minimize`).
7. **Do not make the rank/usability decisions here.** They belong to callers
   (pysimlr's `_usable_retraction`); the kernel returns what it computed.
8. **Verify file state before editing and gate commits on pytest's real exit
   code** (`set -o pipefail`); `pytest | tail` hides failures.

## 5. Benchmarks to re-run when done

```
PYTHONPATH=. python experiments/rapid_primary_benchmarks.py      # 26 s now
PYTHONPATH=. python experiments/adni_cdrsb_repeated.py           # 20 splits
PYTHONPATH=. python experiments/benchmark_new_public_data.py
PYTHONPATH=. python experiments/speed_vs_pca.py --devices cpu,mps
```

Accuracy numbers must be **bit-identical** to 3.1.3 (the solver is
deterministic and the kernel is a translation); only the clocks may move. Put
the new speed table in README §"Speed against PCA" with the same columns, and
state the µs/iteration and ops/iteration alongside, since that is the claim
being made.
