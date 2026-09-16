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

Spectral projected gradient: Barzilai–Borwein steps with Armijo backtracking on
the projected step. Every accumulation point is a stationary point of the
constrained problem, and `result.grad_map` is a computable stationarity
certificate (`result.stop_reason` says why it stopped — never a silent claim of
convergence).

The inner loop forms one Gram product and two `[p,k] x [k,k]` products: `O(p k^2)`,
with **no SVD, eigendecomposition or QR**. Typical convergence is 15–300
deterministic iterations. Pass `compile=True` for a 3–4x speedup via
`torch.compile` at moderate sizes.

Empirically `E_w` has a unique optimum for `w < 1` — 24 random restarts agree to
machine precision on every problem family tested — so there are no restarts,
schedules or step-size heuristics to tune.

## Torch layers

Two routes, both sound:

```python
from nsa_flow import NSAFlowLinear, NSAFlowConv2d

# (preferred) penalty: a standard layer plus a regulariser
layer = NSAFlowLinear(256, 32)
loss  = task_loss(layer(x), y) + 0.1 * layer.defect()

# (parameterisation) effective weight is blended toward the projection
layer = NSAFlowLinear(256, 32, w=0.5)   # w is the true blend fraction
```

`polar_factor` and `project_scaled_stiefel` carry an explicit
Sylvester-equation derivative. Differentiating `torch.linalg.svd` divides by
`sigma_i^2 - sigma_j^2` and returns NaN at repeated singular values — which is
exactly what `nn.init.orthogonal_` produces. The polar factor is smooth wherever
`Y` has full column rank; its derivative divides by `h_i + h_j > 0`.

## API

| Function | Purpose |
|---|---|
| `nsa_flow(target, w, ...)` | solve; returns `NSAResult` |
| `stiefel_defect(Y)` | `D(Y)` |
| `stiefel_defect_normalised(Y)` | `D(Y)/(1-1/k)`, in `[0,1]` |
| `effective_rank(Y)` | `k/(kD+1)`, in `[1,k]` |
| `energy` / `grad_energy` / `value_and_grad` | `E_w` and its gradient |
| `project_nonneg` / `project_scaled_stiefel` / `polar_factor` | projections |
| `NSAFlowLinear` / `NSAFlowConv2d` / `NSAFlowLayer` | torch layers |

For `k > p`, orthonormal columns are impossible and `inf D = 1/p - 1/k > 0`;
this is reported rather than hidden behind a silently row-orthonormal answer.

## Install, test, reproduce

```bash
make install       # editable install with experiment + test extras
make test          # 133 assertions
make theory        # just the property battery (70 assertions)
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
