r"""Signed NSA-Flow: each component a contrast of two non-negative parts.

A non-negative component cannot encode a contrast.  On the ADNI analysis matrix
the leading eigenvalue (15.6) is global size and components 2-5 (1.61, 1.38,
1.23, 1.00) are regional contrasts -- exactly the shape a single non-negative
column cannot represent -- which is the structural ceiling the purely
non-negative basis keeps hitting.

Lift instead:  V = V+ - V-,   V+ >= 0, V- >= 0,

so a component reads as "these regions minus those regions".  Both parts stay
non-negative and interpretable, and the representational capacity of a signed
basis is restored.  Write ``W = [V+ | V-]`` of shape ``[p, 2k]``.  Then

    minimise  (1-w) ||X - X V V'||_F^2 / ||X||_F^2  +  w Dtilde(W),   W >= 0

with ``V = V+ - V-``.  The orthogonality term acts on the ``2k`` *parts*, which is what delivers the
interpretability: near-disjoint lobe supports mean each component names one set
of features against a second, and different components use different features.
Orthogonality of the *signed* ``V`` would not do this, because the
disjoint-support equivalence needs non-negativity and ``V`` is signed by
construction.

Getting the term right took two corrections, both forced by measurement.

``D = ||G - I/(2k)||^2`` was wrong because it demands ``G_ii = 1/(2k)``, i.e. all
``2k`` parts of EQUAL norm.  A contrast is legitimately asymmetric -- a
mostly-positive component has a small or empty negative lobe -- so on ADNI the
part-norm ratio at ``w = 0`` is ~4e8, and raising ``w`` inflates the near-empty
lobes to equalise them: lobe overlap ROSE from 3e-9 to 2.55 between ``w = 0`` and
``w = 0.25``.  Switching to the squared-cosine defect ``C``, which is indifferent
to norms, cut that to 0.122 -- a 21x improvement for 0.4% reconstruction cost.

``C`` with its diagonal was still wrong, more subtly.  Its diagonal charges
``1/(2k(2k-1))`` per column whose norm has been floored, which doubles as a
dead-column penalty -- useful for a plain basis, wrong here, because an empty
negative lobe is the CORRECT answer for a one-signed component such as global
atrophy.  The default is therefore ``C`` restricted to off-diagonal angles
(``angle_defect(..., diagonal=False)``).  Collapse is prevented by the
reconstruction term instead, which is where that job belongs: a dead component
reconstructs nothing.

``lobe`` adds ``sum_i <v+_i, v-_i>`` to forbid a component contrasting a feature
against itself.  The off-diagonal angle term already pushes every pair of lobes
apart, including each component's own pair, so this is a refinement rather than a
necessity; ``lobe=1.0`` drives the overlap to exactly zero.

DEFAULT orth IS NOW "Cg", THE SMOOTH DEFECT.  ``Coff`` traps a descent method:
``C`` is a function of column directions only, so it is discontinuous at a zero
column and an all-zero lobe is a spurious local minimum no step can leave.
``Cg`` mass-weights those cosines by each column's energy share, which cancels
the ``|v_i||v_j|`` and leaves ``||offdiag(V'V)||^2 / tr(V'V)^2`` -- no per-column
normalisation, so it is smooth everywhere, and it is exactly the decorrelation
half of ``D`` (orthogonality without norm balance, which is what ``C`` was for).
Measured over 18 dataset/k/w combinations (ADNI centred and raw, METABRIC):

    orth    reached stationarity   dead lobes
    Coff          7 of 18          1 to 3 in 11 cases
    Cg           12 of 18          0 in ALL 18

and the six remaining ``Cg`` cases are iteration budget, not traps: on the worst
of them |Gmap| falls 7.2e-05 -> 6.2e-06 -> 1.08e-08 as ``max_iter`` goes 2000 ->
8000 -> 20000, converging at 15973 iterations in 13 s.  Every ``Coff`` failure
is a line-search stall at 1e-02 to 1e-03 after 13 to 72 iterations.

FIXED in 2.8.0: the solver used to exit after one iteration.  Recorded because
every signed result produced before this release describes an unoptimised
starting point, not a solution.

The cause was an interaction, not a single mistake.  ``C(diagonal=False)`` is a
function of column directions only and is therefore discontinuous at a zero
column (see ``nsa_flow.angle``), which makes an all-zero lobe a spurious local
minimum that no descent step can leave.  ``init="relax"`` used to build

    W = cat([V0.clamp_min(0), zeros_like(V0)])

putting ``V-`` exactly on that discontinuity.  Energy at the start was 0.1543
and every projected step of any length evaluated to about 0.2182, so the Armijo
line search correctly rejected all 60 halvings and the loop exited at iteration
one.  It then reported ``grad_map=inf`` -- the loop's sentinel, assigned only
after an accepted step -- with ``converged=True``.

``init="relax"`` now seeds both lobes from the relaxed solution,
``V- = (-V0).clamp_min(0)``.  On ADNI thickness (centred, ``k = 5``,
``w = 0.5``) that converges in 446 iterations against 1407 for ``init="split"``,
at a better certificate (1.11e-09 against 2.75e-09), better sparsity (0.433
against 0.427) and better reconstruction (0.2920 against 0.2923).

``init="split"`` is NOT a safe alternative on non-negative uncentred data: it
stalls at two iterations there, with reconstruction up to twelve times worse
(ADNI raw, ``k = 2``, ``w = 0.9``: 0.1117 against 0.0095).  An earlier version
of this docstring claimed the lobes "do not survive a nonzero w" and presented a
sparsity-versus-contrast trade-off; both were artifacts of comparing a converged
solve against a stalled one, and neither is true.

A stall far from stationarity remains possible -- the geometry can genuinely
trap the iterate on strongly positive data -- but it is no longer silent: the
certificate is measured rather than sentinel, ``converged`` requires it to be
finite, and a ``RuntimeWarning`` names the iteration count and ``|Gmap|``.

Capacity.  At ``w = 0`` the lifting reproduces signed PCA's reconstruction to the
digit (0.5723 against 0.572269 on ADNI volumes), settling the question the
construction was built to answer: the purely non-negative ceiling of 0.591 is
representational, not an optimisation failure.

Sparsity.  The relaxation alone does not deliver the practical goal, which is
that BOTH lobes be sparse.  Stopping at an angle defect of ``1e-8`` rather than
``0`` leaves every feature a little weight in several parts, so the parts come out
concentrated but not sparse: on ADNI cortical thickness the largest part uses 39
of 66 features against a disjoint ideal of 6.  ``consolidate=True`` assigns each
feature to its largest part, zeroing it elsewhere, then re-runs the solver with
the support held fixed.  At ``w = 0.5`` that gives a largest part of 18, a mean of
9.1, ``V+`` and ``V-`` each about 10% dense, exactly disjoint supports and no
component lost (effective rank 5.00 of 5).  Lobes may die, which is correct: a
one-signed component such as global atrophy should not be forced to carry a
negative lobe.

Choice of ``w``.  Moderate, and this matters more than it looks.  Across nine
ADNI cognitive outcomes, 20 paired folds each, mean out-of-sample \(\Delta R^2\)
against PCA over an age, sex, education and APOE4 baseline:

                        w = 0.5                 w = 0.75
    basis          linear      forest      linear      forest
    signed        +0.0098     +0.0473     -0.0134     +0.0370
    consolidated  +0.0113     +0.0443     -0.0175     +0.0249
    subspace      +0.0010     +0.0398     +0.0030     +0.0456
    data          +0.0028     +0.0464     +0.0024     +0.0441

At ``w = 0.5`` the lifting is the best of the four variants under both models.  At
``w = 0.75`` it is worse than PCA on 0 of 9 outcomes under a linear model.  Since
a linear model on projected scores is exactly invariant to reparametrising the
basis, it sees only the span, so the collapse is a loss of span quality: pushing
\(w\) up past about 0.5 rounds the contrasts toward a partition that no longer
spans what the data needs.  ``w = 0.5`` is also the best setting for sparsity
(largest part 18, against 22 at ``w = 0.75``), so the two objectives do not
conflict here and there is nothing to trade off.

Replication.  PPMI was tried as a second cohort and is not evidence either way.
It was picked for being on hand rather than for being a good test of anything
here, and it does not have the signal to settle a basis comparison: with all 66
features and every confound in the model, the ceiling over a confound-only model
is +0.067 AUC for SAA and R^2 ~ 0 for UPDRS-I, against between-basis differences
of about 0.015.  ``experiments/exp20`` (retracted, confounded) and
``experiments/exp22`` (confounds modelled, null) hold the record; neither belongs
in a claim about the method and neither is in the paper.

Replication on confound-light data splits, and it splits on ``p``.  Two UCI
sets (``experiments/exp21_uci_replication.py``) were run instead, chosen because
they are single-source with no acquisition covariates to adjust for.  On
diabetes (``n = 442``, ``p = 10``, regression) the lifting beats PCA by
``+0.013`` R^2 under a linear model and ``+0.044`` under a forest, both
``p < 0.002``, and consolidation is neutral.  On Cleveland heart disease
(``n = 297``, ``p = 13``, classification) PCA wins: the lifting loses ``0.007``
to ``0.012`` AUC and consolidation loses ``0.017`` to ``0.018``, small but
consistent across folds.  At ``p = 13`` with ``k = 3`` a disjoint partition
leaves about four features per part, so the consolidated basis is close to a
hard feature partition and pays for it.  The honest summary across all four
datasets is that these bases are competitive-to-better on regression and pay a
modest reproducible price on small-``p`` classification, and that consolidation
is a sparsity control rather than an accuracy one.

Caveats.  The positive imaging result is one cohort, one modality (``p = 66``,
``n ~ 300``), and nine outcomes that share subjects and are therefore not nine
independent tests.  The two UCI sets go one each way.  There is no second
imaging cohort with enough signal to replicate or refute it.  What is solid is the sparsity behaviour, which is structural rather than
statistical: consolidation gives exactly disjoint supports in both lobes at
roughly 10% density with no component lost, on every dataset tried.  What is not
established is that this buys predictive accuracy in general.  ``w`` and ``k``
were not selected by nested cross-validation anywhere.

Gradient.  With ``F(V)`` the reconstruction term, ``dF/dV+ = dF/dV`` and
``dF/dV- = -dF/dV`` by the chain rule, so the data term costs one extra sign flip
and the defect term acts on ``W`` directly.
"""
import math
import time
import warnings

import torch

from .angle import (angle_defect, grad_angle_defect,
                    gram_offdiag_defect, grad_gram_offdiag_defect)
from .energy import (grad_stiefel_defect, stiefel_defect,
                     stiefel_defect_normalised, effective_rank)
from .project import project_nonneg
from .reconstruct import (grad_reconstruction_fidelity, reconstruction_fidelity,
                          relax_into_nonneg)
from .solve import NSAResult

__all__ = ["nsa_flow_signed", "consolidate_supports", "part_sparsity"]


def part_sparsity(W, rel_tol=0.0, abs_tol=1e-10):
    r"""Per-part support statistics for ``W = [V+ | V-]``.

    The practical goal for the lifting is that BOTH lobes be sparse, so the
    aggregate fraction of zeros is the wrong summary: it is satisfied by one
    empty lobe and one dense one.  These are the quantities that are not.

    ``rel_tol`` counts an entry as used only if it exceeds that fraction of its
    part's largest entry.  Driving the angle defect to ``1e-8`` rather than to
    ``0`` leaves a tail of small but non-zero entries, so the count at
    ``rel_tol = 0`` overstates the support considerably.
    """
    A = W.abs()
    thr = (rel_tol * A.amax(dim=-2, keepdim=True)) if rel_tol else abs_tol
    used = A > thr
    nnz = used.sum(-2)
    live = nnz > 0
    top = A.sort(dim=-2, descending=True).values
    kpart = max(1, W.shape[-2] // W.shape[-1])          # p / 2k, the disjoint ideal
    mass = top[:kpart].sum(-2) / A.sum(-2).clamp_min(1e-300)
    return dict(
        nnz_per_part=nnz,
        n_dead=int((~live).sum()),
        max_nnz=int(nnz.max()),
        mean_nnz=float(nnz[live].to(W.dtype).mean()) if bool(live.any()) else 0.0,
        disjoint_ideal=kpart,
        mass_in_ideal=float(mass.mean()),
    )


def consolidate_supports(W):
    r"""Round the relaxed parts to exactly disjoint supports.

    For ``W >= 0`` the angle defect vanishes exactly when the parts have pairwise
    disjoint supports, so the relaxation's ideal endpoint is a hard assignment of
    each feature to one part.  A finite run stops at a small but non-zero defect,
    which leaves every feature a little weight in several parts; that tail is why
    the parts look concentrated but not sparse.  On ADNI thickness at ``w = 0.75``
    the largest part has 39 non-zero entries of 66 before rounding and 22 after,
    with the mean falling from 15.4 to 8.2 against a disjoint ideal of 6.

    Each feature is assigned to the part holding its largest magnitude and zeroed
    elsewhere.  Magnitudes are NOT rescaled here: with the supports fixed the
    objective is still quartic in the parts, so a closed-form rescale is wrong.
    Use ``nsa_flow_signed(..., consolidate=True)``, which re-runs the solver with
    the support held fixed and recovers most of the cost.
    """
    A = W.abs()
    win = A.argmax(dim=-1, keepdim=True)
    mask = torch.zeros_like(W, dtype=torch.bool).scatter_(-1, win, True)
    mask &= A > 0
    return W * mask


def _split(W):
    k = W.shape[-1] // 2
    return W[..., :k], W[..., k:]


def nsa_flow_signed(X, k=None, w=0.5, *, init="relax", orth="Cg", lobe=1.0,
                    max_iter=5000, tol=None, sigma=1e-4, dtype=None, device=None,
                    verbose=False, keep_trace=False, consolidate=False):
    """Fit ``V = V+ - V-`` with ``[V+|V-] >= 0`` near-disjoint, reconstructing ``X``.

    Returns an ``NSAResult`` whose ``Y`` is the signed ``V`` of shape ``[p, k]``;
    ``parts`` holds the ``[p, 2k]`` non-negative ``W = [V+|V-]``.
    """
    Xt = torch.as_tensor(X)
    if not torch.is_floating_point(Xt):
        Xt = Xt.double()
    if dtype is not None:
        Xt = Xt.to(dtype)
    if device is not None:
        Xt = Xt.to(device)
    Xt = Xt.detach()
    if Xt.ndim != 2:
        raise ValueError(f"X must be 2-D [n, p]; got {tuple(Xt.shape)}")
    if not torch.isfinite(Xt).all():
        raise ValueError("X contains non-finite values")
    if not (0.0 <= float(w) <= 1.0):
        raise ValueError(f"w must lie in [0, 1]; got {w}")

    n, p = Xt.shape
    S = Xt.transpose(-2, -1) @ Xt
    c = S.diagonal().sum()
    if float(c) <= 0:
        raise ValueError("X is all zeros; fidelity is undefined")
    if tol is None:
        tol = 1e-9 if Xt.dtype == torch.float64 else 1e-6

    if isinstance(init, str):
        if k is None:
            raise ValueError("give k when init is a strategy name")
        evals, evecs = torch.linalg.eigh(S)
        E = evecs[:, -k:].flip(-1)
        if init == "split":
            # The exact signed basis, losslessly: V+ = max(0,E), V- = max(0,-E).
            W = torch.cat([E.clamp_min(0.0), (-E).clamp_min(0.0)], dim=-1).clone()
        elif init == "relax":
            # V- must NOT start at exactly zero.  C(diagonal=False) depends only
            # on column directions, so it is discontinuous at a zero column and
            # an all-zero lobe is a spurious local minimum the line search
            # cannot leave.  Seeding both lobes from the relaxed solution costs
            # nothing and converges in 446 iterations against 1407 for "split"
            # on ADNI thickness (k=5, w=0.5), at a better grad_map, sparsity and
            # reconstruction.  See the module docstring.
            V0 = relax_into_nonneg(S, c, k, float(w), trS=c)
            W = torch.cat([V0.clamp_min(0.0), (-V0).clamp_min(0.0)],
                          dim=-1).clone()
        else:
            raise ValueError(f"unknown init strategy {init!r}")
    else:
        W = torch.as_tensor(init).to(dtype=Xt.dtype, device=Xt.device).detach().clone()
        k = W.shape[-1] // 2
    if W.shape != (p, 2 * k):
        raise ValueError(f"init shape {tuple(W.shape)} != [p, 2k] = {(p, 2 * k)}")

    inv_k = 1.0 / (1.0 - 1.0 / (2 * k))
    if orth == "Cg":            # default: smooth; no zero-column discontinuity
        o_val, o_grad = gram_offdiag_defect, grad_gram_offdiag_defect
    elif orth == "Coff":        # pairwise angles only; traps at a zero lobe
        o_val = lambda Wv: angle_defect(Wv, diagonal=False)
        o_grad = lambda Wv: grad_angle_defect(Wv, diagonal=False)
    elif orth == "C":           # angles plus a dead-lobe penalty
        o_val, o_grad = angle_defect, grad_angle_defect
    elif orth == "D":           # orthoNORMality; retained only for the ablation
        o_val = stiefel_defect_normalised
        def o_grad(Wv):
            return inv_k * grad_stiefel_defect(Wv)
    else:
        raise ValueError(
            f"orth must be 'Cg', 'Coff', 'C' or 'D'; got {orth!r}")

    def parts_energy(Wv):
        Vp, Vm = _split(Wv)
        f = reconstruction_fidelity(Vp - Vm, S, c, c)
        d = o_val(Wv)
        e = (1.0 - w) * f + w * d
        if lobe:
            e = e + lobe * (Vp * Vm).sum() / c
        return e, f, d

    def parts_grad(Wv):
        Vp, Vm = _split(Wv)
        gV = (1.0 - w) * grad_reconstruction_fidelity(Vp - Vm, S, c)
        g = torch.cat([gV, -gV], dim=-1)
        if w != 0.0:
            g = g + w * o_grad(Wv)
        if lobe:
            g = g + (lobe / c) * torch.cat([Vm, Vp], dim=-1)
        return g

    W = project_nonneg(W)
    E, F, D = parts_energy(W)
    E = float(E)
    g = parts_grad(W)
    t = 1.0 / max(float(g.norm()), 1e-12)
    W_prev = g_prev = None
    trace = [] if keep_trace else None
    gmap, stop, it = float("inf"), "max_iter", 0
    t0 = time.time()

    for it in range(1, max_iter + 1):
        if W_prev is not None:
            s_ = W - W_prev
            r_ = g - g_prev
            sr = float((s_ * r_).sum())
            t = float((s_ * s_).sum()) / sr if sr > 0 else 1e12
            t = min(max(t, 1e-12), 1e12)
        accepted = False
        t_first, dn2_first = t, None
        for _ in range(60):
            W_new = project_nonneg(W - t * g)
            d_ = W_new - W
            dn2 = float((d_ * d_).sum())
            if dn2_first is None:
                dn2_first = dn2
            if float(parts_energy(W_new)[0]) <= E - sigma * dn2 / t:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            stop = "line_search"
            # Report a measured certificate rather than the sentinel.  Without
            # this a first-iteration failure returns grad_map=inf, which reads
            # as a computed value and let a stalled solve claim convergence.
            if not math.isfinite(gmap):
                gmap = (dn2_first ** 0.5) / t_first
            # A finite but large certificate is not stationarity.  The caller
            # cannot be expected to inspect grad_map on every call, so say so.
            if gmap > max(tol, 0.0) * 1e3:
                warnings.warn(
                    "nsa_flow_signed: line search stalled after "
                    f"{it} iteration(s) with |Gmap|={gmap:.2e} against "
                    f"tol={tol:.1e}; the returned point is not stationary. "
                    "Inspect stop_reason and grad_map.",
                    RuntimeWarning, stacklevel=3)
            break
        gmap = (dn2 ** 0.5) / t
        W_prev, g_prev = W, g
        W = W_new
        E, F, D = parts_energy(W)
        E = float(E)
        g = parts_grad(W)
        if trace is not None:
            trace.append(dict(iter=it, energy=E, fidelity=float(F),
                              defect=float(D), grad_map=gmap, step=t))
        if verbose and (it % max(1, max_iter // 10) == 0 or it == 1):
            print(f"    [w={w:.3f} it={it:5d}] E={E:.8e} |Gmap|={gmap:.3e}")
        if gmap <= tol:
            stop = "grad_map"
            break

    if consolidate:
        # Round to exactly disjoint supports, then keep optimising with the
        # support fixed.  The projection onto {W >= 0, support subset of mask} is
        # the clamp followed by the mask, so the same line search applies.
        mask = (consolidate_supports(W) != 0)
        proj_masked = lambda A: A.clamp_min(0.0) * mask
        W = proj_masked(W)
        E = float(parts_energy(W)[0])
        g = parts_grad(W)
        t = 1.0 / max(float(g.norm()), 1e-12)
        Wp = gp = None
        for _ in range(max_iter):
            if Wp is not None:
                s_, r_ = W - Wp, g - gp
                sr = float((s_ * r_).sum())
                t = min(max(float((s_ * s_).sum()) / sr if sr > 0 else 1e12,
                            1e-12), 1e12)
            ok = False
            for _ in range(60):
                W_new = proj_masked(W - t * g)
                d2 = float((W_new - W).pow(2).sum())
                if float(parts_energy(W_new)[0]) <= E - sigma * d2 / t:
                    ok = True
                    break
                t *= 0.5
            if not ok:
                stop = "line_search"
                if math.isfinite(gmap):
                    gmap = (d2 ** 0.5) / t
                break
            Wp, gp = W, g
            W = W_new
            E = float(parts_energy(W)[0])
            g = parts_grad(W)
            gmap = (d2 ** 0.5) / t
            if gmap <= tol:
                stop = "grad_map"
                break

    if consolidate:
        E, F, D = parts_energy(W)          # F and D were pre-consolidation
        E = float(E)

    Vp, Vm = _split(W)
    V = Vp - Vm
    r = NSAResult(
        Y=V, target=None, w=float(w), energy=E, fidelity=float(F),
        defect=float(D), raw_defect=float(stiefel_defect(W)),
        effective_rank=float(effective_rank(W)),
        scale_ratio=float("nan"), iters=it,
        converged=stop != "max_iter" and math.isfinite(gmap),
        stop_reason=stop, grad_map=float(gmap), seconds=time.time() - t0,
        w_schedule=[float(w)], trace=trace, nonneg=True, align=False,
    )
    r["parts"] = W
    r["lobe_overlap"] = float((Vp * Vm).sum())        # -> 0 as D(W) -> 0
    r["consolidated"] = bool(consolidate)
    r.update({f"parts_{key}": val for key, val in part_sparsity(W).items()
              if key != "nnz_per_part"})
    r["parts_nnz"] = part_sparsity(W)["nnz_per_part"]
    return r
