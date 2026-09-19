r"""Single source of truth for every number NSA-Flow reports.

Three kinds of drift had crept into the code base and into the paper's tables,
all of them the same mistake -- a quantity defined more than once:

1. **Three orthogonality registries.**  ``solve._orth_terms_anchor``,
   ``reconstruct._orth_terms`` and an inline ``if/elif`` in ``signed`` each
   mapped the same names ``"D"/"C"/"Cg"`` to functionals, with different
   coverage and different normalisation.  They are now one registry, :data:`ORTH`.

2. **A ``defect`` field that meant a different function in every mode.**
   ``nsa_flow_data`` reported ``C``, ``nsa_flow_signed`` reported ``Cg`` on the
   ``[p, 2k]`` parts, ``nsa_flow`` reported ``Dtilde``, and benchmark tables put
   all three in one column beside a PCA baseline measured with ``Dtilde``.  On a
   disjoint non-negative basis with unequal column norms that is a 0.77 spread
   from the choice of functional alone.  Every result now carries
   :func:`basis_report`, which evaluates *all* the functionals on the returned
   basis, so a table can never accidentally compare two of them.

3. **A stationarity certificate that meant a different function in every
   optimiser.**  SPG reported ``||Y+ - Y|| / t_BB`` with the Barzilai-Borwein
   step, L-BFGS-B reported the *unprojected* ``max |grad|`` (which is nonzero at
   any active bound -- 3.6e-03 at a point whose true residual is 1.0e-05), and
   the Adam/RMSprop probes in the optimiser benchmark divided a *preconditioned*
   step by ``lr``.  There is now one definition, :func:`gradient_mapping`, and
   every loop is scored with it.

The certificate
---------------
For ``min E(Y)`` over a closed convex set with projection ``P``, the gradient
mapping at step ``t`` is ``G_t(Y) = (Y - P(Y - t grad E)) / t``; it vanishes
exactly at the stationary points of the constrained problem.  The step ``t`` is
a free parameter, and the choice is what makes the number comparable or not.

``E`` here is dimensionless by construction (both terms are normalised), so
``grad E`` carries units ``1/[Y]`` and any fixed ``t`` gives a certificate whose
value depends on the scale of ``Y``.  Taking ``t = ||Y||_F^2`` cancels that:
under ``Y -> cY`` we have ``grad E -> grad E / c``, so ``t grad E -> c t grad E``
and the whole mapping is homogeneous of degree one.  Dividing by ``||Y||_F``
leaves

    gmap(Y) = || P(Y - ||Y||_F^2 grad E(Y)) - Y ||_F / ||Y||_F

which is **dimensionless, scale-invariant and zero exactly at stationarity**.
One ``tol`` therefore means the same thing on every problem, every mode and
every optimiser -- which is the same property ``w`` was designed to have.

Unlike the raw KKT residual it also handles near-boundary iterates correctly: a
coordinate sitting at ``1e-8`` with a positive gradient is projected to zero and
contributes ``1e-8``, rather than being charged its full gradient for not being
exactly on the bound.  That matters because the ``Y = Z^2`` reparameterisation
never reaches the boundary exactly.
"""
import math

import torch

from .angle import (angle_defect, grad_angle_defect, gram_offdiag_defect,
                    grad_gram_offdiag_defect)
from .energy import (effective_rank, grad_stiefel_defect, stiefel_defect,
                     stiefel_defect_normalised)
from .project import project_nonneg

__all__ = [
    "ORTH", "orth_terms", "orth_names",
    "gradient_mapping", "basis_report", "support_stats",
    "make_result", "CERTIFICATES", "CONVERGED_STOPS", "STOP_REASONS",
    "default_tol",
]

#: Stop reasons any loop may return.  ``grad_map`` is the only positive claim of
#: stationarity; the others describe why iteration ceased.
STOP_REASONS = ("grad_map", "plateau", "line_search", "max_iter")

#: Stop reasons that are permitted to set ``converged``, and the certificate
#: each one carries.  Everything else leaves ``converged`` False.
#:
#: ``"grad_map"`` -> ``"stationary"``
#:     ``grad_map <= tol``.  The strong claim: ``Y`` satisfies the first-order
#:     conditions of the constrained problem to the requested tolerance.
#: ``"plateau"`` -> ``"numerical_floor"``
#:     The energy has stopped moving *and* the certificate has stopped
#:     improving (see :func:`nsa_flow.optim._plateau`).  Weaker but still a
#:     statement about ``Y``: no further progress is available in this working
#:     precision.  Earlier versions set ``converged`` on an energy plateau
#:     alone, which certified points 3 to 7 orders from stationarity whose
#:     supports were still moving.
CERTIFICATES = {"grad_map": "stationary", "plateau": "numerical_floor"}
CONVERGED_STOPS = tuple(CERTIFICATES)


def default_tol(dtype):
    """Working-precision default for the stationarity tolerance: "good enough".

    ``1e-6`` in float64, ``1e-4`` in float32.  Both are on the shared
    scale-invariant certificate, so they mean the same thing on every problem.

    Chosen by measurement, not by taste.  On the medium synthetic grid
    (anchored / data / signed x planted / random, k=10) against a reference
    solved to ``1e-9``:

        tol     support Jaccard (min)   min |cos|   gradient evals saved
        1e-6           0.985            identical        30 - 50 %
        1e-5           0.887            0.932            (data/random breaks)
        1e-4           0.422            0.393            unacceptable

    ``1e-6`` returns the same basis as ``1e-9`` for a fraction of the work;
    ``1e-5`` already changes the support on the hard family.  In float32 the
    certificate bottoms out around ``1e-4`` (eps = 1.2e-7 on a quartic), so the
    float32 default is the precision floor rather than a quality choice, and
    the plateau detector is what actually ends those solves.  Pass ``tol``
    explicitly when the support matters more than the wall clock.
    """
    return 1e-6 if dtype == torch.float64 else 1e-4


# --------------------------------------------------------------------------
# the orthogonality functionals, defined once
# --------------------------------------------------------------------------
def _grad_D(Y):
    k = Y.shape[-1]
    if k <= 1:
        return torch.zeros_like(Y)
    return (1.0 / (1.0 - 1.0 / k)) * grad_stiefel_defect(Y)


def _C_off(Y):
    return angle_defect(Y, diagonal=False)


def _grad_C_off(Y):
    return grad_angle_defect(Y, diagonal=False)


#: ``name -> (value, gradient, one-line description)``.
#:
#: Every entry is normalised to ``[0, 1]`` with ``0`` on the structure the method
#: exists to produce and ``1`` at full collinearity, so the convex weight ``w``
#: means the same thing whichever is selected.  They are *not* interchangeable
#: as reported diagnostics -- see :func:`basis_report`.
ORTH = {
    "D": (stiefel_defect_normalised, _grad_D,
          "full Stiefel defect ||G - I/k||^2: decorrelation AND equal column norms"),
    "Cg": (gram_offdiag_defect, grad_gram_offdiag_defect,
           "decorrelation half of D: ||offdiag(V'V)||^2 / tr(V'V)^2, smooth everywhere"),
    "C": (angle_defect, grad_angle_defect,
          "mean squared cosine plus a dead-column penalty; discontinuous at a zero column"),
    "Coff": (_C_off, _grad_C_off,
             "mean squared cosine only; discontinuous at a zero column"),
}


def orth_names():
    """The valid ``orth`` selectors, in a stable order."""
    return tuple(ORTH)


def orth_terms(orth):
    """``(value_fn, grad_fn)`` for ``orth``; raises with the full valid set."""
    try:
        val, grad, _ = ORTH[orth]
    except KeyError:
        raise ValueError(
            f"orth must be one of {orth_names()}; got {orth!r}") from None
    return val, grad


# --------------------------------------------------------------------------
# the stationarity certificate, defined once
# --------------------------------------------------------------------------
def gradient_mapping(Y, grad, proj=project_nonneg):
    r"""``|| P(Y - ||Y||_F^2 grad) - Y ||_F / ||Y||_F`` -- see the module docstring.

    Dimensionless, invariant under ``Y -> cY``, and zero exactly at a stationary
    point of ``min E(Y) s.t. Y in dom(proj)``.  Pass ``proj=None`` for an
    unconstrained problem, where this reduces to ``||Y||_F ||grad||_F``.

    Returns a Python float so it is directly comparable across optimisers,
    modes, devices and dtypes.
    """
    nY = float(Y.norm())
    if not math.isfinite(nY) or nY <= 0.0:
        return float("inf")
    step = Y - (nY * nY) * grad
    if proj is not None:
        step = proj(step)
    return float((step - Y).norm()) / nY


# --------------------------------------------------------------------------
# the reported diagnostics, defined once
# --------------------------------------------------------------------------
def support_stats(Y, rel_tol=1e-8):
    """Support size, sparsity and cross-component overlap of a loading matrix.

    ``rel_tol`` is relative to the largest magnitude in the whole matrix, so the
    statistic is scale-invariant like everything else here.  An entry at exactly
    zero and an entry at ``1e-12`` of the peak are both "unused"; counting only
    exact zeros overstates the support badly, because a finite solve stops at a
    small nonzero defect rather than at zero.
    """
    A = Y.abs()
    peak = float(A.max()) if A.numel() else 0.0
    thr = rel_tol * peak
    used = A > thr
    per_feature = used.sum(-1)
    live = per_feature > 0
    nnz = used.sum(-2)
    return dict(
        sparsity=float((~used).to(Y.dtype).mean()),
        max_support=int(nnz.max()) if nnz.numel() else 0,
        mean_support=float(nnz.to(Y.dtype).mean()) if nnz.numel() else 0.0,
        n_dead_columns=int((nnz == 0).sum()),
        # mean components per active feature, minus one: 0 == a clean partition
        support_overlap=(float(per_feature[live].to(Y.dtype).mean()) - 1.0
                         if bool(live.any()) else 0.0),
    )


def basis_report(Y, rel_tol=1e-8):
    """Every canonical diagnostic of a basis ``Y``, under fixed definitions.

    Returned on *every* result from *every* mode, so a benchmark table never has
    to decide which defect it is looking at and can never mix two of them.

    Keys
    ----
    ``defect_D``, ``defect_Cg``, ``defect_C``, ``defect_Coff``
        The orthogonality functionals of :data:`ORTH`, all in ``[0, 1]``, all
        evaluated on the *same* returned basis.  ``defect_D`` is the one to use
        when comparing against PCA, NMF or sparse PCA, because it is the only
        one of the four that is not invariant to column rescaling and is
        therefore the strictest.
    ``raw_defect``
        Unnormalised ``D = ||G - I/k||_F^2``, in ``[0, 1 - 1/k]``.  Kept because
        the theory bounds are stated for it.
    ``effective_rank``
        Participation ratio ``k / (k D + 1)``, in ``[1, k]``.
    ``column_norm_ratio``
        ``max_i ||y_i|| / min_i ||y_i||``.  ``D`` charges for this and the other
        three functionals do not, which is the whole reason they differ.
    ``sparsity``, ``max_support``, ``mean_support``, ``n_dead_columns``,
    ``support_overlap``
        From :func:`support_stats`.
    """
    out = {f"defect_{name}": float(fn(Y)) for name, (fn, _, _) in ORTH.items()}
    out["raw_defect"] = float(stiefel_defect(Y))
    out["effective_rank"] = float(effective_rank(Y))
    cn = Y.norm(dim=-2)
    lo = float(cn.min())
    out["column_norm_ratio"] = (float(cn.max()) / lo) if lo > 0 else float("inf")
    out.update(support_stats(Y, rel_tol=rel_tol))
    return out


# --------------------------------------------------------------------------
# the result object, built once
# --------------------------------------------------------------------------
def make_result(cls, *, Y, w, orth, energy, fidelity, defect, iters, stop,
                grad_map, tol, seconds, mode, optimizer, energy_start=None,
                grad_map_start=None, **extra):
    """Assemble an ``NSAResult`` with uniform field semantics.

    Every solver entry point builds its result here, so the meaning of
    ``defect``, ``grad_map`` and ``converged`` cannot drift between them.

    ``defect`` is the value of *the functional that was optimised*, named by
    ``orth``; the report adds ``defect_D``/``defect_Cg``/``defect_C``/
    ``defect_Coff`` evaluated on the same basis for comparison purposes.

    ``converged`` is set only by :data:`CONVERGED_STOPS`, and ``"grad_map"``
    additionally requires ``grad_map <= tol``.  ``certificate`` names *which*
    claim is being made, so a caller can require the strong one.  A line-search
    stall and an iteration cap are reasons iteration ceased, never evidence that
    ``Y`` is a solution; each has at some point been reported as success by this
    code base while returning a point orders from stationarity.
    """
    gm = float(grad_map)
    if stop == "grad_map" and math.isfinite(gm) and gm <= tol:
        certificate = "stationary"
    elif stop == "plateau" and math.isfinite(gm):
        certificate = "numerical_floor"
    else:
        certificate = "none"
    res = cls(
        Y=Y, w=float(w), orth=str(orth), mode=str(mode), optimizer=str(optimizer),
        energy=float(energy), fidelity=float(fidelity), defect=float(defect),
        iters=int(iters), stop_reason=str(stop), grad_map=gm, tol=float(tol),
        converged=(certificate != "none"), certificate=certificate,
        seconds=float(seconds),
        # Where the solve started, so "did this actually optimise anything?" is
        # answerable from the result rather than needing a traced re-run.
        energy_start=(float("nan") if energy_start is None else float(energy_start)),
        grad_map_start=(float("nan") if grad_map_start is None
                        else float(grad_map_start)),
        energy_reduction=(float("nan") if energy_start is None
                          else float(energy_start) - float(energy)),
    )
    res.update(basis_report(Y))
    res.update(extra)
    return res
