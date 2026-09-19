"""NSA-Flow solver: spectral projected gradient on a single, calibration-free energy.

The problem solved is

    minimise   E_w(Y) = (1 - w) ||Y - X0||_F^2 / ||X0||_F^2  +  w Orth(Y)
    subject to Y >= 0                                        (when ``nonneg``)

where ``Orth`` is one of three orthogonality functionals selected by ``orth``:

* ``"D"`` (default):  ``Dtilde = ||G - I/k||_F^2 / (1 - 1/k)`` for
  the trace-normalised Gram ``G = Y'Y / tr(Y'Y)``.
* ``"Cg"``:  ``||offdiag(V'V)||_F^2 / tr(V'V)^2``, smooth everywhere.
* ``"C"``:  mean squared cosine between column pairs, per-column normalised.

The default optimiser is L-BFGS-B implemented in pure PyTorch
(:mod:`nsa_flow.lbfgsb`); ``spg``, ``fista`` and ``pqn`` are available via
``optimizer=``.  All of them are scored by ONE certificate,
:func:`nsa_flow.diagnostics.gradient_mapping` -- scale-invariant, projected --
and stop for one of four reasons:

* ``"grad_map"``: certificate <= ``tol``  (``certificate="stationary"``).
* ``"plateau"``: energy AND certificate both stopped improving
  (``certificate="numerical_floor"``).
* ``"line_search"``: no descent step at working precision, far from
  stationarity -- NOT converged; a ``RuntimeWarning`` names the certificate.
* ``"max_iter"``: the gradient-evaluation budget ran out -- NOT converged.

``result.converged`` is true only in the first two cases.  No SVD or
eigendecomposition appears in the loop; the top-k initialisation is exact and
on-device (:mod:`nsa_flow.linalg`).
"""
import time
import warnings

import torch

from .diagnostics import default_tol, make_result, orth_terms
from .energy import (energy, stiefel_defect_normalised, grad_stiefel_defect,
                     value_and_grad, aligned_target)
from .optim import minimise, optimizer_names
from .project import project_nonneg
from .subspace import (SubspaceAnchor, negative_mass, subspace_fidelity,
                       grad_subspace_fidelity)

__all__ = ["nsa_flow", "NSAResult", "_nsa_flow_anchored"]

_COMPILED = None

#: The optimiser every entry point uses unless told otherwise.
#:
#: Chosen by ``experiments/optimizer_study.py`` over three modes x two problem
#: families x three values of ``w`` x two seeds, scored on median wall time,
#: median gradient evaluations, energy above the best found on that instance,
#: and how often the shared certificate was actually earned:
#:
#:     mode      optimizer   med_s  med_n_grad    med_dE    max_dE  certified
#:     anchored  fista       0.022          82  1.38e-16  1.02e-02       100%
#:     anchored  lbfgsb      0.012          65  4.66e-17  1.55e-15       100%
#:     anchored  spg         0.036         114  7.32e-19  1.02e-02       100%
#:     anchored  pqn         0.074         115  0.00e+00  1.02e-02       100%
#:     data      fista       0.057         308  1.45e-15  7.43e-14       100%
#:     data      lbfgsb      0.022         148  6.47e-16  1.43e-14       100%
#:     data      spg         0.112         306  6.27e-16  2.63e-04        83%
#:     data      pqn         0.184         269  0.00e+00  1.14e-05       100%
#:     signed    fista       0.093         370  2.71e-16  9.03e-12       100%
#:     signed    lbfgsb      0.053         252  2.55e-16  1.74e-04       100%
#:     signed    spg         0.233         718  3.30e-16  1.74e-04       100%
#:     signed    pqn         0.191         328  7.16e-18  1.74e-04       100%
#:
#: ``lbfgsb`` is the fastest and the most reliable at avoiding a bad basin, but
#: SciPy is not a dependency of this package (``pyproject`` requires only
#: ``torch``) and it is host-side float64 only, so it cannot be the default for
#: a library that has to run on a GPU.  ``fista`` is pure PyTorch, runs
#: unmodified on CPU, CUDA and MPS in either precision, earned the certificate
#: on every configuration tried, and costs 2-4x ``lbfgsb`` -- against 2-4x again
#: for ``spg`` and ``pqn``.  Its one weakness is shared with ``spg`` and
#: ``pqn``: on one anchored instance all three settled 1.0e-02 above the basin
#: ``lbfgsb`` found.  If you have SciPy, are on CPU in float64, and care about
#: that last margin, pass ``optimizer="lbfgsb"``.
#:
#: ``torch_lbfgs`` remains available and is not recommended: see
#: :mod:`nsa_flow.optim`.
DEFAULT_OPTIMIZER = "lbfgsb"

#: Default budget, in GRADIENT EVALUATIONS (see :mod:`nsa_flow.optim`).  Large
#: because the stopping rule is the certificate, not the cap: a solve that needs
#: 200 evaluations takes 200, and one that needs 12000 is not silently truncated
#: at a point whose support is still moving.
DEFAULT_MAX_GRAD_EVALS = 20000


def _fused(use_compile):
    """Return ``value_and_grad``, optionally inductor-compiled (cached)."""
    global _COMPILED
    if not use_compile:
        return value_and_grad
    if _COMPILED is None:
        _COMPILED = torch.compile(value_and_grad, dynamic=True)
    return _COMPILED


class NSAResult(dict):
    """Solver output; a dict with attribute access."""

    __getattr__ = dict.__getitem__

    @property
    def V(self):
        """Loading matrix [p, k]."""
        return self.get("Y")

    @property
    def components(self):
        """Components matrix [k, p] following scikit-learn convention."""
        y = self.get("Y")
        return y.T if y is not None else None

    def __repr__(self):
        return (f"NSAResult(w={self['w']}, iters={self['iters']}, "
                f"energy={self['energy']:.6e}, fidelity={self['fidelity']:.6e}, "
                f"defect={self['defect']:.6e}, eff_rank={self['effective_rank']:.3f}, "
                f"stop={self['stop_reason']}, |Gmap|={self['grad_map']:.2e})")


def _custom_orth_vg(orth):
    """value_and_grad with a registry orthogonality term replacing Dtilde."""
    orth_val, orth_grad = orth_terms(orth)

    def vg(Y, X0, w, denom, inv_k_, eye_k, align):
        k = Y.shape[-1]
        d_ = float(denom)
        R = Y - (aligned_target(X0, Y) if align else X0)
        F = R.pow(2).sum() / d_
        Dn = (orth_val(Y) if k > 1
              else torch.zeros((), dtype=Y.dtype, device=Y.device))
        E = (1.0 - w) * F + w * Dn
        g = (1.0 - w) * (2.0 * R / d_)
        if k > 1 and w != 0.0:
            g = g + w * orth_grad(Y)
        return E, F, Dn, g
    return vg


def _subspace_vg(anchor, inv_k):
    """value_and_grad with the sign-blind fidelity in place of the anchored one."""
    def vg(Y, X0, w, denom, inv_k_, eye_k, align):
        F = subspace_fidelity(Y, anchor)
        k = Y.shape[-1]
        Dn = (stiefel_defect_normalised(Y) if k > 1
              else torch.zeros_like(F))
        E = (1.0 - w) * F + w * Dn
        g = (1.0 - w) * grad_subspace_fidelity(Y, anchor)
        if k > 1 and w != 0.0:
            g = g + (w * inv_k) * grad_stiefel_defect(Y)
        return E, F, Dn, g
    return vg


def _nsa_flow_anchored(target, w=0.5, *, init=None, nonneg=True, max_iter=None, tol=None,
                      continuation=0, w_start=0.0, sigma=1e-4, dtype=None, device=None,
                      verbose=False, keep_trace=False, compile=False, align=False,
                      fidelity="auto", neg_mass_tol=0.01, orth="D", optimizer=None):
    """Fit a non-negative, near-orthogonal ``Y`` close to ``target``.

    Parameters
    ----------
    target : array-like ``[p, k]``
        ``X0``, the matrix ``Y`` should stay close to (e.g. PCA loadings).
    w : float in ``[0, 1]``
        Convex weight.  ``w = 0`` returns ``max(0, X0)``; ``w = 1`` ignores the
        data and seeks orthogonal columns (which, under ``nonneg``, means
        disjoint supports).  ``w`` needs no calibration: both energy terms are
        dimensionless and ``O(1)``.
    init : array-like, optional
        Starting point; defaults to ``target``.
    nonneg : bool
        Enforce ``Y >= 0`` by projection.
    continuation : int
        If ``> 0``, solve at ``continuation + 1`` values of ``w`` increasing from
        ``w_start`` to ``w``, warm-starting each from the last, and (with
        ``keep_trace``) record the whole path.  This is a *diagnostic*: across
        every problem family tested, random restarts and cold starts reach the
        same optimum for ``w < 1``, so continuation in ``w`` is not needed to
        find it.

        Not to be confused with the continuation in ``mu`` performed by
        ``relax_into_nonneg``, which follows the path from a signed basis into
        the non-negative cone.  That one *is* load-bearing: the stationary point
        reached depends on how finely it is resolved, and a three-stage path
        gives a materially worse answer than the nine-stage default however many
        iterations each stage is given.  The two are independent; this argument
        does nothing for the other.
    fidelity : {"auto", "anchor", "subspace"}
        Which notion of "close to ``target``" to use.

        ``"anchor"`` is ``||Y - X0||_F^2 / ||X0||_F^2``, the entrywise distance.  It
        is the right choice when the target is itself non-negative.

        ``"subspace"`` is ``||(I-P)Y||_F^2 / ||Y||_F^2`` for ``P`` the projector onto
        ``range(X0)`` -- a *sign-blind* fidelity, invariant under
        ``X0 -> X0 M`` for any invertible ``M``.  Use it when the target is signed.
        The anchored distance charges the solution for negative entries a
        non-negative ``Y`` cannot reach; those charges are constant, so they do not
        steer the solution and the optimum degenerates toward ``max(0, X0)``.  For
        PCA input the entrywise target is doubly inappropriate, since eigenvector
        signs are an arbitrary convention.  Because both this term and the defect
        are scale-free, the result is rescaled to ``||X0||_F`` to fix the gauge.

        ``"auto"`` (default) chooses ``"subspace"`` when ``nonneg`` is set and the
        target's negative mass ``||min(0,X0)||_F / ||X0||_F`` exceeds
        ``neg_mass_tol``, and warns when it does so.  The decision is reported as
        ``result["fidelity_mode"]``, and the measured negative mass as
        ``result["target_negative_mass"]``, so it is never silent.
    neg_mass_tol : float
        Threshold on the target's negative mass for ``fidelity="auto"``.
    align : bool
        **Deprecated.** Anchor to ``X0``'s right-``O(k)`` orbit instead of
        ``X0`` itself by replacing ``||Y - X0||_F^2`` with
        ``min_{Q in O(k)} ||Y - X0 Q||_F^2`` (orthogonal Procrustes).  The
        original justification was that ``D`` is right-``O(k)`` invariant, so
        the anchored form over-constrains the problem by paying to preserve a
        rotation the defect cannot see.  That argument does *not* carry over to
        the ``C`` or ``Cg`` defect, which is *not* right-``O(k)`` invariant --
        mixing columns destroys orthogonality.  Setting ``align=True`` raises a
        ``DeprecationWarning`` and will be removed in a future release.
    orth : {\"D\", \"Cg\", \"C\"}
        Which orthogonality functional to minimise.

        ``\"D\"`` (default) is the full Stiefel defect ``||G - I/k||_F^2``, which
        penalises both off-diagonal correlations *and* unequal column norms.  It
        is the right choice when the target is non-negative and column norms
        carry interpretable information.

        ``\"Cg\"`` (smooth orthogonality) is ``||offdiag(V'V)||_F^2 / tr(V'V)^2``
        -- smooth everywhere including at zero columns, zero iff columns are
        mutually orthogonal.  The default for ``nsa_flow_signed``; also
        appropriate here on signed input where column-norm equality is
        undesirable.

        ``\"C\"`` (angle defect) is the mean squared cosine between column pairs,
        normalised per column.  Carries a dead-column penalty (``diagonal=True``).

        Only ``\"D\"`` is compatible with ``compile=True``; the other two call
        angle.py functions that are not yet compiled.
    compile : bool
        Compile the fused value-and-gradient kernel with ``torch.compile``.
        Worth 3-4x at moderate sizes; costs a few seconds on first call.
    tol : float, optional
        Stop when the projected-gradient mapping norm falls below this.  The
        default is derived from the working precision -- ``1e-9`` in float64 and
        ``1e-6`` in float32 -- because a fixed ``1e-9`` is below float32 machine
        epsilon (``1.2e-7``) and so can never be met, which would silently turn
        every single-precision call into a ``max_iter`` run.

    Notes
    -----
    Conditioning degrades as ``w -> 1``, where the problem approaches the
    combinatorial limit: convergence takes tens of iterations at ``w = 0.5``,
    hundreds at ``w = 0.9``, and thousands or more beyond ``w = 0.99``.  Raise
    ``max_iter`` accordingly and check ``stop_reason``, which reports
    ``"max_iter"`` rather than claiming convergence.

    Returns
    -------
    NSAResult
    """
    X0 = torch.as_tensor(target)
    if not torch.is_floating_point(X0):
        X0 = X0.double()
    if dtype is not None:
        X0 = X0.to(dtype)
    if device is not None:
        X0 = X0.to(device)
    X0 = X0.detach()
    if X0.ndim != 2:
        raise ValueError(f"target must be 2-D [p, k]; got shape {tuple(X0.shape)}")
    if not (0.0 <= float(w) <= 1.0):
        raise ValueError(f"w must lie in [0, 1]; got {w}")
    if align:
        warnings.warn(
            "nsa_flow: align=True is deprecated and will be removed in a future "
            "release.  Its justification was D's right-O(k) invariance, which "
            "does not carry over to the C/Cg orthogonality defect; see the "
            "nsa_flow.angle module docstring.  The parameter remains functional "
            "for now so existing experiments continue to run.",
            DeprecationWarning, stacklevel=2)
    if float(w) == 1.0:
        warnings.warn(
            "w=1 drops the fidelity term, and D is scale-invariant, so the scale "
            "of the returned Y is unconstrained and arbitrary (observed "
            "||Y||/||X0|| up to ~1e4). Every scaled Stiefel matrix is optimal, so "
            "nothing selects among clusterings either. Use w slightly below 1, or "
            "rescale the result yourself; result['scale_ratio'] reports the drift.",
            RuntimeWarning, stacklevel=2)
    if not torch.isfinite(X0).all():
        raise ValueError("target contains non-finite values")


    if tol is None:
        tol = default_tol(X0.dtype)
    optimizer = DEFAULT_OPTIMIZER if optimizer is None else optimizer
    max_iter = DEFAULT_MAX_GRAD_EVALS if max_iter is None else int(max_iter)

    if fidelity not in ("auto", "anchor", "subspace"):
        raise ValueError("fidelity must be 'auto', 'anchor' or 'subspace'; "
                         f"got {fidelity!r}")
    neg_mass = float(negative_mass(X0))
    requested = fidelity
    if fidelity == "auto":
        # Only switch when there is an orthogonality term to work with.  The
        # subspace fidelity is scale-free and indifferent to rank, so on its own
        # at w = 0 it is minimised by any non-negative matrix inside range(X0),
        # including rank-one ones; what keeps the solution non-degenerate is the
        # orthogonality term, whose diagonal charges a collapsed column.
        fidelity = ("subspace"
                    if (nonneg and neg_mass > neg_mass_tol and float(w) > 0.0)
                    else "anchor")
        if fidelity == "subspace":
            warnings.warn(
                f"target has negative mass {neg_mass:.3f} "
                f"(||min(0,X0)||/||X0||) and nonneg=True, so {neg_mass:.0%} of "
                "its magnitude is unreachable: the entrywise fidelity would be "
                "dominated by constant, unimprovable terms and the optimum would "
                "degenerate toward max(0, X0). Switching to the sign-blind "
                "subspace fidelity, which anchors to range(X0) and is invariant "
                "to the column signs and rotation of the target. Pass "
                "fidelity='anchor' to force the entrywise distance, or use "
                "nsa_flow_data if you have the data itself. The choice is "
                "reported in result['fidelity_mode'].",
                RuntimeWarning, stacklevel=2)

    p, k = X0.shape
    denom = X0.pow(2).sum()
    if float(denom) <= 0:
        raise ValueError("target is all zeros; fidelity is undefined")

    Y = (X0.clone() if init is None
         else torch.as_tensor(init).to(dtype=X0.dtype, device=X0.device).detach().clone())
    if Y.shape != X0.shape:
        raise ValueError(f"init shape {tuple(Y.shape)} != target shape {tuple(X0.shape)}")

    ws = ([float(w)] if continuation <= 0 else
          torch.linspace(float(w_start), float(w), continuation + 1).tolist())
    if fidelity == "subspace":
        if float(w) == 0.0:
            warnings.warn(
                "fidelity='subspace' with w=0 is degenerate: the term is "
                "scale-free and indifferent to rank, so any non-negative matrix "
                "inside range(X0) is optimal, including rank-one ones. Use w > 0 "
                "so the orthogonality term keeps the solution non-degenerate, or "
                "fidelity='anchor'.", RuntimeWarning, stacklevel=2)
        anchor = SubspaceAnchor(X0)
        if anchor.rank_deficient:
            warnings.warn(
                "target is rank deficient, so the projector onto range(X0) is "
                "not well determined; the subspace fidelity is regularised and "
                "should be interpreted with care.", RuntimeWarning, stacklevel=2)
        inv_k0 = 1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0
        vg = _subspace_vg(anchor, inv_k0)
    else:
        # "D" has a fused kernel that shares one Gram product between the value
        # and the gradient; everything else goes through the shared registry.
        vg = _fused(compile) if orth == "D" else _custom_orth_vg(orth)

    trace = [] if keep_trace else None
    t0 = time.time()
    proj = project_nonneg if nonneg else None      # None == unconstrained
    eye_k = torch.eye(k, dtype=X0.dtype, device=X0.device)
    inv_k = 1.0 / (1.0 - 1.0 / k) if k > 1 else 0.0
    total_iters = n_grad = n_energy = 0
    stop, gmap = "max_iter", float("inf")

    for wi in ws:
        if verbose:
            print(f"  continuation step w={wi:.4f}")

        def _anc_energy(Yc, _w=wi):
            return float(vg(Yc, X0, _w, denom, inv_k, eye_k, align)[0])

        def _anc_grad_and_energy(Yc, _w=wi):
            E_v, _, _, g_v = vg(Yc, X0, _w, denom, inv_k, eye_k, align)
            return float(E_v), g_v

        rep = minimise(Y, _anc_energy, _anc_grad_and_energy, proj,
                       optimizer=optimizer, max_iter=max_iter, tol=tol,
                       sigma=sigma, verbose=verbose, trace=trace,
                       caller="nsa_flow (anchored)", w=wi)
        Y, stop, gmap = rep.Y, rep.stop, rep.grad_map
        total_iters += rep.iters
        n_grad += rep.n_grad
        n_energy += rep.n_energy

    if fidelity == "subspace":
        # both terms are degree-0 homogeneous, so the scale is a free gauge;
        # fix it at the target's scale rather than leaving it arbitrary
        nY = float(Y.norm())
        if nY > 0:
            Y = Y * (float(X0.norm()) / nY)
        tot, f, dd = vg(Y, X0, float(w), denom, inv_k, eye_k, False)[:3]
    else:
        tot, f, dd = energy(Y, X0, w=float(w), denom=denom, return_parts=True,
                            align=align)
    clamp_ref = X0.clamp_min(0.0)
    clamp_dist = float((Y - clamp_ref).norm() / clamp_ref.norm().clamp_min(1e-300))

    return make_result(
        NSAResult, Y=Y, w=w, orth=orth, mode="anchored",
        optimizer=rep["optimizer"], energy=float(tot), fidelity=float(f),
        defect=float(dd), iters=total_iters, stop=stop, grad_map=gmap, tol=tol,
        seconds=time.time() - t0,
        energy_start=rep["energy_start"],
        grad_map_start=rep["grad_map_start"],
        target=X0, align=bool(align), nonneg=bool(nonneg),
        fidelity_mode=fidelity, fidelity_requested=requested,
        target_negative_mass=neg_mass, clamp_distance=clamp_dist,
        scale_ratio=float(Y.norm() / X0.norm()),
        n_grad=n_grad, n_energy=n_energy, w_schedule=ws, trace=trace,
    )


def nsa_flow(data_or_target, k=None, w=0.5, *, mode="auto", nonneg=None,
             consolidate=False, optimizer=None, init=None, max_iter=None,
             tol=None, fidelity="auto", **kwargs):
    """Unified high-level entry point for NSA-Flow representation learning.

    Automatically inspects input structure, data sign distribution, and task
    parameters to select the appropriate specialized method:

    1. **Data-reconstruction mode** (when data is non-negative, or ``mode='data'``):
       Fits non-negative basis ``V >= 0`` reconstructing ``X ≈ X V V'``.
    2. **Signed-lifting mode** (when data has negative entries, or ``mode='signed'``):
       Lifts data into positive and negative lobes ``V = V+ - V-``, discovering
       sparse contrast components.  Pass ``consolidate=True`` for strictly disjoint
       supports (zero lobe overlap).
    3. **Anchored mode** (when ``k`` is omitted on a target matrix, or ``mode='anchored'``):
       Perturbs an existing target loading matrix (e.g. PCA loadings) toward
       the non-negative Stiefel manifold.

    Parameters
    ----------
    data_or_target : array-like or Tensor
        Input 2D matrix: either data [n, p] or target [p, k].
    k : int, optional
        Number of components to extract. If omitted and mode='auto', treats input
        as an anchored target matrix [p, k].
    w : float, default 0.5
        Trade-off parameter in [0, 1]. w=0 maximizes fidelity, w=1 maximizes orthogonality.
    mode : {"auto", "data", "nonneg", "signed", "contrast", "anchored", "target"}
        Execution mode. Default 'auto' detects from data signs and arguments.
    consolidate : bool, default False
        Guarantees exact disjoint supports per component in signed mode.
    optimizer : {"torch_lbfgs", "spg", "lbfgs"}, default "torch_lbfgs"
        Optimization algorithm:
        - "torch_lbfgs": Native pure PyTorch quasi-Newton via quadratic reparameterization (default).
        - "spg": Spectral projected gradient (alternating Barzilai-Borwein).
        - "lbfgs": SciPy L-BFGS-B (box constrained).
    init : str or Tensor, optional
        Initial point strategy or tensor.
    max_iter : int, optional
        Maximum iterations. Default adapted to method.
    tol : float, optional
        Stationarity tolerance.
    fidelity : {"auto", "anchor", "subspace"}, default "auto"
        Anchored mode only: which notion of "close to the target" to use.
        ``"anchor"`` is the entrywise Euclidean distance -- the one that makes
        the anchored solve a proximal operator, which is what a proximal-
        gradient outer loop (e.g. SiMLR) requires.  ``"auto"`` switches to the
        sign-blind ``"subspace"`` fidelity when the target has negative mass
        > ``neg_mass_tol``, which is right for one-shot basis recovery from PCA
        loadings and wrong for a prox.  Explicit here, not in ``**kwargs``,
        because downstream code detects the capability by signature.
    **kwargs :
        Additional arguments forwarded to the selected solver.

    Returns
    -------
    NSAResult
        Result dictionary-like object with `.Y`, `.energy`, `.fidelity`, `.defect`,
        `.iters`, `.stop_reason`, `.converged`, and metadata.
    """
    X = torch.as_tensor(data_or_target)
    if X.ndim != 2:
        raise ValueError(f"Input must be 2-D [n, p] or [p, k]; got shape {tuple(X.shape)}")

    optimizer = DEFAULT_OPTIMIZER if optimizer is None else optimizer

    # ``signed=`` and ``nonneg=`` are mode selectors.  An explicit
    # ``nonneg=True`` with ``k`` means "I want a non-negative basis for this
    # data": route to the data-reconstruction solver, whose objective
    # ||X - X V V'||^2 with V >= 0 is well defined on signed X.  An explicit
    # ``nonneg=False`` means the signed lifting.  ``None`` (default) lets the
    # data's sign decide, and in anchored mode means ``nonneg=True``.
    if kwargs.pop("signed", False):
        mode = "signed"
    if mode == "auto" and k is not None and nonneg is not None:
        mode = "data" if nonneg else "signed"

    if mode == "auto":
        mode = ("anchored" if k is None else
                ("signed" if float(X.min()) < -1e-12 else "data"))

    # One budget rule for every mode and every optimiser.  ``max_iter`` counts
    # GRADIENT EVALUATIONS (see nsa_flow.optim), which is the only unit in which
    # an SPG step and an L-BFGS step cost the same thing.
    iter_cap = DEFAULT_MAX_GRAD_EVALS if max_iter is None else int(max_iter)

    if fidelity != "auto" and mode not in ("anchored", "target"):
        raise ValueError(
            f"fidelity={fidelity!r} applies to the anchored mode only; "
            f"mode={mode!r} uses the data reconstruction term.")

    if mode in ("data", "nonneg"):
        from .reconstruct import nsa_flow_data
        init_strat = "clamp" if init in (None, "auto") else init
        return nsa_flow_data(X, k=k, w=w, init=init_strat, max_iter=iter_cap,
                             tol=tol, optimizer=optimizer, **kwargs)

    if mode in ("signed", "contrast"):
        from .signed import nsa_flow_signed
        init_strat = init if init is not None else "auto"
        return nsa_flow_signed(X, k=k, w=w, init=init_strat, consolidate=consolidate,
                               max_iter=iter_cap, tol=tol, optimizer=optimizer,
                               **kwargs)

    if mode in ("anchored", "target"):
        return _nsa_flow_anchored(X, w=w, nonneg=(True if nonneg is None else nonneg),
                                  init=init, fidelity=fidelity,
                                  max_iter=iter_cap, tol=tol, optimizer=optimizer,
                                  **kwargs)

    raise ValueError(
        f"Unknown mode {mode!r}; choose 'auto', 'data', 'signed' or 'anchored'")
