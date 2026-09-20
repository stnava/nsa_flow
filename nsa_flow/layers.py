"""Torch layers built on the NSA-Flow energy.

Two routes are offered, both theoretically clean:

*penalty* (preferred) -- keep a standard layer and add ``w * layer.defect()`` to
the task loss.  ``defect()`` is ``O(p k^2)``, needs no factorisation, and its
gradient is exact, so this is a plain regulariser with no reparameterisation.

*parameterisation* -- set ``w > 0`` and the effective weight is driven toward
the feasible set inside the forward pass.  What that map is depends on whether
non-negativity is requested, because the two constraints do not compose in an
arbitrary order:

``nonneg=None``
    ``(1 - w) W + w P(W)`` with ``P`` the Euclidean projection onto the scaled
    Stiefel manifold.  ``P(W)`` carries a scale matched to ``W`` (it is
    ``(sum sigma_i / k) U V'``), so ``w`` is a true blend fraction, and ``w = 1``
    gives ``D = 0`` exactly.

``nonneg="hard"`` (also ``True``) or ``"softplus"``
    Non-negativity FIRST -- ``relu(W)`` or ``softplus(W)`` -- and then ``w``
    controls a short *projected-gradient flow on the defect inside the
    non-negative orthant*: a few steps of ``Y <- max(0, Y - eta grad Dtilde(Y))``
    where ``w`` sets how many fixed-size steps run (``round(8 w)``), each kept
    only if it lowers ``Dtilde``.  Every step is differentiable (almost everywhere), keeps
    ``Y >= 0`` exactly, and cannot increase the defect, so ``defect()`` is
    non-increasing in ``w`` at fixed ``W``.  ``w = 0`` returns ``relu(W)`` /
    ``softplus(W)`` unchanged.

    Why not the blend here.  Versions before 3.2.1 blended onto the Stiefel
    manifold and *then* applied the non-negativity map.  A clamp or softplus
    after an orthogonalisation destroys the orthogonality just paid for:
    measured on a random 66 x 5 start, ``defect_D`` of the effective weight
    was 0.528 at ``w = 0`` and 0.497 at ``w = 1`` under softplus, 0.134 and
    0.109 under the clamp -- ``w`` did essentially nothing.  Composing in that
    order is the one order guaranteed not to give the feasible set.

The exact operator for the non-negative case is the anchored prox
``nsa_flow(W, w, mode="anchored", fidelity="anchor", nonneg=True)``;
:meth:`project_` applies it in place (no autograd) for proximal-gradient
training loops that want the true projection between optimiser steps.

``nonneg=True`` means the hard clamp.  It used to mean softplus, which cannot
produce a zero (sparsity is identically 0) and maps the default initialisation
to ``softplus(~0) ~ 0.69`` everywhere -- a dense, near-uniform basis.  Ask for
``"softplus"`` explicitly if that is what you want.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .energy import grad_stiefel_defect, stiefel_defect_normalised
from .project import project_scaled_stiefel

#: Maximum steps of the in-orthant defect flow (reached at ``w = 1``).  ``w``
#: selects how many of these fixed-size steps run, so the ``w = 0.5`` iterate
#: is literally a prefix of the ``w = 1`` path: the defect is non-increasing in
#: ``w`` by construction, not by tuning.  Resolution in ``w`` is ``1/_FLOW_STEPS``.
_FLOW_STEPS = 8
#: Fraction of ``||Y||_F`` moved per flow step along ``-grad Dtilde``, before the
#: accept-on-decrease halvings.  Measured: 8 steps at 0.2 take a random 66 x 5
#: start from Dtilde 0.10 to 0.002 (hard) and 0.48 to 0.003 (softplus).
_FLOW_TAU = 0.2


def _nonneg(W, mode):
    if mode in (None, False, "none"):
        return W
    if mode in (True, "hard"):
        return W.clamp_min(0.0)
    if mode in ("softplus", "soft"):
        return F.softplus(W)
    raise ValueError(f"nonneg must be one of None/'none', 'softplus', 'hard'/True; got {mode!r}")


def _defect_flow_nonneg(Y, w, steps=_FLOW_STEPS, tau=_FLOW_TAU):
    r"""``round(w * steps)`` projected-gradient steps on ``Dtilde`` inside ``Y >= 0``.

    Operates on ``[..., p, k]``.  Each step moves a fraction ``tau`` of
    ``||Y||_F`` along ``-grad Dtilde`` (scale-free: ``grad D`` scales as
    ``1/||Y||``), clamps to the orthant, and is kept only if ``Dtilde`` did not
    increase (the full step, then two halvings, are tried).  The step size does
    not depend on ``w``; ``w`` sets the number of steps, so the iterate at a
    smaller ``w`` is a prefix of the path at a larger one and the defect is
    non-increasing in ``w`` exactly.  Differentiable a.e. through the accepted
    branch.
    """
    k = Y.shape[-1]
    n_steps = int(round(float(w) * steps))
    if n_steps <= 0 or k <= 1:
        return Y
    inv_k = 1.0 / (1.0 - 1.0 / k)
    tiny = torch.finfo(Y.dtype).tiny
    D = stiefel_defect_normalised(Y)
    for _ in range(n_steps):
        g = grad_stiefel_defect(Y) * inv_k
        gn = g.flatten(-2).norm(dim=-1).clamp_min(tiny)
        yn = Y.flatten(-2).norm(dim=-1)
        eta = (tau * yn / gn).reshape(*yn.shape, 1, 1)
        accepted = None
        for _h in range(3):                        # full step, then two halvings
            Y_try = (Y - eta * g).clamp_min(0.0)
            D_try = stiefel_defect_normalised(Y_try)
            ok = D_try <= D
            if accepted is None:
                accepted, Y_acc, D_acc = ok, Y_try, D_try
            else:
                take = ok & ~accepted
                Y_acc = torch.where(take.reshape(*take.shape, 1, 1), Y_try, Y_acc)
                D_acc = torch.where(take, D_try, D_acc)
                accepted = accepted | ok
            if bool(accepted.all()):
                break
            eta = eta * 0.5
        Y = torch.where(accepted.reshape(*accepted.shape, 1, 1), Y_acc, Y)
        D = torch.where(accepted, D_acc, D)
    return Y


class _NSAMixin:
    """Shared effective-weight logic.  Subclasses provide ``_pk_view``."""

    def _effective(self, W):
        if self.nonneg in (None, False, "none"):
            if self.w > 0.0:
                M = self._pk_view(W)
                M = (1.0 - self.w) * M + self.w * project_scaled_stiefel(M)
                W = self._pk_unview(M, W)
            return W
        # non-negativity first, then the in-orthant defect flow (see module doc)
        M = self._pk_view(_nonneg(W, self.nonneg))
        M = _defect_flow_nonneg(M, self.w)
        return self._pk_unview(M, W)

    @torch.no_grad()
    def project_(self, w=None, **kwargs):
        """Replace the raw weight by its exact anchored prox, in place.

        ``argmin_{Y >= 0 (if nonneg)} (1 - w)||Y - W||^2/||W||^2 + w Dtilde(Y)``
        via :func:`nsa_flow.nsa_flow` with ``fidelity="anchor"``.  For
        proximal-gradient training: call between optimiser steps.  Returns the
        solver's result for its certificate and diagnostics.
        """
        from .solve import nsa_flow
        w = self.w if w is None else float(w)
        M = self._pk_view(self.weight.detach())
        nonneg = self.nonneg not in (None, False, "none")
        res = nsa_flow(M.double(), w=w, mode="anchored", fidelity="anchor",
                       nonneg=nonneg, **kwargs)
        self.weight.copy_(self._pk_unview(res.Y.to(self.weight.dtype), self.weight))
        return res

    def defect(self):
        """Normalised Stiefel defect of the effective weight, in ``[0, 1]``."""
        return stiefel_defect_normalised(self._pk_view(self.effective_weight()))


class NSAFlowLinear(_NSAMixin, nn.Module):
    """``nn.Linear`` whose ``out_features`` filters are driven toward orthogonality.

    Weight layout matches ``nn.Linear`` (``[out_features, in_features]``), so
    ``state_dict`` is interchangeable.  Orthogonality is measured over the
    ``out_features`` filters, i.e. on ``W'`` viewed as ``[p, k] = [in, out]``.
    """

    def __init__(self, in_features, out_features, bias=True, w=0.0, nonneg=None):
        super().__init__()
        if not 0.0 <= float(w) <= 1.0:
            raise ValueError(f"w must lie in [0, 1]; got {w}")
        self.in_features, self.out_features = in_features, out_features
        self.w, self.nonneg = float(w), nonneg
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # Orthogonal init is the right start only for the Stiefel BLEND
        # (nonneg=None, w>0).  For w=0 this is a drop-in nn.Linear and must
        # initialise like one; for the non-negative modes an orthogonal start
        # is ~half negative and clamps to a random half-support, and softplus of
        # its ~0.1-magnitude entries is ~0.69 everywhere -- dense and uniform.
        if self.w > 0.0 and self.nonneg in (None, False, "none"):
            nn.init.orthogonal_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _pk_view(W):
        return W.transpose(-2, -1)          # [in, out] = [p, k]

    @staticmethod
    def _pk_unview(M, _like):
        return M.transpose(-2, -1)

    def effective_weight(self):
        return self._effective(self.weight)

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, w={self.w}, nonneg={self.nonneg!r}")


class NSAFlowConv2d(_NSAMixin, nn.Conv2d):
    """``nn.Conv2d`` whose ``out_channels`` filters are driven toward orthogonality.

    Filters are flattened to ``[out_channels, in_channels * kh * kw]`` and
    measured as ``[p, k] = [in * kh * kw, out]``.
    """

    def __init__(self, *args, w=0.0, nonneg=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= float(w) <= 1.0:
            raise ValueError(f"w must lie in [0, 1]; got {w}")
        self.w, self.nonneg = float(w), nonneg
        if self.w > 0.0 and nonneg in (None, False, "none"):
            nn.init.orthogonal_(self.weight)          # else keep nn.Conv2d's init

    @staticmethod
    def _pk_view(W):
        return W.reshape(W.shape[0], -1).transpose(0, 1)

    @staticmethod
    def _pk_unview(M, like):
        return M.transpose(0, 1).reshape(like.shape)

    def effective_weight(self):
        return self._effective(self.weight)

    def forward(self, x):
        return self._conv_forward(x, self.effective_weight(), self.bias)


class NSAFlowLayer(nn.Module):
    """Per-sample NSA transform of a batch of matrices, ``[B, p, k] -> [B, p, k]``.

    Each sample is transformed independently.  Input must be 3-D: a 2-D input is
    ambiguous (is the first axis samples or features?) and the old behaviour of
    treating a ``[N, k]`` batch as one ``[p, k]`` matrix coupled every sample to
    its batch-mates, so that outputs depended on batch composition.  That is
    rejected rather than guessed at.
    """

    def __init__(self, w=0.5, nonneg=None):
        super().__init__()
        if not 0.0 <= float(w) <= 1.0:
            raise ValueError(f"w must lie in [0, 1]; got {w}")
        self.w, self.nonneg = float(w), nonneg

    def forward(self, Y):
        if Y.ndim != 3:
            raise ValueError(
                f"NSAFlowLayer expects batched [B, p, k] input; got {tuple(Y.shape)}. "
                "Add a leading batch axis to state the intended per-sample semantics."
            )
        if self.nonneg in (None, False, "none"):
            if self.w > 0.0:
                Y = (1.0 - self.w) * Y + self.w * project_scaled_stiefel(Y)
            return Y
        return _defect_flow_nonneg(_nonneg(Y, self.nonneg), self.w)

    def defect(self, Y):
        return stiefel_defect_normalised(self.forward(Y))

    def extra_repr(self):
        return f"w={self.w}, nonneg={self.nonneg!r}"
