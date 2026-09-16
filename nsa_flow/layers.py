"""Torch layers built on the NSA-Flow energy.

Two routes are offered, both theoretically clean:

*penalty* (preferred) -- keep a standard layer and add ``w * layer.defect()`` to
the task loss.  ``defect()`` is ``O(p k^2)``, needs no factorisation, and its
gradient is exact, so this is a plain regulariser with no reparameterisation.

*parameterisation* -- set ``w > 0`` and the effective weight becomes
``(1 - w) W + w P(W)`` where ``P`` is the Euclidean projection onto the scaled
Stiefel manifold.  Because ``P(W)`` carries a scale matched to ``W`` (it is
``(sum sigma_i / k) U V'``), ``w`` is a true blend fraction here, unlike a blend
against a unit-norm polar factor whose effective weight drifts with ``||W||``.

Non-negativity, when requested, uses ``softplus`` rather than a hard clamp: the
clamp is not surjective onto the positive orthant, so its Jacobian drops rank
and stationary points of the reparameterised problem need not be stationary for
the constrained one.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .energy import stiefel_defect_normalised
from .project import project_scaled_stiefel

__all__ = ["NSAFlowLinear", "NSAFlowConv2d", "NSAFlowLayer"]


def _nonneg(W, mode):
    if mode in (None, False, "none"):
        return W
    if mode in (True, "softplus", "soft"):
        return F.softplus(W)
    if mode == "hard":
        return W.clamp_min(0.0)
    raise ValueError(f"nonneg must be one of None/'none', 'softplus', 'hard'; got {mode!r}")


class _NSAMixin:
    """Shared effective-weight logic.  Subclasses provide ``_pk_view``."""

    def _effective(self, W):
        if self.w > 0.0:
            M = self._pk_view(W)
            M = (1.0 - self.w) * M + self.w * project_scaled_stiefel(M)
            W = self._pk_unview(M, W)
        return _nonneg(W, self.nonneg)

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
        nn.init.orthogonal_(self.weight)
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
        nn.init.orthogonal_(self.weight)

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
        if self.w > 0.0:
            Y = (1.0 - self.w) * Y + self.w * project_scaled_stiefel(Y)
        return _nonneg(Y, self.nonneg)

    def defect(self, Y):
        return stiefel_defect_normalised(self.forward(Y))

    def extra_repr(self):
        return f"w={self.w}, nonneg={self.nonneg!r}"
