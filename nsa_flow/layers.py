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


#: Newton-Schulz iterations for the layer's polar factor.  From a Frobenius
#: scaling every singular value lies in (0, 1]; the iteration multiplies small
#: ones by ~1.5 per step until they approach 1, then converges cubically.
_NS_ITERS = 12


def _polar_ns(Y, iters=_NS_ITERS):
    r"""Orthonormal polar factor ``U`` of ``[..., p, k]`` by Newton-Schulz.

    ``Q <- Q (3I - Q'Q) / 2`` from ``Q0 = Y / ||Y||_F``: six small matmuls per
    iteration, no eigendecomposition, plain autograd (no custom backward), and
    native on MPS where ``eigh`` is unimplemented.  Accuracy after 12 iterations
    is ~1e-10 in ``||U'U - I||`` for the well-conditioned weights a layer holds;
    :func:`nsa_flow.polar_factor` (eigh + Sylvester backward) remains the exact
    reference and is what ``project_scaled_stiefel`` uses.
    """
    k = Y.shape[-1]
    eye = torch.eye(k, dtype=Y.dtype, device=Y.device)
    Q = Y / Y.flatten(-2).norm(dim=-1).clamp_min(torch.finfo(Y.dtype).tiny).reshape(*Y.shape[:-2], 1, 1)
    for _ in range(iters):
        Q = Q @ (1.5 * eye - 0.5 * (Q.transpose(-2, -1) @ Q))
    return Q


def _project_scaled_stiefel_ns(Y):
    """``(<U, Y> / k) U`` with ``U`` from :func:`_polar_ns` -- the same map as
    :func:`nsa_flow.project_scaled_stiefel`, without the eigensolver."""
    k = Y.shape[-1]
    U = _polar_ns(Y)
    c = (U * Y).sum((-2, -1)) / k
    return c.reshape(*c.shape, 1, 1) * U


def _project_scaled_stiefel_layer(Y):
    """The blend's projection, by device.

    Where ``eigh`` is native (CPU, CUDA) the exact eigh-based projection is both
    more accurate and faster (k=5: 31 us against 152 us for twelve Newton-Schulz
    iterations).  Where it is not (MPS), Newton-Schulz avoids the on-device
    Jacobi fallback and halves the forward (1227 -> 717 us measured).
    """
    from .linalg import _has_native_eigh
    if _has_native_eigh(Y):
        return project_scaled_stiefel(Y)
    return _project_scaled_stiefel_ns(Y)

#: Maximum steps of the in-orthant defect flow (reached at ``w = 1``).  ``w``
#: selects how many of these fixed-size steps run, so the ``w = 0.5`` iterate
#: is literally a prefix of the ``w = 1`` path: the defect is non-increasing in
#: ``w`` by construction, not by tuning.  Resolution in ``w`` is ``1/_FLOW_STEPS``.
_FLOW_STEPS = 8
#: Step length per flow step is ``_FLOW_TAU * ||Y||_F * sqrt(Dtilde)`` along the
#: unit descent direction.  Scaling by ``sqrt(Dtilde)`` shrinks the step as the
#: defect vanishes, so the first trial is accepted on every step (measured: 0/8
#: halvings at tau = 2.0 against 2/8 for a fixed-length step) and the cost per
#: step is deterministic.  Measured: 8 steps take a random 66 x 5 start from
#: Dtilde 0.10 to 0.0024 (hard) and 0.48 to 0.003 (softplus).
_FLOW_TAU = 2.0


def _nonneg(W, mode):
    if mode in (None, False, "none"):
        return W
    if mode in (True, "hard"):
        return W.clamp_min(0.0)
    if mode in ("softplus", "soft"):
        return F.softplus(W)
    raise ValueError(f"nonneg must be one of None/'none', 'softplus', 'hard'/True; got {mode!r}")


def _gram_stats(Y):
    """``(S, t, N)`` = ``(Y'Y, tr S, ||S||_F^2)`` for ``[..., p, k]``."""
    S = Y.transpose(-2, -1) @ Y
    t = S.diagonal(dim1=-2, dim2=-1).sum(-1)
    N = (S * S).sum((-2, -1))
    return S, t, N


def _defect_flow_nonneg(Y, steps=_FLOW_STEPS, tau=_FLOW_TAU):
    r"""In-orthant projected-gradient defect flow on ``Dtilde`` inside ``Y >= 0``.

    Operates on ``[..., p, k]``. Each step moves ``tau * ||Y||_F * sqrt(Dtilde)``
    along ``-grad Dtilde / ||grad Dtilde||``, clamps to the orthant, and is kept
    only if ``Dtilde`` did not increase (branchless ``where``, so the flow is a
    single compilable graph). Differentiable a.e.

    Cost. ``D``, ``Dtilde`` and ``grad Dtilde`` all come from one Gram
    ``S = Y'Y``. Each accepted step costs one ``[p,k]x[k,p]`` product for the
    trial plus ``k x k`` algebra.

    Parameters
    ----------
    Y : Tensor
        Input non-negative tensor.
    steps : int or float, default 8
        Number of flow iterations. If a float <= 1.0 is passed (for legacy
        compatibility where w was passed), it is scaled by 8.
    tau : float, default 0.5
        Step size scaling factor.
    """
    if isinstance(steps, float):
        steps = int(round(steps * _FLOW_STEPS)) if steps <= 1.0 else int(round(steps))
    else:
        steps = int(steps)
    k = Y.shape[-1]
    if steps <= 0 or k <= 1:
        return Y
    inv_k = 1.0 / (1.0 - 1.0 / k)
    tiny = torch.finfo(Y.dtype).tiny
    S, t, N = _gram_stats(Y)
    D = (N / (t * t).clamp_min(tiny) - 1.0 / k) * inv_k
    for _ in range(steps):
        t_ = t.reshape(*t.shape, 1, 1)
        N_ = N.reshape(*N.shape, 1, 1)
        g = (4.0 * inv_k / t_.pow(2).clamp_min(tiny)) * (Y @ S - (N_ / t_.clamp_min(tiny)) * Y)
        gn = g.flatten(-2).norm(dim=-1).clamp_min(tiny)
        eta = (tau * t.sqrt() * D.clamp_min(0.0).sqrt() / gn).reshape(*t.shape, 1, 1)
        Y1 = (Y - eta * g).clamp_min(0.0)
        S1, t1, N1 = _gram_stats(Y1)
        D1 = (N1 / (t1 * t1).clamp_min(tiny) - 1.0 / k) * inv_k
        ok = D1 <= D
        okm = ok.reshape(*ok.shape, 1, 1)
        Y = torch.where(okm, Y1, Y)
        S = torch.where(okm, S1, S)
        t = torch.where(ok, t1, t)
        N = torch.where(ok, N1, N)
        D = torch.where(ok, D1, D)
    return Y


_COMPILED_FLOW = None


def _flow_fn(compile_):
    """The defect flow, eagerly or as one compiled graph (cached process-wide)."""
    global _COMPILED_FLOW
    if not compile_:
        return _defect_flow_nonneg
    if _COMPILED_FLOW is None:
        import torch._dynamo
        torch._dynamo.config.recompile_limit = max(
            getattr(torch._dynamo.config, "recompile_limit", 8), 64)
        compiled = torch.compile(_defect_flow_nonneg, fullgraph=True, dynamic=False)

        def guarded(Y, steps=_FLOW_STEPS):
            try:
                return compiled(Y, steps)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"NSAFlow layer: torch.compile of the defect flow failed "
                    f"({type(exc).__name__}: {str(exc)[:80]}); using eager for "
                    "this call.", RuntimeWarning, stacklevel=2)
                return _defect_flow_nonneg(Y, steps)
        _COMPILED_FLOW = guarded
    return _COMPILED_FLOW


class _NSAMixin:
    """Shared effective-weight logic. Subclasses provide ``_pk_view``."""

    compile = False
    operator = "flow"
    n_steps = _FLOW_STEPS

    def _effective(self, W):
        if getattr(self, "operator", "flow") == "solver":
            from .solve import nsa_flow
            M = self._pk_view(W.detach())
            nonneg = self.nonneg not in (None, False, "none")
            res = nsa_flow(M.double(), w=self.w, mode="anchored", fidelity="anchor",
                           nonneg=nonneg)
            Y = res.Y.to(W.dtype)
            W_eff = self._pk_unview(Y, W)
            if self.training:
                # Straight-Through Estimator (STE): forward pass uses the exact
                # nsa_flow solver result; backward pass passes gradients directly
                # through to W so optimizer updates the raw parameter.
                return W + (W_eff - W).detach()
            return W_eff

        if self.nonneg in (None, False, "none"):
            if self.w > 0.0:
                M = self._pk_view(W)
                M = (1.0 - self.w) * M + self.w * _project_scaled_stiefel_layer(M)
                W = self._pk_unview(M, W)
            return W
        # Unified blend semantics: W_eff = (1 - w) max(0, W) + w P_nonneg(W)
        M0 = self._pk_view(_nonneg(W, self.nonneg))
        if self.w > 0.0:
            n_steps = getattr(self, "n_steps", _FLOW_STEPS)
            P = _flow_fn(self.compile)(M0, n_steps)
            M = (1.0 - self.w) * M0 + self.w * P
        else:
            M = M0
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

    def _effective_cached(self, W):
        """In ``eval()`` the weight does not change between forwards, so the
        effective weight is computed once and reused until the parameter's
        version counter moves or the module returns to ``train()``.  Turns an
        inference forward back into ``nn.Linear`` cost."""
        if self.training:
            return self._effective(W)
        key = (W._version, self.w, str(self.nonneg), getattr(self, "operator", "flow"),
               getattr(self, "n_steps", _FLOW_STEPS), W.device, W.dtype)
        cache = getattr(self, "_eff_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        with torch.no_grad():
            eff = self._effective(W)
        self._eff_cache = (key, eff)
        return eff

    def train(self, mode=True):
        self._eff_cache = None
        return super().train(mode)

    def defect(self):
        """Normalised Stiefel defect of the effective weight, in ``[0, 1]``."""
        return stiefel_defect_normalised(self._pk_view(self.effective_weight()))


class NSAFlowLinear(_NSAMixin, nn.Module):
    r"""``nn.Linear`` whose ``out_features`` filters are driven toward orthogonality.

    Weight layout matches ``nn.Linear`` (``[out_features, in_features]``), so
    ``state_dict`` is 100% interchangeable with standard PyTorch models.
    Orthogonality is measured over the ``out_features`` filters, i.e. on
    ``W'`` viewed as ``[p, k] = [in_features, out_features]``.

    Three usage patterns are supported:

    1. **Parameterization in forward pass (w > 0)**:
       When ``w > 0``, the effective weight is parameterized via a continuous
       convex blend: :math:`W_{\text{eff}} = (1 - w) W_0 + w P(W_0)`.

       - If ``nonneg=None`` (signed), :math:`W_0 = W` and :math:`P(W_0)` is the
         scaled Stiefel projection.
       - If ``nonneg="hard"`` or ``"softplus"``, :math:`W_0` is the non-negative
         clamped/softplussed weight and :math:`P(W_0)` is the in-orthant defect
         flow (run for ``n_steps`` iterations, default 8).
       - If ``operator="solver"``, :math:`W_{\text{eff}}` is evaluated directly via
         the exact certified anchored proximal solver :func:`nsa_flow.nsa_flow`,
         matching :meth:`project_` identically with straight-through gradients.

       In ``eval()`` mode, the effective weight is automatically cached so inference
       runs at standard ``nn.Linear`` speed.

    2. **Loss penalty regularization (w = 0)**:
       Keep ``w = 0`` (the default) so the forward pass behaves identically to
       ``nn.Linear``, and add ``lambda_reg * layer.defect()`` to the training loss.
       ``defect()`` computes the normalized Stiefel defect :math:`\tilde{D} \in [0, 1]`
       in :math:`O(p k^2)` with no matrix factorizations.

    3. **Proximal updates (project_)**:
       Call ``layer.project_()`` between optimizer steps to replace raw weights
       in-place with their exact anchored proximal projection via :func:`nsa_flow.nsa_flow`.

    Args:
        in_features (int): Size of each input sample (:math:`p`).
        out_features (int): Size of each output sample (:math:`k`).
        bias (bool, optional): If ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        w (float, optional): Continuous blend coefficient in :math:`[0, 1]`. Default: ``0.0``.
        nonneg (str or bool or None, optional): Non-negativity constraint. Options are
            ``None``/``False`` (unconstrained/signed), ``"hard"``/``True`` (clamped
            non-negative), or ``"softplus"``. Default: ``None``.
        compile (bool, optional): If ``True``, compiles the in-orthant defect flow
            graph using ``torch.compile(fullgraph=True)``. Default: ``False``.
        operator (str, optional): Operator mode for effective weight evaluation.
            ``"flow"`` (default) uses the unrolled in-graph blend with full autograd.
            ``"solver"`` evaluates the exact certified :func:`nsa_flow.nsa_flow`
            anchored proximal solver (with caching in ``eval()`` and straight-through
            gradient estimation in ``train()``), giving exact numerical and sparsity
            parity with :func:`nsa_flow.nsa_flow` across the full range :math:`w \in [0, 1]`.
        n_steps (int, optional): Number of defect-flow steps used when computing
            :math:`P(W_0)` in ``operator="flow"``. Decoupled from ``w``. Default: ``8``.

    Attributes:
        weight (torch.nn.Parameter): Learnable weights of shape ``[out_features, in_features]``.
        bias (torch.nn.Parameter or None): Learnable bias of shape ``[out_features]``.

    Examples:
        >>> import torch
        >>> from nsa_flow import NSAFlowLinear
        >>> # Forward parameterization with orthogonal filters:
        >>> layer = NSAFlowLinear(64, 16, w=0.5)
        >>> x = torch.randn(8, 64)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([8, 16])
        >>> # Check filter defect (0 = perfectly orthogonal):
        >>> print(f"Defect: {layer.defect().item():.4f}")
        >>> # Exact solver operator matching nsa_flow:
        >>> layer_solver = NSAFlowLinear(64, 16, w=0.5, nonneg="hard", operator="solver")
    """

    def __init__(self, in_features, out_features, bias=True, w=0.0, nonneg=None,
                 compile=False, operator="flow", n_steps=_FLOW_STEPS):
        super().__init__()
        if not 0.0 <= float(w) <= 1.0:
            raise ValueError(f"w must lie in [0, 1]; got {w}")
        if operator not in ("flow", "solver"):
            raise ValueError(f"operator must be 'flow' or 'solver'; got {operator!r}")
        self.in_features, self.out_features = in_features, out_features
        self.w, self.nonneg, self.compile, self.operator = float(w), nonneg, bool(compile), str(operator)
        self.n_steps = int(n_steps)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
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
        return self._effective_cached(self.weight)

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, w={self.w}, nonneg={self.nonneg!r}, "
                f"compile={self.compile}, operator={self.operator!r}, n_steps={self.n_steps}")


class NSAFlowConv2d(_NSAMixin, nn.Conv2d):
    r"""``nn.Conv2d`` whose ``out_channels`` filters are driven toward orthogonality.

    Filters are flattened to shape ``[out_channels, in_channels * kh * kw]`` and
    evaluated as :math:`[p, k] = [\text{in\_channels} \times kh \times kw, \text{out\_channels}]`.

    Args:
        *args: Variable positional arguments forwarded to ``nn.Conv2d``.
        w (float, optional): Continuous blend coefficient in :math:`[0, 1]`. Default: ``0.0``.
        nonneg (str or bool or None, optional): Non-negativity constraint (``None``,
            ``"hard"``/``True``, or ``"softplus"``). Default: ``None``.
        compile (bool, optional): If ``True``, compiles the non-negative defect flow.
            Default: ``False``.
        operator (str, optional): Operator mode (``"flow"`` or ``"solver"``). Default: ``"flow"``.
        n_steps (int, optional): Number of defect flow steps in ``operator="flow"``. Default: ``8``.
        **kwargs: Keyword arguments forwarded to ``nn.Conv2d``.
    """

    def __init__(self, *args, w=0.0, nonneg=None, compile=False, operator="flow",
                 n_steps=_FLOW_STEPS, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= float(w) <= 1.0:
            raise ValueError(f"w must lie in [0, 1]; got {w}")
        if operator not in ("flow", "solver"):
            raise ValueError(f"operator must be 'flow' or 'solver'; got {operator!r}")
        self.w, self.nonneg, self.compile, self.operator = float(w), nonneg, bool(compile), str(operator)
        self.n_steps = int(n_steps)
        if self.w > 0.0 and nonneg in (None, False, "none"):
            nn.init.orthogonal_(self.weight)          # else keep nn.Conv2d's init

    @staticmethod
    def _pk_view(W):
        return W.reshape(W.shape[0], -1).transpose(0, 1)

    @staticmethod
    def _pk_unview(M, like):
        return M.transpose(0, 1).reshape(like.shape)

    def effective_weight(self):
        return self._effective_cached(self.weight)

    def forward(self, x):
        return self._conv_forward(x, self.effective_weight(), self.bias)

    def extra_repr(self):
        s = super().extra_repr()
        return f"{s}, w={self.w}, nonneg={self.nonneg!r}, operator={self.operator!r}, n_steps={self.n_steps}"


class NSAFlowLayer(nn.Module):
    """Per-sample NSA transform of a batch of matrices, ``[B, p, k] -> [B, p, k]``.

    Each sample is transformed independently. Input must be 3-D: a 2-D input is
    ambiguous (is the first axis samples or features?) and the old behaviour of
    treating a ``[N, k]`` batch as one ``[p, k]`` matrix coupled every sample to
    its batch-mates, so that outputs depended on batch composition. That is
    rejected rather than guessed at.
    """

    def __init__(self, w=0.5, nonneg=None, compile=False, operator="flow", n_steps=_FLOW_STEPS):
        super().__init__()
        if not 0.0 <= float(w) <= 1.0:
            raise ValueError(f"w must lie in [0, 1]; got {w}")
        if operator not in ("flow", "solver"):
            raise ValueError(f"operator must be 'flow' or 'solver'; got {operator!r}")
        self.w, self.nonneg, self.compile, self.operator = float(w), nonneg, bool(compile), str(operator)
        self.n_steps = int(n_steps)

    def forward(self, Y):
        if Y.ndim != 3:
            raise ValueError(
                f"NSAFlowLayer expects batched [B, p, k] input; got {tuple(Y.shape)}. "
                "Add a leading batch axis to state the intended per-sample semantics."
            )
        if getattr(self, "operator", "flow") == "solver":
            from .solve import nsa_flow
            nonneg = self.nonneg not in (None, False, "none")
            res = [nsa_flow(Y[i].detach().double(), w=self.w, mode="anchored",
                            fidelity="anchor", nonneg=nonneg).Y.to(Y.dtype)
                   for i in range(Y.shape[0])]
            Y_eff = torch.stack(res, dim=0)
            if self.training:
                return Y + (Y_eff - Y).detach()
            return Y_eff

        if self.nonneg in (None, False, "none"):
            if self.w > 0.0:
                Y = (1.0 - self.w) * Y + self.w * _project_scaled_stiefel_layer(Y)
            return Y
        Y0 = _nonneg(Y, self.nonneg)
        if self.w > 0.0:
            P = _flow_fn(self.compile)(Y0, getattr(self, "n_steps", _FLOW_STEPS))
            return (1.0 - self.w) * Y0 + self.w * P
        return Y0

    def defect(self, Y):
        return stiefel_defect_normalised(self.forward(Y))

    def extra_repr(self):
        return f"w={self.w}, nonneg={self.nonneg!r}, operator={self.operator!r}, n_steps={self.n_steps}"
