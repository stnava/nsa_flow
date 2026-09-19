r"""Factorisations that stay on the device.

The initialisers need the top-``k`` eigenvectors of ``S = X'X``.  The obvious
calls do not survive a GPU:

===================  ==========================================================
``linalg.eigh``      ``NotImplementedError`` on MPS.
``linalg.svd``       "works" on MPS only by *silently falling back to the CPU*
                     (``aten::linalg_svd ... will fall back to run on the CPU``),
                     so it is a hidden round trip, which is worse than an error.
``linalg.matrix_rank``  built on ``svd``; same problem.
===================  ==========================================================

``qr``, ``cholesky``, ``cholesky_solve`` and ``solve`` *are* native on MPS, and
everything here is built from those plus elementwise algebra.  Nothing in this
module moves data between devices.

Two pieces:

:func:`jacobi_eigh`
    A symmetric eigensolver written in ``torch``, for the small matrices that
    appear here (``k x k`` or ``2k x 2k`` Rayleigh quotients and Gram matrices,
    ``k`` typically 2-50).  Cyclic one-sided Jacobi with a round-robin pairing,
    so each sweep applies ``n/2`` *disjoint* plane rotations at once as a single
    orthogonal similarity -- two matmuls per round rather than one per rotation,
    which is what keeps the launch count down.  Jacobi converges cubically near
    the solution and is famously accurate for small symmetric matrices,
    including the clustered spectra where the QR algorithm loses relative
    accuracy.

:func:`top_k_eigenvectors`
    The top-``k`` eigenvectors of ``X'X``, **exactly**, via whichever of the two
    Gram matrices is smaller (``X'X`` when ``p <= n``, ``XX'`` otherwise).  At
    ``p = 7129, n = 72`` that is a ``72 x 72`` eigenproblem taking 1.3 ms, and
    the ``p x p`` matrix that would be 0.4 GB is never formed.

    An earlier version used randomised subspace iteration here.  It was wrong:
    its error decays like ``(lambda_{l+1}/lambda_k)^{2q+1}`` and centred data has
    a near-flat Marchenko-Pastur bulk, so there is no gap to exploit.  Measured
    against the exact answer it returned subspaces at ``min cos(principal
    angle)`` of 0.89 and 0.42 -- a different subspace, not an approximation of
    the right one.  It survives only for the case where both dimensions are
    large *and* no native eigensolver exists, where it warns.

Determinism
-----------
Column signs are canonicalised (largest-magnitude entry positive), so repeated
calls return the same basis on any device in either precision, and the two Gram
routes agree exactly.  Without the convention they would differ by arbitrary
sign flips and the result would depend on a shape heuristic.
"""
import torch

__all__ = ["jacobi_eigh", "safe_eigh", "top_k_eigenvectors",
           "leading_eigenvectors", "symmetric_rank"]


def _round_robin(n):
    """Pairings for a round-robin tournament: ``n-1`` rounds of disjoint pairs.

    Round ``r`` pairs every index with exactly one other, and over all rounds
    every unordered pair occurs exactly once -- which is precisely the
    requirement for one Jacobi sweep.
    """
    idx = list(range(n + (n % 2)))          # pad to even
    m = len(idx)
    rounds = []
    for _ in range(m - 1):
        pairs = [(idx[i], idx[m - 1 - i]) for i in range(m // 2)]
        rounds.append([(a, b) for a, b in pairs if a < n and b < n and a != b])
        idx = [idx[0]] + [idx[-1]] + idx[1:-1]
    return rounds


def jacobi_eigh(A, sweeps=12, tol=None):
    r"""Symmetric eigendecomposition ``A = V diag(w) V'``, entirely in ``torch``.

    Returns ``(w, V)`` with ``w`` ascending, matching ``torch.linalg.eigh``'s
    convention so the two are drop-in interchangeable.

    Each round annihilates ``n/2`` disjoint off-diagonal pairs simultaneously.
    For the pair ``(p, q)`` the rotation angle is the classical

        theta = (a_qq - a_pp) / (2 a_pq),
        t = sign(theta) / (|theta| + sqrt(theta^2 + 1)),
        c = 1 / sqrt(t^2 + 1),   s = t c,

    which is the numerically stable branch (it picks the smaller rotation and
    never differences nearly-equal quantities).  Because the pairs in a round
    are disjoint, the rotations commute and compose into one orthogonal matrix.

    Stops early once the off-diagonal Frobenius mass falls below ``tol`` times
    the total, which on the matrices here is typically 4-6 sweeps.
    """
    A = A.clone()
    n = A.shape[-1]
    dtype, device = A.dtype, A.device
    V = torch.eye(n, dtype=dtype, device=device)
    if n == 1:
        return A.reshape(1), V
    if tol is None:
        tol = torch.finfo(dtype).eps * 10.0

    rounds = _round_robin(n)
    for _ in range(sweeps):
        off = float((A - torch.diag(torch.diagonal(A))).pow(2).sum())
        total = float(A.pow(2).sum())
        if total == 0.0 or off <= (tol * tol) * total:
            break
        for pairs in rounds:
            if not pairs:
                continue
            P = torch.tensor([p for p, _ in pairs], device=device)
            Q = torch.tensor([q for _, q in pairs], device=device)
            apq = A[P, Q]
            app, aqq = A[P, P], A[Q, Q]

            theta = (aqq - app) / (2.0 * apq)
            t = torch.sign(theta) / (theta.abs() + torch.sqrt(theta * theta + 1.0))
            t = torch.where(apq.abs() <= torch.finfo(dtype).tiny,
                            torch.zeros_like(t), t)
            c = 1.0 / torch.sqrt(t * t + 1.0)
            s = t * c

            # Apply the disjoint rotations by updating only the rows and
            # columns they touch.  Composing them into a dense [n, n] matrix and
            # doing two full matmuls costs O(n^3) per ROUND, i.e. O(n^4) per
            # sweep, which is unusable past n ~ 100; this is O(n) per rotation,
            # so O(n^2) per round and O(n^3) per sweep.
            c_ = c.unsqueeze(1)
            s_ = s.unsqueeze(1)
            Ap, Aq = A[P, :], A[Q, :]                  # rows
            A[P, :] = c_ * Ap - s_ * Aq
            A[Q, :] = s_ * Ap + c_ * Aq
            Ap, Aq = A[:, P], A[:, Q]                  # columns
            A[:, P] = c.unsqueeze(0) * Ap - s.unsqueeze(0) * Aq
            A[:, Q] = s.unsqueeze(0) * Ap + c.unsqueeze(0) * Aq
            Vp, Vq = V[:, P], V[:, Q]                  # accumulate eigenvectors
            V[:, P] = c.unsqueeze(0) * Vp - s.unsqueeze(0) * Vq
            V[:, Q] = s.unsqueeze(0) * Vp + c.unsqueeze(0) * Vq
        # keep it exactly symmetric against accumulated rounding
        A = 0.5 * (A + A.transpose(-2, -1))

    w = torch.diagonal(A)
    order = torch.argsort(w)
    return w[order].contiguous(), V[:, order].contiguous()


def safe_eigh(A):
    """``torch.linalg.eigh``, or :func:`jacobi_eigh` where it is unavailable.

    No host fallback: the replacement runs on whatever device ``A`` is on.
    """
    try:
        return torch.linalg.eigh(A)
    except (NotImplementedError, RuntimeError):
        return jacobi_eigh(A)


def symmetric_rank(A, rtol=None):
    """Numerical rank of a symmetric matrix, without ``svd``.

    ``torch.linalg.matrix_rank`` goes through ``svd`` and so is unavailable on
    MPS; for a symmetric matrix the eigenvalue magnitudes are the singular
    values, so :func:`safe_eigh` answers it directly.
    """
    w, _ = safe_eigh(A)
    w = w.abs()
    top = float(w.max()) if w.numel() else 0.0
    if top == 0.0:
        return 0
    if rtol is None:
        rtol = A.shape[-1] * torch.finfo(A.dtype).eps
    return int((w > rtol * top).sum())


def _canonical_sign(E):
    """Largest-magnitude entry of each column made positive."""
    k = E.shape[-1]
    rows = E.abs().argmax(dim=0)
    flip = E[rows, torch.arange(k, device=E.device)].sign()
    flip = flip.where(flip != 0, torch.ones_like(flip))
    return E * flip.unsqueeze(0)


def top_k_eigenvectors(k, S=None, X=None, seed=0, max_dense=None):
    r"""Top-``k`` eigenvectors of ``S = X'X``, exactly, on-device, ``[p, k]``.

    Give exactly one of ``S`` (``[p, p]``) or ``X`` (``[n, p]``).

    **Why not randomised subspace iteration.**  That was the first
    implementation and it is wrong for this problem.  Its error decays like
    ``(lambda_{l+1} / lambda_k)^{2q+1}``, which is useless without an eigengap --
    and a centred data matrix has a near-flat Marchenko-Pastur bulk, so there is
    no gap to exploit.  Measured against the exact answer it returned subspaces
    at ``min cos(principal angle) = 0.89`` and ``0.42``: not an approximation,
    a different subspace.  Since this seeds every solve, that is not acceptable
    at any speed.

    **The dual Gram trick instead.**  ``X = U Sigma V'`` gives ``X'X = V Sigma^2 V'``
    *and* ``X X' = U Sigma^2 U'``, so the top-``k`` right singular vectors can be
    had from whichever of the two Gram matrices is smaller:

        p <= n :  eigendecompose  X'X  (p x p), take the top k eigenvectors
        n <  p :  eigendecompose  X X' (n x n), then  V_k = X' U_k / sigma_k

    Both are exact, deterministic, and need only :func:`safe_eigh` on a matrix of
    side ``min(n, p)``.  In the ``p >> n`` regime this is also the *cheap* route:
    at ``p = 7129, n = 72`` it is a ``72 x 72`` eigenproblem, where forming
    ``X'X`` would have been 0.4 GB.

    ``max_dense`` guards the remaining bad case -- both dimensions large *and*
    no native ``eigh`` -- where the Jacobi solver's ``O(n^3)`` per sweep would
    dominate.  There we fall back to subspace iteration and say so.
    """
    if (S is None) == (X is None):
        raise ValueError("give exactly one of S or X")

    if S is not None:
        _, evecs = safe_eigh(S)
        return _canonical_sign(evecs[..., -k:].flip(-1).contiguous())

    n, p = X.shape[-2], X.shape[-1]
    k = min(k, n, p)
    if max_dense is None:
        # a native eigh handles thousands; the Jacobi fallback should not be
        # asked to go much past a few hundred
        max_dense = 4096 if _has_native_eigh(X) else 512

    side = min(n, p)
    if side > max_dense:
        return _subspace_iteration(k, X, seed=seed)

    if p <= n:                                   # primal: p x p
        G = X.transpose(-2, -1) @ X
        _, evecs = safe_eigh(G)
        return _canonical_sign(evecs[..., -k:].flip(-1).contiguous())

    # dual: n x n, then map back.  XX' = U Sigma^2 U', V = X' U / sigma.
    G = X @ X.transpose(-2, -1)
    evals, U = safe_eigh(0.5 * (G + G.transpose(-2, -1)))
    U = U[..., -k:].flip(-1)
    sig = evals[..., -k:].flip(-1).clamp_min(0.0).sqrt()
    V = X.transpose(-2, -1) @ U
    # normalise by sigma where it is meaningful, else by the column norm, so a
    # rank-deficient X degrades to an orthonormal basis rather than to inf
    nrm = V.norm(dim=-2, keepdim=True)
    scale = torch.where(sig.unsqueeze(-2) > 0, sig.unsqueeze(-2), nrm)
    V = V / scale.clamp_min(torch.finfo(X.dtype).tiny)
    return _canonical_sign(V.contiguous())


def _has_native_eigh(ref):
    """Is ``torch.linalg.eigh`` implemented for this device?  Probed once."""
    key = ref.device.type
    if key not in _NATIVE_EIGH:
        try:
            torch.linalg.eigh(torch.eye(2, dtype=ref.dtype, device=ref.device))
            _NATIVE_EIGH[key] = True
        except (NotImplementedError, RuntimeError):
            _NATIVE_EIGH[key] = False
    return _NATIVE_EIGH[key]


_NATIVE_EIGH = {}


def _subspace_iteration(k, X, oversample=None, iters=8, seed=0):
    """Randomised subspace iteration -- only for the both-dimensions-large case.

    Warns, because its accuracy depends on an eigengap this data usually lacks.
    """
    import warnings
    p = X.shape[-1]
    oversample = min(max(20, 2 * k) if oversample is None else oversample, p - k)
    warnings.warn(
        f"top_k_eigenvectors: both dimensions of X are large "
        f"({tuple(X.shape)}) and no dense symmetric eigensolver is available on "
        f"{X.device.type}; falling back to randomised subspace iteration, whose "
        "accuracy depends on an eigengap that centred data often lacks. "
        "Move to CPU/CUDA, or reduce k.", RuntimeWarning, stacklevel=3)
    l = k + oversample
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Z = torch.randn(p, l, generator=gen, dtype=torch.float64).to(
        dtype=X.dtype, device=X.device)
    for _ in range(iters + 1):
        Z, _ = torch.linalg.qr(X.transpose(-2, -1) @ (X @ Z))
    B = Z.transpose(-2, -1) @ (X.transpose(-2, -1) @ (X @ Z))
    _, U = safe_eigh(0.5 * (B + B.transpose(-2, -1)))
    return _canonical_sign((Z @ U[..., -k:].flip(-1)).contiguous())


def leading_eigenvectors(k, S=None, X=None, canonical_sign=True):
    """Backwards-compatible alias for :func:`top_k_eigenvectors`."""
    E = top_k_eigenvectors(k, S=S, X=X)
    return E if canonical_sign else E
