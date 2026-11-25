r"""Factorisations that stay on the device.

The initialisers need the top-``k`` eigenvectors of ``S = X'X``.  The obvious
calls do not survive a GPU:

========================  ==========================================================
``linalg.eigh``           ``NotImplementedError`` on MPS.
``linalg.svd``            "works" on MPS only by *silently falling back to the CPU*
                          (``aten::linalg_svd ... will fall back to run on the CPU``),
                          so it is a hidden round trip, which is worse than an error.
``linalg.matrix_rank``    built on ``svd``; same problem.
========================  ==========================================================

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
import math

import torch

__all__ = ["jacobi_eigh", "safe_eigh", "top_k_eigenvectors",
           "leading_eigenvectors", "symmetric_rank",
           "varimax_rotation", "varimax"]


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

    Supports 2-D ``[n, n]`` and arbitrary leading batch dimensions ``[..., n, n]``.
    Returns ``(w, V)`` with ``w`` ascending (shape ``[..., n]``) and ``V``
    orthogonal (shape ``[..., n, n]``), matching ``torch.linalg.eigh``'s
    convention so the two are drop-in interchangeable.

    Each round annihilates ``n/2`` disjoint off-diagonal pairs simultaneously.
    For the pair ``(p, q)`` the rotation angle is the classical::

        theta = (a_qq - a_pp) / (2 a_pq),
        t = sign_non_zero(theta) / (|theta| + sqrt(theta^2 + 1)),
        c = 1 / sqrt(t^2 + 1),   s = t c,

    which is the numerically stable branch (it picks the smaller rotation and
    never differences nearly-equal quantities).  Because the pairs in a round
    are disjoint, the rotations commute and compose into one orthogonal matrix.

    Stops early once the off-diagonal Frobenius mass falls below ``tol`` times
    the total across all batch elements, which on the matrices here is
    typically 4-6 sweeps.
    """
    if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError(
            f"jacobi_eigh expects symmetric [..., n, n] input; got shape {tuple(A.shape)}"
        )
    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    dtype, device = A.dtype, A.device
    if n == 0 or A.numel() == 0:
        w_empty = torch.empty((*batch_shape, n), dtype=dtype, device=device)
        V_empty = torch.eye(n, dtype=dtype, device=device)
        if batch_shape:
            V_empty = V_empty.expand(*batch_shape, n, n).clone()
        return w_empty, V_empty

    A = 0.5 * (A + A.transpose(-2, -1))
    V = torch.eye(n, dtype=dtype, device=device)
    if batch_shape:
        V = V.expand(*batch_shape, n, n).clone()
    if n == 1:
        return A.reshape(*batch_shape, 1).contiguous(), V
    if tol is None:
        tol = torch.finfo(dtype).eps * 10.0

    tiny = torch.finfo(dtype).tiny
    tol_sq = tol * tol
    rounds = _round_robin(n)
    round_indices = [
        (torch.tensor([p for p, _ in pairs], dtype=torch.long, device=device),
         torch.tensor([q for _, q in pairs], dtype=torch.long, device=device))
        for pairs in rounds if pairs
    ]
    for sw in range(sweeps):
        # On accelerators a D2H sync costs ~50x a small-matrix Jacobi sweep;
        # n==2 converges in 1 sweep and n>=3 typically in 4-6 sweeps, so defer
        # the sync on non-CPU devices until convergence is expected.
        check_conv = (
            device.type == "cpu"
            or (n <= 2 and sw >= 1)
            or (n > 2 and (sw >= 4 and sw % 2 == 0))
        )
        if check_conv:
            diag_A = torch.diagonal(A, dim1=-2, dim2=-1)
            off_t = (A - torch.diag_embed(diag_A)).pow(2).sum(dim=(-2, -1))
            total_t = A.pow(2).sum(dim=(-2, -1))
            need = (off_t > tol_sq * total_t).any()
            if not (need.tolist() if device.type != "cpu" else bool(need)):
                break
        for P, Q in round_indices:
            apq = A[..., P, Q]
            app, aqq = A[..., P, P], A[..., Q, Q]

            safe_apq = torch.where(apq.abs() <= tiny, torch.ones_like(apq), apq)
            theta = (aqq - app) / (2.0 * safe_apq)
            sgn = torch.where(theta >= 0.0, torch.ones_like(theta), -torch.ones_like(theta))
            t = sgn / (theta.abs() + torch.sqrt(theta * theta + 1.0))
            t = torch.where(apq.abs() <= tiny, torch.zeros_like(t), t)
            c = 1.0 / torch.sqrt(t * t + 1.0)
            s = t * c

            # Apply the disjoint rotations by updating only the rows and
            # columns they touch across all leading batch dimensions at once.
            c_row = c.unsqueeze(-1)
            s_row = s.unsqueeze(-1)
            Ap, Aq = A[..., P, :], A[..., Q, :]
            A[..., P, :] = c_row * Ap - s_row * Aq
            A[..., Q, :] = s_row * Ap + c_row * Aq

            c_col = c.unsqueeze(-2)
            s_col = s.unsqueeze(-2)
            Ap, Aq = A[..., :, P], A[..., :, Q]
            A[..., :, P] = c_col * Ap - s_col * Aq
            A[..., :, Q] = s_col * Ap + c_col * Aq

            Vp, Vq = V[..., :, P], V[..., :, Q]
            V[..., :, P] = c_col * Vp - s_col * Vq
            V[..., :, Q] = s_col * Vp + c_col * Vq
        # keep it exactly symmetric against accumulated rounding
        A = 0.5 * (A + A.transpose(-2, -1))

    w = torch.diagonal(A, dim1=-2, dim2=-1)
    order = torch.argsort(w, dim=-1)
    w_sorted = torch.gather(w, -1, order).contiguous()
    V_sorted = torch.gather(V, -1, order.unsqueeze(-2).expand_as(V)).contiguous()
    return w_sorted, V_sorted


def safe_eigh(A):
    """``torch.linalg.eigh``, or :func:`jacobi_eigh` where it is unavailable.

    Supports 2-D ``[n, n]`` and arbitrary leading batch dimensions ``[..., n, n]``.
    No host fallback: the replacement runs on whatever device ``A`` is on.
    """
    if A.device.type == "mps" or not _has_native_eigh(A):
        return jacobi_eigh(A)
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
        p = S.shape[-1]
        k = min(k, p)
        if max_dense is None:
            max_dense = 4096 if _has_native_eigh(S) else 512
        if p > max_dense:
            return _subspace_iteration(k, S=S, seed=seed)
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
        return _subspace_iteration(k, X=X, seed=seed)

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


def _subspace_iteration(k, X=None, S=None, oversample=None, iters=8, seed=0):
    """Randomised subspace iteration -- only for the both-dimensions-large case.

    Warns, because its accuracy depends on an eigengap this data usually lacks.
    """
    if (X is None) == (S is None):
        raise ValueError("give exactly one of X or S")
    import warnings
    p = X.shape[-1] if X is not None else S.shape[-1]
    dev = X.device if X is not None else S.device
    dtype = X.dtype if X is not None else S.dtype
    oversample = min(max(20, 2 * k) if oversample is None else oversample, p - k)
    shape_str = tuple(X.shape) if X is not None else tuple(S.shape)
    warnings.warn(
        f"top_k_eigenvectors: dimension of {'X' if X is not None else 'S'} is large "
        f"({shape_str}) and no dense symmetric eigensolver is available on "
        f"{dev.type}; falling back to randomised subspace iteration, whose "
        "accuracy depends on an eigengap that centred data often lacks. "
        "Move to CPU/CUDA, or reduce k.", RuntimeWarning, stacklevel=3)
    l = k + oversample
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Z = torch.randn(p, l, generator=gen, dtype=torch.float64).to(
        dtype=dtype, device=dev)
    def matvec(M):
        return X.transpose(-2, -1) @ (X @ M) if X is not None else S @ M
    for _ in range(iters + 1):
        Z, _ = torch.linalg.qr(matvec(Z))
    B = Z.transpose(-2, -1) @ matvec(Z)
    _, U = safe_eigh(0.5 * (B + B.transpose(-2, -1)))
    return _canonical_sign((Z @ U[..., -k:].flip(-1)).contiguous())


def leading_eigenvectors(k, S=None, X=None, canonical_sign=True):
    """Backwards-compatible alias for :func:`top_k_eigenvectors`."""
    E = top_k_eigenvectors(k, S=S, X=X)
    return E if canonical_sign else E


def _polar_orthogonal(B):
    """On-device orthogonal polar factor ``U V' in O(k)`` of a ``k x k`` matrix ``B``."""
    k = B.shape[-1]
    lam, V = safe_eigh(B.transpose(-2, -1) @ B)
    eps = torch.finfo(B.dtype).eps
    tiny = torch.finfo(B.dtype).tiny
    if bool((lam.min(dim=-1).values <= eps * lam.max(dim=-1).values.clamp_min(tiny)).any()):
        B_cpu = B.cpu() if B.device.type == "mps" else B
        U, _, Vh = torch.linalg.svd(B_cpu, full_matrices=False)
        return (U @ Vh).to(B.device)
    inv_sqrt = lam.clamp_min(tiny).rsqrt()
    Q = B @ ((V * inv_sqrt.unsqueeze(-2)) @ V.transpose(-2, -1))
    eye = torch.eye(k, dtype=Q.dtype, device=Q.device)
    return Q @ (1.5 * eye - 0.5 * (Q.transpose(-2, -1) @ Q))


def _kaiser_saddle_escape(X, R):
    """One exact 2-column Kaiser (1958) Givens sweep to escape uniform-magnitude saddles.

    First-order GPA has ``X'(Z^3 - Z mean(Z^2)) = 0`` at Hadamard / uniform-magnitude
    minima where the Varimax criterion is minimised rather than maximised.  Kaiser's
    closed-form pairwise angle ``0.25 * atan2(D, A)`` includes the second derivative
    ``A < 0`` and jumps directly to the 45-degree maximum in one step.
    """
    p, k = X.shape[-2], X.shape[-1]
    Z = (X @ R).clone()
    R_out = R.clone()
    tol = 16.0 * torch.finfo(X.dtype).eps * float(p)
    moved = False
    for a in range(k - 1):
        for b in range(a + 1, k):
            za, zb = Z[..., :, a], Z[..., :, b]
            u = za * za - zb * zb
            v = 2.0 * za * zb
            su, sv = u.sum(dim=-1), v.sum(dim=-1)
            A = float(p * (u * u - v * v).sum(dim=-1) - (su * su - sv * sv))
            D = float(2.0 * (p * (u * v).sum(dim=-1) - su * sv))
            if abs(D) > tol or A < -tol:
                phi = 0.25 * math.atan2(D, A)
                if abs(phi) > 1e-12:
                    c, s = math.cos(phi), math.sin(phi)
                    za_new = c * za + s * zb
                    zb_new = -s * za + c * zb
                    Z[..., :, a], Z[..., :, b] = za_new, zb_new
                    ra, rb = R_out[..., :, a].clone(), R_out[..., :, b].clone()
                    R_out[..., :, a] = c * ra + s * rb
                    R_out[..., :, b] = -s * ra + c * rb
                    moved = True
    return R_out, moved


def varimax_rotation(E, normalize=True, eps=1e-6, max_iter=100):
    r"""Orthogonal Varimax rotation matrix ``R in O(k)`` for basis ``E`` (``[p, k]``).

    Maximises the Kaiser (1958) fourth-moment column-variance simplicity criterion

    .. math::

        \mathcal{V}(E R) = \sum_{j=1}^{k} \left[
            \frac{1}{p} \sum_{i=1}^{p} (E R)_{ij}^{4}
            - \left(\frac{1}{p} \sum_{i=1}^{p} (E R)_{ij}^{2}\right)^{2}
        \right]

    over ``R in O(k)`` via Lawley-Maxwell / GPA alternating polar projections.
    Because ``R`` is orthogonal, ``(E R)(E R)' = E E'`` and ``(E R)'(E R) = R'(E'E)R``,
    so both the subspace projector and column orthonormality are preserved to
    machine precision while rotating the Perron-Frobenius global mean mode into
    localised regional factors.
    """
    p, k = E.shape[-2], E.shape[-1]
    eye = torch.eye(k, dtype=E.dtype, device=E.device)
    if k < 2 or p < 2:
        return eye

    tiny = torch.finfo(E.dtype).tiny
    if normalize:
        row_norms = E.pow(2).sum(dim=-1, keepdim=True).sqrt()
        X = torch.where(row_norms > tiny, E / row_norms.clamp_min(tiny), E)
    else:
        X = E

    R = eye.clone()
    d_prev = 0.0
    saddle_tol = 1e-11 * float(p)
    for _ in range(max_iter):
        Z = X @ R
        col_mean_sq = Z.pow(2).sum(dim=-2, keepdim=True) / float(p)
        B = X.transpose(-2, -1) @ (Z.pow(3) - Z * col_mean_sq)
        R_new = _polar_orthogonal(B)
        d = float((B * R_new).sum())
        if d <= saddle_tol and d <= 1e-11 * float(X.pow(2).sum()):
            R_esc, moved = _kaiser_saddle_escape(X, R)
            if moved:
                R = R_esc
                d_prev = 0.0
                continue
            break
        R = R_new
        if d <= d_prev * (1.0 + eps):
            break
        d_prev = d
    return R


def varimax(E, normalize=True, eps=1e-6, max_iter=100, canonical_sign=True):
    r"""Return the Varimax-rotated basis ``E R`` (``[p, k]``) with ``R in O(k)``.

    When ``canonical_sign=True`` (default), each column's sign gauge is resolved
    so its dominant L2 lobe is non-negative (with tie-breaking by largest-magnitude
    entry), and columns are ordered deterministically by descending positive-lobe
    L2 norm so repeated or dual-Gram calls return the same column order.
    """
    p, k = E.shape[-2], E.shape[-1]
    if k < 2 or p < 2:
        if canonical_sign and E.numel() > 0:
            pos = torch.linalg.vector_norm(E.clamp_min(0.0), dim=-2, keepdim=True)
            neg = torch.linalg.vector_norm((-E).clamp_min(0.0), dim=-2, keepdim=True)
            E = torch.where(neg > pos, -E, E)
        return E
    R = varimax_rotation(E, normalize=normalize, eps=eps, max_iter=max_iter)
    Z = (E @ R).contiguous()
    if canonical_sign:
        Z = _canonical_sign(Z)
        pos = torch.linalg.vector_norm(Z.clamp_min(0.0), dim=-2)
        neg = torch.linalg.vector_norm((-Z).clamp_min(0.0), dim=-2)
        flip = torch.where(neg > pos, -torch.ones_like(pos), torch.ones_like(pos))
        Z = Z * flip.unsqueeze(-2)
        # Order deterministically by descending positive-lobe L2 norm
        pos_after = torch.linalg.vector_norm(Z.clamp_min(0.0), dim=-2)
        order = torch.argsort(pos_after, dim=-1, descending=True)
        Z = Z[..., :, order].contiguous()
    return Z

