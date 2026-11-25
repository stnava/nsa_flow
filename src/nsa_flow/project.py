"""Projections used by NSA-Flow.

The polar factor is computed with an explicit, numerically robust derivative.
Routing it through ``torch.linalg.svd`` and letting autograd differentiate the
factorisation fails exactly where the polar factor is best behaved: SVD backward
divides by ``sigma_i^2 - sigma_j^2``, which vanishes whenever two singular values
coincide -- as they do, for instance, at any orthogonal initialisation.  The
polar factor itself is smooth wherever ``Y`` has full column rank; its derivative
divides by ``h_i + h_j > 0`` instead, and so never degenerates.
"""
import torch

from .linalg import safe_eigh

__all__ = ["project_nonneg", "project_scaled_stiefel", "polar_factor",
           "simlr_sparseness", "_resolve_column_sign_gauge"]


def project_nonneg(Y):
    """Euclidean projection onto ``{Y >= 0}``."""
    return Y.clamp_min(0.0)


def _resolve_sign_ties_2d(Yd, sign, tied):
    """Resolve 2-D sign-gauge ties against unambiguous positive row mass."""
    unambig_pos = (Yd[:, ~tied] * sign[~tied].unsqueeze(0)).clamp_min(0.0)
    row_mass = unambig_pos.pow(2).sum(dim=1)
    sign_out = sign.clone()
    for j in torch.nonzero(tied, as_tuple=False).flatten().tolist():
        c_pos = Yd[:, j].clamp_min(0.0)
        c_neg = (-Yd[:, j]).clamp_min(0.0)
        neg_ov, pos_ov = torch.stack([
            (c_neg.pow(2) * row_mass).sum(),
            (c_pos.pow(2) * row_mass).sum(),
        ]).tolist()
        if neg_ov < pos_ov:
            sign_out[j] = -1.0
            row_mass = row_mass + c_neg.pow(2)
        else:
            sign_out[j] = 1.0
            row_mass = row_mass + c_pos.pow(2)
    return sign_out


def _resolve_column_sign_gauge(Y):
    """Flip any column j whose negative-part L2 norm exceeds its positive-part L2 norm.

    Supports 2-D ``[p, k]`` and 3-D ``[B, p, k]`` tensors natively on-device.
    When positive and negative lobe norms tie within working precision (e.g. a
    Hadamard contrast column ``[+s, -s]``), resolves the tie against the row
    energy of unambiguous columns so tied columns orient onto distinct rows.
    """
    if Y.numel() == 0 or Y.ndim < 2:
        return Y
    if Y.ndim not in (2, 3):
        raise ValueError(
            f"_resolve_column_sign_gauge expects 2-D [p, k] or 3-D [B, p, k] input; got shape {tuple(Y.shape)}"
        )
    pos = torch.linalg.vector_norm(Y.clamp_min(0.0), dim=-2)
    neg = torch.linalg.vector_norm((-Y).clamp_min(0.0), dim=-2)
    sign = torch.where(neg > pos, -torch.ones_like(pos), torch.ones_like(pos))
    if Y.shape[-1] > 1:
        eps_rel = 16.0 * torch.finfo(Y.dtype).eps
        scale = pos + neg
        tied = (scale > 0) & ((pos - neg).abs() <= eps_rel * scale)
        has_tie = tied.any()
        if (has_tie.tolist() if Y.device.type != "cpu" else bool(has_tie)):
            Yd = Y.detach()
            if Y.ndim == 2:
                sign = _resolve_sign_ties_2d(Yd, sign, tied)
            else:
                sign = sign.clone()
                tied_b = tied.any(dim=-1)
                for b in torch.nonzero(tied_b, as_tuple=False).flatten().tolist():
                    sign[b] = _resolve_sign_ties_2d(Yd[b], sign[b], tied[b])
    return Y * sign.unsqueeze(-2)


def _simlr_sparseness_tie_break_2d(Y_2d, out_2d, tied_1d, mag_d_2d, vals_2d,
                                   tol_j_1d, n_keep, use_soft, eps_dtype):
    """Pass-2 joint overlap-minimising tie-breaker for a single 2-D ``[p, k]`` matrix."""
    p, k = Y_2d.shape
    cols_out = [None] * k
    pending = []
    tied_flags = tied_1d.tolist()
    for j in range(k):
        if not tied_flags[j]:
            cols_out[j] = out_2d[:, j]
            continue
        thresh = vals_2d[n_keep, j]
        tol_j = tol_j_1d[j]
        mag_dj = mag_d_2d[:, j]
        if use_soft:
            strict_mask = torch.zeros(p, dtype=torch.bool, device=Y_2d.device)
            tie_mask = mag_dj >= thresh - tol_j
            pending.append((int(tie_mask.sum().tolist()), j, strict_mask, tie_mask))
        else:
            strict_mask = mag_dj > thresh + tol_j
            tie_mask = (~strict_mask) & (mag_dj >= thresh - tol_j)
            pending.append((int(tie_mask.sum().tolist()), j, strict_mask, tie_mask))

    if not pending:
        return out_2d

    tiny = torch.finfo(Y_2d.dtype).tiny
    row_load = torch.zeros(p, dtype=torch.int64, device=Y_2d.device)
    committed = []
    for j in range(k):
        if cols_out[j] is not None:
            cj = cols_out[j].detach()
            nz = cj != 0
            if nz.any().tolist():
                row_load = row_load + nz.to(torch.int64)
                committed.append(cj / torch.linalg.vector_norm(cj).clamp_min(tiny))
    for _, _, strict_mask, _ in pending:
        if strict_mask.any().tolist():
            row_load = row_load + strict_mask.to(torch.int64)

    pending.sort(key=lambda item: (item[0], item[1]))
    q_step = 64.0 * eps_dtype
    is_mps = (Y_2d.device.type == "mps")
    for _, j, strict_mask, tie_mask in pending:
        col = Y_2d[:, j]
        col_d = col.detach()
        need = n_keep - int(strict_mask.sum().tolist())
        tie_idx = torch.nonzero(tie_mask, as_tuple=False).flatten()
        chosen_mask = strict_mask.clone()
        cur_vec = torch.where(chosen_mask, col_d, torch.zeros_like(col_d))
        cur_norm_sq = cur_vec.pow(2).sum()
        U = torch.stack(committed, dim=1) if committed else None
        g = (U.transpose(0, 1) @ cur_vec) if U is not None else None
        avail = torch.ones(tie_idx.numel(), dtype=torch.bool, device=Y_2d.device)

        for _ in range(min(need, tie_idx.numel())):
            pos_avail = torch.nonzero(avail, as_tuple=False).flatten()
            cand_idx = tie_idx[pos_avail]
            y_c = col_d[cand_idx]
            if U is not None:
                denom = (cur_norm_sq + y_c.pow(2)).clamp_min(tiny)
                dots = g.unsqueeze(0) + y_c.unsqueeze(1) * U[cand_idx]
                overlap = dots.pow(2).sum(dim=1) / denom
                ov_scaled = (overlap.float() if is_mps else overlap.double()) / q_step
                ov_bucket = torch.round(ov_scaled)
                m1 = ov_bucket == ov_bucket.min()
            else:
                m1 = torch.ones(cand_idx.numel(), dtype=torch.bool, device=Y_2d.device)
            sub_pos = pos_avail[m1]
            rl_sub = row_load[tie_idx[sub_pos]]
            pick_slot_t = sub_pos[rl_sub == rl_sub.min()][0]
            pick_row_t = tie_idx[pick_slot_t]
            pick_slot, pick_row = torch.stack([pick_slot_t, pick_row_t]).tolist()
            avail[pick_slot] = False
            chosen_mask[pick_row] = True
            val_pick = col_d[pick_row]
            cur_vec[pick_row] = val_pick
            cur_norm_sq = cur_norm_sq + val_pick * val_pick
            row_load[pick_row] += 1
            if U is not None:
                g = g + val_pick * U[pick_row]

        out_j = torch.where(chosen_mask, col, torch.zeros_like(col))
        cols_out[j] = out_j
        out_d = out_j.detach()
        if (out_d.abs().max() > 0.0).tolist():
            committed.append(out_d / torch.linalg.vector_norm(out_d).clamp_min(tiny))

    return torch.stack(cols_out, dim=1)


def simlr_sparseness(Y, sparseness=None, nonneg=False, alg="soft", only_if_denser=False):
    r"""Column-wise quantile soft/hard thresholding (R ``ANTsR::simlr`` convention).

    Accepts 2-D ``[p, k]`` or 3-D batched ``[B, p, k]`` tensors natively on-device.
    For each column ``j``, retains at most the top
    ``n_keep = max(1, round(p * (1 - sparseness)))`` entries by magnitude and
    shrinks surviving entries by the ``(1 - sparseness)`` order-statistic
    threshold when ``alg == 'soft'``.

    When magnitudes tie at the ``n_keep``-th order-statistic boundary, breaks
    ties jointly across columns to minimise squared cosine overlap with already-
    committed columns (and row usage count), preserving column rank and
    orthogonality on Hadamard and standardised symmetric bases while leaving
    untied columns unchanged.

    When ``nonneg=True``, resolves the per-column sign gauge first (flipping any
    column whose negative-part L2 norm exceeds its positive-part L2 norm) before
    clamping at zero, so a predominantly negative or all-negative column is not
    zeroed out or inverted onto noise.

    When ``only_if_denser=True``, any column whose existing zero fraction is
    already ``>= sparseness`` is left untouched so a naturally sparser non-negative
    parts column is never degraded by redundant shrinkage.
    """
    if Y.ndim not in (2, 3):
        raise ValueError(
            f"simlr_sparseness expects 2-D [p, k] or 3-D [B, p, k] input; got shape {tuple(Y.shape)}"
        )
    if nonneg:
        Y = _resolve_column_sign_gauge(Y).clamp_min(0.0)
    if sparseness is None:
        return Y
    q = float(sparseness)
    if q <= 0.0 or Y.numel() == 0:
        return Y
    q = min(q, 0.999)
    p, k = Y.shape[-2], Y.shape[-1]
    if p <= 1 or k == 0:
        return Y
    n_keep = max(1, int(round(p * (1.0 - q))))
    if n_keep >= p:
        return Y

    eps_dtype = torch.finfo(Y.dtype).eps
    eps_tie_rel = 16.0 * eps_dtype
    use_soft = (str(alg).lower() != "hard")

    mag = Y.abs()
    mag_d = mag.detach()
    max_mag = mag_d.amax(dim=-2, keepdim=True)
    zero_col = max_mag <= 0.0

    if only_if_denser:
        clean_mask = mag_d > (1e-12 * max_mag)
        nnz = clean_mask.sum(dim=-2, keepdim=True)
        already_sparse = (~zero_col) & (nnz <= n_keep)
        clean_Y = torch.where(clean_mask, Y, torch.zeros_like(Y))
    else:
        already_sparse = torch.zeros_like(zero_col)
        clean_Y = None

    tol_j = eps_tie_rel * max_mag
    vals, idx_sort = torch.sort(mag_d, dim=-2, descending=True)
    thresh = vals[..., n_keep:n_keep + 1, :]
    val_last_keep = vals[..., n_keep - 1:n_keep, :]

    if use_soft:
        val_top = vals[..., 0:1, :]
        shrunk = torch.sign(Y) * torch.clamp(mag - thresh, min=0.0)
        near_tie = (val_last_keep - thresh) <= tol_j
        shrunk_clean = torch.where(mag_d > thresh + tol_j, shrunk, torch.zeros_like(shrunk))
        untied_out = torch.where(near_tie, shrunk_clean, shrunk)
        tied_col = (~zero_col) & (~already_sparse) & ((val_top - thresh) <= tol_j)
    else:
        keep_mask = torch.zeros_like(mag_d, dtype=torch.bool)
        keep_mask.scatter_(-2, idx_sort[..., :n_keep, :], True)
        untied_out = torch.where(keep_mask, Y, torch.zeros_like(Y))
        tied_col = (~zero_col) & (~already_sparse) & (thresh > tol_j) & ((val_last_keep - thresh) <= tol_j)

    out = torch.where(already_sparse, clean_Y, untied_out) if only_if_denser else untied_out
    out = torch.where(zero_col, torch.zeros_like(Y), out)

    has_tie = tied_col.any()
    if not (has_tie.tolist() if Y.device.type != "cpu" else bool(has_tie)):
        return out

    if Y.ndim == 2:
        return _simlr_sparseness_tie_break_2d(
            Y, out, tied_col.squeeze(0), mag_d, vals, tol_j.squeeze(0),
            n_keep, use_soft, eps_dtype,
        )

    tied_b = tied_col.squeeze(-2).any(dim=-1)
    out_slices = list(out.unbind(0))
    for b in torch.nonzero(tied_b, as_tuple=False).flatten().tolist():
        out_slices[b] = _simlr_sparseness_tie_break_2d(
            Y[b], out[b], tied_col[b, 0], mag_d[b], vals[b], tol_j[b, 0],
            n_keep, use_soft, eps_dtype,
        )
    return torch.stack(out_slices, dim=0)


class _PolarFactor(torch.autograd.Function):
    r"""``U`` in the polar decomposition ``Y = U H``, ``U'U = I``, ``H = (Y'Y)^{1/2}``.

    Forward: eigendecompose ``S = Y'Y = Q diag(lambda) Q'`` and set
    ``U = Y Q diag(lambda^{-1/2}) Q'``.

    Backward: differentiating ``Y = U H`` gives
    ``dU = U Omega + (I - U U') dY H^{-1}`` with ``Omega`` skew solving the
    Sylvester equation ``Omega H + H Omega = U' dY - dY' U``.  Taking adjoints,

        Ybar = 2 U Psi + (I - U U') Ubar H^{-1},
        Psi H + H Psi = skew(U' Ubar),

    which in the eigenbasis of ``H`` is ``Psi_ij = R_ij / (h_i + h_j)`` with
    ``h_i = sqrt(lambda_i) > 0``.  The denominator is a *sum* of singular values,
    so repeated singular values are harmless.
    """

    @staticmethod
    def forward(ctx, Y, eps):
        S = Y.transpose(-2, -1) @ Y
        lam, Q = safe_eigh(S)
        eps_eff = max(float(eps), float(torch.finfo(Y.dtype).tiny))
        lam = lam.clamp_min(eps_eff)
        h = lam.sqrt()
        inv_h = h.reciprocal()
        H_inv = (Q * inv_h.unsqueeze(-2)) @ Q.transpose(-2, -1)
        U = Y @ H_inv
        ctx.save_for_backward(U, Q, h)
        return U

    @staticmethod
    def backward(ctx, U_bar):
        U, Q, h = ctx.saved_tensors
        B = U.transpose(-2, -1) @ U_bar
        R = 0.5 * (B - B.transpose(-2, -1))                     # skew part
        R_t = Q.transpose(-2, -1) @ R @ Q                       # eigenbasis of H
        denom = h.unsqueeze(-1) + h.unsqueeze(-2)               # h_i + h_j > 0
        Psi = Q @ (R_t / denom) @ Q.transpose(-2, -1)
        H_inv = (Q * h.reciprocal().unsqueeze(-2)) @ Q.transpose(-2, -1)
        tangential = U_bar - U @ (U.transpose(-2, -1) @ U_bar)   # (I - UU') Ubar
        return 2.0 * (U @ Psi) + tangential @ H_inv, None


def polar_factor(Y, eps=1e-30):
    """Orthonormal polar factor of ``Y`` (columns orthonormal), safely differentiable.

    Requires ``Y`` to have full column rank; ``eps`` floors the Gram eigenvalues
    so that a numerically rank-deficient input degrades gracefully rather than
    producing ``inf``.
    """
    return _PolarFactor.apply(Y, eps)


def project_scaled_stiefel(Y, eps=1e-30):
    r"""Euclidean projection onto ``R_{>0} * St(p, k)`` -- the zero set of ``D``.

    ``argmin_{c > 0, Q'Q = I} ||Y - cQ||_F^2`` is attained at ``Q = U`` (the polar
    factor) and ``c = (sum_i sigma_i) / k = <U, Y> / k``, giving

        P(Y) = (<U, Y> / k) U .

    This is *not* ``(||Y||_F / sqrt(k)) U``: rescaling the polar factor to
    preserve the Frobenius norm is a different, non-optimal map.
    """
    k = Y.shape[-1]
    if k <= 1:
        return Y
    U = polar_factor(Y, eps=eps)
    c = (U * Y).sum((-2, -1)) / k
    return c.reshape(*c.shape, 1, 1) * U
