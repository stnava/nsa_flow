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


def _resolve_column_sign_gauge(Y):
    """Flip any column j whose negative-part L2 norm exceeds its positive-part L2 norm.

    When positive and negative lobe norms tie within working precision (e.g. a
    Hadamard contrast column ``[+s, -s]``), resolves the tie against the row
    energy of unambiguous columns so tied columns orient onto distinct rows.
    """
    if Y.numel() == 0 or Y.ndim < 2:
        return Y
    pos = torch.linalg.vector_norm(Y.clamp_min(0.0), dim=0)
    neg = torch.linalg.vector_norm((-Y).clamp_min(0.0), dim=0)
    sign = torch.where(neg > pos, -torch.ones_like(pos), torch.ones_like(pos))
    if Y.shape[1] > 1:
        eps_rel = 16.0 * torch.finfo(Y.dtype).eps
        scale = pos + neg
        tied = (scale > 0) & ((pos - neg).abs() <= eps_rel * scale)
        if bool(tied.any()):
            Yd = Y.detach()
            unambig_pos = (Yd[:, ~tied] * sign[~tied].unsqueeze(0)).clamp_min(0.0)
            row_mass = unambig_pos.pow(2).sum(dim=1)
            sign = sign.clone()
            for j in torch.nonzero(tied, as_tuple=False).flatten().tolist():
                c_pos = Yd[:, j].clamp_min(0.0)
                c_neg = (-Yd[:, j]).clamp_min(0.0)
                if float((c_neg.pow(2) * row_mass).sum()) < float((c_pos.pow(2) * row_mass).sum()):
                    sign[j] = -1.0
                    row_mass = row_mass + c_neg.pow(2)
                else:
                    sign[j] = 1.0
                    row_mass = row_mass + c_pos.pow(2)
    return Y * sign.unsqueeze(0)


def simlr_sparseness(Y, sparseness=None, nonneg=False, alg="soft", only_if_denser=False):
    r"""Column-wise quantile soft/hard thresholding (R ``ANTsR::simlr`` convention).

    For each column ``j`` of ``Y`` (shape ``[p, k]``), retains at most the top
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
    if nonneg:
        Y = _resolve_column_sign_gauge(Y).clamp_min(0.0)
    if sparseness is None:
        return Y
    q = float(sparseness)
    if q <= 0.0 or Y.numel() == 0:
        return Y
    q = min(q, 0.999)
    p, k = Y.shape
    if p <= 1 or k == 0:
        return Y
    n_keep = max(1, int(round(p * (1.0 - q))))
    if n_keep >= p:
        return Y

    eps_dtype = torch.finfo(Y.dtype).eps
    eps_tie_rel = 16.0 * eps_dtype
    cols_out = [None] * k
    pending = []
    use_soft = (str(alg).lower() != "hard")

    for j in range(k):
        col = Y[:, j]
        mag = col.abs()
        mag_d = mag.detach()
        max_mag = float(mag_d.max())
        if max_mag <= 0.0:
            cols_out[j] = torch.zeros_like(col)
            continue
        if only_if_denser:
            nnz_j = int((mag_d > 1e-12 * max_mag).sum())
            if nnz_j <= n_keep:
                # Zero out tiny numerical dust below 1e-12 * max_mag and keep intact
                clean_mask = mag_d > 1e-12 * max_mag
                cols_out[j] = torch.where(clean_mask, col, torch.zeros_like(col))
                continue
        tol_j = eps_tie_rel * max_mag
        vals, idx_sort = torch.sort(mag_d, descending=True)
        thresh = vals[n_keep]
        if use_soft:
            if float(vals[0] - thresh) > tol_j:
                shrunk = torch.sign(col) * torch.clamp(mag - thresh, min=0.0)
                if float(vals[n_keep - 1] - thresh) <= tol_j:
                    # Clean 1-ULP dust on entries tied with thresh
                    shrunk = torch.where(mag_d > thresh + tol_j, shrunk, torch.zeros_like(shrunk))
                cols_out[j] = shrunk
            else:
                # All top n_keep + 1 magnitudes tie with thresh (> 0); soft shrinkage
                # would zero the entire column, so defer hard selection from the tie pool.
                strict_mask = torch.zeros(p, dtype=torch.bool, device=Y.device)
                tie_mask = mag_d >= thresh - tol_j
                pending.append((int(tie_mask.sum()), j, strict_mask, tie_mask))
        else:
            if float(thresh) > tol_j and float(vals[n_keep - 1] - thresh) <= tol_j:
                strict_mask = mag_d > thresh + tol_j
                tie_mask = (~strict_mask) & (mag_d >= thresh - tol_j)
                pending.append((int(tie_mask.sum()), j, strict_mask, tie_mask))
            else:
                keep_mask = torch.zeros(p, dtype=torch.bool, device=Y.device)
                keep_mask[idx_sort[:n_keep]] = True
                cols_out[j] = torch.where(keep_mask, col, torch.zeros_like(col))

    if not pending:
        return torch.stack(cols_out, dim=1)

    # Pass 2 (only reached when at least one column has a positive boundary tie):
    # select tied support entries to minimise squared overlap with committed columns.
    tiny = torch.finfo(Y.dtype).tiny
    row_load = torch.zeros(p, dtype=torch.int64, device=Y.device)
    committed = []
    for j in range(k):
        if cols_out[j] is not None:
            cj = cols_out[j].detach()
            nz = cj != 0
            if bool(nz.any()):
                row_load = row_load + nz.to(torch.int64)
                committed.append(cj / torch.linalg.vector_norm(cj).clamp_min(tiny))
    for _, _, strict_mask, _ in pending:
        if bool(strict_mask.any()):
            row_load = row_load + strict_mask.to(torch.int64)

    pending.sort(key=lambda item: (item[0], item[1]))
    q_step = 64.0 * eps_dtype
    for _, j, strict_mask, tie_mask in pending:
        col = Y[:, j]
        col_d = col.detach()
        need = n_keep - int(strict_mask.sum())
        tie_idx = torch.nonzero(tie_mask, as_tuple=False).flatten()
        chosen_mask = strict_mask.clone()
        cur_vec = torch.where(chosen_mask, col_d, torch.zeros_like(col_d))
        cur_norm_sq = float(cur_vec.pow(2).sum())
        U = torch.stack(committed, dim=1) if committed else None
        g = (U.transpose(0, 1) @ cur_vec) if U is not None else None
        avail = torch.ones(tie_idx.numel(), dtype=torch.bool, device=Y.device)

        for _ in range(min(need, int(tie_idx.numel()))):
            pos_avail = torch.nonzero(avail, as_tuple=False).flatten()
            cand_idx = tie_idx[pos_avail]
            y_c = col_d[cand_idx]
            if U is not None:
                denom = (cur_norm_sq + y_c.pow(2)).clamp_min(tiny)
                dots = g.unsqueeze(0) + y_c.unsqueeze(1) * U[cand_idx]
                overlap = dots.pow(2).sum(dim=1) / denom
                ov_bucket = torch.round(overlap.double() / q_step)
                m1 = ov_bucket == ov_bucket.min()
            else:
                m1 = torch.ones(cand_idx.numel(), dtype=torch.bool, device=Y.device)
            sub_pos = pos_avail[m1]
            rl_sub = row_load[tie_idx[sub_pos]]
            pick_slot = int(sub_pos[rl_sub == rl_sub.min()][0])
            pick_row = int(tie_idx[pick_slot])
            avail[pick_slot] = False
            chosen_mask[pick_row] = True
            val_pick = col_d[pick_row]
            cur_vec[pick_row] = val_pick
            cur_norm_sq += float(val_pick * val_pick)
            row_load[pick_row] += 1
            if U is not None:
                g = g + val_pick * U[pick_row]

        out_j = torch.where(chosen_mask, col, torch.zeros_like(col))
        cols_out[j] = out_j
        out_d = out_j.detach()
        if float(out_d.abs().max()) > 0.0:
            committed.append(out_d / torch.linalg.vector_norm(out_d).clamp_min(tiny))

    return torch.stack(cols_out, dim=1)


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
        lam = lam.clamp_min(eps)
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
    U = polar_factor(Y, eps=eps)
    c = (U * Y).sum((-2, -1)) / k
    return c.reshape(*c.shape, 1, 1) * U
