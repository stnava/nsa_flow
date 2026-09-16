r"""Fast exploration harness: matrix-free data-anchored NSA-Flow + small datasets.

For p >> n, forming S = X'X is wasteful (400 MB at p=7129) and unnecessary:
every quantity needed is available from X V (n x k) and X'(X V) (p x k), so cost
is O(n p k) per iteration instead of O(p^2 k).  Verified against the S-based
implementation in ``test_fastlab_matches_library``.
"""
import numpy as np
import torch

F64 = torch.float64
_EPS = 1e-12


# --------------------------------------------------------------------------
# matrix-free reconstruction fidelity
# --------------------------------------------------------------------------
def _fid_and_grad(V, X, c, want_grad=True):
    XV = X @ V                                  # n x k
    A = XV.transpose(-2, -1) @ XV               # k x k  (= V' X'X V)
    B = V.transpose(-2, -1) @ V                 # k x k
    f = (c - 2.0 * A.diagonal().sum() + (A * B.transpose(-2, -1)).sum()) / c
    if not want_grad:
        return f, None
    SV = X.transpose(-2, -1) @ XV               # p x k  (= X'X V)
    g = (2.0 / c) * (-2.0 * SV + SV @ B + V @ A)
    return f, g


def _angle(V):
    k = V.shape[-1]
    if k == 1:
        return torch.zeros((), dtype=V.dtype)
    U = V / V.norm(dim=-2, keepdim=True).clamp_min(_EPS)
    M = U.transpose(-2, -1) @ U
    return (M - torch.eye(k, dtype=V.dtype)).pow(2).sum() / (k * (k - 1))


def _grad_angle(V):
    k = V.shape[-1]
    if k == 1:
        return torch.zeros_like(V)
    nrm = V.norm(dim=-2, keepdim=True).clamp_min(_EPS)
    U = V / nrm
    M = U.transpose(-2, -1) @ U
    dU = (4.0 / (k * (k - 1))) * (U @ (M - torch.eye(k, dtype=V.dtype)))
    return (dU - (U * dU).sum(dim=-2, keepdim=True) * U) / nrm


def fit_basis(X, k, w=0.9, max_iter=400, tol=1e-8, nonneg=True,
              mus=(0.0, 1e-2, 1.0, 1e2), V0=None):
    """Data-anchored basis, matrix-free, with a short negativity homotopy.

    ``V0`` starts the homotopy somewhere other than the signed PCA basis, so the
    effect of the starting point can be separated from the effect of running the
    homotopy at all.
    """
    X = torch.as_tensor(np.asarray(X), dtype=F64)
    c = X.pow(2).sum()
    n, p = X.shape
    # mu = 0 optimum: leading right singular vectors (signed)
    if V0 is None:
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh[:k].transpose(0, 1).clone()
    else:
        V = torch.as_tensor(np.asarray(V0), dtype=F64).clone()

    def run(mu, hard, iters):
        nonlocal V
        def obj(Vv):
            f, _ = _fid_and_grad(Vv, X, c, want_grad=False)
            e = (1.0 - w) * f + w * _angle(Vv)
            if mu:
                e = e + mu * Vv.clamp_max(0.0).pow(2).sum()
            return float(e)

        def grd(Vv):
            _, gf = _fid_and_grad(Vv, X, c)
            g = (1.0 - w) * gf + w * _grad_angle(Vv)
            if mu:
                g = g + 2.0 * mu * Vv.clamp_max(0.0)
            return g

        proj = (lambda A: A.clamp_min(0.0)) if hard else (lambda A: A)
        V = proj(V)
        E, g = obj(V), grd(V)
        t = 1.0 / max(float(g.norm()), _EPS)
        Vp = gp = None
        for _ in range(iters):
            if Vp is not None:
                s_, r_ = V - Vp, g - gp
                sr = float((s_ * r_).sum())
                t = min(max(float((s_ * s_).sum()) / sr if sr > 0 else 1e12, 1e-12), 1e12)
            ok = False
            for _ in range(40):
                Vn = proj(V - t * g)
                d2 = float((Vn - V).pow(2).sum())
                if obj(Vn) <= E - 1e-4 * d2 / t:
                    ok = True
                    break
                t *= 0.5
            if not ok:
                break
            Vp, gp = V, g
            V = Vn
            E, g = obj(V), grd(V)
            if (d2 ** 0.5) / t <= tol:
                break

    if nonneg:
        for mu in mus:
            run(mu, False, 120)
        run(0.0, True, max_iter)
    else:
        run(0.0, False, max_iter)
    return V.numpy()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def overlap(L, tol=1e-8):
    s = np.abs(L) > tol
    live = s.any(1)
    return float((s.sum(1)[live] - 1).mean()) if live.any() else 0.0


def sparsity(L, tol=1e-8):
    return float((np.abs(L) <= tol).mean())


def recon_err(L, X):
    X = np.asarray(X)
    return float(np.linalg.norm(X - X @ L @ L.T) / np.linalg.norm(X))


# --------------------------------------------------------------------------
# Findings from the w=0.5 vs w=0.9 exploration, recorded so they are not lost
# --------------------------------------------------------------------------
NOTES = """
1. The homotopy SCHEDULE, not the iteration count, selects the local minimum.
   ADNI basis reproducibility: 3-stage mu path gives 0.708 at 150 iterations and
   0.725 at 3000; a 9-stage path gives 0.975.  Refining the path is worth 0.25,
   converging 20x harder on a coarse path is worth 0.02.  Any result computed
   with a coarse schedule is therefore suspect -- this invalidated one of our own
   cross-dataset runs mid-exploration.

2. Moderate w DOMINATES aggressive w on both axes simultaneously.  Going from
   w=0.9 to w=0.5 improved accuracy AND reproducibility nearly everywhere:
   ADNI stability -0.053 -> +0.038 vs PCA, Golub p=20 accuracy -0.097 -> +0.039,
   faces (standardised) -0.014 -> +0.019.  At w=0.9 the solution is pinned near a
   combinatorial vertex (overlap ~0.35) and small data changes flip which vertex;
   at w=0.5 (overlap ~0.8-1.2) it sits in a smooth interior region, still far
   sparser than PCA.  Every number we previously reported at w=0.9 understated
   the method.

3. Prediction: no consistent win over PCA, at either w.  At w=0.5, 6 of 11
   dataset/dimension cells are non-negative and 5 are negative.  The clear wins
   are the two extremes of Golub's p sweep (p/n=99, where PCA is inconsistent,
   and p=20, where PCA has too few features) and faces standardised.  The
   p/n hypothesis is NOT supported in between: the middle of the sweep is a flat
   tie.

4. Reproducibility: 4 wins, 2 losses at w=0.5 (adni +0.038, diabetes +0.019,
   breast_cancer +0.012, digits +0.008; faces -0.102, wine -0.120).  Real but
   dataset-specific, not the structural property we had claimed.

5. Unexplained losses: faces raw non-negative (-0.094 accuracy) and wine
   (-0.120 stability).  Opposite extremes of p (4096 vs 13), so not a dimension
   story.  20news tf-idf also loses consistently (-0.022) despite being the
   nominally ideal case -- non-negative, sparse, high-dimensional.

6. Cost tension: the 9-stage homotopy that buys reproducibility is ~9x the work.
   A p=4000 cross-dataset sweep went from 59s to 185s when the schedule was
   refined.  Speed and the stability property are in direct conflict.
"""
