"""
Figure 4 in two versions, each solved by the method built for its input.

(a) NON-NEGATIVE data  -> nsa_flow_data: V >= 0, near-orthogonal. Non-negative
    columns with disjoint supports are the natural object here, so a sequential
    colour scale and a plain heatmap say everything.

(b) SIGNED data -> nsa_flow_signed: V = V+ - V-, both parts non-negative, the
    orthogonality term acting on the 2k parts. Running the plain non-negative
    solver on a signed input is the wrong tool -- a non-negative column cannot
    encode a contrast -- and it shows: 7 of 40 components die at w=0.1 and the
    effective rank falls to 3.6, because the sign-blind subspace fidelity the
    solver falls back to is rank-indifferent and a low w cannot hold components
    up. The lifting restores the representational capacity, so a diverging
    scale is the honest display: each component reads as "these features minus
    those".

Presentation choices shared by both, and why:
  * warm-started along w, so consecutive panels are deformations of one another
    rather than independent solves in unrelated basins;
  * one row order for every panel, taken from the GROUND-TRUTH block structure
    rather than from any fit, so no panel is flattered by being sorted with its
    own answer.
"""
import warnings, numpy as np, torch
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import nsa_flow
from pysimlr.utils import orthogonality_defect, angle_defect

N, P, K = 300, 150, 10
CORR, OVERLAP = 0.35, 8   # correlated latents, overlapping supports
W_SEQ = [0.1, 0.25, 0.5, 0.75, 0.9]
BLK = P // K                                   # 15 features per component
OUT = "/Users/stnava/data/repos/nsa_flow/paper"

def frac_zero(Y, rel=1e-6):
    return float((Y.abs() <= rel * Y.abs().max()).float().mean())

def stats(Y):
    dead = int((Y.abs().sum(0) <= 1e-10 * Y.abs().max()).sum())
    return dict(D=float(orthogonality_defect(Y)), C=float(angle_defect(Y)),
                spar=frac_zero(Y), dead=dead)

# ----------------------------------------------------------------- (a) data
rng = np.random.default_rng(7)

# Correlated latents, as in the published chunk (corrval = 0.35). Without the
# correlation the fidelity and orthogonality terms do not compete and every w
# returns the same answer -- D = 0 at w = 0.1 already, so the sweep shows
# nothing. The overlap does the same job on the feature side: supports that
# already partition give the solver nothing to concentrate.
Sig = np.full((K, K), CORR); np.fill_diagonal(Sig, 1.0)

def overlapping_support(j):
    lo = max(0, j*BLK - OVERLAP); hi = min(P, (j+1)*BLK + OVERLAP)
    return lo, hi

Vpos = np.zeros((P, K))
for j in range(K):
    lo, hi = overlapping_support(j)
    Vpos[lo:hi, j] = rng.uniform(0.3, 1.5, hi-lo)
Vpos /= np.linalg.norm(Vpos, axis=0, keepdims=True)

# (a) stays in Figure 4's original framing: REFINE a noisy non-negative target.
# The data-anchored form was tried and is the wrong vehicle for this figure --
# there `X V V'` is an orthogonal projection exactly when `V'V = I`, so the
# reconstruction term already prefers orthonormal columns and D is 0.0016 at
# w = 0.1 before the orthogonality term does anything. Nothing left for w to
# show. The anchored form has the two terms genuinely competing.
TARGET_POS = torch.tensor(
    np.clip(Vpos + 0.30 * np.abs(rng.standard_normal((P, K))), 0, None),
    dtype=torch.float64)

# (b) REAL signed data. The earlier synthetic truth was a staircase -- contiguous
# runs of features, positive lobe then negative lobe, identical for every
# component -- so sorting rows by it produced clean bands that were an artefact
# of the construction rather than anything the method found. Real contrasts are
# between scattered feature sets.
#
# Digits is the right regime for this figure: p = 64 against the p = 66 of the
# ADNI cortical-thickness analysis the lifting was built for, n = 1797, and
# centring makes it genuinely signed. Pixel index is a meaningful order already
# (raster order over the 8x8 grid), so no row sorting is needed and no panel is
# ordered by its own answer.
from sklearn.datasets import load_digits
_dig = load_digits()
Xsgn = torch.tensor(_dig.data - _dig.data.mean(0), dtype=torch.float64)
N_SGN, P_SGN = Xsgn.shape
K_SGN = 10

# row order from the TRUTH: block, then positives before negatives by weight
order_pos = sorted(range(P), key=lambda i: (int(np.argmax(np.abs(Vpos[i]))), -Vpos[i].max()))
order_sgn = list(range(P_SGN))          # raster order, unsorted

def _signed_pca_reference():
    """Signed PCA of the digits data: the reference the lifting is measured
    against, standing in for the ground truth real data does not have."""
    Xc = Xsgn - Xsgn.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Vh[:K_SGN].t().contiguous()


def match_to_truth(V, truth, allow_sign_flip):
    """
    Permute (and for a signed basis, sign-align) the recovered components to
    best match the truth.

    Component order and overall column sign are gauge: the solver has no reason
    to return them in the truth's order. Left unmatched, each panel shows the
    right structure at the wrong coordinates, so the sweep does not visibly
    line up with the truth or with itself. This is a relabelling for display
    and changes no reported quantity -- every metric here is invariant to it.
    """
    from scipy.optimize import linear_sum_assignment
    a = V / (V.norm(dim=0, keepdim=True) + 1e-12)
    b = truth / (truth.norm(dim=0, keepdim=True) + 1e-12)
    corr = (b.t() @ a)
    rows, cols = linear_sum_assignment(-corr.abs().numpy())
    out = V[:, cols].clone()
    if allow_sign_flip:
        signs = torch.sign(corr[rows, cols])
        signs[signs == 0] = 1.0
        out = out * signs.unsqueeze(0)
    return out


def sweep(kind):
    panels, init = [], None
    for w in W_SEQ:
        if kind == "pos":
            kw = {"init": init} if init is not None else {}
            r = nsa_flow.nsa_flow(TARGET_POS.clone(), w=w, nonneg=True,
                                  max_iter=5000, **kw)
            init = r["Y"].clone()
        else:
            r = nsa_flow.nsa_flow_signed(Xsgn, k=K_SGN, w=w,
                                         init=init if init is not None else "relax",
                                         max_iter=5000)
            init = r["parts"].clone()
        panels.append((w, r["Y"], r))
    return panels

for kind, truth, order, cmap, title in [
        ("pos", TARGET_POS, order_pos, "YlGnBu_r",
         "(a) NON-NEGATIVE target, refined with nsa_flow  (V >= 0)"),
        ("sgn", _signed_pca_reference(), order_sgn, "RdBu_r",
         "(b) REAL signed data (digits, centred), solved with nsa_flow_signed  (V = V+ - V-)")]:
    panels = sweep(kind)
    print(f"\n=== {title}")
    hdr = f"{'w':>6} {'D':>8} {'C':>8} {'zeros':>7} {'dead':>5} {'eff.rank':>9}"
    if kind == "sgn": hdr += f" {'lobe ovl':>9}"
    print(hdr)
    tru = stats(truth)
    print(f"{'truth':>6} {tru['D']:8.4f} {tru['C']:8.4f} {tru['spar']:7.3f} {tru['dead']:5d} {'-':>9}")
    metas = []
    for w, V, r in panels:
        m = stats(V); m["w"] = w
        m["er"] = float(r.get("effective_rank", float("nan")))
        m["lobe"] = float(r.get("lobe_overlap", float("nan")))
        metas.append(m)
        line = (f"{w:6.2f} {m['D']:8.4f} {m['C']:8.4f} {m['spar']:7.3f} "
                f"{m['dead']:5d} {m['er']:9.2f}")
        if kind == "sgn": line += f" {m['lobe']:9.2e}"
        print(line)

    shown = [("target" if kind == "pos" else "signed PCA (reference)", truth, tru)] + [
        (f"w = {m['w']}", match_to_truth(V, truth, allow_sign_flip=(kind == "sgn")), m)
        for (w, V, r), m in zip(panels, metas)]
    lim = max(float(torch.quantile(V.abs().flatten(), 0.995)) for _, V, _ in shown)
    fig = plt.figure(figsize=(14, 8.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.20)
    for i, (label, V, m) in enumerate(shown):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        kw = dict(vmin=-lim, vmax=lim) if kind == "sgn" else dict(vmin=0.0, vmax=lim)
        im = ax.imshow(V.numpy()[order, :], aspect="auto", cmap=cmap,
                       interpolation="nearest", **kw)
        sub = f"D={m['D']:.4f}  zeros={m['spar']:.3f}  dead={m['dead']}"
        if not np.isnan(m.get("er", float("nan"))): sub += f"\neff.rank={m['er']:.1f}"
        if kind == "sgn" and not np.isnan(m.get("lobe", float("nan"))):
            sub += f"  lobe ovl={m['lobe']:.1e}"
        ax.set_title(f"{label}\n{sub}", fontsize=8.5)
        ax.set_xlabel("component", fontsize=8)
        ax.set_ylabel("pixel (raster order)" if kind == "sgn" else "feature (true block order)", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle(f"{title}\nn={N}, p={P}, k={K}; warm-started along w; rows ordered by "
                 "ground-truth blocks", fontsize=11)
    path = f"{OUT}/figure4_{'a_nonneg' if kind=='pos' else 'b_signed'}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
