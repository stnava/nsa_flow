"""
Figure-4 style sweeps for every NSA-Flow variant that produces a basis.

One figure per variant: a reference panel plus the same five w values, with the
metrics each variant is judged on in the titles, and a final summary figure
comparing them.

Every panel is an INDEPENDENT solve. An earlier version warm-started each w
from the previous solution to make consecutive panels look continuous; that
overrides init="relax", the penalty homotopy from signed PCA, and for the
signed lifting it is destructive -- effective rank 1.79 of 20 with lobe overlap
20.4, against 9.5 and exactly 0.00 from a cold start at the same w. The
relaxation path is what makes the solve good, so it is not something to skip
for the sake of a smooth-looking figure.

Variants
  nsa_flow                     anchored refinement of a non-negative target
  nsa_flow (signed target)     the same, on a signed target -- the WRONG tool,
                               included because the failure is instructive
  nsa_flow_data                data-anchored non-negative fit
  nsa_flow_signed              signed lifting V = V+ - V-, real signed data
  nsa_flow_signed consolidated the same with supports consolidated
"""
import warnings, numpy as np, torch
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import linear_sum_assignment

import nsa_flow
from pysimlr.utils import orthogonality_defect, angle_defect

W_SEQ = [0.1, 0.25, 0.5, 0.75, 0.9]
OUT = "/Users/stnava/data/repos/nsa_flow/paper"
P, K, OVERLAP = 150, 10, 8
BLK = P // K
CORR = 0.35

# ---------------------------------------------------------------- fixtures
rng = np.random.default_rng(7)
Sig = np.full((K, K), CORR); np.fill_diagonal(Sig, 1.0)
span = lambda j: (max(0, j*BLK - OVERLAP), min(P, (j+1)*BLK + OVERLAP))

Vtrue = np.zeros((P, K))
for j in range(K):
    lo, hi = span(j); Vtrue[lo:hi, j] = rng.uniform(0.3, 1.5, hi-lo)
Vtrue /= np.linalg.norm(Vtrue, axis=0, keepdims=True)
VTRUE = torch.tensor(Vtrue, dtype=torch.float64)

TARGET_POS = torch.tensor(np.clip(Vtrue + 0.30*np.abs(rng.standard_normal((P, K))), 0, None),
                          dtype=torch.float64)
Upos = np.abs(rng.multivariate_normal(np.zeros(K), Sig, size=300))
X_POS = torch.tensor(Upos @ Vtrue.T + 0.15*np.abs(rng.standard_normal((300, P))),
                     dtype=torch.float64)

TARGET_SGN = torch.tensor(Vtrue + 0.9*rng.standard_normal((P, K)), dtype=torch.float64)

from sklearn.datasets import load_digits
_d = load_digits()
_keep = _d.data.std(0) > 1e-6
X_SGN = torch.tensor((_d.data - _d.data.mean(0))[:, _keep], dtype=torch.float64)
P_SGN, K_SGN = X_SGN.shape[1], 10
_, _, _Vh = torch.linalg.svd(X_SGN - X_SGN.mean(0, keepdim=True), full_matrices=False)
PCA_SGN = _Vh[:K_SGN].t().contiguous()

ORDER_TRUE = sorted(range(P), key=lambda i: (int(np.argmax(np.abs(Vtrue[i]))), -Vtrue[i].max()))

def frac_zero(Y, rel=1e-6):
    return float((Y.abs() <= rel*Y.abs().max()).float().mean())

def stats(Y, r=None):
    m = dict(D=float(orthogonality_defect(Y)), C=float(angle_defect(Y)),
             spar=frac_zero(Y),
             dead=int((Y.abs().sum(0) <= 1e-10*Y.abs().max()).sum()))
    m["er"] = float(r["effective_rank"]) if r is not None and "effective_rank" in r else float("nan")
    m["lobe"] = float(r["lobe_overlap"]) if r is not None and "lobe_overlap" in r else float("nan")
    return m

def match(V, ref, flip):
    a = V/(V.norm(dim=0, keepdim=True)+1e-12); b = ref/(ref.norm(dim=0, keepdim=True)+1e-12)
    corr = b.t() @ a
    rows, cols = linear_sum_assignment(-corr.abs().numpy())
    out = V[:, cols].clone()
    if flip:
        sg = torch.sign(corr[rows, cols]); sg[sg == 0] = 1.0
        out = out * sg.unsqueeze(0)
    return out

VARIANTS = [
  dict(key="nsa_flow", label="nsa_flow -- anchored refinement, NON-NEGATIVE target",
       ref=("target", TARGET_POS), order=ORDER_TRUE, signed=False, ylab="feature (true block order)",
       run=lambda w: nsa_flow.nsa_flow(TARGET_POS.clone(), w=w, nonneg=True, max_iter=5000)),
  dict(key="nsa_flow_signed_target",
       label="nsa_flow on a SIGNED target -- the wrong tool (falls back to subspace fidelity)",
       ref=("signed target", TARGET_SGN), order=ORDER_TRUE, signed=True, ylab="feature (true block order)",
       run=lambda w: nsa_flow.nsa_flow(TARGET_SGN.clone(), w=w, nonneg=True, max_iter=5000)),
  dict(key="nsa_flow_data", label="nsa_flow_data -- data-anchored, NON-NEGATIVE data",
       ref=("true V", VTRUE), order=ORDER_TRUE, signed=False, ylab="feature (true block order)",
       run=lambda w: nsa_flow.nsa_flow_data(X_POS, k=K, w=w, max_iter=5000)),
  dict(key="nsa_flow_data_signed",
       label="nsa_flow_data on REAL SIGNED data -- the canonical case (V >= 0, reconstruction fidelity)",
       ref=("signed PCA", PCA_SGN), order=list(range(P_SGN)), signed=False, ylab="pixel (raster order)",
       run=lambda w: nsa_flow.nsa_flow_data(X_SGN, k=K_SGN, w=w, max_iter=5000)),
  dict(key="nsa_flow_signed", label="nsa_flow_signed -- signed lifting, REAL signed data (digits)",
       ref=("signed PCA", PCA_SGN), order=list(range(P_SGN)), signed=True, ylab="pixel (raster order)",
       run=lambda w: nsa_flow.nsa_flow_signed(X_SGN, k=K_SGN, w=w, max_iter=5000)),
  dict(key="nsa_flow_signed_consolidated",
       label="nsa_flow_signed, consolidate=True -- exactly disjoint lobe supports",
       ref=("signed PCA", PCA_SGN), order=list(range(P_SGN)), signed=True, ylab="pixel (raster order)",
       run=lambda w: nsa_flow.nsa_flow_signed(X_SGN, k=K_SGN, w=w, max_iter=5000, consolidate=True)),
]

summary = {}
for v in VARIANTS:
    reflabel, ref = v["ref"]
    panels = [(reflabel, ref, stats(ref))]
    print(f"\n=== {v['label']}")
    print(f"{'w':>7} {'D':>8} {'C':>8} {'zeros':>7} {'dead':>5} {'eff.rank':>9} {'lobe ovl':>10}")
    r0 = panels[0][2]
    print(f"{reflabel[:7]:>7} {r0['D']:8.4f} {r0['C']:8.4f} {r0['spar']:7.3f} {r0['dead']:5d}")
    rows = []
    for w in W_SEQ:
        r = v["run"](w); V = r["Y"]; m = stats(V, r); m["w"] = w
        rows.append(m)
        panels.append((f"w = {w}", match(V, ref, v["signed"]), m))
        print(f"{w:7.2f} {m['D']:8.4f} {m['C']:8.4f} {m['spar']:7.3f} {m['dead']:5d} "
              f"{m['er']:9.2f} {m['lobe']:10.2e}")
    summary[v["label"]] = rows

    lim = max(float(torch.quantile(Y.abs().flatten(), 0.995)) for _, Y, _ in panels)
    cmap = "RdBu_r" if v["signed"] else "YlGnBu_r"
    fig = plt.figure(figsize=(14, 8.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.22)
    for i, (label, Y, m) in enumerate(panels):
        ax = fig.add_subplot(gs[i//3, i%3])
        kw = dict(vmin=-lim, vmax=lim) if v["signed"] else dict(vmin=0.0, vmax=lim)
        im = ax.imshow(Y.numpy()[v["order"], :], aspect="auto", cmap=cmap,
                       interpolation="nearest", **kw)
        sub = f"D={m['D']:.4f}  zeros={m['spar']:.3f}  dead={m['dead']}"
        if not np.isnan(m["er"]):   sub += f"\neff.rank={m['er']:.1f}"
        if not np.isnan(m["lobe"]): sub += f"  lobe ovl={m['lobe']:.1e}"
        ax.set_title(f"{label}\n{sub}", fontsize=8.5)
        ax.set_xlabel("component", fontsize=8); ax.set_ylabel(v["ylab"], fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle(f"{v['label']}\nindependent solve per panel (no warm start); "
                 "components matched to the reference for display only", fontsize=11)
    path = f"{OUT}/figure4_variant_{v['key']}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {path}")

# ------------------------------------------------------------ summary figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for label, rows in summary.items():
    w = [r["w"] for r in rows]
    short = label.split(" -- ")[0]
    axes[0].plot(w, [r["D"] for r in rows], marker="o", label=short)
    axes[1].plot(w, [r["spar"] for r in rows], marker="o", label=short)
    axes[2].plot(w, [r["er"] for r in rows], marker="o", label=short)
for ax, t, yl in zip(axes, ["orthogonality defect D", "sparsity (fraction of zeros)",
                            "effective rank"], ["D", "zeros", "eff. rank"]):
    ax.set_title(t, fontsize=10); ax.set_xlabel("w"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
axes[0].set_yscale("log")
axes[0].legend(fontsize=7.5, loc="best")
fig.suptitle("NSA-Flow variants across w  (each point an independent solve)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/figure4_variants_summary.png", dpi=150, bbox_inches="tight")
print(f"\nwrote {OUT}/figure4_variants_summary.png")
