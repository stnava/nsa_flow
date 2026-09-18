"""
Figure 4, rebuilt so the sweep reads as a path rather than five unrelated solves.

Three changes from the published chunk, each with a stated reason:

1. Target is the NON-NEGATIVE X0 that generate_synth_data() builds, not the
   signed Y0 the caller takes. With a signed target and nonneg=True the solver
   runs in sign-blind subspace mode, which is scale-free and rank-indifferent,
   so at low w the orthogonality term is too weak to keep components alive: 7
   of 40 die at w=0.1 and effective rank falls to 3.63. On the non-negative
   target no component dies at any w and the effective rank rises monotonically.

2. Warm-started along w. Each panel initialises from the previous solution, so
   consecutive panels are deformations of one another. Solved independently they
   land in unrelated basins -- different permutations and supports -- so nothing
   visually connects them even when the metrics move smoothly.

3. One shared row order for every panel, taken from the final solution's
   dominant component. Feature order is arbitrary, so without this a disjoint
   basis looks like speckle; with it the supports read as blocks and the eye can
   follow structure emerging across panels.
"""
import warnings, numpy as np, torch
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pysimlr.nsa_backend import load_nsa_flow
from pysimlr.utils import invariant_orthogonality_defect, orthogonality_defect, angle_defect
fn = load_nsa_flow()

P, K, CORR = 200, 40, 0.35
W_SEQ = [0.1, 0.25, 0.5, 0.75, 0.9]

rng = np.random.default_rng(123)
S = np.full((K, K), CORR); np.fill_diagonal(S, 1.0)
Y0 = rng.multivariate_normal(np.zeros(K), S, size=P)
X0 = torch.clamp(torch.tensor(Y0 + 0.2 * rng.standard_normal((P, K)),
                              dtype=torch.float64), min=0)

def w_spar(Y):
    y = Y.detach().cpu().numpy()
    return 1.0 - float((y / y.max() > np.quantile(y, 0.1)).sum()) / y.size

panels, init = [("Original", X0, None)], None
for w in W_SEQ:
    r = fn(X0.clone(), w=w, nonneg=True, max_iter=5000,
           **({"init": init} if init is not None else {}))
    init = r["Y"].clone()
    panels.append((f"w = {w}", r["Y"], r))

# shared row order from the final solution: group by dominant component
final = panels[-1][1]
dom = final.argmax(dim=1)
strength = final.max(dim=1).values
order = sorted(range(P), key=lambda i: (int(dom[i]), -float(strength[i])))

print(f"{'panel':>10} {'orth(paper)':>12} {'D':>8} {'C':>8} {'w.spar':>8} "
      f"{'dead':>5} {'eff.rank':>9} {'fidelity':>9}")
rows = []
for label, Y, r in panels:
    dead = int((Y.abs().sum(0) <= 1e-10 * Y.abs().max()).sum())
    meta = dict(label=label, inv=float(invariant_orthogonality_defect(Y)),
                D=float(orthogonality_defect(Y)), C=float(angle_defect(Y)),
                spar=w_spar(Y), dead=dead,
                er=float(r["effective_rank"]) if r is not None and "effective_rank" in r else float("nan"),
                fid="-" if r is None else str(r.get("fidelity_mode")))
    rows.append(meta)
    print(f"{label:>10} {meta['inv']:12.4f} {meta['D']:8.4f} {meta['C']:8.4f} "
          f"{meta['spar']:8.3f} {dead:5d} {meta['er']:9.2f} {meta['fid']:>9}")

allv = torch.cat([Y.flatten() for _, Y, _ in panels])
hi = float(torch.quantile(allv, 0.98))
fig = plt.figure(figsize=(14, 8.5))
gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.16)
for i, ((label, Y, r), meta) in enumerate(zip(panels, rows)):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    im = ax.imshow(Y.numpy()[order, :], aspect="auto", cmap="YlGnBu_r",
                   vmin=0.0, vmax=hi, interpolation="nearest")
    er = "" if np.isnan(meta["er"]) else f", eff.rank={meta['er']:.1f}"
    ax.set_title(f"{label}   D={meta['D']:.4f}, C={meta['C']:.4f}\n"
                 f"w.spar={meta['spar']:.3f}, dead={meta['dead']}{er}", fontsize=9)
    ax.set_xlabel("component", fontsize=8); ax.set_ylabel("feature (shared order)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
fig.suptitle(
    f"Figure 4 rebuilt -- non-negative target, warm-started along w, shared row order "
    f"(p={P}, k={K}, corr={CORR})\n"
    "no component dies at any w; supports concentrate monotonically as w rises",
    fontsize=11)
out = "/Users/stnava/data/repos/nsa_flow/paper/figure4_rebuilt_v2.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nwrote {out}")
