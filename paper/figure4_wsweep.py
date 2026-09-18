"""Figure 4 style: w-sweep response plots for all canonical NSA-Flow variants.

Shows fidelity, orthogonality defect, sparsity (and lobe overlap for signed)
as w increases from 0 to 1.  Runs on two datasets:
  - Golub 3-class (n=72, p=2000, k=3) — gene expression, mixed signed data
  - UCI diabetes (n=442, p=10, k=3) — small structured data

Outputs: paper/figs/fig4_wsweep_{dataset}.png

Usage:
    PYTHONPATH=. python paper/figure4_wsweep.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from nsa_flow.reconstruct import nsa_flow_data
from nsa_flow.signed import nsa_flow_signed

FIGS = Path("paper/figs")
FIGS.mkdir(exist_ok=True)

WS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
K  = 3

# ── colour scheme ─────────────────────────────────────────────────────────────
C_DATA   = "#2166ac"   # blue — nsa_flow_data (non-negative)
C_SIGNED = "#d6604d"   # red  — nsa_flow_signed
C_CONSOL = "#4d9221"   # green — signed + consolidated

# ── dataset loaders ───────────────────────────────────────────────────────────
def load_golub():
    from experiments.data import load_golub3
    X_raw, _, _ = load_golub3()
    Xl = np.log2(np.clip(X_raw, 1.0, None))
    top = np.argsort(Xl.var(0))[-2000:]
    Xs = Xl[:, top]; Xs = (Xs - Xs.mean(0)) / (Xs.std(0, ddof=1) + 1e-12)
    return torch.as_tensor(Xs, dtype=torch.float64), "Golub (n=72, p=2000, k=3)"

def load_diabetes():
    from sklearn.datasets import load_diabetes
    from sklearn.preprocessing import StandardScaler
    d = load_diabetes()
    Z = StandardScaler().fit_transform(d.data)
    return torch.as_tensor(Z, dtype=torch.float64), "UCI Diabetes (n=442, p=10, k=3)"

DATASETS = [("golub", load_golub), ("diabetes", load_diabetes)]


def sweep(T, k, ws):
    """Run all three variants over ws. Returns dict of lists.

    The signed variants use warm-starting (continuation): each w is
    initialized from the previous w's parts tensor ``W = [Vp|Vm]``.
    This traces a continuous path through the landscape and prevents
    basin-switching between adjacent w values.  ``nsa_flow_data`` uses
    the default init="clamp" (independent for each w) since it is
    fast and has a well-behaved landscape.
    """
    out = dict(
        data_fid=[], data_defect=[], data_sp=[],
        signed_fid=[], signed_defect=[], signed_sp=[], signed_ov=[],
        consol_fid=[], consol_defect=[], consol_sp=[], consol_ov=[],
        signed_stop=[], consol_stop=[], data_stop=[],
    )
    W_signed_prev = None   # warm-start state for signed relaxed
    W_consol_prev = None   # warm-start state for signed+consol

    for w in ws:
        print(f"  w={w:.2f} ... ", end="", flush=True)

        # nsa_flow_data (fresh init each w — fast, well-behaved landscape)
        r = nsa_flow_data(T, k=k, w=w)
        V = r.Y.numpy()
        out["data_fid"].append(float(r.fidelity))
        out["data_defect"].append(float(r.defect))
        out["data_sp"].append(float((np.abs(V) < 1e-10).mean()))
        out["data_stop"].append(r.stop_reason)

        # nsa_flow_signed — warm-start from previous w (continuation)
        init_s = W_signed_prev if W_signed_prev is not None else "relax"
        r = nsa_flow_signed(T, k=k, w=w, consolidate=False, init=init_s)
        W_signed_prev = r["parts"].clone()    # [p, 2k] parts for next w
        V = r.Y.numpy()
        out["signed_fid"].append(float(r.fidelity))
        out["signed_defect"].append(float(r.defect))
        out["signed_sp"].append(float((np.abs(V) < 1e-10).mean()))
        out["signed_ov"].append(float(r.lobe_overlap))
        out["signed_stop"].append(r.stop_reason)

        # nsa_flow_signed + consolidate — warm-start similarly
        init_c = W_consol_prev if W_consol_prev is not None else "relax"
        r = nsa_flow_signed(T, k=k, w=w, consolidate=True, init=init_c)
        W_consol_prev = r["parts"].clone()
        V = r.Y.numpy()
        out["consol_fid"].append(float(r.fidelity))
        out["consol_defect"].append(float(r.defect))
        out["consol_sp"].append(float((np.abs(V) < 1e-10).mean()))
        out["consol_ov"].append(float(r.lobe_overlap))
        out["consol_stop"].append(r.stop_reason)
        print("done")
    return out


def plot_sweep(ws, out, title, outpath):
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"NSA-Flow variant w-sweep — {title}", fontsize=13, y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_fid  = fig.add_subplot(gs[0, 0])
    ax_def  = fig.add_subplot(gs[0, 1])
    ax_sp   = fig.add_subplot(gs[0, 2])
    ax_ov   = fig.add_subplot(gs[1, 0])
    ax_ileg = fig.add_subplot(gs[1, 1])  # legend panel
    ax_note = fig.add_subplot(gs[1, 2])  # stop reasons

    # ── fidelity ──────────────────────────────────────────────────────────────
    ax_fid.plot(ws, out["data_fid"],   "o-", color=C_DATA,   label="NSA-Flow data")
    ax_fid.plot(ws, out["signed_fid"], "s-", color=C_SIGNED, label="Signed")
    ax_fid.plot(ws, out["consol_fid"], "^-", color=C_CONSOL, label="Signed+consol")
    ax_fid.set_xlabel("w"); ax_fid.set_ylabel("Reconstruction fidelity")
    ax_fid.set_title("Fidelity (↓ better)")
    ax_fid.set_xlim(-0.02, 1.02)

    # ── orth defect ───────────────────────────────────────────────────────────
    ax_def.plot(ws, out["data_defect"],   "o-", color=C_DATA)
    ax_def.plot(ws, out["signed_defect"], "s-", color=C_SIGNED)
    ax_def.plot(ws, out["consol_defect"], "^-", color=C_CONSOL)
    ax_def.set_xlabel("w"); ax_def.set_ylabel("Orth defect")
    ax_def.set_title("Orth defect (↓ better)")
    ax_def.set_xlim(-0.02, 1.02)

    # ── sparsity ──────────────────────────────────────────────────────────────
    ax_sp.plot(ws, out["data_sp"],   "o-", color=C_DATA)
    ax_sp.plot(ws, out["signed_sp"], "s-", color=C_SIGNED)
    ax_sp.plot(ws, out["consol_sp"], "^-", color=C_CONSOL)
    ax_sp.set_xlabel("w"); ax_sp.set_ylabel("Sparsity (fraction zeros)")
    ax_sp.set_title("Sparsity (↑ better for interpretability)")
    ax_sp.set_xlim(-0.02, 1.02); ax_sp.set_ylim(-0.02, 1.02)

    # ── lobe overlap (signed variants only) ───────────────────────────────────
    ax_ov.plot(ws, out["signed_ov"], "s-", color=C_SIGNED, label="Signed")
    ax_ov.plot(ws, out["consol_ov"], "^-", color=C_CONSOL, label="Signed+consol")
    ax_ov.set_xlabel("w"); ax_ov.set_ylabel("Lobe overlap ‖Vp∘Vm‖₁")
    ax_ov.set_title("Lobe overlap (↓ = more disjoint lobes)")
    ax_ov.set_xlim(-0.02, 1.02)
    ax_ov.legend(fontsize=8)

    # ── legend panel ──────────────────────────────────────────────────────────
    ax_ileg.axis("off")
    handles = [
        plt.Line2D([0],[0], marker="o", color=C_DATA,   linestyle="-", label="NSA-Flow data\n(V≥0, reconstruction)"),
        plt.Line2D([0],[0], marker="s", color=C_SIGNED, linestyle="-", label="Signed\n(V=Vp-Vm, relaxed)"),
        plt.Line2D([0],[0], marker="^", color=C_CONSOL, linestyle="-", label="Signed+consol\n(hard disjoint support)"),
    ]
    ax_ileg.legend(handles=handles, loc="center", fontsize=9, frameon=True)
    ax_ileg.set_title("Variants", fontsize=10)

    # ── stop reasons table ────────────────────────────────────────────────────
    ax_note.axis("off")
    stops = [["w", "data", "signed", "consol"]]
    for i, w in enumerate(ws):
        stops.append([f"{w:.2f}",
                      out["data_stop"][i][:4],
                      out["signed_stop"][i][:4],
                      out["consol_stop"][i][:4]])
    tbl = ax_note.table(cellText=stops[1:], colLabels=stops[0],
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    tbl.scale(1, 1.1)
    ax_note.set_title("Stop reasons", fontsize=10)

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


for ds_name, loader in DATASETS:
    print(f"\n=== {ds_name} ===")
    T, title = loader()
    print(f"  Shape: {tuple(T.shape)}")
    out = sweep(T, K, WS)
    plot_sweep(WS, out, title, FIGS / f"fig4_wsweep_{ds_name}.png")

print("\nAll figures saved.")
