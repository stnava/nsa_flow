"""Generate publication-quality figures for the new public benchmark datasets.

Generates:
1. paper/figs/fig_new_public_wsweep.pdf (and .png)
   - Panel A: Tecator NIR Spectroscopy (R^2 and Defect vs w)
   - Panel B: Sonar Acoustic Chirps (ROC-AUC and Balanced Acc vs w)
   - Panel C: Prostate Cancer Transcriptomics (Balanced Acc and Disjoint Sparsity vs w)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})

def generate_figure(json_path="paper/results/new_public_data_sweeps.json",
                    out_pdf="paper/figs/fig_new_public_wsweep.pdf",
                    out_png="paper/figs/fig_new_public_wsweep.png"):
    if not os.path.exists(json_path):
        print(f"File {json_path} does not exist yet.")
        return False

    with open(json_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # ----------------------------------------------------
    # Panel A: Tecator NIR Spectroscopy
    # ----------------------------------------------------
    ax = axes[0]
    tec = data["tecator_sweep"]
    w_vals = sorted(list(set(r["w"] for r in tec)))
    
    # Non-negative
    nonneg_r2 = [r["r2_mean"] for r in tec if r["variant"] == "Non-negative (V >= 0)"]
    nonneg_std = [r["r2_std"] for r in tec if r["variant"] == "Non-negative (V >= 0)"]
    
    # Signed Relaxed
    signed_r2 = [r["r2_mean"] for r in tec if r["variant"] == "Signed Relaxed"]
    
    # Signed Consolidated
    consol_r2 = [r["r2_mean"] for r in tec if r["variant"] == "Signed Consolidated"]
    
    ax.plot(w_vals, nonneg_r2, "o-", color="#10b981", label="Non-negative ($V \\geq 0$)")
    ax.fill_between(w_vals, np.array(nonneg_r2) - np.array(nonneg_std), np.array(nonneg_r2) + np.array(nonneg_std),
                    color="#10b981", alpha=0.15)
    ax.plot(w_vals, signed_r2, "s--", color="#3b82f6", label="Signed Relaxed")
    ax.plot(w_vals, consol_r2, "d-", color="#8b5cf6", label="Signed Consolidated")
    
    # Baseline PCA & NMF horizontal lines
    ax.axhline(0.9109, color="#64748b", linestyle=":", label="Dense PCA (0.9109)")
    ax.axhline(0.3858, color="#ef4444", linestyle=":", label="NMF Collinear (0.3858)")

    ax.set_title("(a) Tecator NIR Spectrometry ($n=240, p=100$)")
    ax.set_xlabel("Trade-off parameter $w$")
    ax.set_ylabel("Linear Ridge $R^2$")
    ax.set_ylim([0.30, 0.95])
    ax.legend(loc="lower left", framealpha=0.9)

    # ----------------------------------------------------
    # Panel B: Sonar Acoustic Chirps
    # ----------------------------------------------------
    ax = axes[1]
    son = data["sonar_sweep"]
    w_son = sorted(list(set(r["w"] for r in son)))
    
    son_rel_auc = [r["auc_mean"] for r in son if r["variant"] == "Signed Relaxed"]
    son_rel_std = [r["auc_std"] for r in son if r["variant"] == "Signed Relaxed"]
    son_con_auc = [r["auc_mean"] for r in son if r["variant"] == "Signed Consolidated"]
    son_con_std = [r["auc_std"] for r in son if r["variant"] == "Signed Consolidated"]

    ax.plot(w_son, son_rel_auc, "s-", color="#3b82f6", label="Signed Relaxed AUC")
    ax.fill_between(w_son, np.array(son_rel_auc) - np.array(son_rel_std), np.array(son_rel_auc) + np.array(son_rel_std),
                    color="#3b82f6", alpha=0.15)
    ax.plot(w_son, son_con_auc, "d-", color="#8b5cf6", label="Signed Consolidated AUC")
    ax.fill_between(w_son, np.array(son_con_auc) - np.array(son_con_std), np.array(son_con_auc) + np.array(son_con_std),
                    color="#8b5cf6", alpha=0.15)

    ax.axhline(0.8185, color="#64748b", linestyle=":", label="Dense PCA (0.8185)")
    ax.axhline(0.8022, color="#ef4444", linestyle=":", label="NMF (0.8022)")

    ax.set_title("(b) Sonar Acoustic Returns ($n=208, p=60$)")
    ax.set_xlabel("Trade-off parameter $w$")
    ax.set_ylabel("Linear ROC-AUC")
    ax.set_ylim([0.75, 0.85])
    ax.legend(loc="lower left", framealpha=0.9)

    # ----------------------------------------------------
    # Panel C: Prostate Cancer Transcriptomics
    # ----------------------------------------------------
    ax = axes[2]
    pros = data["prostate_sweep"]
    w_pros = sorted(list(set(r["w"] for r in pros)))

    pros_rel_bacc = [r["bacc_mean"] for r in pros if r["variant"] == "Signed Relaxed"]
    pros_con_bacc = [r["bacc_mean"] for r in pros if r["variant"] == "Signed Consolidated"]
    pros_con_std = [r["bacc_std"] for r in pros if r["variant"] == "Signed Consolidated"]

    ax.plot(w_pros, pros_con_bacc, "d-", color="#8b5cf6", linewidth=2.5, label="Consolidated Balanced Acc")
    ax.fill_between(w_pros, np.array(pros_con_bacc) - np.array(pros_con_std),
                    np.array(pros_con_bacc) + np.array(pros_con_std), color="#8b5cf6", alpha=0.15)
    ax.plot(w_pros, pros_rel_bacc, "s--", color="#3b82f6", label="Relaxed Balanced Acc")

    ax.axhline(0.8045, color="#64748b", linestyle=":", label="Dense PCA (0.8045)")

    # Secondary y-axis for disjoint sparsity
    ax2 = ax.twinx()
    pros_con_spar = [r["sparsity_mean"] * 100 for r in pros if r["variant"] == "Signed Consolidated"]
    ax2.plot(w_pros, pros_con_spar, "^:", color="#f59e0b", linewidth=2.0, label="Disjoint Sparsity %")
    ax2.set_ylabel("Exact Disjoint Sparsity (%)", color="#f59e0b")
    ax2.tick_params(axis="y", labelcolor="#f59e0b")
    ax2.set_ylim([0, 100])
    ax2.grid(False)

    ax.set_title("(c) Prostate Transcriptomics ($n=102, p=12,600$)")
    ax.set_xlabel("Trade-off parameter $w$")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim([0.74, 0.88])
    ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated {out_pdf} and {out_png}")
    return True

if __name__ == "__main__":
    generate_figure()
