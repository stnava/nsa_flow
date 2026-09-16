"""Run every paper experiment, writing CSVs, LaTeX tables and figures.

    python experiments/run_all.py [--quick]

Outputs land in paper/results/ (CSV + .tex) and paper/figs/ (PDF).
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "paper" / "results"
FIGS = ROOT / "paper" / "figs"
for d in (RES, FIGS):
    d.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, GREY, RED = "#1f78b4", "#d9860a", "#33a02c", "#888888", "#c0392b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "legend.frameon": False,
})


def save(df, name):
    df.to_csv(RES / f"{name}.csv", index=False)
    print(f"    wrote results/{name}.csv  ({len(df)} rows)")
    return df


# ----------------------------------------------------------------- E0: functionals
def fig_functionals():
    """Why D and not the v1 defect: zero sets, blindness to imbalance, collapse."""
    import torch
    from nsa_flow import gram, stiefel_defect
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    def old(Y):
        S = gram(Y)
        return ((S * S).sum() - S.diagonal().pow(2).sum()) / Y.pow(2).sum() ** 2

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))

    # (a) orthogonal columns, one rescaled: v1 sees nothing, D sees it
    Q = torch.linalg.qr(torch.randn(60, 6))[0]
    rs = np.linspace(1, 8, 60)
    do, dn = [], []
    for r in rs:
        s = torch.ones(6, dtype=torch.float64)
        s[-1] = float(r)
        do.append(old(Q * s).item())
        dn.append(stiefel_defect(Q * s).item())
    ax[0].plot(rs, do, color=GREY, lw=2, label="v1 defect")
    ax[0].plot(rs, dn, color=BLUE, lw=2, label="$D$")
    ax[0].set_xlabel("norm of one column ($\\times$ others)")
    ax[0].set_ylabel("value")
    ax[0].set_title("(a) columns stay orthogonal", fontsize=9)
    ax[0].legend()

    # (b) rank collapse: v1 reaches zero, D respects 1/r - 1/k, attained exactly
    k = 6
    ranks = list(range(1, k + 1))
    obs_old, obs_new = [], []
    for r in ranks:
        # A matrix of rank r whose r non-zero Gram eigenvalues are equal attains
        # the bound: take Q (p x r) orthonormal times R (r x k) with R R' = I_r.
        Q = torch.linalg.qr(torch.randn(60, r))[0]
        R = torch.linalg.qr(torch.randn(k, r))[0].T          # r x k, R R' = I_r
        A = Q @ R
        lo_o = old(A).item()
        for _ in range(300):                                  # and random rank-r
            Yr = torch.randn(60, r) @ torch.randn(r, k)
            lo_o = min(lo_o, old(Yr).item())
        Qd = torch.linalg.qr(torch.randn(60, k))[0].clone()
        Qd[:, r:] = 0.0
        obs_old.append(min(lo_o, old(Qd).item()))
        obs_new.append(stiefel_defect(A).item())
    ax[1].plot(ranks, obs_old, "o-", color=GREY, lw=2, label="v1 defect")
    ax[1].plot(ranks, obs_new, "o-", color=BLUE, lw=2, label="$D$")
    ax[1].plot(ranks, [1 / r - 1 / k for r in ranks], ":", color=RED, lw=1.6,
               label="$1/r-1/k$ (bound)")
    ax[1].set_xlabel("rank $r$ of $Y$")
    ax[1].set_ylabel("minimum attainable")
    ax[1].set_title("(b) rank-deficient matrices", fontsize=9)
    ax[1].legend()

    # (c) D = 1/EffectiveRank - 1/k, spanning the whole range ER in [1, k]
    Q = torch.linalg.qr(torch.randn(60, k))[0]
    v = torch.randn(60, 1) @ torch.randn(1, k)
    v = v / v.norm() * Q.norm()
    Ds, ers = [], []
    for a in torch.linspace(0.0, 1.0, 260):
        Y = (1 - a) * Q + a * v
        if Y.norm() < 1e-8:
            continue
        d = stiefel_defect(Y).item()
        Ds.append(d)
        ers.append(k / (k * d + 1))
    ax[2].plot(Ds, ers, "-", color=BLUE, lw=2)
    ax[2].axhline(k, ls=":", color=GREY, lw=1.2)
    ax[2].axhline(1, ls=":", color=GREY, lw=1.2)
    ax[2].set_xlabel("$D(Y)$")
    ax[2].set_ylabel("effective rank")
    ax[2].set_ylim(0.8, k + 0.4)
    ax[2].set_title("(c) $D = 1/\\mathrm{ER} - 1/k$", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGS / "functionals.pdf")
    plt.close(fig)
    print("    wrote figs/functionals.pdf")


# ------------------------------------------------------------------------- driver
ALL_STEPS = ["e0", "e1", "e2", "e3", "e4", "e5", "e6"]


def main(quick=False, only=None):
    reps = 2 if quick else 10
    seeds = 3 if quick else 10
    steps = set(ALL_STEPS if not only else only)
    unknown = steps - set(ALL_STEPS)
    if unknown:
        raise SystemExit(f"unknown step(s) {sorted(unknown)}; choose from {ALL_STEPS}")
    t_start = time.time()

    if "e0" in steps:
        print("\n[E0] functional comparison figure")
        fig_functionals()

    if "e2" in steps:
        print("\n[E2] calibration: does w mean what it says?")
        from experiments import exp2_calibration
        cal = exp2_calibration.run()
        save(cal["calibration"], "e2_term_calibration")
        save(cal["effective_fraction"], "e2_effective_fraction")
        save(cal["monotone"], "e2_monotone")


        ef = cal["effective_fraction"]

        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
        for w, mk in zip([0.25, 0.5, 0.9], ["o", "s", "^"]):
            d = ef[ef.w_nominal == w]
            ax[0].plot(d.p, d.v1_effective, mk + "-", color=GREY, lw=1.6, ms=4,
                       label=f"v1, $w={w}$")
            ax[0].plot(d.p, d.v2_effective, mk + "-", color=BLUE, lw=1.6, ms=4,
                       label=f"v2, $w={w}$")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("$p$")
        ax[0].set_ylabel("realised blend fraction")
        ax[0].set_title("(a) what $w$ actually does", fontsize=9)
        ax[0].legend(fontsize=6.5, ncol=2)

        m = cal["monotone"]
        m2 = m[m.version == "v2"].sort_values("w")
        ax[1].plot(m2.w, m2.fidelity, "o-", color=ORANGE, lw=1.8, ms=4, label="fidelity $F$")
        ax[1].plot(m2.w, m2.defect, "s-", color=BLUE, lw=1.8, ms=4, label="defect $D$")
        ax[1].set_xlabel("$w$")
        ax[1].set_ylabel("value at the optimum")
        ax[1].set_yscale("log")
        ax[1].set_title("(b) monotone trade-off", fontsize=9)
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(FIGS / "calibration.pdf")
        plt.close(fig)
        print("    wrote figs/calibration.pdf")

    if "e1" in steps:
        print("\n[E1] NSA-Flow as a refinement operator on a planted basis")
        from experiments import exp1_recovery
        exp1_recovery.SEEDS = range(seeds)
        r1 = save(exp1_recovery.run(), "e1_recovery")
        s1 = save(exp1_recovery.summarise(r1), "e1_recovery_summary")
        save(exp1_recovery.refinement_table(r1), "e1_refinement")


        fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
        cols = {"PCA": BLUE, "NMF": GREEN, "SparsePCA": ORANGE}
        for ax, metric, lab in zip(axes, ["cosine", "support_f1", "overlap"],
                                   ["matched $|\\cos|$ to truth", "support F1",
                                    "mean extra components per feature"]):
            for base, col in cols.items():
                d = (s1[(s1.base == base) & (s1.refined) & (s1.w >= 0)]
                     .groupby("w")[metric].mean())
                ax.plot(d.index, d.values, "o-", color=col, lw=1.8, ms=4, label=f"{base}+NSA")
                b = s1[(s1.base == base) & (~s1.refined)][metric].mean()
                ax.axhline(b, ls=":", color=col, lw=1.3)
            ax.set_xlabel("$w$")
            ax.set_ylabel(lab)
        axes[0].legend(fontsize=6.5)
        axes[0].set_title("(a) basis recovery", fontsize=9)
        axes[1].set_title("(b) support recovery", fontsize=9)
        axes[2].set_title("(c) factor readability", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS / "recovery.pdf")
        plt.close(fig)
        print("    wrote figs/recovery.pdf  (dotted = unrefined base)")

    if "e3" in steps:
        print("\n[E3] the w=1 limit is a feature clustering")
        from experiments import exp3_clustering
        r3 = save(exp3_clustering.run(), "e3_clustering")
        save(exp3_clustering.summarise(r3), "e3_clustering_summary")

    if "e4" in steps:
        print("\n[E4] Golub leukemia")
        from experiments import exp4_golub
        r4 = save(exp4_golub.run(n_repeats=max(2, reps // 2)), "e4_golub")

    if "e5" in steps:
        print("\n[E5] ADNI")
        from experiments import exp5_adni
        r5 = save(exp5_adni.run(hemisphere="right", n_repeats=reps), "e5_adni")
        save(exp5_adni.run(hemisphere="both", n_repeats=max(2, reps // 2)),
             "e5_adni_both_hemispheres")

        def add_delta(df, by=None):
            """Signed AUC difference from the PCA baseline, within each task."""
            df = df.copy()
            if by is None:
                base = df.loc[df.family == "PCA", "auc"].iloc[0]
                df["dAUC_vs_PCA"] = df.auc - base
            else:
                df["dAUC_vs_PCA"] = np.nan
                for t in df[by].unique():
                    m = df[by] == t
                    base = df.loc[m & (df.family == "PCA"), "auc"].iloc[0]
                    df.loc[m, "dAUC_vs_PCA"] = df.loc[m, "auc"] - base
            return df

        r4 = add_delta(r4)
        r5 = add_delta(r5, by="task")
        save(r4, "e4_golub")
        save(r5, "e5_adni")

        fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9))
        d = r4[r4.family == "NSA-PCA"].sort_values("w")
        axes[0].errorbar(d.w, d.auc, yerr=d.auc_sd, fmt="o-", color=BLUE, lw=1.8,
                         ms=4, capsize=2, label="NSA-PCA")
        for fam, ls in [("PCA", ":"), ("SparsePCA", "--"), ("NMF", "-.")]:
            v = r4[r4.family == fam]
            if len(v):
                axes[0].axhline(v.auc.iloc[0], ls=ls, color=GREY, lw=1.4, label=fam)
        axes[0].set_xlabel("$w$")
        axes[0].set_ylabel("CV AUC")
        axes[0].set_title("(a) Golub: ALL vs AML", fontsize=9)
        axes[0].legend(fontsize=7)

        for ax, task in zip(axes[1:], ["CN vs DEM", "CN vs MCI", "MCI vs DEM"]):
            t = r5[r5.task == task]
            d = t[t.family == "NSA-PCA"].sort_values("w")
            ax.errorbar(d.w, d.auc, yerr=d.auc_sd, fmt="o-", color=BLUE, lw=1.8,
                        ms=4, capsize=2, label="NSA-PCA")
            for fam, ls in [("PCA", ":"), ("SparsePCA", "--"), ("NMF", "-.")]:
                v = t[t.family == fam]
                if len(v):
                    ax.axhline(v.auc.iloc[0], ls=ls, color=GREY, lw=1.4, label=fam)
            ax.set_xlabel("$w$")
            ax.set_ylabel("CV AUC")
            ax.set_title(f"({'bcd'[list(['CN vs DEM','CN vs MCI','MCI vs DEM']).index(task)]}) "
                         f"ADNI: {task}", fontsize=9)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "realdata.pdf")
        plt.close(fig)
        print("    wrote figs/realdata.pdf")


        print("\n[E5b] ADNI loadings figure")
        regions, Lpca, Lnsa = exp5_adni.loadings_for_figure(w=0.9)
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 4.4), sharey=True)
        for a, L, t in [(ax[0], np.abs(Lpca), "PCA loadings"),
                        (ax[1], np.abs(Lnsa), "NSA-PCA loadings ($w=0.9$)")]:
            L = L / (L.max(axis=0, keepdims=True) + 1e-12)
            order = np.argsort(L.argmax(axis=1) * 10 - L.max(axis=1))
            im = a.imshow(L[order], aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
            a.set_xticks(range(L.shape[1]))
            a.set_xticklabels([f"C{i+1}" for i in range(L.shape[1])])
            a.set_title(t, fontsize=9)
            a.grid(False)
        ax[0].set_yticks(range(len(regions)))
        ax[0].set_yticklabels([regions[i] for i in order], fontsize=5.5)
        fig.colorbar(im, ax=ax, shrink=0.6, label="|loading| (column-normalised)")
        fig.savefig(FIGS / "adni_loadings.pdf")
        plt.close(fig)
        print("    wrote figs/adni_loadings.pdf")

    if "e6" in steps:
        print("\n[E6] cost and numerical robustness")
        from experiments import exp6_speed
        r6 = exp6_speed.run()
        for nm, d in r6.items():
            save(d, f"e6_{nm}")

        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
        pi = r6["per_iteration"]
        ax[0].scatter(pi.mflop, pi.us_per_iter, s=26, color=BLUE)
        for _, r in pi.iterrows():
            ax[0].annotate(f"{r.p}$\\times${r.k}", (r.mflop, r.us_per_iter),
                           fontsize=5.5, xytext=(3, 2), textcoords="offset points")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("MFLOP per iteration ($4pk^2$)")
        ax[0].set_ylabel("$\\mu$s per iteration")
        ax[0].set_title("(a) factorisation-free inner loop", fontsize=9)

        ee = r6["end_to_end"]
        wd = ee.pivot_table(index=["p", "k", "w"], columns="version", values="seconds")
        wd = wd.dropna()
        idx = np.arange(len(wd))
        ax[1].barh(idx - 0.2, wd["v1"], height=0.38, color=GREY, label="v1")
        ax[1].barh(idx + 0.2, wd["v2"], height=0.38, color=BLUE, label="v2")
        ax[1].set_yticks(idx)
        ax[1].set_yticklabels([f"{p}$\\times${k}, $w$={w}" for p, k, w in wd.index],
                              fontsize=6)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("seconds to convergence")
        ax[1].set_title("(b) end-to-end", fontsize=9)
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(FIGS / "speed.pdf")
        plt.close(fig)
        print("    wrote figs/speed.pdf")


    print("\n[tables] rebuilding LaTeX tables from the saved CSVs")
    from experiments.build_tables import build
    build()

    print(f"\ndone in {time.time() - t_start:.1f}s -> {RES} and {FIGS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer seeds and CV repeats")
    ap.add_argument("--only", nargs="+", metavar="STEP",
                    help=f"run only these steps ({' '.join(ALL_STEPS)})")
    main(**vars(ap.parse_args()))
