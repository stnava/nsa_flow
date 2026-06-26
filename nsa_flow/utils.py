import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def safe_to_tensor(X, dtype=torch.float64, device=None, name="input"):
    """
    Safely convert an input (NumPy, pandas, or torch) into a clean torch.Tensor.

    Handles:
      - pd.DataFrame with mixed or non-numeric columns
      - np.ndarray with dtype=object or NaNs
      - torch.Tensor (passed through)
    """
    if torch.is_tensor(X):
        return X.to(dtype=dtype, device=device)

    if isinstance(X, pd.DataFrame):
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy()

    if isinstance(X, np.ndarray):
        if X.dtype == object:
            def _safe_float(v):
                try:
                    return float(v)
                except Exception:
                    return 0.0
            X = np.vectorize(_safe_float)(X)
        has_nan = np.isnan(X).any()
        has_inf = np.isinf(X).any()
        if has_nan or has_inf:
            warnings.warn(
                f"safe_to_tensor({name}): replaced "
                f"{'NaN' if has_nan else ''}{'/' if has_nan and has_inf else ''}"
                f"{'Inf' if has_inf else ''} values with 0.0/±1e6.",
                UserWarning, stacklevel=2,
            )
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X = np.array(X, dtype=np.float64)
        return torch.tensor(X, dtype=dtype, device=device)

    raise TypeError(f"Unsupported type for {name}: {type(X)}")

def apply_nonnegativity(Y, mode="softplus"):
    """
    Apply nonnegativity transformation to a tensor.
    """
    if mode in [False, "none", None]:
        return Y
    elif mode == "softplus":
        return F.softplus(Y)
    elif mode == "relu":
        return F.relu(Y)
    elif mode in [True, "hard", "Hard"]:
        return torch.clamp(Y, min=0.0)
    else:
        raise ValueError(f"Invalid apply_nonneg mode: {mode}. "
                         "Use 'none', 'softplus', or True/'hard'.")

def traces_to_dataframe(traces):
    """
    Convert list of trace dictionaries into a clean pandas DataFrame.
    """
    clean = []
    for t in traces:
        clean.append({
            "iter": t["iter"],
            "time": float(t["time"]),
            "fidelity": float(t["fidelity"].detach().cpu().item() if isinstance(t["fidelity"], torch.Tensor) else t["fidelity"]),
            "orthogonality": float(t["orthogonality"]),
            "total_energy": float(t["total_energy"].detach().cpu().item() if isinstance(t["total_energy"], torch.Tensor) else t["total_energy"]),
        })
    return pd.DataFrame(clean)

def plot_nsa_trace(trace_df, retraction="soft_polar", figsize=(8,5)):
    fig, ax1 = plt.subplots(figsize=figsize)

    color_fid = "#1f78b4"      # Fidelity (blue)
    color_orth = "#33a02c"     # Orthogonality (green)

    max_fid = trace_df["fidelity"].max()
    max_orth = trace_df["orthogonality"].max()
    ratio = max_fid / max_orth if max_orth > 0 else 1.0

    ax1.plot(trace_df["iter"], trace_df["fidelity"], color=color_fid, label="Fidelity", linewidth=2)
    ax1.scatter(trace_df["iter"], trace_df["fidelity"], color=color_fid, s=20, alpha=0.7)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fidelity Energy", color=color_fid)
    ax1.tick_params(axis='y', labelcolor=color_fid)

    ax2 = ax1.twinx()
    ax2.plot(trace_df["iter"], trace_df["orthogonality"], color=color_orth, label="Orthogonality", linewidth=2)
    ax2.scatter(trace_df["iter"], trace_df["orthogonality"], color=color_orth, s=20, alpha=0.7)
    ax2.set_ylabel("Orthogonality Defect", color=color_orth)
    ax2.tick_params(axis='y', labelcolor=color_orth)

    plt.title(f"NSA-Flow Optimization Trace: {retraction}\nFidelity and Orthogonality (Dual Scales)", fontsize=13, weight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()

def run_single_experiment(
    size=[50,10],
    w=0.5,
    strategy='auto',
    optimizer_name='lars',
    aggression=0.5,
    fidelity_type="scale_invariant",
    orth_type="scale_invariant",
    device="cpu",
    seed=42,
    verbose=False,
):
    """
    Run a single NSA-Flow optimization with given configuration and aggression level.
    """
    import time
    from .flow import nsa_flow_orth
    from .energy import compute_energy

    torch.manual_seed(seed + int(aggression * 1000))
    np.random.seed(seed + int(aggression * 1000))

    p, k = size
    X0 = torch.randn(p, k, device=device)
    Y0 = torch.randn_like(X0)

    try:
        t0 = time.time()
        res = nsa_flow_orth(
            Y0,
            X0=X0,
            w=w,
            retraction="soft_polar",
            optimizer=optimizer_name,
            lr_strategy=strategy,
            aggression=aggression,
            max_iter=200,
            tol=1e-5,
            verbose=verbose,
            apply_nonneg=True,
            seed=seed,
            record_every=10,
            precision="float32",
        )
        elapsed = time.time() - t0

        Y_final = res.get("Y", None)
        if Y_final is None and "Y_final" in res:
            Y_final = res["Y_final"]
        if Y_final is not None:
            energy_dict = compute_energy(
                Y_final,
                X0,
                w=w,
                fidelity_type=fidelity_type,
                orth_type=orth_type,
                fid_eta=1.0,
                c_orth=1.0,
                track_grad=False,
                return_dict=True,
            )
            total_energy = energy_dict["total"].item()
            fid_energy = energy_dict["fidelity"].item()
            orth_energy = energy_dict["orthogonality"].item()
        else:
            total_energy = fid_energy = orth_energy = np.nan

        best_iter = int(res.get("best_Y_iteration", np.nan))
        success = True

    except Exception as e:
        total_energy = fid_energy = orth_energy = np.nan
        best_iter = np.nan
        elapsed = np.nan
        success = False
        print(f"⚠️ Failure: {strategy} / {optimizer_name} (agg={aggression:.2f}) — {e}")

    return {
        "size": f"{p}x{k}",
        "w": w,
        "strategy": strategy,
        "optimizer": optimizer_name,
        "aggression": aggression,
        "fidelity_type": fidelity_type,
        "orth_type": orth_type,
        "final_energy": total_energy,
        "total_energy": total_energy,
        "fidelity_energy": fid_energy,
        "orth_energy": orth_energy,
        "best_iter": best_iter,
        "elapsed_sec": elapsed,
        "success": success,
    }

def evaluate(seed: int = 42, fast: bool = False, verbose: bool = True):
    """
    Comprehensive benchmark of NSA-Flow configurations.
    """
    import time
    from .optimizer import get_lr_estimation_strategies, get_torch_optimizer
    from .energy import compute_energy

    torch.manual_seed(seed)
    np.random.seed(seed)

    strategies = get_lr_estimation_strategies()
    optimizers = get_torch_optimizer(return_list=True)
    optimizers = [opt for opt in optimizers if opt.lower() not in ["test", "none"]]
    aggressions = [0.0, 0.25, 0.5, 0.75, 1.0]
    ws = [0.01, 0.1, 0.25, 0.5, 0.9, 0.99]
    matrix_sizes = [(20, 5), (50, 10), (100, 20), (200, 40), (40, 200)]

    if fast == 1:
        optimizers = ["lars", "asgd"]
        matrix_sizes = [(50, 20), (20, 50)]
        ws = [0.05, 0.25, 0.5, 0.7, 0.9]
        aggressions = [0.1, 0.25, 0.5, 0.75]
        strategies = ["random", "armijo_aggressive", "armijo", "bayes"]
    elif fast == 2:
        optimizers = ["lars"]
        matrix_sizes = [(50, 10)]
        ws = [0.5, 0.9]
        aggressions = [0.5, 0.75]
        strategies = ["armijo", "bayes"]

    fidelity_types = ["basic", "scale_invariant", "symmetric"]
    orth_types = ["basic", "scale_invariant"]

    all_results = []
    total_jobs = (
        len(strategies)
        * len(optimizers)
        * len(aggressions)
        * len(ws)
        * len(matrix_sizes)
        * len(fidelity_types)
        * len(orth_types)
    )

    print(f"🔍 Evaluating {total_jobs} configurations...\n")

    job = 0
    t0 = time.time()

    for size in matrix_sizes:
        for strategy in strategies:
            for optimizer_name in optimizers:
                for agg in aggressions:
                    for fid_type in fidelity_types:
                        for orth_type in orth_types:
                            for w in ws:
                                job += 1
                                if verbose:
                                    print(
                                        f"[{job:5d}/{total_jobs}] size={size}, strat={strategy}, "
                                        f"opt={optimizer_name}, agg={agg:.2f}, "
                                        f"fid={fid_type}, orth={orth_type}, w={w}"
                                    )

                                try:
                                    res = run_single_experiment(
                                        size=size,
                                        w=w,
                                        strategy=strategy,
                                        optimizer_name=optimizer_name,
                                        aggression=agg,
                                        fidelity_type=fid_type,
                                        orth_type=orth_type,
                                        seed=seed,
                                    )

                                    if "fidelity_energy" not in res or "orth_energy" not in res:
                                        Y_final = res.get("Y_final", None)
                                        X0 = res.get("X0", None)
                                        if Y_final is not None and X0 is not None:
                                            e_dict = compute_energy(
                                                Y_final,
                                                X0,
                                                w=w,
                                                fid_eta=res.get("fid_eta", 1.0),
                                                c_orth=res.get("c_orth", 1.0),
                                                fidelity_type=fid_type,
                                                orth_type=orth_type,
                                                track_grad=False,
                                                return_dict=True,
                                            )
                                            res["fidelity_energy"] = e_dict["fidelity"].item()
                                            res["orth_energy"] = e_dict["orthogonality"].item()
                                            res["total_energy"] = e_dict["total"].item()

                                    res.update(
                                        dict(
                                            size=size,
                                            w=w,
                                            strategy=strategy,
                                            optimizer=optimizer_name,
                                            aggression=agg,
                                            fidelity_type=fid_type,
                                            orth_type=orth_type,
                                            seed=seed,
                                        )
                                    )
                                    all_results.append(res)

                                except Exception as e:
                                    if verbose:
                                        print(
                                            f"⚠️ Failure: {strategy}/{optimizer_name} "
                                            f"(agg={agg}, w={w}) — {e}"
                                        )
                                    all_results.append(
                                        dict(
                                            size=size,
                                            w=w,
                                            strategy=strategy,
                                            optimizer=optimizer_name,
                                            aggression=agg,
                                            fidelity_type=fid_type,
                                            orth_type=orth_type,
                                            error=str(e),
                                        )
                                    )

    df = pd.DataFrame(all_results)

    if "total_energy" in df.columns:
        df["rank_total"] = df["total_energy"].rank(method="dense", ascending=True)
    if "fidelity_energy" in df.columns:
        df["rank_fid"] = df["fidelity_energy"].rank(method="dense", ascending=True)
    if "orth_energy" in df.columns:
        df["rank_orth"] = df["orth_energy"].rank(method="dense", ascending=True)

    elapsed_total = time.time() - t0
    print(f"\n✅ Completed all {job} experiments in {elapsed_total/60:.2f} minutes.")

    if "rank_total" in df.columns:
        summary = (
            df.groupby(["strategy", "optimizer", "fidelity_type", "orth_type", "w"])
            .agg(
                mean_total_energy=("total_energy", "mean"),
                mean_rank=("rank_total", "mean"),
                std_total_energy=("total_energy", "std"),
                n=("total_energy", "count"),
            )
            .sort_values("mean_rank")
        )
        print("\n🏆 Top Performing Configurations:\n")
        print(summary.head(10))

    return df

def plot_evaluation_summary(df, save_dir=None, timestamp=None, show_plots=True):
    """
    Generate and optionally save detailed summary visualizations for NSA-Flow evaluation results.
    """
    import os
    import seaborn as sns

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    def _robust_clip(df, cols, lower=1, upper=99):
        df_clipped = df.copy()
        for col in cols:
            if col in df.columns:
                lo, hi = np.percentile(df[col].dropna(), [lower, upper])
                df_clipped[col] = df[col].clip(lo, hi)
        return df_clipped

    if "error" in df.columns:
        df_plot = df[df["error"].isna()].copy()
    else:
        df_plot = df.copy()

    df_plot = df_plot[df_plot["total_energy"].notna() & np.isfinite(df_plot["total_energy"])]
    df_plot = _robust_clip(df_plot, ["total_energy", "fidelity_energy", "orth_energy"])

    summary = (
        df_plot.groupby(["strategy", "optimizer", "w"])
        .agg(
            mean_total=("total_energy", "mean"),
            std_total=("total_energy", "std"),
            n=("total_energy", "count"),
        )
        .reset_index()
        .sort_values("mean_total", ascending=True)
    )
    print("\n🏆 Top 10 Configurations by Mean Total Energy:\n")
    print(summary.head(10).to_string(index=False, float_format="%.4e"))

    if save_dir:
        summary_csv = f"{save_dir}/summary_ranking_{timestamp or ''}.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"💾 Saved ranking summary: {summary_csv}")

    plt.figure(figsize=(10, 6))
    pivot_strat = (
        df_plot.groupby(["strategy", "aggression"])["total_energy"].mean().unstack()
    )
    sns.heatmap(pivot_strat, annot=True, fmt=".2e", cmap="viridis", cbar_kws={"label": "Mean Total Energy"})
    plt.title("NSA-Flow: Strategy vs Aggression (Mean Total Energy)")
    plt.ylabel("Strategy")
    plt.xlabel("Aggression Level")
    plt.tight_layout()
    if save_dir:
        fname = f"{save_dir}/heatmap_strategy_vs_aggression_{timestamp or ''}.png"
        plt.savefig(fname, dpi=300)
        print(f"💾 Saved: {fname}")
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    pivot_opt = (
        df_plot.groupby(["optimizer", "aggression"])["total_energy"].mean().unstack()
    )
    sns.heatmap(pivot_opt, annot=True, fmt=".2e", cmap="magma", cbar_kws={"label": "Mean Total Energy"})
    plt.title("NSA-Flow: Optimizer vs Aggression (Mean Total Energy)")
    plt.ylabel("Optimizer")
    plt.xlabel("Aggression Level")
    plt.tight_layout()
    if save_dir:
        fname = f"{save_dir}/heatmap_optimizer_vs_aggression_{timestamp or ''}.png"
        plt.savefig(fname, dpi=300)
        print(f"💾 Saved: {fname}")
    if show_plots:
        plt.show()
    plt.close()

    if "w" in df_plot.columns:
        plt.figure(figsize=(10, 6))
        pivot_w = df_plot.groupby(["strategy", "w"])["total_energy"].mean().unstack()
        sns.heatmap(pivot_w, annot=True, fmt=".2e", cmap="plasma", cbar_kws={"label": "Mean Total Energy"})
        plt.title("NSA-Flow: Strategy vs Weight (w)")
        plt.ylabel("Strategy")
        plt.xlabel("Weight (w)")
        plt.tight_layout()
        if save_dir:
            fname = f"{save_dir}/heatmap_strategy_vs_w_{timestamp or ''}.png"
            plt.savefig(fname, dpi=300)
            print(f"💾 Saved: {fname}")
        if show_plots:
            plt.show()
        plt.close()

        g = sns.FacetGrid(df_plot, col="optimizer", row="strategy", margin_titles=True, sharey=False)
        g.map_dataframe(sns.lineplot, x="aggression", y="total_energy", hue="w", marker="o")
        g.add_legend(title="Weight (w)")
        g.set_axis_labels("Aggression", "Total Energy")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("NSA-Flow: Total Energy vs Aggression (Faceted by Optimizer × Strategy)")

        best_row = df_plot.loc[df_plot["total_energy"].idxmin()]
        best_opt, best_strat = best_row["optimizer"], best_row["strategy"]
        best_agg, best_w, best_E = best_row["aggression"], best_row["w"], best_row["total_energy"]

        for (opt, strat), ax in g.axes_dict.items():
            if opt == best_opt and strat == best_strat:
                ax.scatter(best_agg, best_E, s=150, color="gold", edgecolor="black", marker="*", zorder=10, label="⭐ Best")
                ax.legend()

        if save_dir:
            fname = f"{save_dir}/facet_energy_vs_aggression_{timestamp or ''}_highlighted.png"
            g.savefig(fname, dpi=300)
            print(f"💾 Saved with highlight: {fname}")

        if show_plots:
            plt.show()
        plt.close()

    mean_energy = df_plot.groupby(["strategy", "aggression"])["total_energy"].mean().reset_index()
    top_strats = mean_energy.groupby("strategy")["total_energy"].mean().nsmallest(5).index

    plt.figure(figsize=(8, 5))
    for strat in top_strats:
        subset = mean_energy[mean_energy["strategy"] == strat]
        plt.plot(subset["aggression"], subset["total_energy"], "-o", label=strat)
    plt.xlabel("Aggression Level")
    plt.ylabel("Mean Total Energy")
    plt.title("Top Strategies: Energy vs Aggression")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        fname = f"{save_dir}/lineplot_top_strategies_{timestamp or ''}.png"
        plt.savefig(fname, dpi=300)
        print(f"💾 Saved: {fname}")
    if show_plots:
        plt.show()
    plt.close()

    if "w" in df_plot.columns:
        mean_w = df_plot.groupby(["optimizer", "w"])["total_energy"].mean().reset_index()
        top_opts = mean_w.groupby("optimizer")["total_energy"].mean().nsmallest(5).index
        plt.figure(figsize=(8, 5))
        for opt in top_opts:
            subset = mean_w[mean_w["optimizer"] == opt]
            plt.plot(subset["w"], subset["total_energy"], "-o", label=opt)
        plt.xlabel("Weight (w)")
        plt.ylabel("Mean Total Energy")
        plt.title("Top Optimizers: Energy vs Weight")
        plt.legend()
        plt.tight_layout()
        if save_dir:
            fname = f"{save_dir}/lineplot_top_optimizers_vs_w_{timestamp or ''}.png"
            plt.savefig(fname, dpi=300)
            print(f"💾 Saved: {fname}")
        if show_plots:
            plt.show()
        plt.close()

    print("✅ All summary plots and rankings generated successfully.")
