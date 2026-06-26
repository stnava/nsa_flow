import time
import numpy as np
import torch
import pandas as pd

from nsa_flow import nsa_flow_orth, invariant_orthogonality_defect, fidelity_scaled

def run_retraction_benchmark(num_runs=5, p=100, k=20, max_iter=300):
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration list: (retraction_type, ns_iter, label)
    configs = [
        ("soft_polar", 5, "soft_polar"),
        ("soft_cayley", 5, "soft_cayley (ns_iter=5)"),
        ("soft_cayley", 10, "soft_cayley (ns_iter=10)"),
        ("soft_newton_schulz", 5, "soft_newton_schulz (ns_iter=5)"),
        ("soft_newton_schulz", 10, "soft_newton_schulz (ns_iter=10)"),
        ("soft_newton_schulz", 50, "soft_newton_schulz (ns_iter=50)"),
        ("soft_newton_schulz", 100, "soft_newton_schulz (ns_iter=100)"),
    ]
    w_values = [0.1, 0.5, 0.9]
    
    results = []
    
    for w in w_values:
        for run in range(num_runs):
            # Generate target Stiefel matrix and noisy inputs
            X_true, _ = torch.linalg.qr(torch.randn(p, k))
            Y0 = X_true + 0.1 * torch.randn(p, k)
            
            for ret, ns_iter, label in configs:
                t0 = time.time()
                # Run optimization
                res = nsa_flow_orth(
                    Y0, X_true, w=w,
                    retraction=ret,
                    ns_iter=ns_iter,
                    max_iter=max_iter,
                    tol=1e-6,
                    verbose=False,
                    precision="float64",
                    optimizer="adam"
                )
                elapsed = time.time() - t0
                
                Y_final = res["Y"]
                
                # Metrics
                fid = fidelity_scaled(Y_final, X_true).item()
                orth = invariant_orthogonality_defect(Y_final).item()
                total_energy = res["best_total_energy"]
                final_iter = res["final_iter"]
                
                results.append({
                    "w": w,
                    "run": run,
                    "label": label,
                    "total_energy": total_energy,
                    "fidelity": fid,
                    "orthogonality": orth,
                    "iterations": final_iter,
                    "time_sec": elapsed
                })
                
    df = pd.DataFrame(results)
    
    # Compute aggregates
    summary = df.groupby(["w", "label"]).agg({
        "total_energy": ["mean", "std"],
        "fidelity": ["mean", "std"],
        "orthogonality": ["mean", "std"],
        "iterations": ["mean"],
        "time_sec": ["mean", "sum"]
    }).round(6)
    
    print("\n=== Retraction Performance Summary ===")
    print(summary.to_string())
    
    # Save results to a CSV
    df.to_csv("retraction_benchmark_results.csv", index=False)
    print("💾 Saved benchmark results to retraction_benchmark_results.csv")
    
    # Generate Markdown Table
    print("\n### Markdown Summary Table")
    print("| w | Retraction Configuration | Mean Total Energy | Mean Fidelity | Mean Orth Defect | Mean Iters | Total Time (s) |")
    print("|---|---|---|---|---|---|---|")
    for w in w_values:
        for ret, ns_iter, label in configs:
            sub = df[(df["w"] == w) & (df["label"] == label)]
            mean_energy = sub["total_energy"].mean()
            mean_fid = sub["fidelity"].mean()
            mean_orth = sub["orthogonality"].mean()
            mean_iters = sub["iterations"].mean()
            total_time = sub["time_sec"].sum()
            print(f"| {w} | {label} | {mean_energy:.6f} | {mean_fid:.6f} | {mean_orth:.6e} | {mean_iters:.1f} | {total_time:.4f} |")

if __name__ == "__main__":
    run_retraction_benchmark()
