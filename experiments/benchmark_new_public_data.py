"""Comprehensive benchmark of NSA-Flow on new public datasets (OpenML).

Datasets evaluated (all new, not previously in repo/paper):
1. Tecator NIR Meat Spectroscopy (OpenML ID 505):
   - n=240 samples, p=100 continuous absorbance channels (850-1050 nm, X >= 0).
   - Target: continuous chemical fat percentage (regression).
   - Tests: non-negative physical flow (V >= 0) vs signed flow vs PCA/NMF.

2. Sonar Mines vs Rocks (OpenML ID 40):
   - n=208 acoustic returns, p=60 frequency energy bands (X in [0, 1]).
   - Target: metal cylinder (mine) vs rock (classification).
   - Tests: continuous acoustic sensor representation without ceiling effects.

3. Prostate Cancer Transcriptomics (OpenML ID 45099, Singh et al. 2002):
   - n=102 subjects (52 tumor vs 50 normal), p=12,600 genes.
   - Target: tumor vs normal prostate tissue (high-dimensional classification).
   - Tests: signed contrast module discovery and consolidation in p >> n biology.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
try:
    from sklearn.decomposition import SparsePCA
except ImportError:
    SparsePCA = None
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, balanced_accuracy_score, accuracy_score

import torch
from nsa_flow import nsa_flow, NSAFlow
from nsa_flow.energy import stiefel_defect_normalised

warnings.filterwarnings("ignore")


def evaluate_tecator(k=5, n_splits=5, seed=42):
    """Tecator NIR Spectrometry -> Chemical Fat % regression."""
    print("\n" + "="*70)
    print(f"BENCHMARK 1: Tecator NIR Spectroscopy (n=240, p=100) -> Fat % [k={k}]")
    print("="*70)
    
    tec = fetch_openml(data_id=505, as_frame=True, parser="auto")
    abs_cols = [c for c in tec.data.columns if "absorbance" in c]
    X_raw = tec.data[abs_cols].to_numpy(dtype=float)
    y = tec.target.to_numpy(dtype=float)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    configs = [
        ("PCA", "pca"),
        ("NMF", "nmf"),
        ("NSA-Flow Nonneg (w=0.0)", "nsa_nonneg_w0"),
        ("NSA-Flow Nonneg (w=0.5)", "nsa_nonneg_w05"),
        ("NSA-Flow Signed (w=0.5)", "nsa_signed_w05"),
        ("NSA-Flow Consol (w=0.5)", "nsa_consol_w05"),
    ]
    
    results = {cfg[0]: {"r2_lin": [], "r2_rf": [], "rmse_lin": [], "time": [], "defect": []} for cfg in configs}
    
    for fold, (tr, te) in enumerate(kf.split(X_raw)):
        X_tr_raw, X_te_raw = X_raw[tr], X_raw[te]
        y_tr, y_te = y[tr], y[te]
        
        # Standardized features for signed methods and PCA
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_raw)
        X_te_s = scaler.transform(X_te_raw)
        
        for name, method in configs:
            t0 = time.time()
            defect_val = 0.0
            
            if method == "pca":
                model = PCA(n_components=k).fit(X_tr_s)
                Z_tr = model.transform(X_tr_s)
                Z_te = model.transform(X_te_s)
                V_t = torch.tensor(model.components_.T, dtype=torch.float32)
                defect_val = float(stiefel_defect_normalised(V_t))
                
            elif method == "nmf":
                # NMF on raw non-negative absorbance
                model = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=seed).fit(X_tr_raw)
                Z_tr = model.transform(X_tr_raw)
                Z_te = model.transform(X_te_raw)
                V_t = torch.tensor(model.components_.T, dtype=torch.float32)
                defect_val = float(stiefel_defect_normalised(V_t))
                
            elif method == "nsa_nonneg_w0":
                # NSA-Flow nonneg on raw absorbance, w=0.0
                res = nsa_flow(X_tr_raw, k=k, w=0.0, mode="data", max_iter=200)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_raw @ V
                Z_te = X_te_raw @ V
                defect_val = float(res["defect_D"])
                
            elif method == "nsa_nonneg_w05":
                # NSA-Flow nonneg on raw absorbance, w=0.5
                res = nsa_flow(X_tr_raw, k=k, w=0.5, mode="data", max_iter=200)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_raw @ V
                Z_te = X_te_raw @ V
                defect_val = float(res["defect_D"])
                
            elif method == "nsa_signed_w05":
                # NSA-Flow signed on standardized spectra, w=0.5
                res = nsa_flow(X_tr_s, k=k, w=0.5, signed=True, max_iter=150)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_s @ V
                Z_te = X_te_s @ V
                defect_val = float(res["defect_D"])
                
            elif method == "nsa_consol_w05":
                # NSA-Flow consolidated signed, w=0.5
                res = nsa_flow(X_tr_s, k=k, w=0.5, signed=True, consolidate=True, max_iter=150)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_s @ V
                Z_te = X_te_s @ V
                defect_val = float(res["defect_D"])
                
            elapsed = time.time() - t0
            
            # Linear regression
            reg_lin = Ridge(alpha=1.0).fit(Z_tr, y_tr)
            y_pred_lin = reg_lin.predict(Z_te)
            r2_l = r2_score(y_te, y_pred_lin)
            rmse_l = np.sqrt(mean_squared_error(y_te, y_pred_lin))
            
            # Non-linear Random Forest
            reg_rf = RandomForestRegressor(n_estimators=100, random_state=seed, max_depth=6).fit(Z_tr, y_tr)
            y_pred_rf = reg_rf.predict(Z_te)
            r2_rf = r2_score(y_te, y_pred_rf)
            
            results[name]["r2_lin"].append(r2_l)
            results[name]["r2_rf"].append(r2_rf)
            results[name]["rmse_lin"].append(rmse_l)
            results[name]["time"].append(elapsed)
            results[name]["defect"].append(defect_val)
            
    # Summarize
    summary = []
    for name in results:
        r2_l_m = np.mean(results[name]["r2_lin"])
        r2_l_s = np.std(results[name]["r2_lin"])
        r2_rf_m = np.mean(results[name]["r2_rf"])
        r2_rf_s = np.std(results[name]["r2_rf"])
        rmse_m = np.mean(results[name]["rmse_lin"])
        t_m = np.mean(results[name]["time"])
        d_m = np.mean(results[name]["defect"])
        summary.append({
            "Method": name,
            "R2 (Ridge)": f"{r2_l_m:.4f} ± {r2_l_s:.3f}",
            "R2 (Forest)": f"{r2_rf_m:.4f} ± {r2_rf_s:.3f}",
            "RMSE (Ridge)": f"{rmse_m:.2f}",
            "Defect D(V)": f"{d_m:.4f}",
            "Fit Time (s)": f"{t_m:.3f}",
            "raw": {
                "r2_lin_mean": r2_l_m, "r2_lin_std": r2_l_s,
                "r2_rf_mean": r2_rf_m, "r2_rf_std": r2_rf_s,
                "rmse_mean": rmse_m, "defect_mean": d_m, "time_mean": t_m
            }
        })
    df = pd.DataFrame(summary).drop(columns=["raw"])
    print(df.to_string(index=False))
    return summary


def evaluate_sonar(k=6, n_splits=5, seed=42):
    """Sonar Mines vs Rocks (Acoustic Chirp Energy Bands) classification."""
    print("\n" + "="*70)
    print(f"BENCHMARK 2: Sonar Mines vs Rocks (n=208, p=60) [k={k}]")
    print("="*70)
    
    son = fetch_openml(data_id=40, as_frame=True, parser="auto")
    X_raw = son.data.to_numpy(dtype=float)
    y = (son.target.to_numpy() == son.target.iloc[0]).astype(int)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    configs = [
        ("PCA", "pca"),
        ("NMF", "nmf"),
        ("NSA-Flow Nonneg (w=0.5)", "nsa_nonneg_w05"),
        ("NSA-Flow Signed (w=0.0)", "nsa_signed_w0"),
        ("NSA-Flow Signed (w=0.5)", "nsa_signed_w05"),
        ("NSA-Flow Consol (w=0.5)", "nsa_consol_w05"),
    ]
    
    results = {cfg[0]: {"auc": [], "bal_acc": [], "acc": [], "time": [], "defect": []} for cfg in configs}
    
    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        X_tr_raw, X_te_raw = X_raw[tr], X_raw[te]
        y_tr, y_te = y[tr], y[te]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_raw)
        X_te_s = scaler.transform(X_te_raw)
        
        for name, method in configs:
            t0 = time.time()
            defect_val = 0.0
            
            if method == "pca":
                model = PCA(n_components=k).fit(X_tr_s)
                Z_tr = model.transform(X_tr_s)
                Z_te = model.transform(X_te_s)
                V_t = torch.tensor(model.components_.T, dtype=torch.float32)
                defect_val = float(stiefel_defect_normalised(V_t))
                
            elif method == "nmf":
                # Raw acoustic chirp energy is in [0, 1]
                model = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=seed).fit(X_tr_raw)
                Z_tr = model.transform(X_tr_raw)
                Z_te = model.transform(X_te_raw)
                V_t = torch.tensor(model.components_.T, dtype=torch.float32)
                defect_val = float(stiefel_defect_normalised(V_t))
                
            elif method == "nsa_nonneg_w05":
                res = nsa_flow(X_tr_raw, k=k, w=0.5, mode="data", max_iter=150)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_raw @ V
                Z_te = X_te_raw @ V
                defect_val = float(res["defect_D"])
                
            elif method == "nsa_signed_w0":
                res = nsa_flow(X_tr_s, k=k, w=0.0, signed=True, max_iter=100)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_s @ V
                Z_te = X_te_s @ V
                defect_val = float(res["defect_D"])
                
            elif method == "nsa_signed_w05":
                res = nsa_flow(X_tr_s, k=k, w=0.5, signed=True, max_iter=100)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_s @ V
                Z_te = X_te_s @ V
                defect_val = float(res["defect_D"])
                
            elif method == "nsa_consol_w05":
                res = nsa_flow(X_tr_s, k=k, w=0.5, signed=True, consolidate=True, max_iter=100)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_s @ V
                Z_te = X_te_s @ V
                defect_val = float(res["defect_D"])
                
            elapsed = time.time() - t0
            
            clf = LogisticRegression(C=1.0, max_iter=500).fit(Z_tr, y_tr)
            y_prob = clf.predict_proba(Z_te)[:, 1]
            y_pred = clf.predict(Z_te)
            
            auc = roc_auc_score(y_te, y_prob)
            bacc = balanced_accuracy_score(y_te, y_pred)
            acc = accuracy_score(y_te, y_pred)
            
            results[name]["auc"].append(auc)
            results[name]["bal_acc"].append(bacc)
            results[name]["acc"].append(acc)
            results[name]["time"].append(elapsed)
            results[name]["defect"].append(defect_val)
            
    summary = []
    for name in results:
        auc_m = np.mean(results[name]["auc"])
        auc_s = np.std(results[name]["auc"])
        bacc_m = np.mean(results[name]["bal_acc"])
        bacc_s = np.std(results[name]["bal_acc"])
        acc_m = np.mean(results[name]["acc"])
        t_m = np.mean(results[name]["time"])
        d_m = np.mean(results[name]["defect"])
        summary.append({
            "Method": name,
            "ROC-AUC": f"{auc_m:.4f} ± {auc_s:.3f}",
            "Balanced Acc": f"{bacc_m:.4f} ± {bacc_s:.3f}",
            "Accuracy": f"{acc_m:.4f}",
            "Defect D(V)": f"{d_m:.4f}",
            "Fit Time (s)": f"{t_m:.3f}",
            "raw": {
                "auc_mean": auc_m, "auc_std": auc_s,
                "bacc_mean": bacc_m, "bacc_std": bacc_s,
                "acc_mean": acc_m, "defect_mean": d_m, "time_mean": t_m
            }
        })
    df = pd.DataFrame(summary).drop(columns=["raw"])
    print(df.to_string(index=False))
    return summary


def evaluate_prostate(k=6, n_splits=5, top_genes=2000, seed=42):
    """Prostate Cancer Transcriptomics (Singh et al. 2002) classification."""
    print("\n" + "="*70)
    print(f"BENCHMARK 3: Prostate Cancer (n=102, p=12,600, top {top_genes} in-fold) [k={k}]")
    print("="*70)
    
    pros = fetch_openml(data_id=45099, as_frame=True, parser="auto")
    X_full = pros.data.to_numpy(dtype=float)
    y = (pros.target.to_numpy() == pros.target.iloc[0]).astype(int)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    configs = [
        ("PCA", "pca"),
        ("NSA-Flow Signed (w=0.0)", "nsa_signed_w0"),
        ("NSA-Flow Signed (w=0.5)", "nsa_signed_w05"),
        ("NSA-Flow Consol (w=0.5)", "nsa_consol_w05"),
    ]
    
    results = {cfg[0]: {"auc": [], "bal_acc": [], "rf_auc": [], "time": [], "defect": [], "sparsity": []} for cfg in configs}
    
    for fold, (tr, te) in enumerate(skf.split(X_full, y)):
        X_tr_raw = X_full[tr]
        X_te_raw = X_full[te]
        y_tr, y_te = y[tr], y[te]
        
        # In-fold gene selection (strictly inside training fold)
        vars_ = np.var(X_tr_raw, axis=0)
        top_idx = np.argsort(vars_)[-top_genes:]
        X_tr = X_tr_raw[:, top_idx]
        X_te = X_te_raw[:, top_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        for name, method in configs:
            t0 = time.time()
            defect_val = 0.0
            sparsity_val = 0.0
            
            if method == "pca":
                model = PCA(n_components=k).fit(X_tr_s)
                Z_tr = model.transform(X_tr_s)
                Z_te = model.transform(X_te_s)
                V_t = torch.tensor(model.components_.T, dtype=torch.float32)
                defect_val = float(stiefel_defect_normalised(V_t))
                sparsity_val = 0.0
                
            elif method == "nsa_signed_w0":
                res = nsa_flow(X_tr_s, k=k, w=0.0, signed=True, max_iter=80)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_s @ V
                Z_te = X_te_s @ V
                defect_val = float(res["defect_D"])
                sparsity_val = float(np.mean(V == 0.0))
                
            elif method == "nsa_signed_w05":
                res = nsa_flow(X_tr_s, k=k, w=0.5, signed=True, max_iter=80)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_s @ V
                Z_te = X_te_s @ V
                defect_val = float(res["defect_D"])
                sparsity_val = float(np.mean(V == 0.0))
                
            elif method == "nsa_consol_w05":
                res = nsa_flow(X_tr_s, k=k, w=0.5, signed=True, consolidate=True, max_iter=80)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_s @ V
                Z_te = X_te_s @ V
                defect_val = float(res["defect_D"])
                sparsity_val = float(np.mean(V == 0.0))
                
            elapsed = time.time() - t0
            
            # Logistic Regression
            clf = LogisticRegression(C=1.0, max_iter=500).fit(Z_tr, y_tr)
            y_prob = clf.predict_proba(Z_te)[:, 1]
            y_pred = clf.predict(Z_te)
            auc = roc_auc_score(y_te, y_prob)
            bacc = balanced_accuracy_score(y_te, y_pred)
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=5).fit(Z_tr, y_tr)
            rf_prob = rf.predict_proba(Z_te)[:, 1]
            rf_auc = roc_auc_score(y_te, rf_prob)
            
            results[name]["auc"].append(auc)
            results[name]["bal_acc"].append(bacc)
            results[name]["rf_auc"].append(rf_auc)
            results[name]["time"].append(elapsed)
            results[name]["defect"].append(defect_val)
            results[name]["sparsity"].append(sparsity_val)
            
    summary = []
    for name in results:
        auc_m = np.mean(results[name]["auc"])
        auc_s = np.std(results[name]["auc"])
        bacc_m = np.mean(results[name]["bal_acc"])
        bacc_s = np.std(results[name]["bal_acc"])
        rf_auc_m = np.mean(results[name]["rf_auc"])
        rf_auc_s = np.std(results[name]["rf_auc"])
        t_m = np.mean(results[name]["time"])
        d_m = np.mean(results[name]["defect"])
        sp_m = np.mean(results[name]["sparsity"])
        summary.append({
            "Method": name,
            "Linear AUC": f"{auc_m:.4f} ± {auc_s:.3f}",
            "Balanced Acc": f"{bacc_m:.4f} ± {bacc_s:.3f}",
            "Forest AUC": f"{rf_auc_m:.4f} ± {rf_auc_s:.3f}",
            "Disjoint Sparsity": f"{sp_m:.1%}",
            "Defect D(V)": f"{d_m:.4f}",
            "Fit Time (s)": f"{t_m:.2f}",
            "raw": {
                "auc_mean": auc_m, "auc_std": auc_s,
                "bacc_mean": bacc_m, "bacc_std": bacc_s,
                "rf_auc_mean": rf_auc_m, "rf_auc_std": rf_auc_s,
                "sparsity_mean": sp_m, "defect_mean": d_m, "time_mean": t_m
            }
        })
    df = pd.DataFrame(summary).drop(columns=["raw"])
    print(df.to_string(index=False))
    return summary


if __name__ == "__main__":
    t_start = time.time()
    res_tec = evaluate_tecator(k=5, n_splits=5)
    res_son = evaluate_sonar(k=6, n_splits=5)
    res_pros = evaluate_prostate(k=6, n_splits=5, top_genes=2000)
    total_time = time.time() - t_start
    print(f"\nAll 3 public benchmarks completed in {total_time:.2f} seconds!")
    
    # Save raw results JSON for the HTML report
    all_out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": total_time,
        "tecator": res_tec,
        "sonar": res_son,
        "prostate": res_pros
    }
    with open("paper/results/new_public_data_benchmark_results.json", "w") as f:
        json.dump(all_out, f, indent=2)
    print("Saved results to paper/results/new_public_data_benchmark_results.json")
