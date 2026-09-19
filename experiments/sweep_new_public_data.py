"""Parametric sweep of w and algorithmic variants on new public benchmarks.

Evaluates:
1. Trade-off parameter w sweep: w in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
2. Algorithmic variants:
   - Non-negative mode (V >= 0) vs Signed relaxed (V = V+ - V-) vs Signed consolidated (disjoint)
   - Optimizer variants: torch_lbfgs vs SPG (speed, defect, energy)
3. Datasets:
   - Tecator NIR Spectroscopy (n=240, p=100) -> Fat % regression
   - Sonar Mines vs Rocks (n=208, p=60) -> Classification
   - Prostate Cancer (n=102, p=12,600, top 2000 in-fold) -> Classification
"""

import time
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score, balanced_accuracy_score

import torch
from nsa_flow import nsa_flow

warnings.filterwarnings("ignore")


def sweep_tecator(w_list=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], k=5, n_splits=5, seed=42):
    print("\n" + "="*70)
    print(f"TECATOR NIR SWEEP: w in {w_list} across Nonneg, Signed, Consol")
    print("="*70)
    tec = fetch_openml(data_id=505, as_frame=True, parser="auto")
    abs_cols = [c for c in tec.data.columns if "absorbance" in c]
    X_raw = tec.data[abs_cols].to_numpy(dtype=float)
    y = tec.target.to_numpy(dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    sweep_records = []

    for w in w_list:
        modes = [
            ("Non-negative (V >= 0)", "nonneg", False),
            ("Signed Relaxed", "signed", False),
            ("Signed Consolidated", "signed", True),
        ]
        for mode_name, mode_type, consol in modes:
            r2_folds = []
            defect_folds = []
            fid_folds = []
            time_folds = []

            for tr, te in kf.split(X_raw):
                X_tr_raw, X_te_raw = X_raw[tr], X_raw[te]
                y_tr, y_te = y[tr], y[te]

                t0 = time.time()
                if mode_type == "nonneg":
                    res = nsa_flow(X_tr_raw, k=k, w=w, mode="data", max_iter=200)
                    V = res.V.cpu().numpy()
                    Z_tr = X_tr_raw @ V
                    Z_te = X_te_raw @ V
                else:
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr_raw)
                    X_te_s = scaler.transform(X_te_raw)
                    res = nsa_flow(X_tr_s, k=k, w=w, signed=True, consolidate=consol, max_iter=150)
                    V = res.V.cpu().numpy()
                    Z_tr = X_tr_s @ V
                    Z_te = X_te_s @ V

                elapsed = time.time() - t0
                reg = Ridge(alpha=1.0).fit(Z_tr, y_tr)
                r2 = r2_score(y_te, reg.predict(Z_te))

                r2_folds.append(r2)
                defect_folds.append(float(res.defect))
                fid_folds.append(float(res.fidelity))
                time_folds.append(elapsed)

            rec = {
                "dataset": "tecator",
                "w": w,
                "variant": mode_name,
                "r2_mean": float(np.mean(r2_folds)),
                "r2_std": float(np.std(r2_folds)),
                "defect_mean": float(np.mean(defect_folds)),
                "fidelity_mean": float(np.mean(fid_folds)),
                "time_mean": float(np.mean(time_folds)),
            }
            sweep_records.append(rec)
            print(f"w={w:<4} | {mode_name:<22} | R2 = {rec['r2_mean']:.4f} ± {rec['r2_std']:.3f} | Defect = {rec['defect_mean']:.4e} | Time = {rec['time_mean']:.3f}s")

    return sweep_records


def sweep_sonar(w_list=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], k=6, n_splits=5, seed=42):
    print("\n" + "="*70)
    print(f"SONAR ACOUSTIC SWEEP: w in {w_list} across Signed and Consol")
    print("="*70)
    son = fetch_openml(data_id=40, as_frame=True, parser="auto")
    X_raw = son.data.to_numpy(dtype=float)
    y = (son.target.to_numpy() == son.target.iloc[0]).astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sweep_records = []

    for w in w_list:
        modes = [
            ("Signed Relaxed", False),
            ("Signed Consolidated", True),
        ]
        for mode_name, consol in modes:
            auc_folds = []
            bacc_folds = []
            defect_folds = []
            time_folds = []

            for tr, te in skf.split(X_raw, y):
                X_tr, X_te = X_raw[tr], X_raw[te]
                y_tr, y_te = y[tr], y[te]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                t0 = time.time()
                res = nsa_flow(X_tr_s, k=k, w=w, signed=True, consolidate=consol, max_iter=100)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_s @ V
                Z_te = X_te_s @ V
                elapsed = time.time() - t0

                clf = LogisticRegression(C=1.0, max_iter=500).fit(Z_tr, y_tr)
                auc = roc_auc_score(y_te, clf.predict_proba(Z_te)[:, 1])
                bacc = balanced_accuracy_score(y_te, clf.predict(Z_te))

                auc_folds.append(auc)
                bacc_folds.append(bacc)
                defect_folds.append(float(res.defect))
                time_folds.append(elapsed)

            rec = {
                "dataset": "sonar",
                "w": w,
                "variant": mode_name,
                "auc_mean": float(np.mean(auc_folds)),
                "auc_std": float(np.std(auc_folds)),
                "bacc_mean": float(np.mean(bacc_folds)),
                "bacc_std": float(np.std(bacc_folds)),
                "defect_mean": float(np.mean(defect_folds)),
                "time_mean": float(np.mean(time_folds)),
            }
            sweep_records.append(rec)
            print(f"w={w:<4} | {mode_name:<20} | AUC = {rec['auc_mean']:.4f} ± {rec['auc_std']:.3f} | BalAcc = {rec['bacc_mean']:.4f} | Defect = {rec['defect_mean']:.4e}")

    return sweep_records


def sweep_prostate(w_list=[0.0, 0.25, 0.5, 0.75, 0.9], k=6, n_splits=5, top_genes=2000, seed=42):
    print("\n" + "="*70)
    print(f"PROSTATE TRANSCRIPTOMICS SWEEP: w in {w_list} across Signed and Consol")
    print("="*70)
    pros = fetch_openml(data_id=45099, as_frame=True, parser="auto")
    X_full = pros.data.to_numpy(dtype=float)
    y = (pros.target.to_numpy() == pros.target.iloc[0]).astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sweep_records = []

    for w in w_list:
        modes = [
            ("Signed Relaxed", False),
            ("Signed Consolidated", True),
        ]
        for mode_name, consol in modes:
            auc_folds = []
            bacc_folds = []
            defect_folds = []
            sparsity_folds = []
            time_folds = []

            for tr, te in skf.split(X_full, y):
                X_tr_raw, X_te_raw = X_full[tr], X_full[te]
                y_tr, y_te = y[tr], y[te]

                # In-fold gene selection
                vars_ = np.var(X_tr_raw, axis=0)
                top_idx = np.argsort(vars_)[-top_genes:]
                X_tr = X_tr_raw[:, top_idx]
                X_te = X_te_raw[:, top_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                t0 = time.time()
                res = nsa_flow(X_tr_s, k=k, w=w, signed=True, consolidate=consol, max_iter=60)
                V = res.V.cpu().numpy()
                Z_tr = X_tr_s @ V
                Z_te = X_te_s @ V
                elapsed = time.time() - t0

                clf = LogisticRegression(C=1.0, max_iter=500).fit(Z_tr, y_tr)
                auc = roc_auc_score(y_te, clf.predict_proba(Z_te)[:, 1])
                bacc = balanced_accuracy_score(y_te, clf.predict(Z_te))

                auc_folds.append(auc)
                bacc_folds.append(bacc)
                defect_folds.append(float(res.defect))
                sparsity_folds.append(float(np.mean(V == 0.0)))
                time_folds.append(elapsed)

            rec = {
                "dataset": "prostate",
                "w": w,
                "variant": mode_name,
                "auc_mean": float(np.mean(auc_folds)),
                "auc_std": float(np.std(auc_folds)),
                "bacc_mean": float(np.mean(bacc_folds)),
                "bacc_std": float(np.std(bacc_folds)),
                "sparsity_mean": float(np.mean(sparsity_folds)),
                "defect_mean": float(np.mean(defect_folds)),
                "time_mean": float(np.mean(time_folds)),
            }
            sweep_records.append(rec)
            print(f"w={w:<4} | {mode_name:<20} | BalAcc = {rec['bacc_mean']:.4f} ± {rec['bacc_std']:.3f} | AUC = {rec['auc_mean']:.4f} | Sparsity = {rec['sparsity_mean']:.1%}")

    return sweep_records


def sweep_optimizers(k=5, seed=42):
    print("\n" + "="*70)
    print("OPTIMIZER SHOOTOUT ON NEW DATA: torch_lbfgs vs SPG on Tecator")
    print("="*70)
    tec = fetch_openml(data_id=505, as_frame=True, parser="auto")
    abs_cols = [c for c in tec.data.columns if "absorbance" in c]
    X_raw = tec.data[abs_cols].to_numpy(dtype=float)

    records = []
    for opt in ["torch_lbfgs", "spg"]:
        for w in [0.1, 0.5, 0.9]:
            t0 = time.time()
            res = nsa_flow(X_raw, k=k, w=w, mode="data", optimizer=opt, max_iter=1000)
            elapsed = time.time() - t0
            rec = {
                "optimizer": opt,
                "w": w,
                "energy": float(res.energy),
                "defect": float(res.defect),
                "iters": int(res.iters),
                "stop_reason": res.stop_reason,
                "seconds": elapsed,
            }
            records.append(rec)
            print(f"{opt:<12} | w={w} | Energy={res.energy:.6e} | Defect={res.defect:.4e} | iters={res.iters:<4} | stop={res.stop_reason:<12} | time={elapsed:.3f}s")
    return records


if __name__ == "__main__":
    t0 = time.time()
    tec_res = sweep_tecator()
    son_res = sweep_sonar()
    pros_res = sweep_prostate()
    opt_res = sweep_optimizers()
    total_time = time.time() - t0
    print(f"\nAll sweeps completed in {total_time:.2f} seconds!")

    all_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time": total_time,
        "tecator_sweep": tec_res,
        "sonar_sweep": son_res,
        "prostate_sweep": pros_res,
        "optimizer_shootout": opt_res,
    }
    with open("paper/results/new_public_data_sweeps.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print("Saved all sweep data to paper/results/new_public_data_sweeps.json")
