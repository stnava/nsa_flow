#!/usr/bin/env python3
"""
NSA-Flow: Comprehensive Evaluation Runner
=========================================

This script runs the full cross-evaluation benchmark across:
  - Learning rate estimation strategies
  - Torch optimizers
  - Aggression levels
  - Matrix sizes
  - Energy computation variants

It stores a summary DataFrame and generates visual heatmaps and diagnostics.

Usage:
    python main_evaluation.py

Outputs:
    results/evaluation_results.csv
    results/heatmap_total_energy.png
    results/heatmap_rank.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import nsa_flow


def main():
    # -------------------------------------------------------------------------
    # 1️⃣ Setup output directory and seed
    # -------------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n🚀 Starting NSA-Flow Comprehensive Evaluation\n")

    # -------------------------------------------------------------------------
    # 2️⃣ Run all experiments
    # -------------------------------------------------------------------------
    df = nsa_flow.evaluate(seed=42, fast=1, verbose=True)

    # Save results to CSV
    results_path = f"results/nsa_flow_eval_{timestamp}.csv"
    df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")

    # -------------------------------------------------------------------------
    # 3️⃣ Generate summary plots
    # -------------------------------------------------------------------------
    print("\n📊 Generating summary visualizations...\n")
    nsa_flow.plot_evaluation_summary(df, save_dir="results", timestamp=timestamp)

    print("\n✅ Evaluation complete.\n")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()