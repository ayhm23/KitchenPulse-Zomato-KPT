"""
KitchenPulse — Robustness & Sensitivity Analysis
==================================================
Runs three independent evaluations treating the prediction mechanism as a
black box, operating on processed_orders.csv.

Tests:
  1. Bootstrap confidence intervals — 2000 resamples, 95% CI for MAE,
     avg rider wait, and % orders > 5 min wait.
  2. Ablation study — drop one KLI signal at a time, recompute KLI,
     measure MAE impact.
  3. OOD stress test — scale hidden_load from 0.5x to 2.0x, measure
     KitchenPulse vs Baseline MAE across load regimes.

FIX SUMMARY:
  - Required columns check: replaced 'acceptance_latency_zscore' with
    'acceptance_latency_seconds' (the raw input column that must be present).
    The z-score 'acceptance_latency_zscore' is a derived output that may not
    exist if pipeline was run with an older version of kitchen_load_index.py.
  - Rider wait bootstrap uses bootstrap_mean() on the model-independent
    actual_rider_wait_minutes column (not dependent on KPT model).
  - Clarified that the ablation is within-model (same processed df), not
    out-of-sample — see docstring in run_ablation().

Run:
    python analysis/robustness_tests.py
    (requires data/processed_orders.csv — run simulation/run_simulation.py first)
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

DATA_PATH = "data/processed_orders.csv"
os.makedirs('analysis', exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_pred, n_boot=2000, alpha=0.05):
    """Bootstrap MAE confidence interval."""
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        stats.append(mean_absolute_error(y_true[idx], y_pred[idx]))
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return np.mean(stats), lo, hi


def bootstrap_mean(series, n_boot=2000, alpha=0.05):
    """Bootstrap the mean of a series (model-independent)."""
    stats = []
    data = series.dropna().values
    for _ in range(n_boot):
        idx = np.random.randint(0, len(data), len(data))
        stats.append(np.mean(data[idx]))
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return np.mean(stats), lo, hi


def run_bootstrap(df):
    print("\n=== Bootstrap Confidence Intervals (n_boot=2000) ===")

    y_true = df["true_kpt_minutes"].values
    baseline = df["naive_kpt_estimate"].values
    kp = df["kli_adjusted_kpt"].values

    base_mae, base_lo, base_hi = bootstrap_ci(y_true, baseline)
    kp_mae, kp_lo, kp_hi = bootstrap_ci(y_true, kp)

    print(f"KPT MAE   : baseline {base_mae:.2f} [{base_lo:.2f},{base_hi:.2f}]  |  "
          f"KP {kp_mae:.2f} [{kp_lo:.2f},{kp_hi:.2f}]")
    if kp_hi < base_lo:
        print("=> MAE improvement is statistically significant (CIs do not overlap).")
    else:
        print("=> Improvement may not be statistically significant; CIs overlap.")

    # Rider wait metrics are model-independent (they depend on actual_ready_time,
    # not on our KPT prediction), so we just bootstrap the observed distribution.
    wait = df["actual_rider_wait_minutes"]

    w_mean, w_lo, w_hi = bootstrap_mean(wait)
    print(f"Avg rider wait : {w_mean:.2f} [{w_lo:.2f},{w_hi:.2f}] (model-independent, observed)")

    pct_over_5 = (wait > 5.0).astype(float)
    pct_mean, pct_lo, pct_hi = bootstrap_mean(pct_over_5)
    print(f"% >5min wait   : {pct_mean*100:.1f}% [{pct_lo*100:.1f}%,{pct_hi*100:.1f}%] "
          f"(model-independent, observed)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────

def recompute_kli_ablation(df, drop_signal=None):
    """
    Recompute KLI from normalised signals already in the processed df,
    optionally zeroing out one signal and redistributing its weight.

    NOTE: This is a within-model ablation — it measures how much each signal
    contributes to the KLI score on the same processed dataset. It is NOT an
    out-of-sample test.
    """
    signals = {
        "concurrent":   df["zomato_concurrent_orders"] / 15.0,
        "latency":      df["latency_norm"] if "latency_norm" in df.columns
                        else df["acceptance_latency_seconds"].clip(lower=0) / df["acceptance_latency_seconds"].max(),
        "foot_traffic": df["local_foot_traffic_index"] / 100.0,
        "competitor":   df["competitor_platform_orders"].clip(0, 15) / 15.0,
    }

    weights = {
        "concurrent":   0.30,
        "latency":      0.25,
        "foot_traffic": 0.30,
        "competitor":   0.15,
    }

    if drop_signal and drop_signal in weights:
        weights[drop_signal] = 0.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

    kli = sum(weights[k] * signals[k] for k in signals)
    # Re-normalise to 0–100
    kli_min, kli_max = kli.min(), kli.max()
    if kli_max > kli_min:
        kli = 100 * (kli - kli_min) / (kli_max - kli_min)
    else:
        kli = kli * 0 + 50
    return kli


def run_ablation(df):
    """
    Ablation study: drop one signal at a time and measure the impact on MAE.

    The clean base signal (POS for T1, corrected_for_kpt for T2/T3) is
    recovered by reversing the KLI multiplier from the processed df.
    """
    print("\n=== Ablation Study ===")
    print("  Note: within-model ablation on processed_orders.csv (not out-of-sample)")

    y_true = df["true_kpt_minutes"]
    results = []

    # Recover clean base signal by reversing the KLI multiplier
    original_multiplier = 1 + (df["kitchen_load_index"] - 50) / 200
    clean_base = df["kli_adjusted_kpt"] / original_multiplier

    for drop in [None, "concurrent", "latency", "foot_traffic", "competitor"]:
        kli = recompute_kli_ablation(df, drop_signal=drop)
        adjusted = clean_base * (1 + (kli - 50) / 200)
        m = mean_absolute_error(y_true, adjusted)
        key = drop if drop else "all_signals"
        results.append((key, m))
        print(f"  Dropped {key:<15} → MAE {m:.2f}")

    labels, maes = zip(*results)
    plt.figure(figsize=(8, 5))
    colors = ['#27AE60' if l == 'all_signals' else '#3498DB' for l in labels]
    plt.bar(labels, maes, color=colors)
    plt.ylabel("MAE vs True KPT (minutes)")
    plt.xticks(rotation=45)
    plt.title("Ablation Study — Signal Contribution")
    plt.ylim(0, max(maes) + 1)
    for i, v in enumerate(maes):
        plt.text(i, v + 0.1, f'{v:.2f}m', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig("analysis/ablation_results.png")
    plt.close()
    print("  Chart → analysis/ablation_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. OOD STRESS TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_ood_test(df):
    """
    Scale hidden_load from 0.5x to 2.0x to simulate out-of-distribution load
    conditions. Measures KitchenPulse vs Baseline MAE at each level.
    """
    print("\n=== OOD Hidden Load Multiplier Test ===")
    multipliers = np.linspace(0.5, 2.0, 8)
    maes_kp = []
    maes_baseline = []

    original_multiplier = 1 + (df["kitchen_load_index"] - 50) / 200
    clean_base = df["kli_adjusted_kpt"] / original_multiplier

    for m in multipliers:
        temp = df.copy()
        temp["true_kpt_minutes_mod"] = (
            temp["base_kpt_minutes"]
            + temp["hidden_load"] * m * 1.5
            + temp["zomato_concurrent_orders"] * 0.8
        )
        kli = recompute_kli_ablation(temp)
        adjusted = clean_base * (1 + (kli - 50) / 200)

        mae_kp = mean_absolute_error(temp["true_kpt_minutes_mod"], adjusted)
        mae_base = mean_absolute_error(temp["true_kpt_minutes_mod"], temp["naive_kpt_estimate"])

        maes_kp.append(mae_kp)
        maes_baseline.append(mae_base)
        print(f"  Hidden Load x{m:.2f} → KP MAE: {mae_kp:.2f} | Baseline MAE: {mae_base:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(multipliers, maes_baseline, marker="x", color="#E74C3C", linewidth=2, label="Zomato Baseline")
    plt.plot(multipliers, maes_kp, marker="o", color="#27AE60", linewidth=2, label="KitchenPulse")
    plt.xlabel("Hidden Load Multiplier (1.0 = Normal)")
    plt.ylabel("MAE vs True KPT (minutes)")
    plt.title("OOD Stress Test — Hidden Load Sensitivity")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("analysis/ood_stress_test.png")
    plt.close()
    print("  Chart → analysis/ood_stress_test.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# FIX: removed 'acceptance_latency_zscore' from required columns — that is a
# derived output column, not guaranteed to be present. The raw input column
# 'acceptance_latency_seconds' must be present (it is always in the dataset).
REQUIRED_COLUMNS = [
    "true_kpt_minutes",
    "naive_kpt_estimate",
    "kli_adjusted_kpt",
    "kitchen_load_index",
    "base_kpt_minutes",
    "hidden_load",
    "zomato_concurrent_orders",
    "acceptance_latency_seconds",
    "local_foot_traffic_index",
    "competitor_platform_orders",
    "actual_rider_wait_minutes",
]

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run simulation/run_simulation.py first."
        )

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    run_bootstrap(df)
    run_ablation(df)
    run_ood_test(df)

    print("\nRobustness tests complete.")
    print("  analysis/ablation_results.png")
    print("  analysis/ood_stress_test.png\n")
