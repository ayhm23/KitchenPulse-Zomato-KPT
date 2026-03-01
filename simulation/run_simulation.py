"""
simulation/run_simulation.py
-----------------------------
Before vs after comparison across all pipeline strategies.

Original logic (preserved):
  - Baseline: raw FOR button signal into naive KPT
  - Denoised FOR: bias-corrected FOR signal, no KLI
  - KitchenPulse (Full): corrected FOR + KLI-adjusted KPT

New additions (v2):
  [Feature 1] Adversarial Noise Injection
    inject_adversarial_noise() randomly corrupts a configurable fraction
    of rows in the dataset with extreme, non-linear anomalies:
      - Massive spikes in competitor_orders (uncorrelated with true_kpt)
      - Random 20-minute FOR delays for otherwise 'honest' merchants

    The simulation runs FOUR strategies: Baseline, Denoised FOR,
    KitchenPulse (Full), and KitchenPulse under adversarial conditions,
    demonstrating the system's fault tolerance.

  [Feature 4] EMA strategy is also benchmarked alongside the static median.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from pipeline.signal_denoiser import (
    flag_rider_proximate_events,
    compute_bias_correction,
    apply_correction,
    apply_ema_offset,
)
from pipeline.kitchen_load_index import (
    compute_acceptance_latency_zscore,
    compute_concurrent_order_pressure,
    compute_kli,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

ADVERSARIAL_FRACTION: float = 0.08   # 8% of rows get adversarial noise
RANDOM_SEED: int = 99


# ──────────────────────────────────────────────────────────────────────────────
# [Feature 1] Adversarial noise injection
# ──────────────────────────────────────────────────────────────────────────────

def inject_adversarial_noise(
    df: pd.DataFrame,
    fraction: float = ADVERSARIAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    [Feature 1: Adversarial Noise Injection]

    Randomly injects two categories of extreme non-linear anomalies into
    a `fraction` of rows to stress-test pipeline fault tolerance:

    Type A — Competitor order spike (uncorrelated garbage):
      Sets competitor_orders to a massive value (50–200) that has NO
      relationship with true_kpt_minutes. A naive model would be badly
      misled; a robust pipeline should absorb this gracefully.

    Type B — Honest merchant FOR delay flip:
      For rows from 'honest' merchants (rider_present_at_press == False),
      injects a random 18–22 minute delay into for_button_time, simulating
      a day where even an otherwise honest merchant gaming the system.
      This is deliberately non-linear: the delay is sampled from a heavy-
      tailed distribution rather than the Gaussian used for normal bias.

    Both types are flagged in a new column `adversarial_noise_type` for
    post-hoc analysis of where the pipeline succeeded or failed.

    Returns:
        A copy of df with adversarial rows mutated in-place.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    n_noisy = int(n * fraction)
    noisy_indices = rng.choice(df.index, size=n_noisy, replace=False)

    # Split evenly between the two noise types
    type_a_idx = noisy_indices[:n_noisy // 2]
    type_b_idx = noisy_indices[n_noisy // 2:]

    df['adversarial_noise_type'] = 'none'

    # Type A: Competitor order spike (uncorrelated with true KPT)
    if 'competitor_orders' not in df.columns:
        df['competitor_orders'] = 0.0
    df.loc[type_a_idx, 'competitor_orders'] = rng.uniform(50, 200, size=len(type_a_idx))
    df.loc[type_a_idx, 'adversarial_noise_type'] = 'competitor_spike'

    # Type B: Honest merchant FOR delay flip (heavy-tailed, ~20 min)
    # Use Pareto distribution for heavy tail rather than Gaussian
    delay_minutes = rng.pareto(a=1.5, size=len(type_b_idx)) * 6 + 18  # mean ~20 min
    delay_seconds = delay_minutes * 60

    df['for_button_time'] = pd.to_datetime(df['for_button_time'])
    new_times = (
        df.loc[type_b_idx, 'for_button_time']
        + pd.to_timedelta(np.round(delay_seconds).astype(int), unit='s')
    )
    df.loc[type_b_idx, 'for_button_time'] = new_times
    df.loc[type_b_idx, 'adversarial_noise_type'] = 'honest_merchant_flip'

    print(
        f"[Adversarial] Injected noise into {n_noisy} rows "
        f"({fraction:.0%} of {n}):\n"
        f"  Type A (competitor spike):        {len(type_a_idx)} rows\n"
        f"  Type B (honest merchant flip):    {len(type_b_idx)} rows"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def mean_absolute_error(pred: pd.Series, actual: pd.Series) -> float:
    return float(np.mean(np.abs(pred - actual)))


def avg_rider_wait(df: pd.DataFrame, kpt_col: str) -> float:
    """
    Estimate average rider wait using the KPT estimate vs actual ready time.
    Rider wait = max(0, rider_arrival - actual_ready_time).
    Here we approximate from KPT error: wait = max(0, estimated_ready - actual_ready).
    """
    estimated_ready = pd.to_datetime(df['order_time']) + pd.to_timedelta(
        df[kpt_col], unit='m'
    )
    actual_ready = pd.to_datetime(df['actual_ready_time'])
    wait = (estimated_ready - actual_ready).dt.total_seconds() / 60
    return float(wait.clip(lower=0).mean())


def pct_wait_over_5min(df: pd.DataFrame, kpt_col: str) -> float:
    estimated_ready = pd.to_datetime(df['order_time']) + pd.to_timedelta(
        df[kpt_col], unit='m'
    )
    actual_ready = pd.to_datetime(df['actual_ready_time'])
    wait = (estimated_ready - actual_ready).dt.total_seconds() / 60
    return float((wait > 5).mean() * 100)


def print_strategy_row(
    label: str, df: pd.DataFrame, kpt_col: str, baseline_mae: float
) -> dict:
    mae  = mean_absolute_error(df[kpt_col], df['true_kpt_minutes'])
    wait = avg_rider_wait(df, kpt_col)
    pct5 = pct_wait_over_5min(df, kpt_col)
    delta = (baseline_mae - mae) / baseline_mae * 100

    print(
        f"  {label:<40} MAE={mae:.2f}m  "
        f"AvgWait={wait:.2f}m  "
        f">5min={pct5:.1f}%  "
        f"vs Baseline={delta:+.1f}%"
    )
    return {"label": label, "mae": mae, "avg_wait": wait,
            "pct_over_5min": pct5, "mae_delta_pct": delta}


# ──────────────────────────────────────────────────────────────────────────────
# Main simulation runner
# ──────────────────────────────────────────────────────────────────────────────

def run(inject_noise: bool = True) -> pd.DataFrame:
    """
    Run the full before/after simulation across all strategies.

    Parameters
    ----------
    inject_noise : bool
        If True, runs adversarial noise injection as an additional test.

    Returns
    -------
    DataFrame with results per strategy.
    """
    df = pd.read_csv('data/synthetic_orders.csv')

    print("\n" + "=" * 65)
    print("  KitchenPulse — Signal Pipeline Simulation")
    print("=" * 65)

    # ── Strategy 1: Baseline (naive POS estimate) ────────────────────────
    # The baseline is the naive_kpt_estimate from the dataset
    # (what Zomato's initial model would predict without KitchenPulse)
    
    df['raw_kpt'] = (
        pd.to_datetime(df['for_button_time'])
        - pd.to_datetime(df['order_time'])
    ).dt.total_seconds() / 60

    baseline_mae = mean_absolute_error(df['naive_kpt_estimate'], df['true_kpt_minutes'])

    print(f"\n{'─'*65}")
    print("  RESULTS (lower MAE = better)")
    print(f"{'─'*65}")

    results = []
    results.append(print_strategy_row("1. Baseline (naive POS estimate)", df, 'naive_kpt_estimate', baseline_mae))

    # ── Strategy 2: Denoised FOR (static median, no KLI) ──────────────────
    df_d = flag_rider_proximate_events(df)
    offsets = compute_bias_correction(df_d)
    df_d = apply_correction(df_d, offsets)
    results.append(print_strategy_row("2. Denoised FOR (static median)", df_d, 'corrected_kpt', baseline_mae))

    # ── Strategy 3: EMA Denoised FOR (Feature 4) ──────────────────────────
    df_ema = apply_ema_offset(df_d)
    results.append(print_strategy_row("3. EMA Denoised FOR (adaptive)", df_ema, 'corrected_kpt_ema', baseline_mae))

    # ── Strategy 4: KitchenPulse Full (corrected FOR + KLI) ───────────────
    df_kp = compute_acceptance_latency_zscore(df_d)
    df_kp = compute_concurrent_order_pressure(df_kp)
    df_kp = compute_kli(df_kp)
    df_kp['kli_adjusted_kpt'] = df_kp['corrected_kpt'] * (
        1 + (df_kp['kitchen_load_index'] - 50) / 200
    )
    results.append(print_strategy_row("4. KitchenPulse Full (KLI)", df_kp, 'kli_adjusted_kpt', baseline_mae))

    # ── Strategy 5: KitchenPulse under Adversarial Noise (Feature 1) ──────
    if inject_noise:
        print(f"\n{'─'*65}")
        print("  [Feature 1] Adversarial Noise Test")
        print(f"{'─'*65}")

        df_adv = inject_adversarial_noise(df)
        df_adv['raw_kpt'] = (
            pd.to_datetime(df_adv['for_button_time'])
            - pd.to_datetime(df_adv['order_time'])
        ).dt.total_seconds() / 60

        adv_baseline_mae = mean_absolute_error(
            df_adv['naive_kpt_estimate'], df_adv['true_kpt_minutes']
        )
        print(f"\n  Adversarial baseline MAE: {adv_baseline_mae:.2f}m "
              f"(vs clean baseline: {baseline_mae:.2f}m)")

        df_adv2 = flag_rider_proximate_events(df_adv)
        offsets_adv = compute_bias_correction(df_adv2)
        df_adv2 = apply_correction(df_adv2, offsets_adv)
        df_adv2 = compute_acceptance_latency_zscore(df_adv2)
        df_adv2 = compute_concurrent_order_pressure(df_adv2)
        df_adv2 = compute_kli(df_adv2)
        df_adv2['kli_adjusted_kpt'] = df_adv2['corrected_kpt'] * (
            1 + (df_adv2['kitchen_load_index'] - 50) / 200
        )

        print("\n  Performance under adversarial conditions:")
        results.append(
            print_strategy_row(
                "5a. Baseline (adversarial data)",
                df_adv2, 'naive_kpt_estimate', adv_baseline_mae
            )
        )
        results.append(
            print_strategy_row(
                "5b. KitchenPulse Full (adversarial)",
                df_adv2, 'kli_adjusted_kpt', adv_baseline_mae
            )
        )

    print(f"\n{'='*65}\n")

    results_df = pd.DataFrame(results)
    df_kp.to_csv('data/processed_orders.csv', index=False)
    results_df.to_csv('data/simulation_results_summary.csv', index=False)
    print("Saved: data/processed_orders.csv")
    print("Saved: data/simulation_results_summary.csv")

    return df_kp


if __name__ == '__main__':
    run(inject_noise=True)
