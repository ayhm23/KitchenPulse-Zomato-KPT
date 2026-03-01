"""
KitchenPulse — Kitchen Load Index (KLI)
========================================
Purpose: Build a 0–100 score representing true kitchen congestion,
         combining BOTH visible Zomato signals AND hidden load proxies.

Signal sources:
  ┌─────────────────────────────────┬────────────┬──────────────────────────┐
  │ Signal                          │ Source     │ What it captures         │
  ├─────────────────────────────────┼────────────┼──────────────────────────┤
  │ zomato_concurrent_orders        │ Existing   │ Zomato's own load        │
  │ acceptance_latency_zscore       │ Existing   │ Kitchen stress indicator  │
  │ local_foot_traffic_index        │ Proposed   │ Dine-in / offline rush   │
  │ competitor_platform_orders      │ Proposed   │ Swiggy/UberEats load     │
  └─────────────────────────────────┴────────────┴──────────────────────────┘

Weights:
  KLI = 0.30 * concurrent_norm
      + 0.25 * latency_norm
      + 0.30 * foot_traffic_norm
      + 0.15 * competitor_norm

Run standalone:
    python pipeline/kitchen_load_index.py
"""

import pandas as pd
import numpy as np


# ── Component 1: Zomato concurrent orders (normalised) ───────────────────────
def normalise_concurrent_orders(df: pd.DataFrame,
                                 max_concurrent: int = 15) -> pd.DataFrame:
    """
    Clip to max_concurrent then scale to [0, 1].
    A restaurant with 15+ active Zomato orders in the last 15 min is at max load.
    """
    df['concurrent_norm'] = (
        df['zomato_concurrent_orders']
        .clip(0, max_concurrent)
        .div(max_concurrent)
    )
    return df


# ── Component 2: Acceptance latency z-score ───────────────────────────────────
def normalise_acceptance_latency(df: pd.DataFrame) -> pd.DataFrame:
    """
    How unusual is this restaurant's acceptance latency right now vs its history?
    High z-score = kitchen overwhelmed, struggling to even acknowledge new orders.
    Clip z-score to [-3, 3] then map to [0, 1].
    """
    stats = (
        df.groupby('restaurant_id')['acceptance_latency_seconds']
        .agg(['mean', 'std'])
        .rename(columns={'mean': 'lat_mean', 'std': 'lat_std'})
    )
    df = df.merge(stats, on='restaurant_id', how='left')
    df['lat_std'] = df['lat_std'].replace(0, 1)   # avoid divide-by-zero

    df['latency_zscore'] = (
        (df['acceptance_latency_seconds'] - df['lat_mean']) / df['lat_std']
    )
    df['latency_norm'] = (
        df['latency_zscore'].clip(-3, 3).add(3).div(6)
    )
    df = df.drop(columns=['lat_mean', 'lat_std'])
    return df


# ── Component 3: Local foot traffic index ─────────────────────────────────────
def normalise_foot_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Already on 0–100 scale from Google Popular Times proxy.
    Divide by 100 to bring to [0, 1].
    High score = packed restaurant = dine-in kitchen pressure Zomato can't see.
    """
    df['foot_traffic_norm'] = df['local_foot_traffic_index'].clip(0, 100).div(100)
    return df


# ── Component 4: Competitor platform orders ───────────────────────────────────
def normalise_competitor_orders(df: pd.DataFrame,
                                  max_competitor: int = 15) -> pd.DataFrame:
    """
    Competitor app orders (Swiggy / UberEats) in same time window.
    Completely invisible to Zomato today — this is hidden kitchen load.
    """
    df['competitor_norm'] = (
        df['competitor_platform_orders']
        .clip(0, max_competitor)
        .div(max_competitor)
    )
    return df


# ── Composite KLI Score ───────────────────────────────────────────────────────
WEIGHTS = {
    'concurrent_norm'    : 0.30,
    'latency_norm'       : 0.25,
    'foot_traffic_norm'  : 0.30,
    'competitor_norm'    : 0.15,
}

def compute_kli(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted combination of all four normalised components → KLI [0, 100].
    """
    df['kitchen_load_index'] = (
        WEIGHTS['concurrent_norm']   * df['concurrent_norm']
      + WEIGHTS['latency_norm']      * df['latency_norm']
      + WEIGHTS['foot_traffic_norm'] * df['foot_traffic_norm']
      + WEIGHTS['competitor_norm']   * df['competitor_norm']
    ).mul(100).round(2)

    return df


# ── KLI-adjusted KPT Estimate ─────────────────────────────────────────────────
def apply_kli_to_kpt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use KLI to scale the clean POS-based KPT signal.

    Logic:
      - KLI = 50  →  no adjustment (kitchen at baseline)
      - KLI > 50  →  KPT scaled up   (busier than expected)
      - KLI < 50  →  KPT scaled down (quieter than expected)

    Scaling factor: 1 + (KLI - 50) / 200
      KLI=100 → factor = 1.25  (+25% KPT)
      KLI=50  → factor = 1.00  (no change)
      KLI=0   → factor = 0.75  (-25% KPT)
    """
    kli_factor = 1 + (df['kitchen_load_index'] - 50) / 200

    # Tiered routing — matches the architecture we proposed:
    # T1 (large chains) → POS ticket signal (accurate, unbiased)
    # T2/T3 (independent restaurants) → de-biased FOR button (no POS available)
    # KLI adjustment applied to both tiers equally on top of whichever base signal
    base_clean_kpt = np.where(
        df['tier'] == 'T1',
        df['pos_kpt'],
        df['corrected_for_kpt']
    )

    df['kli_adjusted_kpt'] = (base_clean_kpt * kli_factor).clip(lower=1).round(3)
    return df


# ── Stats ─────────────────────────────────────────────────────────────────────
def print_kli_stats(df: pd.DataFrame):
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │        KITCHEN LOAD INDEX SUMMARY           │")
    print("  ├─────────────────────────────────────────────┤")
    print(f"  │  Mean KLI                : {df['kitchen_load_index'].mean():6.2f}         │")
    print(f"  │  Std  KLI                : {df['kitchen_load_index'].std():6.2f}         │")
    print(f"  │  Orders KLI > 70 (high)  : {(df['kitchen_load_index']>70).mean()*100:5.1f}%         │")
    print(f"  │  Orders KLI < 30 (low)   : {(df['kitchen_load_index']<30).mean()*100:5.1f}%         │")
    print("  ├─────────────────────────────────────────────┤")

    # Correlation of each component with true KPT
    for col, label in [
        ('concurrent_norm',   'Zomato concurrent (norm)'),
        ('latency_norm',      'Acceptance latency (norm)'),
        ('foot_traffic_norm', 'Foot traffic (norm)     '),
        ('competitor_norm',   'Competitor orders (norm)'),
        ('kitchen_load_index','KLI composite           '),
    ]:
        corr = df[col].corr(df['true_kpt_minutes'])
        print(f"  │  Corr({label}) w/ true KPT: {corr:+.3f}  │")

    print("  └─────────────────────────────────────────────┘")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_kli(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[KLI] Normalising concurrent orders...")
    df = normalise_concurrent_orders(df)

    print("[KLI] Normalising acceptance latency...")
    df = normalise_acceptance_latency(df)

    print("[KLI] Normalising foot traffic index...")
    df = normalise_foot_traffic(df)

    print("[KLI] Normalising competitor order volume...")
    df = normalise_competitor_orders(df)

    print("[KLI] Computing composite Kitchen Load Index...")
    df = compute_kli(df)

    print("[KLI] Applying KLI adjustment to POS-based KPT...")
    df = apply_kli_to_kpt(df)

    print_kli_stats(df)

    return df


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pipeline.signal_denoiser import run_denoiser

    df = pd.read_csv('data/synthetic_orders.csv')
    df = run_denoiser(df)
    df = run_kli(df)
    print(f"\nKLI complete. Output shape: {df.shape}")
