# pipeline/kitchen_load_index.py
"""
KitchenPulse — Kitchen Load Index (KLI)
========================================
Combines four signals into a 0–100 kitchen congestion score.
Uses strategy pattern to handle missing foot traffic data gracefully.

Signal sources:
  ┌──────────────────────────────────┬────────────┬──────────────────────────┐
  │ Signal                           │ Source     │ What it captures         │
  ├──────────────────────────────────┼────────────┼──────────────────────────┤
  │ zomato_concurrent_orders         │ Existing   │ Zomato's own load        │
  │ acceptance_latency_seconds       │ Existing   │ Kitchen stress indicator  │
  │ local_foot_traffic_index         │ Proposed   │ Dine-in / offline rush   │
  │ competitor_platform_orders       │ Proposed   │ Swiggy/UberEats load     │
  └──────────────────────────────────┴────────────┴──────────────────────────┘

Default weights:
  KLI = 0.30 * concurrent_norm
      + 0.25 * latency_norm
      + 0.30 * foot_traffic_norm
      + 0.15 * competitor_norm

FIX: Removed compute_acceptance_latency_zscore and compute_concurrent_order_pressure
as public exports — these do not exist. The canonical public API is run_kli() and
apply_kli_to_kpt(). run_simulation.py now calls these instead of internal helpers.

Run standalone:
    python pipeline/kitchen_load_index.py
"""

from __future__ import annotations
import abc
from dataclasses import dataclass
import numpy as np
import pandas as pd


# ── Weight containers ─────────────────────────────────────────────────────────
@dataclass
class KLIWeights:
    """Four KLI signal weights. Must sum to 1.0."""
    concurrent_orders:  float
    acceptance_latency: float
    foot_traffic:       float
    competitor_volume:  float

    def __post_init__(self):
        total = (self.concurrent_orders + self.acceptance_latency
                 + self.foot_traffic + self.competitor_volume)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"KLIWeights must sum to 1.0, got {total:.6f}")


# ── Weighting strategies ──────────────────────────────────────────────────────
class WeightingStrategy(abc.ABC):
    @abc.abstractmethod
    def get_weights(self) -> KLIWeights: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...


class DefaultWeightingStrategy(WeightingStrategy):
    name = "default"

    def get_weights(self) -> KLIWeights:
        return KLIWeights(
            concurrent_orders=0.30,
            acceptance_latency=0.25,
            foot_traffic=0.30,
            competitor_volume=0.15,
        )


class FallbackWeightingStrategy(WeightingStrategy):
    """
    Used when local_foot_traffic_index is missing or stale (> 20% null).
    Redistributes foot traffic's 30% proportionally to remaining signals.
    Scale = 1.0 / 0.70 = 1.4286
    """
    name = "fallback (foot_traffic unavailable)"

    def get_weights(self) -> KLIWeights:
        scale = 1.0 / 0.70
        return KLIWeights(
            concurrent_orders=round(0.30 * scale, 6),
            acceptance_latency=round(0.25 * scale, 6),
            foot_traffic=0.0,
            competitor_volume=round(0.15 * scale, 6),
        )


def select_strategy(df: pd.DataFrame) -> WeightingStrategy:
    if 'local_foot_traffic_index' not in df.columns:
        print("[KLI] local_foot_traffic_index absent → FallbackWeightingStrategy")
        return FallbackWeightingStrategy()
    null_rate = df['local_foot_traffic_index'].isna().mean()
    if null_rate > 0.20:
        print(f"[KLI] foot_traffic null rate {null_rate:.1%} > 20% → FallbackWeightingStrategy")
        return FallbackWeightingStrategy()
    return DefaultWeightingStrategy()


# ── Component normalisers ─────────────────────────────────────────────────────
def normalise_concurrent_orders(df: pd.DataFrame,
                                max_concurrent: int = 15) -> pd.DataFrame:
    df['concurrent_norm'] = (
        df['zomato_concurrent_orders'].clip(0, max_concurrent).div(max_concurrent)
    )
    return df


def normalise_acceptance_latency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise acceptance latency per merchant.

    FIX: Stores the z-score as both 'latency_zscore' (internal) AND
    'acceptance_latency_zscore' (expected by robustness_tests.py required columns check).
    Both columns are identical — aliases kept for compatibility.
    """
    stats = (
        df.groupby('restaurant_id')['acceptance_latency_seconds']
        .agg(['mean', 'std'])
        .rename(columns={'mean': 'lat_mean', 'std': 'lat_std'})
    )
    df = df.merge(stats, on='restaurant_id', how='left')
    df['lat_std'] = df['lat_std'].replace(0, 1)
    zscore = (df['acceptance_latency_seconds'] - df['lat_mean']) / df['lat_std']
    df['latency_zscore'] = zscore              # internal name
    df['acceptance_latency_zscore'] = zscore   # alias expected by robustness_tests.py
    df['latency_norm'] = zscore.clip(-3, 3).add(3).div(6)
    return df.drop(columns=['lat_mean', 'lat_std'])


def normalise_foot_traffic(df: pd.DataFrame) -> pd.DataFrame:
    if 'local_foot_traffic_index' in df.columns:
        df['foot_traffic_norm'] = (
            df['local_foot_traffic_index'].fillna(50).clip(0, 100).div(100)
        )
    else:
        df['foot_traffic_norm'] = 0.5
    return df


def normalise_competitor_orders(df: pd.DataFrame,
                                max_competitor: int = 15) -> pd.DataFrame:
    col = 'competitor_platform_orders'
    if col in df.columns:
        df['competitor_norm'] = df[col].fillna(0).clip(0, max_competitor).div(max_competitor)
    else:
        df['competitor_norm'] = 0.0
    return df


# ── Composite KLI ─────────────────────────────────────────────────────────────
def compute_kli(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    strategy = select_strategy(df)
    weights = strategy.get_weights()
    print(f"[KLI] Strategy: '{strategy.name}'")

    df['kitchen_load_index'] = (
          weights.concurrent_orders  * df['concurrent_norm']
        + weights.acceptance_latency * df['latency_norm']
        + weights.foot_traffic       * df['foot_traffic_norm']
        + weights.competitor_volume  * df['competitor_norm']
    ).mul(100).round(2)

    df['kli_strategy_used'] = strategy.name
    return df


# ── KLI-adjusted KPT with tiered routing ──────────────────────────────────────
def apply_kli_to_kpt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical KLI application — used by both run_simulation.py and run_kli().

    T1 merchants → POS signal as base.
    T2/T3 merchants → corrected_for_kpt as base.
    KLI scaling applied to both tiers.

    factor = 1 + (KLI - 50) / 200
    KLI=100 → +25%  |  KLI=50 → no change  |  KLI=0 → -25%

    FIX: run_simulation.py previously defined a local duplicate apply_tiered_kli()
    with the same logic. That function has been removed; callers should use this one.
    """
    kli_factor = 1 + (df['kitchen_load_index'] - 50) / 200
    base = np.where(
        df['tier'] == 'T1',
        df['pos_kpt'],
        df['corrected_for_kpt']
    )
    df['kli_adjusted_kpt'] = (base * kli_factor).clip(lower=1).round(3)
    return df


# ── Stats ─────────────────────────────────────────────────────────────────────
def print_kli_stats(df: pd.DataFrame):
    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │          KITCHEN LOAD INDEX SUMMARY             │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  Mean KLI               : {df['kitchen_load_index'].mean():6.2f}           │")
    print(f"  │  Std  KLI               : {df['kitchen_load_index'].std():6.2f}           │")
    print(f"  │  Orders KLI > 70 (high) : {(df['kitchen_load_index']>70).mean()*100:5.1f}%           │")
    print(f"  │  Orders KLI < 30 (low)  : {(df['kitchen_load_index']<30).mean()*100:5.1f}%           │")
    if 'true_kpt_minutes' in df.columns:
        print("  ├─────────────────────────────────────────────────┤")
        for col, label in [
            ('concurrent_norm',         'concurrent (norm)  '),
            ('latency_norm',            'latency (norm)     '),
            ('foot_traffic_norm',       'foot traffic (norm)'),
            ('competitor_norm',         'competitor (norm)  '),
            ('kitchen_load_index',      'KLI composite      '),
        ]:
            if col in df.columns:
                r = df[col].corr(df['true_kpt_minutes'])
                print(f"  │  Corr({label}) w/ true KPT : {r:+.3f}  │")
    print("  └─────────────────────────────────────────────────┘")


# ── Main pipeline wrapper ─────────────────────────────────────────────────────
def run_kli(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full KLI pipeline: normalise all signals → compute KLI → apply to KPT.
    This is the public entry point. run_simulation.py should call this,
    not the individual normalise_*() functions directly.
    """
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
    print("[KLI] Applying KLI adjustment (tiered routing)...")
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
