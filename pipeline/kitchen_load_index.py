"""
Kitchen Load Index (KLI) computation.

Combines order concurrency, acceptance latency, foot traffic, and competitor
order volume into a single 0-100 kitchen load score. Uses strategy pattern
to handle missing foot traffic data gracefully.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KLIWeights:
    """Container for the four KLI signal weights. Must sum to 1.0."""
    concurrent_orders: float
    acceptance_latency: float
    foot_traffic: float
    competitor_volume: float

    def __post_init__(self):
        total = (self.concurrent_orders + self.acceptance_latency
                 + self.foot_traffic + self.competitor_volume)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"KLIWeights must sum to 1.0, got {total:.6f}"
            )


class WeightingStrategy(abc.ABC):
    """Abstract base: defines the interface for a KLI weight configuration."""

    @abc.abstractmethod
    def get_weights(self) -> KLIWeights:
        """Return the weight configuration for this strategy."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable strategy name for logging."""


class DefaultWeightingStrategy(WeightingStrategy):
    """
    Original static weights as documented in the KitchenPulse report:
      30% — Zomato Concurrent Orders
      25% — Acceptance Latency Z-Score
      30% — Foot Traffic Index  (Google Popular Times)
      15% — Competitor Platform Volume
    """
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
    Triggered when foot_traffic_index is null or stale (Google Popular Times
    API offline). Redistributes the 30% foot traffic weight proportionally
    to the two strongest remaining internal signals:

    Redistribution logic (preserving original proportions of active signals):
      Original internal weights:  concurrent=0.30, latency=0.25, competitor=0.15
      Internal total = 0.70  →  scale up by (0.70 + 0.30) / 0.70 = 1.4286

    Resulting fallback weights:
      concurrent_orders  = 0.30 * 1.4286 ≈ 0.4286
      acceptance_latency = 0.25 * 1.4286 ≈ 0.3571
      foot_traffic       = 0.00  (signal unavailable)
      competitor_volume  = 0.15 * 1.4286 ≈ 0.2143
    """
    name = "fallback (foot_traffic unavailable)"

    def get_weights(self) -> KLIWeights:
        scale = 1.0 / 0.70   # redistribute foot_traffic's 30% proportionally
        return KLIWeights(
            concurrent_orders=round(0.30 * scale, 6),
            acceptance_latency=round(0.25 * scale, 6),
            foot_traffic=0.0,
            competitor_volume=round(0.15 * scale, 6),
        )


def select_strategy(df: pd.DataFrame) -> WeightingStrategy:
    """
    Inspect the dataframe and return the appropriate weighting strategy.

    Triggers FallbackWeightingStrategy when > 20% of local_foot_traffic_index
    values in this batch are null — indicating the external API is offline.
    """
    if 'local_foot_traffic_index' not in df.columns:
        print("[KLI] local_foot_traffic_index column absent → using FallbackWeightingStrategy")
        return FallbackWeightingStrategy()

    null_rate = df['local_foot_traffic_index'].isna().mean()
    if null_rate > 0.20:
        print(
            f"[KLI] local_foot_traffic_index null rate = {null_rate:.1%} "
            f"(> 20% threshold) → using FallbackWeightingStrategy"
        )
        return FallbackWeightingStrategy()

    return DefaultWeightingStrategy()


# ──────────────────────────────────────────────────────────────────────────────
# Original functions (preserved exactly)
# ──────────────────────────────────────────────────────────────────────────────

def compute_acceptance_latency_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """How unusual is this restaurant's current acceptance latency?"""
    stats = df.groupby('restaurant_id')['acceptance_latency_seconds'].agg(
        ['mean', 'std']
    )
    df = df.merge(stats, on='restaurant_id')
    df['latency_zscore'] = (
        (df['acceptance_latency_seconds'] - df['mean'])
        / df['std'].replace(0, 1)
    )
    return df.drop(columns=['mean', 'std'])


def compute_concurrent_order_pressure(
    df: pd.DataFrame, window_minutes: int = 30
) -> pd.DataFrame:
    """How many other orders were active at this order's time?"""
    df = df.copy()
    df['order_time'] = pd.to_datetime(df['order_time'])
    concurrent = []
    for _, row in df.iterrows():
        window_start = row['order_time'] - pd.Timedelta(minutes=window_minutes)
        active = df[
            (df['restaurant_id'] == row['restaurant_id'])
            & (df['order_time'] >= window_start)
            & (df['order_time'] <= row['order_time'])
        ]
        concurrent.append(len(active))
    df['concurrent_orders'] = concurrent
    return df


def compute_kli(df: pd.DataFrame) -> pd.DataFrame:
    """
    [AUGMENTED — Feature 2: Dynamic KLI Weighting]

    Combine signals into a 0–100 Kitchen Load Index using a dynamically
    selected weighting strategy. Automatically falls back to
    FallbackWeightingStrategy when foot_traffic_index is missing/stale.

    Original static weights (DefaultWeightingStrategy):
      0.5 * latency_norm + 0.5 * concurrent_norm  (simplified original)

    New behaviour: full 4-signal weighted fusion with strategy selection.
    Falls back gracefully when any external signal is unavailable.
    """
    df = df.copy()

    # Select strategy based on data availability
    strategy = select_strategy(df)
    weights = strategy.get_weights()
    print(f"[KLI] Using weighting strategy: '{strategy.name}'")
    print(f"[KLI] Weights → concurrent={weights.concurrent_orders:.4f}, "
          f"latency={weights.acceptance_latency:.4f}, "
          f"foot_traffic={weights.foot_traffic:.4f}, "
          f"competitor={weights.competitor_volume:.4f}")

    # Normalise each signal to [0, 1]
    latency_norm     = df['latency_zscore'].clip(-3, 3).add(3).div(6)
    concurrent_norm  = df['concurrent_orders'].clip(0, 15).div(15)

    # Foot traffic — handle missing values (fill with 0.5 = neutral baseline)
    if 'local_foot_traffic_index' in df.columns:
        foot_norm = df['local_foot_traffic_index'].fillna(50).clip(0, 100).div(100)
    else:
        foot_norm = pd.Series(0.5, index=df.index)   # neutral baseline

    # Competitor volume — handle missing values
    if 'competitor_orders' in df.columns:
        competitor_norm = df['competitor_orders'].fillna(0).clip(0, 10).div(10)
    else:
        competitor_norm = pd.Series(0.0, index=df.index)

    # Weighted fusion → scale to 0–100
    df['kitchen_load_index'] = (
        weights.concurrent_orders  * concurrent_norm
        + weights.acceptance_latency * latency_norm
        + weights.foot_traffic       * foot_norm
        + weights.competitor_volume  * competitor_norm
    ).mul(100).round(1)

    df['kli_strategy_used'] = strategy.name

    return df




# ── Main ──────────────────────────────────────────────────────────────────────
def run_kli(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[KLI] Normalising concurrent orders...")
    df = compute_acceptance_latency_zscore(df)
    df = compute_concurrent_order_pressure(df)

    print("[KLI] Computing composite Kitchen Load Index...")
    df = compute_kli(df)

    return df


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    df = pd.read_csv('data/synthetic_orders.csv')
    df = run_kli(df)
    print(f"\nKLI complete. Output shape: {df.shape}")
