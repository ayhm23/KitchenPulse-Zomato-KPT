"""
pipeline/signal_denoiser.py
---------------------------
FOR button bias detection and correction layer.

Original logic (preserved):
  - flag_rider_proximate_events()  — identifies biased FOR presses
  - compute_bias_correction()      — per-merchant median offset
  - apply_correction()             — subtracts offset from biased timestamps

New additions (v2):
  [Feature 3] Variance-Gated Threshold
    compute_bias_correction() now calculates σ of each merchant's RP-FOR
    delays. The offset is ONLY applied when σ < SIGMA_THRESHOLD, i.e. the
    delay is a consistent habit rather than random kitchen chaos.

  [Feature 4] EMA for Data-Drift Mitigation
    apply_ema_offset() replaces the static median with an online EMA so the
    denoiser adapts as merchants change behavior (e.g., after early dispatch
    training makes them start pressing FOR earlier).
    Formula: EMA_t = α * x_t + (1 - α) * EMA_{t-1}
    α = 0.3  (≈ 7-day half-life at typical order rates)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ──────────────────────────────────────────────────────────────────────────────

# [Feature 3] Only apply the bias offset when σ of RP-FOR delays is below
# this threshold (seconds).  A high σ means chaotic operations, not gaming.
SIGMA_THRESHOLD: float = 180.0   # 3 minutes

# [Feature 4] EMA smoothing factor. 0.3 gives a ~7-day effective memory at
# ~10 orders/day. Lower α = longer memory; higher α = faster adaptation.
EMA_ALPHA: float = 0.3


# ──────────────────────────────────────────────────────────────────────────────
# Original Layer 1 functions (preserved exactly)
# ──────────────────────────────────────────────────────────────────────────────

def flag_rider_proximate_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag FOR events where the rider was already present — these are biased.
    Adds columns:
      - for_bias_flag      (bool)
      - for_delay_seconds  (float) — how late vs actual_ready_time
    """
    df = df.copy()
    df['for_bias_flag'] = df['rider_present_at_press'].astype(bool)

    df['for_delay_seconds'] = (
        pd.to_datetime(df['for_button_time']) -
        pd.to_datetime(df['actual_ready_time'])
    ).dt.total_seconds()

    return df


def compute_bias_correction(df: pd.DataFrame) -> pd.Series:
    """
    [AUGMENTED — Feature 3: Variance-Gated Threshold]

    Per-merchant: learn the bias offset from flagged (rider-proximate) events.
    Original behaviour: return the median delay per merchant as the offset.

    New behaviour:
      1. Also compute σ of each merchant's RP-FOR delays.
      2. If σ >= SIGMA_THRESHOLD, the merchant's operations are chaotic
         (not gaming); set their offset to 0 so we don't over-correct.
      3. If σ < SIGMA_THRESHOLD, the delay is a consistent habit — apply
         the median offset as before.

    Returns a Series indexed by restaurant_id with the effective
    bias_offset_seconds (0 for high-variance merchants).
    """
    biased = df[df['for_bias_flag'] == True]

    stats = biased.groupby('restaurant_id')['for_delay_seconds'].agg(
        median_offset='median',
        sigma='std'
    )

    # Gate: zero out the offset for chaotic merchants
    stats['sigma'] = stats['sigma'].fillna(0.0)
    stats['bias_offset_seconds'] = np.where(
        stats['sigma'] < SIGMA_THRESHOLD,
        stats['median_offset'],
        0.0   # high variance → do not apply correction
    )

    # Log which merchants were gated for transparency
    gated = stats[stats['sigma'] >= SIGMA_THRESHOLD]
    if not gated.empty:
        print(
            f"[Denoiser] Variance gate triggered for "
            f"{len(gated)} merchant(s) with σ ≥ {SIGMA_THRESHOLD}s — "
            f"offset suppressed: {gated.index.tolist()}"
        )

    return stats['bias_offset_seconds']


def apply_correction(df: pd.DataFrame, offsets: pd.Series) -> pd.DataFrame:
    """
    Subtract the learned bias offset from flagged FOR timestamps.
    Adds columns:
      - bias_offset_seconds
      - corrected_for_time
      - corrected_kpt       (minutes, corrected)
      - raw_kpt             (minutes, uncorrected)
    """
    df = df.merge(offsets.rename('bias_offset_seconds'), on='restaurant_id',
                  how='left')
    df['bias_offset_seconds'] = df['bias_offset_seconds'].fillna(0)

    df['corrected_for_time'] = (
        pd.to_datetime(df['for_button_time'])
        - pd.to_timedelta(
            df['bias_offset_seconds'] * df['for_bias_flag'], unit='s'
        )
    )

    df['corrected_kpt'] = (
        df['corrected_for_time'] - pd.to_datetime(df['order_time'])
    ).dt.total_seconds() / 60

    df['raw_kpt'] = (
        pd.to_datetime(df['for_button_time']) - pd.to_datetime(df['order_time'])
    ).dt.total_seconds() / 60

    return df


# ──────────────────────────────────────────────────────────────────────────────
# [Feature 4] EMA-based online offset — replaces static median for drift
# ──────────────────────────────────────────────────────────────────────────────

def apply_ema_offset(
    df: pd.DataFrame,
    alpha: float = EMA_ALPHA,
    sigma_threshold: float = SIGMA_THRESHOLD,
) -> pd.DataFrame:
    """
    [Feature 4: EMA for Data-Drift Mitigation]

    Replaces the static median offset with an Exponential Moving Average
    computed per merchant over the ordered sequence of their RP-FOR events.

    This simulates online learning: as merchants adapt their behaviour (e.g.,
    start pressing FOR earlier once they notice riders arrive sooner), the
    correction factor decays toward zero instead of staying locked at a
    historical median.

    Formula: EMA_t = α * x_t + (1 − α) * EMA_{t-1}
    α = 0.3  by default (reasonable for daily drift at typical order rates)

    Also applies the variance gate from Feature 3: if a merchant's final EMA
    sequence has σ ≥ sigma_threshold, the offset is suppressed.

    Adds column: ema_offset_seconds  (per-row EMA value at that order's time)
    Returns a df with corrected_kpt_ema computed from the EMA offset.
    """
    df = df.copy()
    df['order_time'] = pd.to_datetime(df['order_time'])
    df = df.sort_values(['restaurant_id', 'order_time']).reset_index(drop=True)

    ema_offsets = np.zeros(len(df))

    for restaurant_id, group in df.groupby('restaurant_id'):
        indices = group.index
        biased_mask = group['for_bias_flag'].values
        delays = group['for_delay_seconds'].values

        # Compute σ of this merchant's RP-FOR delays (variance gate)
        rp_delays = delays[biased_mask]
        sigma = float(np.std(rp_delays)) if len(rp_delays) > 1 else 0.0

        if sigma >= sigma_threshold:
            # Chaotic operations: EMA offset stays at 0 for all rows
            ema_offsets[indices] = 0.0
            continue

        # Initialise EMA from first biased event's delay, or 0
        ema_val = float(rp_delays[0]) if len(rp_delays) > 0 else 0.0

        row_emas = np.zeros(len(indices))
        for i, (is_biased, delay) in enumerate(zip(biased_mask, delays)):
            if is_biased:
                ema_val = alpha * delay + (1 - alpha) * ema_val
            row_emas[i] = ema_val

        ema_offsets[indices] = row_emas

    df['ema_offset_seconds'] = ema_offsets

    # Apply the EMA offset to compute an EMA-corrected KPT
    df['corrected_for_time_ema'] = (
        pd.to_datetime(df['for_button_time'])
        - pd.to_timedelta(
            df['ema_offset_seconds'] * df['for_bias_flag'], unit='s'
        )
    )
    df['corrected_kpt_ema'] = (
        df['corrected_for_time_ema'] - df['order_time']
    ).dt.total_seconds() / 60

    return df
