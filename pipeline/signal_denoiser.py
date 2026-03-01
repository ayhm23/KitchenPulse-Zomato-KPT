"""
FOR button bias detection and correction.

Detects when merchants press the food-ready button after the rider arrives
(rider-proximate bias) and corrects for it. Also handles adaptive smoothing
for merchants who change their behavior over time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Only apply bias correction when delay is consistent (not chaotic)
SIGMA_THRESHOLD = 180.0  # seconds (3 min threshold for variance)

# EMA smoothing for adaptive denoising
EMA_ALPHA = 0.3


def flag_rider_proximate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Identify FOR presses that happened after rider arrived."""
    df = df.copy()
    df['for_bias_flag'] = df['rider_present_at_press'].astype(bool)

    df['for_delay_seconds'] = (
        pd.to_datetime(df['for_button_time']) -
        pd.to_datetime(df['actual_ready_time'])
    ).dt.total_seconds()

    return df


def compute_bias_correction(df: pd.DataFrame) -> pd.Series:
    """Learn per-merchant bias offset, but skip if delays are too variable."""
    biased = df[df['for_bias_flag'] == True]

    stats = biased.groupby('restaurant_id')['for_delay_seconds'].agg(
        median_offset='median',
        sigma='std'
    )

    # Don't apply correction if variance is high (means operations are chaotic)
    stats['sigma'] = stats['sigma'].fillna(0.0)
    stats['bias_offset_seconds'] = np.where(
        stats['sigma'] < SIGMA_THRESHOLD,
        stats['median_offset'],
        0.0
    )

    # Log which merchants had too much variance
    gated = stats[stats['sigma'] >= SIGMA_THRESHOLD]
    if not gated.empty:
        print(f"Skipping correction for {len(gated)} merchants (high variance): {gated.index.tolist()}")

    return stats['bias_offset_seconds']


def apply_correction(df: pd.DataFrame, offsets: pd.Series) -> pd.DataFrame:
    """Apply the bias correction to FOR timestamps."""
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
