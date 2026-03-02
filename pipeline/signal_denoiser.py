# pipeline/signal_denoiser.py
"""
FOR button bias detection and correction.

This module exposes a clean wrapper `run_denoiser(df)` used by analysis and
simulation scripts. Internals are split into testable functions:

- flag_rider_proximate    : mark rider-proximate FOR presses and compute delays (no leakage)
- compute_bias_offsets    : compute per-merchant median offset + variance gate
- apply_for_correction    : apply the per-merchant offset to produce corrected_for_kpt and raw_kpt
- compute_pos_kpt         : derive POS-based KPT when POS timestamps exist
- apply_ema_offset        : adaptive EMA drift mitigation
- print_signal_quality    : lightweight reporting about corrections applied
- run_denoiser            : wrapper that executes the full pipeline

Important: for_delay_seconds is computed vs. rider_arrival_time (NO data leakage).
actual_ready_time is NEVER used as a pipeline input — only for post-hoc evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# Only apply bias correction when delay is consistent (not chaotic)
SIGMA_THRESHOLD = 180.0  # seconds (3 min threshold for variance)

# EMA smoothing for adaptive denoising
EMA_ALPHA = 0.3


def flag_rider_proximate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Rider-Proximate FOR presses and compute the FOR delay relative to
    the rider arrival time (no access to ground-truth actual_ready_time is used).

    Adds:
      - for_bias_flag (bool): True when rider was present at or before FOR press.
      - for_delay_seconds (float|NaN): (for_button_time - rider_arrival_time).seconds
    """
    df = df.copy()

    if 'for_button_time' in df.columns:
        df['for_button_time'] = pd.to_datetime(df['for_button_time'], errors='coerce')
    if 'rider_arrival_time' in df.columns:
        df['rider_arrival_time'] = pd.to_datetime(df['rider_arrival_time'], errors='coerce')

    # Use dataset flag if present, else infer from timestamps
    if 'rider_present_at_press' in df.columns:
        df['for_bias_flag'] = df['rider_present_at_press'].astype(bool)
    else:
        df['for_bias_flag'] = (
            df['rider_arrival_time'].notna()
            & df['for_button_time'].notna()
            & (df['rider_arrival_time'] <= df['for_button_time'])
        )

    # Delay relative to rider arrival — no leakage from actual_ready_time
    df['for_delay_seconds'] = np.nan
    mask = df['for_button_time'].notna() & df['rider_arrival_time'].notna()
    df.loc[mask, 'for_delay_seconds'] = (
        df.loc[mask, 'for_button_time'] - df.loc[mask, 'rider_arrival_time']
    ).dt.total_seconds()

    return df


def compute_bias_offsets(df: pd.DataFrame) -> pd.Series:
    """
    Compute per-merchant bias offsets (seconds) using only rider-proximate FOR events.

    Returns:
      Series indexed by restaurant_id -> bias_offset_seconds

    Behavior:
      - For each merchant, compute median(for_delay_seconds) and sigma.
      - If sigma < SIGMA_THRESHOLD -> use median as offset; else offset = 0 (gate).
    """
    rp = df[df.get('for_bias_flag', pd.Series(False, index=df.index)) == True].copy()
    rp = rp[rp['for_delay_seconds'].notna()]

    if rp.empty:
        return pd.Series(dtype=float, name='bias_offset_seconds')

    stats = rp.groupby('restaurant_id')['for_delay_seconds'].agg(
        median_offset='median',
        sigma='std'
    )

    stats['sigma'] = stats['sigma'].fillna(0.0)
    stats['bias_offset_seconds'] = np.where(
        stats['sigma'] < SIGMA_THRESHOLD,
        stats['median_offset'],
        0.0
    )

    gated = stats[stats['sigma'] >= SIGMA_THRESHOLD]
    if not gated.empty:
        print(f"[signal_denoiser] Skipping correction for {len(gated)} merchants (high variance).")

    return stats['bias_offset_seconds']


def apply_for_correction(df: pd.DataFrame, offsets: pd.Series) -> pd.DataFrame:
    """
    Apply bias offsets to compute corrected FOR timestamps and KPTs.

    FIX: column is now named 'corrected_for_kpt' (was 'corrected_kpt') to match
    all downstream consumers: run_simulation.py, kitchen_load_index.py,
    correlation_analysis.py, robustness_tests.py.

    Adds/Preserves columns:
      - bias_offset_seconds (per-row)
      - corrected_for_time
      - corrected_for_kpt  (minutes)  ← canonical name used everywhere downstream
      - raw_kpt            (minutes)
    """
    df = df.copy()

    if offsets is None or offsets.empty:
        df['bias_offset_seconds'] = 0.0
    else:
        offsets = offsets.rename('bias_offset_seconds')
        df = df.merge(offsets, on='restaurant_id', how='left')
        df['bias_offset_seconds'] = df['bias_offset_seconds'].fillna(0.0)

    df['for_button_time'] = pd.to_datetime(df['for_button_time'], errors='coerce')
    df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')

    # Apply offset only to rider-proximate events
    bias_seconds = df['bias_offset_seconds'] * df['for_bias_flag'].astype(int)

    df['corrected_for_time'] = df['for_button_time'] - pd.to_timedelta(bias_seconds, unit='s')

    # corrected_for_kpt: minutes from order_time to corrected_for_time
    df['corrected_for_kpt'] = np.nan
    mask = df['corrected_for_time'].notna() & df['order_time'].notna()
    df.loc[mask, 'corrected_for_kpt'] = (
        df.loc[mask, 'corrected_for_time'] - df.loc[mask, 'order_time']
    ).dt.total_seconds() / 60.0

    # raw_kpt: minutes from order_time to as-pressed for_button_time (uncorrected)
    df['raw_kpt'] = np.nan
    mask2 = df['for_button_time'].notna() & df['order_time'].notna()
    df.loc[mask2, 'raw_kpt'] = (
        df.loc[mask2, 'for_button_time'] - df.loc[mask2, 'order_time']
    ).dt.total_seconds() / 60.0

    return df


def compute_pos_kpt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute POS-based KPT (when available).

    Adds:
      - pos_kpt (minutes) or NaN if timestamps missing
    """
    df = df.copy()
    if 'pos_ticket_cleared_time' not in df.columns:
        df['pos_kpt'] = np.nan
        return df

    df['pos_ticket_cleared_time'] = pd.to_datetime(df['pos_ticket_cleared_time'], errors='coerce')
    df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')

    df['pos_kpt'] = np.nan
    mask = df['pos_ticket_cleared_time'].notna() & df['order_time'].notna()
    df.loc[mask, 'pos_kpt'] = (
        df.loc[mask, 'pos_ticket_cleared_time'] - df.loc[mask, 'order_time']
    ).dt.total_seconds() / 60.0

    return df


def apply_ema_offset(
    df: pd.DataFrame,
    alpha: float = EMA_ALPHA,
    sigma_threshold: float = SIGMA_THRESHOLD,
) -> pd.DataFrame:
    """
    Apply per-merchant EMA offsets over ordered RP-FOR events.
    If merchant's RP-FOR delays have sigma >= sigma_threshold the offsets are suppressed.

    Produces:
      - ema_offset_seconds
      - corrected_for_time_ema
      - corrected_kpt_ema  (minutes)
    """
    df = df.copy()
    df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')
    df = df.sort_values(['restaurant_id', 'order_time']).reset_index(drop=True)

    n = len(df)
    ema_offsets = np.zeros(n, dtype=float)

    for restaurant_id, group in df.groupby('restaurant_id'):
        indices = group.index
        biased_mask = group.get('for_bias_flag', pd.Series(False, index=group.index)).values
        delays = group.get('for_delay_seconds', pd.Series(np.nan, index=group.index)).values

        try:
            rp_delays = delays[biased_mask & ~np.isnan(delays)]
        except Exception:
            rp_delays = delays[biased_mask]

        sigma = float(np.std(rp_delays)) if len(rp_delays) > 1 else 0.0

        if sigma >= sigma_threshold:
            ema_offsets[indices] = 0.0
            continue

        ema_val = float(rp_delays[0]) if len(rp_delays) > 0 else 0.0

        row_emas = np.zeros(len(indices), dtype=float)
        for i, (is_biased, delay) in enumerate(zip(biased_mask, delays)):
            if is_biased and not np.isnan(delay):
                ema_val = alpha * float(delay) + (1.0 - alpha) * ema_val
            row_emas[i] = ema_val

        ema_offsets[indices] = row_emas

    df['ema_offset_seconds'] = ema_offsets

    bias_seconds = df['ema_offset_seconds'] * df.get(
        'for_bias_flag', pd.Series(0, index=df.index)
    ).astype(int)
    df['corrected_for_time_ema'] = (
        pd.to_datetime(df['for_button_time'], errors='coerce')
        - pd.to_timedelta(bias_seconds, unit='s')
    )

    df['corrected_kpt_ema'] = np.nan
    mask = df['corrected_for_time_ema'].notna() & df['order_time'].notna()
    df.loc[mask, 'corrected_kpt_ema'] = (
        df.loc[mask, 'corrected_for_time_ema'] - df.loc[mask, 'order_time']
    ).dt.total_seconds() / 60.0

    return df


def print_signal_quality(df: pd.DataFrame, offsets: Optional[pd.Series] = None) -> None:
    """Print a short summary about the denoiser's output."""
    try:
        total_orders = len(df)
        merchants = df['restaurant_id'].nunique() if 'restaurant_id' in df.columns else 'N/A'
        corrected_orders = int(df.get('for_bias_flag', pd.Series([], dtype=bool)).sum())
        offset_median = float(offsets.median()) / 60.0 if (offsets is not None and not offsets.empty) else 0.0
        gated_count = 0
        if offsets is not None and not offsets.empty:
            gated_count = int((offsets == 0.0).sum())

        print("[signal_denoiser] Summary:")
        print(f"  Orders processed       : {total_orders}")
        print(f"  Unique merchants       : {merchants}")
        print(f"  Rider-proximate presses: {corrected_orders}")
        print(f"  Median bias offset     : {offset_median:.2f} min")
        print(f"  Merchants gated (σ≥{SIGMA_THRESHOLD}s): {gated_count}")
    except Exception:
        pass


def run_denoiser(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full denoiser pipeline wrapper.

    Returns an augmented DataFrame with:
      corrected_for_kpt, corrected_kpt_ema, pos_kpt, raw_kpt,
      ema_offset_seconds, for_bias_flag, for_delay_seconds, bias_offset_seconds.

    EVAL-ONLY columns (do NOT use as training features):
      actual_ready_time  — ground truth, only used for MAE / rider wait computation.

    Steps:
      1. flag_rider_proximate  — no leakage; uses rider_arrival_time only
      2. compute_bias_offsets  — per-merchant median + variance gate
      3. apply_for_correction  — produces corrected_for_kpt, raw_kpt
      4. compute_pos_kpt       — derives pos_kpt from POS timestamps
      5. apply_ema_offset      — produces corrected_kpt_ema
      6. print_signal_quality  — informational summary
    """
    df_work = df.copy()

    df_work = flag_rider_proximate(df_work)
    offsets = compute_bias_offsets(df_work)
    df_work = apply_for_correction(df_work, offsets)
    df_work = compute_pos_kpt(df_work)
    df_work = apply_ema_offset(df_work)
    print_signal_quality(df_work, offsets=offsets)

    return df_work
