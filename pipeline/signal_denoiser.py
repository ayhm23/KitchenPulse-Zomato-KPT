"""
KitchenPulse — Signal Denoiser
================================
Purpose: Identify and correct the FOR button bias.

Steps:
  1. Flag every FOR event where rider was present at press (RP-FOR)
  2. Learn per-merchant bias offset from flagged events
  3. Apply correction → produce clean corrected_for_kpt
  4. Swap in POS signal where available → produce pos_kpt
  5. Output enriched dataframe ready for KLI layer

Run standalone:
    python pipeline/signal_denoiser.py
"""

import pandas as pd
import numpy as np


# ── Step 1: Flag rider-proximate FOR events ───────────────────────────────────
def flag_rider_proximate(df: pd.DataFrame) -> pd.DataFrame:
    """
    A FOR event is 'rider-proximate' (biased) when the merchant pressed the
    button AFTER the rider had already arrived, or very close to arrival.
    We know this from rider_present_at_press in our dataset.
    In production: derived from GPS proximity (rider within 50m at press time).
    """
    df = df.copy()

    df['for_button_time']    = pd.to_datetime(df['for_button_time'])
    df['actual_ready_time']  = pd.to_datetime(df['actual_ready_time'])
    df['order_time']         = pd.to_datetime(df['order_time'])
    df['rider_arrival_time'] = pd.to_datetime(df['rider_arrival_time'])

    # FOR delay = how long AFTER the rider arrived did the merchant press the button?
    # Using rider_arrival_time because that's what Zomato can actually observe in
    # production via GPS — actual_ready_time is the unknown we're trying to predict.
    df['for_delay_seconds'] = (
        df['for_button_time'] - df['rider_arrival_time']
    ).dt.total_seconds()
    # Note: NOT clipping to 0 here — negative values (merchant pressed before
    # rider arrived) are real signal identifying honest merchants. We clip
    # only when applying the offset to flagged biased events downstream.

    # Raw KPT using the noisy FOR button (what Zomato uses today)
    df['raw_for_kpt'] = (
        df['for_button_time'] - df['order_time']
    ).dt.total_seconds() / 60

    # Flag biased events (rider present at press)
    df['is_rp_for'] = df['rider_present_at_press'].astype(bool)

    return df


# ── Step 2: Learn per-merchant bias offset ────────────────────────────────────
def compute_bias_offsets(df: pd.DataFrame) -> pd.Series:
    """
    For each restaurant, compute the median FOR delay on biased (RP-FOR) events.
    This is how many seconds late the merchant consistently presses the button.
    Restaurants with no biased events get offset = 0.
    """
    biased_events = df[df['is_rp_for'] == True]

    offsets = (
        biased_events
        .groupby('restaurant_id')['for_delay_seconds']
        .median()
        .rename('learned_bias_offset_sec')
    )

    print(f"  [Denoiser] Learned bias offsets for "
          f"{len(offsets)} / {df['restaurant_id'].nunique()} restaurants")
    print(f"  [Denoiser] Median offset across biased restaurants: "
          f"{offsets.mean():.1f} sec ({offsets.mean()/60:.2f} min)")

    return offsets


# ── Step 3: Apply correction to FOR button timestamps ────────────────────────
def apply_for_correction(df: pd.DataFrame, offsets: pd.Series) -> pd.DataFrame:
    """
    Subtract the learned bias offset from biased FOR events.
    Honest merchant events are passed through unchanged.
    Result: corrected_for_kpt — a de-biased version of raw_for_kpt.
    """
    df = df.merge(offsets, on='restaurant_id', how='left')
    df['learned_bias_offset_sec'] = df['learned_bias_offset_sec'].fillna(0)

    # Only subtract offset on flagged (biased) events
    correction_sec = df['learned_bias_offset_sec'] * df['is_rp_for'].astype(int)

    corrected_for_time = df['for_button_time'] - pd.to_timedelta(correction_sec, unit='s')

    df['corrected_for_kpt'] = (
        corrected_for_time - df['order_time']
    ).dt.total_seconds() / 60

    df['corrected_for_kpt'] = df['corrected_for_kpt'].clip(lower=1)

    return df


# ── Step 4: POS-based KPT (clean proposed signal) ────────────────────────────
def compute_pos_kpt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use pos_ticket_cleared_time as the ready signal instead of FOR button.
    POS systems record when the kitchen closes the ticket — no rider bias.
    This is the core of KitchenPulse's proposed architecture.
    """
    df['pos_ticket_cleared_time'] = pd.to_datetime(df['pos_ticket_cleared_time'])

    df['pos_kpt'] = (
        df['pos_ticket_cleared_time'] - df['order_time']
    ).dt.total_seconds() / 60

    df['pos_kpt'] = df['pos_kpt'].clip(lower=1)

    # POS signal error vs ground truth
    df['pos_signal_error'] = (df['pos_kpt'] - df['true_kpt_minutes']).abs()

    return df


# ── Step 5: Signal quality comparison ────────────────────────────────────────
def print_signal_quality(df: pd.DataFrame):
    raw_mae       = (df['raw_for_kpt']      - df['true_kpt_minutes']).abs().mean()
    corrected_mae = (df['corrected_for_kpt'] - df['true_kpt_minutes']).abs().mean()
    pos_mae       = (df['pos_kpt']           - df['true_kpt_minutes']).abs().mean()

    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │         SIGNAL QUALITY COMPARISON (MAE)         │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  Raw FOR button (Zomato today)   : {raw_mae:6.3f} min  │")
    print(f"  │  Corrected FOR (de-biased)       : {corrected_mae:6.3f} min  │")
    print(f"  │  POS ticket signal (proposed)    : {pos_mae:6.3f} min  │")
    print("  └─────────────────────────────────────────────────┘")
    print(f"\n  FOR→Corrected improvement : {(raw_mae-corrected_mae)/raw_mae*100:.1f}%")
    print(f"  FOR→POS improvement       : {(raw_mae-pos_mae)/raw_mae*100:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_denoiser(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Denoiser] Flagging rider-proximate FOR events...")
    df = flag_rider_proximate(df)

    print("[Denoiser] Learning per-merchant bias offsets...")
    offsets = compute_bias_offsets(df)

    print("[Denoiser] Applying correction...")
    df = apply_for_correction(df, offsets)

    print("[Denoiser] Computing POS-based KPT signal...")
    df = compute_pos_kpt(df)

    print_signal_quality(df)

    return df


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    df = pd.read_csv('data/synthetic_orders.csv')
    df = run_denoiser(df)
    print(f"\nDenoiser complete. Output shape: {df.shape}")
