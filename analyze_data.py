"""
KitchenPulse — Quick Dataset Summary
======================================
Prints baseline problem metrics from synthetic_orders.csv.

Note: FOR delay analysis uses rider_arrival_time (not actual_ready_time)
to remain consistent with the production-safe signal denoiser logic.
"""

import pandas as pd
import numpy as np

df = pd.read_csv('data/synthetic_orders.csv')

print('=' * 65)
print('DATASET SUMMARY — synthetic_orders.csv')
print('=' * 65)
print(f'Total orders        : {len(df):,}')
print(f'Restaurants         : {df["restaurant_id"].nunique()}')
print(f'Date range          : {df["order_time"].min()[:10]} → {df["order_time"].max()[:10]}')
print()
print('--- Baseline Problem Metrics ---')
print(f'Avg true KPT        : {df["true_kpt_minutes"].mean():.2f} min')
print(f'Avg naive KPT error : {df["naive_kpt_error"].mean():.2f} min')
print(f'Avg rider wait      : {df["actual_rider_wait_minutes"].mean():.3f} min')
print(f'Rider wait > 5 min  : {(df["actual_rider_wait_minutes"] > 5).mean()*100:.1f}%')
print()
print('--- POS Signal Accuracy ---')
pos_error = (
    pd.to_datetime(df["pos_ticket_cleared_time"]) -
    pd.to_datetime(df["actual_ready_time"])
).dt.total_seconds().div(60).abs().mean()
print(f'Avg POS signal error: {pos_error:.2f} min')
print()
print('--- FOR Button Bias (production-safe: vs rider_arrival_time) ---')
biased = df[df['honest_merchant'] == False]
honest = df[df['honest_merchant'] == True]

# Use rider_arrival_time — this is what the denoiser uses in production
# (actual_ready_time is ground truth unavailable in production)
biased_delay = (
    pd.to_datetime(biased["for_button_time"]) -
    pd.to_datetime(biased["rider_arrival_time"])
).dt.total_seconds().div(60).mean()

honest_delay = (
    pd.to_datetime(honest["for_button_time"]) -
    pd.to_datetime(honest["rider_arrival_time"])
).dt.total_seconds().div(60).mean()

print(f'Biased merchants (n={len(biased):,}): FOR delay vs rider = {biased_delay:+.2f} min')
print(f'Honest merchants (n={len(honest):,}): FOR delay vs rider = {honest_delay:+.2f} min')
print()
print('--- New Signal Stats ---')
print(f'Avg foot traffic index   : {df["local_foot_traffic_index"].mean():.1f} / 100')
print(f'Avg competitor orders    : {df["competitor_platform_orders"].mean():.1f}')
print(f'Avg Zomato concurrent    : {df["zomato_concurrent_orders"].mean():.1f}')
print('=' * 65)
