"""
KitchenPulse — Synthetic Data Generator (v2)
=============================================
Fixes applied:
  1. New proposed signal columns added (POS, foot traffic, competitor orders)
  2. naive_kpt now reflects Zomato's model trained on biased FOR labels
  3. Rolling-window Zomato concurrent order counter per restaurant
  4. actual_rider_wait_minutes computed directly as core success metric

Output: data/synthetic_orders.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Config ───────────────────────────────────────────────────────────────────
N_RESTAURANTS   = 50
N_DAYS          = 30
BASE_DATE       = datetime(2025, 1, 1)
CONCURRENCY_WINDOW_MIN = 15   # rolling window for Zomato concurrent orders


# ── 1. Restaurant Master ─────────────────────────────────────────────────────
def build_restaurants(n):
    restaurant_names = [
        "Spice Junction", "Biryani Bros", "The Curry House", "Mumbai Bites",
        "Delhi Darbar", "Taste of Bengal", "Southern Spice", "Punjabi Tadka",
        "Coastal Kitchen", "Street Food Co", "Masala Twist", "Dosa Palace",
        "Chaat Corner", "Tandoor Express", "Royal Feast", "Kebab Station",
        "Thali World", "Noodle Nook", "Wrap & Roll", "The Lunch Box",
        "Grill Master", "Curry Leaf", "Flavour Town", "Saffron Nights",
        "Momo Magic", "Pav Bhaji Point", "The Idli Shop", "Quick Bites",
        "Hunger Fix", "Urban Dhaba", "Spicy Wok", "The Roti Hub",
        "Tikka Trails", "Bowl of India", "Frankie Fort", "Samosa House",
        "Rice Bowl", "The Dal Kitchen", "Chaipani", "Snack Attack",
        "Burger Barn", "Pizza Peak", "Sandwich Stop", "Juice Junction",
        "Taco Treats", "Waffle World", "Crepe Cafe", "The Dessert Lab",
        "Sweet Tooth", "Chill & Grill"
    ]
    records = []
    for i in range(n):
        tier = random.choices(['T1', 'T2', 'T3'], weights=[0.1, 0.4, 0.5])[0]
        honest = random.random() > 0.40   # 60% merchants are biased (delay FOR button)
        base_kpt = random.randint(10, 35)

        # FIX 2: Pre-compute this merchant's FOR button bias offset
        # Dishonest merchants press FOR ~8–20 min late on average
        if honest:
            bias_offset_min = np.random.uniform(0, 1.0)   # trivial noise for honest ones
        else:
            bias_offset_min = np.random.uniform(8.0, 20.0)

        # FIX 2: Zomato's model was TRAINED on corrupted FOR labels.
        # So its learned KPT for this restaurant is inflated by the bias.
        # naive_kpt_base ≈ base_kpt + absorbed bias
        naive_kpt_base = base_kpt + bias_offset_min * 0.85  # model partially absorbs the bias

        records.append({
            'restaurant_id'    : f'R{str(i).zfill(3)}',
            'name'             : restaurant_names[i % len(restaurant_names)],
            'tier'             : tier,
            'base_kpt_minutes' : base_kpt,
            'honest_merchant'  : honest,
            'bias_offset_min'  : round(bias_offset_min, 2),
            'naive_kpt_base'   : round(naive_kpt_base, 2),   # what Zomato "thinks" base KPT is
        })
    return records


# ── 2. Hourly demand profile ──────────────────────────────────────────────────
HOUR_WEIGHTS = [
    0.3, 0.2, 0.1, 0.1, 0.1, 0.2,   # 00–05  (dead hours)
    0.5, 1.0, 1.5, 1.2, 0.8, 0.9,   # 06–11  (breakfast / morning)
    2.0, 1.8, 1.2, 0.9, 0.8, 1.0,   # 12–17  (lunch + afternoon)
    1.4, 2.2, 2.0, 1.6, 1.0, 0.5,   # 18–23  (dinner rush)
]

def orders_this_day(tier):
    """Number of Zomato orders placed on this restaurant today."""
    base = {'T1': (20, 50), 'T2': (8, 25), 'T3': (2, 12)}[tier]
    return random.randint(*base)


# ── 3. FIX 1 — New proposed signal helpers ────────────────────────────────────
def simulate_pos_cleared_time(actual_ready_time, tier):
    """
    POS ticket-cleared timestamp — close to actual_ready_time.
    T1 restaurants have tighter POS integration (±20 s).
    T2/T3 are noisier (±60 s) but still far better than FOR button.
    """
    noise_sec = {'T1': 20, 'T2': 60, 'T3': 90}[tier]
    delta = np.random.normal(0, noise_sec)
    return actual_ready_time + timedelta(seconds=delta)

def simulate_foot_traffic_index(hour, day_of_week, restaurant_popularity):
    """
    0–100 score mimicking Google Popular Times.
    Peaks at lunch (12–14) and dinner (19–21).
    Weekend multiplier applied.
    """
    base = HOUR_WEIGHTS[hour] / max(HOUR_WEIGHTS) * 100
    weekend_bump = 1.25 if day_of_week >= 5 else 1.0
    popularity_factor = restaurant_popularity           # restaurant-specific scalar (0.7–1.3)
    noise = np.random.normal(0, 5)
    return float(np.clip(base * weekend_bump * popularity_factor + noise, 0, 100))

def simulate_competitor_orders(hidden_load, hour):
    """
    Competitor platform (Swiggy etc.) order count in last 15 min.
    Correlated with hidden_load but not identical — adds independent signal value.
    """
    base = max(0, hidden_load * 0.6 + np.random.normal(0, 1.5))
    # Dinner spike on competitor apps too
    if 19 <= hour <= 22:
        base *= 1.3
    return int(np.clip(round(base), 0, 20))


# ── 4. Main generation loop ──────────────────────────────────────────────────
def generate_orders(restaurants):
    all_orders = []

    for r in restaurants:
        popularity = np.random.uniform(0.7, 1.3)   # per-restaurant foot traffic scalar

        for day in range(N_DAYS):
            date = BASE_DATE + timedelta(days=day)
            day_of_week = date.weekday()
            n_orders = orders_this_day(r['tier'])

            for _ in range(n_orders):
                hour   = random.choices(range(24), weights=HOUR_WEIGHTS)[0]
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                order_time = date.replace(hour=hour, minute=minute, second=second)

                # ── Hidden load (dine-in + competitor platforms — invisible to Zomato) ──
                peak_hour = (12 <= hour <= 14) or (19 <= hour <= 22)
                hidden_load = max(0, np.random.normal(6 if peak_hour else 2, 2))

                # ── FIX 3: Zomato concurrent orders computed after full dataset built ──
                # Placeholder — filled in post-generation via rolling window
                zomato_concurrent = 0   # updated below

                # ── True KPT — affected by hidden load (concurrency will be added later) ──
                # Will be recalculated after concurrency is known
                true_kpt = (
                    r['base_kpt_minutes']
                    + hidden_load * 1.5
                    + np.random.normal(0, 2)
                )
                true_kpt = max(5, true_kpt)
                actual_ready_time = order_time + timedelta(minutes=true_kpt)

                # ── FIX 2: Naive KPT (what Zomato's biased model predicts) ──
                naive_kpt = max(5, r['naive_kpt_base'] + np.random.normal(0, 2.5))

                # Rider dispatched based on naive_kpt
                rider_arrival_time = order_time + timedelta(minutes=naive_kpt)

                # ── FOR button signal (noisy) ──
                if r['honest_merchant']:
                    for_button_time = actual_ready_time + timedelta(
                        seconds=np.random.normal(0, 30)
                    )
                    rider_present_at_press = False
                else:
                    # Merchant waits for rider before pressing
                    for_button_time = (
                        max(actual_ready_time, rider_arrival_time)
                        + timedelta(seconds=random.randint(15, 90))
                    )
                    rider_present_at_press = (rider_arrival_time <= actual_ready_time
                                              + timedelta(minutes=2))

                # ── Acceptance latency (high when kitchen is slammed) ──
                acceptance_latency = max(5, np.random.normal(
                    20 + hidden_load * 4, 5
                ))

                # ── FIX 1: New proposed signals ──────────────────────────────
                pos_ticket_cleared_time   = simulate_pos_cleared_time(actual_ready_time, r['tier'])
                local_foot_traffic_index  = simulate_foot_traffic_index(hour, day_of_week, popularity)
                competitor_platform_orders = simulate_competitor_orders(hidden_load, hour)

                # ── FIX 4: Rider wait time (core success metric) ──────────────
                # Positive = rider arrived and had to wait; 0 = food wasn't ready on rider arrival
                rider_wait_seconds = (actual_ready_time - rider_arrival_time).total_seconds()
                actual_rider_wait_minutes = max(0, rider_wait_seconds / 60)

                # ── Naive KPT error (baseline error metric) ───────────────────
                naive_kpt_error = abs(naive_kpt - true_kpt)

                all_orders.append({
                    # ── Identity ──────────────────────────────────────────────
                    'restaurant_id'              : r['restaurant_id'],
                    'restaurant_name'            : r['name'],
                    'tier'                       : r['tier'],
                    'honest_merchant'            : r['honest_merchant'],
                    'day_of_week'                : day_of_week,
                    'hour_of_day'                : hour,

                    # ── Timestamps ────────────────────────────────────────────
                    'order_time'                 : order_time,
                    'actual_ready_time'          : actual_ready_time,
                    'for_button_time'            : for_button_time,
                    'rider_arrival_time'         : rider_arrival_time,
                    'pos_ticket_cleared_time'    : pos_ticket_cleared_time,   # FIX 1

                    # ── Ground truth (simulation only) ────────────────────────
                    'base_kpt_minutes'           : r['base_kpt_minutes'],
                    'true_kpt_minutes'           : round(true_kpt, 3),
                    'hidden_load'                : round(hidden_load, 3),
                    'rider_present_at_press'     : rider_present_at_press,
                    'merchant_bias_offset_min'   : r['bias_offset_min'],

                    # ── Zomato-visible signals (existing today) ───────────────
                    'naive_kpt_estimate'         : round(naive_kpt, 3),       # FIX 2
                    'acceptance_latency_seconds' : round(acceptance_latency, 2),
                    'zomato_concurrent_orders'   : zomato_concurrent,         # FIX 3 (updated below)

                    # ── Proposed new signals (KitchenPulse architecture) ──────
                    'local_foot_traffic_index'   : round(local_foot_traffic_index, 2),  # FIX 1
                    'competitor_platform_orders' : competitor_platform_orders,           # FIX 1

                    # ── Success metrics ───────────────────────────────────────
                    'actual_rider_wait_minutes'  : round(actual_rider_wait_minutes, 3), # FIX 4
                    'naive_kpt_error'            : round(naive_kpt_error, 3),           # FIX 4
                })

    return pd.DataFrame(all_orders)


# ── 5. FIX 3: Post-generation rolling concurrency counter ────────────────────
def add_zomato_concurrency(df):
    """
    Efficient rolling concurrency counter per restaurant using
    sliding window (two-pointer technique). Then recalculates dependent fields.
    """

    df = df.sort_values(['restaurant_id', 'order_time']).reset_index(drop=True)
    df['order_time'] = pd.to_datetime(df['order_time'])
    window = pd.Timedelta(minutes=CONCURRENCY_WINDOW_MIN)

    concurrency = np.zeros(len(df), dtype=int)

    # Process each restaurant independently
    for restaurant_id, group in df.groupby('restaurant_id', sort=False):
        indices = group.index
        times = group['order_time'].values

        left = 0
        for right in range(len(times)):
            while times[right] - times[left] > window:
                left += 1

            # exclude current order
            concurrency[indices[right]] = right - left

    df['zomato_concurrent_orders'] = concurrency

    # Recalculate true_kpt using correct concurrency
    df['true_kpt_minutes'] = (
        df['base_kpt_minutes']
        + df['hidden_load'] * 1.5
        + df['zomato_concurrent_orders'] * 0.8
        + np.random.normal(0, 0.5, len(df))
    ).clip(lower=5).round(3)

    # Recalculate actual_ready_time
    df['actual_ready_time'] = df['order_time'] + pd.to_timedelta(
        df['true_kpt_minutes'], unit='m'
    )

    # Recalculate rider wait
    wait_sec = (
        pd.to_datetime(df['actual_ready_time']) -
        pd.to_datetime(df['rider_arrival_time'])
    ).dt.total_seconds()

    df['actual_rider_wait_minutes'] = wait_sec.clip(lower=0).div(60).round(3)

    # Recalculate naive error
    df['naive_kpt_error'] = (
        df['naive_kpt_estimate'] - df['true_kpt_minutes']
    ).abs().round(3)

    return df

# ── 6. Summary stats ─────────────────────────────────────────────────────────
def print_summary(df):
    print("\n" + "="*60)
    print("  KitchenPulse — Synthetic Dataset Summary")
    print("="*60)
    print(f"  Total orders          : {len(df):,}")
    print(f"  Restaurants           : {df['restaurant_id'].nunique()}")
    print(f"  Date range            : {df['order_time'].min().date()} → {df['order_time'].max().date()}")
    print(f"\n  --- Baseline Problem Metrics ---")
    print(f"  Avg true KPT          : {df['true_kpt_minutes'].mean():.2f} min")
    print(f"  Avg naive KPT error   : {df['naive_kpt_error'].mean():.2f} min")
    print(f"  Avg rider wait time   : {df['actual_rider_wait_minutes'].mean():.2f} min")
    print(f"  Orders w/ rider wait >5 min : {(df['actual_rider_wait_minutes'] > 5).mean()*100:.1f}%")
    print(f"\n  --- Signal Quality Check ---")
    biased = df[df['honest_merchant'] == False]
    honest = df[df['honest_merchant'] == True]
    for_delay_biased = (
        pd.to_datetime(biased['for_button_time']) -
        pd.to_datetime(biased['actual_ready_time'])
    ).dt.total_seconds().div(60).mean()
    for_delay_honest = (
        pd.to_datetime(honest['for_button_time']) -
        pd.to_datetime(honest['actual_ready_time'])
    ).dt.total_seconds().div(60).mean()
    print(f"  Avg FOR delay (biased merchants)  : +{for_delay_biased:.2f} min")
    print(f"  Avg FOR delay (honest merchants)  : +{for_delay_honest:.2f} min")
    pos_error = (
        pd.to_datetime(df['pos_ticket_cleared_time']) -
        pd.to_datetime(df['actual_ready_time'])
    ).dt.total_seconds().div(60).abs().mean()
    print(f"  Avg POS signal error (absolute)   :  {pos_error:.2f} min  ← new signal accuracy")
    print(f"\n  --- New Signals Distribution ---")
    print(f"  Avg foot traffic index : {df['local_foot_traffic_index'].mean():.1f} / 100")
    print(f"  Avg competitor orders  : {df['competitor_platform_orders'].mean():.1f}")
    print(f"  Avg Zomato concurrent  : {df['zomato_concurrent_orders'].mean():.1f}")
    print("="*60)


# ── 7. Run ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    print("Building restaurant profiles...")
    restaurants = build_restaurants(N_RESTAURANTS)

    print("Generating orders...")
    df = generate_orders(restaurants)

    print("Computing rolling concurrency window (this takes ~30s)...")
    df = add_zomato_concurrency(df)

    output_path = 'data/synthetic_orders.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} rows → {output_path}")

    print_summary(df)

    # Also save restaurant master for reference
    pd.DataFrame(restaurants).to_csv('data/restaurants.csv', index=False)
    print("\nRestaurant master saved → data/restaurants.csv")