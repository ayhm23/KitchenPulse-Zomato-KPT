"""
KitchenPulse — Master Simulation Runner
=========================================
Purpose: Prove that replacing Zomato's current broken inputs with
         KitchenPulse signals reduces rider wait time and KPT error.

Three prediction strategies compared head-to-head:
  A) BASELINE   — Zomato today: naive_kpt_estimate (trained on biased FOR labels)
  B) DENOISED   — De-biased FOR button + per-merchant bias correction
  C) KITCHENPULSE — POS signal + KLI adjustment (full proposed architecture)

Output metrics (what judges score you on):
  • MAE vs true_kpt_minutes
  • Mean rider wait time
  • % orders with rider wait > 5 min
  • % improvement across all three metrics

Run:
    python simulation/run_simulation.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.signal_denoiser import run_denoiser
from pipeline.kitchen_load_index import run_kli


# ── Helpers ──────────────────────────────────────────────────────────────────
def mae(pred, actual):
    return np.mean(np.abs(pred - actual))

def rider_wait_from_kpt(df, kpt_col):
    """
    Given a KPT prediction column, compute what rider wait would be
    if Zomato dispatched riders based on that prediction.
    Wait = max(0, actual_ready_time - predicted_arrival_time)
    """
    predicted_arrival = df['order_time'] + pd.to_timedelta(df[kpt_col], unit='m')
    wait_sec = (df['actual_ready_time'] - predicted_arrival).dt.total_seconds()
    return wait_sec.clip(lower=0) / 60   # convert to minutes

def pct_long_wait(wait_series, threshold=5):
    return (wait_series > threshold).mean() * 100


# ── Load and process data ─────────────────────────────────────────────────────
def load_and_process():
    print("Loading synthetic_orders.csv...")
    df = pd.read_csv('data/synthetic_orders.csv')

    # Parse all datetime columns
    for col in ['order_time', 'actual_ready_time', 'for_button_time',
                'rider_arrival_time', 'pos_ticket_cleared_time']:
        df[col] = pd.to_datetime(df[col])

    print(f"Loaded {len(df):,} orders\n")

    df = run_denoiser(df)
    df = run_kli(df)

    return df


# ── Strategy definitions ──────────────────────────────────────────────────────
def evaluate_strategies(df):
    results = {}

    # ── Strategy A: Baseline (Zomato today) ──────────────────────────────────
    wait_A = rider_wait_from_kpt(df, 'naive_kpt_estimate')
    results['A_Baseline'] = {
        'label'         : 'Baseline\n(Zomato Today)',
        'short_label'   : 'Baseline',
        'color'         : '#E74C3C',
        'kpt_mae'       : mae(df['naive_kpt_estimate'], df['true_kpt_minutes']),
        'avg_wait'      : wait_A.mean(),
        'pct_long_wait' : pct_long_wait(wait_A),
        'wait_series'   : wait_A,
        'kpt_series'    : df['naive_kpt_estimate'],
    }

    # ── Strategy B: Denoised FOR button ──────────────────────────────────────
    wait_B = rider_wait_from_kpt(df, 'corrected_for_kpt')
    results['B_Denoised'] = {
        'label'         : 'Denoised FOR\n(Bias Corrected)',
        'short_label'   : 'Denoised FOR',
        'color'         : '#F39C12',
        'kpt_mae'       : mae(df['corrected_for_kpt'], df['true_kpt_minutes']),
        'avg_wait'      : wait_B.mean(),
        'pct_long_wait' : pct_long_wait(wait_B),
        'wait_series'   : wait_B,
        'kpt_series'    : df['corrected_for_kpt'],
    }

    # ── Strategy C: KitchenPulse (full system) ────────────────────────────────
    wait_C = rider_wait_from_kpt(df, 'kli_adjusted_kpt')
    results['C_KitchenPulse'] = {
        'label'         : 'KitchenPulse\n(POS + KLI)',
        'short_label'   : 'KitchenPulse',
        'color'         : '#27AE60',
        'kpt_mae'       : mae(df['kli_adjusted_kpt'], df['true_kpt_minutes']),
        'avg_wait'      : wait_C.mean(),
        'pct_long_wait' : pct_long_wait(wait_C),
        'wait_series'   : wait_C,
        'kpt_series'    : df['kli_adjusted_kpt'],
    }

    return results, df


# ── Print results table ───────────────────────────────────────────────────────
def print_results_table(results):
    base = results['A_Baseline']

    print("\n" + "="*70)
    print("  KITCHENPULSE SIMULATION RESULTS")
    print("="*70)
    print(f"  {'Strategy':<28} {'KPT MAE':>8} {'Avg Wait':>10} {'Wait>5m':>10}")
    print("-"*70)

    for key, r in results.items():
        kpt_imp = (base['kpt_mae']   - r['kpt_mae'])   / base['kpt_mae']   * 100
        wt_imp  = (base['avg_wait']  - r['avg_wait'])  / base['avg_wait']  * 100
        lw_imp  = (base['pct_long_wait'] - r['pct_long_wait']) / base['pct_long_wait'] * 100

        if key == 'A_Baseline':
            print(f"  {r['short_label']:<28} {r['kpt_mae']:>7.2f}m {r['avg_wait']:>9.2f}m {r['pct_long_wait']:>9.1f}%")
        else:
            print(f"  {r['short_label']:<28} {r['kpt_mae']:>7.2f}m {r['avg_wait']:>9.2f}m {r['pct_long_wait']:>9.1f}%")
            print(f"  {'  → vs Baseline':<28} {kpt_imp:>+7.1f}% {wt_imp:>+9.1f}% {lw_imp:>+9.1f}%")
    print("="*70)


# ── Charts ────────────────────────────────────────────────────────────────────
def build_charts(results, df):
    os.makedirs('report/figures', exist_ok=True)
    STRATEGIES = list(results.values())
    LABELS     = [s['short_label'] for s in STRATEGIES]
    COLORS     = [s['color']       for s in STRATEGIES]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0F1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    title_kw  = dict(color='white', fontsize=11, fontweight='bold', pad=10)
    label_kw  = dict(color='#AAAAAA', fontsize=9)
    tick_kw   = dict(colors='#AAAAAA', labelsize=8)

    def style_ax(ax):
        ax.set_facecolor('#1A1D27')
        ax.tick_params(axis='x', **tick_kw)
        ax.tick_params(axis='y', **tick_kw)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333344')
        ax.grid(axis='y', color='#333344', linestyle='--', linewidth=0.5, alpha=0.7)
        return ax

    # ── Chart 1: KPT MAE Comparison ──────────────────────────────────────────
    ax1 = style_ax(fig.add_subplot(gs[0, 0]))
    vals = [s['kpt_mae'] for s in STRATEGIES]
    bars = ax1.bar(LABELS, vals, color=COLORS, width=0.5, edgecolor='none')
    ax1.set_title('KPT Prediction Error (MAE)', **title_kw)
    ax1.set_ylabel('Minutes', **label_kw)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.05, f'{v:.2f}m',
                 ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')

    # ── Chart 2: Average Rider Wait ───────────────────────────────────────────
    ax2 = style_ax(fig.add_subplot(gs[0, 1]))
    vals2 = [s['avg_wait'] for s in STRATEGIES]
    bars2 = ax2.bar(LABELS, vals2, color=COLORS, width=0.5, edgecolor='none')
    ax2.set_title('Avg Rider Wait at Pickup', **title_kw)
    ax2.set_ylabel('Minutes', **label_kw)
    for bar, v in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}m',
                 ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')

    # ── Chart 3: % Orders with Long Wait ─────────────────────────────────────
    ax3 = style_ax(fig.add_subplot(gs[0, 2]))
    vals3 = [s['pct_long_wait'] for s in STRATEGIES]
    bars3 = ax3.bar(LABELS, vals3, color=COLORS, width=0.5, edgecolor='none')
    ax3.set_title('% Orders: Rider Wait > 5 Min', **title_kw)
    ax3.set_ylabel('Percentage (%)', **label_kw)
    for bar, v in zip(bars3, vals3):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.2, f'{v:.1f}%',
                 ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')

    # ── Chart 4: Rider Wait Distribution (KDE) ────────────────────────────────
    ax4 = style_ax(fig.add_subplot(gs[1, 0:2]))
    for s in STRATEGIES:
        wait = s['wait_series'].clip(0, 20)
        wait.plot.kde(ax=ax4, color=s['color'], linewidth=2, label=s['short_label'])
    ax4.axvline(5, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax4.text(5.2, ax4.get_ylim()[1]*0.9 if ax4.get_ylim()[1] > 0 else 0.1,
             '5 min\nthreshold', color='white', fontsize=7, alpha=0.7)
    ax4.set_title('Rider Wait Time Distribution', **title_kw)
    ax4.set_xlabel('Wait Time (minutes)', **label_kw)
    ax4.set_ylabel('Density', **label_kw)
    ax4.legend(facecolor='#1A1D27', edgecolor='#333344',
               labelcolor='white', fontsize=8)
    ax4.set_xlim(left=0)

    # ── Chart 5: KLI vs True KPT Scatter ─────────────────────────────────────
    ax5 = style_ax(fig.add_subplot(gs[1, 2]))
    sample = df.sample(min(2000, len(df)), random_state=42)
    sc = ax5.scatter(
        sample['kitchen_load_index'], sample['true_kpt_minutes'],
        c=sample['kitchen_load_index'], cmap='RdYlGn_r',
        alpha=0.4, s=8, linewidths=0
    )
    # Trend line
    z = np.polyfit(sample['kitchen_load_index'], sample['true_kpt_minutes'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(sample['kitchen_load_index'].min(),
                           sample['kitchen_load_index'].max(), 100)
    ax5.plot(x_range, p(x_range), color='white', linewidth=1.5,
             linestyle='--', alpha=0.8)
    corr = df['kitchen_load_index'].corr(df['true_kpt_minutes'])
    ax5.set_title(f'KLI vs True KPT  (r = {corr:.3f})', **title_kw)
    ax5.set_xlabel('Kitchen Load Index', **label_kw)
    ax5.set_ylabel('True KPT (min)', **label_kw)

    # ── Super title ───────────────────────────────────────────────────────────
    fig.suptitle('KitchenPulse — Simulation Results\n'
                 'De-noised Signals + Kitchen Load Index vs Zomato Baseline',
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    out_path = 'report/figures/simulation_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Chart saved → {out_path}")


# ── Tier breakdown ────────────────────────────────────────────────────────────
def tier_breakdown(results, df):
    """Show improvement broken down by merchant tier T1 / T2 / T3."""
    print("\n  IMPROVEMENT BY MERCHANT TIER (KitchenPulse vs Baseline)")
    print("  " + "-"*55)
    print(f"  {'Tier':<6} {'Base MAE':>9} {'KP MAE':>9} {'Improvement':>12} {'Merchants':>10}")
    print("  " + "-"*55)

    for tier in ['T1', 'T2', 'T3']:
        mask = df['tier'] == tier
        sub  = df[mask]
        if len(sub) == 0:
            continue
        base_mae = mae(sub['naive_kpt_estimate'], sub['true_kpt_minutes'])
        kp_mae   = mae(sub['kli_adjusted_kpt'],   sub['true_kpt_minutes'])
        imp      = (base_mae - kp_mae) / base_mae * 100
        n_rest   = sub['restaurant_id'].nunique()
        print(f"  {tier:<6} {base_mae:>8.2f}m {kp_mae:>8.2f}m {imp:>+11.1f}% {n_rest:>10}")

    print("  " + "-"*55)


# ── Save processed dataset ────────────────────────────────────────────────────
def save_processed(df):
    out = 'data/processed_orders.csv'
    # Keep only the columns useful for the PDF / further analysis
    keep = [
        'restaurant_id', 'restaurant_name', 'tier', 'hour_of_day', 'day_of_week',
        'order_time', 'true_kpt_minutes', 'hidden_load', 'base_kpt_minutes',
        'zomato_concurrent_orders', 'acceptance_latency_seconds',
        'for_delay_seconds', 'is_rp_for', 'honest_merchant',
        'raw_for_kpt', 'corrected_for_kpt', 'pos_kpt', 'kli_adjusted_kpt',
        'kitchen_load_index', 'local_foot_traffic_index',
        'competitor_platform_orders', 'naive_kpt_estimate',
        'actual_rider_wait_minutes', 'naive_kpt_error',
        'concurrent_norm', 'latency_norm', 'foot_traffic_norm', 'competitor_norm',
    ]
    df[[c for c in keep if c in df.columns]].to_csv(out, index=False)
    print(f"  Processed dataset saved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load_and_process()

    print("\nEvaluating three strategies...")
    results, df = evaluate_strategies(df)

    print_results_table(results)
    tier_breakdown(results, df)

    print("\nGenerating charts...")
    build_charts(results, df)

    save_processed(df)

    print("\n✅ Simulation complete.")
    print("   → Charts in  : report/figures/simulation_results.png")
    print("   → Data in     : data/processed_orders.csv")
    print("   → Use these numbers directly in your PDF.\n")
