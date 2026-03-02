"""
KitchenPulse — Master Simulation Runner
=========================================
Before vs after comparison across pipeline strategies.

Strategies:
  A) Baseline         — naive_kpt_estimate (Zomato today)
  B) Denoised FOR     — static median bias correction
  C) EMA Denoised     — adaptive EMA correction
  D) KitchenPulse     — corrected signal + KLI (full system)
  E) Adversarial test — KitchenPulse under injected noise

FIX SUMMARY:
  1. Removed nonexistent imports: compute_acceptance_latency_zscore,
     compute_concurrent_order_pressure — these are internal normalisers,
     not public exports. run_kli() handles them internally.
  2. Removed local apply_tiered_kli() — was an exact duplicate of
     apply_kli_to_kpt() in kitchen_load_index.py. Now imports and uses
     the canonical version.
  3. Corrected column reference: corrected_for_kpt (was corrected_kpt).

Run:
    python simulation/run_simulation.py
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.signal_denoiser import (
    flag_rider_proximate,
    compute_bias_offsets,
    apply_for_correction,
    apply_ema_offset,
    compute_pos_kpt,
    run_denoiser,
)
from pipeline.kitchen_load_index import (
    normalise_concurrent_orders,
    normalise_acceptance_latency,
    normalise_foot_traffic,
    normalise_competitor_orders,
    compute_kli,
    apply_kli_to_kpt,   # canonical — replaces local apply_tiered_kli()
    run_kli,
)

ADVERSARIAL_FRACTION = 0.08
RANDOM_SEED = 99
BG = '#0F1117'; PANEL = '#1A1D27'; GRID = '#2A2D3A'
TEXT = '#FFFFFF'; SUB = '#AAAAAA'
RED = '#E74C3C'; AMBER = '#F39C12'; BLUE = '#3498DB'; GREEN = '#27AE60'


def inject_adversarial_noise(df, fraction=ADVERSARIAL_FRACTION, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    df = df.copy()
    n_noisy = int(len(df) * fraction)
    noisy_idx = rng.choice(df.index, size=n_noisy, replace=False)
    type_a = noisy_idx[:n_noisy // 2]
    type_b = noisy_idx[n_noisy // 2:]
    df['adversarial_noise_type'] = 'none'
    if 'competitor_platform_orders' not in df.columns:
        df['competitor_platform_orders'] = 0.0
    df.loc[type_a, 'competitor_platform_orders'] = rng.uniform(50, 200, size=len(type_a))
    df.loc[type_a, 'adversarial_noise_type'] = 'competitor_spike'
    delay_sec = (rng.pareto(a=1.5, size=len(type_b)) * 6 + 18) * 60
    df['for_button_time'] = pd.to_datetime(df['for_button_time'])
    df.loc[type_b, 'for_button_time'] = (
        df.loc[type_b, 'for_button_time']
        + pd.to_timedelta(np.round(delay_sec).astype(int), unit='s')
    )
    df.loc[type_b, 'adversarial_noise_type'] = 'honest_merchant_flip'
    print(f"  [Adversarial] {n_noisy} rows injected: "
          f"{len(type_a)} competitor spikes, {len(type_b)} FOR flips")
    return df


def mae(pred, actual):
    return float(np.mean(np.abs(pred - actual)))


def rider_wait_from_kpt(df, kpt_col):
    # EVAL ONLY — actual_ready_time is ground truth, not a training feature
    pred_arrival = (
        pd.to_datetime(df['order_time'])
        + pd.to_timedelta(df[kpt_col], unit='m')
    )
    wait_sec = (pd.to_datetime(df['actual_ready_time']) - pred_arrival).dt.total_seconds()
    return wait_sec.clip(lower=0) / 60


def pct_long_wait(series, threshold=5.0):
    return float((series > threshold).mean() * 100)


def print_row(label, df, kpt_col, base_mae):
    m = mae(df[kpt_col], df['true_kpt_minutes'])
    wait = rider_wait_from_kpt(df, kpt_col)
    w = wait.mean()
    p5 = pct_long_wait(wait)
    d = (base_mae - m) / base_mae * 100
    print(f"  {label:<44} MAE={m:.2f}m  Wait={w:.2f}m  >5min={p5:.1f}%  Δ={d:+.1f}%")
    return {"label": label, "mae": m, "avg_wait": w,
            "pct_over_5min": p5, "mae_delta_pct": d, "wait_series": wait}


def tier_breakdown(df):
    print("\n  IMPROVEMENT BY MERCHANT TIER")
    print("  " + "-" * 55)
    print(f"  {'Tier':<6} {'Base MAE':>9} {'KP MAE':>9} {'Improvement':>12} {'Merchants':>10}")
    print("  " + "-" * 55)
    for tier in ['T1', 'T2', 'T3']:
        sub = df[df['tier'] == tier]
        if len(sub) < 10:
            continue
        bm = mae(sub['naive_kpt_estimate'], sub['true_kpt_minutes'])
        km = mae(sub['kli_adjusted_kpt'], sub['true_kpt_minutes'])
        print(f"  {tier:<6} {bm:>8.2f}m {km:>8.2f}m "
              f"{(bm-km)/bm*100:>+11.1f}% {sub['restaurant_id'].nunique():>10}")
    print("  " + "-" * 55)


def build_charts(results, df):
    os.makedirs('report/figures', exist_ok=True)
    main = [r for r in results if not r['label'].startswith('5')]
    labels = [r['label'].split('.')[1].strip() for r in main]
    colors = [RED, AMBER, BLUE, GREEN]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    def sax(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUB, labelsize=8)
        ax.xaxis.label.set_color(SUB)
        ax.yaxis.label.set_color(SUB)
        ax.title.set_color(TEXT)
        for s in ax.spines.values():
            s.set_edgecolor(GRID)
        ax.grid(axis='y', color=GRID, linestyle='--', linewidth=0.5, alpha=0.7)
        return ax

    tkw = dict(color=TEXT, fontsize=11, fontweight='bold', pad=10)

    # Chart 1 — KPT MAE
    ax1 = sax(fig.add_subplot(gs[0, 0]))
    vals = [r['mae'] for r in main]
    bars = ax1.bar(labels, vals, color=colors, width=0.5, edgecolor='none')
    ax1.set_title('KPT Prediction Error (MAE)', **tkw)
    ax1.set_ylabel('Minutes', color=SUB)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.05, f'{v:.2f}m',
                 ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')

    # Chart 2 — Avg rider wait
    ax2 = sax(fig.add_subplot(gs[0, 1]))
    vals2 = [r['avg_wait'] for r in main]
    bars2 = ax2.bar(labels, vals2, color=colors, width=0.5, edgecolor='none')
    ax2.set_title('Avg Rider Wait at Pickup', **tkw)
    ax2.set_ylabel('Minutes', color=SUB)
    for b, v in zip(bars2, vals2):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.2f}m',
                 ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')

    # Chart 3 — % long wait
    ax3 = sax(fig.add_subplot(gs[0, 2]))
    vals3 = [r['pct_over_5min'] for r in main]
    bars3 = ax3.bar(labels, vals3, color=colors, width=0.5, edgecolor='none')
    ax3.set_title('% Orders: Rider Wait > 5 Min', **tkw)
    ax3.set_ylabel('Percentage (%)', color=SUB)
    for b, v in zip(bars3, vals3):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.2, f'{v:.1f}%',
                 ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')

    # Chart 4 — KDE distribution
    ax4 = sax(fig.add_subplot(gs[1, 0:2]))
    for r, c in zip(main, colors):
        if 'wait_series' in r:
            r['wait_series'].clip(0, 20).plot.kde(
                ax=ax4, color=c, linewidth=2,
                label=r['label'].split('.')[1].strip()
            )
    ax4.axvline(5, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_title('Rider Wait Time Distribution', **tkw)
    ax4.set_xlabel('Wait Time (minutes)', color=SUB)
    ax4.set_ylabel('Density', color=SUB)
    ax4.legend(facecolor=PANEL, edgecolor=GRID, labelcolor='white', fontsize=8)
    ax4.set_xlim(left=0)

    # Chart 5 — KLI scatter
    ax5 = sax(fig.add_subplot(gs[1, 2]))
    if 'kitchen_load_index' in df.columns:
        s = df.sample(min(2000, len(df)), random_state=42)
        ax5.scatter(s['kitchen_load_index'], s['true_kpt_minutes'],
                    c=s['kitchen_load_index'], cmap='RdYlGn_r',
                    alpha=0.4, s=8, linewidths=0)
        z = np.polyfit(s['kitchen_load_index'], s['true_kpt_minutes'], 1)
        xr = np.linspace(s['kitchen_load_index'].min(), s['kitchen_load_index'].max(), 100)
        ax5.plot(xr, np.poly1d(z)(xr), color='white', linewidth=1.5, linestyle='--', alpha=0.8)
        corr = df['kitchen_load_index'].corr(df['true_kpt_minutes'])
        ax5.set_title(f'KLI vs True KPT  (r = {corr:.3f})', **tkw)
        ax5.set_xlabel('Kitchen Load Index', color=SUB)
        ax5.set_ylabel('True KPT (min)', color=SUB)

    fig.suptitle('KitchenPulse — Simulation Results',
                 color=TEXT, fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('report/figures/simulation_results.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("  Chart → report/figures/simulation_results.png")


def run(inject_noise=True):
    df = pd.read_csv('data/synthetic_orders.csv')
    for c in ['order_time', 'actual_ready_time', 'for_button_time',
              'rider_arrival_time', 'pos_ticket_cleared_time']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])

    print("\n" + "=" * 65)
    print("  KitchenPulse — Signal Pipeline Simulation")
    print("=" * 65)

    base_mae_val = mae(df['naive_kpt_estimate'], df['true_kpt_minutes'])
    print(f"\n  Baseline MAE: {base_mae_val:.2f} min\n{'─'*65}")

    results = []

    # ── A: Baseline ──────────────────────────────────────────────────────────
    results.append(print_row("1. Baseline (Zomato today)", df, 'naive_kpt_estimate', base_mae_val))

    # ── B: Denoised FOR static ────────────────────────────────────────────────
    df_d = flag_rider_proximate(df)
    offsets = compute_bias_offsets(df_d)
    df_d = apply_for_correction(df_d, offsets)
    df_d = compute_pos_kpt(df_d)
    # corrected_for_kpt is now correct (was corrected_kpt in the broken version)
    results.append(print_row("2. Denoised FOR (static median)", df_d, 'corrected_for_kpt', base_mae_val))

    # ── C: EMA adaptive ──────────────────────────────────────────────────────
    df_ema = apply_ema_offset(df_d)
    results.append(print_row("3. EMA Denoised FOR (adaptive)", df_ema, 'corrected_kpt_ema', base_mae_val))

    # ── D: Full KitchenPulse ─────────────────────────────────────────────────
    # Normalise all KLI signals then compute KLI and apply to KPT
    df_kp = normalise_concurrent_orders(df_d)
    df_kp = normalise_acceptance_latency(df_kp)
    df_kp = normalise_foot_traffic(df_kp)
    df_kp = normalise_competitor_orders(df_kp)
    df_kp = compute_kli(df_kp)
    df_kp = apply_kli_to_kpt(df_kp)   # canonical — no local duplicate
    results.append(print_row("4. KitchenPulse Full (POS + KLI)", df_kp, 'kli_adjusted_kpt', base_mae_val))
    tier_breakdown(df_kp)

    # ── E: Adversarial ───────────────────────────────────────────────────────
    if inject_noise:
        print(f"\n{'─'*65}\n  Adversarial Noise Test\n{'─'*65}")
        df_adv = inject_adversarial_noise(df)
        for c in ['order_time', 'actual_ready_time', 'for_button_time',
                  'rider_arrival_time', 'pos_ticket_cleared_time']:
            if c in df_adv.columns:
                df_adv[c] = pd.to_datetime(df_adv[c])

        # Adversarial baseline uses the same naive_kpt_estimate (unaffected by FOR noise)
        adv_base = mae(df_adv['naive_kpt_estimate'], df_adv['true_kpt_minutes'])

        df_adv2 = flag_rider_proximate(df_adv)
        offsets2 = compute_bias_offsets(df_adv2)
        df_adv2 = apply_for_correction(df_adv2, offsets2)
        df_adv2 = compute_pos_kpt(df_adv2)
        df_adv2 = normalise_concurrent_orders(df_adv2)
        df_adv2 = normalise_acceptance_latency(df_adv2)
        df_adv2 = normalise_foot_traffic(df_adv2)
        df_adv2 = normalise_competitor_orders(df_adv2)
        df_adv2 = compute_kli(df_adv2)
        df_adv2 = apply_kli_to_kpt(df_adv2)

        results.append(print_row("5a. Baseline (adversarial)", df_adv2, 'naive_kpt_estimate', adv_base))
        results.append(print_row("5b. KitchenPulse (adversarial)", df_adv2, 'kli_adjusted_kpt', adv_base))

    print(f"\n{'='*65}\n")

    os.makedirs('data', exist_ok=True)
    df_kp.to_csv('data/processed_orders.csv', index=False)
    clean = [{k: v for k, v in r.items() if k != 'wait_series'} for r in results]
    pd.DataFrame(clean).to_csv('data/simulation_results_summary.csv', index=False)
    print("  Saved → data/processed_orders.csv")
    print("  Saved → data/simulation_results_summary.csv")

    build_charts(results, df_kp)
    print("\n✅ Simulation complete.\n")
    return df_kp


if __name__ == '__main__':
    run(inject_noise=True)
