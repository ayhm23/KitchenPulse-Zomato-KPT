"""
Correlation analysis and chart generation for KitchenPulse.

Produces 6 analytical charts:
- Signal correlations with true KPT
- Merchant bias distribution
- Hidden load impact on wait times
- Kitchen load heatmap by hour
- Improvement by tier
- Signal accuracy comparison
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.signal_denoiser import run_denoiser
from pipeline.kitchen_load_index import run_kli

os.makedirs('report/figures', exist_ok=True)

# ── Styling constants ─────────────────────────────────────────────────────────
BG       = '#0F1117'
PANEL    = '#1A1D27'
GRID     = '#2A2D3A'
TEXT     = '#FFFFFF'
SUBTEXT  = '#AAAAAA'
RED      = '#E74C3C'
AMBER    = '#F39C12'
GREEN    = '#27AE60'
BLUE     = '#3498DB'
PURPLE   = '#9B59B6'

def base_style(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.yaxis.label.set_color(SUBTEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
    return fig

def save(name):
    path = f'report/figures/{name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  ✓ Saved → {path}')


# ── Load data ─────────────────────────────────────────────────────────────────
def load():
    print('Loading processed_orders.csv...')
    try:
        df = pd.read_csv('data/processed_orders.csv')
        print(f'  Loaded {len(df):,} rows from processed_orders.csv')
    except FileNotFoundError:
        print('  processed_orders.csv not found — running full pipeline...')
        df = pd.read_csv('data/synthetic_orders.csv')
        for col in ['order_time', 'actual_ready_time', 'for_button_time',
                    'rider_arrival_time', 'pos_ticket_cleared_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        df = run_denoiser(df)
        df = run_kli(df)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Signal Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def chart_correlation_heatmap(df):
    print('\n[Chart 1] Signal correlation heatmap...')

    signals = {
        'Zomato Concurrent Orders' : 'zomato_concurrent_orders',
        'Acceptance Latency (s)'   : 'acceptance_latency_seconds',
        'Foot Traffic Index'       : 'local_foot_traffic_index',
        'Competitor Orders'        : 'competitor_platform_orders',
        'Kitchen Load Index'       : 'kitchen_load_index',
        'Raw FOR KPT'              : 'raw_for_kpt',
        'Corrected FOR KPT'        : 'corrected_for_kpt',
        'POS KPT'                  : 'pos_kpt',
        'KLI-Adjusted KPT'         : 'kli_adjusted_kpt',
        'True KPT (ground truth)'  : 'true_kpt_minutes',
    }

    available = {k: v for k, v in signals.items() if v in df.columns}
    corr_df   = df[[v for v in available.values()]].corr()
    corr_df.index   = list(available.keys())
    corr_df.columns = list(available.keys())

    fig, ax = plt.subplots(figsize=(12, 10))
    base_style(fig, ax)

    cmap = plt.cm.RdYlGn
    im   = ax.imshow(corr_df.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(corr_df)))
    ax.set_yticks(range(len(corr_df)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha='right',
                       fontsize=8, color=SUBTEXT)
    ax.set_yticklabels(corr_df.index, fontsize=8, color=SUBTEXT)

    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            val = corr_df.values[i, j]
            color = 'white' if abs(val) > 0.5 else SUBTEXT
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=SUBTEXT, labelsize=8)
    cbar.set_label('Pearson Correlation', color=SUBTEXT, fontsize=9)

    ax.set_title('Signal Correlation Matrix — KitchenPulse vs True KPT',
                 color=TEXT, fontsize=12, fontweight='bold', pad=15)

    # Highlight the True KPT row
    ax.axhline(len(corr_df)-1, color=GREEN, linewidth=2, alpha=0.6)
    ax.axvline(len(corr_df)-1, color=GREEN, linewidth=2, alpha=0.6)

    fig.tight_layout()
    save('chart1_correlation_heatmap')


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — FOR Button Bias: Honest vs Biased Merchant Distribution
# ══════════════════════════════════════════════════════════════════════════════
def chart_for_bias_distribution(df):
    print('[Chart 2] FOR button bias distribution...')

    if 'for_delay_seconds' not in df.columns or 'honest_merchant' not in df.columns:
        print('  Skipping — required columns not found')
        return

    honest = df[df['honest_merchant'] == True]['for_delay_seconds'].dropna()
    biased = df[df['honest_merchant'] == False]['for_delay_seconds'].dropna()

    # Convert to minutes, clip for readable chart
    honest_min = (honest / 60).clip(-5, 20)
    biased_min = (biased / 60).clip(-5, 20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    base_style(fig, axes)

    bins = np.linspace(-5, 20, 60)

    # Left: histogram overlay
    axes[0].hist(honest_min, bins=bins, color=GREEN,  alpha=0.7,
                 label=f'Honest ({len(honest_min):,} events)', density=True)
    axes[0].hist(biased_min, bins=bins, color=RED,    alpha=0.7,
                 label=f'Biased ({len(biased_min):,} events)', density=True)
    axes[0].axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)
    axes[0].set_title('FOR Button Delay Distribution', color=TEXT,
                       fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Delay vs Rider Arrival (minutes)', color=SUBTEXT)
    axes[0].set_ylabel('Density', color=SUBTEXT)
    axes[0].legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # Right: box summary
    data_box  = [honest_min.values, biased_min.values]
    bp = axes[1].boxplot(data_box, patch_artist=True,
                          medianprops=dict(color='white', linewidth=2),
                          whiskerprops=dict(color=SUBTEXT),
                          capprops=dict(color=SUBTEXT),
                          flierprops=dict(marker='o', markersize=2,
                                          markerfacecolor=SUBTEXT, alpha=0.3))
    bp['boxes'][0].set_facecolor(GREEN)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(RED)
    bp['boxes'][1].set_alpha(0.7)

    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['Honest\nMerchants', 'Biased\nMerchants'],
                              color=SUBTEXT, fontsize=9)
    axes[1].set_ylabel('Delay (minutes)', color=SUBTEXT)
    axes[1].set_title('FOR Button Delay — Box Summary', color=TEXT,
                       fontsize=11, fontweight='bold')
    axes[1].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.4)

    # Annotate medians
    med_h = np.median(honest_min)
    med_b = np.median(biased_min)
    axes[1].text(1, med_h + 0.3, f'Median: {med_h:.1f}m', ha='center',
                  color=GREEN, fontsize=8, fontweight='bold')
    axes[1].text(2, med_b + 0.3, f'Median: {med_b:.1f}m', ha='center',
                  color=RED,   fontsize=8, fontweight='bold')

    fig.suptitle('Problem Evidence: Merchant FOR Button Bias',
                 color=TEXT, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save('chart2_for_bias_distribution')


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Hidden Load vs Rider Wait (the invisible problem)
# ══════════════════════════════════════════════════════════════════════════════
def chart_hidden_load_impact(df):
    print('[Chart 3] Hidden load vs rider wait...')

    if 'hidden_load' not in df.columns or 'actual_rider_wait_minutes' not in df.columns:
        print('  Skipping — required columns not found')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    base_style(fig, axes)

    sample = df.sample(min(3000, len(df)), random_state=42)

    # Left: scatter hidden_load vs rider wait
    sc = axes[0].scatter(
        sample['hidden_load'],
        sample['actual_rider_wait_minutes'].clip(0, 20),
        c=sample['hidden_load'], cmap='RdYlGn_r',
        alpha=0.35, s=10, linewidths=0
    )
    z = np.polyfit(sample['hidden_load'],
                   sample['actual_rider_wait_minutes'].clip(0, 20), 1)
    xr = np.linspace(sample['hidden_load'].min(), sample['hidden_load'].max(), 100)
    axes[0].plot(xr, np.poly1d(z)(xr), color='white', linewidth=2,
                  linestyle='--', alpha=0.8)
    corr = df['hidden_load'].corr(df['actual_rider_wait_minutes'])
    axes[0].set_title(f'Hidden Load vs Rider Wait  (r = {corr:.3f})',
                       color=TEXT, fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Hidden Load (dine-in + competitor orders)', color=SUBTEXT)
    axes[0].set_ylabel('Rider Wait Time (minutes)', color=SUBTEXT)

    # Right: binned average wait by hidden load quartile
    df2 = df.copy()
    df2['load_bin'] = pd.qcut(df2['hidden_load'], q=4,
                               labels=['Q1\nLow', 'Q2\nMed-Low',
                                       'Q3\nMed-High', 'Q4\nHigh'])
    avg_wait = df2.groupby('load_bin', observed=True)['actual_rider_wait_minutes'].mean()
    colors_q = [GREEN, AMBER, '#E67E22', RED]
    bars = axes[1].bar(avg_wait.index, avg_wait.values,
                        color=colors_q, width=0.5, edgecolor='none')
    axes[1].set_title('Avg Rider Wait by Hidden Load Quartile',
                       color=TEXT, fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Hidden Load Quartile', color=SUBTEXT)
    axes[1].set_ylabel('Avg Rider Wait (minutes)', color=SUBTEXT)
    axes[1].grid(axis='y', color=GRID, linestyle='--', alpha=0.5)
    for bar, v in zip(bars, avg_wait.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.05,
                      f'{v:.2f}m', ha='center', va='bottom',
                      color='white', fontsize=9, fontweight='bold')

    fig.suptitle('The Hidden Load Problem: What Zomato Cannot See',
                 color=TEXT, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save('chart3_hidden_load_impact')


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Hourly Kitchen Load Heatmap (across all restaurants)
# ══════════════════════════════════════════════════════════════════════════════
def chart_hourly_kli_heatmap(df):
    print('[Chart 4] Hourly KLI heatmap...')

    if 'kitchen_load_index' not in df.columns or 'hour_of_day' not in df.columns:
        print('  Skipping — required columns not found')
        return

    if 'day_of_week' not in df.columns:
        df['day_of_week'] = 0

    pivot = df.pivot_table(
        values='kitchen_load_index',
        index='day_of_week',
        columns='hour_of_day',
        aggfunc='mean'
    )

    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot.index = [day_labels[i] for i in pivot.index if i < len(day_labels)]

    fig, ax = plt.subplots(figsize=(16, 5))
    base_style(fig, ax)

    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                    vmin=20, vmax=80)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{h:02d}:00' for h in pivot.columns],
                        rotation=45, ha='right', fontsize=7, color=SUBTEXT)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, color=SUBTEXT)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=6, color='white', fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.9)
    cbar.ax.tick_params(colors=SUBTEXT, labelsize=8)
    cbar.set_label('Kitchen Load Index', color=SUBTEXT, fontsize=9)

    ax.set_title('Kitchen Load Index by Hour & Day of Week\n'
                  'Red = High Load (kitchen under pressure) | Green = Low Load',
                  color=TEXT, fontsize=11, fontweight='bold', pad=12)

    fig.tight_layout()
    save('chart4_hourly_kli_heatmap')


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Tier Improvement Breakdown
# ══════════════════════════════════════════════════════════════════════════════
def chart_tier_improvement(df):
    print('[Chart 5] Tier improvement breakdown...')

    if 'tier' not in df.columns:
        print('  Skipping — tier column not found')
        return

    def mae(pred, actual):
        return np.mean(np.abs(pred - actual))

    tiers, base_maes, kp_maes, improvements, counts = [], [], [], [], []
    for tier in ['T1', 'T2', 'T3']:
        sub = df[df['tier'] == tier]
        if len(sub) < 10:
            continue
        bm = mae(sub['naive_kpt_estimate'], sub['true_kpt_minutes'])
        km = mae(sub['kli_adjusted_kpt'],   sub['true_kpt_minutes'])
        tiers.append(tier)
        base_maes.append(bm)
        kp_maes.append(km)
        improvements.append((bm - km) / bm * 100)
        counts.append(sub['restaurant_id'].nunique())

    x     = np.arange(len(tiers))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    base_style(fig, axes)

    # Left: grouped MAE bars
    b1 = axes[0].bar(x - width/2, base_maes, width, color=RED,   alpha=0.85,
                      label='Baseline (Zomato today)', edgecolor='none')
    b2 = axes[0].bar(x + width/2, kp_maes,   width, color=GREEN, alpha=0.85,
                      label='KitchenPulse',            edgecolor='none')

    tier_labels = [
        f'{t}\n({c} restaurants)' for t, c in zip(tiers, counts)
    ]
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tier_labels, color=SUBTEXT, fontsize=9)
    axes[0].set_ylabel('KPT MAE (minutes)', color=SUBTEXT)
    axes[0].set_title('KPT Error by Merchant Tier', color=TEXT,
                       fontsize=11, fontweight='bold')
    axes[0].legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    axes[0].grid(axis='y', color=GRID, linestyle='--', alpha=0.5)

    for bar, v in zip(b1, base_maes):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.05,
                      f'{v:.2f}m', ha='center', va='bottom',
                      color=RED, fontsize=8, fontweight='bold')
    for bar, v in zip(b2, kp_maes):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.05,
                      f'{v:.2f}m', ha='center', va='bottom',
                      color=GREEN, fontsize=8, fontweight='bold')

    # Right: improvement % bars
    colors_imp = [GREEN if i > 0 else RED for i in improvements]
    bars = axes[1].bar(tiers, improvements, color=colors_imp,
                        width=0.4, edgecolor='none', alpha=0.85)
    axes[1].set_ylabel('MAE Improvement (%)', color=SUBTEXT)
    axes[1].set_title('% Improvement by Tier', color=TEXT,
                       fontsize=11, fontweight='bold')
    axes[1].axhline(0, color='white', linewidth=1, alpha=0.3)
    axes[1].grid(axis='y', color=GRID, linestyle='--', alpha=0.5)
    axes[1].tick_params(colors=SUBTEXT, labelsize=10)

    for bar, v in zip(bars, improvements):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                      v + (1 if v >= 0 else -2),
                      f'+{v:.1f}%' if v >= 0 else f'{v:.1f}%',
                      ha='center', va='bottom', color='white',
                      fontsize=11, fontweight='bold')

    tier_desc = 'T1 = Large chains (POS integrated) | T2 = Mid-size | T3 = Small stalls'
    fig.text(0.5, -0.02, tier_desc, ha='center', color=SUBTEXT, fontsize=8)
    fig.suptitle('KitchenPulse Scales Across All 300,000+ Merchants',
                 color=TEXT, fontsize=13, fontweight='bold')
    fig.tight_layout()
    save('chart5_tier_improvement')


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Signal Accuracy Ladder (ranked MAE of every signal)
# ══════════════════════════════════════════════════════════════════════════════
def chart_signal_accuracy_ladder(df):
    print('[Chart 6] Signal accuracy ladder...')

    signal_map = {
        'Naive KPT\n(Zomato today)'   : 'naive_kpt_estimate',
        'Raw FOR Button'               : 'raw_for_kpt',
        'Corrected FOR Button'         : 'corrected_for_kpt',
        'POS Ticket Signal'            : 'pos_kpt',
        'KLI-Adjusted KPT\n(Full System)' : 'kli_adjusted_kpt',
    }

    labels, maes = [], []
    for label, col in signal_map.items():
        if col in df.columns:
            m = np.mean(np.abs(df[col] - df['true_kpt_minutes']))
            labels.append(label)
            maes.append(m)

    # Sort worst → best
    paired = sorted(zip(maes, labels), reverse=True)
    maes   = [p[0] for p in paired]
    labels = [p[1] for p in paired]

    colors = []
    for l in labels:
        if 'Full System' in l or 'KLI' in l:
            colors.append(GREEN)
        elif 'POS' in l or 'Corrected' in l:
            colors.append(BLUE)
        elif 'Naive' in l or 'Raw' in l:
            colors.append(RED)
        else:
            colors.append(AMBER)

    fig, ax = plt.subplots(figsize=(12, 6))
    base_style(fig, ax)

    bars = ax.barh(labels, maes, color=colors, alpha=0.85,
                    height=0.5, edgecolor='none')
    ax.set_xlabel('Mean Absolute Error vs True KPT (minutes)', color=SUBTEXT)
    ax.set_title('Signal Accuracy Ladder — Lower is Better',
                  color=TEXT, fontsize=12, fontweight='bold')
    ax.grid(axis='x', color=GRID, linestyle='--', alpha=0.5)
    ax.tick_params(colors=SUBTEXT, labelsize=9)
    ax.invert_yaxis()

    for bar, v in zip(bars, maes):
        ax.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                f'{v:.2f} min', va='center', color='white',
                fontsize=9, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=RED,   label='Current Zomato signals (broken)'),
        Patch(facecolor=BLUE,  label='Intermediate KitchenPulse signals'),
        Patch(facecolor=GREEN, label='Full KitchenPulse system'),
    ]
    ax.legend(handles=legend_elements, facecolor=PANEL, edgecolor=GRID,
               labelcolor=TEXT, fontsize=9, loc='lower right')

    fig.tight_layout()
    save('chart6_signal_accuracy_ladder')


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATS — Print everything useful for the PDF
# ══════════════════════════════════════════════════════════════════════════════
def print_pdf_stats(df):
    def mae(a, b): return np.mean(np.abs(a - b))

    print('\n' + '='*65)
    print('  PDF-READY NUMBERS — copy these directly into your report')
    print('='*65)

    if all(c in df.columns for c in ['naive_kpt_estimate', 'kli_adjusted_kpt', 'true_kpt_minutes']):
        base_mae = mae(df['naive_kpt_estimate'], df['true_kpt_minutes'])
        kp_mae   = mae(df['kli_adjusted_kpt'],   df['true_kpt_minutes'])
        print(f'  KPT MAE — Baseline      : {base_mae:.2f} min')
        print(f'  KPT MAE — KitchenPulse  : {kp_mae:.2f} min')
        print(f'  KPT MAE Improvement     : {(base_mae-kp_mae)/base_mae*100:.1f}%')

    if 'actual_rider_wait_minutes' in df.columns:
        print(f'\n  Avg rider wait (baseline): {df["actual_rider_wait_minutes"].mean():.2f} min')
        print(f'  Orders w/ wait > 5 min  : {(df["actual_rider_wait_minutes"]>5).mean()*100:.1f}%')

    if 'for_delay_seconds' in df.columns and 'honest_merchant' in df.columns:
        biased_delay = df[df['honest_merchant']==False]['for_delay_seconds'].median()/60
        honest_delay = df[df['honest_merchant']==True]['for_delay_seconds'].median()/60
        print(f'\n  FOR delay — biased merchants  : +{biased_delay:.2f} min (median)')
        print(f'  FOR delay — honest merchants  : +{honest_delay:.2f} min (median)')

    if 'pos_kpt' in df.columns:
        pos_mae = mae(df['pos_kpt'], df['true_kpt_minutes'])
        raw_mae = mae(df['raw_for_kpt'], df['true_kpt_minutes'])
        print(f'\n  POS signal MAE            : {pos_mae:.2f} min')
        print(f'  Raw FOR signal MAE        : {raw_mae:.2f} min')
        print(f'  POS accuracy advantage    : {raw_mae/pos_mae:.1f}x better than FOR button')

    if 'kitchen_load_index' in df.columns:
        for col, label in [
            ('zomato_concurrent_orders', 'Zomato concurrent  '),
            ('acceptance_latency_seconds','Acceptance latency '),
            ('local_foot_traffic_index', 'Foot traffic index '),
            ('competitor_platform_orders','Competitor orders  '),
            ('kitchen_load_index',        'KLI composite      '),
        ]:
            if col in df.columns:
                r = df[col].corr(df['true_kpt_minutes'])
                print(f'  Corr({label}) w/ true KPT : {r:+.3f}')

    print('='*65)
    print('\n  Charts saved to report/figures/:')
    for i, name in enumerate([
        'chart1_correlation_heatmap',
        'chart2_for_bias_distribution',
        'chart3_hidden_load_impact',
        'chart4_hourly_kli_heatmap',
        'chart5_tier_improvement',
        'chart6_signal_accuracy_ladder',
    ], 1):
        print(f'    {i}. {name}.png')
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load()

    chart_correlation_heatmap(df)
    chart_for_bias_distribution(df)
    chart_hidden_load_impact(df)
    chart_hourly_kli_heatmap(df)
    chart_tier_improvement(df)
    chart_signal_accuracy_ladder(df)

    print_pdf_stats(df)

    print('✅ Phase 3 complete. All charts ready for PDF.\n')
