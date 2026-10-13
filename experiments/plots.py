"""Plots for the paper.

Main figures (read from main.csv):
    fig_shd_timeseries.png     SHD per window, 5 scenarios, 4 baselines
    fig_drift_score.png        Drift score per window, 5 scenarios
    fig_update_speedup.png     Update vs retrain time (log scale)
    fig_fallback_rate.png      Fallback rate by scenario, ours only
    fig_delay_fpr.png          Detection delay vs FPR (from cusum_sweep.csv)

Ablation figures (read from ablation_*/summary_by_scenario.csv):
    fig_ablation_signals.png   Detection recall by scenario x signal mode
    fig_ablation_update.png    SHD and update time by scenario x update scope
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------

BASELINE_LABELS = {
    'B1': 'Full retrain',
    'B2': 'Detect + full',
    'B3': 'Frozen',
    'B4': 'Ours',
}

SCENARIO_LABELS = {
    'S1': 'S1: structure / abrupt',
    'S2': 'S2: coef / abrupt',
    'S3': 'S3: both / abrupt',
    'S4': 'S4: structure / gradual',
    'S5': 'S5: coef / gradual',
}

SIGNAL_LABELS = {
    'A1': 'ED only',
    'A2': 'ΔB only',
    'A3': 'SHD only',
    'A4': 'ED + ΔB',
    'A5': 'ED + ΔB + SHD',
    'A6': 'ΔB + SHD',
}

UPDATE_LABELS = {
    'U1': 'Seeds only',
    'U2': 'Seeds + parents',
    'U3': 'Seeds + MB (default)',
    'U4': 'Full retrain',
}


# ---------------------------------------------------------------------
# Main figures
# ---------------------------------------------------------------------

def fig_shd_timeseries(df: pd.DataFrame, out: Path) -> None:
    scenarios = sorted(df['scenario'].unique())
    baselines = sorted(df['baseline'].unique())

    fig, axes = plt.subplots(
        1, len(scenarios), figsize=(4 * len(scenarios), 3.2),
        sharey=True,
    )
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scen in zip(axes, scenarios):
        sub = df[df['scenario'] == scen]
        for bl in baselines:
            s = sub[sub['baseline'] == bl].groupby('window')['shd'].agg(
                ['mean', 'std']
            )
            ax.plot(s.index, s['mean'], label=BASELINE_LABELS.get(bl, bl))
            ax.fill_between(
                s.index,
                s['mean'] - s['std'],
                s['mean'] + s['std'],
                alpha=0.15,
            )
        ax.axvline(5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(SCENARIO_LABELS.get(scen, scen), fontsize=9)
        ax.set_xlabel('window')
    axes[0].set_ylabel('SHD vs ground truth')
    axes[-1].legend(loc='upper left', fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'fig_shd_timeseries.png', dpi=150)
    plt.close(fig)


def fig_drift_score(df: pd.DataFrame, out: Path) -> None:
    scenarios = sorted(df['scenario'].unique())
    fig, axes = plt.subplots(
        1, len(scenarios), figsize=(4 * len(scenarios), 3.0),
        sharey=True,
    )
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scen in zip(axes, scenarios):
        sub = df[(df['scenario'] == scen) & (df['baseline'] == 'B4')]
        s = sub.groupby('window')['drift_score'].agg(['mean', 'std'])
        ax.plot(s.index, s['mean'], color='C3')
        ax.fill_between(
            s.index,
            s['mean'] - s['std'],
            s['mean'] + s['std'],
            alpha=0.2, color='C3',
        )
        ax.axvline(5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(SCENARIO_LABELS.get(scen, scen), fontsize=9)
        ax.set_xlabel('window')
    axes[0].set_ylabel('Drift score')
    fig.tight_layout()
    fig.savefig(out / 'fig_drift_score.png', dpi=150)
    plt.close(fig)


def fig_update_speedup(df: pd.DataFrame, out: Path) -> None:
    sub = df[(df['baseline'] == 'B4') & (df['drift_confirmed'] == 1)]
    if sub.empty:
        print("no confirmed drift events, skipping speedup plot")
        return

    agg = sub.groupby('scenario').agg(
        update_time=('update_time', 'mean'),
        retrain_time=('retrain_time', 'mean'),
    )
    x = np.arange(len(agg))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - width / 2, agg['retrain_time'] * 1000, width,
           label='Full retrain')
    ax.bar(x + width / 2, agg['update_time'] * 1000, width,
           label='Incremental update')
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.set_ylabel('time (ms)')
    ax.set_yscale('log')
    ax.set_title('Update vs full-retrain time on drift windows')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'fig_update_speedup.png', dpi=150)
    plt.close(fig)


def fig_fallback_rate(df: pd.DataFrame, out: Path) -> None:
    sub = df[df['baseline'] == 'B4']
    agg = sub.groupby('scenario')['fallback'].mean()

    fig, ax = plt.subplots(figsize=(5, 3))
    agg.plot(kind='bar', ax=ax, color='C2')
    ax.set_ylabel('Fallback rate')
    ax.set_ylim(0, 1.05)
    ax.set_title('Ours: fallback rate by scenario')
    fig.tight_layout()
    fig.savefig(out / 'fig_fallback_rate.png', dpi=150)
    plt.close(fig)


def fig_delay_fpr(out_root: Path, fig_root: Path) -> None:
    sweep_path = out_root / 'cusum_sweep.csv'
    if not sweep_path.exists():
        print("cusum_sweep.csv not found, skipping delay-FPR plot")
        return

    sweep = pd.read_csv(sweep_path)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for scen in sorted(sweep['scenario'].unique()):
        s = sweep[sweep['scenario'] == scen].sort_values('fpr_mean')
        # Drop points where delay is NaN (no detection) to keep the line clean
        s = s[s['delay_mean'].notna()]
        if s.empty:
            continue
        ax.plot(s['fpr_mean'], s['delay_mean'], marker='o',
                label=SCENARIO_LABELS.get(scen, scen))
    ax.set_xlabel('False positive rate')
    ax.text(
        0.02, 0.98,
        'each point: one kappa in {2,3,4,5,6,7,8}',
        transform=ax.transAxes,
        fontsize=7, color='gray', va='top', ha='left',
    )
    ax.set_ylabel('Detection delay (windows)')
    ax.set_title('Delay vs FPR (CUSUM sensitivity sweep)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_root / 'fig_delay_fpr.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Ablation figures
# ---------------------------------------------------------------------

def _load_ablation_summary(out_root: Path, subdir: str) -> pd.DataFrame | None:
    csv_path = out_root / subdir / 'summary_by_scenario.csv'
    if not csv_path.exists():
        print(f"{csv_path} not found, skipping")
        return None
    return pd.read_csv(csv_path)


def fig_ablation_signals(out_root: Path, fig_root: Path) -> None:
    df = _load_ablation_summary(out_root, 'ablation_signals')
    if df is None:
        return

    mode_order = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    modes = [m for m in mode_order if m in df['mode_id'].unique()]
    scenarios = sorted(df['scenario'].unique())

    pivot = df.pivot(
        index='scenario', columns='mode_id', values='det_recall'
    ).reindex(index=scenarios, columns=modes)

    fig, ax = plt.subplots(figsize=(1.4 * len(scenarios) + 2.5, 4))
    width = 0.8 / len(modes)
    x = np.arange(len(scenarios))

    for i, mode in enumerate(modes):
        offset = (i - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, pivot[mode].values, width,
               label=SIGNAL_LABELS.get(mode, mode))

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Detection recall')
    ax.set_ylim(0, 1.05)
    ax.set_title('Signal ablation: detection recall by scenario')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_root / 'fig_ablation_signals.png', dpi=150)
    plt.close(fig)


def fig_ablation_update(out_root: Path, fig_root: Path) -> None:
    df = _load_ablation_summary(out_root, 'ablation_update')
    if df is None:
        return

    mode_order = ['U1', 'U2', 'U3', 'U4']
    modes = [m for m in mode_order if m in df['mode_id'].unique()]
    scenarios = sorted(df['scenario'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # --- Left: SHD delta relative to U4 (full retrain) ---
    pivot_shd = df.pivot(
        index='scenario', columns='mode_id', values='shd_post_mean'
    ).reindex(index=scenarios, columns=modes)

    if 'U4' in pivot_shd.columns:
        ref = pivot_shd['U4']
        pivot_shd_delta = pivot_shd.sub(ref, axis=0)
    else:
        pivot_shd_delta = pivot_shd

    width = 0.8 / len(modes)
    x = np.arange(len(scenarios))

    for i, mode in enumerate(modes):
        offset = (i - (len(modes) - 1) / 2) * width
        axes[0].bar(x + offset, pivot_shd_delta[mode].values, width,
                    label=UPDATE_LABELS.get(mode, mode))
    axes[0].axhline(0, color='gray', linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios)
    axes[0].set_ylabel('SHD − SHD(U4)   (positive = worse)')
    axes[0].set_title('Graph quality by update scope (Δ vs full retrain)')
    axes[0].legend(fontsize=8)

    # --- Right: update time on drift windows (log) ---
    pivot_upd = df.pivot(
        index='scenario', columns='mode_id', values='update_time_drift_mean'
    ).reindex(index=scenarios, columns=modes)

    for i, mode in enumerate(modes):
        offset = (i - (len(modes) - 1) / 2) * width
        vals = pivot_upd[mode].values * 1000.0
        vals_plot = np.where(vals > 0, vals, np.nan)
        axes[1].bar(x + offset, vals_plot, width,
                    label=UPDATE_LABELS.get(mode, mode))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios)
    axes[1].set_ylabel('Update time on drift windows (ms)')
    axes[1].set_title('Incremental update cost by scope (lower is better)')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=8)

    # Annotate that U4 has no incremental path
    axes[1].text(
        0.02, 0.10,
        'Full retrain (U4) has no incremental path',
        transform=axes[1].transAxes,
        fontsize=8, color='gray', va='bottom', ha='left',
    )

    fig.tight_layout()
    fig.savefig(fig_root / 'fig_ablation_update.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', default='results/main.csv')
    parser.add_argument('--out', default='results/figures')
    parser.add_argument('--root', default='results',
                        help='root dir containing ablation_* subdirs')
    args = parser.parse_args()

    fig_root = Path(args.out)
    fig_root.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.root)

    df = pd.read_csv(args.main)

    # Main figures
    fig_shd_timeseries(df, fig_root)
    fig_drift_score(df, fig_root)
    fig_update_speedup(df, fig_root)
    fig_fallback_rate(df, fig_root)
    fig_delay_fpr(out_root, fig_root)

    # Ablation figures
    fig_ablation_signals(out_root, fig_root)
    fig_ablation_update(out_root, fig_root)

    print(f"wrote figures to {fig_root}")


if __name__ == '__main__':
    main()