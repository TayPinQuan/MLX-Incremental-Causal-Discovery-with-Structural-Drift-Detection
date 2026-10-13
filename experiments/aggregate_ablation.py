"""Aggregate signal ablation results.

Reads ablation_signals.csv and produces:
    results/ablation_signals/summary_by_scenario.csv
    results/ablation_signals/summary_overall.csv
    results/ablation_signals/summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _to_md(df: pd.DataFrame, float_fmt: str = '{:.4f}') -> str:
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |',
             '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(float_fmt.format(v))
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def build_summary(main_csv: Path, out_dir: Path):
    df = pd.read_csv(main_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Support both signal and update ablations
    if 'signal_mode' in df.columns:
        mode_label_col = 'signal_mode'
    elif 'mb_mode' in df.columns:
        mode_label_col = 'mb_mode'
    else:
        raise ValueError("no mode label column found")

    rows = []
    for (scenario, mode_id, mode_label), sub in df.groupby(
        ['scenario', 'mode_id', mode_label_col]
    ):
        per_seed = sub.groupby('seed').agg(
            det_precision=('det_precision', 'first'),
            det_recall=('det_recall', 'first'),
            det_f1=('det_f1', 'first'),
        )
        post = sub[sub['window'] >= 5]
        drift_sub = sub[sub['drift_confirmed'] == 1]

        # Handle U4: update_time is 0 (goes straight to full)
        mean_update = drift_sub['update_time'].mean() if not drift_sub.empty else 0.0
        mean_retrain = drift_sub['retrain_time'].mean() if not drift_sub.empty else 0.0
        fallback_rate = drift_sub['fallback'].mean() if not drift_sub.empty else 0.0
        if mean_update > 0:
            ratio = mean_retrain / mean_update
        else:
            ratio = float('nan')  # U4 has no incremental path

        rows.append({
            'scenario': scenario,
            'mode_id': mode_id,
            'mode_label': mode_label,
            'det_precision': per_seed['det_precision'].mean(),
            'det_recall': per_seed['det_recall'].mean(),
            'det_f1': per_seed['det_f1'].mean(),
            'shd_post_mean': post['shd'].mean(),
            'f1_post_mean': post['f1'].mean(),
            'n_drift_confirmed': int(sub['drift_confirmed'].sum()),
            'update_time_drift_mean': mean_update,
            'retrain_time_drift_mean': mean_retrain,
            'fallback_rate_drift': fallback_rate,
            'update_vs_retrain_ratio': ratio,
            'n_seeds': sub['seed'].nunique(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'summary_by_scenario.csv', index=False)

    overall = summary.groupby(['mode_id', 'mode_label']).agg(
        det_precision=('det_precision', 'mean'),
        det_recall=('det_recall', 'mean'),
        det_f1=('det_f1', 'mean'),
        shd_post_mean=('shd_post_mean', 'mean'),
        f1_post_mean=('f1_post_mean', 'mean'),
        n_drift_confirmed=('n_drift_confirmed', 'sum'),
        update_time_drift_mean=('update_time_drift_mean', 'mean'),
        retrain_time_drift_mean=('retrain_time_drift_mean', 'mean'),
        fallback_rate_drift=('fallback_rate_drift', 'mean'),
        update_vs_retrain_ratio=('update_vs_retrain_ratio', 'mean'),
        n_seeds=('n_seeds', 'mean'),
    ).reset_index()
    overall.to_csv(out_dir / 'summary_overall.csv', index=False)

    # pivots
    pivot_shd = summary.pivot(
        index='scenario', columns='mode_id', values='shd_post_mean'
    )
    pivot_update = summary.pivot(
        index='scenario', columns='mode_id', values='update_time_drift_mean'
    )
    pivot_fallback = summary.pivot(
        index='scenario', columns='mode_id', values='fallback_rate_drift'
    )
    pivot_shd.to_csv(out_dir / 'pivot_shd.csv')
    pivot_update.to_csv(out_dir / 'pivot_update.csv')
    pivot_fallback.to_csv(out_dir / 'pivot_fallback.csv')

    print(f"wrote summary to {out_dir}")


def _to_md(df, fmt='{:.4f}'):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |',
             '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(fmt.format(v))
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--main',
        default='results/ablation_signals/ablation_signals.csv',
    )
    parser.add_argument('--out', default='results/ablation_signals')
    args = parser.parse_args()

    build_summary(Path(args.main), Path(args.out))


if __name__ == '__main__':
    main()