"""Aggregate main.csv into summary tables.

Outputs:
    results/summary_by_scenario.csv
    results/summary_overall.csv
    results/summary.md

Speedup and fallback rate are computed only on drift-confirmed
windows for B4, to avoid dilution by the many quiet windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _split_pre_post(df: pd.DataFrame, drift_window: int = 5):
    pre = df[df['window'] < drift_window]
    post = df[df['window'] >= drift_window]
    return pre, post


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

    # ------------------------------------------------------------------
    # Per-scenario summary
    # ------------------------------------------------------------------
    rows = []
    for scenario in sorted(df['scenario'].unique()):
        for baseline in sorted(df['baseline'].unique()):
            sub = df[(df['scenario'] == scenario)
                     & (df['baseline'] == baseline)]
            pre, post = _split_pre_post(sub, drift_window=5)

            det_prec = sub['det_precision'].mean()
            det_rec = sub['det_recall'].mean()
            det_f1 = sub['det_f1'].mean()

            drift_sub = sub[sub['drift_confirmed'] == 1]
            if not drift_sub.empty:
                mean_update = drift_sub['update_time'].mean()
                mean_retrain = drift_sub['retrain_time'].mean()
                fallback_rate = drift_sub['fallback'].mean()
                speedup = (
                    mean_retrain / mean_update
                    if mean_update > 0 else float('nan')
                )
            else:
                mean_update = 0.0
                mean_retrain = sub['retrain_time'].mean()
                fallback_rate = 0.0
                speedup = 1.0

            rows.append({
                'scenario': scenario,
                'baseline': baseline,
                'shd_pre_mean': pre['shd'].mean(),
                'shd_pre_std': pre['shd'].std(),
                'shd_post_mean': post['shd'].mean(),
                'shd_post_std': post['shd'].std(),
                'f1_post_mean': post['f1'].mean(),
                'coef_mse_post_mean': post['coef_mse'].mean(),
                'update_time_drift_mean': mean_update,
                'retrain_time_drift_mean': mean_retrain,
                'fallback_rate_drift': fallback_rate,
                'update_vs_retrain_ratio': speedup,
                'det_precision': det_prec,
                'det_recall': det_rec,
                'det_f1': det_f1,
                'n_rows': len(sub),
                'n_drift_windows': len(drift_sub),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'summary_by_scenario.csv', index=False)

    # ------------------------------------------------------------------
    # Overall (average across scenarios)
    # ------------------------------------------------------------------
    overall = summary.groupby('baseline').agg(
        shd_pre_mean=('shd_pre_mean', 'mean'),
        shd_post_mean=('shd_post_mean', 'mean'),
        f1_post_mean=('f1_post_mean', 'mean'),
        coef_mse_post_mean=('coef_mse_post_mean', 'mean'),
        update_time_drift_mean=('update_time_drift_mean', 'mean'),
        retrain_time_drift_mean=('retrain_time_drift_mean', 'mean'),
        fallback_rate_drift=('fallback_rate_drift', 'mean'),
        update_vs_retrain_ratio=('update_vs_retrain_ratio', 'mean'),
        det_precision=('det_precision', 'mean'),
        det_recall=('det_recall', 'mean'),
        det_f1=('det_f1', 'mean'),
    ).reset_index()
    overall.to_csv(out_dir / 'summary_overall.csv', index=False)

    # ------------------------------------------------------------------
    # Markdown (no tabulate dependency)
    # ------------------------------------------------------------------
    md_lines = ['# Summary by scenario\n']
    md_lines.append(_to_md(summary))
    md_lines.append('\n# Overall\n')
    md_lines.append(_to_md(overall))
    (out_dir / 'summary.md').write_text('\n'.join(md_lines))

    print(f"wrote {len(summary)} rows to {out_dir / 'summary_by_scenario.csv'}")
    print(f"wrote {len(overall)} rows to {out_dir / 'summary_overall.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', default='results/main.csv')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()

    build_summary(Path(args.main), Path(args.out))


if __name__ == '__main__':
    main()