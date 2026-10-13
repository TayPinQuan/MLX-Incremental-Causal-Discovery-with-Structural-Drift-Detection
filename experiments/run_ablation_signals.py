"""Signal ablation: A1-A5.

A1  ED only      residual energy distance
A2  ΔB only      coefficient change
A3  SHD only     structural Hamming distance
A4  ED + ΔB      two continuous signals
A5  all three    default in main experiment

Runs B4 (full pipeline) under each signal mode, one CSV per mode.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
import numpy as np

from data.synthetic import generate_drifting_data
from lingam_mlx import DirectLiNGAMEngine
from drift_detector import DriftDetector, compute_window_stats
from incremental import IncrementalUpdater
from online_causal import OnlineCausalDiscovery

from experiments.config import RunConfig, SCENARIOS
from experiments.metrics import (
    collapse_node_adj, true_var_adj,
    shd, edge_f1, coefficient_mse, detection_metrics,
)


MODE_LABELS = {
    'A1': 'ed_only',
    'A2': 'db_only',
    'A3': 'shd_only',
    'A4': 'ed_db',
    'A5': 'all',
    'A6': 'db_shd',
}


def _true_drift_windows(meta):
    dw = meta['drift_window']
    if meta['drift_mode'] == 'abrupt':
        return [dw]
    return [dw, dw + 1]


def _run_with_signal_mode(data, cfg, seed, signal_mode):
    """Run full pipeline with a specific signal_mode."""
    ocd = OnlineCausalDiscovery(
        window_size=cfg.data.window_size,
        max_lag=cfg.data.max_lag,
        warmup=cfg.model.warmup,
        kappa=cfg.model.kappa,
        n_perm=cfg.model.n_perm,
        z_threshold=cfg.model.z_threshold,
        use_mlx=cfg.model.use_mlx,
        ridge_alpha=cfg.model.ridge_alpha,
        coeff_tol=cfg.model.coeff_tol,
        max_parents_per_node=cfg.model.max_parents_per_node,
        residual_factor=cfg.model.residual_factor,
        coeff_explosion_factor=cfg.model.coeff_explosion_factor,
        shd_absolute_threshold=cfg.model.shd_absolute_threshold,
        max_local_fraction=cfg.model.max_local_fraction,
        min_block_edges=cfg.model.min_block_edges,
        random_state=seed,
    )
    # Override the detector's signal_mode
    ocd.detector_.signal_mode = signal_mode

    results = ocd.run(data)
    retrain_times = results['retrain_times']

    rows = []
    for r in results['records']:
        upd = r.update
        rows.append({
            'window': r.index,
            'A': r.A, 'B': r.B,
            'mode': upd.mode if upd else 'none',
            'fallback': upd.fallback if upd else False,
            'update_time': upd.update_time if upd else 0.0,
            'retrain_time': float(retrain_times[r.index]),
            'drift_score': r.drift_score,
            'drift_confirmed': r.drift_confirmed,
        })
    return rows


def _evaluate(rows, meta, cfg, scenario, mode_id):
    L = cfg.data.max_lag
    records = []
    drift_scores = np.array([r['drift_score'] for r in rows])
    true_drift = _true_drift_windows(meta)

    for r in rows:
        w = r['window']
        A_est = collapse_node_adj(r['A'], L)
        A_true = true_var_adj(
            meta, min(w, len(meta['true_inst_adj']) - 1)
        )
        s = shd(A_est, A_true)
        _, _, f1 = edge_f1(A_est, A_true)
        cmse = coefficient_mse(
            r['B'],
            meta['true_inst_coef'][min(w, len(meta['true_inst_coef']) - 1)],
            meta['true_lag_coef'][min(w, len(meta['true_lag_coef']) - 1)],
            L,
        )
        records.append({
            'scenario': scenario,
            'mode_id': mode_id,
            'signal_mode': MODE_LABELS[mode_id],
            'window': w,
            'shd': s,
            'f1': f1,
            'coef_mse': cmse,
            'mode': r['mode'],
            'fallback': int(r['fallback']),
            'update_time': r['update_time'],
            'retrain_time': r['retrain_time'],
            'drift_score': r['drift_score'],
            'drift_confirmed': int(r['drift_confirmed']),
        })

    det = detection_metrics(drift_scores, true_drift)
    for rec in records:
        rec['det_precision'] = det['precision']
        rec['det_recall'] = det['recall']
        rec['det_f1'] = det['f1']
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/ablation_signals')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument(
        '--modes', nargs='+', default=['A1', 'A2', 'A3', 'A4', 'A5'],
    )
    args = parser.parse_args()

    cfg = RunConfig()
    cfg.seeds = list(range(args.seeds))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for mode_id in args.modes:
        signal_mode = MODE_LABELS[mode_id]
        for scenario in cfg.scenarios:
            sconf = SCENARIOS[scenario]
            for seed in cfg.seeds:
                data, meta = generate_drifting_data(
                    n_samples=cfg.data.n_samples,
                    n_vars=cfg.data.n_vars,
                    max_lag=cfg.data.max_lag,
                    window_size=cfg.data.window_size,
                    drift_at=cfg.data.drift_at,
                    drift_type=sconf['drift_type'],
                    drift_mode=sconf['drift_mode'],
                    edge_prob=cfg.data.edge_prob,
                    noise_std=cfg.data.noise_std,
                    random_state=seed,
                )
                rows = _run_with_signal_mode(data, cfg, seed, signal_mode)
                records = _evaluate(rows, meta, cfg, scenario, mode_id)
                for rec in records:
                    rec['seed'] = seed
                    all_rows.append(rec)
            print(f"[done] {mode_id} ({signal_mode}) {scenario}")

    csv_path = out_dir / 'ablation_signals.csv'
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows to {csv_path}")


if __name__ == '__main__':
    main()