"""Main experiment: 5 scenarios x N seeds x 4 baselines.

Baselines:
    B1  Full retrain every window (no warm start, no update)
    B2  Detect drift, then full retrain (no incremental)
    B3  Sliding window without update (freeze graph after warmup)
    B4  Ours: detect + incremental MB local re-learning + fallback
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
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


# ---------------------------------------------------------------------
# Baseline runners
# ---------------------------------------------------------------------

def _run_full_every_window(data, cfg, seed):
    """B1: refit every window with warm start (quality upper bound)."""
    ws = cfg.data.window_size
    n_windows = data.shape[0] // ws
    engine = DirectLiNGAMEngine(
        max_lag=cfg.data.max_lag,
        use_mlx=cfg.model.use_mlx,
        ridge_alpha=cfg.model.ridge_alpha,
        coeff_tol=cfg.model.coeff_tol,
        max_parents_per_node=cfg.model.max_parents_per_node,
    )
    rows = []
    prev_B = prev_order = None
    for w in range(n_windows):
        W = data[w * ws:(w + 1) * ws]
        t0 = time.perf_counter()
        engine.fit(W, warm_start_B=prev_B, warm_start_order=prev_order)
        rt = time.perf_counter() - t0
        A = engine.get_adjacency()
        B = engine.get_coefficients()
        prev_B = B
        prev_order = engine.get_causal_order()
        rows.append({
            'window': w, 'A': A, 'B': B,
            'mode': 'full', 'fallback': False,
            'update_time': 0.0, 'retrain_time': rt,
            'drift_score': 0.0, 'drift_confirmed': False,
        })
    return rows


def _run_detect_full(data, cfg, seed):
    """B2: same detector as ours, but on drift just use full result."""
    ws = cfg.data.window_size
    n_windows = data.shape[0] // ws
    engine = DirectLiNGAMEngine(
        max_lag=cfg.data.max_lag,
        use_mlx=cfg.model.use_mlx,
        ridge_alpha=cfg.model.ridge_alpha,
        coeff_tol=cfg.model.coeff_tol,
        max_parents_per_node=cfg.model.max_parents_per_node,
    )
    detector = DriftDetector(
        warmup=cfg.model.warmup,
        kappa=cfg.model.kappa,
        n_perm=cfg.model.n_perm,
        z_threshold=cfg.model.z_threshold,
        random_state=seed,
    )
    rows = []
    prev_A = prev_B = prev_R = None
    prev_order = None
    for w in range(n_windows):
        W = data[w * ws:(w + 1) * ws]
        t0 = time.perf_counter()
        engine.fit(W, warm_start_B=prev_B, warm_start_order=prev_order)
        rt = time.perf_counter() - t0
        A = engine.get_adjacency()
        B = engine.get_coefficients()
        R = engine.get_residuals(W)
        order = engine.get_causal_order()

        stats = compute_window_stats(A, B, R, prev_A, prev_B, prev_R)
        res = detector.update(stats)

        rows.append({
            'window': w, 'A': A, 'B': B,
            'mode': 'full', 'fallback': False,
            'update_time': 0.0, 'retrain_time': rt,
            'drift_score': res.score,
            'drift_confirmed': res.confirmed,
        })
        prev_A, prev_B, prev_R = A, B, R
        prev_order = order
    return rows


def _run_freeze(data, cfg, seed):
    """B3: freeze graph after the first window."""
    ws = cfg.data.window_size
    n_windows = data.shape[0] // ws
    engine = DirectLiNGAMEngine(
        max_lag=cfg.data.max_lag,
        use_mlx=cfg.model.use_mlx,
        ridge_alpha=cfg.model.ridge_alpha,
        coeff_tol=cfg.model.coeff_tol,
        max_parents_per_node=cfg.model.max_parents_per_node,
    )
    rows = []
    A0 = B0 = None
    for w in range(n_windows):
        W = data[w * ws:(w + 1) * ws]
        if w == 0:
            t0 = time.perf_counter()
            engine.fit(W)
            rt = time.perf_counter() - t0
            A0 = engine.get_adjacency()
            B0 = engine.get_coefficients()
        rows.append({
            'window': w, 'A': A0, 'B': B0,
            'mode': 'frozen', 'fallback': False,
            'update_time': 0.0, 'retrain_time': 0.0,
            'drift_score': 0.0, 'drift_confirmed': False,
        })
    return rows


def _run_ours(data, cfg, seed):
    """B4: full pipeline."""
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


BASELINE_RUNNERS = {
    'B1': _run_full_every_window,
    'B2': _run_detect_full,
    'B3': _run_freeze,
    'B4': _run_ours,
}


# ---------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------
def _true_drift_windows(meta, cfg, scenario):
    """Return windows considered 'drift' for detection metrics.

    Abrupt: single window where the change is observed.
    Gradual: two windows where the transition is observed, because
    the coefficient weight changes between (dw-1, dw) and (dw, dw+1).
    """
    dw = meta['drift_window']
    mode = meta['drift_mode']
    if mode == 'abrupt':
        return [dw]
    return [dw, dw + 1]

def _evaluate(rows, meta, cfg, scenario):
    ws = cfg.data.window_size
    L = cfg.data.max_lag
    drift_window = meta['drift_window']

    records = []
    drift_scores = np.array([r['drift_score'] for r in rows])
    true_drift = _true_drift_windows(meta, cfg, scenario)

    for r in rows:
        w = r['window']
        A_est = collapse_node_adj(r['A'], L)
        A_true = true_var_adj(meta, min(w, len(meta['true_inst_adj']) - 1))
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
    return records, drift_scores, true_drift


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results')
    parser.add_argument('--seeds', type=int, default=None)
    args = parser.parse_args()

    cfg = RunConfig()
    if args.seeds is not None:
        cfg.seeds = list(range(args.seeds))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
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
            for baseline in cfg.baselines:
                runner = BASELINE_RUNNERS[baseline]
                rows = runner(data, cfg, seed)
                records, drift_scores, true_drift = _evaluate(
                    rows, meta, cfg, scenario
                )
                det = detection_metrics(
                    drift_scores, true_drift
                )
                for rec in records:
                    rec['baseline'] = baseline
                    rec['seed'] = seed
                    rec['det_precision'] = det['precision']
                    rec['det_recall'] = det['recall']
                    rec['det_f1'] = det['f1']
                    all_rows.append(rec)
            print(f"[done] {scenario} seed={seed}")

    csv_path = out_dir / 'main.csv'
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"wrote {len(all_rows)} rows to {csv_path}")


if __name__ == '__main__':
    main()