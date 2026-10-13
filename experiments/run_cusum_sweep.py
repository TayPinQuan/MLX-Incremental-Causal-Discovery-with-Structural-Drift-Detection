"""CUSUM parameter sweep for the delay-FPR curve.

For each kappa in a grid, run the full pipeline on all scenarios and
record:
    - detection delay on the primary drift window (w=drift_window)
    - false positive rate on quiet windows (excluding warmup and the
      primary/secondary drift windows)

Output: results/cusum_sweep.csv with columns
    scenario, kappa, fpr, delay, precision, recall, f1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from data.synthetic import generate_drifting_data
from online_causal import OnlineCausalDiscovery

from experiments.config import RunConfig, SCENARIOS


def _true_drift_windows(meta):
    dw = meta['drift_window']
    if meta['drift_mode'] == 'abrupt':
        return [dw]
    return [dw, dw + 1]


def _evaluate_run(drift_scores, drift_confirmed, meta, kappa):
    """Compute delay, FPR, precision, recall, F1 for a single run.

    - delay: first confirmed window >= drift_window minus drift_window.
             If never confirmed on or after drift_window, delay = -1.
    - FPR: fraction of quiet windows (w >= warmup_end and w not in
           true_drift) that got confirmed.
    """
    n = len(drift_scores)
    true_drift = set(_true_drift_windows(meta))
    dw = meta['drift_window']

    # Delay: first confirmed window at or after dw
    delay = -1
    for w in range(dw, n):
        if drift_confirmed[w]:
            delay = w - dw
            break

    # FPR: quiet windows = not in true_drift and w >= 3
    quiet = [w for w in range(3, n) if w not in true_drift]
    fp = sum(1 for w in quiet if drift_confirmed[w])
    fpr = fp / len(quiet) if quiet else 0.0

    # Per-window detection labels
    labels = np.zeros(n, dtype=int)
    for w in true_drift:
        if 0 <= w < n:
            labels[w] = 1
    preds = np.array(drift_confirmed, dtype=int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp_all = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    prec = tp / (tp + fp_all) if (tp + fp_all) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return delay, fpr, prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument(
        '--kappas', type=float, nargs='+',
        default=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    args = parser.parse_args()

    cfg = RunConfig()
    cfg.seeds = list(range(args.seeds))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for kappa in args.kappas:
        for scenario in cfg.scenarios:
            sconf = SCENARIOS[scenario]
            delays: list[int] = []
            fprs: list[float] = []
            precs: list[float] = []
            recs: list[float] = []
            f1s: list[float] = []

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

                ocd = OnlineCausalDiscovery(
                    window_size=cfg.data.window_size,
                    max_lag=cfg.data.max_lag,
                    warmup=cfg.model.warmup,
                    kappa=kappa,
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
                drift_scores = results['drift_scores']
                drift_confirmed = [r.drift_confirmed for r in results['records']]

                delay, fpr, prec, rec, f1 = _evaluate_run(
                    drift_scores, drift_confirmed, meta, kappa
                )
                delays.append(delay)
                fprs.append(fpr)
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)

            # Delay: ignore -1 (no detection) in mean, but count rate
            valid_delays = [d for d in delays if d >= 0]
            mean_delay = float(np.mean(valid_delays)) if valid_delays else float('nan')
            missed = sum(1 for d in delays if d < 0)

            rows.append({
                'scenario': scenario,
                'kappa': kappa,
                'delay_mean': mean_delay,
                'delay_std': float(np.std(valid_delays)) if valid_delays else float('nan'),
                'missed': missed,
                'n_seeds': len(delays),
                'fpr_mean': float(np.mean(fprs)),
                'fpr_std': float(np.std(fprs)),
                'precision_mean': float(np.mean(precs)),
                'recall_mean': float(np.mean(recs)),
                'f1_mean': float(np.mean(f1s)),
            })
            print(f"[kappa={kappa}] {scenario} delay={mean_delay:.2f} "
                  f"fpr={np.mean(fprs):.3f} missed={missed}/{len(delays)}")

    csv_path = out_dir / 'cusum_sweep.csv'
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {csv_path}")


if __name__ == '__main__':
    main()