"""Metrics against ground truth."""

from __future__ import annotations

import numpy as np


def collapse_node_adj(A: np.ndarray, max_lag: int) -> np.ndarray:
    """Collapse a node-level (d, d) adjacency to variable-level (p, p)."""
    d = A.shape[0]
    L = max_lag
    p = d // (L + 1)
    A_var = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            block = A[
                i * (L + 1):(i + 1) * (L + 1),
                j * (L + 1):(j + 1) * (L + 1),
            ]
            if block.max() > 0:
                A_var[i, j] = 1.0
    return A_var


def true_var_adj(meta: dict, window: int) -> np.ndarray:
    """True variable-level adjacency for a given window."""
    A = meta['true_inst_adj'][window].copy()
    for A_l in meta['true_lag_adj'][window]:
        A = np.maximum(A, A_l)
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(np.float64)


def shd(A_est: np.ndarray, A_true: np.ndarray) -> int:
    return int(np.sum((A_est > 0) != (A_true > 0)))


def edge_f1(A_est: np.ndarray, A_true: np.ndarray) -> tuple[float, float, float]:
    est = (A_est > 0)
    tru = (A_true > 0)
    tp = int(np.sum(est & tru))
    fp = int(np.sum(est & ~tru))
    fn = int(np.sum(~est & tru))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def coefficient_mse(
    B_est: np.ndarray,
    true_inst: np.ndarray,
    true_lag_list: list[np.ndarray],
    max_lag: int,
) -> float:
    """MSE on true-edge positions only, variable-level."""
    p = true_inst.shape[0]
    L = max_lag
    errs: list[float] = []
    for j in range(p):
        # Instantaneous
        for i in range(p):
            if true_inst[i, j] != 0:
                est = B_est[i * (L + 1) + 0, j * (L + 1) + 0]
                errs.append(float(est - true_inst[i, j]) ** 2)
        # Lagged
        for lag_idx, A_l in enumerate(true_lag_list, start=1):
            for i in range(p):
                if A_l[i, j] != 0:
                    est = B_est[
                        i * (L + 1) + lag_idx, j * (L + 1) + 0
                    ]
                    errs.append(float(est) ** 2)
    if not errs:
        return 0.0
    return float(np.mean(errs))


def detection_metrics(
    drift_scores: np.ndarray,
    true_drift_windows: list[int],
    threshold: float | None = None,
) -> dict:
    """Per-window detection metrics.

    If threshold is None, uses `mean + 2 * std` of drift scores
    computed on non-drift windows only.
    """
    n = len(drift_scores)
    labels = np.zeros(n, dtype=int)
    for w in true_drift_windows:
        if 0 <= w < n:
            labels[w] = 1

    if threshold is None:
        non_drift = drift_scores[labels == 0]
        if non_drift.size >= 2:
            mu = float(non_drift.mean())
            sd = float(non_drift.std())
            threshold = max(mu + 2 * sd, 0.0)
        elif non_drift.size == 1:
            threshold = float(non_drift[0])
        else:
            threshold = 0.0

    preds = (drift_scores > threshold).astype(int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'threshold': float(threshold),
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
    }