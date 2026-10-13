"""Statistical primitives for drift detection.

All functions are NumPy-only. No MLX here, because these operate on
small arrays (window-level statistics), where MLX overhead is not
justified. MLX is reserved for the LiNGAM engine's linear algebra.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------
# Energy distance
# ---------------------------------------------------------------------

def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Symmetric energy distance between two samples.

    ED(X, Y) = 2 E||x - y|| - E||x - x'|| - E||y - y'||

    Parameters
    ----------
    X, Y : (n, d) and (m, d) arrays. d must match. n and m may differ.

    Returns
    -------
    Scalar energy distance >= 0.

    Notes
    -----
    - Uses pairwise Euclidean distances via broadcasting.
    - O(n^2 + m^2 + nm) memory. For n=m=500 this is fine.
    - If a sample has < 2 rows, returns 0.0 (undefined distance).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"dim mismatch: {X.shape[1]} vs {Y.shape[1]}")
    if X.shape[0] < 2 or Y.shape[0] < 2:
        return 0.0

    def _mean_pairwise(A: np.ndarray, B: np.ndarray) -> float:
        # ||a - b|| for all pairs
        diff = A[:, None, :] - B[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        return float(dist.mean())

    xy = _mean_pairwise(X, Y)
    xx = _mean_pairwise(X, X)
    yy = _mean_pairwise(Y, Y)
    ed = 2.0 * xy - xx - yy
    return max(ed, 0.0)


def energy_distance_per_variable(
    R_prev: np.ndarray, R_curr: np.ndarray
) -> np.ndarray:
    """Per-column energy distance between two residual matrices.

    Parameters
    ----------
    R_prev, R_curr : (n, d) arrays. Same d. n may differ.

    Returns
    -------
    (d,) array of energy distances.
    """
    R_prev = np.asarray(R_prev, dtype=np.float64)
    R_curr = np.asarray(R_curr, dtype=np.float64)
    if R_prev.shape[1] != R_curr.shape[1]:
        raise ValueError("residual dim mismatch")
    d = R_prev.shape[1]
    out = np.zeros(d, dtype=np.float64)
    for j in range(d):
        out[j] = energy_distance(R_prev[:, j], R_curr[:, j])
    return out


# ---------------------------------------------------------------------
# Coefficient change
# ---------------------------------------------------------------------

def coefficient_change(
    B_prev: np.ndarray,
    B_curr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Relative Frobenius change between coefficient matrices.

    ||B_curr - B_prev||_F / (||B_prev||_F + eps)

    Parameters
    ----------
    B_prev, B_curr : (d, d) coefficient matrices.
    mask : optional (d, d) boolean. If given, only entries where
           mask is True are compared. Use A_prev | A_curr to focus
           on edges present in either graph.

    Returns
    -------
    Non-negative scalar.
    """
    B_prev = np.asarray(B_prev, dtype=np.float64)
    B_curr = np.asarray(B_curr, dtype=np.float64)
    if B_prev.shape != B_curr.shape:
        raise ValueError("coefficient shape mismatch")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != B_prev.shape:
            raise ValueError("mask shape mismatch")
        num = np.linalg.norm((B_curr - B_prev)[mask])
        den = np.linalg.norm(B_prev[mask]) + 1e-12
    else:
        num = np.linalg.norm(B_curr - B_prev)
        den = np.linalg.norm(B_prev) + 1e-12
    return float(num / den)


def coefficient_change_per_variable(
    B_prev: np.ndarray,
    B_curr: np.ndarray,
    A_prev: np.ndarray,
    A_curr: np.ndarray,
) -> np.ndarray:
    """Per-column coefficient change, aligned to incoming edges.

    For column j, sum |B_curr[i,j] - B_prev[i,j]| over i where
    A_prev[i,j] or A_curr[i,j] is 1, normalized by the same sum
    for B_prev.

    Returns
    -------
    (d,) array of non-negative scores.
    """
    d = B_prev.shape[0]
    out = np.zeros(d, dtype=np.float64)
    for j in range(d):
        incoming = (A_prev[:, j] > 0) | (A_curr[:, j] > 0)
        if not incoming.any():
            continue
        num = np.sum(np.abs(B_curr[incoming, j] - B_prev[incoming, j]))
        den = np.sum(np.abs(B_prev[incoming, j])) + 1e-12
        out[j] = num / den
    return out


# ---------------------------------------------------------------------
# Structural Hamming distance
# ---------------------------------------------------------------------

def structural_hamming(A_prev: np.ndarray, A_curr: np.ndarray) -> int:
    """SHD between two binary adjacency matrices of the same shape."""
    A_prev = np.asarray(A_prev)
    A_curr = np.asarray(A_curr)
    if A_prev.shape != A_curr.shape:
        raise ValueError("adjacency shape mismatch")
    return int(np.sum(A_prev != A_curr))


def shd_per_variable(
    A_prev: np.ndarray, A_curr: np.ndarray
) -> np.ndarray:
    """Per-variable SHD: sum of incoming and outgoing edge changes."""
    d = A_prev.shape[0]
    out = np.zeros(d, dtype=np.float64)
    diff = (A_prev != A_curr).astype(np.float64)
    for j in range(d):
        out[j] = diff[:, j].sum() + diff[j, :].sum()
    return out


# ---------------------------------------------------------------------
# Standardization and fusion
# ---------------------------------------------------------------------

def standardize(scores: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Z-score with precomputed mean and std."""
    sigma = max(sigma, 1e-12)
    return (np.asarray(scores, dtype=np.float64) - mu) / sigma


def variable_drift_scores(
    ed_per_var: np.ndarray,
    delta_b_per_var: np.ndarray,
    shd_per_var: np.ndarray,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Fuse three per-variable signals into one score.

    Each input is standardized internally by its own std across
    variables, so that no single signal dominates by scale.
    """
    def _z(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        s = x.std()
        if s < 1e-12:
            return np.zeros_like(x)
        return (x - x.mean()) / s

    w1, w2, w3 = weights
    return w1 * _z(ed_per_var) + w2 * _z(delta_b_per_var) + w3 * _z(shd_per_var)


# ---------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------

def permutation_test_energy(
    R_prev: np.ndarray,
    R_curr: np.ndarray,
    n_perm: int = 200,
    random_state: Optional[int] = None,
) -> tuple[float, float]:
    """Permutation test on energy distance between two residual matrices.

    Returns
    -------
    (stat, p_value)
        stat : observed energy distance
        p_value : fraction of permutations with ED >= observed
    """
    rng = np.random.default_rng(random_state)
    R_prev = np.asarray(R_prev, dtype=np.float64)
    R_curr = np.asarray(R_curr, dtype=np.float64)
    if R_prev.shape[1] != R_curr.shape[1]:
        raise ValueError("residual dim mismatch")

    observed = energy_distance(R_prev, R_curr)
    pooled = np.vstack([R_prev, R_curr])
    n_prev = R_prev.shape[0]
    n_total = pooled.shape[0]
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(n_total)
        a = pooled[idx[:n_prev]]
        b = pooled[idx[n_prev:]]
        if energy_distance(a, b) >= observed:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(observed), float(p)