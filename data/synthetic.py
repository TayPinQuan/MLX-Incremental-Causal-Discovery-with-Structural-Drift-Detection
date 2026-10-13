"""Synthetic drifting time-series generator.

Layout convention (must match graph_ops.node_index):
    node_index(var, lag, max_lag) = var * (max_lag + 1) + lag

Data generating process (per time step t):
    x_j(t) = sum_{i in Pa_inst(j)} B_inst[i, j] * x_i(t)
           + sum_{l=1..max_lag} sum_{i in Pa_lag(j, l)} B_lag[l][i, j] * x_i(t - l)
           + eps_j(t)
    eps ~ N(0, noise_std^2)

Drift is defined at the window level: each window has its own
instantaneous and lagged coefficient matrices. Abrupt drift switches
coefficients at the drift window; gradual drift linearly interpolates
over a span of windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np


# ---------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------

def _random_dag(
    n: int, edge_prob: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random DAG with a random topological order."""
    topo = rng.permutation(n)
    A = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < edge_prob:
                A[topo[a], topo[b]] = 1.0
    return A, topo


def _random_coefficients(
    A: np.ndarray, rng: np.random.Generator,
    low: float = 0.3, high: float = 1.0,
) -> np.ndarray:
    """Non-zero coefficients with random signs and magnitudes."""
    B = np.zeros_like(A, dtype=np.float64)
    mask = A > 0
    n_edges = int(mask.sum())
    if n_edges == 0:
        return B
    signs = rng.choice([-1.0, 1.0], size=n_edges)
    mags = rng.uniform(low, high, size=n_edges)
    B[mask] = signs * mags
    return B


def _stabilize_lag_matrix(
    B: np.ndarray, max_col_sum: float = 0.5
) -> np.ndarray:
    """Scale columns so max |column sum| <= max_col_sum.

    For the recursion x_j(t) = sum_i B[i, j] * x_i(t-1) + ..., the
    per-target stability condition is sum_i |B[i, j]| < 1.
    """
    col_abs_sum = np.abs(B).sum(axis=0)
    scale = np.maximum(col_abs_sum / max_col_sum, 1.0)
    return B / scale[None, :]

def _ensure_stability(
    B_inst: np.ndarray,
    B_lag_list: list[np.ndarray],
    target_rho: float = 0.9,
) -> list[np.ndarray]:
    """Scale lag matrices so the companion matrix has spectral radius
    below target_rho. Accounts for instantaneous amplification via
    (I - B_inst)^{-1}.
    """
    p = B_inst.shape[0]
    L = len(B_lag_list)
    if L == 0:
        return B_lag_list

    # (I - B_inst) may be near-singular if B_inst has huge column sums.
    I = np.eye(p)
    try:
        M = np.linalg.inv(I - B_inst)
    except np.linalg.LinAlgError:
        # Fall back: shrink B_inst first
        B_inst_scaled = B_inst * 0.5
        M = np.linalg.inv(I - B_inst_scaled)

    eff_lag = [M @ B_l for B_l in B_lag_list]

    # Companion matrix of size p*L
    C = np.zeros((p * L, p * L), dtype=np.float64)
    for l, E in enumerate(eff_lag):
        C[:p, l * p:(l + 1) * p] = E
    for l in range(1, L):
        C[l * p:(l + 1) * p, (l - 1) * p:l * p] = I

    try:
        rho = float(np.max(np.abs(np.linalg.eigvals(C))))
    except np.linalg.LinAlgError:
        rho = np.inf

    if not np.isfinite(rho) or rho <= target_rho:
        return B_lag_list

    scale = target_rho / rho
    return [B_l * scale for B_l in B_lag_list]


def _topo_order(A: np.ndarray) -> np.ndarray:
    """Kahn's algorithm; assumes A is a DAG."""
    A = (A > 0).astype(np.int32)
    n = A.shape[0]
    indeg = A.sum(axis=0).astype(np.int32)
    queue = [i for i in range(n) if indeg[i] == 0]
    order: list[int] = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in np.where(A[u, :] > 0)[0]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(int(v))
    if len(order) != n:
        raise ValueError("A is not a DAG")
    return np.array(order, dtype=int)


# ---------------------------------------------------------------------
# Drift mutation helpers
# ---------------------------------------------------------------------

def _mutate_structure(
    A: np.ndarray, B: np.ndarray, topo: np.ndarray,
    rng: np.random.Generator, p_add: float = 0.15, p_del: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Add and remove edges while staying a DAG."""
    A_new = A.copy()
    B_new = B.copy()

    edges = np.argwhere(A_new > 0)
    for i, j in edges:
        if rng.random() < p_del:
            A_new[i, j] = 0.0
            B_new[i, j] = 0.0

    n = A_new.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            i, j = int(topo[a]), int(topo[b])
            if A_new[i, j] == 0 and rng.random() < p_add:
                A_new[i, j] = 1.0
                sign = rng.choice([-1.0, 1.0])
                mag = rng.uniform(0.3, 1.0)
                B_new[i, j] = sign * mag
    return A_new, B_new


def _mutate_coefficients(
    B: np.ndarray, A: np.ndarray, rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.5, 1.8),
) -> np.ndarray:
    """Perturb existing non-zero coefficients multiplicatively."""
    B_new = B.copy()
    mask = A > 0
    factors = rng.uniform(scale_range[0], scale_range[1], size=int(mask.sum()))
    B_new[mask] = B_new[mask] * factors
    return B_new


# ---------------------------------------------------------------------
# Window-level coefficient trajectories
# ---------------------------------------------------------------------

@dataclass
class WindowCoefficients:
    B_inst: np.ndarray
    B_lag: list[np.ndarray]
    A_inst: np.ndarray
    A_lag: list[np.ndarray]


def _build_window_coefficients(
    B_inst_0: np.ndarray, B_lag_0: list[np.ndarray],
    B_inst_1: np.ndarray, B_lag_1: list[np.ndarray],
    A_inst_0: np.ndarray, A_lag_0: list[np.ndarray],
    A_inst_1: np.ndarray, A_lag_1: list[np.ndarray],
    n_windows: int, drift_window: int, drift_mode: str,
    gradual_span: int = 3,
) -> list[WindowCoefficients]:
    """Return per-window coefficients and adjacency."""
    out: list[WindowCoefficients] = []

    if drift_mode == "abrupt":
        for w in range(n_windows):
            if w < drift_window:
                out.append(WindowCoefficients(
                    B_inst=B_inst_0.copy(),
                    B_lag=[B.copy() for B in B_lag_0],
                    A_inst=A_inst_0.copy(),
                    A_lag=[A.copy() for A in A_lag_0],
                ))
            else:
                out.append(WindowCoefficients(
                    B_inst=B_inst_1.copy(),
                    B_lag=[B.copy() for B in B_lag_1],
                    A_inst=A_inst_1.copy(),
                    A_lag=[A.copy() for A in A_lag_1],
                ))
    elif drift_mode == "gradual":
        start = drift_window - gradual_span // 2
        end = drift_window + gradual_span // 2
        for w in range(n_windows):
            if w <= start:
                wt = 0.0
            elif w >= end:
                wt = 1.0
            else:
                wt = (w - start) / (end - start)

            B_inst_w = (1 - wt) * B_inst_0 + wt * B_inst_1
            B_lag_w = [
                (1 - wt) * B0 + wt * B1
                for B0, B1 in zip(B_lag_0, B_lag_1)
            ]
            # A is derived from B's non-zero pattern so that
            # adjacency and coefficients describe the same system.
            A_inst_w = (np.abs(B_inst_w) > 1e-12).astype(np.float64)
            A_lag_w = [
                (np.abs(B_w) > 1e-12).astype(np.float64)
                for B_w in B_lag_w
            ]

            out.append(WindowCoefficients(
                B_inst=B_inst_w, B_lag=B_lag_w,
                A_inst=A_inst_w, A_lag=A_lag_w,
            ))
    else:
        raise ValueError(f"unknown drift_mode: {drift_mode}")
    return out


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

def _simulate(
    window_coeffs: list[WindowCoefficients],
    topo: np.ndarray,
    n_samples: int,
    max_lag: int,
    window_size: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_vars = window_coeffs[0].B_inst.shape[0]
    X = np.zeros((n_samples, n_vars), dtype=np.float64)

    X[:max_lag] = rng.normal(0, noise_std, size=(max_lag, n_vars))

    for t in range(max_lag, n_samples):
        w = min(t // window_size, len(window_coeffs) - 1)
        wc = window_coeffs[w]
        for j in topo:
            val = 0.0
            for i in np.where(wc.B_inst[:, j] != 0)[0]:
                val += wc.B_inst[i, j] * X[t, i]
            for lag_idx, B_lag in enumerate(wc.B_lag, start=1):
                for i in np.where(B_lag[:, j] != 0)[0]:
                    val += B_lag[i, j] * X[t - lag_idx, i]
            X[t, j] = val + rng.normal(0, noise_std)

        if not np.isfinite(X[t]).all():
            raise RuntimeError(
                f"non-finite value at t={t}, window={w}; "
                f"check coefficient stability"
            )
    return X


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def generate_drifting_data(
    n_samples: int = 5000,
    n_vars: int = 8,
    max_lag: int = 2,
    window_size: int = 500,
    drift_at: int = 2500,
    drift_type: Literal["structure", "coefficients", "both"] = "both",
    drift_mode: Literal["abrupt", "gradual"] = "abrupt",
    edge_prob: float = 0.2,
    noise_std: float = 0.5,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a drifting time series."""
    rng = np.random.default_rng(random_state)
    n_windows = n_samples // window_size
    drift_window = drift_at // window_size

    # Initial structure
    A_inst_0, topo = _random_dag(n_vars, edge_prob, rng)
    B_inst_0 = _random_coefficients(A_inst_0, rng)

    A_lag_0: list[np.ndarray] = []
    B_lag_0: list[np.ndarray] = []
    for _ in range(max_lag):
        A_l = (rng.random((n_vars, n_vars)) < edge_prob).astype(np.float64)
        np.fill_diagonal(A_l, 0.0)
        B_l = _random_coefficients(A_l, rng)
        A_lag_0.append(A_l)
        B_lag_0.append(B_l)

    # Stabilize B_lag_0 BEFORE mutation, so drift magnitudes are
    # comparable to the initial magnitudes.
    # Initial stabilization: per-column cap, then companion-based scaling
    B_lag_0 = [_stabilize_lag_matrix(B, max_col_sum=0.5) for B in B_lag_0]
    B_lag_0 = _ensure_stability(B_inst_0, B_lag_0, target_rho=0.9)

    # Drifted structure
    A_inst_1 = A_inst_0.copy()
    B_inst_1 = B_inst_0.copy()
    if drift_type in ("structure", "both"):
        A_inst_1, B_inst_1 = _mutate_structure(
            A_inst_0, B_inst_0, topo, rng
        )
    if drift_type in ("coefficients", "both"):
        B_inst_1 = _mutate_coefficients(B_inst_1, A_inst_1, rng)

    A_lag_1 = [A.copy() for A in A_lag_0]
    B_lag_1 = [B.copy() for B in B_lag_0]
    if drift_type in ("structure", "both"):
        for l in range(max_lag):
            A_lag_1[l], B_lag_1[l] = _mutate_structure(
                A_lag_0[l], B_lag_0[l], topo, rng,
                p_add=0.10, p_del=0.10,
            )
    if drift_type in ("coefficients", "both"):
        for l in range(max_lag):
            B_lag_1[l] = _mutate_coefficients(
                B_lag_1[l], A_lag_1[l], rng
            )
    # Post-mutation stabilization
    B_lag_1 = [_stabilize_lag_matrix(B, max_col_sum=0.5) for B in B_lag_1]
    B_lag_1 = _ensure_stability(B_inst_1, B_lag_1, target_rho=0.9)

    window_coeffs = _build_window_coefficients(
        B_inst_0, B_lag_0, B_inst_1, B_lag_1,
        A_inst_0, A_lag_0, A_inst_1, A_lag_1,
        n_windows=n_windows,
        drift_window=drift_window,
        drift_mode=drift_mode,
        gradual_span=3,
    )

    data = _simulate(
        window_coeffs=window_coeffs,
        topo=topo,
        n_samples=n_samples,
        max_lag=max_lag,
        window_size=window_size,
        noise_std=noise_std,
        rng=rng,
    )

    meta = {
        "window_size": window_size,
        "max_lag": max_lag,
        "n_vars": n_vars,
        "n_windows": n_windows,
        "drift_at": drift_at,
        "drift_window": drift_window,
        "drift_type": drift_type,
        "drift_mode": drift_mode,
        "true_inst_adj": [wc.A_inst for wc in window_coeffs],
        "true_lag_adj": [wc.A_lag for wc in window_coeffs],
        "true_inst_coef": [wc.B_inst for wc in window_coeffs],
        "true_lag_coef": [wc.B_lag for wc in window_coeffs],
        "topo_order": topo,
    }
    return data, meta