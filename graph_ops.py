"""Graph operations for causal structure manipulation.

All functions operate on binary adjacency matrices A where
A[i, j] = 1 means an edge i -> j.

For delay-embedded graphs, node indices are laid out as:
    index = var * (max_lag + 1) + lag
so that node `var*L + 0` is the instantaneous copy of `var`,
and `var*L + k` is the k-th lag copy.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable


# ---------------------------------------------------------------------
# Markov blanket
# ---------------------------------------------------------------------

def parents(A: np.ndarray, j: int) -> set[int]:
    return set(np.where(A[:, j] > 0)[0].tolist())


def children(A: np.ndarray, j: int) -> set[int]:
    return set(np.where(A[j, :] > 0)[0].tolist())


def markov_blanket(A: np.ndarray, j: int) -> set[int]:
    """Directed Markov blanket of node j.

    MB(j) = Pa(j) ∪ Ch(j) ∪ Pa(Ch(j)) \\ {j}
    """
    pa = parents(A, j)
    ch = children(A, j)
    pa_ch: set[int] = set()
    for c in ch:
        pa_ch |= parents(A, c)
    mb = (pa | ch | pa_ch) - {j}
    return mb


def expand_mb_set(A: np.ndarray, seeds: Iterable[int]) -> set[int]:
    """Union of MB(j) for j in seeds, plus the seeds themselves."""
    S: set[int] = set()
    for j in seeds:
        S.add(j)
        S |= markov_blanket(A, j)
    return S


# ---------------------------------------------------------------------
# DAG check
# ---------------------------------------------------------------------

def is_dag(A: np.ndarray) -> bool:
    """Kahn's algorithm on a binary adjacency matrix."""
    A = (np.asarray(A) > 0).astype(np.int32)
    n = A.shape[0]
    indeg = A.sum(axis=0).astype(np.int32)
    queue = [i for i in range(n) if indeg[i] == 0]
    seen = 0
    while queue:
        u = queue.pop()
        seen += 1
        for v in np.where(A[u, :] > 0)[0]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(int(v))
    return seen == n


# ---------------------------------------------------------------------
# Coefficient alignment
# ---------------------------------------------------------------------

def align_coefficients(
    B_prev: np.ndarray,
    B_curr: np.ndarray,
    A_prev: np.ndarray,
    A_curr: np.ndarray,
) -> np.ndarray:
    """Zero out entries in B_prev/B_curr where neither graph has an edge.

    Returns a copy of B_curr with the union mask applied. This makes
    coefficient comparison meaningful when the edge set changed.
    """
    mask = (A_prev > 0) | (A_curr > 0)
    B = np.array(B_curr, dtype=np.float64, copy=True)
    B[~mask] = 0.0
    return B


# ---------------------------------------------------------------------
# Subgraph extraction and merge
# ---------------------------------------------------------------------

def submatrix(A: np.ndarray, S: Iterable[int]) -> np.ndarray:
    idx = sorted(set(S))
    return A[np.ix_(idx, idx)]


def submatrix_B(B: np.ndarray, S: Iterable[int]) -> np.ndarray:
    idx = sorted(set(S))
    return B[np.ix_(idx, idx)]


def merge_subgraph(
    A_global: np.ndarray,
    B_global: np.ndarray,
    A_local: np.ndarray,
    B_local: np.ndarray,
    S: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Replace the submatrix on S with the locally relearned one.

    Edges from S to outside and outside to S are preserved from
    the global graph. This is the known limitation of local relearning:
    new edges crossing the boundary are not discovered here, and are
    handled by the fallback consistency check.
    """
    idx = sorted(set(S))
    A_new = np.array(A_global, copy=True)
    B_new = np.array(B_global, copy=True)
    A_new[np.ix_(idx, idx)] = A_local
    B_new[np.ix_(idx, idx)] = B_local
    return A_new, B_new


# ---------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------

def check_no_cycle(A: np.ndarray) -> bool:
    return is_dag(A)


def check_shd_jump(
    shd_t: float, shd_history: np.ndarray, quantile: float = 0.99
) -> bool:
    """True if shd_t exceeds the given quantile of history."""
    if shd_history.size == 0:
        return False
    thr = np.quantile(shd_history, quantile)
    return shd_t > thr


def check_coefficient_explosion(
    B_new: np.ndarray, B_prev: np.ndarray, factor: float = 3.0
) -> bool:
    """True if ||B_new||_F > factor * ||B_prev||_F."""
    n_new = np.linalg.norm(B_new)
    n_prev = np.linalg.norm(B_prev) + 1e-12
    return n_new > factor * n_prev


# ---------------------------------------------------------------------
# Delay-embedding index helpers
# ---------------------------------------------------------------------

def node_index(var: int, lag: int, max_lag: int) -> int:
    return var * (max_lag + 1) + lag


def node_label(var: int, lag: int) -> str:
    return f"X{var}(t-{lag})" if lag > 0 else f"X{var}(t)"


def all_node_labels(n_vars: int, max_lag: int) -> list[str]:
    labels = []
    for lag in range(max_lag + 1):
        for var in range(n_vars):
            labels.append(node_label(var, lag))
    return labels