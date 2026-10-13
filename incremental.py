"""Incremental update via Markov-blanket local re-learning.

Pipeline:
    1. Collapse the delay-embedded node graph to a variable-level graph.
    2. Expand changed variables to their Markov blanket (MB) on the
       variable-level graph.
    3. Project the MB variable set back to node indices.
    4. Re-fit DirectLiNGAM only on X[:, S_vars], warm-started with the
       previous window's coefficients and order restricted to S.
    5. Merge the local subgraph back into the global graph.
    6. Run consistency checks; on failure, fall back to full retrain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np

from graph_ops import is_dag, check_coefficient_explosion
from stats import structural_hamming
from lingam_mlx import DirectLiNGAMEngine


# ---------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------

@dataclass
class UpdateResult:
    mode: str
    fallback: bool
    A: np.ndarray
    B: np.ndarray
    changed_vars: list[int] = field(default_factory=list)
    mb_set: list[int] = field(default_factory=list)
    update_time: float = 0.0
    retrain_time: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------

def _submatrix_index(S_vars: list[int], max_lag: int) -> list[int]:
    idx: list[int] = []
    for v in S_vars:
        for lag in range(max_lag + 1):
            idx.append(v * (max_lag + 1) + lag)
    return idx


def _extract_submatrix_B(
    B: np.ndarray, S_vars: list[int], max_lag: int
) -> np.ndarray:
    idx = _submatrix_index(S_vars, max_lag)
    return B[np.ix_(idx, idx)]


def _merge_local_into_global(
    A_full: np.ndarray,
    B_full: np.ndarray,
    A_loc: np.ndarray,
    B_loc: np.ndarray,
    S_vars: list[int],
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = _submatrix_index(S_vars, max_lag)
    A_new = A_full.copy()
    B_new = B_full.copy()
    A_new[np.ix_(idx, idx)] = A_loc
    B_new[np.ix_(idx, idx)] = B_loc
    return A_new, B_new


def _remap_order(
    global_order: list[int],
    S_vars: list[int],
    max_lag: int,
) -> Optional[list[int]]:
    var_to_local = {v: i for i, v in enumerate(S_vars)}
    L = max_lag
    filtered: list[int] = []
    for u in global_order:
        v = u // (L + 1)
        lag = u % (L + 1)
        if v in var_to_local:
            filtered.append(var_to_local[v] * (L + 1) + lag)

    expected = set(range(len(S_vars) * (L + 1)))
    if set(filtered) != expected or len(filtered) != len(expected):
        return None
    return filtered


def _residuals_with_B(
    engine: DirectLiNGAMEngine, X: np.ndarray, B: np.ndarray
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X_emb = engine._delay_embed(X)
    pred = X_emb @ B
    L = engine.max_lag
    p = engine._n_vars
    residuals = np.zeros((X_emb.shape[0], p), dtype=np.float64)
    for var in range(p):
        node = var * (L + 1) + 0
        residuals[:, var] = X_emb[:, node] - pred[:, node]
    return residuals


# ---------------------------------------------------------------------
# Variable-level graph helpers
# ---------------------------------------------------------------------

def _collapse_to_var_graph(
    A: np.ndarray, max_lag: int, min_block_edges: int = 2
) -> np.ndarray:
    """Collapse delay-embedded node graph to a variable-level graph.

    A_var[i, j] = 1 iff at least `min_block_edges` edges exist in
    the block from lag copies of i to lag copies of j. Requiring at
    least 2 edges suppresses isolated noise edges.
    """
    d = A.shape[0]
    L = max_lag
    n_vars = d // (L + 1)
    A_var = np.zeros((n_vars, n_vars), dtype=np.float64)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            block = A[
                i * (L + 1):(i + 1) * (L + 1),
                j * (L + 1):(j + 1) * (L + 1),
            ]
            if int(block.sum()) >= min_block_edges:
                A_var[i, j] = 1.0
    return A_var


def _var_markov_blanket(A_var: np.ndarray, j: int) -> set[int]:
    """Directed Markov blanket on a variable-level DAG.

    MB(j) = Pa(j) ∪ Ch(j) ∪ Pa(Ch(j)) \\ {j}
    """
    pa = set(np.where(A_var[:, j] > 0)[0].tolist())
    ch = set(np.where(A_var[j, :] > 0)[0].tolist())
    pa_ch: set[int] = set()
    for c in ch:
        pa_ch |= set(np.where(A_var[:, c] > 0)[0].tolist())
    return (pa | ch | pa_ch) - {j}


# ---------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------

class IncrementalUpdater:
    """Markov-blanket local re-learning with consistency-checked fallback."""

    def __init__(
        self,
        residual_factor: float = 2.0,
        coeff_explosion_factor: float = 3.0,
        shd_absolute_threshold: float = 25.0,
        shd_quantile: float = 0.95,
        min_local_vars: int = 1,
        max_local_fraction: float = 0.75,
        min_block_edges: int = 3,
        mb_mode: str = 'mb',
    ):
        valid_modes = {'seeds', 'parents', 'mb', 'full'}
        if mb_mode not in valid_modes:
            raise ValueError(
                f"mb_mode must be one of {valid_modes}, got {mb_mode}"
            )
        self.mb_mode = mb_mode
        self.residual_factor = float(residual_factor)
        self.coeff_explosion_factor = float(coeff_explosion_factor)
        self.shd_absolute_threshold = float(shd_absolute_threshold)
        self.shd_quantile = float(shd_quantile)
        self.min_local_vars = int(min_local_vars)
        self.max_local_fraction = float(max_local_fraction)
        self.min_block_edges = int(min_block_edges)

    # -----------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------

    def update(
        self,
        engine: DirectLiNGAMEngine,
        window: np.ndarray,
        drift_result,
        prev_B: Optional[np.ndarray],
        prev_A: Optional[np.ndarray],
        prev_order: Optional[list[int]],
        prev_residuals: Optional[np.ndarray],
        shd_history: np.ndarray,
        retrain_time: float,
    ) -> UpdateResult:
        A_full = engine.get_adjacency()
        B_full = engine.get_coefficients()

        if not drift_result.confirmed:
            return UpdateResult(
                mode='none', fallback=False,
                A=A_full, B=B_full, retrain_time=retrain_time,
                reason='no_drift',
            )
        if not drift_result.changed_vars:
            return UpdateResult(
                mode='none', fallback=False,
                A=A_full, B=B_full, retrain_time=retrain_time,
                reason='no_changed_vars',
            )
        if prev_A is None or prev_B is None or prev_order is None:
            return UpdateResult(
                mode='full_retrain', fallback=True,
                A=A_full, B=B_full,
                changed_vars=list(drift_result.changed_vars),
                retrain_time=retrain_time,
                reason='no_prev_state',
            )

        L = engine.max_lag
        n_vars = engine._n_vars

        # 1. Variable-level MB expansion
        # 0. Full mode: immediate fallback (equivalent to B2)
        if self.mb_mode == 'full':
            return UpdateResult(
                mode='full_retrain', fallback=True,
                A=A_full, B=B_full,
                changed_vars=list(drift_result.changed_vars),
                retrain_time=retrain_time,
                reason='mb_mode_full',
            )

        # 1. Variable-level MB expansion with mode control
        A_var = _collapse_to_var_graph(
            prev_A, L, min_block_edges=self.min_block_edges
        )
        seed_vars = list(drift_result.changed_vars)

        if self.mb_mode == 'seeds':
            S_var_set: set[int] = set(seed_vars)
        elif self.mb_mode == 'parents':
            S_var_set = set(seed_vars)
            for v in seed_vars:
                S_var_set |= set(
                    np.where(A_var[:, v] > 0)[0].tolist()
                )
        elif self.mb_mode == 'mb':
            S_var_set = set(seed_vars)
            for v in seed_vars:
                S_var_set |= _var_markov_blanket(A_var, v)
        else:
            raise ValueError(f"unknown mb_mode: {self.mb_mode}")

        S_vars = sorted(S_var_set)
        if len(S_vars) < self.min_local_vars:
            return UpdateResult(
                mode='full_retrain', fallback=True,
                A=A_full, B=B_full,
                changed_vars=list(drift_result.changed_vars),
                mb_set=S_vars, retrain_time=retrain_time,
                reason='empty_mb',
            )
        if len(S_vars) > self.max_local_fraction * n_vars:
            return UpdateResult(
                mode='full_retrain', fallback=True,
                A=A_full, B=B_full,
                changed_vars=list(drift_result.changed_vars),
                mb_set=S_vars, retrain_time=retrain_time,
                reason='mb_too_large',
            )

        # 2. Local order
        local_order = _remap_order(prev_order, S_vars, L)
        if local_order is None:
            return UpdateResult(
                mode='full_retrain', fallback=True,
                A=A_full, B=B_full,
                changed_vars=list(drift_result.changed_vars),
                mb_set=S_vars, retrain_time=retrain_time,
                reason='order_mismatch',
            )

        # 3. Local fit
        W_sub = window[:, S_vars]
        B_sub = _extract_submatrix_B(prev_B, S_vars, L)

        t0 = time.perf_counter()
        try:
            local_engine = DirectLiNGAMEngine(
                max_lag=L,
                use_mlx=engine.use_mlx,
                ridge_alpha=engine.ridge_alpha,
                bic_tol=engine.bic_tol,
                coeff_tol=engine.coeff_tol,
                max_backward_rounds=engine.max_backward_rounds,
                max_parents_per_node=engine.max_parents_per_node,
            )
            local_engine.fit(
                W_sub,
                warm_start_B=B_sub,
                warm_start_order=local_order,
            )
        except Exception as exc:
            return UpdateResult(
                mode='full_retrain', fallback=True,
                A=A_full, B=B_full,
                changed_vars=list(drift_result.changed_vars),
                mb_set=S_vars,
                update_time=time.perf_counter() - t0,
                retrain_time=retrain_time,
                reason=f'local_fit_failed:{type(exc).__name__}',
            )
        update_time = time.perf_counter() - t0

        # 4. Merge
        A_loc = local_engine.get_adjacency()
        B_loc = local_engine.get_coefficients()
        A_inc, B_inc = _merge_local_into_global(
            A_full, B_full, A_loc, B_loc, S_vars, L
        )

        # 5. Consistency checks
        ok, reason = self._consistency_check(
            A_inc=A_inc,
            B_inc=B_inc,
            A_full=A_full,
            B_full=B_full,
            engine=engine,
            window=window,
            shd_history=np.asarray(shd_history, dtype=np.float64),
        )
        if not ok:
            return UpdateResult(
                mode='full_retrain', fallback=True,
                A=A_full, B=B_full,
                changed_vars=list(drift_result.changed_vars),
                mb_set=S_vars,
                update_time=update_time,
                retrain_time=retrain_time,
                reason=f'consistency_failed:{reason}',
            )

        return UpdateResult(
            mode='incremental', fallback=False,
            A=A_inc, B=B_inc,
            changed_vars=list(drift_result.changed_vars),
            mb_set=S_vars,
            update_time=update_time,
            retrain_time=retrain_time,
            reason='ok',
        )

    # -----------------------------------------------------------------
    # Consistency checks
    # -----------------------------------------------------------------

    def _consistency_check(
        self,
        A_inc: np.ndarray,
        B_inc: np.ndarray,
        A_full: np.ndarray,
        B_full: np.ndarray,
        engine: DirectLiNGAMEngine,
        window: np.ndarray,
        shd_history: np.ndarray,
    ) -> tuple[bool, str]:
        # 1. Acyclicity
        if not is_dag(A_inc):
            return False, 'cycle'

        # 2. SHD deviation from full result
        shd_delta = structural_hamming(A_full, A_inc)
        if shd_history.size > 0:
            thr = max(
                self.shd_absolute_threshold,
                float(np.quantile(shd_history, self.shd_quantile)),
            )
        else:
            thr = self.shd_absolute_threshold
        if shd_delta > thr:
            return False, f'shd_delta={shd_delta}>{thr:.1f}'

        # 3. Residual variance must not blow up vs full
        try:
            R_full = _residuals_with_B(engine, window, B_full)
            R_inc = _residuals_with_B(engine, window, B_inc)
        except Exception:
            return False, 'residual_compute_failed'

        var_full = float(np.mean(R_full ** 2)) + 1e-12
        var_inc = float(np.mean(R_inc ** 2))
        if not np.isfinite(var_inc):
            return False, 'residual_nonfinite'
        if var_inc > self.residual_factor * var_full:
            return False, (
                f'residual_var={var_inc:.4f}'
                f'>{self.residual_factor}*{var_full:.4f}'
            )

        # 4. Coefficient explosion
        if check_coefficient_explosion(
            B_inc, B_full, factor=self.coeff_explosion_factor
        ):
            return False, 'coeff_explosion'

        return True, 'ok'