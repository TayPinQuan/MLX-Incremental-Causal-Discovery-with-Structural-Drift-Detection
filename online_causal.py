"""Online causal discovery orchestration.

Sliding-window pipeline:
    1. Full DirectLiNGAM fit per window (with warm start).
    2. Drift detection on residual, coefficient, and structure signals.
    3. On confirmed drift: incremental update via Markov-blanket local
       re-learning, with consistency-checked fallback to the full result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

from lingam_mlx import DirectLiNGAMEngine
from drift_detector import DriftDetector, compute_window_stats
from incremental import IncrementalUpdater, UpdateResult


@dataclass
class WindowRecord:
    index: int
    A: np.ndarray
    B: np.ndarray
    R: np.ndarray
    order: list[int]
    drift_detected: bool
    drift_confirmed: bool
    drift_score: float
    update: Optional[UpdateResult] = None


class OnlineCausalDiscovery:
    """Sliding-window online causal discovery with drift adaptation."""

    def __init__(
        self,
        window_size: int = 500,
        max_lag: int = 2,
        warmup: int = 3,
        kappa: float = 5.0,
        n_perm: int = 200,
        z_threshold: float = 2.5,
        use_mlx: bool = True,
        ridge_alpha: float = 0.1,
        coeff_tol: float = 0.1,
        max_parents_per_node: int = 4,
        residual_factor: float = 2.0,
        coeff_explosion_factor: float = 3.0,
        shd_absolute_threshold: float = 25.0,
        max_local_fraction: float = 0.75,
        min_block_edges: int = 3,
        random_state: int = 42,
    ):
        self.window_size = int(window_size)
        self.max_lag = int(max_lag)
        self.random_state = int(random_state)

        self.engine_ = DirectLiNGAMEngine(
            max_lag=self.max_lag,
            use_mlx=use_mlx,
            ridge_alpha=ridge_alpha,
            coeff_tol=coeff_tol,
            max_parents_per_node=max_parents_per_node,
        )
        self.detector_ = DriftDetector(
            warmup=warmup,
            kappa=kappa,
            n_perm=n_perm,
            z_threshold=z_threshold,
            random_state=random_state,
        )
        self.updater_ = IncrementalUpdater(
            residual_factor=residual_factor,
            coeff_explosion_factor=coeff_explosion_factor,
            shd_absolute_threshold=shd_absolute_threshold,
            max_local_fraction=max_local_fraction,
            min_block_edges=min_block_edges,
        )

    # -----------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------

    def run(self, data: np.ndarray) -> dict:
        data = np.asarray(data, dtype=np.float64)
        n_samples = data.shape[0]
        n_windows = n_samples // self.window_size

        records: list[WindowRecord] = []
        drift_points: list[int] = []
        update_records: list[UpdateResult] = []
        drift_scores: list[float] = []
        shd_history: list[float] = []
        retrain_times: list[float] = []

        prev_A: Optional[np.ndarray] = None
        prev_B: Optional[np.ndarray] = None
        prev_R: Optional[np.ndarray] = None
        prev_order: Optional[list[int]] = None

        for w in range(n_windows):
            W = data[w * self.window_size:(w + 1) * self.window_size]

            t0 = time.perf_counter()
            self.engine_.fit(
                W, warm_start_B=prev_B, warm_start_order=prev_order
            )
            retrain_time = time.perf_counter() - t0
            retrain_times.append(retrain_time)

            A_full = self.engine_.get_adjacency()
            B_full = self.engine_.get_coefficients()
            R_full = self.engine_.get_residuals(W)
            order = self.engine_.get_causal_order()

            stats = compute_window_stats(
                A_full, B_full, R_full, prev_A, prev_B, prev_R
            )
            res = self.detector_.update(stats)
            drift_scores.append(res.score)

            update_result: Optional[UpdateResult] = None
            A_use, B_use, R_use = A_full, B_full, R_full

            if res.confirmed:
                drift_points.append(w)
                upd = self.updater_.update(
                    engine=self.engine_,
                    window=W,
                    drift_result=res,
                    prev_B=prev_B,
                    prev_A=prev_A,
                    prev_order=prev_order,
                    prev_residuals=prev_R,
                    shd_history=np.array(shd_history, dtype=np.float64),
                    retrain_time=retrain_time,
                )
                update_records.append(upd)
                update_result = upd
                A_use, B_use = upd.A, upd.B
                R_use = self.engine_.get_residuals(W)

            records.append(WindowRecord(
                index=w,
                A=A_use,
                B=B_use,
                R=R_use,
                order=order,
                drift_detected=res.detected,
                drift_confirmed=res.confirmed,
                drift_score=res.score,
                update=update_result,
            ))

            shd_history.append(stats.shd)
            prev_A, prev_B, prev_R = A_use, B_use, R_use
            prev_order = order

        return {
            'records': records,
            'adj_matrices': [r.A for r in records],
            'coefficients': [r.B for r in records],
            'residuals': [r.R for r in records],
            'causal_orders': [r.order for r in records],
            'drift_points': drift_points,
            'drift_scores': np.array(drift_scores),
            'update_records': update_records,
            'retrain_times': np.array(retrain_times, dtype=np.float64),
            'n_windows': n_windows,
        }