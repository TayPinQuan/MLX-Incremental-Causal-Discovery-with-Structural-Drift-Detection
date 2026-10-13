"""Multi-signal drift detection with double-sided CUSUM.

Signals:
    1. Residual energy distance (ED)
    2. Relative coefficient change (ΔB)
    3. Normalized structural Hamming distance (SHD)

Fusion:
    Standardize each signal using warmup statistics (robust MAD),
    then take the signal-wise max:
        d_t = max(z_ED, z_ΔB, z_SHD)

Detection:
    Double-sided CUSUM with adaptive threshold
        C_pos = max(0, C_pos + d_t - k)
        C_neg = max(0, C_neg - d_t - k)
        h = mu_d + kappa * sigma_d
    A trigger is confirmed by a permutation test on residuals.

Changed variable selection:
    For each variable j, compute the robust z-score of its drift
    score v_j with respect to the warmup distribution of v_j.
    Variables with z > z_threshold are selected. If none crosses,
    the argmax variable is returned so that a confirmed drift always
    yields at least one seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from stats import (
    energy_distance,
    energy_distance_per_variable,
    coefficient_change,
    coefficient_change_per_variable,
    structural_hamming,
    shd_per_variable,
    variable_drift_scores,
    permutation_test_energy,
)


# ---------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------

@dataclass
class WindowStats:
    """Statistics for one window relative to the previous window."""
    ed: float
    delta_b: float
    shd: float
    ed_per_var: np.ndarray
    db_per_var: np.ndarray
    shd_per_var: np.ndarray
    residuals: np.ndarray
    adjacency: np.ndarray
    coefficients: np.ndarray
    prev_residuals: Optional[np.ndarray] = None
    prev_adjacency: Optional[np.ndarray] = None
    prev_coefficients: Optional[np.ndarray] = None


@dataclass
class DriftResult:
    """Outcome of drift detection for one window."""
    detected: bool = False
    confirmed: bool = False
    in_warmup: bool = False
    score: float = 0.0
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    p_value: float = 1.0
    changed_vars: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------

def _aggregate_to_variables(
    per_node_scores: np.ndarray,
    n_vars: int,
    n_lags_plus_1: int,
) -> np.ndarray:
    """Sum per-node scores over lag copies to per-variable scores.

    Node layout (matches graph_ops.node_index):
        node = var * (max_lag + 1) + lag
    """
    out = np.zeros(n_vars, dtype=np.float64)
    for var in range(n_vars):
        idx = [var * n_lags_plus_1 + lag for lag in range(n_lags_plus_1)]
        out[var] = float(per_node_scores[idx].sum())
    return out


def compute_window_stats(
    adjacency: np.ndarray,
    coefficients: np.ndarray,
    residuals: np.ndarray,
    prev_adjacency: Optional[np.ndarray] = None,
    prev_coefficients: Optional[np.ndarray] = None,
    prev_residuals: Optional[np.ndarray] = None,
) -> WindowStats:
    """Compute the three signals and per-variable scores.

    Per-variable arrays are all shape (n_vars,).
    Per-node arrays from the delay-embedded graph are aggregated
    over lag copies before returning.

    If any `prev_*` is None, all signals are set to 0.
    """
    n_vars = residuals.shape[1]
    d = adjacency.shape[0]
    if d % n_vars != 0:
        raise ValueError(
            f"adjacency dim {d} not divisible by n_vars {n_vars}"
        )
    n_lags_plus_1 = d // n_vars

    if (
        prev_adjacency is None
        or prev_coefficients is None
        or prev_residuals is None
    ):
        zeros = np.zeros(n_vars, dtype=np.float64)
        return WindowStats(
            ed=0.0, delta_b=0.0, shd=0.0,
            ed_per_var=zeros.copy(),
            db_per_var=zeros.copy(),
            shd_per_var=zeros.copy(),
            residuals=residuals,
            adjacency=adjacency,
            coefficients=coefficients,
            prev_residuals=None,
            prev_adjacency=None,
            prev_coefficients=None,
        )

    # Signal 1: residual energy distance, per variable (already (p,))
    ed = energy_distance(prev_residuals, residuals)
    ed_per_var = energy_distance_per_variable(prev_residuals, residuals)

    # Signal 2: coefficient change, per node -> aggregate to per variable
    union_mask = (prev_adjacency > 0) | (adjacency > 0)
    delta_b = coefficient_change(
        prev_coefficients, coefficients, mask=union_mask
    )
    db_per_node = coefficient_change_per_variable(
        prev_coefficients, coefficients, prev_adjacency, adjacency
    )
    db_per_var = _aggregate_to_variables(db_per_node, n_vars, n_lags_plus_1)

    # Signal 3: SHD, per node -> aggregate to per variable
    shd = float(structural_hamming(prev_adjacency, adjacency))
    shd_per_node = shd_per_variable(prev_adjacency, adjacency)
    shd_per_var = _aggregate_to_variables(shd_per_node, n_vars, n_lags_plus_1)

    return WindowStats(
        ed=ed,
        delta_b=delta_b,
        shd=shd,
        ed_per_var=ed_per_var,
        db_per_var=db_per_var,
        shd_per_var=shd_per_var,
        residuals=residuals,
        adjacency=adjacency,
        coefficients=coefficients,
        prev_residuals=prev_residuals,
        prev_adjacency=prev_adjacency,
        prev_coefficients=prev_coefficients,
    )


# ---------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------

class DriftDetector:
    """Multi-signal drift detector with CUSUM and permutation test.

    Parameters
    ----------
    warmup : int
        Number of windows used to calibrate statistics. No detection
        happens during warmup.
    kappa : float
        Threshold multiplier: h = mu_d + kappa * sigma_d.
    n_perm : int
        Number of permutations for the confirmation test.
    weights : tuple
        Reserved for weighted fusion. Currently unused because the
        default fusion is signal-wise max. Kept for ablation.
    cusum_k : float or None
        CUSUM slack. If None, defaults to 0.5 * sigma_d from warmup.
    z_threshold : float
        Per-variable robust z-score threshold for changed-variable
        selection.
    perm_alpha : float
        Significance level for permutation-test confirmation.
    random_state : int
        Seed for permutation test RNG.
    use_ema : bool
        If True, apply EMA smoothing before CUSUM. Default False.
    ema_alpha : float
        EMA coefficient when use_ema is True.
    """

    def __init__(
        self,
        warmup: int = 10,
        kappa: float = 5.0,
        n_perm: int = 200,
        weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        cusum_k: Optional[float] = None,
        z_threshold: float = 2.5,
        perm_alpha: float = 0.05,
        random_state: int = 42,
        use_ema: bool = False,
        ema_alpha: float = 0.3,
        signal_mode: str = 'all',
    ):
        self.warmup = int(warmup)
        self.kappa = float(kappa)
        self.n_perm = int(n_perm)
        self.w1, self.w2, self.w3 = weights
        self.cusum_k = cusum_k
        self.z_threshold = float(z_threshold)
        self.perm_alpha = float(perm_alpha)
        self.rng = np.random.default_rng(random_state)
        self.use_ema = bool(use_ema)
        self.ema_alpha = float(ema_alpha)

        # warmup buffers
        self._warm_ed: list[float] = []
        self._warm_db: list[float] = []
        self._warm_shd: list[float] = []
        self._warm_v: list[np.ndarray] = []

        # post-warmup state
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._mu_v: Optional[np.ndarray] = None
        self._sigma_v: Optional[np.ndarray] = None

        self._mu_d: float = 0.0
        self._sigma_d: float = 1.0
        self._h: float = np.inf
        self._k_cusum: float = 0.5

        self._c_pos: float = 0.0
        self._c_neg: float = 0.0
        self._t: int = 0
        self._finalized: bool = False

        self._ema_d: Optional[float] = None

        valid_modes = {'ed_only', 'db_only', 'shd_only', 'ed_db', 'all'}
        if signal_mode not in valid_modes:
            raise ValueError(
                f"signal_mode must be one of {valid_modes}, "
                f"got {signal_mode}"
            )
        self.signal_mode = signal_mode


    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def is_warmup(self) -> bool:
        return not self._finalized

    def update(self, stats: WindowStats) -> DriftResult:
        # Warmup phase: accumulate and return neutral result
        if not self._finalized:
            self._warm_ed.append(stats.ed)
            self._warm_db.append(stats.delta_b)
            self._warm_shd.append(stats.shd)
            v = variable_drift_scores(
                stats.ed_per_var, stats.db_per_var, stats.shd_per_var
            )
            self._warm_v.append(v)
            self._t += 1
            if self._t >= self.warmup:
                self._finalize_warmup()
            return DriftResult(in_warmup=True)

        # Post-warmup phase
        z_ed = (stats.ed - self._mu[0]) / self._sigma[0]
        z_db = (stats.delta_b - self._mu[1]) / self._sigma[1]
        z_shd = (stats.shd - self._mu[2]) / self._sigma[2]

        d_t = float(self._fuse(z_ed, z_db, z_shd))

        if self.use_ema:
            if self._ema_d is None:
                self._ema_d = d_t
            else:
                self._ema_d = (
                    self.ema_alpha * d_t
                    + (1 - self.ema_alpha) * self._ema_d
                )
            d_t = self._ema_d

        self._c_pos = max(0.0, self._c_pos + d_t - self._k_cusum)
        self._c_neg = max(0.0, self._c_neg - d_t - self._k_cusum)

        detected = (
            self._c_pos > self._h
            or self._c_neg > self._h
            or d_t > self._h
        )
        confirmed = detected
        p_value = 1.0

        if detected and stats.prev_residuals is not None:
            # Permutation test is a diagnostic, not a gate. Residual
            # energy distance does not respond to pure structure drift.
            _, p_value = permutation_test_energy(
                stats.prev_residuals,
                stats.residuals,
                n_perm=self.n_perm,
                random_state=int(self.rng.integers(1 << 31)),
            )
            self._c_pos = 0.0
            self._c_neg = 0.0

        changed_vars: list[int] = []
        if confirmed and self._mu_v is not None and self._sigma_v is not None:
            v = variable_drift_scores(
                stats.ed_per_var, stats.db_per_var, stats.shd_per_var
            )
            z_v = (v - self._mu_v) / self._sigma_v
            changed_vars = [
                int(j) for j in np.where(z_v > self.z_threshold)[0]
            ]
            # Fallback: confirmed drift must yield at least one seed
            if not changed_vars:
                changed_vars = [int(np.argmax(z_v))]

        return DriftResult(
            detected=detected,
            confirmed=confirmed,
            in_warmup=False,
            score=float(d_t),
            cusum_pos=float(self._c_pos),
            cusum_neg=float(self._c_neg),
            p_value=float(p_value),
            changed_vars=changed_vars,
        )

    def _fuse(self, z_ed, z_db, z_shd):
        """Combine standardized signals into a single drift score.

        z_ed, z_db, z_shd: scalars or 1-D arrays (warmup uses arrays).
        """
        mode = self.signal_mode
        if mode == 'ed_only':
            return z_ed
        if mode == 'db_only':
            return z_db
        if mode == 'shd_only':
            return z_shd
        if mode == 'ed_db':
            return np.maximum(z_ed, z_db)
        if mode == 'db_shd':
            return np.maximum(z_db, z_shd)
        # 'all'
        return np.maximum(np.maximum(z_ed, z_db), z_shd)

    # -----------------------------------------------------------------
    # Warmup finalization
    # -----------------------------------------------------------------

    def _finalize_warmup(self) -> None:
        arr_ed = np.array(self._warm_ed, dtype=np.float64)
        arr_db = np.array(self._warm_db, dtype=np.float64)
        arr_shd = np.array(self._warm_shd, dtype=np.float64)

        def _mu_sigma(a: np.ndarray) -> tuple[float, float]:
            """Robust location and scale via median and MAD."""
            if a.size == 0:
                return 0.0, 1.0
            med = float(np.median(a))
            mad = float(np.median(np.abs(a - med)))
            s = 1.4826 * mad
            if not np.isfinite(s) or s < 1e-8:
                s = 1.0
            return med, s

        mu_ed, sg_ed = _mu_sigma(arr_ed)
        mu_db, sg_db = _mu_sigma(arr_db)
        mu_shd, sg_shd = _mu_sigma(arr_shd)
        self._mu = np.array([mu_ed, mu_db, mu_shd])
        self._sigma = np.array([sg_ed, sg_db, sg_shd])

        # Compute d_t for warmup windows using the SAME fusion as runtime
        z_ed = (arr_ed - mu_ed) / sg_ed
        z_db = (arr_db - mu_db) / sg_db
        z_shd = (arr_shd - mu_shd) / sg_shd
        d_warm = np.asarray(self._fuse(z_ed, z_db, z_shd), dtype=np.float64)

        if d_warm.size:
            med_d = float(np.median(d_warm))
            mad_d = float(np.median(np.abs(d_warm - med_d)))
            sigma_robust = 1.4826 * mad_d
        else:
            med_d = 0.0
            sigma_robust = 1.0
        if not np.isfinite(sigma_robust) or sigma_robust < 1e-6:
            sigma_robust = 1.0
        self._mu_d = med_d
        self._sigma_d = sigma_robust
        self._h = self._mu_d + self.kappa * self._sigma_d

        if self.cusum_k is None:
            self._k_cusum = 0.5 * self._sigma_d
        else:
            self._k_cusum = float(self.cusum_k)

        # Per-variable robust location and scale from warmup v_j
        if self._warm_v:
            v_stack = np.stack(self._warm_v, axis=0)  # (warmup, n_vars)
            mu_v = np.median(v_stack, axis=0)
            mad_v = np.median(np.abs(v_stack - mu_v), axis=0)
            sigma_v = 1.4826 * mad_v
            sigma_v = np.where(
                (~np.isfinite(sigma_v)) | (sigma_v < 1e-6),
                1.0,
                sigma_v,
            )
            self._mu_v = mu_v
            self._sigma_v = sigma_v
        else:
            self._mu_v = None
            self._sigma_v = None

        self._finalized = True