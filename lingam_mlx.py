"""MLX-native DirectLiNGAM engine with delay embedding.

Layout convention (must match graph_ops.node_index):
    node_index(var, lag, max_lag) = var * (max_lag + 1) + lag

Order search:
    Nodes with lag > 0 are forced before nodes with lag == 0. Within the
    lag == 0 layer, a DirectLiNGAM style exogeneity search using
    third-order cross-cumulants ranks the nodes. If warm_start_order is
    valid, it is reused directly.

Coefficient estimation:
    Ridge regression with an optional target prior from the previous
    window (warm_start_B). Loss is
        ||y - X beta||^2 + lam * ||beta - target||^2
    which keeps coefficients stable across windows when the data does
    not push them away.

Edge selection:
    Backward elimination using BIC, then a hard cap on the number of
    parents per node (max_parents_per_node), keeping the largest
    |beta| coefficients.

Post-processing:
    Only edges pointing into lag == 0 nodes are kept. All other edges
    are zeroed out, because the target model is
        X_j(t) = f(X_i(t), X_i(t-1), ..., X_i(t-max_lag)).
"""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    mx = None
    _MLX_AVAILABLE = False


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------

class DirectLiNGAMEngine:
    """DirectLiNGAM with delay embedding, MLX-accelerated order search."""

    def __init__(
        self,
        max_lag: int = 2,
        use_mlx: bool = True,
        ridge_alpha: float = 0.1,
        bic_tol: float = 1e-6,
        coeff_tol: float = 0.1,
        max_backward_rounds: int = 20,
        max_parents_per_node: int = 4,
    ):
        self.max_lag = int(max_lag)
        self.use_mlx = bool(use_mlx and _MLX_AVAILABLE)
        self.ridge_alpha = float(ridge_alpha)
        self.bic_tol = float(bic_tol)
        self.coeff_tol = float(coeff_tol)
        self.max_backward_rounds = int(max_backward_rounds)
        self.max_parents_per_node = int(max_parents_per_node)

        self._A: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None
        self._order: Optional[list[int]] = None
        self._X_emb: Optional[np.ndarray] = None
        self._n_vars: Optional[int] = None
        self._n_samples: Optional[int] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        warm_start_B: Optional[np.ndarray] = None,
        warm_start_order: Optional[list[int]] = None,
    ) -> "DirectLiNGAMEngine":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_vars)")
        self._n_samples, self._n_vars = X.shape

        X_emb = self._delay_embed(X)
        self._X_emb = X_emb

        if warm_start_order is not None and self._valid_order(
            warm_start_order
        ):
            order = list(warm_start_order)
        else:
            try:
                order = self._search_order(X_emb)
            except Exception:
                if not self.use_mlx:
                    raise
                self.use_mlx = False
                order = self._search_order(X_emb)
        self._order = order

        B = self._estimate_coefficients(X_emb, order, warm_start_B)
        A, B = self._select_edges_bic(X_emb, B, order, warm_start_B)
        A, B = self._postprocess_temporal(A, B)

        self._A = A
        self._B = B
        return self

    def get_adjacency(self) -> np.ndarray:
        self._check_fitted()
        return self._A.copy()

    def get_coefficients(self) -> np.ndarray:
        self._check_fitted()
        return self._B.copy()

    def get_residuals(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        X_emb = self._delay_embed(X)
        T = X_emb.shape[0]
        L = self.max_lag
        p = self._n_vars

        pred = X_emb @ self._B
        residuals = np.zeros((T, p), dtype=np.float64)
        for var in range(p):
            node = var * (L + 1) + 0
            residuals[:, var] = X_emb[:, node] - pred[:, node]
        return residuals

    def get_causal_order(self) -> list[int]:
        self._check_fitted()
        return list(self._order)

    def get_node_count(self) -> int:
        return self._n_vars * (self.max_lag + 1)

    # -----------------------------------------------------------------
    # Delay embedding
    # -----------------------------------------------------------------

    def _delay_embed(self, X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        L = self.max_lag
        T = n - L
        if T <= 0:
            raise ValueError("n_samples must exceed max_lag")
        X_emb = np.zeros((T, p * (L + 1)), dtype=np.float64)
        for lag in range(L + 1):
            cols = [var * (L + 1) + lag for var in range(p)]
            X_emb[:, cols] = X[L - lag : n - lag, :]
        return X_emb

    # -----------------------------------------------------------------
    # Order validation and search
    # -----------------------------------------------------------------

    def _valid_order(self, order: list[int]) -> bool:
        if order is None:
            return False
        d = self._n_vars * (self.max_lag + 1)
        if len(order) != d:
            return False
        if set(order) != set(range(d)):
            return False
        L = self.max_lag
        seen_lag0 = False
        for node in order:
            lag = node % (L + 1)
            if lag == 0:
                seen_lag0 = True
            elif seen_lag0:
                return False
        return True

    def _search_order(self, X_emb: np.ndarray) -> list[int]:
        L = self.max_lag
        p = self._n_vars

        order: list[int] = []
        for lag in range(L, 0, -1):
            for var in range(p):
                order.append(var * (L + 1) + lag)

        remaining = [var * (L + 1) + 0 for var in range(p)]

        if self.use_mlx:
            X_mx = mx.array(X_emb.astype(np.float32))
            mx.eval(X_mx)
        else:
            X_mx = None

        while remaining:
            best_node = None
            best_measure = np.inf
            for i in remaining:
                total = 0.0
                for j in remaining:
                    if j == i:
                        continue
                    if self.use_mlx:
                        m = self._pairwise_independence_mlx(X_mx, i, j)
                    else:
                        m = self._pairwise_independence_numpy(X_emb, i, j)
                    if not np.isfinite(m):
                        m = 0.0
                    total += m
                if total < best_measure:
                    best_measure = total
                    best_node = i
            if best_node is None:
                best_node = remaining[0]
            remaining.remove(best_node)
            order.append(best_node)

        return order

    # -----------------------------------------------------------------
    # Independence measures
    # -----------------------------------------------------------------

    def _pairwise_independence_mlx(self, X_mx, i: int, j: int) -> float:
        xi = X_mx[:, i]
        xj = X_mx[:, j]

        xi_c = xi - mx.mean(xi)
        xj_c = xj - mx.mean(xj)

        num = mx.sum(xi_c * xj_c)
        den = mx.sum(xj_c * xj_c)
        beta = num / (den + 1e-12)

        residual = xi_c - beta * xj_c

        r_std = mx.sqrt(mx.mean(residual * residual)) + 1e-8
        xj_std = mx.sqrt(mx.mean(xj_c * xj_c)) + 1e-8

        m = mx.mean(
            (residual / r_std) * (xj_c / xj_std) * (xj_c / xj_std)
        )
        m = mx.abs(m)
        mx.eval(m)

        val = float(m)
        if not np.isfinite(val):
            return 0.0
        return val

    def _pairwise_independence_numpy(
        self, X: np.ndarray, i: int, j: int
    ) -> float:
        xi = X[:, i]
        xj = X[:, j]
        xi_c = xi - xi.mean()
        xj_c = xj - xj.mean()
        beta = float((xi_c * xj_c).sum() / ((xj_c * xj_c).sum() + 1e-12))
        residual = xi_c - beta * xj_c
        r_std = float(np.sqrt((residual * residual).mean())) + 1e-8
        xj_std = float(np.sqrt((xj_c * xj_c).mean())) + 1e-8
        val = float(abs(np.mean((residual / r_std) * (xj_c / xj_std) ** 2)))
        if not np.isfinite(val):
            return 0.0
        return val

    # -----------------------------------------------------------------
    # Coefficient estimation (target-prior Ridge)
    # -----------------------------------------------------------------

    def _estimate_coefficients(
        self,
        X_emb: np.ndarray,
        order: list[int],
        warm_start_B: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        T, d = X_emb.shape
        B = np.zeros((d, d), dtype=np.float64)

        if warm_start_B is None:
            B_prev = np.zeros((d, d), dtype=np.float64)
        else:
            if warm_start_B.shape != (d, d):
                raise ValueError(
                    f"warm_start_B shape {warm_start_B.shape} "
                    f"does not match ({d}, {d})"
                )
            B_prev = warm_start_B

        for pos, j in enumerate(order):
            if pos == 0:
                continue
            preds = order[:pos]
            X_pred = X_emb[:, preds]
            y = X_emb[:, j]

            XtX = X_pred.T @ X_pred
            scale = float(np.trace(XtX)) / XtX.shape[0]
            lam = self.ridge_alpha * scale

            target = B_prev[preds, j]

            A_reg = XtX + lam * np.eye(len(preds))
            b_reg = X_pred.T @ y + lam * target
            beta = np.linalg.solve(A_reg, b_reg)
            for k, pred in enumerate(preds):
                B[pred, j] = beta[k]
        return B

    # -----------------------------------------------------------------
    # BIC edge selection
    # -----------------------------------------------------------------

    def _select_edges_bic(
        self,
        X_emb: np.ndarray,
        B: np.ndarray,
        order: list[int],
        warm_start_B: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        T, d = X_emb.shape
        B_new = np.zeros_like(B)
        A_new = np.zeros((d, d), dtype=np.float64)

        if warm_start_B is None:
            B_prev = np.zeros((d, d), dtype=np.float64)
        else:
            B_prev = warm_start_B

        for pos, j in enumerate(order):
            if pos == 0:
                continue
            preds = order[:pos]
            y = X_emb[:, j]
            target_full = B_prev[preds, j]

            active = list(range(len(preds)))
            best_bic = self._bic_for_subset(
                X_emb, y, preds, active, T, target_full
            )
            for _ in range(self.max_backward_rounds):
                improved = False
                drop_idx = None
                for k in range(len(active)):
                    if len(active) <= 1:
                        break
                    trial = active[:k] + active[k + 1 :]
                    bic_trial = self._bic_for_subset(
                        X_emb, y, preds, trial, T, target_full
                    )
                    if bic_trial < best_bic - self.bic_tol:
                        best_bic = bic_trial
                        drop_idx = k
                        improved = True
                if not improved or drop_idx is None:
                    break
                active.pop(drop_idx)

            if not active:
                continue

            idx_active = [preds[a] for a in active]
            X_act = X_emb[:, idx_active]
            target_act = B_prev[idx_active, j]

            XtX = X_act.T @ X_act
            scale = float(np.trace(XtX)) / XtX.shape[0]
            lam = self.ridge_alpha * scale
            A_reg = XtX + lam * np.eye(XtX.shape[0])
            b_reg = X_act.T @ y + lam * target_act
            beta_act = np.linalg.solve(A_reg, b_reg)

            candidates = [
                (idx_active[k], float(beta_act[k]))
                for k in range(len(idx_active))
                if abs(beta_act[k]) > self.coeff_tol
            ]
            candidates.sort(key=lambda x: abs(x[1]), reverse=True)
            candidates = candidates[: self.max_parents_per_node]

            for node, val in candidates:
                B_new[node, j] = val
                A_new[node, j] = 1.0

        return A_new, B_new

    def _bic_for_subset(
        self,
        X_emb: np.ndarray,
        y: np.ndarray,
        preds: list[int],
        active: list[int],
        T: int,
        target_full: Optional[np.ndarray] = None,
    ) -> float:
        if not active:
            rss = float(np.sum((y - y.mean()) ** 2))
            k = 0
        else:
            idx_active = [preds[a] for a in active]
            X_sub = X_emb[:, idx_active]
            if target_full is None:
                target = np.zeros(len(active))
            else:
                target = target_full[active]
            XtX = X_sub.T @ X_sub
            scale = float(np.trace(XtX)) / XtX.shape[0]
            lam = self.ridge_alpha * scale
            A_reg = XtX + lam * np.eye(XtX.shape[0])
            b_reg = X_sub.T @ y + lam * target
            beta = np.linalg.solve(A_reg, b_reg)
            rss = float(np.sum((y - X_sub @ beta) ** 2))
            k = len(active)
        rss = max(rss, 1e-12)
        return T * np.log(rss / T) + k * np.log(T)

    # -----------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------

    def _postprocess_temporal(
        self, A: np.ndarray, B: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        L = self.max_lag
        d = A.shape[0]
        A_new = A.copy()
        B_new = B.copy()
        for v in range(d):
            lag_v = v % (L + 1)
            if lag_v != 0:
                A_new[:, v] = 0.0
                B_new[:, v] = 0.0
        np.fill_diagonal(A_new, 0.0)
        np.fill_diagonal(B_new, 0.0)
        return A_new, B_new

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._A is None or self._B is None:
            raise RuntimeError("engine is not fitted yet")