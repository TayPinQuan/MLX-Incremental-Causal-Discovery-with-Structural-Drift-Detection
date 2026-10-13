"""
tests/test_lingam.py
--------------------
Unit tests for lingam_mlx.py (CausalEngine).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from lingam_mlx import CausalEngine, build_delay_embedded
from graph_ops import total_nodes


class TestDelayEmbedding:
    def test_shape(self):
        T, n_vars, max_lag = 100, 4, 2
        data = np.random.randn(T, n_vars).astype(np.float32)
        X, meta = build_delay_embedded(data, max_lag)
        assert X.shape == (T - max_lag, total_nodes(n_vars, max_lag))
        assert meta["n_nodes"] == total_nodes(n_vars, max_lag)

    def test_lag_values(self):
        """Verify that the lagged column indeed has shifted values."""
        T, n_vars, max_lag = 50, 2, 1
        data = np.arange(T * n_vars, dtype=np.float32).reshape(T, n_vars)
        X, _ = build_delay_embedded(data, max_lag)
        # node_index(0, 0, max_lag=1) = 0  → current
        # node_index(0, 1, max_lag=1) = 1  → lag-1
        n_valid = T - max_lag
        np.testing.assert_array_equal(X[:, 0], data[max_lag:, 0])
        np.testing.assert_array_equal(X[:, 1], data[:n_valid, 0])


class TestCausalEngine:
    def _make_data(self, n_samples=300, n_vars=3, seed=0):
        rng = np.random.default_rng(seed)
        X = np.zeros((n_samples, n_vars), dtype=np.float32)
        for t in range(1, n_samples):
            X[t, 0] = rng.normal(0, 0.5)
            X[t, 1] = 0.6 * X[t, 0] + rng.normal(0, 0.5)
            X[t, 2] = 0.4 * X[t, 1] + rng.normal(0, 0.5)
        return X

    def test_fit_returns_self(self):
        data = self._make_data()
        engine = CausalEngine(max_lag=1)
        result = engine.fit(data)
        assert result is engine

    def test_adjacency_shape(self):
        data = self._make_data()
        engine = CausalEngine(max_lag=1)
        engine.fit(data)
        n_nodes = total_nodes(3, 1)
        adj = engine.get_adjacency()
        assert adj.shape == (n_nodes, n_nodes)

    def test_coefficients_shape(self):
        data = self._make_data()
        engine = CausalEngine(max_lag=1)
        engine.fit(data)
        coef = engine.get_coefficients()
        n_nodes = total_nodes(3, 1)
        assert coef.shape == (n_nodes, n_nodes)

    def test_residuals_shape(self):
        data = self._make_data()
        engine = CausalEngine(max_lag=1)
        engine.fit(data)
        res = engine.get_residuals()
        n_nodes = total_nodes(3, 1)
        assert res.shape[1] == n_nodes

    def test_causal_order_length(self):
        data = self._make_data()
        engine = CausalEngine(max_lag=1)
        engine.fit(data)
        order = engine.get_causal_order()
        n_nodes = total_nodes(3, 1)
        assert len(order) == n_nodes
        assert set(order) == set(range(n_nodes))

    def test_adjacency_is_dag(self):
        data = self._make_data()
        engine = CausalEngine(max_lag=1)
        engine.fit(data)
        from graph_ops import is_dag
        assert is_dag(engine.get_adjacency())

    def test_fit_subset(self):
        data = self._make_data(n_vars=4)
        engine = CausalEngine(max_lag=1)
        engine.fit(data)
        subset = [0, 1, 2]
        local_adj, local_coef, local_order = engine.fit_subset(data, subset)
        assert local_adj.shape == (3, 3)
        assert local_coef.shape == (3, 3)
        assert len(local_order) == 3
