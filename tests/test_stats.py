"""Unit tests for stats.py."""

import numpy as np
import pytest

from stats import (
    energy_distance,
    energy_distance_per_variable,
    coefficient_change,
    coefficient_change_per_variable,
    structural_hamming,
    shd_per_variable,
    standardize,
    variable_drift_scores,
    permutation_test_energy,
)


def test_energy_distance_same_distribution():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    Y = rng.normal(size=(200, 3))
    ed = energy_distance(X, Y)
    assert ed < 0.2  # should be near zero


def test_energy_distance_shifted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    Y = rng.normal(loc=3.0, size=(200, 3))
    ed = energy_distance(X, Y)
    assert ed > 1.0


def test_energy_distance_per_variable_localizes_shift():
    rng = np.random.default_rng(1)
    R_prev = rng.normal(size=(150, 4))
    R_curr = rng.normal(size=(150, 4))
    R_curr[:, 2] += 2.0  # only column 2 drifted
    ed = energy_distance_per_variable(R_prev, R_curr)
    assert ed.shape == (4,)
    assert ed[2] == ed.max()
    assert ed[2] > 1.0


def test_energy_distance_handles_short_sample():
    X = np.zeros((1, 2))
    Y = np.zeros((5, 2))
    assert energy_distance(X, Y) == 0.0


def test_coefficient_change_scaling():
    B_prev = np.array([[0.0, 0.5], [0.0, 0.0]])
    B_curr = 2.0 * B_prev
    rel = coefficient_change(B_prev, B_curr)
    assert abs(rel - 1.0) < 1e-6


def test_coefficient_change_with_mask():
    B_prev = np.array([[0.0, 0.5], [0.0, 0.0]])
    B_curr = np.array([[0.0, 1.0], [0.0, 0.0]])
    mask = np.array([[False, True], [False, False]])
    rel = coefficient_change(B_prev, B_curr, mask=mask)
    assert abs(rel - 1.0) < 1e-6


def test_coefficient_change_per_variable():
    B_prev = np.array([[0.0, 0.5], [0.3, 0.0]])
    B_curr = np.array([[0.0, 1.0], [0.3, 0.0]])
    A_prev = (B_prev != 0).astype(float)
    A_curr = (B_curr != 0).astype(float)
    scores = coefficient_change_per_variable(B_prev, B_curr, A_prev, A_curr)
    assert scores[0] == 0.0
    assert scores[1] > 0.0


def test_structural_hamming_single_flip():
    A = np.zeros((3, 3))
    A[0, 1] = 1
    B = A.copy()
    B[0, 1] = 0
    assert structural_hamming(A, B) == 1


def test_shd_per_variable_single_flip():
    A = np.zeros((3, 3))
    A[0, 1] = 1
    B = A.copy()
    B[0, 1] = 0
    scores = shd_per_variable(A, B)
    assert scores[0] == 1.0
    assert scores[1] == 1.0
    assert scores[2] == 0.0


def test_standardize():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    z = standardize(x, mu=2.5, sigma=1.0)
    assert np.allclose(z, [-1.5, -0.5, 0.5, 1.5])


def test_variable_drift_scores_shape_and_sign():
    ed = np.array([0.1, 0.2, 0.3, 0.4])
    db = np.array([0.4, 0.3, 0.2, 0.1])
    shd = np.array([0.0, 0.0, 0.0, 1.0])
    s = variable_drift_scores(ed, db, shd)
    assert s.shape == (4,)
    assert s[3] > s[0]  # high ED + SHD dominates


def test_permutation_test_same_distribution():
    rng = np.random.default_rng(0)
    R_prev = rng.normal(size=(100, 3))
    R_curr = rng.normal(size=(100, 3))
    stat, p = permutation_test_energy(R_prev, R_curr, n_perm=100, random_state=0)
    assert p > 0.05


def test_permutation_test_shifted():
    rng = np.random.default_rng(0)
    R_prev = rng.normal(size=(100, 3))
    R_curr = rng.normal(loc=2.0, size=(100, 3))
    stat, p = permutation_test_energy(R_prev, R_curr, n_perm=100, random_state=0)
    assert p < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])