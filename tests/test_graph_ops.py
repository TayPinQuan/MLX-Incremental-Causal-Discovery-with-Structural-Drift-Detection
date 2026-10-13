"""Unit tests for graph_ops.py."""

import numpy as np
import pytest

from graph_ops import (
    parents, children, markov_blanket, expand_mb_set,
    is_dag, align_coefficients, submatrix, submatrix_B,
    merge_subgraph,
    check_no_cycle, check_shd_jump, check_coefficient_explosion,
    node_index, node_label, all_node_labels,
)


def _chain():
    # 0 -> 1 -> 2
    A = np.zeros((3, 3))
    A[0, 1] = 1
    A[1, 2] = 1
    return A


def _collider():
    # 0 -> 2 <- 1
    A = np.zeros((3, 3))
    A[0, 2] = 1
    A[1, 2] = 1
    return A


def _fork():
    # 1 -> 0, 1 -> 2
    A = np.zeros((3, 3))
    A[1, 0] = 1
    A[1, 2] = 1
    return A


def test_parents_children():
    A = _chain()
    assert parents(A, 1) == {0}
    assert children(A, 0) == {1}


def test_mb_chain():
    A = _chain()
    assert markov_blanket(A, 1) == {0, 2}
    assert markov_blanket(A, 0) == {1}


def test_mb_collider():
    A = _collider()
    assert markov_blanket(A, 2) == {0, 1}
    # In a collider, 0 has child 2, 2 has parent 1, so MB(0) = Ch(0) ∪ Pa(Ch(0)) = {2} ∪ {1} = {1, 2}
    assert markov_blanket(A, 0) == {1, 2}


def test_mb_fork():
    A = _fork()
    # MB(0) = Pa(0) ∪ Ch(0) ∪ Pa(Ch(0)) = {1} ∪ {} ∪ {} = {1}
    assert markov_blanket(A, 0) == {1}


def test_expand_mb_set():
    A = _chain()
    S = expand_mb_set(A, seeds=[1])
    assert S == {0, 1, 2}


def test_is_dag_true():
    A = _chain()
    assert is_dag(A)


def test_is_dag_false():
    A = np.zeros((3, 3))
    A[0, 1] = 1
    A[1, 2] = 1
    A[2, 0] = 1
    assert not is_dag(A)


def test_align_coefficients():
    A_prev = np.array([[0, 1], [0, 0]], dtype=float)
    A_curr = np.array([[0, 1], [1, 0]], dtype=float)
    B_prev = np.array([[0.0, 0.5], [0.0, 0.0]])
    B_curr = np.array([[0.0, 1.0], [0.7, 0.0]])
    B_aligned = align_coefficients(B_prev, B_curr, A_prev, A_curr)
    assert B_aligned[0, 1] == 1.0
    assert B_aligned[1, 0] == 0.7


def test_submatrix():
    A = _chain()
    S = [1, 2]
    sub = submatrix(A, S)
    assert sub.shape == (2, 2)
    assert sub[0, 1] == 1


def test_merge_subgraph():
    A_global = np.zeros((3, 3))
    A_global[0, 1] = 1
    B_global = np.zeros((3, 3))
    B_global[0, 1] = 0.5

    A_local = np.zeros((2, 2))
    A_local[0, 1] = 1
    B_local = np.zeros((2, 2))
    B_local[0, 1] = 0.9

    A_new, B_new = merge_subgraph(
        A_global, B_global, A_local, B_local, S=[1, 2]
    )
    # A_global[0,1] (boundary edge from outside to S) preserved
    assert A_new[0, 1] == 1
    assert B_new[0, 1] == 0.5
    # A_new[1,2] replaced by local
    assert A_new[1, 2] == 1
    assert B_new[1, 2] == 0.9


def test_check_no_cycle():
    assert check_no_cycle(_chain())
    assert not check_no_cycle(np.array([[0, 1], [1, 0]]))


def test_check_shd_jump():
    history = np.array([1.0, 2.0, 1.5, 2.5, 1.0])
    assert check_shd_jump(10.0, history, quantile=0.99)
    assert not check_shd_jump(0.5, history, quantile=0.99)


def test_check_coefficient_explosion():
    B_prev = np.array([[0.0, 1.0], [0.0, 0.0]])
    B_big = 4.0 * B_prev
    B_ok = 1.5 * B_prev
    assert check_coefficient_explosion(B_big, B_prev, factor=3.0)
    assert not check_coefficient_explosion(B_ok, B_prev, factor=3.0)


def test_node_index_layout():
    assert node_index(0, 0, 2) == 0
    assert node_index(0, 1, 2) == 1
    assert node_index(0, 2, 2) == 2
    assert node_index(1, 0, 2) == 3
    assert node_index(7, 2, 2) == 23


def test_node_label():
    assert node_label(0, 0) == "X0(t)"
    assert node_label(3, 2) == "X3(t-2)"


def test_all_node_labels_length_and_order():
    labels = all_node_labels(n_vars=2, max_lag=2)
    assert len(labels) == 2 * 3
    assert labels[0] == "X0(t)"
    assert labels[1] == "X1(t)"
    assert labels[2] == "X0(t-1)"
    assert labels[-1] == "X1(t-2)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])