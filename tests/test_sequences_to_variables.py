"""
Tests for sequences-to-variables (Helske et al. 2024) implementation.
"""
import numpy as np
import pytest

from sequenzo.clustering.seqs2vars_utils import max_distance, cluster_labels_to_dummies
from sequenzo.clustering.sequences_to_variables import (
    representativeness_matrix,
    medoid_indices_from_kmedoids_result,
    cluster_labels_from_kmedoids_result,
    hard_classification_variables,
    soft_classification_variables,
)


# -----------------------------------------------------------------------------
# Phase 0: max_distance, cluster_labels_to_dummies
# -----------------------------------------------------------------------------

def test_max_distance_square():
    diss = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0],
    ], dtype=float)
    assert max_distance(diss) == 3.0


def test_max_distance_condensed():
    from scipy.spatial.distance import squareform
    diss_sq = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    cond = squareform(diss_sq)
    assert max_distance(cond) == 3.0


def test_cluster_labels_to_dummies():
    labels = np.array([0, 1, 2, 0, 1])
    d = cluster_labels_to_dummies(labels, k=3, reference=0)
    assert d.shape == (5, 2)
    # Reference is 0; columns for cluster 1 and 2
    np.testing.assert_array_equal(d[0], [0, 0])
    np.testing.assert_array_equal(d[1], [1, 0])
    np.testing.assert_array_equal(d[2], [0, 1])
    np.testing.assert_array_equal(d[3], [0, 0])
    np.testing.assert_array_equal(d[4], [1, 0])


# -----------------------------------------------------------------------------
# Phase 1: representativeness_matrix
# -----------------------------------------------------------------------------

def test_representativeness_matrix():
    # 4x4 distance matrix; d_max = 4 (e.g. 0-3)
    diss = np.array([
        [0, 1, 2, 4],
        [1, 0, 3, 3],
        [2, 3, 0, 2],
        [4, 3, 2, 0],
    ], dtype=float)
    medoids = [0, 3]  # first and last row as medoids
    d_max = 4.0
    R = representativeness_matrix(diss, medoids, d_max=d_max)
    assert R.shape == (4, 2)
    # R[i,k] = 1 - diss[i, med_k] / d_max
    np.testing.assert_allclose(R[:, 0], 1 - diss[:, 0] / 4)
    np.testing.assert_allclose(R[:, 1], 1 - diss[:, 3] / 4)
    # Medoid 0: row 0 has R[0,0]=1, others 1 - 1/4, 1-2/4, 1-4/4
    assert R[0, 0] == 1.0
    assert R[3, 1] == 1.0
    np.testing.assert_allclose(R[0, 1], 0.0)  # 1 - 4/4


def test_representativeness_d_max_none():
    diss = np.array([[0, 2], [2, 0]], dtype=float)
    R = representativeness_matrix(diss, [0], d_max=None)
    assert R.shape == (2, 1)
    assert max_distance(diss) == 2.0
    np.testing.assert_allclose(R[:, 0], [1.0, 0.0])


def test_representativeness_as_dataframe():
    diss = np.array([[0, 1], [1, 0]], dtype=float)
    df = representativeness_matrix(diss, [0, 1], d_max=1.0, ids=["a", "b"], as_dataframe=True)
    assert list(df.columns) == ["R_1", "R_2"]
    assert list(df.index) == ["a", "b"]
    np.testing.assert_allclose(df.values, [[1, 0], [0, 1]])


# -----------------------------------------------------------------------------
# Helpers: medoid indices and cluster labels from KMedoids result
# -----------------------------------------------------------------------------

def test_medoid_indices_from_kmedoids_result():
    # KMedoids returns medoid index per row (not cluster id)
    memb = np.array([2, 2, 5, 5, 5, 2])
    medoids = medoid_indices_from_kmedoids_result(memb)
    np.testing.assert_array_equal(medoids, [2, 5])


def test_cluster_labels_from_kmedoids_result():
    memb = np.array([2, 2, 5, 5, 2])
    labels = cluster_labels_from_kmedoids_result(memb)
    # Medoids sorted = [2, 5]; so 2->0, 5->1
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 0])


# -----------------------------------------------------------------------------
# Phase 2: hard_classification_variables
# -----------------------------------------------------------------------------

def test_hard_classification_variables():
    labels = np.array([0, 1, 2, 0, 1])
    X = hard_classification_variables(labels, k=3, reference=0)
    assert X.shape == (5, 2)
    np.testing.assert_array_equal(X, cluster_labels_to_dummies(labels, k=3, reference=0))


def test_hard_classification_variables_dataframe():
    labels = np.array([1, 2, 3, 1, 2])
    df = hard_classification_variables(labels, k=3, reference=0, ids=["i", "j", "k", "l", "m"], as_dataframe=True)
    assert df.shape == (5, 2)
    assert list(df.index) == ["i", "j", "k", "l", "m"]
