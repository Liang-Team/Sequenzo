"""Tests for align_cluster_orders."""

from __future__ import annotations

import numpy as np

from sequenzo.visualization import (
    align_cluster_orders,
    apply_cluster_order,
    match_clusters_to_reference,
)


def test_match_recovers_permutation():
    ref = np.array(
        [
            [1, 1, 1, 2],
            [2, 2, 2, 2],
            [3, 1, 1, 1],
        ]
    )
    # Shuffle reference rows
    perm = np.array([2, 0, 1])
    other = ref[perm]
    recovered = match_clusters_to_reference(ref, other)
    # display column j should show other[recovered[j]] == ref[j]
    assert np.array_equal(other[recovered], ref)


def test_align_to_reference_identical_medoids_share_columns():
    # Two partitions share medoids in different original orders
    a = np.array([[0, 0, 1], [1, 1, 1], [2, 0, 0], [1, 2, 2]])
    b = np.array([[1, 1, 1], [2, 0, 0], [0, 0, 1], [1, 2, 2]])  # a[[1,2,0,3]]
    orders = align_cluster_orders([a, b], reference=0, sort_reference=False)
    assert np.array_equal(a[orders[0]], a)
    assert np.array_equal(b[orders[1]], a)


def test_apply_cluster_order_renumbers():
    import pandas as pd

    df = pd.DataFrame(
        {
            "Cluster": [1, 2, 3],
            "Cluster_Size": [10, 20, 30],
            "Cluster_Weight_Sum": [1.0, 2.0, 3.0],
        }
    )
    out = apply_cluster_order(df, [2, 0, 1], renumber=True)
    assert list(out["Cluster"]) == [1, 2, 3]
    assert list(out["Cluster_Size"]) == [30, 10, 20]
