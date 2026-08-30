import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

from sequenzo.clustering.hierarchical_clustering import (
    Cluster,
    ClusterQuality,
    ward_labels_only,
)


_DISS = np.array(
    [
        [0.0, 1.0, 1.0, 4.24264068711928, 5.65685424949238],
        [1.0, 0.0, 1.4142135623731, 3.60555127546399, 5.0],
        [1.0, 1.4142135623731, 0.0, 3.60555127546399, 5.0],
        [4.24264068711928, 3.60555127546399, 3.60555127546399, 0.0, 1.4142135623731],
        [5.65685424949238, 5.0, 5.0, 1.4142135623731, 0.0],
    ],
    dtype=np.float64,
)

_FEATURES = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ],
    dtype=np.float64,
)

_R_FASTCLUSTER_WARD_D_HEIGHTS = np.array(
    [1.0, 1.2761423749153999, 1.4142135623731, 9.0852539076258356],
    dtype=np.float64,
)


@pytest.mark.parametrize("distance", [_DISS, squareform(_DISS, checks=False)])
def test_ward_d_heights_match_r_fastcluster(distance):
    result = Cluster(distance, clustering_method="ward_d", fast_path=True)

    assert np.allclose(
        result.linkage_matrix[:, 2],
        _R_FASTCLUSTER_WARD_D_HEIGHTS,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(result.get_cluster_labels(2), [1, 1, 1, 2, 2])
    np.testing.assert_array_equal(result.get_cluster_labels(3), [1, 1, 1, 2, 3])


def test_condensed_core_rejects_a_mismatched_length():
    from sequenzo.clustering import clustering_c_code

    with pytest.raises(RuntimeError, match="size mismatch"):
        clustering_c_code.cluster_from_condensed(
            np.zeros(2, dtype=np.float64), 3, "ward_d", True, False
        )


def test_ward_d_feature_input_matches_r_fastcluster_distance_semantics():
    from_features = Cluster(
        X_features=_FEATURES,
        clustering_method="ward_d",
        fast_path=True,
    )
    from_distances = Cluster(
        pdist(_FEATURES, metric="euclidean"),
        clustering_method="ward_d",
        fast_path=True,
    )

    np.testing.assert_allclose(
        from_features.linkage_matrix,
        from_distances.linkage_matrix,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        from_features.linkage_matrix[:, 2],
        _R_FASTCLUSTER_WARD_D_HEIGHTS,
        rtol=1e-12,
        atol=1e-12,
    )


def test_ward_d2_feature_and_distance_paths_match_scipy():
    original = _FEATURES.copy()
    from_features = Cluster(
        X_features=_FEATURES,
        clustering_method="ward_d2",
        fast_path=True,
    )
    from_distances = Cluster(
        pdist(_FEATURES, metric="euclidean"),
        clustering_method="ward_d2",
        fast_path=True,
    )
    expected = linkage(_FEATURES, method="ward")

    np.testing.assert_allclose(
        from_features.linkage_matrix, expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        from_distances.linkage_matrix, expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(_FEATURES, original)


def test_ward_d2_feature_input_can_be_used_in_place():
    working = _FEATURES.copy()
    expected = Cluster(
        X_features=_FEATURES,
        clustering_method="ward_d2",
        fast_path=True,
    )

    result = Cluster(
        X_features=working,
        clustering_method="ward_d2",
        fast_path=True,
        preserve_input=False,
    )

    np.testing.assert_allclose(
        result.linkage_matrix, expected.linkage_matrix, rtol=0.0, atol=0.0
    )
    assert not np.array_equal(working, _FEATURES)
    assert result.condensed_matrix is None


def test_default_condensed_input_is_owned_by_cluster():
    condensed = squareform(_DISS, checks=False)
    original = condensed.copy()

    result = Cluster(condensed, clustering_method="ward_d", fast_path=True)
    condensed[:] = 0.0

    assert not np.shares_memory(result.condensed_matrix, condensed)
    assert np.array_equal(result.condensed_matrix, original)


def test_default_full_input_is_owned_by_cluster():
    matrix = _DISS.copy()
    expected = squareform(matrix, checks=False)

    result = Cluster(matrix, clustering_method="ward_d", fast_path=True)
    matrix[:] = 0.0

    assert not np.shares_memory(result.condensed_matrix, matrix)
    np.testing.assert_array_equal(result.condensed_matrix, expected)


def test_condensed_input_can_be_used_in_place_without_changing_the_tree():
    condensed = squareform(_DISS, checks=False)
    working = condensed.copy()
    expected = Cluster(condensed, clustering_method="ward_d", fast_path=True)

    result = Cluster(
        working,
        clustering_method="ward_d",
        fast_path=True,
        preserve_input=False,
    )

    assert np.allclose(result.linkage_matrix, expected.linkage_matrix, rtol=0.0, atol=0.0)
    assert np.array_equal(result.get_cluster_labels(2), expected.get_cluster_labels(2))
    assert not np.array_equal(working, condensed)
    assert result.condensed_matrix is None

    with pytest.raises(ValueError, match="preserve_input=False"):
        ClusterQuality(result)


def test_cluster_quality_keeps_condensed_distances_compact_until_requested():
    condensed = squareform(_DISS, checks=False)
    cluster = Cluster(condensed, clustering_method="ward_d", fast_path=True)

    quality = ClusterQuality(cluster, max_clusters=3)

    assert cluster._full_matrix is None
    np.testing.assert_allclose(quality.matrix, _DISS, rtol=1e-12, atol=1e-12)


def test_cluster_quality_keeps_direct_matrix_input():
    quality = ClusterQuality(_DISS, max_clusters=3, clustering_method="ward_d")

    np.testing.assert_allclose(quality.matrix, _DISS, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("ward_variant", ["ward_d", "single"])
def test_non_ward_d2_rejects_sklearn_early_stop_semantics(ward_variant):
    with pytest.raises(ValueError, match="Ward.D2"):
        ward_labels_only(
            2,
            X_features=_FEATURES,
            ward_variant=ward_variant,
            early_stop=True,
        )


def test_ward_labels_only_default_matches_cluster_default():
    features = np.array(
        [
            [0.5969296630302747, 0.3077656870097005],
            [0.1314992599933992, -1.088382071626557],
            [0.5018268154817228, 1.848029865672147],
            [-0.11676844070960773, -0.7806608338156458],
            [0.7924985369783882, -0.647964177382318],
        ]
    )
    expected = Cluster(
        X_features=features,
        clustering_method="ward",
        fast_path=True,
    ).get_cluster_labels(3)

    np.testing.assert_array_equal(
        ward_labels_only(3, X_features=features),
        expected,
    )
