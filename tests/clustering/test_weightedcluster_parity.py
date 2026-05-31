import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

from sequenzo.clustering import aggregate_cases, k_medoids_range
from sequenzo.clustering.utils.weightedcluster_compat import cutree_labels, divisive_hclust_linkage
from sequenzo.clustering.validation.partition_quality import cluster_range_from_partitions


_DISS_5 = np.array(
    [
        [0.0, 1.0, 1.0, 4.24264068711928, 5.65685424949238],
        [1.0, 0.0, 1.4142135623731, 3.60555127546399, 5.0],
        [1.0, 1.4142135623731, 0.0, 3.60555127546399, 5.0],
        [4.24264068711928, 3.60555127546399, 3.60555127546399, 0.0, 1.4142135623731],
        [5.65685424949238, 5.0, 5.0, 1.4142135623731, 0.0],
    ],
    dtype=float,
)


def test_package_imports_keep_callables_after_submodule_imports():
    import sequenzo.clustering.k_medoids_range  # noqa: F401
    import sequenzo.dissimilarity_measures.get_distance_matrix  # noqa: F401

    from sequenzo.clustering import k_medoids_range as imported_k_medoids_range
    from sequenzo.dissimilarity_measures import get_distance_matrix

    assert callable(imported_k_medoids_range)
    assert callable(get_distance_matrix)


def test_fresh_package_imports_keep_callables_after_submodule_imports():
    code = """
import sequenzo.clustering.k_medoids_range
import sequenzo.dissimilarity_measures.get_distance_matrix
from sequenzo.clustering import k_medoids_range
from sequenzo.dissimilarity_measures import get_distance_matrix
assert callable(k_medoids_range), type(k_medoids_range)
assert callable(get_distance_matrix), type(get_distance_matrix)
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_same_named_submodules_remain_importable_with_importlib():
    import importlib

    clustering_module = importlib.import_module("sequenzo.clustering.k_medoids_range")
    distance_module = importlib.import_module("sequenzo.dissimilarity_measures.get_distance_matrix")

    assert isinstance(clustering_module, types.ModuleType)
    assert isinstance(distance_module, types.ModuleType)


def test_same_named_submodules_remain_modules_with_dotted_imports():
    import sequenzo.clustering.k_medoids_range as clustering_module
    import sequenzo.dissimilarity_measures.get_distance_matrix as distance_module

    assert isinstance(clustering_module, types.ModuleType)
    assert isinstance(distance_module, types.ModuleType)


def test_fresh_dotted_submodule_imports_remain_modules():
    code = """
import types
import sequenzo.clustering.k_medoids_range as clustering_module
import sequenzo.dissimilarity_measures.get_distance_matrix as distance_module
assert isinstance(clustering_module, types.ModuleType), type(clustering_module)
assert isinstance(distance_module, types.ModuleType), type(distance_module)
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_plain_dotted_imports_keep_submodule_attribute_chains():
    code = """
import sequenzo.clustering
import sequenzo.clustering.k_medoids_range
import sequenzo.clustering.compare_cluster_methods
import sequenzo.clustering.property_based_clustering
import sequenzo.dissimilarity_measures
import sequenzo.dissimilarity_measures.get_distance_matrix
import sequenzo.dissimilarity_measures.get_substitution_cost_matrix
assert callable(sequenzo.clustering.k_medoids_range)
assert callable(sequenzo.clustering.k_medoids_range.k_medoids_range)
assert callable(sequenzo.clustering.compare_cluster_methods.compare_cluster_methods)
assert callable(sequenzo.clustering.property_based_clustering.property_based_clustering)
assert callable(sequenzo.dissimilarity_measures.get_distance_matrix)
assert callable(sequenzo.dissimilarity_measures.get_distance_matrix.get_distance_matrix)
assert callable(sequenzo.dissimilarity_measures.get_substitution_cost_matrix.get_substitution_cost_matrix)
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_aggregate_cases_matches_weightedcluster_reference():
    ref_dir = pytest.importorskip("pathlib").Path(__file__).resolve().parent
    summary_path = ref_dir / "ref_aggregate_cases_summary.csv"
    rows_path = ref_dir / "ref_aggregate_cases_rows.csv"
    if not summary_path.is_file() or not rows_path.is_file():
        pytest.skip("WeightedCluster aggregate_cases reference not found.")

    frame = pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 1],
            "b": ["x", "x", "y", "y", "x"],
        }
    )
    weights = np.array([1.0, 2.0, 1.0, 3.0, 4.0])
    result = aggregate_cases(frame, weights=weights)

    summary = pd.read_csv(summary_path)
    rows = pd.read_csv(rows_path)

    assert np.allclose(result.agg_weights, summary["aggWeights"].to_numpy(dtype=float))
    assert np.allclose(result.agg_index, summary["aggIndex"].to_numpy(dtype=int))
    assert np.allclose(result.disagg_index, rows["disaggIndex"].to_numpy(dtype=int))
    assert np.allclose(result.disagg_weights, rows["disaggWeight"].to_numpy(dtype=float))


def test_k_medoids_range_is_reproducible_with_random_state():
    weights = np.array([1.0, 2.0, 1.0, 1.0, 3.0])
    result = k_medoids_range(_DISS_5, kvals=[2, 3, 4], weights=weights, random_state=1)
    repeat = k_medoids_range(_DISS_5, kvals=[2, 3, 4], weights=weights, random_state=1)

    assert list(result.clustering.columns) == ["cluster2", "cluster3", "cluster4"]
    assert result.stats.shape[0] == 3
    assert np.array_equal(result.clustering.to_numpy(), repeat.clustering.to_numpy())
    assert np.allclose(result.stats.to_numpy(dtype=float), repeat.stats.to_numpy(dtype=float))


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("diana", {2: [1, 1, 1, 2, 2], 3: [1, 1, 1, 2, 3], 4: [1, 2, 1, 3, 4]}),
        ("beta.flexible", {2: [1, 1, 1, 2, 2], 3: [1, 1, 1, 2, 3], 4: [1, 1, 2, 3, 4]}),
    ],
)
def test_special_linkage_matches_weightedcluster_partitions(method, expected):
    linkage_matrix = divisive_hclust_linkage(_DISS_5, method)
    for k, labels in expected.items():
        assert cutree_labels(linkage_matrix, k).tolist() == labels


def test_partition_quality_matches_weightedcluster_k_medoids_labels():
    ref_dir = pytest.importorskip("pathlib").Path(__file__).resolve().parent
    stats_path = ref_dir / "ref_k_medoids_range_stats.csv"
    clustering_path = ref_dir / "ref_k_medoids_range_clustering.csv"
    if not stats_path.is_file() or not clustering_path.is_file():
        pytest.skip("WeightedCluster k_medoids_range reference not found.")

    diss = np.array(
        [
            [0.0, 1.0, 1.0, 4.24264068711928, 5.65685424949238],
            [1.0, 0.0, 1.4142135623731, 3.60555127546399, 5.0],
            [1.0, 1.4142135623731, 0.0, 3.60555127546399, 5.0],
            [4.24264068711928, 3.60555127546399, 3.60555127546399, 0.0, 1.4142135623731],
            [5.65685424949238, 5.0, 5.0, 1.4142135623731, 0.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 2.0, 1.0, 1.0, 3.0])
    clustering = pd.read_csv(clustering_path)

    result = cluster_range_from_partitions(diss, clustering, weights=weights)
    ref = pd.read_csv(stats_path, index_col=0)

    for metric in ref.columns:
        assert np.allclose(
            result.stats[metric].to_numpy(dtype=float),
            ref[metric].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )
