import numpy as np
import pandas as pd
import pytest

from sequenzo.clustering import aggregate_cases, k_medoids_range
from sequenzo.clustering.validation.partition_quality import cluster_range_from_partitions


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


def test_k_medoids_range_matches_weightedcluster_reference():
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
    result = k_medoids_range(diss, kvals=[2, 3, 4], weights=weights, random_state=1)
    ref_clustering = pd.read_csv(clustering_path)
    ref_stats = pd.read_csv(stats_path, index_col=0)

    assert np.array_equal(result.clustering.to_numpy(), ref_clustering.to_numpy())
    for metric in ref_stats.columns:
        assert np.allclose(
            result.stats[metric].to_numpy(dtype=float),
            ref_stats[metric].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )


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
