import importlib
from unittest import mock

import numpy as np
import pandas as pd
import pytest


def _distance_matrix(points):
    points = np.asarray(points, dtype=np.float64)
    return np.abs(points[:, None] - points[None, :])


def test_kmedoids_range_condensed_input_matches_full_input():
    import scipy.spatial.distance as scipy_distance

    from sequenzo.clustering.k_medoids_range import k_medoids_range

    square = _distance_matrix([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    condensed = scipy_distance.squareform(square, checks=False)
    full = k_medoids_range(
        square,
        kvals=[2, 3],
        method="PAMonce",
        random_state=7,
    )
    compact = k_medoids_range(
        condensed,
        kvals=[2, 3],
        method="PAMonce",
        random_state=7,
    )

    np.testing.assert_array_equal(compact.clustering, full.clustering)
    np.testing.assert_allclose(compact.stats, full.stats, equal_nan=True)
    np.testing.assert_array_equal(compact.clustering["cluster2"], [2, 2, 2, 5, 5, 5])
    np.testing.assert_array_equal(compact.clustering["cluster3"], [2, 2, 2, 4, 4, 6])


def test_kmedoids_range_passes_random_restarts_to_kmedoids(monkeypatch):
    range_module = importlib.import_module("sequenzo.clustering.k_medoids_range")
    calls = []
    original = range_module.KMedoids

    def capture(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(range_module, "KMedoids", capture)
    range_module.k_medoids_range(
        _distance_matrix([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
        kvals=[2, 3],
        method="PAMonce",
        npass=3,
        random_state=7,
    )

    assert all(call["initialclust"] is None for call in calls)
    assert all(call["npass"] == 3 for call in calls)
    assert all(call["random_state"] == 7 for call in calls)


def test_kmedoids_range_accepts_integer_pamonce_method():
    from sequenzo.clustering.k_medoids_range import k_medoids_range

    distance = _distance_matrix([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    named = k_medoids_range(distance, kvals=[2, 3], method="PAMonce")
    numbered = k_medoids_range(distance, kvals=[2, 3], method=3)

    np.testing.assert_array_equal(numbered.clustering, named.clustering)


def test_kmedoids_range_validates_weights_before_shared_build(monkeypatch):
    clustering_c_code = importlib.import_module(
        "sequenzo.clustering.clustering_c_code"
    )
    pamonce = mock.Mock(wraps=clustering_c_code.PAMonce)
    monkeypatch.setattr(clustering_c_code, "PAMonce", pamonce)
    from sequenzo.clustering.k_medoids_range import k_medoids_range

    with pytest.raises(ValueError, match="weights"):
        k_medoids_range(
            _distance_matrix([0.0, 1.0, 2.0, 10.0]),
            kvals=[2],
            weights=np.ones(3),
            method="PAMonce",
        )

    assert pamonce.call_count == 0


def test_kmedoids_range_bootstrap_keeps_point_estimates_and_varies_samples():
    from scipy.spatial.distance import squareform

    from sequenzo.clustering.k_medoids_range import k_medoids_range

    distance = _distance_matrix([0.0, 1.0, 2.0, 9.0, 11.0, 14.0])
    condensed = squareform(distance, checks=False)
    point = k_medoids_range(condensed, kvals=[2, 3])
    boot = k_medoids_range(
        condensed,
        kvals=[2, 3],
        n_boot=20,
        sample_size=6,
        random_state=17,
    )

    np.testing.assert_allclose(boot.stats, point.stats, equal_nan=True)
    assert all(values.shape == (20, 10) for values in boot.boot)
    assert np.nanmax(boot.stderr.to_numpy()) > 0.0


def test_weighted_bootstrap_defaults_to_the_observation_count():
    from sequenzo.clustering.k_medoids_range import k_medoids_range

    distance = _distance_matrix([0.0, 1.0, 2.0, 9.0, 11.0, 14.0])
    weights = np.full(6, 1.0 / 6.0)

    result = k_medoids_range(
        distance,
        kvals=[2],
        weights=weights,
        n_boot=2,
        random_state=17,
    )

    assert result.boot[0].shape == (2, 10)


def test_simple_bootstrap_draws_once_per_replication(monkeypatch):
    from types import SimpleNamespace

    import sequenzo.clustering.k_medoids_range as range_module

    clustering = pd.DataFrame(
        {
            "a": [0, 0, 1, 1],
            "b": [0, 1, 0, 1],
            "c": [0, 1, 1, 0],
        }
    )
    distance = _distance_matrix([0.0, 1.0, 2.0, 3.0])

    class FixedRng:
        def __init__(self):
            self.calls = 0

        def choice(self, *args, **kwargs):
            samples = (np.array([0, 1]), np.array([2, 3]))
            if self.calls >= len(samples):
                raise AssertionError("bootstrap repeated a draw")
            sample = samples[self.calls]
            self.calls += 1
            return sample

    rng = FixedRng()

    def quality(_distance, partitions, weights=None):
        stats = pd.DataFrame(
            np.zeros((partitions.shape[1], len(range_module.METRIC_ORDER))),
            columns=range_module.METRIC_ORDER,
        )
        return SimpleNamespace(
            clustering=partitions,
            kvals=np.ones(partitions.shape[1], dtype=int),
            stats=stats,
        )

    monkeypatch.setattr(range_module.np.random, "default_rng", lambda _: rng)
    monkeypatch.setattr(range_module, "cluster_range_from_partitions", quality)

    range_module._weighted_bootstrap_range(
        distance,
        clustering,
        weights=None,
        n_boot=2,
        sample_size=2,
        random_state=7,
    )

    assert rng.calls == 2


def test_kmedoids_range_allows_bootstrap_samples_smaller_than_k():
    from sequenzo.clustering.k_medoids_range import k_medoids_range

    result = k_medoids_range(
        _distance_matrix([0.0, 1.0, 2.0, 9.0, 11.0, 14.0]),
        kvals=[2, 4],
        n_boot=2,
        sample_size=3,
        random_state=17,
    )

    assert all(values.shape == (2, 10) for values in result.boot)


def test_stratified_bootstrap_keeps_condensed_distances_compact(monkeypatch):
    import scipy.spatial.distance as scipy_distance

    from sequenzo.clustering.k_medoids_range import k_medoids_range

    square = _distance_matrix([0.0, 1.0, 2.0, 9.0, 11.0, 14.0])
    condensed = scipy_distance.squareform(square, checks=False)

    squareform = mock.Mock(wraps=scipy_distance.squareform)
    monkeypatch.setattr(scipy_distance, "squareform", squareform)
    result = k_medoids_range(
        condensed,
        kvals=[2],
        n_boot=2,
        sample_size=4,
        sampling="clustering",
        random_state=17,
    )

    assert squareform.call_count == 0
    assert result.boot[0].shape == (2, 10)


def test_stratified_bootstrap_keeps_full_sample_point_estimates():
    from sequenzo.clustering.k_medoids_range import k_medoids_range

    distance = _distance_matrix([0.0, 1.0, 2.0, 9.0, 11.0, 14.0])
    point = k_medoids_range(distance, kvals=[2, 3])
    boot = k_medoids_range(
        distance,
        kvals=[2, 3],
        n_boot=3,
        sample_size=4,
        sampling="clustering",
        random_state=17,
    )

    np.testing.assert_allclose(boot.stats, point.stats, equal_nan=True)


def test_kmedoids_range_rejects_unsupported_sampling_mode(monkeypatch):
    range_module = importlib.import_module("sequenzo.clustering.k_medoids_range")
    bootstrap = mock.Mock(wraps=range_module.boot_cluster_range)
    monkeypatch.setattr(range_module, "boot_cluster_range", bootstrap)

    with pytest.raises(ValueError, match="sampling"):
        range_module.k_medoids_range(
            _distance_matrix([0.0, 1.0, 2.0, 9.0, 11.0, 14.0]),
            kvals=[2],
            n_boot=2,
            sample_size=4,
            sampling="medoids",
            random_state=17,
        )

    assert bootstrap.call_count == 0
