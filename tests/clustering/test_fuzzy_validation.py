import numpy as np
import pandas as pd
import pytest

from sequenzo.clustering import (
    wfcmdd,
    crispness,
    cluster_range_from_partitions,
    cluster_association,
    boot_cluster_range,
)


def _toy_distance() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 4.0, 5.0, 6.0],
            [1.0, 0.0, 2.0, 3.0, 4.0],
            [4.0, 2.0, 0.0, 1.0, 2.0],
            [5.0, 3.0, 1.0, 0.0, 1.0],
            [6.0, 4.0, 2.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def test_wfcmdd_runs_and_normalizes_membership():
    diss = _toy_distance()
    result = wfcmdd(diss, memb=[0, 2, 4], method="FCMdd", m=2.0, iter_max=50)
    assert result.memb.shape == (5, 3)
    assert np.allclose(result.memb.sum(axis=1), 1.0, atol=1e-8)
    assert result.mobile_centers.shape == (3,)
    assert result.functional > 0.0
    assert crispness(result.memb).shape == (5,)


def test_wfcmdd_rejects_invalid_inputs():
    diss = _toy_distance()
    with pytest.raises(ValueError, match="greater than 1"):
        wfcmdd(diss, memb=[0, 2, 4], method="FCMdd", m=1.0)
    asymmetric = diss.copy()
    asymmetric[0, 1] = 99.0
    with pytest.raises(ValueError, match="symmetric"):
        wfcmdd(asymmetric, memb=[0, 2, 4], method="FCMdd")
    with pytest.raises(ValueError, match="positive"):
        wfcmdd(diss, memb=[0, 2, 4], method="NCdd", dnoise=0.0)


def test_wfcmdd_pcmdd_runs_with_fixed_eta():
    diss = _toy_distance()
    eta = np.array([2.0, 2.0, 2.0], dtype=float)
    result = wfcmdd(diss, memb=[0, 2, 4], method="PCMdd", m=2.0, eta=eta, iter_max=20)
    assert result.memb.shape == (5, 3)
    assert result.functional > 0.0


def test_fuzzy_group_labels_match_membership_flat_order():
    membership = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
        ],
        dtype=float,
    )
    n_obs, n_clusters = membership.shape
    cluster_names = ["C1", "C2"]
    flat_membership = membership.reshape(-1)
    group_labels = np.tile(np.asarray(cluster_names, dtype=object), n_obs)
    expected_labels = np.array(["C1", "C2", "C1", "C2"], dtype=object)
    assert np.array_equal(group_labels, expected_labels)
    assert np.allclose(flat_membership, [0.9, 0.1, 0.2, 0.8])
    for i in range(n_obs * n_clusters):
        assert group_labels[i] == cluster_names[i % n_clusters]


def test_cluster_range_from_partitions_matches_cpp():
    diss = _toy_distance()
    clustering = pd.DataFrame(
        {
            "k2": [1, 1, 2, 2, 2],
            "k3": [1, 1, 2, 2, 3],
        }
    )
    result = cluster_range_from_partitions(diss, clustering)
    assert list(result.stats.columns) == [
        "PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC",
    ]
    assert result.stats.shape[0] == 2


def test_boot_cluster_range_with_distance_builder():
    diss = _toy_distance()
    clustering = pd.DataFrame({"k2": [1, 1, 2, 2, 2], "k3": [1, 1, 2, 2, 3]})
    result = boot_cluster_range(
        clustering=clustering,
        distance_builder=lambda idx: diss[np.ix_(idx, idx)],
        n_boot=5,
        sample_size=4,
        sampling="simple",
        random_state=1,
    )
    assert result.meant.shape == (2, 10)
    assert result.stderr.shape == (2, 10)


def test_cluster_association_returns_baseline_row():
    diss = _toy_distance()
    covar = np.array([0, 0, 1, 1, 1])
    clustering = cluster_range_from_partitions(
        diss,
        pd.DataFrame({"k2": [1, 1, 2, 2, 2]}),
    )
    assoc = cluster_association(clustering, diss, covar)
    assert "No Clustering" in assoc.index
    assert {"Unaccounted", "Remaining", "BIC", "numcluster"}.issubset(assoc.columns)
