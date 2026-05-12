import numpy as np
import pandas as pd

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
