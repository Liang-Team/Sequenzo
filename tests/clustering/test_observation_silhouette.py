import numpy as np
import pytest

from sequenzo.clustering import observation_silhouette


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


def test_observation_silhouette_returns_weightedcluster_measures():
    diss = _toy_distance()
    clustering = np.array([1, 1, 2, 2, 2])
    weights = np.array([1.0, 2.0, 1.0, 1.0, 1.0])

    asw = observation_silhouette(diss, clustering, weights=weights, measure="ASW")
    asww = observation_silhouette(diss, clustering, weights=weights, measure="ASWw")

    assert asw.shape == (5,)
    assert asww.shape == (5,)
    assert np.all(np.isfinite(asw))
    assert np.all(np.isfinite(asww))
    assert not np.allclose(asw, asww)


def test_observation_silhouette_rejects_single_cluster():
    diss = _toy_distance()
    clustering = np.ones(5, dtype=int)
    with pytest.raises(ValueError, match="at least two distinct groups"):
        observation_silhouette(diss, clustering)


def test_observation_silhouette_matches_weightedcluster_reference():
    ref_path = (
        pytest.importorskip("pathlib").Path(__file__).resolve().parent
        / "ref_observation_silhouette.csv"
    )
    if not ref_path.is_file():
        pytest.skip(
            "WeightedCluster reference not found. Run weightedcluster_reference_silhouette.R"
        )

    import pandas as pd

    ref = pd.read_csv(ref_path)
    diss_cols = [col for col in ref.columns if col not in {"clustering", "weight", "ASW", "ASWw"}]
    diss = ref[diss_cols].to_numpy(dtype=float)
    clustering = ref["clustering"].to_numpy()
    weights = ref["weight"].to_numpy(dtype=float)

    asw = observation_silhouette(diss, clustering, weights=weights, measure="ASW")
    asww = observation_silhouette(diss, clustering, weights=weights, measure="ASWw")

    assert np.allclose(asw, ref["ASW"].to_numpy(dtype=float), rtol=1e-10, atol=1e-12)
    assert np.allclose(asww, ref["ASWw"].to_numpy(dtype=float), rtol=1e-10, atol=1e-12)
