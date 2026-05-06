import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.dissimilarity_measures import get_distance_matrix


def _seqdata():
    time_cols = [f"T{i}" for i in range(1, 6)]
    data = pd.DataFrame(
        [
            [1, 1, 2, 2, 3],
            [1, 2, 2, 3, 3],
            [3, 3, 2, 2, 1],
            [1, 1, 2, 2, 3],
        ],
        columns=time_cols,
    )
    data.insert(0, "id", [f"s{i}" for i in range(len(data))])
    return SequenceData(data, time=time_cols, states=[1, 2, 3], id_col="id")


def _expected_euclid(rows, *, norm_auto=False, condensed=False):
    values = np.asarray(rows, dtype=np.int32)
    distances = np.zeros((values.shape[0], values.shape[0]), dtype=np.float64)
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            mismatches = np.count_nonzero(values[i] != values[j])
            if norm_auto:
                value = np.sqrt(mismatches / values.shape[1])
            else:
                value = np.sqrt(2.0 * mismatches)
            distances[i, j] = value
            distances[j, i] = value
    if condensed:
        return distances[np.triu_indices(values.shape[0], k=1)]
    return distances


def test_euclid_categorical_fast_path_matches_expected_full_matrix():
    seqdata = _seqdata()
    rows = seqdata.values

    fast = get_distance_matrix(seqdata, method="EUCLID", norm="none", euclid_backend="categorical")

    np.testing.assert_allclose(
        np.asarray(fast, dtype=np.float64),
        _expected_euclid(rows, norm_auto=False),
        rtol=1e-10,
        atol=1e-10,
    )


def test_euclid_categorical_fast_path_matches_dense_when_full_matrix_is_false():
    seqdata = _seqdata()

    dense = get_distance_matrix(
        seqdata,
        method="EUCLID",
        norm="auto",
        full_matrix=False,
        euclid_backend="dense",
    )
    fast = get_distance_matrix(
        seqdata,
        method="EUCLID",
        norm="auto",
        full_matrix=False,
        euclid_backend="categorical",
    )

    np.testing.assert_allclose(
        np.asarray(fast, dtype=np.float64),
        np.asarray(dense, dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def test_euclid_categorical_fast_path_rejects_unsupported_options():
    seqdata = _seqdata()

    with pytest.raises(ValueError, match="euclid_backend='categorical'"):
        get_distance_matrix(
            seqdata,
            method="EUCLID",
            euclid_backend="categorical",
            overlap=True,
        )
