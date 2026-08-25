import numpy as np
import pandas as pd
import pytest

from sequenzo.clustering import KMedoids, cluster_sequences_pamonce
from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures import get_distance_matrix


def _sequence_data(patterns, weights=None):
    time_columns = [f"T{index + 1}" for index in range(len(patterns[0]))]
    frame = pd.DataFrame(patterns, columns=time_columns)
    frame.insert(0, "id", np.arange(len(frame)))
    return SequenceData(
        frame,
        time=time_columns,
        states=["A", "B"],
        id_col="id",
        weights=weights,
    )


def test_codebook_probe_does_not_build_a_full_inverse_array(monkeypatch):
    import sequenzo.clustering.sequence_pamonce as sequence_pamonce

    distance = np.arange(200_000, dtype=np.float64)
    calls = []
    original_unique = sequence_pamonce.np.unique

    def tracked_unique(values, *args, **kwargs):
        calls.append((values.size, kwargs.get("return_inverse", False)))
        return original_unique(values, *args, **kwargs)

    monkeypatch.setattr(sequence_pamonce.np, "unique", tracked_unique)
    encoded, codebook = sequence_pamonce._lossless_codebook(distance)

    assert encoded is distance
    assert codebook is None
    assert max(size for size, _ in calls) < distance.size
    assert not any(return_inverse for _, return_inverse in calls)


def test_integer_hamming_compares_unique_sequences_only(monkeypatch):
    import sequenzo.clustering.sequence_pamonce as sequence_pamonce

    sequences = np.repeat(
        np.array([[1, 1, 1, 1], [1, 2, 1, 2]], dtype=np.int32),
        50,
        axis=0,
    )
    compared_rows = []
    original_count_nonzero = sequence_pamonce.np.count_nonzero

    def tracked_count_nonzero(values, *args, **kwargs):
        compared_rows.append(values.shape[0])
        return original_count_nonzero(values, *args, **kwargs)

    monkeypatch.setattr(
        sequence_pamonce.np, "count_nonzero", tracked_count_nonzero
    )
    distance = sequence_pamonce._integer_hamming_condensed(sequences)

    assert max(compared_rows) == 1
    assert np.count_nonzero(distance == 2) == 2_500


def test_sequence_workflow_matches_full_classic_pam():
    patterns = [
        ["A", "A", "A", "A"],
        ["B", "B", "B", "B"],
        ["A", "B", "A", "B"],
        ["B", "B", "B", "B"],
        ["A", "A", "A", "A"],
        ["A", "B", "A", "B"],
        ["A", "A", "A", "A"],
    ]
    seqdata = _sequence_data(
        patterns,
        weights=np.array([1.0, 2.0, 1.5, 0.5, 2.0, 1.0, 0.75]),
    )

    full_distance = get_distance_matrix(
        seqdata=seqdata, method="HAM", norm="none", full_matrix=False
    )
    classic = KMedoids(
        full_distance,
        k=2,
        weights=seqdata.weights,
        npass=1,
        method="PAM",
        threads=1,
        verbose=False,
    )
    compressed, diagnostics = cluster_sequences_pamonce(
        seqdata,
        k=2,
        method="HAM",
        distance_kwargs={"norm": "none"},
        threads=1,
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(compressed, classic)
    assert diagnostics["original_n"] == 7
    assert diagnostics["candidate_storage_bytes"] == 0


def test_sequence_hamming_fast_path_uses_lossless_integer_condensed_storage():
    patterns = [
        ["A", "A", "A", "A"],
        ["A", "A", "A", "A"],
        ["A", "B", "A", "B"],
        ["B", "B", "B", "B"],
        ["B", "B", "B", "B"],
    ]
    seqdata = _sequence_data(patterns)

    membership, diagnostics = cluster_sequences_pamonce(
        seqdata,
        k=2,
        method="HAM",
        threads=1,
        return_diagnostics=True,
    )

    assert membership.shape == (5,)
    assert diagnostics["distance_representation"] == "integer_condensed"
    assert diagnostics["distance_dtype"] == "uint8"
    assert diagnostics["distance_storage_bytes"] == 10


def test_sequence_discrete_fractional_distances_use_lossless_codebook():
    patterns = [
        ["A", "A", "A", "A"],
        ["A", "A", "A", "B"],
        ["A", "A", "B", "B"],
        ["A", "B", "B", "B"],
        ["B", "B", "B", "B"],
        ["A", "A", "A", "A"],
    ]
    seqdata = _sequence_data(patterns)

    membership, diagnostics = cluster_sequences_pamonce(
        seqdata,
        k=2,
        method="HAM",
        distance_kwargs={"norm": "maxlength"},
        threads=1,
        return_diagnostics=True,
    )

    full_distance = get_distance_matrix(
        seqdata=seqdata,
        method="HAM",
        norm="maxlength",
        full_matrix=False,
    )
    classic = KMedoids(
        full_distance,
        k=2,
        method="PAM",
        threads=1,
        verbose=False,
    )
    np.testing.assert_array_equal(membership, classic)
    assert diagnostics["distance_representation"] == "codebook_condensed"
    assert diagnostics["distance_codebook_size"] == 5
    assert diagnostics["distance_storage_bytes"] < full_distance.nbytes


def test_sequence_workflow_keeps_data_dependent_distance_costs():
    patterns = [["A", "A", "A", "A"]] * 8 + [
        ["A", "B", "B", "B"],
        ["B", "B", "B", "B"],
        ["B", "A", "A", "A"],
    ]
    seqdata = _sequence_data(patterns)
    distance_kwargs = {"sm": "TRATE", "weighted": False, "norm": "none"}

    full_distance = get_distance_matrix(
        seqdata=seqdata,
        method="HAM",
        full_matrix=False,
        **distance_kwargs,
    )
    classic = KMedoids(
        full_distance,
        k=2,
        npass=1,
        method="PAM",
        threads=1,
        verbose=False,
    )
    result = cluster_sequences_pamonce(
        seqdata,
        k=2,
        method="HAM",
        distance_kwargs=distance_kwargs,
        threads=1,
    )

    np.testing.assert_array_equal(result, classic)


def test_general_distance_path_reads_sequence_values_once(monkeypatch):
    seqdata = _sequence_data(
        [["A", "A"], ["A", "B"], ["B", "A"], ["B", "B"]]
    )
    original_getter = SequenceData.values.fget
    reads = 0

    def counted_values(instance):
        nonlocal reads
        reads += 1
        return original_getter(instance)

    monkeypatch.setattr(SequenceData, "values", property(counted_values))
    cluster_sequences_pamonce(
        seqdata,
        k=2,
        method="HAM",
        distance_kwargs={"sm": "TRATE", "weighted": False},
        threads=1,
    )

    assert reads == 1


def test_default_sequence_weights_use_unit_weight_core_path(monkeypatch):
    import sequenzo.clustering.sequence_pamonce as sequence_pamonce

    seqdata = _sequence_data(
        [["A", "A"], ["A", "B"], ["B", "A"], ["B", "B"]]
    )
    captured = {}

    class PamonceProbe:
        def __init__(self, *args):
            captured["weights"] = args[4]
            self.n = args[0]

        def set_collect_diagnostics(self, enabled):
            pass

        def runclusterloop_one_based(self):
            return np.arange(1, self.n + 1, dtype=np.int32)

    monkeypatch.setattr(sequence_pamonce.clustering_c_code, "PAMonce", PamonceProbe)
    cluster_sequences_pamonce(seqdata, k=2, method="HAM", threads=1)

    assert captured["weights"].size == 0


def test_sequence_workflow_rejects_reference_sequence_output():
    seqdata = _sequence_data(
        [["A", "A"], ["A", "B"], ["B", "A"], ["B", "B"]]
    )

    with pytest.raises(ValueError, match="refseq"):
        cluster_sequences_pamonce(
            seqdata,
            k=2,
            method="HAM",
            distance_kwargs={"norm": "none", "refseq": 0},
            threads=1,
        )


def test_sequence_workflow_rejects_nested_distance_options():
    seqdata = _sequence_data(
        [["A", "A"], ["A", "B"], ["B", "A"], ["B", "B"]]
    )

    with pytest.raises(ValueError, match="directly"):
        cluster_sequences_pamonce(
            seqdata,
            k=2,
            method="HAM",
            distance_kwargs={"opts": {"refseq": 0}},
            threads=1,
        )


def test_sequence_workflow_validates_threads_before_distance_calculation():
    seqdata = _sequence_data([["A", "A"], ["A", "B"], ["B", "B"]])

    with pytest.raises(ValueError, match="threads"):
        cluster_sequences_pamonce(
            seqdata,
            k=2,
            method="HAM",
            distance_kwargs={"norm": "none"},
            threads=0,
        )
