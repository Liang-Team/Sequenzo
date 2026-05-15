import numpy as np

from tests.seqHMM._label_alignment import align_hmm_state_labels


def test_align_hmm_state_labels_recovers_permuted_parameters():
    reference = {
        "initial": np.array([0.2, 0.5, 0.3]),
        "transition": np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ),
        "emission": np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.2, 0.6],
            ]
        ),
    }
    permutation = [2, 0, 1]
    candidate = {
        "initial": reference["initial"][permutation],
        "transition": reference["transition"][np.ix_(permutation, permutation)],
        "emission": reference["emission"][permutation],
    }

    aligned, recovered = align_hmm_state_labels(candidate, reference)

    assert recovered == [1, 2, 0]
    assert np.allclose(aligned["initial"], reference["initial"])
    assert np.allclose(aligned["transition"], reference["transition"])
    assert np.allclose(aligned["emission"], reference["emission"])


def test_align_hmm_state_labels_handles_multichannel_emissions():
    reference = {
        "initial": np.array([0.6, 0.4]),
        "transition": np.array([[0.8, 0.2], [0.1, 0.9]]),
        "emission": [
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.array([[0.7, 0.3], [0.1, 0.9]]),
        ],
    }
    candidate = {
        "initial": reference["initial"][[1, 0]],
        "transition": reference["transition"][np.ix_([1, 0], [1, 0])],
        "emission": [channel[[1, 0]] for channel in reference["emission"]],
    }

    aligned, recovered = align_hmm_state_labels(candidate, reference)

    assert recovered == [1, 0]
    assert np.allclose(aligned["initial"], reference["initial"])
    assert np.allclose(aligned["transition"], reference["transition"])
    for aligned_channel, reference_channel in zip(aligned["emission"], reference["emission"]):
        assert np.allclose(aligned_channel, reference_channel)
