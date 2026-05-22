import numpy as np

from sequenzo.seqhmm.nhmm_utils import (
    compute_emission_probs_with_covariates,
    compute_initial_probs_with_covariates,
    compute_transition_probs_with_covariates,
    softmax,
)


def _transition_loop(eta_A, X, n_states):
    n_sequences, n_timepoints, _ = X.shape
    probs = np.zeros((n_sequences, n_timepoints, n_states, n_states))
    for seq_idx in range(n_sequences):
        for t in range(n_timepoints):
            eta = np.zeros((n_states, n_states))
            for i in range(n_states):
                for j in range(n_states):
                    eta[i, j] = np.sum(X[seq_idx, t, :] * eta_A[:, i, j])
            for i in range(n_states):
                probs[seq_idx, t, i, :] = softmax(eta[i, :])
    return probs


def _emission_loop(eta_B, X, n_states, n_symbols):
    n_sequences, n_timepoints, _ = X.shape
    probs = np.zeros((n_sequences, n_timepoints, n_states, n_symbols))
    for seq_idx in range(n_sequences):
        for t in range(n_timepoints):
            eta = np.zeros((n_states, n_symbols))
            for i in range(n_states):
                for j in range(n_symbols):
                    eta[i, j] = np.sum(X[seq_idx, t, :] * eta_B[:, i, j])
            for i in range(n_states):
                probs[seq_idx, t, i, :] = softmax(eta[i, :])
    return probs


def _initial_loop(eta_pi, X, n_states):
    n_sequences = X.shape[0]
    probs = np.zeros((n_sequences, n_states))
    for seq_idx in range(n_sequences):
        eta = np.zeros(n_states)
        for i in range(n_states):
            eta[i] = np.sum(X[seq_idx, 0, :] * eta_pi[:, i])
        probs[seq_idx, :] = softmax(eta)
    return probs


def test_transition_probability_kernel_matches_loop_baseline():
    rng = np.random.RandomState(123)
    X = rng.normal(size=(7, 5, 4))
    eta_A = rng.normal(size=(4, 3, 3))

    actual = compute_transition_probs_with_covariates(eta_A, X, n_states=3)
    expected = _transition_loop(eta_A, X, n_states=3)

    assert actual.shape == (7, 5, 3, 3)
    assert np.allclose(actual, expected, atol=1e-12)
    assert np.allclose(actual.sum(axis=-1), 1.0)


def test_emission_probability_kernel_matches_loop_baseline():
    rng = np.random.RandomState(456)
    X = rng.normal(size=(6, 4, 5))
    eta_B = rng.normal(size=(5, 3, 4))

    actual = compute_emission_probs_with_covariates(
        eta_B,
        X,
        n_states=3,
        n_symbols=4,
    )
    expected = _emission_loop(eta_B, X, n_states=3, n_symbols=4)

    assert actual.shape == (6, 4, 3, 4)
    assert np.allclose(actual, expected, atol=1e-12)
    assert np.allclose(actual.sum(axis=-1), 1.0)


def test_initial_probability_kernel_matches_loop_baseline():
    rng = np.random.RandomState(789)
    X = rng.normal(size=(8, 3, 4))
    eta_pi = rng.normal(size=(4, 3))

    actual = compute_initial_probs_with_covariates(eta_pi, X, n_states=3)
    expected = _initial_loop(eta_pi, X, n_states=3)

    assert actual.shape == (8, 3)
    assert np.allclose(actual, expected, atol=1e-12)
    assert np.allclose(actual.sum(axis=-1), 1.0)
