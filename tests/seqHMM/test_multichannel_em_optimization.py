import time
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd

from sequenzo import SequenceData
from sequenzo.seqhmm import build_hmm
from sequenzo.seqhmm.multichannel_emission import (
    _accumulate_sequence_statistics,
    _accumulate_equal_length_batch_statistics,
    _auto_equal_length_batch_size,
    _prepare_length_grouped_batches,
    _prepare_equal_length_batch,
    _unique_sequence_counts,
    _unique_equal_length_matrices,
    _unique_rows_with_counts,
)


TIME_COLS = [f"t{i}" for i in range(10)]


def _seqdata_from_patterns(patterns, states, n_repeats, prefix):
    rows = patterns * n_repeats
    df = pd.DataFrame(rows, columns=TIME_COLS)
    df.insert(0, "id", [f"s{i}" for i in range(len(rows))])
    with redirect_stdout(StringIO()):
        return SequenceData(df, time=TIME_COLS, states=states, id_col="id")


def test_multichannel_em_compresses_repeated_sequence_patterns_fast_enough():
    patterns_a = [
        ["A0", "A0", "A1", "A1", "A2", "A2", "A3", "A3", "A0", "A1"],
        ["A1", "A1", "A2", "A2", "A3", "A3", "A0", "A0", "A1", "A2"],
        ["A2", "A2", "A3", "A3", "A0", "A0", "A1", "A1", "A2", "A3"],
    ]
    patterns_b = [
        ["B0", "B1", "B1", "B2", "B2", "B3", "B3", "B0", "B0", "B1"],
        ["B1", "B2", "B2", "B3", "B3", "B0", "B0", "B1", "B1", "B2"],
        ["B2", "B3", "B3", "B0", "B0", "B1", "B1", "B2", "B2", "B3"],
    ]
    patterns_c = [
        ["C3", "C3", "C2", "C2", "C1", "C1", "C0", "C0", "C3", "C2"],
        ["C2", "C2", "C1", "C1", "C0", "C0", "C3", "C3", "C2", "C1"],
        ["C1", "C1", "C0", "C0", "C3", "C3", "C2", "C2", "C1", "C0"],
    ]

    channels = [
        _seqdata_from_patterns(patterns_a, [f"A{i}" for i in range(4)], 180, "A"),
        _seqdata_from_patterns(patterns_b, [f"B{i}" for i in range(4)], 180, "B"),
        _seqdata_from_patterns(patterns_c, [f"C{i}" for i in range(4)], 180, "C"),
    ]
    model = build_hmm(channels, n_states=4, random_state=7)

    start = time.perf_counter()
    model.fit(n_iter=2, tol=0.0, verbose=False)
    elapsed = time.perf_counter() - start

    assert np.isfinite(model.log_likelihood)
    assert model.n_iter == 2
    assert elapsed < 2.0


def test_equal_length_batch_preparation_adapts_to_repetition():
    rng = np.random.default_rng(42)
    seq_length = 8
    n_unique = 600
    unique_observations = [
        rng.integers(0, 1000, size=n_unique * seq_length, dtype=np.int32),
        rng.integers(0, 1000, size=n_unique * seq_length, dtype=np.int32),
    ]
    unique_lengths = np.full(n_unique, seq_length, dtype=np.int32)

    unique_batch = _prepare_equal_length_batch(unique_observations, unique_lengths, n_iter=1)

    assert unique_batch is not None
    _, unique_counts, unique_mode = unique_batch
    assert unique_mode == "direct"
    assert len(unique_counts) == n_unique

    patterns = 4
    n_repeated = 600
    repeated_observations = [
        np.tile(rng.integers(0, 5, size=patterns * seq_length, dtype=np.int32), n_repeated // patterns),
        np.tile(rng.integers(0, 5, size=patterns * seq_length, dtype=np.int32), n_repeated // patterns),
    ]
    repeated_lengths = np.full(n_repeated, seq_length, dtype=np.int32)

    repeated_batch = _prepare_equal_length_batch(repeated_observations, repeated_lengths, n_iter=2)

    assert repeated_batch is not None
    _, repeated_counts, repeated_mode = repeated_batch
    assert repeated_mode == "compressed"
    assert len(repeated_counts) == patterns


def test_equal_length_batch_preparation_compresses_large_unique_batches():
    rng = np.random.default_rng(123)
    seq_length = 8
    n_sequences = 1000
    observations = [
        rng.integers(0, 1000, size=n_sequences * seq_length, dtype=np.int32),
        rng.integers(0, 1000, size=n_sequences * seq_length, dtype=np.int32),
    ]
    lengths = np.full(n_sequences, seq_length, dtype=np.int32)

    batch = _prepare_equal_length_batch(observations, lengths, n_iter=1)

    assert batch is not None
    _, counts, mode = batch
    assert mode == "compressed"
    assert len(counts) == n_sequences


def test_unique_rows_with_counts_matches_numpy_axis_unique():
    rows = np.array(
        [
            [1, 2, 3],
            [3, 2, 1],
            [1, 2, 3],
            [5, 5, 5],
            [3, 2, 1],
        ],
        dtype=np.int32,
    )

    unique_rows, counts = _unique_rows_with_counts(rows)
    expected_rows, expected_counts = np.unique(rows, axis=0, return_counts=True)

    np.testing.assert_array_equal(unique_rows, expected_rows)
    np.testing.assert_array_equal(counts, expected_counts)


def test_direct_and_compressed_equal_length_batches_have_matching_statistics():
    rng = np.random.default_rng(7)
    n_states = 3
    seq_length = 5
    patterns = 3
    repeats = 12
    observations = [
        np.tile(rng.integers(0, 4, size=patterns * seq_length, dtype=np.int32), repeats),
        np.tile(rng.integers(0, 5, size=patterns * seq_length, dtype=np.int32), repeats),
    ]
    lengths = np.full(patterns * repeats, seq_length, dtype=np.int32)
    n_sequences = len(lengths)
    direct_matrices = [
        observation.reshape(n_sequences, seq_length)
        for observation in observations
    ]
    direct_counts = np.ones(n_sequences, dtype=float)
    compressed_matrices, compressed_counts = _unique_equal_length_matrices(observations, lengths)

    initial = rng.dirichlet(np.ones(n_states))
    transition = rng.dirichlet(np.ones(n_states), size=n_states)
    emission_probs = [
        rng.dirichlet(np.ones(4), size=n_states),
        rng.dirichlet(np.ones(5), size=n_states),
    ]

    def accumulate(matrices, counts):
        gamma_0 = np.zeros(n_states, dtype=float)
        xi_sum = np.zeros((n_states, n_states), dtype=float)
        gamma_sum_trans = np.zeros(n_states, dtype=float)
        emission_counts = [
            np.zeros_like(emission_ch, dtype=float)
            for emission_ch in emission_probs
        ]
        emission_denominators = [
            np.zeros(n_states, dtype=float)
            for _ in emission_probs
        ]
        log_likelihood = _accumulate_equal_length_batch_statistics(
            matrices,
            counts,
            initial,
            transition,
            emission_probs,
            gamma_0,
            xi_sum,
            gamma_sum_trans,
            emission_counts,
            emission_denominators,
        )
        return log_likelihood, gamma_0, xi_sum, gamma_sum_trans, emission_counts, emission_denominators

    direct_stats = accumulate(direct_matrices, direct_counts)
    compressed_stats = accumulate(compressed_matrices, compressed_counts)

    for direct, compressed in zip(direct_stats[:4], compressed_stats[:4]):
        np.testing.assert_allclose(direct, compressed, rtol=1e-10, atol=1e-10)
    for direct_counts_ch, compressed_counts_ch in zip(direct_stats[4], compressed_stats[4]):
        np.testing.assert_allclose(direct_counts_ch, compressed_counts_ch, rtol=1e-10, atol=1e-10)
    for direct_denoms_ch, compressed_denoms_ch in zip(direct_stats[5], compressed_stats[5]):
        np.testing.assert_allclose(direct_denoms_ch, compressed_denoms_ch, rtol=1e-10, atol=1e-10)


def test_length_grouped_batches_match_scalar_variable_length_statistics():
    rng = np.random.default_rng(11)
    sequences = [
        ([0, 1, 0], [1, 1, 0]),
        ([1, 0, 1, 0], [0, 1, 0, 1]),
        ([0, 1, 0], [1, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 1, 0], [0, 1, 0, 1]),
    ]
    observations = [
        np.asarray([state for seq in sequences for state in seq[ch_idx]], dtype=np.int32)
        for ch_idx in range(2)
    ]
    lengths = np.asarray([len(seq[0]) for seq in sequences], dtype=np.int32)
    n_states = 3
    initial = rng.dirichlet(np.ones(n_states))
    transition = rng.dirichlet(np.ones(n_states), size=n_states)
    emission_probs = [
        rng.dirichlet(np.ones(2), size=n_states),
        rng.dirichlet(np.ones(2), size=n_states),
    ]

    def empty_stats():
        return (
            np.zeros(n_states, dtype=float),
            np.zeros((n_states, n_states), dtype=float),
            np.zeros(n_states, dtype=float),
            [np.zeros_like(emission_ch, dtype=float) for emission_ch in emission_probs],
            [np.zeros(n_states, dtype=float) for _ in emission_probs],
        )

    grouped_stats = empty_stats()
    grouped_log_likelihood = 0.0
    grouped_batches = _prepare_length_grouped_batches(observations, lengths, n_iter=2)
    assert grouped_batches is not None
    assert {batch[0][0].shape[1] for batch in grouped_batches} == {3, 4}

    for matrices, counts, _mode in grouped_batches:
        grouped_log_likelihood += _accumulate_equal_length_batch_statistics(
            matrices,
            counts,
            initial,
            transition,
            emission_probs,
            *grouped_stats,
        )

    scalar_stats = empty_stats()
    scalar_log_likelihood = 0.0
    for obs_list, count in _unique_sequence_counts(observations, lengths):
        scalar_log_likelihood += _accumulate_sequence_statistics(
            obs_list,
            count,
            initial,
            transition,
            emission_probs,
            *scalar_stats,
        )

    np.testing.assert_allclose(grouped_log_likelihood, scalar_log_likelihood, rtol=1e-10, atol=1e-10)
    for grouped, scalar in zip(grouped_stats[:3], scalar_stats[:3]):
        np.testing.assert_allclose(grouped, scalar, rtol=1e-10, atol=1e-10)
    for grouped_counts, scalar_counts in zip(grouped_stats[3], scalar_stats[3]):
        np.testing.assert_allclose(grouped_counts, scalar_counts, rtol=1e-10, atol=1e-10)
    for grouped_denoms, scalar_denoms in zip(grouped_stats[4], scalar_stats[4]):
        np.testing.assert_allclose(grouped_denoms, scalar_denoms, rtol=1e-10, atol=1e-10)


def test_equal_length_batch_size_shrinks_for_memory_heavier_models():
    small_batch = _auto_equal_length_batch_size(
        n_sequences=100_000,
        seq_length=10,
        n_states=3,
    )
    large_batch = _auto_equal_length_batch_size(
        n_sequences=100_000,
        seq_length=200,
        n_states=12,
    )

    assert 1 <= large_batch < small_batch <= 5000


def test_multichannel_fit_stores_final_parameter_log_likelihood():
    patterns_a = [
        ["A0", "A1", "A0", "A1", "A0", "A1", "A0", "A1", "A0", "A1"],
        ["A1", "A0", "A1", "A0", "A1", "A0", "A1", "A0", "A1", "A0"],
    ]
    patterns_b = [
        ["B0", "B0", "B1", "B1", "B0", "B1", "B0", "B0", "B1", "B1"],
        ["B1", "B1", "B0", "B0", "B1", "B0", "B1", "B1", "B0", "B0"],
    ]
    channels = [
        _seqdata_from_patterns(patterns_a, ["A0", "A1"], 12, "A"),
        _seqdata_from_patterns(patterns_b, ["B0", "B1"], 12, "B"),
    ]
    model = build_hmm(channels, n_states=2, random_state=9)

    model.fit(n_iter=2, tol=0.0, verbose=False)

    np.testing.assert_allclose(model.log_likelihood, model.score(), rtol=1e-10, atol=1e-10)
