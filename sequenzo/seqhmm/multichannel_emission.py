"""
@Author  : Yuqi Liang 梁彧祺; Yapeng Wei 卫亚鹏
@File    : multichannel_emission.py
@Time    : 2025-11-08 13:52
@Desc    : EM algorithm for multichannel HMM

This module provides the EM algorithm implementation for multichannel HMM,
where each sequence has multiple parallel channels (e.g., marriage, children, residence).
"""

from typing import List, Optional, Tuple

import numpy as np

from .hmm import HMM
from .multichannel_utils import multichannel_to_hmmlearn_format


_EPS = np.finfo(float).tiny


def _normalize_vector(values: np.ndarray, fallback_size: int) -> np.ndarray:
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        return np.ones(fallback_size, dtype=float) / fallback_size
    return values / total


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(values, dtype=float).copy()
    row_sums = normalized.sum(axis=1, keepdims=True)
    valid = np.isfinite(row_sums[:, 0]) & (row_sums[:, 0] > 0.0)
    normalized[valid] /= row_sums[valid]
    normalized[~valid] = 1.0 / normalized.shape[1]
    return normalized


def _initialize_multichannel_parameters(model: HMM) -> None:
    n_states = model.n_states

    if model.initial_probs is None:
        model.initial_probs = np.ones(n_states, dtype=float) / n_states
    else:
        model.initial_probs = _normalize_vector(
            np.asarray(model.initial_probs, dtype=float),
            n_states,
        )

    if model.transition_probs is None:
        model.transition_probs = np.ones((n_states, n_states), dtype=float) / n_states
    else:
        model.transition_probs = _normalize_rows(np.asarray(model.transition_probs, dtype=float))

    if model.emission_probs is None or not isinstance(model.emission_probs, list):
        model.emission_probs = []
        for n_symbols_ch in model.n_symbols:
            emission_ch = np.random.rand(n_states, n_symbols_ch)
            model.emission_probs.append(_normalize_rows(emission_ch))
    else:
        model.emission_probs = [
            _normalize_rows(np.asarray(emission_ch, dtype=float))
            for emission_ch in model.emission_probs
        ]


def _multichannel_arrays(model: HMM) -> Tuple[List[np.ndarray], np.ndarray]:
    X_list, lengths = multichannel_to_hmmlearn_format(model.channels)
    observations = [X[:, 0].astype(np.int32, copy=False) for X in X_list]
    return observations, lengths.astype(np.int32, copy=False)


def _unique_sequence_counts(
    observations: List[np.ndarray],
    lengths: np.ndarray,
) -> List[Tuple[List[np.ndarray], int]]:
    counts = {}
    start = 0
    for seq_length in lengths:
        end = start + int(seq_length)
        key = tuple(tuple(obs[start:end].tolist()) for obs in observations)
        counts[key] = counts.get(key, 0) + 1
        start = end

    return [
        ([np.asarray(channel, dtype=np.int32) for channel in key], count)
        for key, count in counts.items()
    ]


def _unique_rows_with_counts(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    contiguous_rows = np.ascontiguousarray(rows)
    row_dtype = np.dtype(
        (np.void, contiguous_rows.dtype.itemsize * contiguous_rows.shape[1])
    )
    row_view = contiguous_rows.view(row_dtype).ravel()
    _, indices, counts = np.unique(row_view, return_index=True, return_counts=True)
    return contiguous_rows[indices], counts


def _unique_equal_length_matrices(
    observations: List[np.ndarray],
    lengths: np.ndarray,
) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
    direct_batch = _equal_length_matrices(observations, lengths)
    if direct_batch is None:
        return None

    matrices, _ = direct_batch
    return _compress_equal_length_matrices(matrices)


def _compress_equal_length_matrices(
    matrices: List[np.ndarray],
) -> Tuple[List[np.ndarray], np.ndarray]:
    seq_length = matrices[0].shape[1]
    combined = np.hstack(matrices)
    unique_rows, counts = _unique_rows_with_counts(combined)

    n_channels = len(matrices)
    unique_matrices = [
        np.ascontiguousarray(
            unique_rows[:, ch_idx * seq_length:(ch_idx + 1) * seq_length].astype(
                np.int32,
                copy=False,
            )
        )
        for ch_idx in range(n_channels)
    ]
    return unique_matrices, counts.astype(float, copy=False)


def _equal_length_matrices(
    observations: List[np.ndarray],
    lengths: np.ndarray,
) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
    if len(lengths) == 0 or not np.all(lengths == lengths[0]):
        return None

    n_sequences = len(lengths)
    seq_length = int(lengths[0])
    matrices = [
        np.ascontiguousarray(obs.reshape(n_sequences, seq_length))
        for obs in observations
    ]
    return matrices, np.ones(n_sequences, dtype=float)


def _sample_duplicate_fraction(
    matrices: List[np.ndarray],
    sample_size: int = 4096,
) -> float:
    n_sequences = matrices[0].shape[0]
    if n_sequences <= 1:
        return 0.0

    sample_n = min(n_sequences, sample_size)
    if sample_n == n_sequences:
        sample_indices = slice(None)
    else:
        sample_indices = np.linspace(0, n_sequences - 1, sample_n, dtype=np.intp)

    sample_rows = np.hstack([matrix[sample_indices] for matrix in matrices])
    n_unique = _unique_rows_with_counts(sample_rows)[0].shape[0]
    return max(0.0, 1.0 - (n_unique / sample_n))


def _should_compress_equal_length_batch(
    n_sequences: int,
    duplicate_fraction: float,
    n_iter: int,
) -> bool:
    return duplicate_fraction >= 0.50 or n_iter > 1 or n_sequences >= 1000


def _prepare_equal_length_batch(
    observations: List[np.ndarray],
    lengths: np.ndarray,
    n_iter: int,
) -> Optional[Tuple[List[np.ndarray], np.ndarray, str]]:
    direct_batch = _equal_length_matrices(observations, lengths)
    if direct_batch is None:
        return None

    direct_matrices, _ = direct_batch
    return _prepare_matrices_batch(direct_matrices, n_iter)


def _prepare_matrices_batch(
    direct_matrices: List[np.ndarray],
    n_iter: int,
) -> Tuple[List[np.ndarray], np.ndarray, str]:
    n_sequences = direct_matrices[0].shape[0]
    direct_counts = np.ones(n_sequences, dtype=float)
    duplicate_fraction = _sample_duplicate_fraction(direct_matrices)
    if _should_compress_equal_length_batch(len(direct_counts), duplicate_fraction, n_iter):
        compressed_matrices, compressed_counts = _compress_equal_length_matrices(direct_matrices)
        return compressed_matrices, compressed_counts, "compressed"

    return direct_matrices, direct_counts, "direct"


def _prepare_length_grouped_batches(
    observations: List[np.ndarray],
    lengths: np.ndarray,
    n_iter: int,
) -> Optional[List[Tuple[List[np.ndarray], np.ndarray, str]]]:
    if len(lengths) == 0:
        return None

    spans_by_length = {}
    start = 0
    for seq_length in lengths:
        end = start + int(seq_length)
        spans_by_length.setdefault(int(seq_length), []).append((start, end))
        start = end

    grouped_batches = []
    for seq_length, spans in spans_by_length.items():
        matrices = []
        for obs in observations:
            matrix = np.empty((len(spans), seq_length), dtype=np.int32)
            for row_idx, (start, end) in enumerate(spans):
                matrix[row_idx] = obs[start:end]
            matrices.append(matrix)
        grouped_batches.append(_prepare_matrices_batch(matrices, n_iter))

    return grouped_batches


def _auto_equal_length_batch_size(
    n_sequences: int,
    seq_length: int,
    n_states: int,
    target_elements: int = 100_000,
    max_batch_size: int = 5000,
    min_batch_size: int = 64,
) -> int:
    if n_sequences <= 0:
        return 1

    state_time_cells = max(1, int(seq_length) * int(n_states))
    batch_size = target_elements // state_time_cells
    batch_size = max(min_batch_size, batch_size)
    batch_size = min(max_batch_size, n_sequences, batch_size)
    return max(1, int(batch_size))


def _emission_matrix(
    emission_probs: List[np.ndarray],
    obs_list: List[np.ndarray],
    n_states: int,
    floor: bool = True,
) -> np.ndarray:
    emission = np.ones((len(obs_list[0]), n_states), dtype=float)
    for ch_idx, obs in enumerate(obs_list):
        emission *= emission_probs[ch_idx][:, obs].T
    if floor:
        return np.maximum(emission, _EPS)
    return emission


def _normalize_probability_rows(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_states = values.shape[1]
    totals = values.sum(axis=1)
    valid = np.isfinite(totals) & (totals > 0.0)

    normalized = np.empty_like(values, dtype=float)
    normalized[valid] = values[valid] / totals[valid, np.newaxis]
    normalized[~valid] = 1.0 / n_states

    scales = totals.astype(float, copy=True)
    scales[~valid] = _EPS
    return normalized, scales


def _scaled_forward_log_likelihood(
    initial: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
) -> float:
    alpha, scale = _normalize_probability_rows(
        (initial * emission[0])[np.newaxis, :]
    )
    alpha_row = alpha[0]
    log_likelihood = float(np.log(scale[0]))

    for t in range(1, emission.shape[0]):
        alpha_t = (alpha_row @ transition) * emission[t]
        alpha, scale = _normalize_probability_rows(alpha_t[np.newaxis, :])
        alpha_row = alpha[0]
        log_likelihood += float(np.log(scale[0]))

    return log_likelihood


def _scaled_forward_backward(
    initial: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    seq_length, n_states = emission.shape
    alpha = np.empty((seq_length, n_states), dtype=float)
    scales = np.empty(seq_length, dtype=float)

    alpha[0] = initial * emission[0]
    scales[0] = max(float(alpha[0].sum()), _EPS)
    alpha[0] /= scales[0]

    for t in range(1, seq_length):
        alpha[t] = (alpha[t - 1] @ transition) * emission[t]
        scales[t] = max(float(alpha[t].sum()), _EPS)
        alpha[t] /= scales[t]

    beta = np.ones((seq_length, n_states), dtype=float)
    for t in range(seq_length - 2, -1, -1):
        beta[t] = transition @ (emission[t + 1] * beta[t + 1])
        beta[t] /= scales[t + 1]

    gamma = alpha * beta
    gamma_sums = gamma.sum(axis=1, keepdims=True)
    gamma_sums[gamma_sums <= 0.0] = 1.0
    gamma /= gamma_sums

    log_likelihood = float(np.sum(np.log(scales)))
    return alpha, gamma, beta, log_likelihood


def _accumulate_sequence_statistics(
    obs_list: List[np.ndarray],
    count: int,
    initial: np.ndarray,
    transition: np.ndarray,
    emission_probs: List[np.ndarray],
    gamma_0: np.ndarray,
    xi_sum: np.ndarray,
    gamma_sum_trans: np.ndarray,
    emission_counts: List[np.ndarray],
    emission_denominators: List[np.ndarray],
) -> float:
    n_states = len(initial)
    emission = _emission_matrix(emission_probs, obs_list, n_states)
    alpha, gamma, beta, log_likelihood = _scaled_forward_backward(initial, transition, emission)

    weight = float(count)
    gamma_0 += weight * gamma[0]

    if len(obs_list[0]) > 1:
        for t in range(len(obs_list[0]) - 1):
            xi = (
                alpha[t][:, np.newaxis]
                * transition
                * (emission[t + 1] * beta[t + 1])[np.newaxis, :]
            )
            xi_total = float(xi.sum())
            if xi_total > 0.0 and np.isfinite(xi_total):
                xi_sum += weight * (xi / xi_total)
            gamma_sum_trans += weight * gamma[t]

    gamma_totals = gamma.sum(axis=0)
    for ch_idx, obs in enumerate(obs_list):
        for state_idx in range(n_states):
            emission_counts[ch_idx][state_idx] += weight * np.bincount(
                obs,
                weights=gamma[:, state_idx],
                minlength=emission_counts[ch_idx].shape[1],
            )
        emission_denominators[ch_idx] += weight * gamma_totals

    return weight * log_likelihood


def _batch_emission_tensor(
    emission_probs: List[np.ndarray],
    obs_matrices: List[np.ndarray],
    floor: bool = True,
) -> np.ndarray:
    n_sequences, seq_length = obs_matrices[0].shape
    n_states = emission_probs[0].shape[0]
    emission = np.ones((n_sequences, seq_length, n_states), dtype=float)

    for ch_idx, obs in enumerate(obs_matrices):
        emission *= emission_probs[ch_idx][:, obs].transpose(1, 2, 0)

    if floor:
        return np.maximum(emission, _EPS)
    return emission


def _scaled_forward_log_likelihood_batch(
    initial: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
) -> np.ndarray:
    n_sequences, seq_length, _ = emission.shape
    alpha, scales = _normalize_probability_rows(
        initial[np.newaxis, :] * emission[:, 0, :]
    )
    log_likelihoods = np.log(scales)

    for t in range(1, seq_length):
        alpha_t = (alpha @ transition) * emission[:, t, :]
        alpha, scales = _normalize_probability_rows(alpha_t)
        log_likelihoods += np.log(scales)

    return log_likelihoods


def _scaled_forward_backward_batch(
    initial: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_sequences, seq_length, n_states = emission.shape
    alpha = np.empty((n_sequences, seq_length, n_states), dtype=float)
    scales = np.empty((n_sequences, seq_length), dtype=float)

    alpha[:, 0, :] = initial[np.newaxis, :] * emission[:, 0, :]
    scales[:, 0] = np.maximum(alpha[:, 0, :].sum(axis=1), _EPS)
    alpha[:, 0, :] /= scales[:, 0, np.newaxis]

    for t in range(1, seq_length):
        alpha[:, t, :] = (alpha[:, t - 1, :] @ transition) * emission[:, t, :]
        scales[:, t] = np.maximum(alpha[:, t, :].sum(axis=1), _EPS)
        alpha[:, t, :] /= scales[:, t, np.newaxis]

    beta = np.ones((n_sequences, seq_length, n_states), dtype=float)
    for t in range(seq_length - 2, -1, -1):
        next_term = emission[:, t + 1, :] * beta[:, t + 1, :]
        beta[:, t, :] = next_term @ transition.T
        beta[:, t, :] /= scales[:, t + 1, np.newaxis]

    gamma = alpha * beta
    gamma_sums = gamma.sum(axis=2, keepdims=True)
    gamma_sums[gamma_sums <= 0.0] = 1.0
    gamma /= gamma_sums

    log_likelihoods = np.sum(np.log(scales), axis=1)
    return alpha, gamma, beta, log_likelihoods


def _accumulate_equal_length_batch_statistics(
    obs_matrices: List[np.ndarray],
    counts: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    emission_probs: List[np.ndarray],
    gamma_0: np.ndarray,
    xi_sum: np.ndarray,
    gamma_sum_trans: np.ndarray,
    emission_counts: List[np.ndarray],
    emission_denominators: List[np.ndarray],
    batch_size: Optional[int] = None,
) -> float:
    n_unique = len(counts)
    total_log_likelihood = 0.0
    if batch_size is None:
        batch_size = _auto_equal_length_batch_size(
            n_unique,
            obs_matrices[0].shape[1],
            initial.shape[0],
        )

    for batch_start in range(0, n_unique, batch_size):
        batch_end = min(batch_start + batch_size, n_unique)
        batch_counts = counts[batch_start:batch_end]
        batch_obs = [
            matrix[batch_start:batch_end]
            for matrix in obs_matrices
        ]

        emission = _batch_emission_tensor(emission_probs, batch_obs)
        alpha, gamma, beta, log_likelihoods = _scaled_forward_backward_batch(
            initial,
            transition,
            emission,
        )

        weighted_gamma = gamma * batch_counts[:, np.newaxis, np.newaxis]
        gamma_0 += weighted_gamma[:, 0, :].sum(axis=0)
        gamma_sum_trans += weighted_gamma[:, :-1, :].sum(axis=(0, 1))

        for t in range(emission.shape[1] - 1):
            next_term = emission[:, t + 1, :] * beta[:, t + 1, :]
            xi = (
                alpha[:, t, :, np.newaxis]
                * transition[np.newaxis, :, :]
                * next_term[:, np.newaxis, :]
            )
            xi_denominator = xi.sum(axis=(1, 2))
            xi_denominator[xi_denominator <= 0.0] = 1.0
            xi /= xi_denominator[:, np.newaxis, np.newaxis]
            xi_sum += np.einsum("n,nij->ij", batch_counts, xi)

        for ch_idx, obs in enumerate(batch_obs):
            flat_obs = obs.ravel()
            for state_idx in range(initial.shape[0]):
                emission_counts[ch_idx][state_idx] += np.bincount(
                    flat_obs,
                    weights=weighted_gamma[:, :, state_idx].ravel(),
                    minlength=emission_counts[ch_idx].shape[1],
                )
            emission_denominators[ch_idx] += weighted_gamma.sum(axis=(0, 1))

        total_log_likelihood += float(np.dot(batch_counts, log_likelihoods))

    return total_log_likelihood


def _score_equal_length_batch(
    obs_matrices: List[np.ndarray],
    counts: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    emission_probs: List[np.ndarray],
    batch_size: Optional[int] = None,
) -> float:
    n_unique = len(counts)
    total_log_likelihood = 0.0
    if batch_size is None:
        batch_size = _auto_equal_length_batch_size(
            n_unique,
            obs_matrices[0].shape[1],
            initial.shape[0],
        )

    for batch_start in range(0, n_unique, batch_size):
        batch_end = min(batch_start + batch_size, n_unique)
        batch_counts = counts[batch_start:batch_end]
        batch_obs = [
            matrix[batch_start:batch_end]
            for matrix in obs_matrices
        ]
        emission = _batch_emission_tensor(emission_probs, batch_obs, floor=False)
        log_likelihoods = _scaled_forward_log_likelihood_batch(
            initial,
            transition,
            emission,
        )
        total_log_likelihood += float(np.dot(batch_counts, log_likelihoods))

    return total_log_likelihood


def _score_sequence_counts(
    sequence_counts: List[Tuple[List[np.ndarray], int]],
    initial: np.ndarray,
    transition: np.ndarray,
    emission_probs: List[np.ndarray],
) -> float:
    n_states = len(initial)
    total_log_likelihood = 0.0

    for obs_list, count in sequence_counts:
        emission = _emission_matrix(emission_probs, obs_list, n_states, floor=False)
        log_likelihood = _scaled_forward_log_likelihood(initial, transition, emission)
        total_log_likelihood += float(count) * log_likelihood

    return total_log_likelihood


def fit_multichannel_hmm(
    model: HMM,
    n_iter: int = 100,
    tol: float = 1e-2,
    verbose: bool = False
) -> HMM:
    """
    Fit a multichannel HMM using an optimized EM algorithm.

    The implementation uses scaled forward-backward recursions, vectorized
    transition/posterior accumulation, and repeated-pattern compression. This
    keeps the same multichannel independence assumption as seqHMM while avoiding
    the previous deeply nested Python loops over every state pair.

    Args:
        model: HMM model object with multichannel data
        n_iter: Maximum number of EM iterations
        tol: Convergence tolerance
        verbose: Whether to print progress

    Returns:
        HMM: Fitted model
    """
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")

    n_states = model.n_states
    _initialize_multichannel_parameters(model)

    observations, lengths = _multichannel_arrays(model)
    equal_length_batch = _prepare_equal_length_batch(observations, lengths, n_iter)
    length_grouped_batches = None
    sequence_counts = None
    if equal_length_batch is None:
        length_grouped_batches = _prepare_length_grouped_batches(observations, lengths, n_iter)
        if length_grouped_batches is None:
            sequence_counts = _unique_sequence_counts(observations, lengths)

    prev_log_likelihood = -np.inf
    current_log_likelihood = -np.inf
    model.converged = False

    for iteration in range(n_iter):
        initial = np.asarray(model.initial_probs, dtype=float)
        transition = np.asarray(model.transition_probs, dtype=float)
        emission_probs = [
            np.asarray(emission_ch, dtype=float)
            for emission_ch in model.emission_probs
        ]

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

        current_log_likelihood = 0.0
        if equal_length_batch is not None:
            obs_matrices, counts, _batch_mode = equal_length_batch
            current_log_likelihood += _accumulate_equal_length_batch_statistics(
                obs_matrices,
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
        elif length_grouped_batches is not None:
            for obs_matrices, counts, _batch_mode in length_grouped_batches:
                current_log_likelihood += _accumulate_equal_length_batch_statistics(
                    obs_matrices,
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
        else:
            for obs_list, count in sequence_counts:
                current_log_likelihood += _accumulate_sequence_statistics(
                    obs_list,
                    count,
                    initial,
                    transition,
                    emission_probs,
                    gamma_0,
                    xi_sum,
                    gamma_sum_trans,
                    emission_counts,
                    emission_denominators,
                )

        model.initial_probs = _normalize_vector(gamma_0, n_states)

        new_transition = np.empty_like(xi_sum)
        for state_idx in range(n_states):
            if gamma_sum_trans[state_idx] > 0.0:
                new_transition[state_idx] = xi_sum[state_idx] / gamma_sum_trans[state_idx]
            else:
                new_transition[state_idx] = 1.0 / n_states
        model.transition_probs = _normalize_rows(new_transition)

        for ch_idx, emission_ch in enumerate(emission_counts):
            new_emission = np.empty_like(emission_ch)
            for state_idx in range(n_states):
                denom = emission_denominators[ch_idx][state_idx]
                if denom > 0.0:
                    new_emission[state_idx] = emission_ch[state_idx] / denom
                else:
                    new_emission[state_idx] = 1.0 / emission_ch.shape[1]
            model.emission_probs[ch_idx] = _normalize_rows(new_emission)

        if iteration > 0:
            change = current_log_likelihood - prev_log_likelihood
            if abs(change) < tol:
                model.converged = True
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

        prev_log_likelihood = current_log_likelihood

        if verbose and (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: log-likelihood = {current_log_likelihood:.4f}")

    final_initial = np.asarray(model.initial_probs, dtype=float)
    final_transition = np.asarray(model.transition_probs, dtype=float)
    final_emission_probs = [
        np.asarray(emission_ch, dtype=float)
        for emission_ch in model.emission_probs
    ]
    if equal_length_batch is not None:
        obs_matrices, counts, _batch_mode = equal_length_batch
        model.log_likelihood = _score_equal_length_batch(
            obs_matrices,
            counts,
            final_initial,
            final_transition,
            final_emission_probs,
        )
    elif length_grouped_batches is not None:
        model.log_likelihood = 0.0
        for obs_matrices, counts, _batch_mode in length_grouped_batches:
            model.log_likelihood += _score_equal_length_batch(
                obs_matrices,
                counts,
                final_initial,
                final_transition,
                final_emission_probs,
            )
    else:
        model.log_likelihood = _score_sequence_counts(
            sequence_counts,
            final_initial,
            final_transition,
            final_emission_probs,
        )
    model.n_iter = iteration + 1

    if not model.converged and verbose:
        print(f"Did not converge after {n_iter} iterations")

    return model
