"""
@Author  : Yapeng Wei 卫亚鹏
@File    : mnhmm.py
@Time    : 2026-05-25 02:04
@Desc    : Core Mixture Non-homogeneous Hidden Markov Model (MNHMM) implementation for Sequenzo

Mixture non-homogeneous hidden Markov models for Sequenzo.

This module provides fixed and covariate-driven MNHMM inference, posterior
cluster responsibilities, multichannel scoring and fitting paths, and direct
likelihood optimization for covariate models.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from sequenzo.define_sequence_data import SequenceData

from .multichannel_utils import multichannel_to_hmmlearn_format, prepare_multichannel_data
from .nhmm_utils import (
    compute_emission_probs_with_covariates,
    compute_initial_probs_with_covariates,
    compute_transition_probs_with_covariates,
    softmax,
)
from .utils import sequence_data_to_hmmlearn_format


_FLOAT_TINY = np.finfo(float).tiny
EmissionProbabilities = Union[Sequence[np.ndarray], Sequence[Sequence[np.ndarray]]]
StateNames = Union[Sequence[str], Sequence[Sequence[str]]]


def _as_state_list(n_states: Union[int, Sequence[int]], n_clusters: int) -> List[int]:
    if isinstance(n_states, int):
        state_list = [int(n_states)] * n_clusters
    else:
        state_list = [int(value) for value in n_states]
    if len(state_list) != n_clusters:
        raise ValueError(
            f"n_states length ({len(state_list)}) must equal n_clusters ({n_clusters})"
        )
    if any(value < 1 for value in state_list):
        raise ValueError("n_states values must be positive integers")
    return state_list


def _givens_rotation(a: float, b: float) -> tuple[float, float]:
    if abs(b) >= abs(a):
        t = -a / b
        s = 1.0 / np.sqrt(1.0 + t * t)
        c = s * t
    else:
        t = -b / a
        c = 1.0 / np.sqrt(1.0 + t * t)
        s = c * t
    return c, s


def seqhmm_contrast_matrix(n_categories: int) -> np.ndarray:
    """
    Return seqHMM's orthonormal sum-to-zero contrast matrix.

    R seqHMM stores multinomial logits in a reduced ``n_categories - 1``
    parameterization and expands them with ``create_Q(n) %*% eta`` before the
    softmax step. This is a direct Python port of seqHMM's C++ ``create_Q``.
    """
    n_categories = int(n_categories)
    if n_categories < 2:
        raise ValueError("n_categories must be at least 2")

    n_reduced = n_categories - 1
    rotations = np.empty((2, n_reduced), dtype=float)
    u = -np.ones(n_reduced, dtype=float)
    r = np.eye(n_reduced, dtype=float)
    for j in range(n_reduced):
        c, s = _givens_rotation(r[j, j], u[j])
        rotations[:, j] = (c, s)
        r[j, j] = c * r[j, j] - s * u[j]
        if j < n_reduced - 1:
            t1 = r[j + 1 :, j].copy()
            t2 = u[j + 1 :].copy()
            r[j + 1 :, j] = c * t1 - s * t2
            u[j + 1 :] = s * t1 + c * t2

    q = np.eye(n_categories, dtype=float)
    for j in range(n_reduced):
        c, s = rotations[:, j]
        t1 = q[:, j].copy()
        t2 = q[:, n_categories - 1].copy()
        q[:, j] = c * t1 - s * t2
        q[:, n_categories - 1] = s * t1 + c * t2
    return q[:, :n_reduced]


def _check_full_and_reduced_exclusive(full, reduced, name: str) -> None:
    if full is not None and reduced is not None:
        raise ValueError(f"Specify either {name} or {name}_reduced, not both")


def _expand_eta_pi_reduced_one(value: np.ndarray, n_states: int, n_covariates: int) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    expected = (n_states - 1, n_covariates)
    if value.shape != expected:
        raise ValueError(f"eta_pi_reduced item shape {value.shape} must be {expected}")
    return (seqhmm_contrast_matrix(n_states) @ value).T


def _expand_eta_A_reduced_one(value: np.ndarray, n_states: int, n_covariates: int) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    expected = (n_states - 1, n_covariates, n_states)
    if value.shape != expected:
        raise ValueError(f"eta_A_reduced item shape {value.shape} must be {expected}")
    q = seqhmm_contrast_matrix(n_states)
    out = np.zeros((n_covariates, n_states, n_states), dtype=float)
    for origin in range(n_states):
        out[:, origin, :] = (q @ value[:, :, origin]).T
    return out


def _expand_eta_B_reduced_one(
    value: np.ndarray,
    n_symbols: int,
    n_states: int,
    n_covariates: int,
) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    expected = (n_symbols - 1, n_covariates, n_states)
    if value.shape != expected:
        raise ValueError(f"eta_B_reduced item shape {value.shape} must be {expected}")
    q = seqhmm_contrast_matrix(n_symbols)
    out = np.zeros((n_covariates, n_states, n_symbols), dtype=float)
    for state in range(n_states):
        out[:, state, :] = (q @ value[:, :, state]).T
    return out


def _expand_multichannel_eta_B_reduced(
    values: Optional[Sequence[Sequence[np.ndarray]]],
    n_clusters: int,
    n_channels: int,
    n_symbols: Sequence[int],
    n_states: Sequence[int],
    n_covariates: int,
) -> Optional[List[List[np.ndarray]]]:
    if values is None:
        return None
    if len(values) != n_clusters:
        raise ValueError(
            f"eta_B_reduced length ({len(values)}) must equal n_clusters ({n_clusters})"
        )
    out: List[List[np.ndarray]] = []
    for cluster_idx, cluster_values in enumerate(values):
        if not isinstance(cluster_values, (list, tuple)):
            raise ValueError(
                "multichannel eta_B_reduced must be a list per cluster, "
                "each containing one reduced eta array per channel"
            )
        if len(cluster_values) != n_channels:
            raise ValueError(
                f"eta_B_reduced[{cluster_idx}] length ({len(cluster_values)}) "
                f"must equal n_channels ({n_channels})"
            )
        cluster_out = []
        for channel_idx, value in enumerate(cluster_values):
            expanded = _expand_eta_B_reduced_one(
                value,
                n_symbols[channel_idx],
                n_states[cluster_idx],
                n_covariates,
            )
            if not np.isfinite(expanded).all():
                raise ValueError("eta_B_reduced must contain finite values")
            cluster_out.append(expanded)
        out.append(cluster_out)
    return out


def _expand_eta_omega_reduced_one(
    value: np.ndarray,
    n_clusters: int,
    n_covariates: int,
) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    expected = (n_clusters - 1, n_covariates)
    if value.shape != expected:
        raise ValueError(f"eta_omega_reduced shape {value.shape} must be {expected}")
    return (seqhmm_contrast_matrix(n_clusters) @ value).T


def _reduce_eta_pi_full(value: np.ndarray) -> np.ndarray:
    return seqhmm_contrast_matrix(value.shape[1]).T @ value.T


def _reduce_eta_A_full(value: np.ndarray) -> np.ndarray:
    n_covariates, n_states, _ = value.shape
    q_t = seqhmm_contrast_matrix(n_states).T
    out = np.zeros((n_states - 1, n_covariates, n_states), dtype=float)
    for origin in range(n_states):
        out[:, :, origin] = q_t @ value[:, origin, :].T
    return out


def _reduce_eta_B_full(value: np.ndarray) -> np.ndarray:
    n_covariates, n_states, n_symbols = value.shape
    q_t = seqhmm_contrast_matrix(n_symbols).T
    out = np.zeros((n_symbols - 1, n_covariates, n_states), dtype=float)
    for state in range(n_states):
        out[:, :, state] = q_t @ value[:, state, :].T
    return out


def _reduce_eta_omega_full(value: np.ndarray) -> np.ndarray:
    return seqhmm_contrast_matrix(value.shape[1]).T @ value.T


def _expand_reduced_eta_list(
    values: Optional[Sequence[np.ndarray]],
    n_clusters: int,
    expand_one,
    name: str,
) -> Optional[List[np.ndarray]]:
    if values is None:
        return None
    if len(values) != n_clusters:
        raise ValueError(f"{name} length ({len(values)}) must equal n_clusters ({n_clusters})")
    out = []
    for cluster_idx, value in enumerate(values):
        expanded = expand_one(value, cluster_idx)
        if not np.isfinite(expanded).all():
            raise ValueError(f"{name} must contain finite values")
        out.append(expanded)
    return out


def _coerce_covariates(
    X: Optional[np.ndarray],
    n_sequences: int,
    n_timepoints: int,
    name: str,
) -> np.ndarray:
    if X is None:
        return np.ones((n_sequences, n_timepoints, 1), dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"{name} must have shape (n_sequences, n_timepoints, n_covariates)")
    if X.shape[0] != n_sequences:
        raise ValueError(f"{name} first dimension must equal n_sequences ({n_sequences})")
    if X.shape[1] < n_timepoints:
        raise ValueError(f"{name} must contain at least {n_timepoints} time points")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} must contain finite values")
    return X


def _coerce_cluster_covariates(
    X: Optional[np.ndarray],
    n_sequences: int,
) -> np.ndarray:
    if X is None:
        return np.ones((n_sequences, 1), dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 3:
        if not np.allclose(X, X[:, :1, :]):
            raise ValueError(
                "X_cluster must be time-constant for MNHMM mixture weights; "
                "pass a 2D matrix or repeat identical values at every time point"
            )
        X = X[:, 0, :]
    if X.ndim != 2:
        raise ValueError("X_cluster must have shape (n_sequences, n_covariates)")
    if X.shape[0] != n_sequences:
        raise ValueError(f"X_cluster first dimension must equal n_sequences ({n_sequences})")
    if not np.isfinite(X).all():
        raise ValueError("X_cluster must contain finite values")
    return X


def _is_intercept_only_tensor(X: np.ndarray) -> bool:
    return X.shape[2] == 1 and np.allclose(X[:, :, 0], 1.0)


def _is_intercept_only_matrix(X: np.ndarray) -> bool:
    return X.shape[1] == 1 and np.allclose(X[:, 0], 1.0)


def _validate_probability_vector(values: np.ndarray, expected: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (expected,):
        raise ValueError(f"{name} shape {values.shape} must be ({expected},)")
    if np.any(values < 0.0) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total probability")
    if not np.isclose(total, 1.0):
        raise ValueError(f"{name} must sum to 1.0")
    return np.array(values, copy=True)


def _log_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    logged = np.full(values.shape, -np.inf, dtype=float)
    positive = values > 0.0
    logged[positive] = np.log(values[positive])
    return logged


def _cluster_responsibilities_from_log_joint(
    log_joint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    log_norm = logsumexp(log_joint, axis=1, keepdims=True)
    if not np.isfinite(log_norm).all():
        raise ValueError("cluster responsibilities are undefined for impossible sequences")
    return np.exp(log_joint - log_norm), log_norm


def _probabilities_to_centered_logits(values: np.ndarray) -> np.ndarray:
    """
    Convert probabilities to seqHMM-compatible centered softmax logits.

    R seqHMM's probability initial values are clipped, normalized, logged, and
    centered before entering the reduced sum-to-zero eta parameterization. This
    helper returns the equivalent full centered logits used internally here.
    """
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    values = values / values.sum()
    logits = np.log(values)
    return logits - logits.mean()


def _validate_probability_matrix(
    values: np.ndarray,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != expected_shape:
        raise ValueError(f"{name} shape {values.shape} must be {expected_shape}")
    if np.any(values < 0.0) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"{name} rows must have positive total probability")
    if not np.allclose(row_sums, 1.0):
        raise ValueError(f"{name} rows must sum to 1.0")
    return np.array(values, copy=True)


def _validate_probability_list(
    values: Optional[Sequence[np.ndarray]],
    n_clusters: int,
    expected_shapes: Sequence[tuple[int, ...]],
    name: str,
) -> Optional[List[np.ndarray]]:
    if values is None:
        return None
    if len(values) != n_clusters:
        raise ValueError(f"{name} length ({len(values)}) must equal n_clusters ({n_clusters})")
    out = []
    for cluster_idx, value in enumerate(values):
        expected_shape = expected_shapes[cluster_idx]
        item_name = f"{name}[{cluster_idx}]"
        if len(expected_shape) == 1:
            out.append(_validate_probability_vector(value, expected_shape[0], item_name))
        else:
            out.append(_validate_probability_matrix(value, expected_shape, item_name))
    return out


def _validate_multichannel_emission_probability_list(
    values: Optional[Sequence[Sequence[np.ndarray]]],
    n_clusters: int,
    n_states: Sequence[int],
    n_symbols: Sequence[int],
) -> Optional[List[List[np.ndarray]]]:
    if values is None:
        return None
    if len(values) != n_clusters:
        raise ValueError(
            f"emission_probs length ({len(values)}) must equal n_clusters ({n_clusters})"
        )
    out: List[List[np.ndarray]] = []
    for cluster_idx, cluster_values in enumerate(values):
        if not isinstance(cluster_values, (list, tuple)):
            raise ValueError(
                "multichannel emission_probs must be a list per cluster, "
                "each containing one matrix per channel"
            )
        if len(cluster_values) != len(n_symbols):
            raise ValueError(
                f"emission_probs[{cluster_idx}] length ({len(cluster_values)}) "
                f"must equal n_channels ({len(n_symbols)})"
            )
        cluster_out = []
        for channel_idx, value in enumerate(cluster_values):
            cluster_out.append(
                _validate_probability_matrix(
                    value,
                    (n_states[cluster_idx], n_symbols[channel_idx]),
                    f"emission_probs[{cluster_idx}][{channel_idx}]",
                )
            )
        out.append(cluster_out)
    return out


def _validate_eta_list(
    values: Optional[Sequence[np.ndarray]],
    n_clusters: int,
    expected_shapes: Sequence[tuple[int, ...]],
    name: str,
) -> Optional[List[np.ndarray]]:
    if values is None:
        return None
    if len(values) != n_clusters:
        raise ValueError(f"{name} length ({len(values)}) must equal n_clusters ({n_clusters})")
    out = []
    for cluster_idx, value in enumerate(values):
        value = np.asarray(value, dtype=float)
        expected_shape = expected_shapes[cluster_idx]
        if value.shape != expected_shape:
            raise ValueError(f"{name}[{cluster_idx}] shape {value.shape} must be {expected_shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name}[{cluster_idx}] must contain finite values")
        out.append(value)
    return out


def _validate_multichannel_eta_list(
    values: Optional[Sequence[Sequence[np.ndarray]]],
    n_clusters: int,
    expected_shapes: Sequence[Sequence[tuple[int, ...]]],
    name: str,
) -> Optional[List[List[np.ndarray]]]:
    if values is None:
        return None
    if len(values) != n_clusters:
        raise ValueError(f"{name} length ({len(values)}) must equal n_clusters ({n_clusters})")
    out: List[List[np.ndarray]] = []
    for cluster_idx, cluster_values in enumerate(values):
        if not isinstance(cluster_values, (list, tuple)):
            raise ValueError(
                f"multichannel {name} must be a list per cluster, "
                "each containing one eta array per channel"
            )
        if len(cluster_values) != len(expected_shapes[cluster_idx]):
            raise ValueError(
                f"{name}[{cluster_idx}] length ({len(cluster_values)}) "
                f"must equal n_channels ({len(expected_shapes[cluster_idx])})"
            )
        cluster_out = []
        for channel_idx, value in enumerate(cluster_values):
            value = np.asarray(value, dtype=float)
            expected_shape = expected_shapes[cluster_idx][channel_idx]
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name}[{cluster_idx}][{channel_idx}] shape {value.shape} "
                    f"must be {expected_shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{name}[{cluster_idx}][{channel_idx}] must contain finite values")
            cluster_out.append(value)
        out.append(cluster_out)
    return out


def _ensure_no_missing_mnhmm_data(channels: Sequence[SequenceData]) -> None:
    for channel_idx, channel in enumerate(channels):
        if bool(getattr(channel, "ismissing", False)):
            raise ValueError(
                "missing values are not supported by MNHMM because preserving "
                f"time/covariate alignment is required; channel {channel_idx} "
                "contains missing values"
            )


def _ensure_mnhmm_alphabets_match(
    observed_alphabets: Sequence[Sequence[str]],
    expected_alphabets: Sequence[Sequence[str]],
) -> None:
    observed = [list(alphabet) for alphabet in observed_alphabets]
    expected = [list(alphabet) for alphabet in expected_alphabets]
    if observed != expected:
        raise ValueError(
            "new data alphabet must match the fitted MNHMM alphabet and state order"
        )


def _normalize_observations_input(
    observations: Union[SequenceData, Sequence[SequenceData]],
) -> Union[SequenceData, Sequence[SequenceData]]:
    return list(observations) if isinstance(observations, tuple) else observations


def _sequence_data_matches(reference: SequenceData, candidate: SequenceData) -> bool:
    return (
        np.array_equal(np.asarray(reference.ids), np.asarray(candidate.ids))
        and np.array_equal(np.asarray(reference.time), np.asarray(candidate.time))
    )


def _unique_rows_with_counts(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.ascontiguousarray(rows)
    row_view = rows.view(
        np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
    ).ravel()
    _, indices, counts = np.unique(row_view, return_index=True, return_counts=True)
    return indices, counts


def _unique_rows_with_counts_and_inverse(
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.ascontiguousarray(rows)
    row_view = rows.view(
        np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
    ).ravel()
    _, indices, inverse, counts = np.unique(
        row_view,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    return indices, inverse, counts


def _array_key(value: np.ndarray) -> tuple[tuple[int, ...], str, bytes]:
    arr = np.ascontiguousarray(value)
    return arr.shape, arr.dtype.str, arr.tobytes()


def _unique_equal_length_sequences_with_inverse(
    observations: Sequence[np.ndarray],
    lengths: np.ndarray,
) -> Optional[tuple[list[np.ndarray], np.ndarray, np.ndarray]]:
    lengths = np.asarray(lengths, dtype=int)
    if lengths.size == 0 or np.any(lengths != lengths[0]):
        return None

    n_sequences = int(lengths.size)
    sequence_length = int(lengths[0])
    matrices = [
        np.asarray(obs, dtype=int).reshape(n_sequences, sequence_length)
        for obs in observations
    ]
    combined = np.concatenate(matrices, axis=1)
    indices, inverse, counts = _unique_rows_with_counts_and_inverse(combined)
    unique = [matrix[indices].copy() for matrix in matrices]
    return unique, counts.astype(float, copy=False), inverse


def _unique_equal_length_sequences(
    observations: Sequence[np.ndarray],
    lengths: np.ndarray,
) -> Optional[tuple[list[np.ndarray], np.ndarray]]:
    unique = _unique_equal_length_sequences_with_inverse(observations, lengths)
    if unique is None:
        return None
    observations_unique, counts, _ = unique
    return observations_unique, counts


class MNHMM:
    """
    Mixture of NHMM-like components.

    The model computes sequence likelihoods under each component, combines them
    with fixed or covariate-driven cluster probabilities, and exposes posterior
    responsibilities. Intercept-only EM and direct covariate optimization are
    implemented by :meth:`fit`.
    """

    def __init__(
        self,
        observations: Union[SequenceData, Sequence[SequenceData]],
        n_clusters: int,
        n_states: Union[int, Sequence[int]],
        X: Optional[np.ndarray] = None,
        X_pi: Optional[np.ndarray] = None,
        X_A: Optional[np.ndarray] = None,
        X_B: Optional[np.ndarray] = None,
        X_cluster: Optional[np.ndarray] = None,
        eta_pi: Optional[Sequence[np.ndarray]] = None,
        eta_A: Optional[Sequence[np.ndarray]] = None,
        eta_B: Optional[Sequence[np.ndarray]] = None,
        eta_omega: Optional[np.ndarray] = None,
        eta_pi_reduced: Optional[Sequence[np.ndarray]] = None,
        eta_A_reduced: Optional[Sequence[np.ndarray]] = None,
        eta_B_reduced: Optional[Sequence[np.ndarray]] = None,
        eta_omega_reduced: Optional[np.ndarray] = None,
        initial_probs: Optional[Sequence[np.ndarray]] = None,
        transition_probs: Optional[Sequence[np.ndarray]] = None,
        emission_probs: Optional[EmissionProbabilities] = None,
        cluster_probs: Optional[np.ndarray] = None,
        cluster_names: Optional[Sequence[str]] = None,
        state_names: Optional[StateNames] = None,
        random_state: Optional[int] = None,
    ):
        if int(n_clusters) < 2:
            raise ValueError("n_clusters must be at least 2")

        channels, default_channel_names, alphabets = prepare_multichannel_data(
            _normalize_observations_input(observations)
        )
        _ensure_no_missing_mnhmm_data(channels)
        self.channels = channels
        self.n_channels = len(channels)
        self.channel_names = default_channel_names
        self.observations = channels[0] if self.n_channels == 1 else channels
        self.n_clusters = int(n_clusters)
        self.n_states = _as_state_list(n_states, self.n_clusters)
        self.alphabets = alphabets
        self.alphabet = alphabets[0]
        self.n_symbols = len(self.alphabet) if self.n_channels == 1 else [
            len(alphabet) for alphabet in alphabets
        ]
        self.n_sequences = len(channels[0].sequences)
        self.sequence_lengths = np.array([len(seq) for seq in channels[0].sequences], dtype=int)
        self.length_of_sequences = int(self.sequence_lengths.max())
        self.random_state = random_state

        self.cluster_names = (
            [str(name) for name in cluster_names]
            if cluster_names is not None
            else [f"Cluster {idx + 1}" for idx in range(self.n_clusters)]
        )
        if len(self.cluster_names) != self.n_clusters:
            raise ValueError("cluster_names length must equal n_clusters")

        if state_names is None:
            self.state_names = [
                [f"State {idx + 1}" for idx in range(n_states_k)]
                for n_states_k in self.n_states
            ]
        else:
            if isinstance(state_names, str):
                raise ValueError("state_names must be a sequence, not a string")
            if all(isinstance(name, str) for name in state_names):
                common_names = [str(name) for name in state_names]
                if any(n_states_k != len(common_names) for n_states_k in self.n_states):
                    raise ValueError(
                        "flat state_names length must equal each n_states value"
                    )
                self.state_names = [common_names.copy() for _ in range(self.n_clusters)]
            else:
                if len(state_names) != self.n_clusters:
                    raise ValueError("state_names length must equal n_clusters")
                self.state_names = [[str(name) for name in names] for names in state_names]
            for cluster_idx, names in enumerate(self.state_names):
                if len(names) != self.n_states[cluster_idx]:
                    raise ValueError(
                        f"state_names[{cluster_idx}] length must equal n_states[{cluster_idx}]"
                    )

        base_X = X
        self.X_pi = _coerce_covariates(
            X_pi if X_pi is not None else base_X,
            self.n_sequences,
            self.length_of_sequences,
            "X_pi",
        )
        self.X_A = _coerce_covariates(
            X_A if X_A is not None else base_X,
            self.n_sequences,
            self.length_of_sequences,
            "X_A",
        )
        self.X_B = _coerce_covariates(
            X_B if X_B is not None else base_X,
            self.n_sequences,
            self.length_of_sequences,
            "X_B",
        )
        self.X_cluster = _coerce_cluster_covariates(X_cluster, self.n_sequences)

        expected_pi = [(self.X_pi.shape[2], n_states_k) for n_states_k in self.n_states]
        expected_A = [
            (self.X_A.shape[2], n_states_k, n_states_k) for n_states_k in self.n_states
        ]
        if self.n_channels == 1:
            expected_B = [
                (self.X_B.shape[2], n_states_k, self.n_symbols)
                for n_states_k in self.n_states
            ]
        else:
            expected_B = [
                [
                    (self.X_B.shape[2], n_states_k, n_symbols_c)
                    for n_symbols_c in self.n_symbols
                ]
                for n_states_k in self.n_states
            ]

        _check_full_and_reduced_exclusive(eta_pi, eta_pi_reduced, "eta_pi")
        _check_full_and_reduced_exclusive(eta_A, eta_A_reduced, "eta_A")
        _check_full_and_reduced_exclusive(eta_B, eta_B_reduced, "eta_B")
        _check_full_and_reduced_exclusive(eta_omega, eta_omega_reduced, "eta_omega")

        self._eta_pi_user_supplied = eta_pi is not None or eta_pi_reduced is not None
        self._eta_A_user_supplied = eta_A is not None or eta_A_reduced is not None
        self._eta_B_user_supplied = eta_B is not None or eta_B_reduced is not None
        self._eta_omega_user_supplied = eta_omega is not None or eta_omega_reduced is not None

        if eta_pi_reduced is not None:
            eta_pi = _expand_reduced_eta_list(
                eta_pi_reduced,
                self.n_clusters,
                lambda value, cluster_idx: _expand_eta_pi_reduced_one(
                    value, self.n_states[cluster_idx], self.X_pi.shape[2]
                ),
                "eta_pi_reduced",
            )
        if eta_A_reduced is not None:
            eta_A = _expand_reduced_eta_list(
                eta_A_reduced,
                self.n_clusters,
                lambda value, cluster_idx: _expand_eta_A_reduced_one(
                    value, self.n_states[cluster_idx], self.X_A.shape[2]
                ),
                "eta_A_reduced",
            )
        if eta_B_reduced is not None and self.n_channels == 1:
            eta_B = _expand_reduced_eta_list(
                eta_B_reduced,
                self.n_clusters,
                lambda value, cluster_idx: _expand_eta_B_reduced_one(
                    value,
                    self.n_symbols,
                    self.n_states[cluster_idx],
                    self.X_B.shape[2],
                ),
                "eta_B_reduced",
            )
        elif eta_B_reduced is not None:
            eta_B = _expand_multichannel_eta_B_reduced(
                eta_B_reduced,
                self.n_clusters,
                self.n_channels,
                self.n_symbols,
                self.n_states,
                self.X_B.shape[2],
            )
        if eta_omega_reduced is not None:
            eta_omega = _expand_eta_omega_reduced_one(
                eta_omega_reduced, self.n_clusters, self.X_cluster.shape[1]
            )

        self.initial_probs = _validate_probability_list(
            initial_probs,
            self.n_clusters,
            [(n_states_k,) for n_states_k in self.n_states],
            "initial_probs",
        )
        self.transition_probs = _validate_probability_list(
            transition_probs,
            self.n_clusters,
            [(n_states_k, n_states_k) for n_states_k in self.n_states],
            "transition_probs",
        )
        if self.n_channels == 1:
            self.emission_probs = _validate_probability_list(
                emission_probs,
                self.n_clusters,
                [(n_states_k, self.n_symbols) for n_states_k in self.n_states],
                "emission_probs",
            )
        else:
            self.emission_probs = _validate_multichannel_emission_probability_list(
                emission_probs,
                self.n_clusters,
                self.n_states,
                self.n_symbols,
            )

        rng = np.random.default_rng(random_state)
        self.eta_pi = _validate_eta_list(eta_pi, self.n_clusters, expected_pi, "eta_pi")
        if self.eta_pi is None:
            self.eta_pi = [
                rng.normal(scale=0.1, size=shape) for shape in expected_pi
            ]
        self.eta_A = _validate_eta_list(eta_A, self.n_clusters, expected_A, "eta_A")
        if self.eta_A is None:
            self.eta_A = [
                rng.normal(scale=0.1, size=shape) for shape in expected_A
            ]
        if self.n_channels == 1:
            self.eta_B = _validate_eta_list(eta_B, self.n_clusters, expected_B, "eta_B")
            if self.eta_B is None:
                self.eta_B = [
                    rng.normal(scale=0.1, size=shape) for shape in expected_B
                ]
        else:
            self.eta_B = _validate_multichannel_eta_list(
                eta_B,
                self.n_clusters,
                expected_B,
                "eta_B",
            )
            if self.eta_B is None:
                self.eta_B = [
                    [
                        rng.normal(scale=0.1, size=channel_shape)
                        for channel_shape in cluster_shapes
                    ]
                    for cluster_shapes in expected_B
                ]

        self._cluster_probs_user_fixed = cluster_probs is not None

        if cluster_probs is not None and eta_omega is not None:
            raise ValueError("Specify either cluster_probs or eta_omega, not both")
        if cluster_probs is None:
            if eta_omega is None:
                self.cluster_probs = np.ones(self.n_clusters, dtype=float) / self.n_clusters
                self.eta_omega = None
            else:
                eta_omega = np.asarray(eta_omega, dtype=float)
                expected_shape = (self.X_cluster.shape[1], self.n_clusters)
                if eta_omega.shape != expected_shape:
                    raise ValueError(f"eta_omega shape {eta_omega.shape} must be {expected_shape}")
                if not np.isfinite(eta_omega).all():
                    raise ValueError("eta_omega must contain finite values")
                self.cluster_probs = None
                self.eta_omega = eta_omega
        else:
            self.cluster_probs = _validate_probability_vector(
                cluster_probs, self.n_clusters, "cluster_probs"
            )
            self.eta_omega = None

        self.log_likelihood = None
        self.n_iter = None
        self.converged = None
        self.responsibilities = None
        self.n_optimized_parameters = None

    @property
    def has_complete_parameters(self) -> bool:
        """Return True when the model can score sequences."""
        component_ready = (
            (self.initial_probs is not None or self.eta_pi is not None)
            and (self.transition_probs is not None or self.eta_A is not None)
            and (self.emission_probs is not None or self.eta_B is not None)
        )
        cluster_ready = self.cluster_probs is not None or self.eta_omega is not None
        return bool(component_ready and cluster_ready)

    def compute_cluster_probs(self) -> np.ndarray:
        """Return per-sequence prior cluster probabilities."""
        if self.cluster_probs is not None:
            return np.tile(self.cluster_probs, (self.n_sequences, 1))
        logits = self.X_cluster @ self.eta_omega
        return softmax(logits, axis=1)

    def _cluster_probs_for_indices(self, sequence_indices: np.ndarray) -> np.ndarray:
        if self.cluster_probs is not None:
            return np.tile(self.cluster_probs, (len(sequence_indices), 1))
        return softmax(self.X_cluster[sequence_indices] @ self.eta_omega, axis=1)

    def _cluster_probs_for_n(self, n_sequences: int) -> np.ndarray:
        if self.cluster_probs is not None:
            return np.tile(self.cluster_probs, (int(n_sequences), 1))
        cluster_probs = self.compute_cluster_probs()
        if cluster_probs.shape[0] != int(n_sequences):
            raise ValueError(
                "MNHMM scoring with new data currently requires the same number "
                "of sequences as the model cluster covariates"
            )
        return cluster_probs

    def _validate_observation_alphabets(
        self,
        channels: Sequence[SequenceData],
    ) -> None:
        _ensure_mnhmm_alphabets_match(
            [channel.alphabet for channel in channels],
            self.alphabets,
        )

    def _validate_covariate_newdata_alignment(
        self,
        channels: Sequence[SequenceData],
    ) -> None:
        reference_channels = self.channels
        if len(channels) != len(reference_channels) or any(
            not _sequence_data_matches(reference, candidate)
            for reference, candidate in zip(reference_channels, channels)
        ):
            raise ValueError(
                "covariate MNHMM new data must have the same sequence IDs "
                "and time order as the fitted model"
            )

    def _uses_covariate_probabilities(self) -> bool:
        return (
            self.initial_probs is None
            or self.transition_probs is None
            or self.emission_probs is None
            or self.eta_omega is not None
        )

    def has_cluster_covariates(self) -> bool:
        """Return True when mixture weights have non-intercept covariates."""
        return not _is_intercept_only_matrix(self.X_cluster)

    def enable_cluster_covariate_estimation(
        self,
        random_state: Optional[int] = None,
        scale: float = 0.05,
    ) -> None:
        """
        Switch mixture weights from fixed probabilities to eta coefficients.

        seqHMM estimates mixture logits even for an intercept-only
        ``cluster_formula``. This helper is used by ``estimate_mnhmm`` when a
        covariate model is requested and the user did not explicitly fix
        ``cluster_probs``.
        """
        if self.cluster_probs is None and self.eta_omega is not None:
            return
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        eta = np.zeros((self.X_cluster.shape[1], self.n_clusters), dtype=float)
        if self.cluster_probs is not None:
            eta[0, :] = np.log(np.maximum(self.cluster_probs, _FLOAT_TINY))
        eta += rng.normal(scale=scale, size=eta.shape)
        self.cluster_probs = None
        self.eta_omega = eta

    def use_probability_parameters_as_covariate_starts(
        self,
        include_cluster: bool = True,
    ) -> None:
        """
        Treat supplied probability arrays as covariate-model starting values.

        This mirrors R seqHMM's ``fit_mnhmm`` semantics: supplied probability
        arrays initialize eta coefficients instead of freezing those probability
        families during fitting. ``build_mnhmm`` keeps its fixed-probability
        interpretation; ``estimate_mnhmm`` calls this only when explicitly
        asked for R-style probability starts.
        """
        if self.initial_probs is not None:
            for cluster_idx, probs in enumerate(self.initial_probs):
                self.eta_pi[cluster_idx] = np.zeros_like(self.eta_pi[cluster_idx])
                self.eta_pi[cluster_idx][0, :] = _probabilities_to_centered_logits(probs)
            self.initial_probs = None
            self._eta_pi_user_supplied = True

        if self.transition_probs is not None:
            for cluster_idx, probs in enumerate(self.transition_probs):
                self.eta_A[cluster_idx] = np.zeros_like(self.eta_A[cluster_idx])
                for origin in range(self.n_states[cluster_idx]):
                    self.eta_A[cluster_idx][0, origin, :] = (
                        _probabilities_to_centered_logits(probs[origin])
                    )
            self.transition_probs = None
            self._eta_A_user_supplied = True

        if self.emission_probs is not None:
            if self.n_channels == 1:
                for cluster_idx, probs in enumerate(self.emission_probs):
                    self.eta_B[cluster_idx] = np.zeros_like(self.eta_B[cluster_idx])
                    for state in range(self.n_states[cluster_idx]):
                        self.eta_B[cluster_idx][0, state, :] = (
                            _probabilities_to_centered_logits(probs[state])
                        )
            else:
                for cluster_idx, cluster_probs in enumerate(self.emission_probs):
                    for channel_idx, probs in enumerate(cluster_probs):
                        self.eta_B[cluster_idx][channel_idx] = np.zeros_like(
                            self.eta_B[cluster_idx][channel_idx]
                        )
                        for state in range(self.n_states[cluster_idx]):
                            self.eta_B[cluster_idx][channel_idx][0, state, :] = (
                                _probabilities_to_centered_logits(probs[state])
                            )
            self.emission_probs = None
            self._eta_B_user_supplied = True

        if include_cluster and self.cluster_probs is not None:
            eta = np.zeros((self.X_cluster.shape[1], self.n_clusters), dtype=float)
            eta[0, :] = _probabilities_to_centered_logits(self.cluster_probs)
            self.cluster_probs = None
            self.eta_omega = eta
            self._cluster_probs_user_fixed = False
            self._eta_omega_user_supplied = True

    def _component_probs(
        self,
        cluster_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_states = self.n_states[cluster_idx]
        if self.initial_probs is not None:
            initial = np.tile(self.initial_probs[cluster_idx], (self.n_sequences, 1))
        else:
            initial = compute_initial_probs_with_covariates(
                self.eta_pi[cluster_idx], self.X_pi, n_states
            )

        if self.transition_probs is not None:
            transition = np.tile(
                self.transition_probs[cluster_idx],
                (self.n_sequences, self.length_of_sequences, 1, 1),
            )
        else:
            transition = compute_transition_probs_with_covariates(
                self.eta_A[cluster_idx], self.X_A, n_states
            )

        if self.emission_probs is not None:
            if self.n_channels == 1:
                emission = np.tile(
                    self.emission_probs[cluster_idx],
                    (self.n_sequences, self.length_of_sequences, 1, 1),
                )
            else:
                emission = [
                    np.tile(
                        channel_emission,
                        (self.n_sequences, self.length_of_sequences, 1, 1),
                    )
                    for channel_emission in self.emission_probs[cluster_idx]
                ]
        else:
            if self.n_channels == 1:
                emission = compute_emission_probs_with_covariates(
                    self.eta_B[cluster_idx], self.X_B, n_states, self.n_symbols
                )
            else:
                emission = [
                    compute_emission_probs_with_covariates(
                        channel_eta,
                        self.X_B,
                        n_states,
                        self.n_symbols[channel_idx],
                    )
                    for channel_idx, channel_eta in enumerate(self.eta_B[cluster_idx])
                ]
        return initial, transition, emission

    def _component_probs_for_indices(
        self,
        cluster_idx: int,
        sequence_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_states = self.n_states[cluster_idx]
        n_sequences = len(sequence_indices)
        max_length = (
            int(np.max(self.sequence_lengths[sequence_indices]))
            if n_sequences
            else self.length_of_sequences
        )
        if self.initial_probs is not None:
            initial = np.tile(self.initial_probs[cluster_idx], (n_sequences, 1))
        else:
            initial = compute_initial_probs_with_covariates(
                self.eta_pi[cluster_idx],
                self.X_pi[sequence_indices, :max_length],
                n_states,
            )

        if self.transition_probs is not None:
            transition = np.tile(
                self.transition_probs[cluster_idx],
                (n_sequences, max_length, 1, 1),
            )
        else:
            transition = compute_transition_probs_with_covariates(
                self.eta_A[cluster_idx],
                self.X_A[sequence_indices, :max_length],
                n_states,
            )

        if self.emission_probs is not None:
            if self.n_channels == 1:
                emission = np.tile(
                    self.emission_probs[cluster_idx],
                    (n_sequences, max_length, 1, 1),
                )
            else:
                emission = [
                    np.tile(
                        channel_emission,
                        (n_sequences, max_length, 1, 1),
                    )
                    for channel_emission in self.emission_probs[cluster_idx]
                ]
        else:
            if self.n_channels == 1:
                emission = compute_emission_probs_with_covariates(
                    self.eta_B[cluster_idx],
                    self.X_B[sequence_indices, :max_length],
                    n_states,
                    self.n_symbols,
                )
            else:
                emission = [
                    compute_emission_probs_with_covariates(
                        channel_eta,
                        self.X_B[sequence_indices, :max_length],
                        n_states,
                        self.n_symbols[channel_idx],
                    )
                    for channel_idx, channel_eta in enumerate(self.eta_B[cluster_idx])
                ]
        return initial, transition, emission

    @staticmethod
    def _forward_log_likelihood(
        initial_probs: np.ndarray,
        transition_probs: np.ndarray,
        emission_probs: np.ndarray,
        observations: np.ndarray,
    ) -> float:
        log_alpha = (
            _log_probabilities(initial_probs)
            + _log_probabilities(emission_probs[0, :, observations[0]])
        )
        log_transition = _log_probabilities(transition_probs)
        log_emission = _log_probabilities(emission_probs)

        for t in range(1, len(observations)):
            # seqHMM uses transition covariates at the destination time; A_1 is
            # constructed for reporting but is not used in the likelihood.
            log_alpha = (
                logsumexp(log_alpha[:, np.newaxis] + log_transition[t], axis=0)
                + log_emission[t, :, observations[t]]
            )
        return float(logsumexp(log_alpha))

    @staticmethod
    def _multichannel_forward_log_likelihood(
        initial_probs: np.ndarray,
        transition_probs: np.ndarray,
        emission_probs: Sequence[np.ndarray],
        observations: Sequence[np.ndarray],
    ) -> float:
        log_alpha = _log_probabilities(initial_probs)
        for obs, emission in zip(observations, emission_probs):
            log_alpha += _log_probabilities(emission[:, obs[0]])

        log_transition = _log_probabilities(transition_probs)
        log_emissions = [_log_probabilities(emission) for emission in emission_probs]
        for t in range(1, len(observations[0])):
            log_emission_t = np.zeros_like(log_alpha)
            for obs, log_emission in zip(observations, log_emissions):
                log_emission_t += log_emission[:, obs[t]]
            log_alpha = logsumexp(log_alpha[:, np.newaxis] + log_transition, axis=0)
            log_alpha += log_emission_t
        return float(logsumexp(log_alpha))

    @staticmethod
    def _multichannel_forward_log_likelihood_timevarying(
        initial_probs: np.ndarray,
        transition_probs: np.ndarray,
        emission_probs: Sequence[np.ndarray],
        observations: Sequence[np.ndarray],
    ) -> float:
        log_alpha = _log_probabilities(initial_probs)
        log_transition = _log_probabilities(transition_probs)
        log_emissions = [_log_probabilities(emission) for emission in emission_probs]

        for obs, log_emission in zip(observations, log_emissions):
            log_alpha += log_emission[0, :, obs[0]]

        for t in range(1, len(observations[0])):
            log_emission_t = np.zeros_like(log_alpha)
            for obs, log_emission in zip(observations, log_emissions):
                log_emission_t += log_emission[t, :, obs[t]]
            log_alpha = (
                logsumexp(log_alpha[:, np.newaxis] + log_transition[t], axis=0)
                + log_emission_t
            )
        return float(logsumexp(log_alpha))

    @staticmethod
    def _forward_backward_timevarying(
        initial_probs: np.ndarray,
        transition_probs: np.ndarray,
        emission_probs: np.ndarray,
        observations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        T = len(observations)
        S = len(initial_probs)
        log_initial = _log_probabilities(initial_probs)
        log_transition = _log_probabilities(transition_probs)
        log_emission = _log_probabilities(emission_probs)

        log_alpha = np.empty((T, S), dtype=float)
        log_alpha[0] = log_initial + log_emission[0, :, observations[0]]
        for t in range(1, T):
            log_alpha[t] = (
                logsumexp(log_alpha[t - 1, :, np.newaxis] + log_transition[t], axis=0)
                + log_emission[t, :, observations[t]]
            )

        log_beta = np.zeros((T, S), dtype=float)
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_transition[t + 1]
                + log_emission[t + 1, :, observations[t + 1]][np.newaxis, :]
                + log_beta[t + 1, np.newaxis, :],
                axis=1,
            )

        log_likelihood = float(logsumexp(log_alpha[-1]))
        xi = np.zeros((max(T - 1, 0), S, S), dtype=float)
        if not np.isfinite(log_likelihood):
            return np.zeros((T, S), dtype=float), xi, log_likelihood
        gamma = np.exp(log_alpha + log_beta - log_likelihood)
        for t in range(1, T):
            log_xi = (
                log_alpha[t - 1, :, np.newaxis]
                + log_transition[t]
                + log_emission[t, :, observations[t]][np.newaxis, :]
                + log_beta[t, np.newaxis, :]
                - log_likelihood
            )
            xi[t - 1] = np.exp(log_xi)
        return gamma, xi, log_likelihood

    @staticmethod
    def _multichannel_forward_backward_timevarying(
        initial_probs: np.ndarray,
        transition_probs: np.ndarray,
        emission_probs: Sequence[np.ndarray],
        observations: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        T = len(observations[0])
        S = len(initial_probs)
        log_initial = _log_probabilities(initial_probs)
        log_transition = _log_probabilities(transition_probs)
        log_emissions = [_log_probabilities(emission) for emission in emission_probs]

        log_observed = np.zeros((T, S), dtype=float)
        for obs, log_emission in zip(observations, log_emissions):
            for t, symbol in enumerate(obs):
                log_observed[t] += log_emission[t, :, symbol]

        log_alpha = np.empty((T, S), dtype=float)
        log_alpha[0] = log_initial + log_observed[0]
        for t in range(1, T):
            log_alpha[t] = (
                logsumexp(log_alpha[t - 1, :, np.newaxis] + log_transition[t], axis=0)
                + log_observed[t]
            )

        log_beta = np.zeros((T, S), dtype=float)
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_transition[t + 1]
                + log_observed[t + 1, np.newaxis, :]
                + log_beta[t + 1, np.newaxis, :],
                axis=1,
            )

        log_likelihood = float(logsumexp(log_alpha[-1]))
        xi = np.zeros((max(T - 1, 0), S, S), dtype=float)
        if not np.isfinite(log_likelihood):
            return np.zeros((T, S), dtype=float), xi, log_likelihood
        gamma = np.exp(log_alpha + log_beta - log_likelihood)
        for t in range(1, T):
            log_xi = (
                log_alpha[t - 1, :, np.newaxis]
                + log_transition[t]
                + log_observed[t, np.newaxis, :]
                + log_beta[t, np.newaxis, :]
                - log_likelihood
            )
            xi[t - 1] = np.exp(log_xi)
        return gamma, xi, log_likelihood

    def _sequence_log_likelihoods(self, sequences: Optional[SequenceData] = None) -> np.ndarray:
        if not self.has_complete_parameters:
            raise ValueError("MNHMM parameters are incomplete")
        if sequences is None:
            sequences = self.observations

        components_fixed = (
            self.initial_probs is not None
            and self.transition_probs is not None
            and self.emission_probs is not None
        )
        if self.n_channels > 1:
            channels, _, _ = prepare_multichannel_data(
                _normalize_observations_input(sequences)
            )
            _ensure_no_missing_mnhmm_data(channels)
            if len(channels) != self.n_channels:
                raise ValueError(f"Expected {self.n_channels} channels, got {len(channels)}")
            self._validate_observation_alphabets(channels)
            observations, lengths = multichannel_to_hmmlearn_format(channels)
            observations = [X[:, 0].astype(int, copy=False) for X in observations]
            starts = np.zeros(len(lengths) + 1, dtype=int)
            starts[1:] = np.cumsum(lengths)
            log_likelihoods = np.zeros((len(lengths), self.n_clusters), dtype=float)
            if self._uses_covariate_probabilities():
                self._validate_covariate_newdata_alignment(channels)
            if components_fixed:
                for cluster_idx in range(self.n_clusters):
                    for seq_idx, seq_length in enumerate(lengths):
                        start = starts[seq_idx]
                        end = starts[seq_idx + 1]
                        obs_list = [obs[start:end] for obs in observations]
                        log_likelihoods[seq_idx, cluster_idx] = (
                            self._multichannel_forward_log_likelihood(
                                self.initial_probs[cluster_idx],
                                self.transition_probs[cluster_idx],
                                self.emission_probs[cluster_idx],
                                obs_list,
                            )
                        )
                return log_likelihoods

            if len(lengths) != self.n_sequences:
                raise ValueError(
                    "MNHMM scoring with new data currently requires the same number "
                    "of sequences as the model covariates"
                )
            self._validate_covariate_newdata_alignment(channels)
            if np.any(lengths > self.length_of_sequences):
                raise ValueError("new sequences are longer than the model covariate arrays")

            for cluster_idx in range(self.n_clusters):
                initial, transition, emissions = self._component_probs(cluster_idx)
                for seq_idx, seq_length in enumerate(lengths):
                    start = starts[seq_idx]
                    end = starts[seq_idx + 1]
                    obs_list = [obs[start:end] for obs in observations]
                    log_likelihoods[seq_idx, cluster_idx] = (
                        self._multichannel_forward_log_likelihood_timevarying(
                            initial[seq_idx],
                            transition[seq_idx, :seq_length],
                            [emission[seq_idx, :seq_length] for emission in emissions],
                            obs_list,
                        )
                    )
            return log_likelihoods

        _ensure_no_missing_mnhmm_data([sequences])
        self._validate_observation_alphabets([sequences])
        X_int, lengths = sequence_data_to_hmmlearn_format(sequences)
        starts = np.zeros(len(lengths) + 1, dtype=int)
        starts[1:] = np.cumsum(lengths)
        log_likelihoods = np.zeros((len(lengths), self.n_clusters), dtype=float)
        if self._uses_covariate_probabilities():
            self._validate_covariate_newdata_alignment([sequences])

        if components_fixed:
            for cluster_idx in range(self.n_clusters):
                for seq_idx, seq_length in enumerate(lengths):
                    start = starts[seq_idx]
                    end = starts[seq_idx + 1]
                    obs = X_int[start:end, 0].astype(int, copy=False)
                    log_likelihoods[seq_idx, cluster_idx] = self._forward_backward_fixed(
                        self.initial_probs[cluster_idx],
                        self.transition_probs[cluster_idx],
                        self.emission_probs[cluster_idx],
                        obs,
                    )[2]
            return log_likelihoods

        if len(lengths) != self.n_sequences:
            raise ValueError(
                "MNHMM scoring with new data currently requires the same number "
                "of sequences as the model covariates"
            )
        self._validate_covariate_newdata_alignment([sequences])
        if np.any(lengths > self.length_of_sequences):
            raise ValueError("new sequences are longer than the model covariate arrays")

        for cluster_idx in range(self.n_clusters):
            initial, transition, emission = self._component_probs(cluster_idx)
            for seq_idx, seq_length in enumerate(lengths):
                start = starts[seq_idx]
                end = starts[seq_idx + 1]
                obs = X_int[start:end, 0].astype(int, copy=False)
                log_likelihoods[seq_idx, cluster_idx] = self._forward_log_likelihood(
                    initial[seq_idx],
                    transition[seq_idx, :seq_length],
                    emission[seq_idx, :seq_length],
                    obs,
                )

        return log_likelihoods

    def _supports_fixed_probability_em(self) -> bool:
        if self.initial_probs is None and self._eta_pi_user_supplied:
            return False
        if self.transition_probs is None and self._eta_A_user_supplied:
            return False
        if self.emission_probs is None and self._eta_B_user_supplied:
            return False
        covariates = (self.X_pi, self.X_A, self.X_B)
        for X in covariates:
            if not _is_intercept_only_tensor(X):
                return False
        if self.has_cluster_covariates() and not self._cluster_probs_user_fixed:
            return False
        if self.eta_omega is not None:
            return False
        return True

    def _mixture_log_likelihood(self, sequences: Optional[SequenceData] = None) -> float:
        log_likelihoods = self._sequence_log_likelihoods(sequences)
        cluster_probs = self._cluster_probs_for_n(log_likelihoods.shape[0])
        log_joint = log_likelihoods + _log_probabilities(cluster_probs)
        return float(np.sum(logsumexp(log_joint, axis=1)))

    def _fixed_probability_score_compressed(
        self,
        sequences: Optional[Union[SequenceData, Sequence[SequenceData]]] = None,
    ) -> float:
        if (
            self.initial_probs is None
            or self.transition_probs is None
            or self.emission_probs is None
            or self.cluster_probs is None
        ):
            return self._mixture_log_likelihood(sequences)

        if sequences is None:
            sequences = self.observations

        log_cluster_probs = _log_probabilities(self.cluster_probs)
        if self.n_channels > 1:
            channels, _, _ = prepare_multichannel_data(
                _normalize_observations_input(sequences)
            )
            _ensure_no_missing_mnhmm_data(channels)
            if len(channels) != self.n_channels:
                raise ValueError(f"Expected {self.n_channels} channels, got {len(channels)}")
            self._validate_observation_alphabets(channels)
            observations, lengths = multichannel_to_hmmlearn_format(channels)
            observations = [X[:, 0].astype(int, copy=False) for X in observations]
            unique = _unique_equal_length_sequences(observations, lengths)
            if unique is not None:
                unique_observations, counts = unique
                total = 0.0
                for row_idx, count in enumerate(counts):
                    obs_list = [obs[row_idx] for obs in unique_observations]
                    component_terms = np.empty(self.n_clusters, dtype=float)
                    for cluster_idx in range(self.n_clusters):
                        component_terms[cluster_idx] = self._multichannel_forward_log_likelihood(
                            self.initial_probs[cluster_idx],
                            self.transition_probs[cluster_idx],
                            self.emission_probs[cluster_idx],
                            obs_list,
                        )
                    total += float(count) * float(logsumexp(component_terms + log_cluster_probs))
                return total

            counts: dict[tuple, int] = {}
            starts = np.zeros(len(lengths) + 1, dtype=int)
            starts[1:] = np.cumsum(lengths)
            for seq_idx in range(len(lengths)):
                start = starts[seq_idx]
                end = starts[seq_idx + 1]
                key = tuple(tuple(obs[start:end].tolist()) for obs in observations)
                counts[key] = counts.get(key, 0) + 1

            total = 0.0
            for key, count in counts.items():
                obs_list = [np.asarray(channel_key, dtype=int) for channel_key in key]
                component_terms = np.empty(self.n_clusters, dtype=float)
                for cluster_idx in range(self.n_clusters):
                    component_terms[cluster_idx] = self._multichannel_forward_log_likelihood(
                        self.initial_probs[cluster_idx],
                        self.transition_probs[cluster_idx],
                        self.emission_probs[cluster_idx],
                        obs_list,
                    )
                total += count * float(logsumexp(component_terms + log_cluster_probs))
            return total

        _ensure_no_missing_mnhmm_data([sequences])
        self._validate_observation_alphabets([sequences])
        X_int, lengths = sequence_data_to_hmmlearn_format(sequences)
        observations = [X_int[:, 0].astype(int, copy=False)]
        unique = _unique_equal_length_sequences(observations, lengths)
        if unique is not None:
            unique_observations, counts = unique
            total = 0.0
            for row_idx, count in enumerate(counts):
                obs = unique_observations[0][row_idx]
                component_terms = np.empty(self.n_clusters, dtype=float)
                for cluster_idx in range(self.n_clusters):
                    component_terms[cluster_idx] = self._forward_backward_fixed(
                        self.initial_probs[cluster_idx],
                        self.transition_probs[cluster_idx],
                        self.emission_probs[cluster_idx],
                        obs,
                    )[2]
                total += float(count) * float(logsumexp(component_terms + log_cluster_probs))
            return total

        counts: dict[tuple, int] = {}
        starts = np.zeros(len(lengths) + 1, dtype=int)
        starts[1:] = np.cumsum(lengths)
        for seq_idx in range(len(lengths)):
            start = starts[seq_idx]
            end = starts[seq_idx + 1]
            key = tuple(X_int[start:end, 0].astype(int, copy=False).tolist())
            counts[key] = counts.get(key, 0) + 1

        total = 0.0
        for key, count in counts.items():
            obs = np.asarray(key, dtype=int)
            component_terms = np.empty(self.n_clusters, dtype=float)
            for cluster_idx in range(self.n_clusters):
                component_terms[cluster_idx] = self._forward_backward_fixed(
                    self.initial_probs[cluster_idx],
                    self.transition_probs[cluster_idx],
                    self.emission_probs[cluster_idx],
                    obs,
                )[2]
            total += count * float(logsumexp(component_terms + log_cluster_probs))
        return total

    def _compressed_fixed_fit_data(
        self,
        observations: Sequence[np.ndarray],
        lengths: np.ndarray,
        compress: bool,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        lengths = np.asarray(lengths, dtype=int)
        counts = np.ones(len(lengths), dtype=float)
        inverse = np.arange(len(lengths), dtype=int)
        if not compress:
            return [np.asarray(obs, dtype=int) for obs in observations], lengths, counts, inverse

        unique = _unique_equal_length_sequences_with_inverse(observations, lengths)
        if unique is None:
            return [np.asarray(obs, dtype=int) for obs in observations], lengths, counts, inverse

        unique_observations, unique_counts, unique_inverse = unique
        if len(unique_counts) == len(lengths):
            return [np.asarray(obs, dtype=int) for obs in observations], lengths, counts, inverse

        unique_lengths = np.repeat(int(lengths[0]), len(unique_counts)).astype(int)
        flattened = [matrix.reshape(-1).astype(int, copy=False) for matrix in unique_observations]
        return flattened, unique_lengths, unique_counts, unique_inverse.astype(int, copy=False)

    def _fixed_probability_log_likelihoods_from_observations(
        self,
        observations: Sequence[np.ndarray],
        lengths: np.ndarray,
    ) -> np.ndarray:
        log_likelihoods = np.zeros((len(lengths), self.n_clusters), dtype=float)
        starts = np.zeros(len(lengths) + 1, dtype=int)
        starts[1:] = np.cumsum(lengths)

        for cluster_idx in range(self.n_clusters):
            initial = self.initial_probs[cluster_idx]
            transition = self.transition_probs[cluster_idx]
            emission = self.emission_probs[cluster_idx]
            for seq_idx in range(len(lengths)):
                start = starts[seq_idx]
                end = starts[seq_idx + 1]
                if self.n_channels == 1:
                    obs = observations[0][start:end]
                    log_likelihoods[seq_idx, cluster_idx] = self._forward_backward_fixed(
                        initial,
                        transition,
                        emission,
                        obs,
                    )[2]
                else:
                    obs_list = [obs[start:end] for obs in observations]
                    log_likelihoods[seq_idx, cluster_idx] = (
                        self._multichannel_forward_log_likelihood(
                            initial,
                            transition,
                            emission,
                            obs_list,
                        )
                    )
        return log_likelihoods

    def _ensure_fixed_probability_parameters(self) -> None:
        rng = np.random.default_rng(self.random_state)
        if self.cluster_probs is None:
            self.cluster_probs = np.ones(self.n_clusters, dtype=float) / self.n_clusters

        if self.initial_probs is None:
            self.initial_probs = [
                rng.dirichlet(np.ones(n_states_k)) for n_states_k in self.n_states
            ]
        if self.transition_probs is None:
            self.transition_probs = [
                rng.dirichlet(np.ones(n_states_k), size=n_states_k)
                for n_states_k in self.n_states
            ]
        if self.emission_probs is None:
            self.emission_probs = [
                rng.dirichlet(np.ones(self.n_symbols), size=n_states_k)
                for n_states_k in self.n_states
            ]

    def _ensure_multichannel_fixed_probability_parameters(self) -> None:
        rng = np.random.default_rng(self.random_state)
        if self.cluster_probs is None and self.eta_omega is None:
            self.cluster_probs = np.ones(self.n_clusters, dtype=float) / self.n_clusters

        if self.initial_probs is None:
            self.initial_probs = [
                rng.dirichlet(np.ones(n_states_k)) for n_states_k in self.n_states
            ]
        if self.transition_probs is None:
            self.transition_probs = [
                rng.dirichlet(np.ones(n_states_k), size=n_states_k)
                for n_states_k in self.n_states
            ]
        if self.emission_probs is None:
            self.emission_probs = [
                [
                    rng.dirichlet(np.ones(n_symbols), size=n_states_k)
                    for n_symbols in self.n_symbols
                ]
                for n_states_k in self.n_states
            ]

    @staticmethod
    def _forward_backward_fixed(
        initial_probs: np.ndarray,
        transition_probs: np.ndarray,
        emission_probs: np.ndarray,
        observations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        T = len(observations)
        S = len(initial_probs)
        log_initial = _log_probabilities(initial_probs)
        log_transition = _log_probabilities(transition_probs)
        log_emission = _log_probabilities(emission_probs)

        log_alpha = np.empty((T, S), dtype=float)
        log_alpha[0] = log_initial + log_emission[:, observations[0]]
        for t in range(1, T):
            log_alpha[t] = (
                logsumexp(log_alpha[t - 1, :, np.newaxis] + log_transition, axis=0)
                + log_emission[:, observations[t]]
            )

        log_beta = np.zeros((T, S), dtype=float)
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_transition + log_emission[:, observations[t + 1]] + log_beta[t + 1],
                axis=1,
            )

        log_likelihood = float(logsumexp(log_alpha[-1]))
        xi = np.zeros((max(T - 1, 0), S, S), dtype=float)
        if not np.isfinite(log_likelihood):
            return np.zeros((T, S), dtype=float), xi, log_likelihood
        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)

        for t in range(T - 1):
            log_xi = (
                log_alpha[t, :, np.newaxis]
                + log_transition
                + log_emission[:, observations[t + 1]][np.newaxis, :]
                + log_beta[t + 1, np.newaxis, :]
                - log_likelihood
            )
            xi[t] = np.exp(log_xi)
        return gamma, xi, log_likelihood

    @staticmethod
    def _multichannel_forward_backward_fixed(
        initial_probs: np.ndarray,
        transition_probs: np.ndarray,
        emission_probs: Sequence[np.ndarray],
        observations: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        T = len(observations[0])
        S = len(initial_probs)
        log_initial = _log_probabilities(initial_probs)
        log_transition = _log_probabilities(transition_probs)
        log_emissions = [_log_probabilities(emission) for emission in emission_probs]
        log_emission_matrix = np.zeros((T, S), dtype=float)
        for obs, log_emission in zip(observations, log_emissions):
            log_emission_matrix += log_emission[:, obs].T

        log_alpha = np.empty((T, S), dtype=float)
        log_alpha[0] = log_initial + log_emission_matrix[0]

        for t in range(1, T):
            log_alpha[t] = (
                logsumexp(log_alpha[t - 1, :, np.newaxis] + log_transition, axis=0)
                + log_emission_matrix[t]
            )

        log_beta = np.zeros((T, S), dtype=float)
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_transition
                + log_emission_matrix[t + 1, np.newaxis, :]
                + log_beta[t + 1, np.newaxis, :],
                axis=1,
            )

        log_likelihood = float(logsumexp(log_alpha[-1]))
        xi = np.zeros((max(T - 1, 0), S, S), dtype=float)
        if not np.isfinite(log_likelihood):
            return np.zeros((T, S), dtype=float), xi, log_likelihood
        gamma = np.exp(log_alpha + log_beta - log_likelihood)
        for t in range(T - 1):
            log_xi = (
                log_alpha[t, :, np.newaxis]
                + log_transition
                + log_emission_matrix[t + 1, np.newaxis, :]
                + log_beta[t + 1, np.newaxis, :]
                - log_likelihood
            )
            xi[t] = np.exp(log_xi)
        return gamma, xi, log_likelihood

    def _weighted_fixed_mstep_for_cluster(
        self,
        X_int: np.ndarray,
        lengths: np.ndarray,
        resp_k: np.ndarray,
        cluster_idx: int,
        sequence_weights: Optional[np.ndarray] = None,
    ) -> None:
        S = self.n_states[cluster_idx]
        EPS = 1e-10
        initial = self.initial_probs[cluster_idx]
        transition = self.transition_probs[cluster_idx]
        emission = self.emission_probs[cluster_idx]

        acc_pi = np.zeros(S, dtype=float)
        acc_A = np.zeros((S, S), dtype=float)
        acc_A_den = np.zeros(S, dtype=float)
        acc_B = np.zeros((S, self.n_symbols), dtype=float)
        acc_B_den = np.zeros(S, dtype=float)
        total_weight = 0.0

        starts = np.zeros(len(lengths) + 1, dtype=int)
        starts[1:] = np.cumsum(lengths)
        if sequence_weights is None:
            sequence_weights = np.ones(len(lengths), dtype=float)
        for seq_idx, seq_length in enumerate(lengths):
            weight = float(resp_k[seq_idx]) * float(sequence_weights[seq_idx])
            if weight <= EPS:
                continue
            start = starts[seq_idx]
            end = starts[seq_idx + 1]
            obs = X_int[start:end, 0].astype(int, copy=False)
            gamma, xi, _ = self._forward_backward_fixed(initial, transition, emission, obs)

            acc_pi += weight * gamma[0]
            total_weight += weight
            if xi.shape[0] > 0:
                acc_A += weight * xi.sum(axis=0)
                acc_A_den += weight * gamma[:-1].sum(axis=0)
            gamma_sum = gamma.sum(axis=0)
            np.add.at(
                acc_B,
                (np.arange(S)[:, np.newaxis], obs[np.newaxis, :]),
                weight * gamma.T,
            )
            acc_B_den += weight * gamma_sum

        if total_weight <= EPS:
            return

        new_pi = np.maximum(acc_pi / total_weight, EPS)
        new_pi /= new_pi.sum()

        new_A = np.zeros((S, S), dtype=float)
        for state_idx in range(S):
            if acc_A_den[state_idx] > EPS:
                new_A[state_idx] = acc_A[state_idx] / acc_A_den[state_idx]
            else:
                new_A[state_idx] = 1.0 / S
        new_A = np.maximum(new_A, EPS)
        new_A /= new_A.sum(axis=1, keepdims=True)

        new_B = np.zeros((S, self.n_symbols), dtype=float)
        for state_idx in range(S):
            if acc_B_den[state_idx] > EPS:
                new_B[state_idx] = acc_B[state_idx] / acc_B_den[state_idx]
            else:
                new_B[state_idx] = 1.0 / self.n_symbols
        new_B = np.maximum(new_B, EPS)
        new_B /= new_B.sum(axis=1, keepdims=True)

        self.initial_probs[cluster_idx] = new_pi
        self.transition_probs[cluster_idx] = new_A
        self.emission_probs[cluster_idx] = new_B

    def _weighted_multichannel_mstep_for_cluster(
        self,
        observations: Sequence[np.ndarray],
        lengths: np.ndarray,
        resp_k: np.ndarray,
        cluster_idx: int,
        sequence_weights: Optional[np.ndarray] = None,
    ) -> None:
        S = self.n_states[cluster_idx]
        EPS = 1e-10
        initial = self.initial_probs[cluster_idx]
        transition = self.transition_probs[cluster_idx]
        emissions = self.emission_probs[cluster_idx]

        acc_pi = np.zeros(S, dtype=float)
        acc_A = np.zeros((S, S), dtype=float)
        acc_A_den = np.zeros(S, dtype=float)
        acc_B = [np.zeros((S, n_symbols), dtype=float) for n_symbols in self.n_symbols]
        acc_B_den = [np.zeros(S, dtype=float) for _ in range(self.n_channels)]
        total_weight = 0.0

        starts = np.zeros(len(lengths) + 1, dtype=int)
        starts[1:] = np.cumsum(lengths)
        if sequence_weights is None:
            sequence_weights = np.ones(len(lengths), dtype=float)
        for seq_idx, _seq_length in enumerate(lengths):
            weight = float(resp_k[seq_idx]) * float(sequence_weights[seq_idx])
            if weight <= EPS:
                continue
            start = starts[seq_idx]
            end = starts[seq_idx + 1]
            obs_list = [obs[start:end] for obs in observations]
            gamma, xi, _ = self._multichannel_forward_backward_fixed(
                initial,
                transition,
                emissions,
                obs_list,
            )

            acc_pi += weight * gamma[0]
            total_weight += weight
            if xi.shape[0] > 0:
                acc_A += weight * xi.sum(axis=0)
                acc_A_den += weight * gamma[:-1].sum(axis=0)
            gamma_sum = gamma.sum(axis=0)
            for channel_idx, obs_ch in enumerate(obs_list):
                np.add.at(
                    acc_B[channel_idx],
                    (np.arange(S)[:, np.newaxis], obs_ch[np.newaxis, :]),
                    weight * gamma.T,
                )
                acc_B_den[channel_idx] += weight * gamma_sum

        if total_weight <= EPS:
            return

        new_pi = np.maximum(acc_pi / total_weight, EPS)
        new_pi /= new_pi.sum()

        new_A = np.zeros((S, S), dtype=float)
        for state_idx in range(S):
            if acc_A_den[state_idx] > EPS:
                new_A[state_idx] = acc_A[state_idx] / acc_A_den[state_idx]
            else:
                new_A[state_idx] = 1.0 / S
        new_A = np.maximum(new_A, EPS)
        new_A /= new_A.sum(axis=1, keepdims=True)

        new_emissions = []
        for channel_idx, counts in enumerate(acc_B):
            n_symbols = counts.shape[1]
            emission = np.zeros_like(counts)
            for state_idx in range(S):
                if acc_B_den[channel_idx][state_idx] > EPS:
                    emission[state_idx] = counts[state_idx] / acc_B_den[channel_idx][state_idx]
                else:
                    emission[state_idx] = 1.0 / n_symbols
            emission = np.maximum(emission, EPS)
            emission /= emission.sum(axis=1, keepdims=True)
            new_emissions.append(emission)

        self.initial_probs[cluster_idx] = new_pi
        self.transition_probs[cluster_idx] = new_A
        self.emission_probs[cluster_idx] = new_emissions

    def _pack_covariate_parameters(
        self,
    ) -> tuple[np.ndarray, list[tuple[str, int | None, int | None, tuple[int, ...]]]]:
        entries: list[tuple[str, int | None, int | None, tuple[int, ...]]] = []
        chunks: list[np.ndarray] = []

        if self.initial_probs is None:
            for cluster_idx, eta in enumerate(self.eta_pi):
                reduced = _reduce_eta_pi_full(eta)
                entries.append(("eta_pi_reduced", cluster_idx, None, reduced.shape))
                chunks.append(reduced.ravel(order="F"))
        if self.transition_probs is None:
            for cluster_idx, eta in enumerate(self.eta_A):
                reduced = _reduce_eta_A_full(eta)
                entries.append(("eta_A_reduced", cluster_idx, None, reduced.shape))
                chunks.append(reduced.ravel(order="F"))
        if self.emission_probs is None:
            if self.n_channels == 1:
                for cluster_idx, eta in enumerate(self.eta_B):
                    reduced = _reduce_eta_B_full(eta)
                    entries.append(("eta_B_reduced", cluster_idx, None, reduced.shape))
                    chunks.append(reduced.ravel(order="F"))
            else:
                for cluster_idx, cluster_eta in enumerate(self.eta_B):
                    for channel_idx, eta in enumerate(cluster_eta):
                        reduced = _reduce_eta_B_full(eta)
                        entries.append(
                            ("eta_B_reduced", cluster_idx, channel_idx, reduced.shape)
                        )
                        chunks.append(reduced.ravel(order="F"))
        if self.cluster_probs is None:
            if self.eta_omega is None:
                self.enable_cluster_covariate_estimation()
            reduced = _reduce_eta_omega_full(self.eta_omega)
            entries.append(("eta_omega_reduced", None, None, reduced.shape))
            chunks.append(reduced.ravel(order="F"))

        if not chunks:
            return np.array([], dtype=float), entries
        return np.concatenate(chunks).astype(float, copy=False), entries

    def _covariate_direct_compression_indices(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_channels == 1:
            observations_data, lengths = sequence_data_to_hmmlearn_format(self.observations)
            observation_arrays = [observations_data[:, 0].astype(int, copy=False)]
        else:
            observations_data, lengths = multichannel_to_hmmlearn_format(self.channels)
            observation_arrays = [
                X[:, 0].astype(int, copy=False) for X in observations_data
            ]

        starts = np.zeros(len(lengths) + 1, dtype=int)
        starts[1:] = np.cumsum(lengths)
        groups: dict[tuple, int] = {}
        indices: list[int] = []
        counts: list[float] = []

        for seq_idx, seq_length in enumerate(lengths):
            seq_length = int(seq_length)
            start = starts[seq_idx]
            end = starts[seq_idx + 1]
            key_parts: list[tuple] = [("length", seq_length)]
            for obs in observation_arrays:
                key_parts.append(("obs", _array_key(obs[start:end])))
            if self.initial_probs is None:
                key_parts.append(("X_pi", _array_key(self.X_pi[seq_idx, :seq_length])))
            if self.transition_probs is None:
                key_parts.append(("X_A", _array_key(self.X_A[seq_idx, :seq_length])))
            if self.emission_probs is None:
                key_parts.append(("X_B", _array_key(self.X_B[seq_idx, :seq_length])))
            if self.cluster_probs is None and self.eta_omega is not None:
                key_parts.append(("X_cluster", _array_key(self.X_cluster[seq_idx])))

            key = tuple(key_parts)
            group_pos = groups.get(key)
            if group_pos is None:
                groups[key] = len(indices)
                indices.append(seq_idx)
                counts.append(1.0)
            else:
                counts[group_pos] += 1.0

        return np.asarray(indices, dtype=int), np.asarray(counts, dtype=float)

    def _loglik_and_reduced_gradient(
        self,
        sequence_indices: Optional[np.ndarray] = None,
        sequence_weights: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if not self.has_complete_parameters:
            raise ValueError("MNHMM parameters are incomplete")

        params, entries = self._pack_covariate_parameters()
        if self.n_channels == 1:
            observations_data, lengths = sequence_data_to_hmmlearn_format(self.observations)
            observations_data = observations_data[:, 0].astype(int, copy=False)
        else:
            observations_data, lengths = multichannel_to_hmmlearn_format(self.channels)
            observations_data = [
                X[:, 0].astype(int, copy=False) for X in observations_data
            ]
        starts = np.zeros(len(lengths) + 1, dtype=int)
        starts[1:] = np.cumsum(lengths)
        if sequence_indices is None:
            sequence_indices = np.arange(len(lengths), dtype=int)
        else:
            sequence_indices = np.asarray(sequence_indices, dtype=int)
        if sequence_weights is None:
            sequence_weights = np.ones(len(sequence_indices), dtype=float)
        else:
            sequence_weights = np.asarray(sequence_weights, dtype=float)
        if sequence_indices.ndim != 1 or sequence_weights.ndim != 1:
            raise ValueError("sequence_indices and sequence_weights must be one-dimensional")
        if len(sequence_indices) != len(sequence_weights):
            raise ValueError("sequence_indices and sequence_weights lengths must match")
        if (
            np.any(sequence_indices < 0)
            or np.any(sequence_indices >= len(lengths))
            or not np.isfinite(sequence_weights).all()
            or np.any(sequence_weights <= 0)
        ):
            raise ValueError("invalid compressed sequence indices or weights")

        cluster_probs = self._cluster_probs_for_indices(sequence_indices)
        log_likelihoods = np.zeros((len(sequence_indices), self.n_clusters), dtype=float)
        caches = [[] for _ in range(self.n_clusters)]

        for cluster_idx in range(self.n_clusters):
            initial, transition, emission = self._component_probs_for_indices(
                cluster_idx,
                sequence_indices,
            )
            for row_idx, seq_idx in enumerate(sequence_indices):
                seq_length = int(lengths[seq_idx])
                start = starts[seq_idx]
                end = starts[seq_idx + 1]
                if self.n_channels == 1:
                    obs = observations_data[start:end]
                    emission_slice = emission[row_idx, :seq_length]
                    gamma, xi, seq_loglik = self._forward_backward_timevarying(
                        initial[row_idx],
                        transition[row_idx, :seq_length],
                        emission_slice,
                        obs,
                    )
                else:
                    obs = [obs_array[start:end] for obs_array in observations_data]
                    emission_slice = [
                        emission_ch[row_idx, :seq_length] for emission_ch in emission
                    ]
                    gamma, xi, seq_loglik = self._multichannel_forward_backward_timevarying(
                        initial[row_idx],
                        transition[row_idx, :seq_length],
                        emission_slice,
                        obs,
                    )
                log_likelihoods[row_idx, cluster_idx] = seq_loglik
                caches[cluster_idx].append(
                    (
                        gamma,
                        xi,
                        obs,
                        transition[row_idx, :seq_length],
                        emission_slice,
                    )
                )

        log_joint = log_likelihoods + _log_probabilities(cluster_probs)
        sequence_loglik = logsumexp(log_joint, axis=1)
        if not np.isfinite(sequence_loglik).all():
            return -np.inf, params, np.zeros_like(params)
        responsibilities = np.exp(log_joint - sequence_loglik[:, np.newaxis])
        total_loglik = float(np.sum(sequence_loglik * sequence_weights))

        grad_pi = [
            np.zeros_like(eta) if self.initial_probs is None else None
            for eta in self.eta_pi
        ]
        grad_A = [
            np.zeros_like(eta) if self.transition_probs is None else None
            for eta in self.eta_A
        ]
        if self.emission_probs is None:
            if self.n_channels == 1:
                grad_B = [np.zeros_like(eta) for eta in self.eta_B]
            else:
                grad_B = [
                    [np.zeros_like(channel_eta) for channel_eta in cluster_eta]
                    for cluster_eta in self.eta_B
                ]
        else:
            grad_B = [None for _ in range(self.n_clusters)]
        grad_omega = np.zeros_like(self.eta_omega) if self.cluster_probs is None else None

        if grad_omega is not None:
            for row_idx, seq_idx in enumerate(sequence_indices):
                grad_omega += sequence_weights[row_idx] * np.outer(
                    self.X_cluster[seq_idx],
                    responsibilities[row_idx] - cluster_probs[row_idx],
                )

        for cluster_idx in range(self.n_clusters):
            initial, _, _ = self._component_probs_for_indices(
                cluster_idx,
                sequence_indices,
            )
            for row_idx, seq_idx in enumerate(sequence_indices):
                seq_length = int(lengths[seq_idx])
                gamma, xi, obs, transition, emission = caches[cluster_idx][row_idx]
                weight = responsibilities[row_idx, cluster_idx] * sequence_weights[row_idx]

                if grad_pi[cluster_idx] is not None:
                    grad_pi[cluster_idx] += weight * np.outer(
                        self.X_pi[seq_idx, 0],
                        gamma[0] - initial[row_idx],
                    )

                if grad_A[cluster_idx] is not None:
                    for t in range(1, seq_length):
                        x_t = self.X_A[seq_idx, t]
                        for origin in range(self.n_states[cluster_idx]):
                            expected = xi[t - 1, origin]
                            parent = float(expected.sum())
                            diff = expected - parent * transition[t, origin]
                            grad_A[cluster_idx][:, origin, :] += (
                                weight * x_t[:, np.newaxis] * diff[np.newaxis, :]
                            )

                if grad_B[cluster_idx] is not None:
                    if self.n_channels == 1:
                        for t, symbol in enumerate(obs):
                            x_t = self.X_B[seq_idx, t]
                            for state in range(self.n_states[cluster_idx]):
                                diff = -gamma[t, state] * emission[t, state].copy()
                                diff[symbol] += gamma[t, state]
                                grad_B[cluster_idx][:, state, :] += (
                                    weight * x_t[:, np.newaxis] * diff[np.newaxis, :]
                                )
                    else:
                        for channel_idx, obs_channel in enumerate(obs):
                            emission_channel = emission[channel_idx]
                            for t, symbol in enumerate(obs_channel):
                                x_t = self.X_B[seq_idx, t]
                                for state in range(self.n_states[cluster_idx]):
                                    diff = -gamma[t, state] * emission_channel[t, state].copy()
                                    diff[symbol] += gamma[t, state]
                                    grad_B[cluster_idx][channel_idx][:, state, :] += (
                                        weight * x_t[:, np.newaxis] * diff[np.newaxis, :]
                                    )

        chunks: list[np.ndarray] = []
        for family, cluster_idx, channel_idx, _ in entries:
            if family == "eta_pi_reduced":
                chunks.append(_reduce_eta_pi_full(grad_pi[cluster_idx]).ravel(order="F"))
            elif family == "eta_A_reduced":
                chunks.append(_reduce_eta_A_full(grad_A[cluster_idx]).ravel(order="F"))
            elif family == "eta_B_reduced":
                if self.n_channels == 1:
                    chunks.append(_reduce_eta_B_full(grad_B[cluster_idx]).ravel(order="F"))
                else:
                    chunks.append(
                        _reduce_eta_B_full(grad_B[cluster_idx][channel_idx]).ravel(order="F")
                    )
            elif family == "eta_omega_reduced":
                chunks.append(_reduce_eta_omega_full(grad_omega).ravel(order="F"))
            else:  # pragma: no cover - entries are created internally
                raise ValueError(f"Unknown MNHMM parameter family: {family}")
        reduced_gradient = (
            np.concatenate(chunks).astype(float, copy=False)
            if chunks else np.array([], dtype=float)
        )
        return total_loglik, params, reduced_gradient

    def objective_and_gradient(
        self,
        lambda_penalty: float = 0.0,
        sequence_indices: Optional[np.ndarray] = None,
        sequence_weights: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray | float]:
        """
        Return seqHMM-style reduced-eta objective, gradient, and parameters.

        The objective follows R seqHMM's ``make_objective_mnhmm`` convention:
        negative penalized average log-likelihood, normalized by the number of
        observed sequence positions. Parameters and gradients are packed in
        reduced eta order ``pi, A, B, omega``.
        """
        lambda_penalty = float(lambda_penalty)
        if lambda_penalty < 0:
            raise ValueError("lambda_penalty must be non-negative")
        loglik, params, grad_loglik = self._loglik_and_reduced_gradient(
            sequence_indices=sequence_indices,
            sequence_weights=sequence_weights,
        )
        if sequence_indices is None:
            n_obs = float(np.sum(self.sequence_lengths))
        else:
            weights = (
                np.ones(len(sequence_indices), dtype=float)
                if sequence_weights is None
                else np.asarray(sequence_weights, dtype=float)
            )
            selected_lengths = self.sequence_lengths[
                np.asarray(sequence_indices, dtype=int)
            ]
            n_obs = float(np.sum(selected_lengths * weights))
        objective = -(loglik - 0.5 * lambda_penalty * float(np.dot(params, params))) / n_obs
        gradient = -(grad_loglik - lambda_penalty * params) / n_obs
        return {
            "objective": float(objective),
            "gradient": gradient,
            "parameters": params.copy(),
            "log_likelihood": float(loglik),
        }

    def _unpack_covariate_parameters(
        self,
        params: np.ndarray,
        entries: list[tuple[str, int | None, int | None, tuple[int, ...]]],
    ) -> None:
        params = np.asarray(params, dtype=float)
        expected_size = int(sum(np.prod(shape) for _, _, _, shape in entries))
        if params.size != expected_size:
            raise ValueError(
                f"parameter vector length {params.size} does not match expected "
                f"length {expected_size}"
            )
        if not np.isfinite(params).all():
            raise ValueError("parameter vector must contain finite values")

        offset = 0
        for family, cluster_idx, channel_idx, shape in entries:
            size = int(np.prod(shape))
            value = params[offset:offset + size].reshape(shape, order="F")
            if family == "eta_pi_reduced":
                self.eta_pi[cluster_idx] = _expand_eta_pi_reduced_one(
                    value, self.n_states[cluster_idx], self.X_pi.shape[2]
                )
            elif family == "eta_A_reduced":
                self.eta_A[cluster_idx] = _expand_eta_A_reduced_one(
                    value, self.n_states[cluster_idx], self.X_A.shape[2]
                )
            elif family == "eta_B_reduced":
                if self.n_channels == 1:
                    self.eta_B[cluster_idx] = _expand_eta_B_reduced_one(
                        value,
                        self.n_symbols,
                        self.n_states[cluster_idx],
                        self.X_B.shape[2],
                    )
                else:
                    self.eta_B[cluster_idx][channel_idx] = _expand_eta_B_reduced_one(
                        value,
                        self.n_symbols[channel_idx],
                        self.n_states[cluster_idx],
                        self.X_B.shape[2],
                    )
            elif family == "eta_omega_reduced":
                self.eta_omega = _expand_eta_omega_reduced_one(
                    value, self.n_clusters, self.X_cluster.shape[1]
                )
            else:  # pragma: no cover - entries are created internally
                raise ValueError(f"Unknown MNHMM parameter family: {family}")
            offset += size

    def _fit_covariate_direct(
        self,
        n_iter: int,
        tol: float,
        verbose: bool,
        lambda_penalty: float = 0.0,
        compress: bool = False,
    ) -> "MNHMM":
        """
        Fit a covariate MNHMM by direct observed-likelihood optimization.

        Optimize covariate-dependent MNHMM probability families with analytic
        gradients and L-BFGS.
        """
        params0, entries = self._pack_covariate_parameters()
        sequence_indices = None
        sequence_weights = None
        if compress:
            sequence_indices, sequence_weights = self._covariate_direct_compression_indices()
            if (
                len(sequence_indices) == self.n_sequences
                and np.all(sequence_weights == 1.0)
            ):
                sequence_indices = None
                sequence_weights = None
        self.n_optimized_parameters = int(params0.size)
        if params0.size == 0 or int(n_iter) <= 0:
            self.log_likelihood = self._mixture_log_likelihood()
            self.responsibilities = self.compute_responsibilities()
            self.n_iter = 0
            self.converged = True
            self.estimation_method = "direct_l_bfgs"
            self.lambda_penalty = float(lambda_penalty)
            return self

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            try:
                self._unpack_covariate_parameters(params, entries)
            except ValueError:
                return np.inf, np.zeros_like(params)
            out = self.objective_and_gradient(
                lambda_penalty=lambda_penalty,
                sequence_indices=sequence_indices,
                sequence_weights=sequence_weights,
            )
            value = float(out["objective"])
            gradient = np.asarray(out["gradient"], dtype=float)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                return np.inf, np.zeros_like(params)
            return value, gradient

        result = minimize(
            objective,
            params0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": int(n_iter),
                "ftol": float(tol),
                "gtol": float(tol),
                "maxls": 50,
                "disp": bool(verbose),
            },
        )
        self._unpack_covariate_parameters(result.x, entries)
        self.log_likelihood = self.score()
        self.responsibilities = self.compute_responsibilities()
        self.n_iter = int(result.nit)
        self.converged = bool(result.success)
        self.estimation_method = "direct_l_bfgs"
        self.lambda_penalty = float(lambda_penalty)
        self.optimization_result = result
        if verbose:
            print(f"L-BFGS {'converged' if result.success else 'stopped'}: {result.message}")
            print(f"Log-likelihood: {self.log_likelihood:.4f}")
        return self

    def _fit_cluster_covariates_direct(
        self,
        n_iter: int,
        tol: float,
        verbose: bool,
        lambda_penalty: float = 0.0,
        compress: bool = False,
    ) -> "MNHMM":
        """Optimize mixture weights while fixed component likelihoods stay cached."""
        if self.eta_omega is None:
            raise ValueError("eta_omega must be initialized before cluster optimization")

        method_name = (
            "fixed_multichannel_cluster_l_bfgs"
            if self.n_channels > 1
            else "direct_l_bfgs"
        )
        component_log_likelihoods = self._sequence_log_likelihoods()
        X_cluster = self.X_cluster
        counts = np.ones(component_log_likelihoods.shape[0], dtype=float)
        if compress:
            rows = np.concatenate([component_log_likelihoods, self.X_cluster], axis=1)
            indices, _, counts = _unique_rows_with_counts_and_inverse(rows)
            if len(indices) < component_log_likelihoods.shape[0]:
                component_log_likelihoods = component_log_likelihoods[indices]
                X_cluster = self.X_cluster[indices]
        reduced0 = _reduce_eta_omega_full(np.asarray(self.eta_omega, dtype=float))
        params0 = reduced0.ravel(order="F")
        lambda_penalty = float(lambda_penalty)
        self.n_optimized_parameters = int(params0.size)
        if params0.size == 0 or int(n_iter) <= 0:
            self.eta_omega = _expand_eta_omega_reduced_one(
                reduced0, self.n_clusters, self.X_cluster.shape[1]
            )
            self.log_likelihood = self._mixture_log_likelihood()
            self.responsibilities = self.compute_responsibilities()
            self.n_iter = 0
            self.converged = True
            self.estimation_method = method_name
            self.lambda_penalty = lambda_penalty
            return self

        n_obs = max(float(counts.sum()), 1.0)
        reduced_shape = reduced0.shape

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            reduced = params.reshape(reduced_shape, order="F")
            eta = _expand_eta_omega_reduced_one(
                reduced, self.n_clusters, self.X_cluster.shape[1]
            )
            logits = X_cluster @ eta
            priors = softmax(logits, axis=1)
            log_joint = component_log_likelihoods + _log_probabilities(priors)
            try:
                responsibilities, log_norm = _cluster_responsibilities_from_log_joint(
                    log_joint
                )
            except ValueError:
                return np.inf, np.zeros_like(params)
            value = -float(
                np.sum(log_norm[:, 0] * counts) - 0.5 * lambda_penalty * np.dot(params, params)
            ) / n_obs
            grad_full = X_cluster.T @ ((responsibilities - priors) * counts[:, np.newaxis])
            grad_reduced = _reduce_eta_omega_full(grad_full).ravel(order="F")
            gradient = -(grad_reduced - lambda_penalty * params) / n_obs
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                return np.inf, np.zeros_like(params)
            return value, gradient

        result = minimize(
            objective,
            params0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": int(n_iter),
                "ftol": float(tol),
                "gtol": float(tol),
                "maxls": 50,
                "disp": bool(verbose),
            },
        )
        self.eta_omega = _expand_eta_omega_reduced_one(
            result.x.reshape(reduced_shape, order="F"),
            self.n_clusters,
            self.X_cluster.shape[1],
        )
        self.log_likelihood = self.score()
        self.responsibilities = self.compute_responsibilities()
        self.n_iter = int(result.nit)
        self.converged = bool(result.success)
        self.estimation_method = method_name
        self.lambda_penalty = lambda_penalty
        self.optimization_result = result
        if verbose:
            print(f"L-BFGS {'converged' if result.success else 'stopped'}: {result.message}")
            print(f"Log-likelihood: {self.log_likelihood:.4f}")
        return self

    def _fit_cluster_covariates_to_responsibilities(
        self,
        responsibilities: np.ndarray,
        n_iter: int,
        tol: float,
        lambda_penalty: float = 0.0,
        sequence_weights: Optional[np.ndarray] = None,
    ) -> None:
        if self.eta_omega is None:
            self.enable_cluster_covariate_estimation(scale=0.0)

        reduced0 = _reduce_eta_omega_full(np.asarray(self.eta_omega, dtype=float))
        params0 = reduced0.ravel(order="F")
        if params0.size == 0:
            return

        if sequence_weights is None:
            sequence_weights = np.ones(responsibilities.shape[0], dtype=float)
        sequence_weights = np.asarray(sequence_weights, dtype=float)
        lambda_penalty = float(lambda_penalty)
        n_obs = max(float(sequence_weights.sum()), 1.0)
        reduced_shape = reduced0.shape

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            reduced = params.reshape(reduced_shape, order="F")
            eta = _expand_eta_omega_reduced_one(
                reduced, self.n_clusters, self.X_cluster.shape[1]
            )
            priors = softmax(self.X_cluster @ eta, axis=1)
            value = -float(
                np.sum(
                    sequence_weights[:, np.newaxis]
                    * responsibilities
                    * _log_probabilities(priors)
                )
                - 0.5 * lambda_penalty * np.dot(params, params)
            ) / n_obs
            grad_full = self.X_cluster.T @ (
                sequence_weights[:, np.newaxis] * (responsibilities - priors)
            )
            grad_reduced = _reduce_eta_omega_full(grad_full).ravel(order="F")
            gradient = -(grad_reduced - lambda_penalty * params) / n_obs
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                return np.inf, np.zeros_like(params)
            return value, gradient

        result = minimize(
            objective,
            params0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": int(max(n_iter, 1)),
                "ftol": float(tol),
                "gtol": float(tol),
                "maxls": 50,
                "disp": False,
            },
        )
        self.eta_omega = _expand_eta_omega_reduced_one(
            result.x.reshape(reduced_shape, order="F"),
            self.n_clusters,
            self.X_cluster.shape[1],
        )

    def _fit_fixed_component_cluster_probs(
        self,
        n_iter: int,
        tol: float,
        verbose: bool,
        compress: bool = False,
    ) -> "MNHMM":
        self._ensure_multichannel_fixed_probability_parameters()
        observations, lengths = multichannel_to_hmmlearn_format(self.channels)
        observations = [X[:, 0].astype(int, copy=False) for X in observations]
        fit_observations, fit_lengths, fit_counts, fit_inverse = (
            self._compressed_fixed_fit_data(observations, lengths, compress)
        )
        log_likelihoods = self._fixed_probability_log_likelihoods_from_observations(
            fit_observations,
            fit_lengths,
        )
        previous_log_likelihood = -np.inf
        iteration = -1

        for iteration in range(int(n_iter)):
            cluster_probs = self._cluster_probs_for_n(log_likelihoods.shape[0])
            log_joint = log_likelihoods + _log_probabilities(cluster_probs)
            responsibilities, log_norm = _cluster_responsibilities_from_log_joint(
                log_joint
            )
            weighted_resp = responsibilities * fit_counts[:, np.newaxis]
            self.cluster_probs = np.maximum(
                weighted_resp.sum(axis=0) / fit_counts.sum(),
                _FLOAT_TINY,
            )
            self.cluster_probs /= self.cluster_probs.sum()

            post_log_joint = (
                log_likelihoods + _log_probabilities(self.cluster_probs)
            )
            (
                post_responsibilities,
                post_log_norm,
            ) = _cluster_responsibilities_from_log_joint(post_log_joint)
            self.responsibilities = post_responsibilities[fit_inverse]
            log_likelihood = float(np.sum(post_log_norm[:, 0] * fit_counts))
            if verbose:
                print(f"Iteration {iteration + 1}: log-likelihood = {log_likelihood:.4f}")
            if iteration > 0 and abs(log_likelihood - previous_log_likelihood) < tol:
                self.converged = True
                break
            previous_log_likelihood = log_likelihood

        cluster_probs = self._cluster_probs_for_n(log_likelihoods.shape[0])
        final_log_joint = log_likelihoods + _log_probabilities(cluster_probs)
        (
            final_responsibilities,
            final_log_norm,
        ) = _cluster_responsibilities_from_log_joint(final_log_joint)
        self.responsibilities = final_responsibilities[fit_inverse]
        self.log_likelihood = float(np.sum(final_log_norm[:, 0] * fit_counts))
        self.n_iter = max(iteration + 1, 0)
        if self.converged is None:
            self.converged = False
        self.estimation_method = "fixed_component_cluster_em"
        return self

    def _fit_multichannel_em(
        self,
        n_iter: int,
        tol: float,
        verbose: bool,
        lambda_penalty: float = 0.0,
        compress: bool = False,
    ) -> "MNHMM":
        self._ensure_multichannel_fixed_probability_parameters()
        observations, lengths = multichannel_to_hmmlearn_format(self.channels)
        observations = [X[:, 0].astype(int, copy=False) for X in observations]
        fit_observations, fit_lengths, fit_counts, fit_inverse = (
            self._compressed_fixed_fit_data(observations, lengths, compress)
        )
        previous_log_likelihood = -np.inf
        iteration = -1

        for iteration in range(int(n_iter)):
            log_likelihoods = self._fixed_probability_log_likelihoods_from_observations(
                fit_observations,
                fit_lengths,
            )
            cluster_probs = self._cluster_probs_for_n(log_likelihoods.shape[0])
            log_joint = log_likelihoods + _log_probabilities(cluster_probs)
            responsibilities, log_norm = _cluster_responsibilities_from_log_joint(
                log_joint
            )

            if self.eta_omega is not None and not self._cluster_probs_user_fixed:
                self._fit_cluster_covariates_to_responsibilities(
                    responsibilities,
                    n_iter=max(20, int(n_iter)),
                    tol=tol,
                    lambda_penalty=lambda_penalty,
                )
            else:
                weighted_resp = responsibilities * fit_counts[:, np.newaxis]
                self.cluster_probs = weighted_resp.sum(axis=0) / fit_counts.sum()
                self.cluster_probs = np.maximum(self.cluster_probs, _FLOAT_TINY)
                self.cluster_probs /= self.cluster_probs.sum()

            for cluster_idx in range(self.n_clusters):
                self._weighted_multichannel_mstep_for_cluster(
                    fit_observations,
                    fit_lengths,
                    responsibilities[:, cluster_idx],
                    cluster_idx,
                    sequence_weights=fit_counts,
                )

            post_log_likelihoods = self._fixed_probability_log_likelihoods_from_observations(
                fit_observations,
                fit_lengths,
            )
            post_cluster_probs = self._cluster_probs_for_n(post_log_likelihoods.shape[0])
            post_log_joint = (
                post_log_likelihoods
                + _log_probabilities(post_cluster_probs)
            )
            (
                post_responsibilities,
                post_log_norm,
            ) = _cluster_responsibilities_from_log_joint(post_log_joint)
            self.responsibilities = post_responsibilities[fit_inverse]
            log_likelihood = float(np.sum(post_log_norm[:, 0] * fit_counts))
            if verbose:
                print(f"Iteration {iteration + 1}: log-likelihood = {log_likelihood:.4f}")
            if iteration > 0 and abs(log_likelihood - previous_log_likelihood) < tol:
                self.converged = True
                break
            previous_log_likelihood = log_likelihood

        final_log_likelihoods = self._fixed_probability_log_likelihoods_from_observations(
            fit_observations,
            fit_lengths,
        )
        final_cluster_probs = self._cluster_probs_for_n(final_log_likelihoods.shape[0])
        final_log_joint = (
            final_log_likelihoods + _log_probabilities(final_cluster_probs)
        )
        (
            final_responsibilities,
            final_log_norm,
        ) = _cluster_responsibilities_from_log_joint(final_log_joint)
        self.responsibilities = final_responsibilities[fit_inverse]
        self.log_likelihood = float(np.sum(final_log_norm[:, 0] * fit_counts))
        self.n_iter = iteration + 1
        if self.converged is None:
            self.converged = False
        self.estimation_method = "multichannel_em"
        return self

    def fit(
        self,
        n_iter: int = 100,
        tol: float = 1e-2,
        verbose: bool = False,
        lambda_penalty: float = 0.0,
        compress: bool = False,
    ) -> "MNHMM":
        """
        Estimate an intercept-only MNHMM using weighted Baum-Welch EM.

        Intercept-only models use weighted Baum-Welch EM. Covariate-dependent
        models use direct L-BFGS observed-likelihood optimization. ``compress``
        reuses exact repeated statistical units: observations alone for fixed
        probabilities, and observations plus active covariate design arrays for
        covariate-dependent direct fits.
        """
        self.converged = None
        lambda_penalty = float(lambda_penalty)
        if lambda_penalty < 0:
            raise ValueError("lambda_penalty must be non-negative")
        if self.n_channels > 1:
            components_fixed = (
                self.initial_probs is not None
                and self.transition_probs is not None
                and self.emission_probs is not None
            )
            if (
                self.has_cluster_covariates()
                and not self._cluster_probs_user_fixed
                and self.eta_omega is None
            ):
                self.enable_cluster_covariate_estimation()
            if self.eta_omega is not None and not self._cluster_probs_user_fixed:
                if components_fixed:
                    return self._fit_cluster_covariates_direct(
                        n_iter=n_iter,
                        tol=tol,
                        verbose=verbose,
                        lambda_penalty=lambda_penalty,
                        compress=compress,
                    )
            if not self._supports_fixed_probability_em():
                return self._fit_covariate_direct(
                    n_iter=n_iter,
                    tol=tol,
                    verbose=verbose,
                    lambda_penalty=lambda_penalty,
                    compress=compress,
                )
            if (
                self.initial_probs is None
                or self.transition_probs is None
                or self.emission_probs is None
            ):
                return self._fit_multichannel_em(
                    n_iter=n_iter,
                    tol=tol,
                    verbose=verbose,
                    lambda_penalty=lambda_penalty,
                    compress=compress,
                )
            if (
                components_fixed
                and not self._cluster_probs_user_fixed
                and self.eta_omega is None
            ):
                return self._fit_fixed_component_cluster_probs(
                    n_iter=n_iter,
                    tol=tol,
                    verbose=verbose,
                    compress=compress,
                )
            self.log_likelihood = self.score()
            self.responsibilities = self.compute_responsibilities()
            self.n_iter = 0
            self.converged = True
            self.estimation_method = "fixed_multichannel_inference"
            return self

        if (
            self.has_cluster_covariates()
            and not self._cluster_probs_user_fixed
            and self.eta_omega is None
        ):
            self.enable_cluster_covariate_estimation()
        components_fixed = (
            self.initial_probs is not None
            and self.transition_probs is not None
            and self.emission_probs is not None
        )
        if self.eta_omega is not None and not self._cluster_probs_user_fixed:
            if components_fixed:
                return self._fit_cluster_covariates_direct(
                    n_iter=n_iter,
                    tol=tol,
                    verbose=verbose,
                    lambda_penalty=lambda_penalty,
                    compress=compress,
                )
        if not self._supports_fixed_probability_em():
            return self._fit_covariate_direct(
                n_iter=n_iter,
                tol=tol,
                verbose=verbose,
                lambda_penalty=lambda_penalty,
                compress=compress,
            )

        self._ensure_fixed_probability_parameters()
        X_int, lengths = sequence_data_to_hmmlearn_format(self.observations)
        observations = [X_int[:, 0].astype(int, copy=False)]
        fit_observations, fit_lengths, fit_counts, fit_inverse = (
            self._compressed_fixed_fit_data(observations, lengths, compress)
        )
        previous_log_likelihood = -np.inf
        iteration = -1

        for iteration in range(int(n_iter)):
            log_likelihoods = self._fixed_probability_log_likelihoods_from_observations(
                fit_observations,
                fit_lengths,
            )
            log_joint = log_likelihoods + _log_probabilities(self.cluster_probs)
            responsibilities, log_norm = _cluster_responsibilities_from_log_joint(
                log_joint
            )
            weighted_resp = responsibilities * fit_counts[:, np.newaxis]
            self.cluster_probs = weighted_resp.sum(axis=0) / fit_counts.sum()
            self.cluster_probs = np.maximum(self.cluster_probs, _FLOAT_TINY)
            self.cluster_probs /= self.cluster_probs.sum()

            for cluster_idx in range(self.n_clusters):
                self._weighted_fixed_mstep_for_cluster(
                    fit_observations[0].reshape(-1, 1),
                    fit_lengths,
                    responsibilities[:, cluster_idx],
                    cluster_idx,
                    sequence_weights=fit_counts,
                )

            post_log_likelihoods = self._fixed_probability_log_likelihoods_from_observations(
                fit_observations,
                fit_lengths,
            )
            post_log_joint = (
                post_log_likelihoods + _log_probabilities(self.cluster_probs)
            )
            (
                post_responsibilities,
                post_log_norm,
            ) = _cluster_responsibilities_from_log_joint(post_log_joint)
            self.responsibilities = post_responsibilities[fit_inverse]
            log_likelihood = float(np.sum(post_log_norm[:, 0] * fit_counts))
            if verbose:
                print(f"Iteration {iteration + 1}: log-likelihood = {log_likelihood:.4f}")
            if iteration > 0 and abs(log_likelihood - previous_log_likelihood) < tol:
                self.converged = True
                break
            previous_log_likelihood = log_likelihood

        final_log_likelihoods = self._fixed_probability_log_likelihoods_from_observations(
            fit_observations,
            fit_lengths,
        )
        final_log_joint = final_log_likelihoods + _log_probabilities(self.cluster_probs)
        (
            final_responsibilities,
            final_log_norm,
        ) = _cluster_responsibilities_from_log_joint(final_log_joint)
        self.responsibilities = final_responsibilities[fit_inverse]
        self.log_likelihood = float(np.sum(final_log_norm[:, 0] * fit_counts))
        self.n_iter = iteration + 1
        if self.converged is None:
            self.converged = False
        self.estimation_method = "intercept_em"
        return self

    def compute_responsibilities(self, sequences: Optional[SequenceData] = None) -> np.ndarray:
        """
        Compute posterior cluster probabilities for each sequence.
        """
        log_likelihoods = self._sequence_log_likelihoods(sequences)
        cluster_probs = self._cluster_probs_for_n(log_likelihoods.shape[0])
        log_joint = log_likelihoods + _log_probabilities(cluster_probs)
        responsibilities, log_norm = _cluster_responsibilities_from_log_joint(
            log_joint
        )

        if sequences is None:
            self.responsibilities = responsibilities
            self.log_likelihood = float(np.sum(log_norm))
        return responsibilities

    def predict_cluster(self, sequences: Optional[SequenceData] = None) -> np.ndarray:
        """Return the most likely cluster index for each sequence."""
        return np.argmax(self.compute_responsibilities(sequences), axis=1)

    def score(
        self,
        sequences: Optional[Union[SequenceData, Sequence[SequenceData]]] = None,
        compress: bool = False,
    ) -> float:
        """Return the mixture log-likelihood of the sequences."""
        score = (
            self._fixed_probability_score_compressed(sequences)
            if compress
            else self._mixture_log_likelihood(sequences)
        )
        if sequences is None:
            self.log_likelihood = score
        return score

    def __repr__(self) -> str:
        status = "scored" if self.log_likelihood is not None else "unscored"
        return (
            f"MNHMM(n_clusters={self.n_clusters}, n_states={self.n_states}, "
            f"n_sequences={self.n_sequences}, status='{status}')"
        )
