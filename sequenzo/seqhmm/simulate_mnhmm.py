"""
@Author  : Yapeng Wei 卫亚鹏
@File    : simulate_mnhmm.py
@Time    : 2026-05-25 02:04
@Desc    : Simulate Mixture Non-homogeneous Hidden Markov Model (MNHMM) sequences for Sequenzo

Simulation for mixture non-homogeneous hidden Markov models.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .mnhmm import (
    MNHMM,
    _as_state_list,
    _validate_probability_list,
    _validate_probability_vector,
)


def _coefs_value(coefs: Optional[Dict], key: str):
    if coefs is None:
        return None
    return coefs.get(key)


def _default_state_names(n_states: Sequence[int]) -> list[list[str]]:
    return [[f"State {idx + 1}" for idx in range(n_states_k)] for n_states_k in n_states]


def _draw_probability_parameters(
    rng: np.random.Generator,
    n_states: Sequence[int],
    n_symbols: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    initial_probs = []
    transition_probs = []
    emission_probs = []
    for n_states_k in n_states:
        initial_probs.append(rng.dirichlet(np.ones(n_states_k)))
        transition_probs.append(rng.dirichlet(np.ones(n_states_k), size=n_states_k))
        emission_probs.append(rng.dirichlet(np.ones(n_symbols), size=n_states_k))
    return initial_probs, transition_probs, emission_probs


def _simulate_from_probabilities(
    rng: np.random.Generator,
    sequence_lengths: Sequence[int],
    n_clusters: int,
    n_states: Sequence[int],
    initial_probs: Sequence[np.ndarray],
    transition_probs: Sequence[np.ndarray],
    emission_probs: Sequence[np.ndarray],
    cluster_probs: np.ndarray,
    alphabet: Sequence[str],
    state_names: Sequence[Sequence[str]],
    cluster_names: Sequence[str],
) -> dict:
    observations = []
    states = []
    clusters = []

    for seq_length in sequence_lengths:
        seq_idx = len(clusters)
        if np.ndim(cluster_probs) == 2:
            cluster_prob_row = np.asarray(cluster_probs[seq_idx], dtype=float)
        else:
            cluster_prob_row = np.asarray(cluster_probs, dtype=float)
        cluster_idx = int(rng.choice(n_clusters, p=cluster_prob_row))
        clusters.append(cluster_names[cluster_idx])

        initial = initial_probs[cluster_idx]
        transition = transition_probs[cluster_idx]
        emission = emission_probs[cluster_idx]
        n_states_k = n_states[cluster_idx]

        current_state = int(rng.choice(n_states_k, p=initial))
        seq_states = [f"{cluster_names[cluster_idx]}: {state_names[cluster_idx][current_state]}"]
        obs_idx = int(rng.choice(len(alphabet), p=emission[current_state]))
        seq_obs = [alphabet[obs_idx]]

        for _ in range(1, int(seq_length)):
            current_state = int(rng.choice(n_states_k, p=transition[current_state]))
            seq_states.append(f"{cluster_names[cluster_idx]}: {state_names[cluster_idx][current_state]}")
            obs_idx = int(rng.choice(len(alphabet), p=emission[current_state]))
            seq_obs.append(alphabet[obs_idx])

        observations.append(seq_obs)
        states.append(seq_states)

    max_length = max(int(length) for length in sequence_lengths)
    obs_dict = {}
    for time_idx in range(max_length):
        obs_dict[f"time_{time_idx + 1}"] = [
            seq[time_idx] if time_idx < len(seq) else None for seq in observations
        ]
    observations_df = pd.DataFrame(obs_dict)
    observations_df["cluster"] = clusters

    state_rows = []
    for seq_idx, seq_states in enumerate(states):
        for time_idx, state in enumerate(seq_states):
            state_rows.append({"id": seq_idx, "time": time_idx + 1, "state": state})

    return {
        "observations": observations,
        "states": states,
        "clusters": clusters,
        "observations_df": observations_df,
        "states_df": pd.DataFrame(state_rows),
        "alphabet": list(alphabet),
        "state_names": [list(names) for names in state_names],
        "cluster_names": list(cluster_names),
        "model": {
            "n_clusters": n_clusters,
            "n_states": list(n_states),
            "initial_probs": [np.array(value, copy=True) for value in initial_probs],
            "transition_probs": [np.array(value, copy=True) for value in transition_probs],
            "emission_probs": [np.array(value, copy=True) for value in emission_probs],
            "cluster_probs": np.array(cluster_probs, copy=True),
        },
    }


def _validate_multichannel_emission_probs(
    emission_probs: Sequence[Sequence[np.ndarray]],
    n_clusters: int,
    n_states: Sequence[int],
    alphabets: Sequence[Sequence[str]],
) -> list[list[np.ndarray]]:
    if len(emission_probs) != n_clusters:
        raise ValueError("emission_probs length must equal n_clusters")
    validated = []
    for cluster_idx in range(n_clusters):
        cluster_values = emission_probs[cluster_idx]
        if len(cluster_values) != len(alphabets):
            raise ValueError(
                f"emission_probs[{cluster_idx}] length must equal n_channels"
            )
        cluster_validated = []
        for channel_idx, alphabet in enumerate(alphabets):
            arr = np.asarray(cluster_values[channel_idx], dtype=float)
            expected = (int(n_states[cluster_idx]), len(alphabet))
            if arr.shape != expected:
                raise ValueError(
                    f"emission_probs[{cluster_idx}][{channel_idx}] must have "
                    f"shape {expected}, got {arr.shape}"
                )
            rows = [
                _validate_probability_vector(row, expected[1], "emission_probs")
                for row in arr
            ]
            cluster_validated.append(np.vstack(rows))
        validated.append(cluster_validated)
    return validated


def _simulate_multichannel_from_probabilities(
    rng: np.random.Generator,
    sequence_lengths: Sequence[int],
    n_clusters: int,
    n_states: Sequence[int],
    initial_probs: Sequence[np.ndarray],
    transition_probs: Sequence[np.ndarray],
    emission_probs: Sequence[Sequence[np.ndarray]],
    cluster_probs: np.ndarray,
    alphabets: Sequence[Sequence[str]],
    channel_names: Sequence[str],
    state_names: Sequence[Sequence[str]],
    cluster_names: Sequence[str],
) -> dict:
    observations = {name: [] for name in channel_names}
    states = []
    clusters = []

    for seq_idx, seq_length in enumerate(sequence_lengths):
        if np.ndim(cluster_probs) == 2:
            cluster_prob_row = np.asarray(cluster_probs[seq_idx], dtype=float)
        else:
            cluster_prob_row = np.asarray(cluster_probs, dtype=float)
        cluster_idx = int(rng.choice(n_clusters, p=cluster_prob_row))
        clusters.append(cluster_names[cluster_idx])

        initial = initial_probs[cluster_idx]
        transition = transition_probs[cluster_idx]
        emissions = emission_probs[cluster_idx]
        n_states_k = n_states[cluster_idx]

        current_state = int(rng.choice(n_states_k, p=initial))
        seq_states = [f"{cluster_names[cluster_idx]}: {state_names[cluster_idx][current_state]}"]
        channel_sequences = {name: [] for name in channel_names}

        for time_idx in range(int(seq_length)):
            if time_idx > 0:
                current_state = int(rng.choice(n_states_k, p=transition[current_state]))
                seq_states.append(
                    f"{cluster_names[cluster_idx]}: {state_names[cluster_idx][current_state]}"
                )
            for channel_idx, channel_name in enumerate(channel_names):
                obs_idx = int(
                    rng.choice(
                        len(alphabets[channel_idx]),
                        p=emissions[channel_idx][current_state],
                    )
                )
                channel_sequences[channel_name].append(alphabets[channel_idx][obs_idx])

        for channel_name in channel_names:
            observations[channel_name].append(channel_sequences[channel_name])
        states.append(seq_states)

    max_length = max(int(length) for length in sequence_lengths)
    observations_df = {}
    for channel_name in channel_names:
        obs_dict = {}
        for time_idx in range(max_length):
            obs_dict[f"time_{time_idx + 1}"] = [
                seq[time_idx] if time_idx < len(seq) else None
                for seq in observations[channel_name]
            ]
        frame = pd.DataFrame(obs_dict)
        frame["cluster"] = clusters
        observations_df[channel_name] = frame

    state_rows = []
    for seq_idx, seq_states in enumerate(states):
        for time_idx, state in enumerate(seq_states):
            state_rows.append({"id": seq_idx, "time": time_idx + 1, "state": state})

    return {
        "observations": observations,
        "observations_by_channel": observations,
        "observations_list": [observations[name] for name in channel_names],
        "states": states,
        "clusters": clusters,
        "observations_df": observations_df,
        "states_df": pd.DataFrame(state_rows),
        "alphabets": {name: list(alphabet) for name, alphabet in zip(channel_names, alphabets)},
        "channel_names": list(channel_names),
        "state_names": [list(names) for names in state_names],
        "cluster_names": list(cluster_names),
        "model": {
            "n_clusters": n_clusters,
            "n_channels": len(channel_names),
            "n_states": list(n_states),
            "initial_probs": [np.array(value, copy=True) for value in initial_probs],
            "transition_probs": [np.array(value, copy=True) for value in transition_probs],
            "emission_probs": [
                [np.array(channel_value, copy=True) for channel_value in cluster_value]
                for cluster_value in emission_probs
            ],
            "cluster_probs": np.array(cluster_probs, copy=True),
        },
    }


def simulate_mnhmm(
    n_sequences: Optional[int] = None,
    n_clusters: Optional[int] = None,
    n_states: Union[int, Sequence[int], None] = None,
    initial_probs: Optional[Sequence[np.ndarray]] = None,
    transition_probs: Optional[Sequence[np.ndarray]] = None,
    emission_probs: Optional[Sequence[np.ndarray]] = None,
    cluster_probs: Optional[np.ndarray] = None,
    sequence_length: Optional[int] = None,
    sequence_lengths: Optional[Sequence[int]] = None,
    alphabet: Optional[Sequence[str]] = None,
    n_symbols: Optional[int] = None,
    state_names: Optional[Sequence[Sequence[str]]] = None,
    cluster_names: Optional[Sequence[str]] = None,
    coefs: Optional[Dict] = None,
    model: Optional[MNHMM] = None,
    random_state: Optional[int] = None,
) -> dict:
    """
    Simulate observed and hidden sequences from an MNHMM.

    If no probabilities are supplied, random categorical parameters are drawn.
    """
    rng = np.random.default_rng(random_state)

    if model is not None:
        if not isinstance(model, MNHMM):
            raise TypeError("model must be an MNHMM instance")
        if model.initial_probs is None or model.transition_probs is None or model.emission_probs is None:
            raise NotImplementedError(
                "simulate_mnhmm(model=...) currently requires fixed component probabilities"
            )
        n_clusters = model.n_clusters
        n_states_list = model.n_states
        sequence_lengths = model.sequence_lengths
        alphabet = model.alphabet
        state_names = model.state_names
        cluster_names = model.cluster_names
        initial_probs = model.initial_probs
        transition_probs = model.transition_probs
        emission_probs = model.emission_probs
        cluster_probs = (
            model.cluster_probs
            if model.cluster_probs is not None
            else model.compute_cluster_probs()
        )
        if model.n_channels > 1:
            expected_initial = [(n_states_k,) for n_states_k in n_states_list]
            expected_transition = [(n_states_k, n_states_k) for n_states_k in n_states_list]
            initial_probs = _validate_probability_list(
                initial_probs, n_clusters, expected_initial, "initial_probs"
            )
            transition_probs = _validate_probability_list(
                transition_probs, n_clusters, expected_transition, "transition_probs"
            )
            emission_probs = _validate_multichannel_emission_probs(
                emission_probs,
                n_clusters,
                n_states_list,
                model.alphabets,
            )
            if np.ndim(cluster_probs) == 2:
                cluster_probs = np.asarray(cluster_probs, dtype=float)
                if cluster_probs.shape != (len(sequence_lengths), n_clusters):
                    raise ValueError(
                        "cluster_probs matrix must have shape (n_sequences, n_clusters)"
                    )
                for row_idx in range(cluster_probs.shape[0]):
                    cluster_probs[row_idx] = _validate_probability_vector(
                        cluster_probs[row_idx],
                        n_clusters,
                        f"cluster_probs[{row_idx}]",
                    )
            else:
                cluster_probs = _validate_probability_vector(
                    cluster_probs,
                    n_clusters,
                    "cluster_probs",
                )
            return _simulate_multichannel_from_probabilities(
                rng=rng,
                sequence_lengths=sequence_lengths,
                n_clusters=n_clusters,
                n_states=n_states_list,
                initial_probs=initial_probs,
                transition_probs=transition_probs,
                emission_probs=emission_probs,
                cluster_probs=cluster_probs,
                alphabets=model.alphabets,
                channel_names=model.channel_names,
                state_names=state_names,
                cluster_names=cluster_names,
            )
    else:
        if n_clusters is None:
            raise ValueError("n_clusters must be provided")
        n_clusters = int(n_clusters)
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if n_states is None:
            raise ValueError("n_states must be provided")
        n_states_list = _as_state_list(n_states, n_clusters)

        if sequence_lengths is None:
            if n_sequences is None or sequence_length is None:
                raise ValueError("Provide either sequence_lengths or both n_sequences and sequence_length")
            if int(sequence_length) < 1:
                raise ValueError("sequence_length must be at least 1")
            sequence_lengths = [int(sequence_length)] * int(n_sequences)
        else:
            sequence_lengths = [int(value) for value in sequence_lengths]
            if any(value < 1 for value in sequence_lengths):
                raise ValueError("sequence_lengths values must be at least 1")

        if alphabet is None:
            if n_symbols is None:
                n_symbols = 2
            alphabet = [str(idx) for idx in range(int(n_symbols))]
        else:
            alphabet = [str(value) for value in alphabet]
            n_symbols = len(alphabet)

        if initial_probs is None:
            initial_probs = _coefs_value(coefs, "initial_probs")
        if transition_probs is None:
            transition_probs = _coefs_value(coefs, "transition_probs")
        if emission_probs is None:
            emission_probs = _coefs_value(coefs, "emission_probs")
        if cluster_probs is None:
            cluster_probs = _coefs_value(coefs, "cluster_probs")

        if initial_probs is None or transition_probs is None or emission_probs is None:
            initial_probs, transition_probs, emission_probs = _draw_probability_parameters(
                rng, n_states_list, int(n_symbols)
            )

        if cluster_probs is None:
            cluster_probs = np.ones(n_clusters, dtype=float) / n_clusters

        if cluster_names is None:
            cluster_names = [f"Cluster {idx + 1}" for idx in range(n_clusters)]
        else:
            cluster_names = [str(value) for value in cluster_names]
        if len(cluster_names) != n_clusters:
            raise ValueError("cluster_names length must equal n_clusters")

        if state_names is None:
            state_names = _default_state_names(n_states_list)
        else:
            state_names = [[str(value) for value in names] for names in state_names]

    expected_initial = [(n_states_k,) for n_states_k in n_states_list]
    expected_transition = [(n_states_k, n_states_k) for n_states_k in n_states_list]
    expected_emission = [(n_states_k, len(alphabet)) for n_states_k in n_states_list]

    initial_probs = _validate_probability_list(
        initial_probs, n_clusters, expected_initial, "initial_probs"
    )
    transition_probs = _validate_probability_list(
        transition_probs, n_clusters, expected_transition, "transition_probs"
    )
    emission_probs = _validate_probability_list(
        emission_probs, n_clusters, expected_emission, "emission_probs"
    )
    if np.ndim(cluster_probs) == 2:
        cluster_probs = np.asarray(cluster_probs, dtype=float)
        if cluster_probs.shape != (len(sequence_lengths), n_clusters):
            raise ValueError(
                "cluster_probs matrix must have shape (n_sequences, n_clusters)"
            )
        for row_idx in range(cluster_probs.shape[0]):
            cluster_probs[row_idx] = _validate_probability_vector(
                cluster_probs[row_idx],
                n_clusters,
                f"cluster_probs[{row_idx}]",
            )
    else:
        cluster_probs = _validate_probability_vector(cluster_probs, n_clusters, "cluster_probs")

    if len(state_names) != n_clusters:
        raise ValueError("state_names length must equal n_clusters")
    for cluster_idx, names in enumerate(state_names):
        if len(names) != n_states_list[cluster_idx]:
            raise ValueError(f"state_names[{cluster_idx}] length must equal n_states[{cluster_idx}]")

    return _simulate_from_probabilities(
        rng=rng,
        sequence_lengths=sequence_lengths,
        n_clusters=n_clusters,
        n_states=n_states_list,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=cluster_probs,
        alphabet=alphabet,
        state_names=state_names,
        cluster_names=cluster_names,
    )
