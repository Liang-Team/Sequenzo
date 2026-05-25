"""
Small seqHMM-compatibility utilities for fitted and parameterized models.

The functions in this module intentionally keep Python semantics simple: they
return defensive copies by default and do not mutate models unless requested.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from sequenzo.define_sequence_data import SequenceData
from .hmm import HMM
from .mhmm import MHMM
from .mnhmm import MNHMM
from .multichannel_utils import multichannel_to_hmmlearn_format, prepare_multichannel_data
from .nhmm import NHMM


Model = Union[HMM, MHMM, MNHMM, NHMM]


def _copy(value, copy: bool = True):
    if isinstance(value, list):
        return [_copy(item, copy=copy) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item, copy=copy) for item in value)
    if isinstance(value, dict):
        return {key: _copy(item, copy=copy) for key, item in value.items()}
    return np.array(value, copy=copy)


def _normalize_observations_input(observations):
    return list(observations) if isinstance(observations, tuple) else observations


def get_initial_probs(
    model: Model,
    cluster: Optional[int] = None,
    copy: bool = True,
):
    """Return initial state probabilities, matching seqHMM's getter role."""
    if isinstance(model, MHMM):
        if cluster is None:
            return [
                get_initial_probs(component, copy=copy)
                for component in model.clusters
            ]
        return get_initial_probs(model.clusters[cluster], copy=copy)
    if isinstance(model, MNHMM):
        if cluster is None:
            return [
                get_initial_probs(model, cluster=i, copy=copy)
                for i in range(model.n_clusters)
            ]
        initial_probs, _, _ = model._component_probs(cluster)
        return _copy(initial_probs, copy=copy)
    if isinstance(model, NHMM):
        initial_probs, _, _ = model._compute_probs()
        return _copy(initial_probs, copy=copy)
    if isinstance(model, HMM):
        return _copy(model.initial_probs, copy=copy)
    raise TypeError(
        "get_initial_probs expects an HMM, MHMM, NHMM, or MNHMM model"
    )


def get_transition_probs(
    model: Model,
    cluster: Optional[int] = None,
    copy: bool = True,
):
    """Return transition probabilities, preserving model-specific shapes."""
    if isinstance(model, MHMM):
        if cluster is None:
            return [
                get_transition_probs(component, copy=copy)
                for component in model.clusters
            ]
        return get_transition_probs(model.clusters[cluster], copy=copy)
    if isinstance(model, MNHMM):
        if cluster is None:
            return [
                get_transition_probs(model, cluster=i, copy=copy)
                for i in range(model.n_clusters)
            ]
        _, transition_probs, _ = model._component_probs(cluster)
        return _copy(transition_probs, copy=copy)
    if isinstance(model, NHMM):
        _, transition_probs, _ = model._compute_probs()
        return _copy(transition_probs, copy=copy)
    if isinstance(model, HMM):
        return _copy(model.transition_probs, copy=copy)
    raise TypeError(
        "get_transition_probs expects an HMM, MHMM, NHMM, or MNHMM model"
    )


def get_emission_probs(
    model: Model,
    cluster: Optional[int] = None,
    copy: bool = True,
):
    """Return emission probabilities, including one matrix per channel."""
    if isinstance(model, MHMM):
        if cluster is None:
            return [
                get_emission_probs(component, copy=copy)
                for component in model.clusters
            ]
        return get_emission_probs(model.clusters[cluster], copy=copy)
    if isinstance(model, MNHMM):
        if cluster is None:
            return [
                get_emission_probs(model, cluster=i, copy=copy)
                for i in range(model.n_clusters)
            ]
        _, _, emission_probs = model._component_probs(cluster)
        return _copy(emission_probs, copy=copy)
    if isinstance(model, NHMM):
        _, _, emission_probs = model._compute_probs()
        return _copy(emission_probs, copy=copy)
    if isinstance(model, HMM):
        return _copy(model.emission_probs, copy=copy)
    raise TypeError(
        "get_emission_probs expects an HMM, MHMM, NHMM, or MNHMM model"
    )


def _primary_sequence_data(model, newdata=None):
    sequences = newdata
    if sequences is None:
        if isinstance(model, MHMM):
            sequences = model._default_sequences()
        elif isinstance(model, MNHMM):
            sequences = model.observations
        elif getattr(model, "n_channels", 1) > 1:
            sequences = model.channels
        else:
            sequences = model.observations
    if isinstance(sequences, (list, tuple)):
        return sequences[0]
    return sequences


def _sequence_ids_and_times(seq_data: SequenceData, lengths: Sequence[int]):
    ids = list(seq_data.ids)
    if len(ids) != len(lengths):
        ids = list(range(len(lengths)))
    times = list(seq_data.time)
    numeric_times = pd.to_numeric(pd.Index(times), errors="coerce")
    if numeric_times.notna().all():
        times = [
            int(value) if float(value).is_integer() else float(value)
            for value in numeric_times
        ]
    else:
        stripped = (
            pd.Index(times).astype(str).str.replace(r"\D+", "", regex=True)
        )
        stripped_numeric = pd.to_numeric(stripped, errors="coerce")
        if stripped_numeric.notna().all():
            times = [
                int(value) if float(value).is_integer() else float(value)
                for value in stripped_numeric
            ]
        else:
            times = list(range(1, max(lengths) + 1))
    return ids, times


def _flat_paths_to_dataframe(
    model,
    paths: np.ndarray,
    lengths: Sequence[int],
    seq_data: SequenceData,
    log_prob: Optional[Sequence[float]] = None,
    clusters: Optional[Sequence[str]] = None,
    state_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    ids, times = _sequence_ids_and_times(seq_data, lengths)
    state_names = list(getattr(model, "state_names", []))
    lengths = np.asarray(lengths, dtype=int)
    row_ids = np.repeat(np.asarray(ids, dtype=object), lengths)
    row_times = np.concatenate(
        [
            np.asarray(
                (
                    times[:seq_length]
                    if seq_length <= len(times)
                    else range(1, seq_length + 1)
                ),
                dtype=object,
            )
            for seq_length in lengths
        ]
    )
    if state_labels is not None:
        row_states = np.asarray(state_labels, dtype=object)
    elif state_names:
        row_states = np.asarray(state_names, dtype=object)[
            paths.astype(int, copy=False)
        ]
    else:
        row_states = paths.astype(int, copy=False)
    data = {"id": row_ids, "time": row_times, "state": row_states}
    if clusters is not None:
        data["cluster"] = np.repeat(
            np.asarray(clusters, dtype=object), lengths
        )
    out = pd.DataFrame(data)
    numeric_time = pd.to_numeric(out["time"], errors="coerce")
    if numeric_time.notna().all():
        out["time"] = (
            numeric_time.astype(int)
            if np.all(np.equal(np.mod(numeric_time, 1), 0))
            else numeric_time
        )
    if log_prob is not None:
        out.attrs["log_prob"] = np.asarray(log_prob, dtype=float)
    return out


def _hmm_lengths(model: HMM, newdata=None):
    if model.n_channels > 1:
        _, lengths = model._multichannel_arrays(newdata)
    else:
        from .utils import sequence_data_to_hmmlearn_format

        seq = newdata if newdata is not None else model.observations
        _, lengths = sequence_data_to_hmmlearn_format(seq)
    return lengths


def _hmm_path_log_prob(
    model: HMM, paths: np.ndarray, newdata=None
) -> np.ndarray:
    eps = np.finfo(float).tiny
    initial = np.log(
        np.maximum(np.asarray(model.initial_probs, dtype=float), eps)
    )
    transition = np.log(
        np.maximum(np.asarray(model.transition_probs, dtype=float), eps)
    )
    lengths = _hmm_lengths(model, newdata)
    out = []
    offset = 0

    if model.n_channels > 1:
        observations, _ = model._multichannel_arrays(newdata)
        for seq_length in lengths:
            end = offset + int(seq_length)
            path = paths[offset:end].astype(int, copy=False)
            obs_list = [obs[offset:end] for obs in observations]
            emission = np.log(
                np.maximum(model._multichannel_emission_matrix(obs_list), eps)
            )
            value = initial[path[0]] + emission[0, path[0]]
            for t in range(1, len(path)):
                value += (
                    transition[path[t - 1], path[t]] + emission[t, path[t]]
                )
            out.append(float(value))
            offset = end
        return np.asarray(out, dtype=float)

    from .utils import sequence_data_to_hmmlearn_format

    seq = newdata if newdata is not None else model.observations
    X, _ = sequence_data_to_hmmlearn_format(seq)
    obs = X[:, 0].astype(int, copy=False)
    emission = np.log(
        np.maximum(np.asarray(model.emission_probs, dtype=float), eps)
    )
    for seq_length in lengths:
        end = offset + int(seq_length)
        path = paths[offset:end].astype(int, copy=False)
        obs_seq = obs[offset:end]
        value = initial[path[0]] + emission[path[0], obs_seq[0]]
        for t in range(1, len(path)):
            value += (
                transition[path[t - 1], path[t]]
                + emission[path[t], obs_seq[t]]
            )
        out.append(float(value))
        offset = end
    return np.asarray(out, dtype=float)


def _viterbi_timevarying(
    initial_probs: np.ndarray,
    transition_probs: np.ndarray,
    emission_probs: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, float]:
    eps = np.finfo(float).tiny
    initial = np.log(np.maximum(initial_probs, eps))
    transition = np.log(np.maximum(transition_probs, eps))
    emission = np.log(np.maximum(emission_probs, eps))
    T = len(observations)
    S = len(initial_probs)
    delta = np.empty((T, S), dtype=float)
    psi = np.zeros((T, S), dtype=int)
    delta[0] = initial + emission[0, :, observations[0]]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + transition[t]
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = (
            scores[psi[t], np.arange(S)] + emission[t, :, observations[t]]
        )
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, float(delta[-1, path[-1]])


def _hidden_paths_hmm(model: HMM, newdata=None):
    paths = model.predict(newdata)
    lengths = _hmm_lengths(model, newdata)
    return (
        paths,
        lengths,
        _hmm_path_log_prob(model, paths, newdata),
        None,
        None,
    )


def _hidden_paths_mhmm(model: MHMM, newdata=None):
    model._check_fixed_inference_ready()
    sequences = newdata if newdata is not None else model._default_sequences()
    log_likelihoods = model._sequence_log_likelihoods(sequences)
    log_joint = (
        log_likelihoods + np.log(model.cluster_probs + 1e-300)[np.newaxis, :]
    )
    cluster_indices = np.argmax(log_joint, axis=1)
    lengths = (
        model.clusters[0]._multichannel_arrays(sequences)[1]
        if model.n_channels > 1
        else None
    )
    if lengths is None:
        from .utils import sequence_data_to_hmmlearn_format

        _, lengths = sequence_data_to_hmmlearn_format(sequences)
    paths = []
    state_labels = []
    log_probs = []
    clusters = []
    starts = np.zeros(len(lengths) + 1, dtype=int)
    starts[1:] = np.cumsum(lengths)
    cluster_paths = [cluster.predict(sequences) for cluster in model.clusters]
    for seq_idx, cluster_idx in enumerate(cluster_indices):
        cluster = model.clusters[int(cluster_idx)]
        start = starts[seq_idx]
        end = starts[seq_idx + 1]
        sub_paths = cluster_paths[int(cluster_idx)][start:end]
        paths.append(sub_paths)
        state_labels.extend(
            cluster.state_names[int(state_idx)] for state_idx in sub_paths
        )
        log_probs.append(float(log_joint[seq_idx, cluster_idx]))
        clusters.append(model.cluster_names[int(cluster_idx)])
    return (
        np.concatenate(paths),
        lengths,
        np.asarray(log_probs),
        clusters,
        state_labels,
    )


def _hidden_paths_nhmm(model: NHMM, newdata=None):
    from .utils import sequence_data_to_hmmlearn_format

    seq = newdata if newdata is not None else model.observations
    X, lengths = sequence_data_to_hmmlearn_format(seq)
    initial, transition, emission = model._compute_probs()
    starts = np.zeros(len(lengths) + 1, dtype=int)
    starts[1:] = np.cumsum(lengths)
    max_sequence_length = int(np.max(lengths)) if len(lengths) else 0
    paths = []
    log_probs = []
    for seq_idx, seq_length in enumerate(lengths):
        obs = X[starts[seq_idx]:starts[seq_idx + 1], 0].astype(
            int, copy=False
        )
        path, log_prob = _viterbi_timevarying(
            initial[seq_idx],
            transition[seq_idx, :seq_length],
            emission[seq_idx, :seq_length],
            obs,
        )
        paths.append(path)
        log_probs.append(log_prob)
    return np.concatenate(paths), lengths, np.asarray(log_probs), None, None


def _hidden_paths_mnhmm(model: MNHMM, newdata=None):
    sequences = newdata if newdata is not None else model.observations
    if model.n_channels > 1:
        channels, _, _ = prepare_multichannel_data(
            _normalize_observations_input(sequences)
        )
        model._validate_observation_alphabets(channels)
        observations, lengths = multichannel_to_hmmlearn_format(channels)
        observations = [X[:, 0].astype(int, copy=False) for X in observations]
    else:
        from .utils import sequence_data_to_hmmlearn_format

        model._validate_observation_alphabets([sequences])
        X, lengths = sequence_data_to_hmmlearn_format(sequences)
        observations = [X[:, 0].astype(int, copy=False)]
    if model._uses_covariate_probabilities():
        alignment_channels = channels if model.n_channels > 1 else [sequences]
        model._validate_covariate_newdata_alignment(alignment_channels)
    cluster_priors = model._cluster_probs_for_n(len(lengths))
    starts = np.zeros(len(lengths) + 1, dtype=int)
    starts[1:] = np.cumsum(lengths)
    max_sequence_length = int(np.max(lengths)) if len(lengths) else 0
    paths = []
    state_labels = []
    log_probs = []
    clusters = []
    components_fixed = (
        model.initial_probs is not None
        and model.transition_probs is not None
        and model.emission_probs is not None
    )
    component_probs = []
    for cluster_idx in range(model.n_clusters):
        if components_fixed:
            initial = np.tile(
                model.initial_probs[cluster_idx],
                (len(lengths), 1),
            )
            transition = np.tile(
                model.transition_probs[cluster_idx],
                (len(lengths), max_sequence_length, 1, 1),
            )
            if model.n_channels == 1:
                emission = np.tile(
                    model.emission_probs[cluster_idx],
                    (len(lengths), max_sequence_length, 1, 1),
                )
            else:
                emission = [
                    np.tile(
                        channel_emission,
                        (len(lengths), max_sequence_length, 1, 1),
                    )
                    for channel_emission in model.emission_probs[cluster_idx]
                ]
            component_probs.append((initial, transition, emission))
        else:
            component_probs.append(model._component_probs(cluster_idx))

    for seq_idx, seq_length in enumerate(lengths):
        obs = [
            channel[starts[seq_idx]:starts[seq_idx + 1]]
            for channel in observations
        ]
        best_path = None
        best_log_prob = -np.inf
        best_cluster_idx = 0
        for cluster_idx in range(model.n_clusters):
            initial, transition, emission = component_probs[cluster_idx]

            if model.n_channels == 1:
                path, path_log_prob = _viterbi_timevarying(
                    initial[seq_idx],
                    transition[seq_idx, :seq_length],
                    emission[seq_idx, :seq_length],
                    obs[0],
                )
            else:
                combined = np.ones((int(seq_length), model.n_states[cluster_idx], 1))
                for channel_idx, channel_obs in enumerate(obs):
                    channel_emission = emission[channel_idx][seq_idx, :seq_length]
                    combined[:, :, 0] *= channel_emission[
                        np.arange(int(seq_length))[:, None],
                        np.arange(model.n_states[cluster_idx])[None, :],
                        channel_obs[:, None],
                    ]
                path, path_log_prob = _viterbi_timevarying(
                    initial[seq_idx],
                    transition[seq_idx, :seq_length],
                    combined,
                    np.zeros(int(seq_length), dtype=int),
                )
            joint_log_prob = path_log_prob + np.log(
                max(cluster_priors[seq_idx, cluster_idx], 1e-300)
            )
            if joint_log_prob > best_log_prob:
                best_path = path
                best_log_prob = joint_log_prob
                best_cluster_idx = cluster_idx
        cluster_idx = int(best_cluster_idx)
        path = best_path
        paths.append(path)
        names = (
            model.state_names[cluster_idx]
            if model.n_clusters > 1
            else model.state_names
        )
        state_labels.extend(names[int(state_idx)] for state_idx in path)
        log_probs.append(best_log_prob)
        clusters.append(model.cluster_names[cluster_idx])
    return (
        np.concatenate(paths),
        lengths,
        np.asarray(log_probs),
        clusters,
        state_labels,
    )


def hidden_paths(
    model: Model,
    newdata=None,
    *,
    output: str = "dataframe",
    as_stslist: bool = False,
):
    """Return Viterbi hidden paths in seqHMM-style table form by default."""
    if isinstance(model, MHMM):
        paths, lengths, log_prob, clusters, state_labels = _hidden_paths_mhmm(
            model, newdata
        )
    elif isinstance(model, MNHMM):
        paths, lengths, log_prob, clusters, state_labels = _hidden_paths_mnhmm(
            model, newdata
        )
    elif isinstance(model, NHMM):
        paths, lengths, log_prob, clusters, state_labels = _hidden_paths_nhmm(
            model, newdata
        )
    elif isinstance(model, HMM):
        paths, lengths, log_prob, clusters, state_labels = _hidden_paths_hmm(
            model, newdata
        )
    else:
        raise TypeError(
            "hidden_paths expects an HMM, MHMM, NHMM, or MNHMM model"
        )

    if output == "array" and not as_stslist:
        return paths
    if output not in {"dataframe", "array"}:
        raise ValueError("output must be 'dataframe' or 'array'")

    seq_data = _primary_sequence_data(model, newdata)
    out = _flat_paths_to_dataframe(
        model,
        paths,
        lengths,
        seq_data,
        log_prob,
        clusters,
        state_labels,
    )
    if as_stslist:
        return data_to_stslist(out, id="id", time="time", responses="state")
    return out


def _validate_permutation(
    permutation: Sequence[int], n_states: int
) -> List[int]:
    permutation = [int(state) for state in permutation]
    if sorted(permutation) != list(range(n_states)):
        raise ValueError(
            "permutation must contain each state index "
            f"0..{n_states - 1} exactly once"
        )
    return permutation


def _sync_hmmlearn_parameters(model: HMM) -> None:
    if model._hmm_model is None:
        return
    model._hmm_model.n_components = model.n_states
    model._hmm_model.startprob_ = np.array(model.initial_probs, copy=True)
    model._hmm_model.transmat_ = np.array(model.transition_probs, copy=True)
    model._hmm_model.emissionprob_ = np.array(model.emission_probs, copy=True)


def permute_states(
    model,
    permutation=None,
    copy: bool = True,
):
    """Permute HMM states or align seqHMM-style gamma coefficients."""
    if isinstance(model, Mapping):
        if permutation is None or not isinstance(permutation, Mapping):
            raise TypeError("reference gamma coefficients must be provided")
        return _permute_gamma_coefficients(model, permutation, copy=copy)
    if not isinstance(model, HMM):
        raise TypeError(
            "explicit permutation currently supports HMM objects only"
        )

    target = deepcopy(model) if copy else model
    permutation = _validate_permutation(permutation, target.n_states)

    target.initial_probs = np.asarray(target.initial_probs)[permutation]
    target.transition_probs = np.asarray(target.transition_probs)[
        np.ix_(permutation, permutation)
    ]
    if isinstance(target.emission_probs, list):
        target.emission_probs = [
            np.asarray(emission)[permutation]
            for emission in target.emission_probs
        ]
    else:
        target.emission_probs = np.asarray(target.emission_probs)[permutation]
    target.state_names = [target.state_names[i] for i in permutation]
    _sync_hmmlearn_parameters(target)
    return target


class PermutedStateCoefficients(dict):
    """Dictionary with the recovered state permutation attached."""

    def __init__(self, *args, permutation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.permutation = (
            list(permutation) if permutation is not None else None
        )


def _state_alignment_cost(
    estimates: Mapping, reference: Mapping
) -> np.ndarray:
    gamma_pi_est = np.asarray(estimates["gamma_pi"], dtype=float)
    gamma_pi_ref = np.asarray(reference["gamma_pi"], dtype=float)
    n_states = gamma_pi_ref.shape[0]
    cost = np.zeros((n_states, n_states), dtype=float)
    gamma_A_est = np.asarray(estimates["gamma_A"], dtype=float)
    gamma_A_ref = np.asarray(reference["gamma_A"], dtype=float)
    gamma_B_est = estimates.get("gamma_B", [])
    gamma_B_ref = reference.get("gamma_B", [])
    for i in range(n_states):
        for j in range(n_states):
            est_parts = [
                gamma_pi_est[i].ravel(order="F"),
                gamma_A_est[:, :, i].ravel(order="F"),
            ]
            ref_parts = [
                gamma_pi_ref[j].ravel(order="F"),
                gamma_A_ref[:, :, j].ravel(order="F"),
            ]
            for est_B, ref_B in zip(gamma_B_est, gamma_B_ref):
                est_parts.append(np.asarray(est_B)[:, :, i].ravel(order="F"))
                ref_parts.append(np.asarray(ref_B)[:, :, j].ravel(order="F"))
            est_vec = np.concatenate(est_parts)
            ref_vec = np.concatenate(ref_parts)
            cost[i, j] = float(np.mean((est_vec - ref_vec) ** 2))
    return cost


def _permute_gamma_coefficients(
    estimates: Mapping,
    reference: Mapping,
    copy: bool = True,
) -> PermutedStateCoefficients:
    cost = _state_alignment_cost(estimates, reference)
    row_ind, col_ind = linear_sum_assignment(cost)
    permutation = [
        int(row_ind[np.where(col_ind == ref_idx)[0][0]])
        for ref_idx in range(len(col_ind))
    ]
    gamma_A = np.asarray(estimates["gamma_A"])
    cov_axis = list(range(gamma_A.shape[1]))
    out = PermutedStateCoefficients(permutation=permutation)
    out["gamma_pi"] = np.array(estimates["gamma_pi"], copy=copy)[permutation]
    out["gamma_A"] = np.array(gamma_A, copy=copy)[
        np.ix_(permutation, cov_axis, permutation)
    ]
    out["gamma_B"] = [
        np.array(item, copy=copy)[:, :, permutation]
        for item in estimates.get("gamma_B", [])
    ]
    return out


def trim_model(
    model: Union[HMM, MHMM],
    maxit: int = 0,
    return_loglik: bool = False,
    zerotol: float = 1e-8,
    verbose: bool = False,
    copy: bool = True,
) -> Union[HMM, MHMM, dict]:
    """Set tiny positive probabilities to zero without reducing likelihood."""
    if not isinstance(model, (HMM, MHMM)):
        raise TypeError("trim_model expects an HMM or MHMM model")
    target = deepcopy(model) if copy else model
    original = deepcopy(model)
    ll_original = _model_log_likelihood(model)

    changed = _trim_probabilities_in_place(target, zerotol)
    if not changed:
        result = target
        loglik = _model_log_likelihood(result)
    else:
        loglik = _model_log_likelihood(target)
        if not np.isfinite(loglik) or loglik < ll_original - 1e-8:
            result = original
            loglik = ll_original
        else:
            result = target
            if maxit > 0:
                result = _trim_refit_loop(result, maxit, zerotol, verbose)
                loglik = _model_log_likelihood(result)
    if return_loglik:
        return {"model": result, "loglik": loglik}
    return result


def _model_log_likelihood(model) -> float:
    if isinstance(model, MHMM):
        log_likelihoods = model._sequence_log_likelihoods()
        from scipy.special import logsumexp

        return float(
            np.sum(
                logsumexp(
                    log_likelihoods + np.log(model.cluster_probs + 1e-300),
                    axis=1,
                )
            )
        )
    return float(model.score())


def _trim_refit_loop(model, maxit: int, zerotol: float, verbose: bool):
    best = deepcopy(model)
    best_ll = _model_log_likelihood(best)
    for _ in range(int(maxit)):
        candidate = deepcopy(best)
        if hasattr(candidate, "fit"):
            candidate.fit(n_iter=1, verbose=verbose)
        _trim_probabilities_in_place(candidate, zerotol)
        ll = _model_log_likelihood(candidate)
        if not np.isfinite(ll) or ll <= best_ll + 1e-12:
            break
        best, best_ll = candidate, ll
    return best


def _zero_and_normalize_vector(
    values: np.ndarray, zerotol: float
) -> tuple[np.ndarray, bool, bool]:
    out = np.asarray(values, dtype=float).copy()
    mask = (out < zerotol) & (out > 0.0)
    changed = bool(np.any(mask))
    out[mask] = 0.0
    total = out.sum()
    if not np.isfinite(total) or total <= 0.0:
        return out, changed, False
    return out / total, changed, True


def _zero_and_normalize_rows(
    values: np.ndarray, zerotol: float
) -> tuple[np.ndarray, bool, bool]:
    out = np.asarray(values, dtype=float).copy()
    mask = (out < zerotol) & (out > 0.0)
    changed = bool(np.any(mask))
    out[mask] = 0.0
    row_sums = out.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        return out, changed, False
    return out / row_sums, changed, True


def _trim_hmm_probabilities_in_place(model: HMM, zerotol: float) -> bool:
    changed_any = False
    initial, changed, ok = _zero_and_normalize_vector(
        model.initial_probs, zerotol
    )
    if not ok:
        raise ValueError(
            "trimming initial probabilities produced an invalid row"
        )
    model.initial_probs = initial
    changed_any = changed_any or changed

    transition, changed, ok = _zero_and_normalize_rows(
        model.transition_probs, zerotol
    )
    if not ok:
        raise ValueError(
            "trimming transition probabilities produced an invalid row"
        )
    model.transition_probs = transition
    changed_any = changed_any or changed

    if isinstance(model.emission_probs, list):
        emissions = []
        for emission in model.emission_probs:
            trimmed, changed, ok = _zero_and_normalize_rows(emission, zerotol)
            if not ok:
                raise ValueError(
                    "trimming emission probabilities produced an invalid row"
                )
            emissions.append(trimmed)
            changed_any = changed_any or changed
        model.emission_probs = emissions
    else:
        emission, changed, ok = _zero_and_normalize_rows(
            model.emission_probs, zerotol
        )
        if not ok:
            raise ValueError(
                "trimming emission probabilities produced an invalid row"
            )
        model.emission_probs = emission
        changed_any = changed_any or changed
    _sync_hmmlearn_parameters(model)
    return changed_any


def _trim_probabilities_in_place(
    model: Union[HMM, MHMM], zerotol: float
) -> bool:
    if isinstance(model, MHMM):
        changed_any = False
        for cluster in model.clusters:
            changed_any = (
                _trim_hmm_probabilities_in_place(cluster, zerotol)
                or changed_any
            )
        return changed_any
    return _trim_hmm_probabilities_in_place(model, zerotol)


def separate_mhmm(model: MHMM, copy: bool = True) -> List[HMM]:
    """Return the component HMMs from a mixture model."""
    if not isinstance(model, MHMM):
        raise TypeError("separate_mhmm expects an MHMM model")
    if copy:
        return [deepcopy(component) for component in model.clusters]
    return list(model.clusters)


def data_to_stslist(
    data: pd.DataFrame,
    time=None,
    responses=None,
    id: Optional[str] = None,
    states: Optional[List[str]] = None,
    id_col: Optional[str] = None,
    **kwargs,
):
    """Convert long seqHMM-style data or wide data to SequenceData objects."""
    if responses is None:
        return SequenceData(
            data, time=time, states=states, id_col=id_col, **kwargs
        )

    if time is None or id is None:
        raise ValueError(
            "id, time, and responses are required for long-format conversion"
        )
    response_names = (
        [responses] if isinstance(responses, str) else list(responses)
    )
    out = {}
    for response in response_names:
        subset = data[[id, time, response]].copy()
        id_order = pd.unique(subset[id])
        time_order = _ordered_time_values(subset[time])
        subset[id] = pd.Categorical(
            subset[id], categories=id_order, ordered=True
        )
        wide = subset.pivot(index=id, columns=time, values=response).reindex(
            index=id_order,
            columns=time_order,
        )
        time_cols = list(wide.columns)
        wide = wide.reset_index()
        response_states = states
        if response_states is None:
            response_states = sorted(
                pd.Series(data[response].dropna().unique()).tolist()
            )
        out[response] = SequenceData(
            wide,
            time=time_cols,
            states=list(response_states),
            id_col=id,
            **kwargs,
        )
    return out[response_names[0]] if isinstance(responses, str) else out


def _ordered_time_values(values) -> list:
    unique = pd.Index(pd.unique(values))
    numeric = pd.to_numeric(unique, errors="coerce")
    if numeric.notna().all():
        order = np.argsort(numeric.to_numpy(dtype=float), kind="stable")
        return unique.take(order).tolist()

    text = unique.astype(str)
    stripped = text.str.replace(r"\D+", "", regex=True)
    stripped_numeric = pd.to_numeric(stripped, errors="coerce")
    if stripped_numeric.notna().all():
        order = np.argsort(
            stripped_numeric.to_numpy(dtype=float), kind="stable"
        )
        return unique.take(order).tolist()
    return unique.tolist()


def stslist_to_data(
    seq_data,
    id: str = "id",
    time: str = "time",
    responses=None,
    id_col: Optional[str] = None,
    use_labels: bool = False,
    wide: bool = False,
) -> pd.DataFrame:
    """Convert SequenceData objects to long data by default, or wide data."""
    if isinstance(seq_data, Mapping):
        response_names = (
            list(seq_data.keys())
            if responses is None
            else (
                [responses] if isinstance(responses, str) else list(responses)
            )
        )
        pieces = [
            stslist_to_data(
                seq_data[name],
                id=id,
                time=time,
                responses=name,
                use_labels=use_labels,
            )
            for name in response_names
        ]
        out = pieces[0]
        for piece in pieces[1:]:
            out = out.merge(piece, on=[id, time], how="inner")
        return out

    if use_labels:
        mapping = {i + 1: label for i, label in enumerate(seq_data.labels)}
    else:
        mapping = seq_data.inverse_state_mapping

    out = seq_data.to_dataframe().replace(mapping).reset_index(drop=True)
    if wide:
        col = id_col or id
        if col is not None:
            out.insert(0, col, list(seq_data.ids))
        return out

    ids = list(seq_data.ids)
    times = list(seq_data.time)
    response = responses if isinstance(responses, str) else "state"
    return pd.DataFrame(
        {
            id: np.repeat(np.asarray(ids, dtype=object), len(times)),
            time: np.tile(np.asarray(times), len(ids)),
            response: out.to_numpy(dtype=object).reshape(-1),
        }
    )


def _figure_or_current(result):
    if result is not None:
        return result
    import matplotlib.pyplot as plt

    return plt.gcf()


def _plot_stacked_sequences(seq_data: SequenceData, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    numeric = seq_data.to_numeric()
    ax.imshow(numeric, aspect="auto", interpolation="nearest")
    ax.set_title("Sequences")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sequence")
    ax.set_xticks(range(numeric.shape[1]))
    ax.set_xticklabels(seq_data.time)
    return ax.figure


def _plot_state_distribution(seq_data: SequenceData, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    numeric = seq_data.to_numeric()
    x = np.arange(1, numeric.shape[1] + 1)
    proportions = []
    for state_idx in range(1, len(seq_data.states) + 1):
        proportions.append((numeric == state_idx).mean(axis=0))
    ax.stackplot(x, proportions, labels=seq_data.labels)
    ax.set_ylim(0, 1)
    ax.set_title("State Distribution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Proportion")
    return ax.figure


def stacked_sequence_plot(seq_data: SequenceData, **kwargs):
    """seqHMM-style sequence index plot wrapper returning a Figure."""
    return _plot_stacked_sequences(seq_data)


def ssplot(seq_data: SequenceData, **kwargs):
    """seqHMM-style state distribution plot wrapper returning a Figure."""
    return _plot_state_distribution(seq_data)


def gridplot(
    seq_data: SequenceData,
    plots: Sequence[str] = ("stacked", "distribution", "modal"),
    figsize: Optional[tuple] = None,
):
    """Create a compact grid of common sequence plots."""
    import matplotlib.pyplot as plt

    plots = tuple(plots)
    if figsize is None:
        figsize = (5 * len(plots), 4)
    fig, axes = plt.subplots(1, len(plots), figsize=figsize, squeeze=False)
    axes = axes.ravel()
    numeric = seq_data.to_numeric()
    x = np.arange(1, numeric.shape[1] + 1)

    for ax, plot_name in zip(axes, plots):
        if plot_name == "stacked":
            _plot_stacked_sequences(seq_data, ax=ax)
        elif plot_name in {"distribution", "ssplot"}:
            _plot_state_distribution(seq_data, ax=ax)
        elif plot_name == "modal":
            modal_states = []
            for t in range(numeric.shape[1]):
                counts = np.bincount(
                    numeric[:, t], minlength=len(seq_data.states) + 1
                )
                modal_states.append(int(np.argmax(counts[1:]) + 1))
            ax.step(x, modal_states, where="mid")
            ax.set_yticks(range(1, len(seq_data.states) + 1))
            ax.set_yticklabels(seq_data.labels)
            ax.set_title("Modal State")
            ax.set_xlabel("Time")
        else:
            raise ValueError(f"Unknown gridplot panel: {plot_name}")

    fig.tight_layout()
    return fig
