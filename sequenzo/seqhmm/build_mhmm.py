"""
@Author  : Yuqi Liang 梁彧祺
@File    : build_mhmm.py
@Time    : 2025-11-21 10:55
@Desc    : Build Mixture HMM models from SequenceData

This module provides the build_mhmm function, which creates Mixture HMM model objects
similar to seqHMM's build_mhmm() function in R.
"""

import numpy as np
from typing import Optional, List, Union
from sequenzo.define_sequence_data import SequenceData
from .mhmm import MHMM
from .hmm import HMM
from .multichannel_utils import prepare_multichannel_data


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
        raise ValueError(f"{name} must sum to one")
    return values.copy()


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
    if not np.allclose(row_sums.ravel(), 1.0):
        raise ValueError(f"{name} rows must sum to one")
    return values.copy()


def _validate_parameter_lists(
    initial_probs,
    transition_probs,
    emission_probs,
    n_clusters,
    n_states_list,
    n_symbols,
):
    if initial_probs is not None and len(initial_probs) != n_clusters:
        raise ValueError(f"initial_probs length ({len(initial_probs)}) must equal n_clusters ({n_clusters})")
    if transition_probs is not None and len(transition_probs) != n_clusters:
        raise ValueError(f"transition_probs length ({len(transition_probs)}) must equal n_clusters ({n_clusters})")
    if emission_probs is not None and len(emission_probs) != n_clusters:
        raise ValueError(f"emission_probs length ({len(emission_probs)}) must equal n_clusters ({n_clusters})")

    multichannel = isinstance(n_symbols, list)
    if initial_probs is not None:
        initial_probs = [
            _validate_probability_vector(value, n_states_list[k], f"initial_probs[{k}]")
            for k, value in enumerate(initial_probs)
        ]
    if transition_probs is not None:
        transition_probs = [
            _validate_probability_matrix(
                value,
                (n_states_list[k], n_states_list[k]),
                f"transition_probs[{k}]",
            )
            for k, value in enumerate(transition_probs)
        ]
    if emission_probs is not None:
        if multichannel:
            checked = []
            for k, cluster_values in enumerate(emission_probs):
                if not isinstance(cluster_values, (list, tuple)):
                    raise ValueError(
                        "multichannel emission_probs must be a list per cluster"
                    )
                if len(cluster_values) != len(n_symbols):
                    raise ValueError(
                        f"emission_probs[{k}] length ({len(cluster_values)}) "
                        f"must equal n_channels ({len(n_symbols)})"
                    )
                checked.append([
                    _validate_probability_matrix(
                        value,
                        (n_states_list[k], n_symbols[ch_idx]),
                        f"emission_probs[{k}][{ch_idx}]",
                    )
                    for ch_idx, value in enumerate(cluster_values)
                ])
            emission_probs = checked
        else:
            emission_probs = [
                _validate_probability_matrix(
                    value,
                    (n_states_list[k], n_symbols),
                    f"emission_probs[{k}]",
                )
                for k, value in enumerate(emission_probs)
            ]
    return initial_probs, transition_probs, emission_probs


def build_mhmm(
    observations: Union[SequenceData, List[SequenceData]],
    n_clusters: int,
    n_states: Union[int, List[int]],
    initial_probs: Optional[List[np.ndarray]] = None,
    transition_probs: Optional[List[np.ndarray]] = None,
    emission_probs: Optional[List[Union[np.ndarray, List[np.ndarray]]]] = None,
    cluster_probs: Optional[np.ndarray] = None,
    cluster_names: Optional[List[str]] = None,
    state_names: Optional[List[List[str]]] = None,
    channel_names: Optional[List[str]] = None,
    random_state: Optional[int] = None
) -> MHMM:
    """
    Build a Mixture Hidden Markov Model object.
    
    A Mixture HMM consists of multiple HMM submodels (clusters). Each sequence
    belongs to one of these clusters with certain probabilities. This function
    creates the model structure but does not fit it (use fit_mhmm() for that).
    
    It is similar to seqHMM's build_mhmm() function in R.
    
    Args:
        observations: SequenceData object, or a list of SequenceData objects for
                     multichannel fixed-parameter inference
        n_clusters: Number of clusters (submodels)
        n_states: Number of hidden states per cluster. Can be:
                 - int: Same number of states for all clusters
                 - List[int]: Different number of states for each cluster
        initial_probs: Optional list of initial state probabilities, one per cluster.
                      Each element should be (n_states[k],) array.
        transition_probs: Optional list of transition matrices, one per cluster.
                         Each element should be (n_states[k], n_states[k]) array.
        emission_probs: Optional list of emission matrices, one per cluster.
                       For multichannel input, each cluster element should be a
                       list of matrices, one per channel.
        cluster_probs: Optional initial cluster probabilities (n_clusters,).
                     If None, uses uniform probabilities.
        cluster_names: Optional names for clusters
        state_names: Optional names for hidden states. Should be a list of lists,
                    where state_names[k] contains names for cluster k.
        channel_names: Optional names for channels
        random_state: Random seed for initialization
        
    Returns:
        MHMM: A Mixture HMM model object (not yet fitted)
        
    Examples:
        >>> from sequenzo import SequenceData, load_dataset
        >>> from sequenzo.seqhmm import build_mhmm
        >>> 
        >>> # Load example data
        >>> df = load_dataset('mvad')
        >>> seq = SequenceData(df, time=range(15, 86), states=['EM', 'FE', 'HE', 'JL', 'SC', 'TR'])
        >>> 
        >>> # Build MHMM with 3 clusters, 4 states each
        >>> mhmm = build_mhmm(seq, n_clusters=3, n_states=4, random_state=42)
        >>> 
        >>> # Build MHMM with different number of states per cluster
        >>> mhmm = build_mhmm(seq, n_clusters=3, n_states=[4, 4, 6], random_state=42)
    """
    _, default_channel_names, alphabets = prepare_multichannel_data(observations)
    n_symbols = [len(alphabet) for alphabet in alphabets]
    if len(n_symbols) == 1:
        n_symbols = n_symbols[0]
    
    # Handle n_states: convert to list if int
    if isinstance(n_states, int):
        n_states_list = [n_states] * n_clusters
    else:
        n_states_list = n_states
    
    # Validate n_states length
    if len(n_states_list) != n_clusters:
        raise ValueError(
            f"n_states length ({len(n_states_list)}) must equal n_clusters ({n_clusters})"
        )
    initial_probs, transition_probs, emission_probs = _validate_parameter_lists(
        initial_probs,
        transition_probs,
        emission_probs,
        n_clusters,
        n_states_list,
        n_symbols,
    )
    
    # Build HMM clusters
    clusters = []
    for k in range(n_clusters):
        # Get parameters for this cluster
        cluster_initial = initial_probs[k] if initial_probs is not None and k < len(initial_probs) else None
        cluster_transition = transition_probs[k] if transition_probs is not None and k < len(transition_probs) else None
        cluster_emission = emission_probs[k] if emission_probs is not None and k < len(emission_probs) else None
        
        # Get state names for this cluster
        cluster_state_names = None
        if state_names is not None and k < len(state_names):
            cluster_state_names = state_names[k]
        
        # Create HMM for this cluster
        hmm = HMM(
            observations=observations,
            n_states=n_states_list[k],
            initial_probs=cluster_initial,
            transition_probs=cluster_transition,
            emission_probs=cluster_emission,
            state_names=cluster_state_names,
            channel_names=channel_names or default_channel_names,
            random_state=random_state
        )
        clusters.append(hmm)
    
    # Create and return MHMM object
    mhmm = MHMM(
        observations=observations,
        n_clusters=n_clusters,
        n_states=n_states_list,
        clusters=clusters,
        cluster_probs=cluster_probs,
        cluster_names=cluster_names,
        state_names=state_names,
        channel_names=channel_names or default_channel_names,
        random_state=random_state
    )
    
    return mhmm
