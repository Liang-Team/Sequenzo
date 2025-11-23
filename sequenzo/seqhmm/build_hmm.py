"""
@Author  : Yuqi Liang 梁彧祺
@File    : build_hmm.py
@Time    : 2025-11-10 09:05
@Desc    : Build HMM models from SequenceData

This module provides the build_hmm function, which creates HMM model objects
similar to seqHMM's build_hmm() function in R.
"""

import numpy as np
from typing import Optional, List, Union
from sequenzo.define_sequence_data import SequenceData
from .multichannel_utils import prepare_multichannel_data
from .hmm import HMM
from .utils import (
    create_initial_probs,
    create_transition_probs,
    create_emission_probs
)


def build_hmm(
    observations: Union[SequenceData, List[SequenceData]],
    n_states: Optional[int] = None,
    initial_probs: Optional[np.ndarray] = None,
    transition_probs: Optional[np.ndarray] = None,
    emission_probs: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    state_names: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None,
    random_state: Optional[int] = None
) -> HMM:
    """
    Build a Hidden Markov Model object.
    
    This function creates an HMM model object that can be fitted to sequence data.
    It supports both single-channel and multichannel data.
    It is similar to seqHMM's build_hmm() function in R.
    
    Args:
        observations: SequenceData object or list of SequenceData objects (for multichannel)
                     containing the sequences to model
        n_states: Number of hidden states. Required if initial_probs, transition_probs,
                  or emission_probs are not provided.
        initial_probs: Optional initial state probabilities (n_states,).
                      If None, will be randomly initialized.
        transition_probs: Optional transition probability matrix (n_states x n_states).
                         If None, will be randomly initialized.
        emission_probs: Optional emission probability matrix (n_states x n_symbols).
                       If None, will be randomly initialized.
        state_names: Optional names for hidden states. If None, uses "State 1", "State 2", etc.
        channel_names: Optional names for channels. Currently only single-channel is supported.
        random_state: Random seed for initialization of random parameters.
        
    Returns:
        HMM: An HMM model object (not yet fitted)
        
    Examples:
        >>> from sequenzo import SequenceData, load_dataset
        >>> from sequenzo.seqhmm import build_hmm
        >>> 
        >>> # Load example data
        >>> df = load_dataset('mvad')
        >>> seq = SequenceData(df, time=range(15, 86), states=['EM', 'FE', 'HE', 'JL', 'SC', 'TR'])
        >>> 
        >>> # Build HMM with 4 states, random initialization
        >>> hmm = build_hmm(seq, n_states=4, random_state=42)
        >>> 
        >>> # Build HMM with custom initial parameters
        >>> init_probs = np.array([0.3, 0.3, 0.2, 0.2])
        >>> trans_probs = np.array([[0.8, 0.1, 0.05, 0.05],
        ...                         [0.05, 0.8, 0.1, 0.05],
        ...                         [0.05, 0.05, 0.8, 0.1],
        ...                         [0.05, 0.05, 0.1, 0.8]])
        >>> emission_probs = np.random.rand(4, 6)  # 4 states, 6 symbols
        >>> emission_probs = emission_probs / emission_probs.sum(axis=1, keepdims=True)
        >>> hmm = build_hmm(seq, initial_probs=init_probs, 
        ...                  transition_probs=trans_probs,
        ...                  emission_probs=emission_probs)
    """
    # Determine number of states
    if n_states is None:
        if initial_probs is not None:
            n_states = len(initial_probs)
        elif transition_probs is not None:
            n_states = transition_probs.shape[0]
        elif emission_probs is not None:
            n_states = emission_probs.shape[0]
        else:
            raise ValueError(
                "n_states must be provided if initial_probs, transition_probs, "
                "and emission_probs are all None"
            )
    
    # Get alphabet size
    n_symbols = len(observations.alphabet)
    
    # Create initial probabilities if not provided
    if initial_probs is None:
        initial_probs = create_initial_probs(n_states, method='uniform')
    
    # Create transition probabilities if not provided
    if transition_probs is None:
        transition_probs = create_transition_probs(
            n_states, method='random', random_state=random_state
        )
    
    # Create emission probabilities if not provided
    if emission_probs is None:
        emission_probs = create_emission_probs(
            n_states, n_symbols, method='random', random_state=random_state
        )
    
    # Validate dimensions
    if len(initial_probs) != n_states:
        raise ValueError(
            f"initial_probs length ({len(initial_probs)}) must equal n_states ({n_states})"
        )
    
    if transition_probs.shape != (n_states, n_states):
        raise ValueError(
            f"transition_probs shape ({transition_probs.shape}) must be ({n_states}, {n_states})"
        )
    
    if emission_probs.shape != (n_states, n_symbols):
        raise ValueError(
            f"emission_probs shape ({emission_probs.shape}) must be ({n_states}, {n_symbols})"
        )
    
    # Create and return HMM object
    hmm = HMM(
        observations=observations,
        n_states=n_states,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        state_names=state_names,
        channel_names=channel_names,
        random_state=random_state
    )
    
    return hmm
