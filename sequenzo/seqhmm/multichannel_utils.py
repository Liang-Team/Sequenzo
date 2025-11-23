"""
@Author  : Yuqi Liang 梁彧祺
@File    : multichannel_utils.py
@Time    : 2025-11-05 11:26
@Desc    : Utility functions for multichannel HMM support

This module provides helper functions for handling multichannel sequence data,
where each subject has multiple parallel sequences (channels).
"""

import numpy as np
from typing import List, Union, Tuple
from sequenzo.define_sequence_data import SequenceData


def prepare_multichannel_data(
    observations: Union[SequenceData, List[SequenceData]]
) -> Tuple[List[SequenceData], List[str], List[List[str]]]:
    """
    Prepare multichannel data for HMM.
    
    This function handles both single-channel (SequenceData) and
    multichannel (List[SequenceData]) inputs.
    
    Args:
        observations: Either a single SequenceData or a list of SequenceData objects
        
    Returns:
        tuple: (channels, channel_names, alphabets) where:
            - channels: List of SequenceData objects (one per channel)
            - channel_names: List of channel names
            - alphabets: List of alphabets (one per channel)
    """
    if isinstance(observations, SequenceData):
        # Single channel
        return [observations], ["Channel 1"], [observations.alphabet]
    
    elif isinstance(observations, list):
        # Multichannel
        if len(observations) == 0:
            raise ValueError("observations list cannot be empty")
        
        # Validate all channels have same number of sequences
        n_sequences = len(observations[0].sequences)
        for i, obs in enumerate(observations):
            if not isinstance(obs, SequenceData):
                raise ValueError(f"observations[{i}] must be a SequenceData object")
            if len(obs.sequences) != n_sequences:
                raise ValueError(
                    f"All channels must have the same number of sequences. "
                    f"Channel 0 has {n_sequences}, channel {i} has {len(obs.sequences)}"
                )
        
        # Get channel names and alphabets
        channel_names = [f"Channel {i+1}" for i in range(len(observations))]
        alphabets = [obs.alphabet for obs in observations]
        
        return observations, channel_names, alphabets
    
    else:
        raise ValueError(
            f"observations must be SequenceData or List[SequenceData], got {type(observations)}"
        )


def multichannel_to_hmmlearn_format(
    channels: List[SequenceData]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert multichannel SequenceData to format for hmmlearn.
    
    For multichannel data, hmmlearn expects observations to be a tuple
    of arrays, one per channel. However, since hmmlearn's CategoricalHMM
    doesn't directly support multichannel, we'll need to handle this
    differently or use a custom implementation.
    
    For now, this function prepares the data structure. A full implementation
    would require extending hmmlearn or implementing multichannel HMM from scratch.
    
    Args:
        channels: List of SequenceData objects (one per channel)
        
    Returns:
        tuple: (X_list, lengths) where:
            - X_list: List of observation arrays (one per channel)
            - lengths: Array of sequence lengths (same for all channels)
    """
    from .utils import sequence_data_to_hmmlearn_format
    
    X_list = []
    lengths_list = []
    
    for channel in channels:
        X, lengths = sequence_data_to_hmmlearn_format(channel)
        X_list.append(X)
        lengths_list.append(lengths)
    
    # Validate all channels have same lengths
    lengths = lengths_list[0]
    for i, l in enumerate(lengths_list[1:], 1):
        if not np.array_equal(lengths, l):
            raise ValueError(
                f"All channels must have the same sequence lengths. "
                f"Channel 0 and channel {i} differ."
            )
    
    return X_list, lengths


def compute_multichannel_emission_prob(
    emission_probs: List[np.ndarray],
    observations: List[np.ndarray],
    n_states: int
) -> float:
    """
    Compute emission probability for multichannel observations.
    
    For multichannel HMM, the emission probability is the product of
    emission probabilities across all channels (assuming independence).
    
    P(obs | state) = product over channels: P(obs_channel | state)
    
    Args:
        emission_probs: List of emission probability matrices, one per channel
        observations: List of observed symbols (one per channel) at current time
        n_states: Number of hidden states
        
    Returns:
        numpy array: Emission probabilities (n_states,) for current observations
    """
    n_channels = len(emission_probs)
    emission = np.ones(n_states)
    
    # Multiply probabilities across channels
    for ch in range(n_channels):
        emission *= emission_probs[ch][:, observations[ch]]
    
    return emission
