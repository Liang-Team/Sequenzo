"""
@Author  : Yuqi Liang 梁彧祺
@File    : utils.py
@Time    : 2025-11-12 12:18
@Desc    : Utility functions for HMM module

This module provides helper functions for converting between Sequenzo's
SequenceData format and formats required by hmmlearn.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
from sequenzo.define_sequence_data import SequenceData


def sequence_data_to_hmmlearn_format(
    seq_data: SequenceData,
    channel: Optional[int] = None
) -> tuple:
    """
    Convert Sequenzo SequenceData to format required by hmmlearn.
    
    hmmlearn expects:
    - X: array of shape (n_samples, 1) with integer observations (0-indexed)
    - lengths: array of sequence lengths
    
    Args:
        seq_data: SequenceData object from Sequenzo
        channel: Optional channel index for multichannel data (not implemented yet)
        
    Returns:
        tuple: (X, lengths) where:
            - X: numpy array of shape (n_total_timepoints, 1) with integer observations
            - lengths: numpy array of sequence lengths
    """
    # Get the sequence matrix (n_sequences x n_timepoints)
    sequences = seq_data.sequences
    
    # Get the alphabet (state space)
    alphabet = seq_data.alphabet
    n_symbols = len(alphabet)
    
    # Check if sequences are already integer-coded (SequenceData uses 1-indexed integers)
    # The sequences property returns integer-coded sequences (1, 2, 3, ...) not state symbols
    # Missing values are coded as len(states) + 1 (which equals n_symbols + 1 if missing was added)
    # We need to convert from 1-indexed to 0-indexed and filter out missing values
    
    # Convert sequences to integer format
    # Flatten all sequences into a single array
    X_list = []
    lengths = []
    
    for seq_idx in range(len(sequences)):
        seq = sequences[seq_idx]
        seq_int = []
        
        for state in seq:
            # Check if state is already an integer (SequenceData uses 1-indexed)
            if isinstance(state, (int, np.integer)):
                # Convert from 1-indexed to 0-indexed
                # Valid states are 1 to n_symbols, missing is n_symbols + 1
                if 1 <= state <= n_symbols:
                    seq_int.append(state - 1)  # Convert to 0-indexed
                # Skip missing values (state > n_symbols or state == 0)
            else:
                # If state is a symbol (string), map it to integer
                # This shouldn't happen with SequenceData, but handle it for safety
                state_to_int = {state: idx for idx, state in enumerate(alphabet)}
                mapped_int = state_to_int.get(state, -1)
                if mapped_int >= 0:
                    seq_int.append(mapped_int)
        
        # Only add sequence if it has valid states
        if len(seq_int) > 0:
            X_list.extend(seq_int)
            lengths.append(len(seq_int))
    
    # Check if we have any data
    if len(X_list) == 0:
        raise ValueError(
            "No valid sequences found. All sequences are empty or contain only missing/invalid states. "
            f"Number of sequences: {len(sequences)}, Alphabet size: {n_symbols}"
        )
    
    # Convert to numpy array with shape (n_samples, 1) for hmmlearn
    X = np.array(X_list, dtype=np.int32).reshape(-1, 1)
    lengths = np.array(lengths, dtype=np.int32)
    
    return X, lengths


def int_to_state_mapping(alphabet: List[str]) -> dict:
    """
    Create a mapping from integer indices to state symbols.
    
    Args:
        alphabet: List of state symbols
        
    Returns:
        dict: Mapping from integer (0-indexed) to state symbol
    """
    return {idx: state for idx, state in enumerate(alphabet)}


def state_to_int_mapping(alphabet: List[str]) -> dict:
    """
    Create a mapping from state symbols to integer indices.
    
    Args:
        alphabet: List of state symbols
        
    Returns:
        dict: Mapping from state symbol to integer (0-indexed)
    """
    return {state: idx for idx, state in enumerate(alphabet)}


def create_initial_probs(n_states: int, method: str = 'uniform', 
                         custom_probs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create initial state probabilities.
    
    Args:
        n_states: Number of hidden states
        method: Method to create probabilities ('uniform' or 'custom')
        custom_probs: Custom initial probabilities (must sum to 1)
        
    Returns:
        numpy array: Initial state probabilities
    """
    if method == 'uniform':
        return np.ones(n_states) / n_states
    elif method == 'custom':
        if custom_probs is None:
            raise ValueError("custom_probs must be provided when method='custom'")
        if len(custom_probs) != n_states:
            raise ValueError(f"custom_probs length ({len(custom_probs)}) must equal n_states ({n_states})")
        if not np.isclose(np.sum(custom_probs), 1.0):
            raise ValueError("custom_probs must sum to 1.0")
        return np.array(custom_probs)
    else:
        raise ValueError(f"Unknown method: {method}")


def create_transition_probs(n_states: int, method: str = 'random',
                           custom_probs: Optional[np.ndarray] = None,
                           random_state: Optional[int] = None) -> np.ndarray:
    """
    Create transition probability matrix.
    
    Args:
        n_states: Number of hidden states
        method: Method to create probabilities ('random', 'uniform', or 'custom')
        custom_probs: Custom transition matrix (must be n_states x n_states, rows sum to 1)
        random_state: Random seed for random initialization
        
    Returns:
        numpy array: Transition probability matrix (n_states x n_states)
    """
    if method == 'random':
        rng = np.random.RandomState(random_state)
        probs = rng.rand(n_states, n_states)
        # Normalize rows to sum to 1
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs
    elif method == 'uniform':
        return np.ones((n_states, n_states)) / n_states
    elif method == 'custom':
        if custom_probs is None:
            raise ValueError("custom_probs must be provided when method='custom'")
        if custom_probs.shape != (n_states, n_states):
            raise ValueError(f"custom_probs shape ({custom_probs.shape}) must be ({n_states}, {n_states})")
        # Check that rows sum to 1
        row_sums = custom_probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Each row of custom_probs must sum to 1.0")
        return np.array(custom_probs)
    else:
        raise ValueError(f"Unknown method: {method}")


def create_emission_probs(n_states: int, n_symbols: int, method: str = 'random',
                          custom_probs: Optional[np.ndarray] = None,
                          random_state: Optional[int] = None) -> np.ndarray:
    """
    Create emission probability matrix.
    
    Args:
        n_states: Number of hidden states
        n_symbols: Number of observed symbols (alphabet size)
        method: Method to create probabilities ('random', 'uniform', or 'custom')
        custom_probs: Custom emission matrix (must be n_states x n_symbols, rows sum to 1)
        random_state: Random seed for random initialization
        
    Returns:
        numpy array: Emission probability matrix (n_states x n_symbols)
    """
    if method == 'random':
        rng = np.random.RandomState(random_state)
        probs = rng.rand(n_states, n_symbols)
        # Normalize rows to sum to 1
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs
    elif method == 'uniform':
        return np.ones((n_states, n_symbols)) / n_symbols
    elif method == 'custom':
        if custom_probs is None:
            raise ValueError("custom_probs must be provided when method='custom'")
        if custom_probs.shape != (n_states, n_symbols):
            raise ValueError(f"custom_probs shape ({custom_probs.shape}) must be ({n_states}, {n_symbols})")
        # Check that rows sum to 1
        row_sums = custom_probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Each row of custom_probs must sum to 1.0")
        return np.array(custom_probs)
    else:
        raise ValueError(f"Unknown method: {method}")
