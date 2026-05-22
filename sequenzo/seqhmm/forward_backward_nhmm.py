"""
@Author  : Yuqi Liang 梁彧祺; Yapeng Wei 卫亚鹏
@File    : forward_backward_nhmm.py
@Time    : 2025-10-20 09:41
@Desc    : Forward-Backward algorithm for Non-homogeneous HMM

This module provides the forward-backward algorithm implementation for NHMM,
which handles time-varying transition and emission probabilities.
This is similar to seqHMM's forward_backward.nhmm() function in R.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .nhmm import NHMM


def forward_backward_nhmm(
    model: NHMM,
    sequences: Optional = None,
    forward_only: bool = False
) -> pd.DataFrame:
    """
    Compute forward and backward probabilities for a Non-homogeneous HMM.
    
    The forward-backward algorithm computes the probability of being in each
    hidden state at each time point, given the observed sequence. For NHMM,
    this accounts for time-varying transition and emission probabilities.
    
    This is similar to seqHMM's forward_backward.nhmm() function in R.
    
    Args:
        model: Fitted NHMM model object
        sequences: Optional SequenceData (uses model.observations if None)
        forward_only: If True, only compute forward probabilities. Default is False.
        
    Returns:
        pandas DataFrame: Forward and backward probabilities with columns:
            - id: Sequence identifier
            - time: Time point
            - state: Hidden state index
            - log_alpha: Log forward probability
            - log_beta: Log backward probability (if forward_only=False)
            
    Examples:
        >>> from sequenzo.seqhmm import build_nhmm, fit_nhmm, forward_backward_nhmm
        >>> 
        >>> nhmm = build_nhmm(seq, n_states=4, X=X, random_state=42)
        >>> nhmm = fit_nhmm(nhmm)
        >>> 
        >>> # Compute forward-backward probabilities
        >>> fb = forward_backward_nhmm(nhmm)
        >>> print(fb.head())
    """
    if model.log_likelihood is None:
        raise ValueError("Model must be fitted before computing forward-backward probabilities.")
    
    if sequences is None:
        sequences = model.observations
    
    observations = _nhmm_observation_arrays(sequences, model.n_symbols)
    n_sequences = len(observations)
    
    # Compute probabilities for all sequences and time points
    initial_probs, transition_probs, emission_probs = model._compute_probs()
    
    # Initialize results list
    results = []
    
    # Process each sequence
    for seq_idx in range(n_sequences):
        original_idx, obs_seq = observations[seq_idx]
        seq_length = len(obs_seq)
        
        # Get probabilities for this sequence
        seq_initial = initial_probs[original_idx, :]  # Shape: (n_states,)
        seq_transition = transition_probs[original_idx, :seq_length, :, :]  # Shape: (seq_length, n_states, n_states)
        seq_emission = emission_probs[original_idx, :seq_length, :, :]  # Shape: (seq_length, n_states, n_symbols)
        
        # Compute forward probabilities
        log_alpha = _forward_nhmm(seq_initial, seq_transition, seq_emission, obs_seq, model.n_states)
        
        # Compute backward probabilities if requested
        if forward_only:
            log_beta = None
        else:
            log_beta = _backward_nhmm(seq_transition, seq_emission, obs_seq, model.n_states)
        
        # Store results
        for t in range(seq_length):
            for state_idx in range(model.n_states):
                result_row = {
                    'id': sequences.ids[original_idx] if getattr(sequences, "ids", None) is not None else original_idx,
                    'time': sequences.time[t] if getattr(sequences, "time", None) is not None else t + 1,
                    'state': state_idx,
                    'log_alpha': log_alpha[state_idx, t]
                }
                if not forward_only:
                    result_row['log_beta'] = log_beta[state_idx, t]
                results.append(result_row)
    
    return pd.DataFrame(results)


def _forward_nhmm(
    initial_probs: np.ndarray,
    transition_probs: np.ndarray,
    emission_probs: np.ndarray,
    observations: np.ndarray,
    n_states: int
) -> np.ndarray:
    """
    Forward algorithm for Non-homogeneous HMM (log-space implementation).
    
    Computes forward probabilities: alpha[i, t] = P(obs[0:t], state_t = i)
    
    For NHMM, transition and emission probabilities vary with time, so we
    use time-specific probabilities at each step.
    
    Args:
        initial_probs: Initial state probabilities (n_states,)
        transition_probs: Time-varying transition matrix (T, n_states, n_states)
        emission_probs: Time-varying emission matrix (T, n_states, n_symbols)
        observations: Observed sequence (T,) with integer observations (0-indexed)
        n_states: Number of hidden states
        
    Returns:
        numpy array: Log forward probabilities (n_states, T)
    """
    T = len(observations)
    log_alpha = np.zeros((n_states, T))
    
    # Initialization: alpha[i, 0] = pi[i] * B[i, obs[0]]
    for i in range(n_states):
        log_alpha[i, 0] = np.log(initial_probs[i] + 1e-10)
        if observations[0] >= 0:
            log_alpha[i, 0] += np.log(emission_probs[0, i, observations[0]] + 1e-10)
    
    # Recursion uses the destination-time transition matrix.
    for t in range(1, T):
        for j in range(n_states):
            # Compute log-sum-exp for numerical stability
            log_sum = -np.inf
            for i in range(n_states):
                log_term = (
                    log_alpha[i, t-1] +
                    np.log(transition_probs[t, i, j] + 1e-10)
                )
                if observations[t] >= 0:
                    log_term += np.log(emission_probs[t, j, observations[t]] + 1e-10)
                # Log-sum-exp trick
                if log_sum == -np.inf:
                    log_sum = log_term
                else:
                    log_sum = np.logaddexp(log_sum, log_term)
            
            log_alpha[j, t] = log_sum
    
    return log_alpha


def _backward_nhmm(
    transition_probs: np.ndarray,
    emission_probs: np.ndarray,
    observations: np.ndarray,
    n_states: int
) -> np.ndarray:
    """
    Backward algorithm for Non-homogeneous HMM (log-space implementation).
    
    Computes backward probabilities: beta[i, t] = P(obs[t+1:T] | state_t = i)
    
    Args:
        transition_probs: Time-varying transition matrix (T, n_states, n_states)
        emission_probs: Time-varying emission matrix (T, n_states, n_symbols)
        observations: Observed sequence (T,) with integer observations (0-indexed)
        n_states: Number of hidden states
        
    Returns:
        numpy array: Log backward probabilities (n_states, T)
    """
    T = len(observations)
    log_beta = np.zeros((n_states, T))
    
    # Initialization: beta[i, T-1] = 1 for all i
    log_beta[:, T-1] = 0.0  # log(1) = 0
    
    # Recursion uses the destination-time transition matrix.
    for t in range(T-2, -1, -1):
        for i in range(n_states):
            log_sum = -np.inf
            for j in range(n_states):
                log_term = (
                    np.log(transition_probs[t+1, i, j] + 1e-10) +
                    log_beta[j, t+1]
                )
                if observations[t+1] >= 0:
                    log_term += np.log(emission_probs[t+1, j, observations[t+1]] + 1e-10)
                # Log-sum-exp trick
                if log_sum == -np.inf:
                    log_sum = log_term
                else:
                    log_sum = np.logaddexp(log_sum, log_term)
            
            log_beta[i, t] = log_sum
    
    return log_beta


def log_likelihood_nhmm(model: NHMM, sequences: Optional = None) -> float:
    """
    Compute log-likelihood for NHMM using forward algorithm.
    
    The log-likelihood is computed as the sum of log forward probabilities
    at the final time point for each sequence.
    
    This is similar to seqHMM's logLik.nhmm() function in R.
    
    Args:
        model: Fitted NHMM model object
        sequences: Optional SequenceData (uses model.observations if None)
        
    Returns:
        float: Total log-likelihood across all sequences
    """
    if sequences is None:
        sequences = model.observations
    
    observations = _nhmm_observation_arrays(sequences, model.n_symbols)
    n_sequences = len(observations)
    
    # Compute probabilities
    initial_probs, transition_probs, emission_probs = model._compute_probs()
    
    total_log_lik = 0.0
    
    # Process each sequence
    for seq_idx in range(n_sequences):
        original_idx, obs_seq = observations[seq_idx]
        seq_length = len(obs_seq)
        
        # Get probabilities for this sequence
        seq_initial = initial_probs[original_idx, :]
        seq_transition = transition_probs[original_idx, :seq_length, :, :]
        seq_emission = emission_probs[original_idx, :seq_length, :, :]
        
        # Compute forward probabilities
        log_alpha = _forward_nhmm(seq_initial, seq_transition, seq_emission, obs_seq, model.n_states)
        
        # Log-likelihood is log(sum of forward probabilities at final time)
        # Use log-sum-exp for numerical stability
        log_lik_seq = -np.inf
        for i in range(model.n_states):
            if log_lik_seq == -np.inf:
                log_lik_seq = log_alpha[i, seq_length-1]
            else:
                log_lik_seq = np.logaddexp(log_lik_seq, log_alpha[i, seq_length-1])
        
        total_log_lik += log_lik_seq
    
    return total_log_lik


def _nhmm_observation_arrays(sequences, n_symbols: int) -> list[tuple[int, np.ndarray]]:
    observation_arrays = []
    alphabet = getattr(sequences, "alphabet", [])
    state_to_int = {state: idx for idx, state in enumerate(alphabet[:n_symbols])}

    for seq_idx, seq in enumerate(sequences.sequences):
        obs = []
        for state in seq:
            if isinstance(state, (int, np.integer)):
                obs.append(state - 1 if 1 <= state <= n_symbols else -1)
            else:
                obs.append(state_to_int.get(state, -1))
        observation_arrays.append((seq_idx, np.asarray(obs, dtype=np.int32)))

    if not observation_arrays:
        raise ValueError("No valid observations found for NHMM likelihood.")
    return observation_arrays
