"""
@Author  : Yuqi Liang 梁彧祺
@File    : multichannel_em.py
@Time    : 2025-11-08 13:52
@Desc    : EM algorithm for multichannel HMM

This module provides the EM algorithm implementation for multichannel HMM,
where each sequence has multiple parallel channels (e.g., marriage, children, residence).
"""

import numpy as np
from typing import List
from .hmm import HMM
from .multichannel_utils import multichannel_to_hmmlearn_format, compute_multichannel_emission_prob
from .utils import sequence_data_to_hmmlearn_format, state_to_int_mapping


def fit_multichannel_hmm(
    model: HMM,
    n_iter: int = 100,
    tol: float = 1e-2,
    verbose: bool = False
) -> HMM:
    """
    Fit a multichannel HMM using EM algorithm.
    
    For multichannel HMM, the emission probability is the product of
    emission probabilities across all channels (assuming independence).
    
    This is similar to seqHMM's multichannel HMM fitting in R.
    
    Args:
        model: HMM model object with multichannel data
        n_iter: Maximum number of EM iterations
        tol: Convergence tolerance
        verbose: Whether to print progress
        
    Returns:
        HMM: Fitted model
    """
    n_channels = model.n_channels
    n_states = model.n_states
    channels = model.channels
    
    # Get sequence lengths (same for all channels)
    lengths = model.sequence_lengths
    n_sequences = len(lengths)
    
    # Initialize parameters if not provided
    if model.initial_probs is None:
        model.initial_probs = np.ones(n_states) / n_states
    
    if model.transition_probs is None:
        model.transition_probs = np.ones((n_states, n_states)) / n_states
    
    if model.emission_probs is None or not isinstance(model.emission_probs, list):
        # Initialize emission probabilities for each channel
        model.emission_probs = []
        for ch in range(n_channels):
            n_symbols_ch = model.n_symbols[ch]
            emission_ch = np.random.rand(n_states, n_symbols_ch)
            emission_ch = emission_ch / emission_ch.sum(axis=1, keepdims=True)
            model.emission_probs.append(emission_ch)
    
    # Convert channels to integer format
    X_list = []
    state_to_int_list = []
    for ch in range(n_channels):
        X_ch, _ = sequence_data_to_hmmlearn_format(channels[ch])
        X_list.append(X_ch)
        state_to_int_ch = state_to_int_mapping(channels[ch].alphabet)
        state_to_int_list.append(state_to_int_ch)
    
    # EM algorithm
    prev_log_likelihood = -np.inf
    
    for iteration in range(n_iter):
        # E-step: Compute forward and backward probabilities
        # For multichannel, we need to compute emission probabilities
        # as product across channels
        
        # Initialize forward and backward arrays
        log_alpha = {}  # Dictionary: seq_idx -> (n_states, T) array
        log_beta = {}   # Dictionary: seq_idx -> (n_states, T) array
        
        total_log_lik = 0.0
        
        # Forward pass
        for seq_idx in range(n_sequences):
            seq_length = lengths[seq_idx]
            start_idx = lengths[:seq_idx].sum()
            end_idx = start_idx + seq_length
            
            # Get observations for all channels
            obs_list = [X_ch[start_idx:end_idx, 0] for X_ch in X_list]
            
            # Initialize forward probabilities
            alpha = np.zeros((n_states, seq_length))
            
            # Initialization: alpha[i, 0] = pi[i] * product(B_ch[i, obs_ch[0]])
            for i in range(n_states):
                emission_prob = 1.0
                for ch in range(n_channels):
                    emission_prob *= model.emission_probs[ch][i, obs_list[ch][0]]
                alpha[i, 0] = model.initial_probs[i] * emission_prob
            
            # Scale to prevent underflow
            scale = alpha[:, 0].sum()
            alpha[:, 0] /= scale
            log_scale = np.log(scale)
            
            # Recursion
            for t in range(1, seq_length):
                for j in range(n_states):
                    # Compute emission probability for multichannel
                    emission_prob = 1.0
                    for ch in range(n_channels):
                        emission_prob *= model.emission_probs[ch][j, obs_list[ch][t]]
                    
                    # Forward: alpha[j, t] = sum_i(alpha[i, t-1] * A[i, j] * B[j, obs[t]])
                    alpha[j, t] = np.sum(alpha[:, t-1] * model.transition_probs[:, j]) * emission_prob
                
                # Scale
                scale = alpha[:, t].sum()
                alpha[:, t] /= scale
                log_scale += np.log(scale)
            
            log_alpha[seq_idx] = np.log(alpha + 1e-10)
            total_log_lik += log_scale
        
        # Backward pass
        for seq_idx in range(n_sequences):
            seq_length = lengths[seq_idx]
            start_idx = lengths[:seq_idx].sum()
            end_idx = start_idx + seq_length
            
            obs_list = [X_ch[start_idx:end_idx, 0] for X_ch in X_list]
            
            beta = np.ones((n_states, seq_length))
            
            # Recursion (backward)
            for t in range(seq_length - 2, -1, -1):
                for i in range(n_states):
                    # Compute emission probability for next time
                    emission_prob_next = 1.0
                    for ch in range(n_channels):
                        emission_prob_next *= model.emission_probs[ch][:, obs_list[ch][t+1]]
                    
                    # Backward: beta[i, t] = sum_j(A[i, j] * B[j, obs[t+1]] * beta[j, t+1])
                    beta[i, t] = np.sum(
                        model.transition_probs[i, :] * emission_prob_next * beta[:, t+1]
                    )
                
                # Scale (use same scale as forward)
                beta[:, t] /= beta[:, t].sum()
            
            log_beta[seq_idx] = np.log(beta + 1e-10)
        
        # M-step: Update parameters
        # Update initial probabilities
        gamma_0 = np.zeros(n_states)
        for seq_idx in range(n_sequences):
            gamma_0 += np.exp(log_alpha[seq_idx][:, 0] + log_beta[seq_idx][:, 0] - 
                             np.log(np.sum(np.exp(log_alpha[seq_idx][:, 0] + log_beta[seq_idx][:, 0]))))
        model.initial_probs = gamma_0 / n_sequences
        
        # Update transition probabilities
        xi_sum = np.zeros((n_states, n_states))
        gamma_sum = np.zeros(n_states)
        
        for seq_idx in range(n_sequences):
            seq_length = lengths[seq_idx]
            start_idx = lengths[:seq_idx].sum()
            end_idx = start_idx + seq_length
            
            obs_list = [X_ch[start_idx:end_idx, 0] for X_ch in X_list]
            
            # Compute gamma and xi
            for t in range(seq_length):
                # Gamma: posterior probability of being in state i at time t
                log_gamma = log_alpha[seq_idx][:, t] + log_beta[seq_idx][:, t]
                log_gamma -= np.log(np.sum(np.exp(log_gamma)))
                gamma = np.exp(log_gamma)
                gamma_sum += gamma
                
                if t < seq_length - 1:
                    # Xi: joint probability of state i at t and state j at t+1
                    for i in range(n_states):
                        for j in range(n_states):
                            # Compute emission probability for next time
                            emission_prob_next = 1.0
                            for ch in range(n_channels):
                                emission_prob_next *= model.emission_probs[ch][j, obs_list[ch][t+1]]
                            
                            log_xi = (
                                log_alpha[seq_idx][i, t] +
                                np.log(model.transition_probs[i, j] + 1e-10) +
                                np.log(emission_prob_next + 1e-10) +
                                log_beta[seq_idx][j, t+1]
                            )
                            # Normalize
                            log_xi_sum = -np.inf
                            for i2 in range(n_states):
                                for j2 in range(n_states):
                                    emission_prob_next2 = 1.0
                                    for ch in range(n_channels):
                                        emission_prob_next2 *= model.emission_probs[ch][j2, obs_list[ch][t+1]]
                                    log_xi_term = (
                                        log_alpha[seq_idx][i2, t] +
                                        np.log(model.transition_probs[i2, j2] + 1e-10) +
                                        np.log(emission_prob_next2 + 1e-10) +
                                        log_beta[seq_idx][j2, t+1]
                                    )
                                    if log_xi_sum == -np.inf:
                                        log_xi_sum = log_xi_term
                                    else:
                                        log_xi_sum = np.logaddexp(log_xi_sum, log_xi_term)
                            
                            xi = np.exp(log_xi - log_xi_sum)
                            xi_sum[i, j] += xi
        
        # Normalize transition probabilities
        for i in range(n_states):
            if gamma_sum[i] > 0:
                model.transition_probs[i, :] = xi_sum[i, :] / gamma_sum[i]
            else:
                model.transition_probs[i, :] = 1.0 / n_states
        
        # Update emission probabilities for each channel
        for ch in range(n_channels):
            n_symbols_ch = model.n_symbols[ch]
            emission_ch = np.zeros((n_states, n_symbols_ch))
            gamma_sum_ch = np.zeros(n_states)
            
            for seq_idx in range(n_sequences):
                seq_length = lengths[seq_idx]
                start_idx = lengths[:seq_idx].sum()
                end_idx = start_idx + seq_length
                
                obs_ch = X_list[ch][start_idx:end_idx, 0]
                
                for t in range(seq_length):
                    # Gamma: posterior probability
                    log_gamma = log_alpha[seq_idx][:, t] + log_beta[seq_idx][:, t]
                    log_gamma -= np.log(np.sum(np.exp(log_gamma)))
                    gamma = np.exp(log_gamma)
                    
                    # Update emission counts
                    for i in range(n_states):
                        emission_ch[i, obs_ch[t]] += gamma[i]
                        gamma_sum_ch[i] += gamma[i]
            
            # Normalize
            for i in range(n_states):
                if gamma_sum_ch[i] > 0:
                    model.emission_probs[ch][i, :] = emission_ch[i, :] / gamma_sum_ch[i]
                else:
                    model.emission_probs[ch][i, :] = 1.0 / n_symbols_ch
        
        # Check convergence
        if iteration > 0:
            change = total_log_lik - prev_log_likelihood
            if abs(change) < tol:
                model.converged = True
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
        
        prev_log_likelihood = total_log_lik
        
        if verbose and (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: log-likelihood = {total_log_lik:.4f}")
    
    model.log_likelihood = prev_log_likelihood
    model.n_iter = iteration + 1
    
    if not model.converged:
        model.converged = False
        if verbose:
            print(f"Did not converge after {n_iter} iterations")
    
    return model
