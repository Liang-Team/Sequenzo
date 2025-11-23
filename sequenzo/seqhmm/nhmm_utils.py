"""
@Author  : Yuqi Liang 梁彧祺
@File    : nhmm_utils.py
@Time    : 2025-11-23 10:20
@Desc    : Utility functions for Non-homogeneous HMM

This module provides utility functions for NHMM, including Softmax parameterization
and gradient computation.
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax function for numerical stability.
    
    Softmax converts a vector of real numbers into a probability distribution.
    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    We use the log-sum-exp trick for numerical stability:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    Args:
        x: Input array
        axis: Axis along which to compute softmax
        
    Returns:
        numpy array: Softmax probabilities (sums to 1 along specified axis)
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def eta_to_gamma(eta: np.ndarray, n_categories: int) -> np.ndarray:
    """
    Convert eta (linear predictor) to gamma (probabilities) using Softmax.
    
    In NHMM, we use linear predictors (eta) that are transformed to probabilities
    (gamma) using the Softmax function. This allows covariates to influence
    probabilities while ensuring they sum to 1.
    
    Args:
        eta: Linear predictor array of shape (..., n_categories)
        n_categories: Number of categories (e.g., number of states)
        
    Returns:
        numpy array: Probabilities of shape (..., n_categories), sums to 1 along last axis
    """
    # Reshape eta to (n_samples, n_categories)
    original_shape = eta.shape
    eta_flat = eta.reshape(-1, n_categories)
    
    # Apply softmax
    gamma_flat = softmax(eta_flat, axis=1)
    
    # Reshape back to original shape
    return gamma_flat.reshape(original_shape)


def compute_transition_probs_with_covariates(
    eta_A: np.ndarray,
    X: np.ndarray,
    n_states: int
) -> np.ndarray:
    """
    Compute transition probabilities from covariates using Softmax.
    
    For each time point and each sequence, we compute:
    eta = X @ coefficients
    gamma = softmax(eta)
    
    Args:
        eta_A: Coefficient matrix of shape (n_covariates, n_states, n_states)
               where eta_A[c, i, j] is the coefficient for covariate c,
               transition from state i to state j
        X: Covariate matrix of shape (n_sequences, n_timepoints, n_covariates)
        n_states: Number of hidden states
        
    Returns:
        numpy array: Transition probabilities of shape (n_sequences, n_timepoints, n_states, n_states)
    """
    n_sequences, n_timepoints, n_covariates = X.shape
    
    # Initialize transition probability matrix
    transition_probs = np.zeros((n_sequences, n_timepoints, n_states, n_states))
    
    # For each sequence and time point
    for seq_idx in range(n_sequences):
        for t in range(n_timepoints):
            # Get covariates for this time point
            x_t = X[seq_idx, t, :]  # Shape: (n_covariates,)
            
            # Compute linear predictor for each transition
            # eta[i, j] = sum over covariates: x[c] * eta_A[c, i, j]
            eta = np.zeros((n_states, n_states))
            for i in range(n_states):
                for j in range(n_states):
                    eta[i, j] = np.sum(x_t * eta_A[:, i, j])
            
            # Convert to probabilities using softmax (row-wise)
            for i in range(n_states):
                transition_probs[seq_idx, t, i, :] = softmax(eta[i, :])
    
    return transition_probs


def compute_emission_probs_with_covariates(
    eta_B: np.ndarray,
    X: np.ndarray,
    n_states: int,
    n_symbols: int
) -> np.ndarray:
    """
    Compute emission probabilities from covariates using Softmax.
    
    Similar to transition probabilities, but for emission probabilities.
    
    Args:
        eta_B: Coefficient matrix of shape (n_covariates, n_states, n_symbols)
        X: Covariate matrix of shape (n_sequences, n_timepoints, n_covariates)
        n_states: Number of hidden states
        n_symbols: Number of observed symbols
        
    Returns:
        numpy array: Emission probabilities of shape (n_sequences, n_timepoints, n_states, n_symbols)
    """
    n_sequences, n_timepoints, n_covariates = X.shape
    
    # Initialize emission probability matrix
    emission_probs = np.zeros((n_sequences, n_timepoints, n_states, n_symbols))
    
    # For each sequence and time point
    for seq_idx in range(n_sequences):
        for t in range(n_timepoints):
            # Get covariates for this time point
            x_t = X[seq_idx, t, :]  # Shape: (n_covariates,)
            
            # Compute linear predictor for each emission
            # eta[i, j] = sum over covariates: x[c] * eta_B[c, i, j]
            eta = np.zeros((n_states, n_symbols))
            for i in range(n_states):
                for j in range(n_symbols):
                    eta[i, j] = np.sum(x_t * eta_B[:, i, j])
            
            # Convert to probabilities using softmax (row-wise)
            for i in range(n_states):
                emission_probs[seq_idx, t, i, :] = softmax(eta[i, :])
    
    return emission_probs


def compute_initial_probs_with_covariates(
    eta_pi: np.ndarray,
    X: np.ndarray,
    n_states: int
) -> np.ndarray:
    """
    Compute initial state probabilities from covariates using Softmax.
    
    Args:
        eta_pi: Coefficient matrix of shape (n_covariates, n_states)
        X: Covariate matrix of shape (n_sequences, 1, n_covariates) for initial time
        n_states: Number of hidden states
        
    Returns:
        numpy array: Initial probabilities of shape (n_sequences, n_states)
    """
    n_sequences = X.shape[0]
    
    # Initialize initial probability matrix
    initial_probs = np.zeros((n_sequences, n_states))
    
    # For each sequence
    for seq_idx in range(n_sequences):
        # Get covariates for initial time point
        x_0 = X[seq_idx, 0, :]  # Shape: (n_covariates,)
        
        # Compute linear predictor
        # eta[i] = sum over covariates: x[c] * eta_pi[c, i]
        eta = np.zeros(n_states)
        for i in range(n_states):
            eta[i] = np.sum(x_0 * eta_pi[:, i])
        
        # Convert to probabilities using softmax
        initial_probs[seq_idx, :] = softmax(eta)
    
    return initial_probs
