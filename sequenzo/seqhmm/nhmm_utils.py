"""
@Author  : Yuqi Liang 梁彧祺
@File    : nhmm_utils.py
@Time    : 2025-11-23 10:20
@Desc    : Utility functions for Non-homogeneous HMM

This module provides utility functions for NHMM, including Softmax parameterization
and gradient computation.
"""

import numpy as np


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
    if not np.isfinite(x).all():
        raise ValueError("linear predictors must contain only finite values")

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
    with np.errstate(over="ignore", invalid="ignore"):
        eta = np.einsum("ntc,cij->ntij", X, eta_A)
    return softmax(eta, axis=-1)


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
    with np.errstate(over="ignore", invalid="ignore"):
        eta = np.einsum("ntc,cik->ntik", X, eta_B)
    return softmax(eta, axis=-1)


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
    with np.errstate(over="ignore", invalid="ignore"):
        eta = X[:, 0, :] @ eta_pi
    return softmax(eta, axis=-1)
