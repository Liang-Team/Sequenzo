"""
@Author  : Yuqi Liang 梁彧祺
@File    : gradients_nhmm.py
@Time    : 2025-10-22 15:18
@Desc    : Analytical gradient computation for Non-homogeneous HMM

This module provides functions for computing analytical gradients of the
log-likelihood with respect to model parameters (eta_pi, eta_A, eta_B).
This is similar to seqHMM's gradient computation in objective_functions.R.

Note: This is a complex implementation. The gradients are computed using
the forward-backward algorithm and chain rule through the Softmax function.
"""

import numpy as np
from typing import Tuple
from .nhmm import NHMM
from .forward_backward_nhmm import _forward_nhmm, _backward_nhmm
from .nhmm_utils import softmax
from .utils import sequence_data_to_hmmlearn_format


def compute_gradient_nhmm(model: NHMM) -> np.ndarray:
    """
    Compute analytical gradient of log-likelihood with respect to parameters.
    
    The gradient is computed using the forward-backward algorithm and
    the chain rule through the Softmax parameterization.
    
    This is similar to seqHMM's gradient computation for NHMM.
    
    Args:
        model: Fitted NHMM model object
        
    Returns:
        numpy array: Flattened gradient vector [grad_eta_pi, grad_eta_A, grad_eta_B]
    """
    # Convert sequences to integer format
    X_int, lengths = sequence_data_to_hmmlearn_format(model.observations)
    n_sequences = len(lengths)
    
    # Compute probabilities
    initial_probs, transition_probs, emission_probs = model._compute_probs()
    
    # Initialize gradients
    grad_eta_pi = np.zeros_like(model.eta_pi)
    grad_eta_A = np.zeros_like(model.eta_A)
    grad_eta_B = np.zeros_like(model.eta_B)
    
    # Process each sequence
    for seq_idx in range(n_sequences):
        seq_length = lengths[seq_idx]
        start_idx = lengths[:seq_idx].sum()
        end_idx = start_idx + seq_length
        
        # Get sequence observations
        obs_seq = X_int[start_idx:end_idx, 0]
        
        # Get probabilities for this sequence
        seq_initial = initial_probs[seq_idx, :]
        seq_transition = transition_probs[seq_idx, :seq_length, :, :]
        seq_emission = emission_probs[seq_idx, :seq_length, :, :]
        
        # Get covariates for this sequence
        X_seq = model.X[seq_idx, :seq_length, :]  # Shape: (seq_length, n_covariates)
        
        # Compute forward and backward probabilities
        log_alpha = _forward_nhmm(seq_initial, seq_transition, seq_emission, obs_seq, model.n_states)
        log_beta = _backward_nhmm(seq_transition, seq_emission, obs_seq, model.n_states)
        
        # Compute log-likelihood for this sequence
        log_lik_seq = -np.inf
        for i in range(model.n_states):
            if log_lik_seq == -np.inf:
                log_lik_seq = log_alpha[i, seq_length-1]
            else:
                log_lik_seq = np.logaddexp(log_lik_seq, log_alpha[i, seq_length-1])
        
        # Compute posterior probabilities: gamma[i, t] = P(state_t = i | obs)
        # gamma[i, t] = alpha[i, t] * beta[i, t] / P(obs)
        gamma = np.zeros((model.n_states, seq_length))
        for t in range(seq_length):
            for i in range(model.n_states):
                gamma[i, t] = np.exp(log_alpha[i, t] + log_beta[i, t] - log_lik_seq)
        
        # Compute xi: xi[i, j, t] = P(state_t = i, state_{t+1} = j | obs)
        xi = np.zeros((model.n_states, model.n_states, seq_length - 1))
        for t in range(seq_length - 1):
            for i in range(model.n_states):
                for j in range(model.n_states):
                    log_xi = (
                        log_alpha[i, t] +
                        np.log(seq_transition[t, i, j] + 1e-10) +
                        np.log(seq_emission[t+1, j, obs_seq[t+1]] + 1e-10) +
                        log_beta[j, t+1] -
                        log_lik_seq
                    )
                    xi[i, j, t] = np.exp(log_xi)
        
        # Compute gradients using chain rule through Softmax
        # Gradient w.r.t. eta_pi (initial probabilities)
        grad_pi = _gradient_initial_probs(gamma, seq_initial, model.n_states)
        grad_eta_pi += _gradient_softmax_to_eta(grad_pi, seq_initial, X_seq[0, :], model.n_states)
        
        # Gradient w.r.t. eta_A (transition probabilities)
        for t in range(seq_length - 1):
            grad_A_t = _gradient_transition_probs(xi[:, :, t], gamma[:, t], seq_transition[t, :, :], model.n_states)
            grad_eta_A += _gradient_softmax_to_eta_transition(
                grad_A_t, seq_transition[t, :, :], X_seq[t, :], model.n_states
            )
        
        # Gradient w.r.t. eta_B (emission probabilities)
        for t in range(seq_length):
            grad_B_t = _gradient_emission_probs(gamma[:, t], obs_seq[t], seq_emission[t, :, :], model.n_states, model.n_symbols)
            grad_eta_B += _gradient_softmax_to_eta_emission(
                grad_B_t, seq_emission[t, :, :], X_seq[t, :], model.n_states, model.n_symbols
            )
    
    # Flatten gradients
    grad_flat = np.concatenate([
        grad_eta_pi.flatten(),
        grad_eta_A.flatten(),
        grad_eta_B.flatten()
    ])
    
    return grad_flat


def _gradient_initial_probs(gamma: np.ndarray, initial_probs: np.ndarray, n_states: int) -> np.ndarray:
    """
    Compute gradient of log-likelihood w.r.t. initial probabilities.
    
    Args:
        gamma: Posterior probabilities (n_states, T)
        initial_probs: Initial probabilities (n_states,)
        n_states: Number of states
        
    Returns:
        numpy array: Gradient w.r.t. initial probabilities (n_states,)
    """
    # Gradient: dL/dpi[i] = gamma[i, 0] / pi[i]
    grad = gamma[:, 0] / (initial_probs + 1e-10)
    return grad


def _gradient_transition_probs(
    xi: np.ndarray,
    gamma_t: np.ndarray,
    transition_probs: np.ndarray,
    n_states: int
) -> np.ndarray:
    """
    Compute gradient of log-likelihood w.r.t. transition probabilities.
    
    Args:
        xi: Joint probabilities (n_states, n_states)
        gamma_t: Posterior probabilities at time t (n_states,)
        transition_probs: Transition probabilities (n_states, n_states)
        n_states: Number of states
        
    Returns:
        numpy array: Gradient w.r.t. transition probabilities (n_states, n_states)
    """
    # Gradient: dL/dA[i, j] = xi[i, j] / A[i, j]
    grad = xi / (transition_probs + 1e-10)
    return grad


def _gradient_emission_probs(
    gamma_t: np.ndarray,
    obs: int,
    emission_probs: np.ndarray,
    n_states: int,
    n_symbols: int
) -> np.ndarray:
    """
    Compute gradient of log-likelihood w.r.t. emission probabilities.
    
    Args:
        gamma_t: Posterior probabilities at time t (n_states,)
        obs: Observed symbol (integer, 0-indexed)
        emission_probs: Emission probabilities (n_states, n_symbols)
        n_states: Number of states
        n_symbols: Number of symbols
        
    Returns:
        numpy array: Gradient w.r.t. emission probabilities (n_states, n_symbols)
    """
    # Gradient: dL/dB[i, j] = gamma[i] / B[i, j] if j == obs, else 0
    grad = np.zeros((n_states, n_symbols))
    for i in range(n_states):
        grad[i, obs] = gamma_t[i] / (emission_probs[i, obs] + 1e-10)
    return grad


def _gradient_softmax_to_eta(
    grad_gamma: np.ndarray,
    gamma: np.ndarray,
    x: np.ndarray,
    n_categories: int
) -> np.ndarray:
    """
    Compute gradient w.r.t. eta from gradient w.r.t. gamma (Softmax chain rule).
    
    For initial probabilities: gamma = softmax(eta), where eta = X @ eta_pi
    
    Args:
        grad_gamma: Gradient w.r.t. gamma (n_categories,)
        gamma: Probabilities (n_categories,)
        x: Covariates (n_covariates,)
        n_categories: Number of categories (n_states for initial probs)
        
    Returns:
        numpy array: Gradient w.r.t. eta_pi (n_covariates, n_states)
    """
    n_covariates = len(x)
    grad_eta = np.zeros((n_covariates, n_categories))
    
    # Chain rule: dL/deta = dL/dgamma * dgamma/deta
    # dgamma[i]/deta[j] = gamma[i] * (delta[i,j] - gamma[j])
    for c in range(n_covariates):
        for i in range(n_categories):
            for j in range(n_categories):
                if i == j:
                    dgamma_deta = gamma[i] * (1 - gamma[j])
                else:
                    dgamma_deta = -gamma[i] * gamma[j]
                grad_eta[c, i] += grad_gamma[j] * dgamma_deta * x[c]
    
    return grad_eta


def _gradient_softmax_to_eta_transition(
    grad_A: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    n_states: int
) -> np.ndarray:
    """
    Compute gradient w.r.t. eta_A from gradient w.r.t. A (Softmax chain rule).
    
    For transition probabilities: A[i, :] = softmax(eta[i, :]), where eta[i, j] = X @ eta_A[:, i, j]
    
    Args:
        grad_A: Gradient w.r.t. A (n_states, n_states)
        A: Transition probabilities (n_states, n_states)
        x: Covariates (n_covariates,)
        n_states: Number of states
        
    Returns:
        numpy array: Gradient w.r.t. eta_A (n_covariates, n_states, n_states)
    """
    n_covariates = len(x)
    grad_eta = np.zeros((n_covariates, n_states, n_states))
    
    # For each row i, apply Softmax chain rule
    for i in range(n_states):
        for c in range(n_covariates):
            for j in range(n_states):
                for k in range(n_states):
                    if j == k:
                        dA_deta = A[i, j] * (1 - A[i, k])
                    else:
                        dA_deta = -A[i, j] * A[i, k]
                    grad_eta[c, i, j] += grad_A[i, k] * dA_deta * x[c]
    
    return grad_eta


def _gradient_softmax_to_eta_emission(
    grad_B: np.ndarray,
    B: np.ndarray,
    x: np.ndarray,
    n_states: int,
    n_symbols: int
) -> np.ndarray:
    """
    Compute gradient w.r.t. eta_B from gradient w.r.t. B (Softmax chain rule).
    
    For emission probabilities: B[i, :] = softmax(eta[i, :]), where eta[i, j] = X @ eta_B[:, i, j]
    
    Args:
        grad_B: Gradient w.r.t. B (n_states, n_symbols)
        B: Emission probabilities (n_states, n_symbols)
        x: Covariates (n_covariates,)
        n_states: Number of states
        n_symbols: Number of symbols
        
    Returns:
        numpy array: Gradient w.r.t. eta_B (n_covariates, n_states, n_symbols)
    """
    n_covariates = len(x)
    grad_eta = np.zeros((n_covariates, n_states, n_symbols))
    
    # For each row i, apply Softmax chain rule
    for i in range(n_states):
        for c in range(n_covariates):
            for j in range(n_symbols):
                for k in range(n_symbols):
                    if j == k:
                        dB_deta = B[i, j] * (1 - B[i, k])
                    else:
                        dB_deta = -B[i, j] * B[i, k]
                    grad_eta[c, i, j] += grad_B[i, k] * dB_deta * x[c]
    
    return grad_eta
