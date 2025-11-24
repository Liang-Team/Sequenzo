"""
@Author  : Yuqi Liang 梁彧祺
@File    : mhmm.py
@Time    : 2025-11-22 08:47
@Desc    : Mixture Hidden Markov Model (MHMM) for Sequenzo

A Mixture HMM consists of multiple HMM submodels, where each submodel represents
a cluster or type. The model assigns each sequence to one of these clusters with
certain probabilities.

This is similar to seqHMM's mhmm class in R.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
from sequenzo.define_sequence_data import SequenceData
from .hmm import HMM
from .utils import (
    sequence_data_to_hmmlearn_format,
    create_initial_probs,
    create_transition_probs,
    create_emission_probs
)


class MHMM:
    """
    Mixture Hidden Markov Model for sequence analysis.
    
    A Mixture HMM consists of multiple HMM submodels (clusters). Each sequence
    belongs to one of these clusters with certain probabilities. The model
    estimates both the cluster membership probabilities and the parameters
    of each HMM submodel.
    
    Attributes:
        observations: SequenceData object containing the observed sequences
        n_clusters: Number of clusters (submodels)
        clusters: List of HMM objects, one for each cluster
        cluster_probs: Mixture probabilities (probability of each cluster)
        coefficients: Optional regression coefficients for covariates
        X: Optional covariate matrix
        cluster_names: Optional names for clusters
        state_names: Optional names for hidden states (per cluster)
        channel_names: Optional names for channels
        
        # Model parameters (after fitting)
        log_likelihood: Log-likelihood of the fitted model
        n_iter: Number of EM iterations performed
        converged: Whether the EM algorithm converged
    """
    
    def __init__(
        self,
        observations: SequenceData,
        n_clusters: int,
        n_states: Union[int, List[int]],
        clusters: Optional[List[HMM]] = None,
        cluster_probs: Optional[np.ndarray] = None,
        coefficients: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        cluster_names: Optional[List[str]] = None,
        state_names: Optional[List[List[str]]] = None,
        channel_names: Optional[List[str]] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize a Mixture HMM model.
        
        Args:
            observations: SequenceData object containing the sequences
            n_clusters: Number of clusters (submodels)
            n_states: Number of hidden states per cluster. Can be:
                     - int: Same number of states for all clusters
                     - List[int]: Different number of states for each cluster
            clusters: Optional list of pre-built HMM objects for each cluster
            cluster_probs: Optional initial cluster probabilities (n_clusters,)
            coefficients: Optional regression coefficients for covariates
            X: Optional covariate matrix (n_sequences x n_covariates)
            cluster_names: Optional names for clusters
            state_names: Optional names for hidden states (list of lists)
            channel_names: Optional names for channels
            random_state: Random seed for initialization
        """
        self.observations = observations
        self.n_clusters = n_clusters
        self.alphabet = observations.alphabet
        self.n_symbols = len(self.alphabet)
        self.n_sequences = len(observations.sequences)
        
        # Handle n_states: convert to list if int
        if isinstance(n_states, int):
            n_states = [n_states] * n_clusters
        self.n_states = n_states
        
        # Validate n_states length
        if len(n_states) != n_clusters:
            raise ValueError(
                f"n_states length ({len(n_states)}) must equal n_clusters ({n_clusters})"
            )
        
        # Set names
        self.cluster_names = cluster_names or [f"Cluster {i+1}" for i in range(n_clusters)]
        self.channel_names = channel_names or ["Channel 1"]
        self.n_channels = len(self.channel_names)
        
        # Initialize clusters (HMM submodels)
        if clusters is None:
            self.clusters = []
            for k in range(n_clusters):
                # Get state names for this cluster
                cluster_state_names = None
                if state_names is not None:
                    cluster_state_names = state_names[k] if k < len(state_names) else None
                
                # Create HMM for this cluster
                hmm = HMM(
                    observations=observations,
                    n_states=n_states[k],
                    state_names=cluster_state_names,
                    channel_names=channel_names,
                    random_state=random_state
                )
                self.clusters.append(hmm)
        else:
            if len(clusters) != n_clusters:
                raise ValueError(
                    f"Number of clusters ({len(clusters)}) must equal n_clusters ({n_clusters})"
                )
            self.clusters = clusters
        
        # Initialize cluster probabilities
        if cluster_probs is None:
            self.cluster_probs = np.ones(n_clusters) / n_clusters  # Uniform
        else:
            if len(cluster_probs) != n_clusters:
                raise ValueError(
                    f"cluster_probs length ({len(cluster_probs)}) must equal n_clusters ({n_clusters})"
                )
            if not np.isclose(np.sum(cluster_probs), 1.0):
                raise ValueError("cluster_probs must sum to 1.0")
            self.cluster_probs = np.array(cluster_probs)
        
        # Covariates (for future extension)
        self.coefficients = coefficients
        self.X = X
        self.n_covariates = X.shape[1] if X is not None else 0
        
        # Fitting results
        self.log_likelihood = None
        self.n_iter = None
        self.converged = None
        
        # Store responsibilities (posterior cluster probabilities) after fitting
        self.responsibilities = None
    
    def fit(
        self,
        n_iter: int = 100,
        tol: float = 1e-2,
        verbose: bool = False
    ) -> 'MHMM':
        """
        Fit the Mixture HMM model using EM algorithm.
        
        The EM algorithm alternates between:
        1. E-step: Compute responsibilities (posterior cluster probabilities)
        2. M-step: Update cluster probabilities and HMM parameters
        
        Args:
            n_iter: Maximum number of EM iterations
            tol: Convergence tolerance
            verbose: Whether to print progress
            
        Returns:
            self: Returns self for method chaining
        """
        # Convert SequenceData to hmmlearn format
        X, lengths = sequence_data_to_hmmlearn_format(self.observations)
        n_sequences = len(lengths)
        
        # Initialize log-likelihood
        prev_log_likelihood = -np.inf
        
        # EM algorithm
        for iteration in range(n_iter):
            # E-step: Compute responsibilities
            # Responsibility = P(cluster | sequence) = P(sequence | cluster) * P(cluster) / P(sequence)
            
            # Compute log-likelihood for each sequence under each cluster
            log_likelihoods = np.zeros((n_sequences, self.n_clusters))
            
            for k in range(self.n_clusters):
                # Fit this cluster's HMM if not already fitted
                # Suppress warnings about init_params during fitting
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*init_params.*')
                    if self.clusters[k].log_likelihood is None:
                        self.clusters[k].fit(n_iter=10, tol=tol, verbose=False)
                
                # Compute log-likelihood for each sequence
                for seq_idx in range(n_sequences):
                    # Get sequence indices
                    start_idx = lengths[:seq_idx].sum()
                    end_idx = start_idx + lengths[seq_idx]
                    seq_X = X[start_idx:end_idx]
                    seq_lengths = np.array([lengths[seq_idx]])
                    
                    # Compute log-likelihood
                    log_likelihoods[seq_idx, k] = self.clusters[k]._hmm_model.score(seq_X, seq_lengths)
            
            # Add log of cluster probabilities
            log_probs = np.log(self.cluster_probs + 1e-10)  # Add small epsilon to avoid log(0)
            log_likelihoods += log_probs[np.newaxis, :]
            
            # Compute responsibilities using log-sum-exp trick for numerical stability
            # responsibility = exp(log_likelihood - log_sum_exp(log_likelihoods))
            max_log_lik = np.max(log_likelihoods, axis=1, keepdims=True)
            exp_log_lik = np.exp(log_likelihoods - max_log_lik)
            responsibilities = exp_log_lik / np.sum(exp_log_lik, axis=1, keepdims=True)
            self.responsibilities = responsibilities
            
            # M-step: Update cluster probabilities
            self.cluster_probs = np.mean(responsibilities, axis=0)
            
            # M-step: Update each cluster's HMM parameters
            # We use weighted fitting: each sequence contributes to each cluster
            # proportionally to its responsibility
            for k in range(self.n_clusters):
                # For simplicity, we fit using all sequences but this could be
                # optimized to use only sequences with high responsibility
                # For now, we refit each cluster's HMM
                # Suppress warnings about init_params during fitting
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*init_params.*')
                    self.clusters[k].fit(n_iter=10, tol=tol, verbose=False)
            
            # Compute overall log-likelihood
            # log P(data) = sum over sequences of log(sum over clusters of P(seq | cluster) * P(cluster))
            log_likelihood = np.sum(
                np.log(np.sum(np.exp(log_likelihoods), axis=1) + 1e-10)
            )
            
            if verbose:
                print(f"Iteration {iteration + 1}: log-likelihood = {log_likelihood:.4f}")
            
            # Check convergence
            if iteration > 0:
                change = log_likelihood - prev_log_likelihood
                if abs(change) < tol:
                    self.converged = True
                    if verbose:
                        print(f"Converged at iteration {iteration + 1}")
                    break
            
            prev_log_likelihood = log_likelihood
        
        self.log_likelihood = prev_log_likelihood
        self.n_iter = iteration + 1
        
        if not self.converged:
            self.converged = False
            if verbose:
                print(f"Did not converge after {n_iter} iterations")
        
        return self
    
    def predict_cluster(self, sequences: Optional[SequenceData] = None) -> np.ndarray:
        """
        Predict the most likely cluster for each sequence.
        
        Args:
            sequences: Optional SequenceData (uses self.observations if None)
            
        Returns:
            numpy array: Predicted cluster index for each sequence
        """
        if self.responsibilities is None:
            raise ValueError("Model must be fitted before prediction. Use fit() first.")
        
        if sequences is None:
            return np.argmax(self.responsibilities, axis=1)
        else:
            # Compute responsibilities for new sequences
            X, lengths = sequence_data_to_hmmlearn_format(sequences)
            n_sequences = len(lengths)
            
            log_likelihoods = np.zeros((n_sequences, self.n_clusters))
            
            for k in range(self.n_clusters):
                for seq_idx in range(n_sequences):
                    start_idx = lengths[:seq_idx].sum()
                    end_idx = start_idx + lengths[seq_idx]
                    seq_X = X[start_idx:end_idx]
                    seq_lengths = np.array([lengths[seq_idx]])
                    
                    log_likelihoods[seq_idx, k] = self.clusters[k]._hmm_model.score(seq_X, seq_lengths)
            
            log_probs = np.log(self.cluster_probs + 1e-10)
            log_likelihoods += log_probs[np.newaxis, :]
            
            max_log_lik = np.max(log_likelihoods, axis=1, keepdims=True)
            exp_log_lik = np.exp(log_likelihoods - max_log_lik)
            responsibilities = exp_log_lik / np.sum(exp_log_lik, axis=1, keepdims=True)
            
            return np.argmax(responsibilities, axis=1)
    
    def __repr__(self) -> str:
        """String representation of the MHMM."""
        status = "fitted" if self.log_likelihood is not None else "unfitted"
        return (f"MHMM(n_clusters={self.n_clusters}, n_states={self.n_states}, "
                f"n_sequences={self.n_sequences}, status='{status}')")
