"""
@Author  : Yuqi Liang 梁彧祺
@File    : hmm.py
@Time    : 2025-11-13 16:20
@Desc    : Base HMM class for Sequenzo

This module provides the HMM class that wraps hmmlearn's CategoricalHMM
and adapts it for use with Sequenzo's SequenceData format.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
from hmmlearn.hmm import CategoricalHMM
from sequenzo.define_sequence_data import SequenceData
from .utils import (
    sequence_data_to_hmmlearn_format,
    int_to_state_mapping,
    state_to_int_mapping
)
from .multichannel_utils import prepare_multichannel_data


class HMM:
    """
    Hidden Markov Model for sequence analysis.
    
    This class wraps hmmlearn's CategoricalHMM and provides a Sequenzo-friendly
    interface that works with SequenceData objects.
    
    Attributes:
        observations: SequenceData object containing the observed sequences
        n_states: Number of hidden states
        n_symbols: Number of observed symbols (alphabet size)
        alphabet: List of observed state symbols
        state_names: Optional names for hidden states
        channel_names: Optional names for channels (for multichannel data)
        length_of_sequences: Maximum sequence length
        sequence_lengths: Array of individual sequence lengths
        n_sequences: Number of sequences
        n_channels: Number of channels (currently 1 for single-channel)
        
        # Model parameters (after fitting)
        initial_probs: Initial state probabilities
        transition_probs: Transition probability matrix
        emission_probs: Emission probability matrix
        
        # hmmlearn model
        _hmm_model: Internal hmmlearn CategoricalHMM model
    """
    
    def __init__(
        self,
        observations: Union[SequenceData, List[SequenceData]],
        n_states: int,
        initial_probs: Optional[np.ndarray] = None,
        transition_probs: Optional[np.ndarray] = None,
        emission_probs: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        state_names: Optional[List[str]] = None,
        channel_names: Optional[List[str]] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize an HMM model.
        
        Args:
            observations: SequenceData object or list of SequenceData objects (for multichannel)
            n_states: Number of hidden states
            initial_probs: Optional initial state probabilities (n_states,)
            transition_probs: Optional transition matrix (n_states x n_states)
            emission_probs: Optional emission matrix (n_states x n_symbols) or
                          list of matrices (one per channel for multichannel)
            state_names: Optional names for hidden states
            channel_names: Optional names for channels
            random_state: Random seed for initialization
        """
        # Handle multichannel data
        channels, channel_names_list, alphabets = prepare_multichannel_data(observations)
        self.channels = channels
        self.n_channels = len(channels)
        
        # For single channel, store as observations for backward compatibility
        if self.n_channels == 1:
            self.observations = channels[0]
            self.alphabet = alphabets[0]
        else:
            # For multichannel, store first channel as primary (for compatibility)
            self.observations = channels[0]
            self.alphabet = alphabets[0]
        
        self.alphabets = alphabets
        self.n_symbols = [len(alph) for alph in alphabets]
        
        # For single channel, use single n_symbols
        if self.n_channels == 1:
            self.n_symbols = self.n_symbols[0]
        
        self.n_states = n_states
        
        # Store metadata
        self.state_names = state_names or [f"State {i+1}" for i in range(n_states)]
        self.channel_names = channel_names or channel_names_list
        
        # Get sequence information (use first channel for sequence info)
        self.sequence_lengths = np.array([len(seq) for seq in channels[0].sequences])
        self.length_of_sequences = int(self.sequence_lengths.max())
        self.n_sequences = len(channels[0].sequences)
        
        # Create mappings
        self._int_to_state = int_to_state_mapping(self.alphabet)
        self._state_to_int = state_to_int_mapping(self.alphabet)
        
        # Initialize hmmlearn model (only for single channel)
        # For multichannel, we'll need custom implementation
        if self.n_channels == 1:
            self._hmm_model = CategoricalHMM(
                n_components=n_states,
                n_features=self.n_symbols,
                random_state=random_state,
                n_iter=100,  # Default max iterations
                tol=1e-2,    # Default tolerance
                verbose=False
            )
            
            # Set initial parameters if provided
            # When custom parameters are provided, we need to remove the corresponding
            # letters from init_params to prevent hmmlearn from re-initializing them
            # 's' = startprob, 't' = transmat, 'e' = emissionprob
            if initial_probs is not None:
                self._hmm_model.startprob_ = initial_probs
                # Remove 's' from init_params so startprob won't be re-initialized during fit
                self._hmm_model.init_params = self._hmm_model.init_params.replace('s', '')
            
            if transition_probs is not None:
                self._hmm_model.transmat_ = transition_probs
                # Remove 't' from init_params so transmat won't be re-initialized during fit
                self._hmm_model.init_params = self._hmm_model.init_params.replace('t', '')
            
            if emission_probs is not None:
                self._hmm_model.emissionprob_ = emission_probs
                # Remove 'e' from init_params so emissionprob won't be re-initialized during fit
                self._hmm_model.init_params = self._hmm_model.init_params.replace('e', '')
        else:
            # Multichannel: hmmlearn doesn't support this directly
            # We'll implement custom fitting
            self._hmm_model = None
            if emission_probs is not None and isinstance(emission_probs, list):
                if len(emission_probs) != self.n_channels:
                    raise ValueError(
                        f"emission_probs list length ({len(emission_probs)}) must equal n_channels ({self.n_channels})"
                    )
        
        # Store parameters (will be updated after fitting)
        self.initial_probs = initial_probs
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs
        
        # Fitting results
        self.log_likelihood = None
        self.n_iter = None
        self.converged = None
    
    def fit(
        self,
        n_iter: int = 100,
        tol: float = 1e-2,
        verbose: bool = False
    ) -> 'HMM':
        """
        Fit the HMM model to the observations using EM algorithm.
        
        For single-channel data, uses hmmlearn's EM algorithm.
        For multichannel data, uses custom multichannel EM algorithm.
        
        Args:
            n_iter: Maximum number of EM iterations
            tol: Convergence tolerance
            verbose: Whether to print progress
            
        Returns:
            self: Returns self for method chaining
        """
        if self.n_channels == 1:
            # Single channel: use hmmlearn
            X, lengths = sequence_data_to_hmmlearn_format(self.observations)
            
            # Ensure init_params is correctly set before fitting
            # Remove letters from init_params if we have custom parameters
            if self.initial_probs is not None:
                self._hmm_model.startprob_ = self.initial_probs.copy()
                # Remove 's' from init_params to prevent re-initialization
                if 's' in self._hmm_model.init_params:
                    self._hmm_model.init_params = self._hmm_model.init_params.replace('s', '')
            
            if self.transition_probs is not None:
                self._hmm_model.transmat_ = self.transition_probs.copy()
                # Remove 't' from init_params to prevent re-initialization
                if 't' in self._hmm_model.init_params:
                    self._hmm_model.init_params = self._hmm_model.init_params.replace('t', '')
            
            if self.emission_probs is not None:
                self._hmm_model.emissionprob_ = self.emission_probs.copy()
                # Remove 'e' from init_params to prevent re-initialization
                if 'e' in self._hmm_model.init_params:
                    self._hmm_model.init_params = self._hmm_model.init_params.replace('e', '')
            
            # Update hmmlearn model parameters
            self._hmm_model.n_iter = n_iter
            self._hmm_model.tol = tol
            self._hmm_model.verbose = verbose
            
            # Fit the model, suppressing warnings about init_params
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*init_params.*')
                warnings.filterwarnings('ignore', message='.*overwritten during initialization.*')
                self._hmm_model.fit(X, lengths)
            
            # Extract fitted parameters
            self.initial_probs = self._hmm_model.startprob_.copy()
            self.transition_probs = self._hmm_model.transmat_.copy()
            self.emission_probs = self._hmm_model.emissionprob_.copy()
            
            # Store fitting results
            self.log_likelihood = self._hmm_model.score(X, lengths)
            self.n_iter = self._hmm_model.monitor_.iter
            self.converged = self._hmm_model.monitor_.converged
        else:
            # Multichannel: use custom EM algorithm
            from .multichannel_em import fit_multichannel_hmm
            fit_multichannel_hmm(self, n_iter=n_iter, tol=tol, verbose=verbose)
        
        return self
    
    def predict(self, sequences: Optional[SequenceData] = None) -> np.ndarray:
        """
        Predict the most likely hidden state sequence using Viterbi algorithm.
        
        Args:
            sequences: Optional SequenceData to predict (uses self.observations if None)
            
        Returns:
            numpy array: Predicted hidden states for each sequence
        """
        if sequences is None:
            sequences = self.observations
        
        X, lengths = sequence_data_to_hmmlearn_format(sequences)
        states = self._hmm_model.predict(X, lengths)
        
        return states
    
    def predict_proba(self, sequences: Optional[SequenceData] = None) -> np.ndarray:
        """
        Compute posterior probabilities of hidden states.
        
        Args:
            sequences: Optional SequenceData (uses self.observations if None)
            
        Returns:
            numpy array: Posterior probabilities for each time point
        """
        if sequences is None:
            sequences = self.observations
        
        X, lengths = sequence_data_to_hmmlearn_format(sequences)
        posteriors = self._hmm_model.predict_proba(X, lengths)
        
        return posteriors
    
    def score(self, sequences: Optional[SequenceData] = None) -> float:
        """
        Compute the log-likelihood of sequences under the model.
        
        Args:
            sequences: Optional SequenceData (uses self.observations if None)
            
        Returns:
            float: Log-likelihood
        """
        if sequences is None:
            sequences = self.observations
        
        X, lengths = sequence_data_to_hmmlearn_format(sequences)
        return self._hmm_model.score(X, lengths)
    
    def __repr__(self) -> str:
        """String representation of the HMM."""
        status = "fitted" if self.log_likelihood is not None else "unfitted"
        return (f"HMM(n_states={self.n_states}, n_symbols={self.n_symbols}, "
                f"n_sequences={self.n_sequences}, status='{status}')")
