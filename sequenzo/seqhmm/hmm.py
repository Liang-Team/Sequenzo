"""
@Author  : Yuqi Liang 梁彧祺；Yapeng Wei 卫亚鹏
@File    : hmm.py
@Time    : 2025-11-13 16:20
@Desc    : Base HMM class for Sequenzo

This module provides the HMM class that wraps hmmlearn's CategoricalHMM
and adapts it for use with Sequenzo's SequenceData format.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from hmmlearn.hmm import CategoricalHMM
from sequenzo.define_sequence_data import SequenceData
from .utils import (
    sequence_data_to_hmmlearn_format,
    int_to_state_mapping,
    state_to_int_mapping
)
from .multichannel_utils import multichannel_to_hmmlearn_format, prepare_multichannel_data


_FLOAT_TINY = np.finfo(float).tiny


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

    @property
    def has_complete_parameters(self) -> bool:
        """Return True when the model has enough parameters for inference."""
        if self.initial_probs is None or self.transition_probs is None or self.emission_probs is None:
            return False
        if self.n_channels == 1:
            return isinstance(self.emission_probs, np.ndarray)
        return isinstance(self.emission_probs, list) and len(self.emission_probs) == self.n_channels

    def _check_inference_ready(self) -> None:
        if not self.has_complete_parameters:
            raise ValueError(
                "Model parameters are incomplete. Fit the model or provide "
                "initial_probs, transition_probs, and emission_probs."
            )

    def _multichannel_channels(
        self,
        sequences: Optional[Union[SequenceData, List[SequenceData]]] = None
    ) -> List[SequenceData]:
        if sequences is None:
            return self.channels

        channels, _, _ = prepare_multichannel_data(sequences)
        if len(channels) != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {len(channels)}"
            )
        return channels

    def _multichannel_arrays(
        self,
        sequences: Optional[Union[SequenceData, List[SequenceData]]] = None
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        channels = self._multichannel_channels(sequences)
        X_list, lengths = multichannel_to_hmmlearn_format(channels)
        observations = [X[:, 0].astype(int, copy=False) for X in X_list]
        return observations, lengths

    def _multichannel_emission_matrix(
        self,
        obs_list: List[np.ndarray]
    ) -> np.ndarray:
        emissions = np.ones((len(obs_list[0]), self.n_states), dtype=float)

        for ch_idx, obs in enumerate(obs_list):
            emission_probs = np.asarray(self.emission_probs[ch_idx], dtype=float)
            if np.any(obs < 0) or np.any(obs >= emission_probs.shape[1]):
                raise ValueError(
                    f"Channel {ch_idx} contains observations outside the fitted alphabet."
                )
            emissions *= emission_probs[:, obs].T

        return emissions

    @staticmethod
    def _normalize_probabilities(values: np.ndarray) -> Tuple[np.ndarray, float]:
        total = float(np.sum(values))
        if not np.isfinite(total) or total <= 0.0:
            return np.ones_like(values, dtype=float) / len(values), _FLOAT_TINY
        return values / total, total

    def _multichannel_sequence_log_likelihood(self, obs_list: List[np.ndarray]) -> float:
        initial = np.asarray(self.initial_probs, dtype=float)
        transition = np.asarray(self.transition_probs, dtype=float)
        emission = self._multichannel_emission_matrix(obs_list)

        alpha, scale = self._normalize_probabilities(initial * emission[0])
        log_likelihood = float(np.log(max(scale, _FLOAT_TINY)))

        for t in range(1, emission.shape[0]):
            alpha_t = (alpha @ transition) * emission[t]
            alpha, scale = self._normalize_probabilities(alpha_t)
            log_likelihood += float(np.log(max(scale, _FLOAT_TINY)))

        return log_likelihood

    def _multichannel_forward_backward(
        self,
        sequences: Optional[Union[SequenceData, List[SequenceData]]] = None
    ) -> Tuple[np.ndarray, float]:
        self._check_inference_ready()

        observations, lengths = self._multichannel_arrays(sequences)
        initial = np.asarray(self.initial_probs, dtype=float)
        transition = np.asarray(self.transition_probs, dtype=float)

        posteriors = []
        log_likelihood = 0.0
        start = 0

        for seq_length in lengths:
            end = start + int(seq_length)
            obs_list = [obs[start:end] for obs in observations]
            emission = self._multichannel_emission_matrix(obs_list)

            alpha = np.empty((seq_length, self.n_states), dtype=float)
            scales = np.empty(seq_length, dtype=float)
            alpha[0], scales[0] = self._normalize_probabilities(initial * emission[0])

            for t in range(1, seq_length):
                alpha_t = (alpha[t - 1] @ transition) * emission[t]
                alpha[t], scales[t] = self._normalize_probabilities(alpha_t)

            beta = np.ones((seq_length, self.n_states), dtype=float)
            for t in range(seq_length - 2, -1, -1):
                beta[t] = transition @ (emission[t + 1] * beta[t + 1])
                beta[t] /= max(scales[t + 1], _FLOAT_TINY)

            gamma = alpha * beta
            gamma_sums = gamma.sum(axis=1, keepdims=True)
            gamma_sums[gamma_sums <= 0.0] = 1.0
            posteriors.append(gamma / gamma_sums)
            log_likelihood += float(np.sum(np.log(np.maximum(scales, _FLOAT_TINY))))
            start = end

        return np.vstack(posteriors), log_likelihood

    def _multichannel_viterbi(
        self,
        sequences: Optional[Union[SequenceData, List[SequenceData]]] = None
    ) -> np.ndarray:
        self._check_inference_ready()

        observations, lengths = self._multichannel_arrays(sequences)
        log_initial = np.log(np.maximum(np.asarray(self.initial_probs, dtype=float), _FLOAT_TINY))
        log_transition = np.log(np.maximum(np.asarray(self.transition_probs, dtype=float), _FLOAT_TINY))

        paths = []
        start = 0

        for seq_length in lengths:
            end = start + int(seq_length)
            obs_list = [obs[start:end] for obs in observations]
            emission = self._multichannel_emission_matrix(obs_list)
            log_emission = np.log(np.maximum(emission, _FLOAT_TINY))

            delta = np.empty((seq_length, self.n_states), dtype=float)
            psi = np.zeros((seq_length, self.n_states), dtype=int)
            delta[0] = log_initial + log_emission[0]

            for t in range(1, seq_length):
                scores = delta[t - 1][:, None] + log_transition
                psi[t] = np.argmax(scores, axis=0)
                delta[t] = scores[psi[t], np.arange(self.n_states)] + log_emission[t]

            path = np.empty(seq_length, dtype=int)
            path[-1] = int(np.argmax(delta[-1]))
            for t in range(seq_length - 2, -1, -1):
                path[t] = psi[t + 1, path[t + 1]]

            paths.append(path)
            start = end

        return np.concatenate(paths)

    @staticmethod
    def _sequence_keys_from_lengths(
        observations: List[np.ndarray],
        lengths: np.ndarray
    ) -> Dict[Tuple[Tuple[int, ...], ...], int]:
        counts = {}
        start = 0
        for seq_length in lengths:
            end = start + int(seq_length)
            key = tuple(tuple(obs[start:end].tolist()) for obs in observations)
            counts[key] = counts.get(key, 0) + 1
            start = end
        return counts

    def _single_channel_score_compressed(self, sequences: SequenceData) -> float:
        dense = self._dense_single_channel_matrix(sequences)
        if dense is not None:
            unique_rows, counts = np.unique(dense, axis=0, return_counts=True)

            log_likelihood = 0.0
            seq_length = dense.shape[1]
            seq_length_array = np.array([seq_length], dtype=np.int32)
            for row, count in zip(unique_rows, counts):
                sequence = row.reshape(-1, 1)
                log_likelihood += int(count) * float(
                    self._hmm_model.score(sequence, seq_length_array)
                )
            return log_likelihood

        X, lengths = sequence_data_to_hmmlearn_format(sequences)

        if np.all(lengths == lengths[0]):
            seq_length = int(lengths[0])
            matrix = X[:, 0].reshape(len(lengths), seq_length)
            unique_rows, counts = np.unique(matrix, axis=0, return_counts=True)

            log_likelihood = 0.0
            for row, count in zip(unique_rows, counts):
                sequence = row.astype(np.int32, copy=False).reshape(-1, 1)
                log_likelihood += int(count) * float(
                    self._hmm_model.score(
                        sequence,
                        np.array([seq_length], dtype=np.int32),
                    )
                )
            return log_likelihood

        counts = self._sequence_keys_from_lengths([X[:, 0].astype(int, copy=False)], lengths)

        log_likelihood = 0.0
        for key, count in counts.items():
            sequence = np.asarray(key[0], dtype=np.int32).reshape(-1, 1)
            seq_length = np.array([sequence.shape[0]], dtype=np.int32)
            log_likelihood += count * float(self._hmm_model.score(sequence, seq_length))

        return log_likelihood

    def _multichannel_score_compressed(
        self,
        sequences: Optional[Union[SequenceData, List[SequenceData]]] = None
    ) -> float:
        self._check_inference_ready()
        channels = self._multichannel_channels(sequences)
        dense_channels = self._dense_multichannel_matrices(channels)
        if dense_channels is not None:
            n_sequences, seq_length = dense_channels[0].shape
            combined = np.hstack(dense_channels)
            unique_rows, counts = np.unique(combined, axis=0, return_counts=True)

            log_likelihood = 0.0
            for row, count in zip(unique_rows, counts):
                obs_list = [
                    row[ch * seq_length:(ch + 1) * seq_length].astype(int, copy=False)
                    for ch in range(self.n_channels)
                ]
                log_likelihood += int(count) * self._multichannel_sequence_log_likelihood(obs_list)
            return float(log_likelihood)

        observations, lengths = self._multichannel_arrays(sequences)

        if np.all(lengths == lengths[0]):
            seq_length = int(lengths[0])
            n_sequences = len(lengths)
            matrices = [
                obs.reshape(n_sequences, seq_length) for obs in observations
            ]
            combined = np.hstack(matrices)
            unique_rows, counts = np.unique(combined, axis=0, return_counts=True)

            log_likelihood = 0.0
            for row, count in zip(unique_rows, counts):
                obs_list = [
                    row[ch * seq_length:(ch + 1) * seq_length].astype(int, copy=False)
                    for ch in range(self.n_channels)
                ]
                log_likelihood += int(count) * self._multichannel_sequence_log_likelihood(obs_list)
            return float(log_likelihood)

        counts = self._sequence_keys_from_lengths(observations, lengths)

        log_likelihood = 0.0
        for key, count in counts.items():
            obs_list = [np.asarray(channel, dtype=int) for channel in key]
            log_likelihood += count * self._multichannel_sequence_log_likelihood(obs_list)

        return float(log_likelihood)

    def _dense_single_channel_matrix(
        self,
        sequences: SequenceData
    ) -> Optional[np.ndarray]:
        numeric = sequences.to_numeric()
        if numeric.ndim != 2:
            return None
        if not np.all((numeric >= 1) & (numeric <= self.n_symbols)):
            return None
        return np.ascontiguousarray(numeric.astype(np.int32, copy=False) - 1)

    def _dense_multichannel_matrices(
        self,
        channels: List[SequenceData]
    ) -> Optional[List[np.ndarray]]:
        matrices = []
        shape = None
        for ch_idx, channel in enumerate(channels):
            numeric = channel.to_numeric()
            if numeric.ndim != 2:
                return None
            n_symbols = self.n_symbols[ch_idx]
            if not np.all((numeric >= 1) & (numeric <= n_symbols)):
                return None
            if shape is None:
                shape = numeric.shape
            elif numeric.shape != shape:
                return None
            matrices.append(np.ascontiguousarray(numeric.astype(np.int32, copy=False) - 1))
        return matrices
    
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
            from .multichannel_emission import fit_multichannel_hmm
            fit_multichannel_hmm(self, n_iter=n_iter, tol=tol, verbose=verbose)
        
        return self
    
    def predict(
        self,
        sequences: Optional[Union[SequenceData, List[SequenceData]]] = None
    ) -> np.ndarray:
        """
        Predict the most likely hidden state sequence using Viterbi algorithm.
        
        Args:
            sequences: Optional SequenceData to predict (uses self.observations if None)
            
        Returns:
            numpy array: Predicted hidden states for each sequence
        """
        if self.n_channels > 1:
            return self._multichannel_viterbi(sequences)

        if sequences is None:
            sequences = self.observations
        
        X, lengths = sequence_data_to_hmmlearn_format(sequences)
        states = self._hmm_model.predict(X, lengths)
        
        return states
    
    def predict_proba(
        self,
        sequences: Optional[Union[SequenceData, List[SequenceData]]] = None
    ) -> np.ndarray:
        """
        Compute posterior probabilities of hidden states.
        
        Args:
            sequences: Optional SequenceData (uses self.observations if None)
            
        Returns:
            numpy array: Posterior probabilities for each time point
        """
        if self.n_channels > 1:
            posteriors, _ = self._multichannel_forward_backward(sequences)
            return posteriors

        if sequences is None:
            sequences = self.observations
        
        X, lengths = sequence_data_to_hmmlearn_format(sequences)
        posteriors = self._hmm_model.predict_proba(X, lengths)
        
        return posteriors
    
    def score(
        self,
        sequences: Optional[Union[SequenceData, List[SequenceData]]] = None,
        compress: bool = False
    ) -> float:
        """
        Compute the log-likelihood of sequences under the model.
        
        Args:
            sequences: Optional SequenceData (uses self.observations if None)
            
        Returns:
            float: Log-likelihood
        """
        if self.n_channels > 1:
            if compress:
                return self._multichannel_score_compressed(sequences)
            _, log_likelihood = self._multichannel_forward_backward(sequences)
            return log_likelihood

        if sequences is None:
            sequences = self.observations

        if compress:
            return self._single_channel_score_compressed(sequences)
        
        X, lengths = sequence_data_to_hmmlearn_format(sequences)
        return self._hmm_model.score(X, lengths)
    
    def __repr__(self) -> str:
        """String representation of the HMM."""
        status = "fitted" if self.log_likelihood is not None else "unfitted"
        return (f"HMM(n_states={self.n_states}, n_symbols={self.n_symbols}, "
                f"n_sequences={self.n_sequences}, status='{status}')")
