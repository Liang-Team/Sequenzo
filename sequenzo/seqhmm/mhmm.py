"""
@Author  : Yuqi Liang 梁彧祺; Yapeng Wei 卫亚鹏
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
from scipy.special import logsumexp
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

        # Bug 3 fix: add sequence_lengths for model_comparison.py compatibility
        self.sequence_lengths = np.array(
            [len(seq) for seq in observations.sequences])
        self.length_of_sequences = int(self.sequence_lengths.max())

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
        self.cluster_names = cluster_names or [f"Cluster {i + 1}" for i in range(n_clusters)]
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
           using responsibilities as weights (weighted Baum-Welch)

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

        # Pre-compute cumulative lengths (avoid repeated .sum() in loop)
        cum_lengths = np.zeros(n_sequences + 1, dtype=np.int32)
        cum_lengths[1:] = np.cumsum(lengths)

        # Initialize: fit each cluster once to get starting parameters
        import warnings
        for k in range(self.n_clusters):
            if self.clusters[k].log_likelihood is None:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    self.clusters[k].fit(n_iter=10, tol=tol, verbose=False)

        # Initialize log-likelihood
        prev_log_likelihood = -np.inf

        # EM algorithm
        for iteration in range(n_iter):
            # E-step: Compute responsibilities
            # Responsibility = P(cluster | sequence) = P(sequence | cluster) * P(cluster) / P(sequence)

            # Compute log-likelihood for each sequence under each cluster
            log_likelihoods = np.zeros((n_sequences, self.n_clusters))

            for k in range(self.n_clusters):
                # Compute log-likelihood for each sequence
                for seq_idx in range(n_sequences):
                    start = cum_lengths[seq_idx]
                    end = cum_lengths[seq_idx + 1]
                    seq_X = X[start:end]
                    seq_len = np.array([lengths[seq_idx]])
                    log_likelihoods[seq_idx, k] = \
                        self.clusters[k]._hmm_model.score(seq_X, seq_len)

            # Add log of cluster probabilities
            log_probs = np.log(self.cluster_probs + 1e-300)
            log_joint = log_likelihoods + log_probs[np.newaxis, :]

            # Compute responsibilities using log-sum-exp trick for numerical stability
            log_norm = logsumexp(log_joint, axis=1, keepdims=True)
            responsibilities = np.exp(log_joint - log_norm)
            self.responsibilities = responsibilities

            # M-step: Update cluster probabilities
            self.cluster_probs = np.mean(responsibilities, axis=0)

            # M-step: Update each cluster's HMM parameters
            # Bug 1 fix: use weighted Baum-Welch instead of unweighted refit.
            # Each sequence's sufficient statistics are weighted by r(n,k).
            for k in range(self.n_clusters):
                self._weighted_mstep_for_cluster(
                    X, lengths, cum_lengths,
                    responsibilities[:, k],
                    self.clusters[k]
                )

            # Compute overall log-likelihood
            # log P(data) = sum_n log( sum_k P(k) * P(seq_n | theta_k) )
            log_likelihood = np.sum(logsumexp(log_joint, axis=1))

            if verbose:
                print(f"Iteration {iteration + 1}: log-likelihood = {log_likelihood:.4f}")

            # Check convergence
            if iteration > 0:
                if abs(log_likelihood - prev_log_likelihood) < tol:
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

    def _weighted_mstep_for_cluster(
            self,
            X: np.ndarray,
            lengths: np.ndarray,
            cum_lengths: np.ndarray,
            resp_k: np.ndarray,
            cluster_k: HMM,
    ) -> None:
        """
        Weighted Baum-Welch M-step for a single cluster.

        Runs forward-backward on each sequence using the current cluster
        parameters, then accumulates sufficient statistics weighted by
        the cluster responsibilities.

        The correct MHMM M-step formulas are:
            pi_k(i)    = sum_n r(n,k) * gamma_k(i, t=0 | seq_n)  /  sum_n r(n,k)
            A_k(i,j)   = sum_{n,t} r(n,k) * xi_k(i,j,t | seq_n)  /  sum_{n,t} r(n,k) * gamma_k(i,t)
            B_k(i,v)   = sum_{n,t: o_t=v} r(n,k) * gamma_k(i,t)  /  sum_{n,t} r(n,k) * gamma_k(i,t)

        Args:
            X:            Observations (n_total, 1), 0-indexed integers
            lengths:      Sequence lengths (n_sequences,)
            cum_lengths:  Cumulative lengths (n_sequences + 1,)
            resp_k:       Responsibilities for this cluster (n_sequences,)
            cluster_k:    HMM object for this cluster
        """
        hmm = cluster_k._hmm_model
        S = cluster_k.n_states  # number of hidden states
        V = cluster_k.n_symbols  # vocabulary size
        N = len(lengths)

        # Current parameters in log space
        log_pi = np.log(hmm.startprob_ + 1e-300)
        log_A = np.log(hmm.transmat_ + 1e-300)
        log_B = np.log(hmm.emissionprob_ + 1e-300)

        # Accumulators for weighted sufficient statistics
        acc_pi = np.zeros(S)
        acc_A = np.zeros((S, S))
        acc_A_den = np.zeros(S)
        acc_B = np.zeros((S, V))
        acc_B_den = np.zeros(S)
        total_weight = 0.0

        for n in range(N):
            w = resp_k[n]
            if w < 1e-15:
                continue

            start = cum_lengths[n]
            end = cum_lengths[n + 1]
            obs = X[start:end, 0]  # integer observations for this sequence
            T = len(obs)
            if T == 0:
                continue

            # ── Forward pass ──
            log_alpha = np.full((T, S), -np.inf)
            log_alpha[0] = log_pi + log_B[:, obs[0]]
            for t in range(1, T):
                log_alpha[t] = (
                        logsumexp(log_alpha[t - 1, :, np.newaxis] + log_A, axis=0)
                        + log_B[:, obs[t]]
                )

            # ── Backward pass ──
            log_beta = np.zeros((T, S))
            for t in range(T - 2, -1, -1):
                log_beta[t] = logsumexp(
                    log_A + log_B[:, obs[t + 1]] + log_beta[t + 1],
                    axis=1
                )

            # ── Gamma: P(q_t = i | O, lambda) ──
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)  # (T, S)

            # Accumulate weighted initial probs
            acc_pi += w * gamma[0]
            total_weight += w

            # ── Xi: P(q_t=i, q_{t+1}=j | O, lambda) ──
            for t in range(T - 1):
                log_xi = (
                        log_alpha[t, :, np.newaxis]
                        + log_A
                        + log_B[:, obs[t + 1]][np.newaxis, :]
                        + log_beta[t + 1][np.newaxis, :]
                )
                log_xi -= logsumexp(log_xi.ravel())
                xi = np.exp(log_xi)  # (S, S)

                acc_A += w * xi
                acc_A_den += w * gamma[t]

            # Accumulate weighted emission counts
            for t in range(T):
                acc_B[:, obs[t]] += w * gamma[t]
            acc_B_den += w * gamma.sum(axis=0)

        # ── Normalize accumulators → new parameters ──
        EPS = 1e-10

        if total_weight < EPS:
            return  # cluster has near-zero responsibility, skip update

        # Initial probs
        new_pi = acc_pi / total_weight
        new_pi = np.maximum(new_pi, EPS)
        new_pi /= new_pi.sum()

        # Transition probs
        new_A = np.zeros((S, S))
        for i in range(S):
            if acc_A_den[i] > EPS:
                new_A[i] = acc_A[i] / acc_A_den[i]
            else:
                new_A[i] = 1.0 / S
        new_A = np.maximum(new_A, EPS)
        new_A /= new_A.sum(axis=1, keepdims=True)

        # Emission probs
        new_B = np.zeros((S, V))
        for i in range(S):
            if acc_B_den[i] > EPS:
                new_B[i] = acc_B[i] / acc_B_den[i]
            else:
                new_B[i] = 1.0 / V
        new_B = np.maximum(new_B, EPS)
        new_B /= new_B.sum(axis=1, keepdims=True)

        # Write back to both hmmlearn model and cluster wrapper
        hmm.startprob_ = new_pi
        hmm.transmat_ = new_A
        hmm.emissionprob_ = new_B

        cluster_k.initial_probs = new_pi
        cluster_k.transition_probs = new_A
        cluster_k.emission_probs = new_B

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

            log_probs = np.log(self.cluster_probs + 1e-300)
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