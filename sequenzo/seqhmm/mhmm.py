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
from typing import Optional, List, Dict, Tuple, Union
from scipy.special import logsumexp
from sequenzo.define_sequence_data import SequenceData
from .hmm import HMM
from .multichannel_utils import prepare_multichannel_data
from .utils import sequence_data_to_hmmlearn_format


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
            observations: Union[SequenceData, List[SequenceData]],
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
            observations: SequenceData object, or a list of SequenceData objects
                for multichannel fixed-parameter inference
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
        channels, default_channel_names, alphabets = prepare_multichannel_data(observations)
        self.channels = channels
        self.n_channels = len(channels)
        self.observations = channels[0] if self.n_channels == 1 else channels
        self.n_clusters = n_clusters
        self.alphabet = alphabets[0]
        self.alphabets = alphabets
        self.n_symbols = [len(alphabet) for alphabet in alphabets]
        if self.n_channels == 1:
            self.n_symbols = self.n_symbols[0]
        self.n_sequences = len(channels[0].sequences)

        self.sequence_lengths = np.array(
            [len(seq) for seq in channels[0].sequences])
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
        if cluster_names is None:
            self.cluster_names = [f"Cluster {i + 1}" for i in range(n_clusters)]
        else:
            if len(cluster_names) != n_clusters:
                raise ValueError(
                    f"cluster_names length ({len(cluster_names)}) must equal n_clusters ({n_clusters})"
                )
            self.cluster_names = list(cluster_names)
        self.channel_names = channel_names or default_channel_names
        self.random_state = random_state

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
            cluster_probs = np.asarray(cluster_probs, dtype=float)
            if cluster_probs.shape != (n_clusters,):
                raise ValueError(
                    f"cluster_probs shape {cluster_probs.shape} must be ({n_clusters},)"
                )
            if np.any(cluster_probs < 0.0) or not np.isfinite(cluster_probs).all():
                raise ValueError("cluster_probs must contain finite non-negative probabilities")
            total = float(cluster_probs.sum())
            if total <= 0.0:
                raise ValueError("cluster_probs must have positive total probability")
            if not np.isclose(total, 1.0):
                raise ValueError("cluster_probs must sum to one")
            self.cluster_probs = cluster_probs.copy()

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

    @property
    def has_complete_parameters(self) -> bool:
        """Return True when all mixture components can score sequences."""
        return all(cluster.has_complete_parameters for cluster in self.clusters)

    def _default_sequences(self) -> Union[SequenceData, List[SequenceData]]:
        return self.channels if self.n_channels > 1 else self.observations

    def _check_single_channel_alphabet(self, sequences: SequenceData) -> None:
        if list(sequences.alphabet) != list(self.alphabet):
            raise ValueError("newdata alphabet does not match the fitted model alphabet")

    def _check_fixed_inference_ready(self) -> None:
        if not self.has_complete_parameters:
            raise ValueError(
                "Model must be fitted before prediction, or all cluster HMMs "
                "must have initial_probs, transition_probs, and emission_probs."
            )

    def _cluster_log_priors(self) -> np.ndarray:
        priors = np.asarray(self.cluster_probs, dtype=float)
        out = np.full(priors.shape, -np.inf, dtype=float)
        positive = priors > 0.0
        out[positive] = np.log(priors[positive])
        return out

    def _responsibilities_from_log_likelihoods(
            self,
            log_likelihoods: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        log_joint = log_likelihoods + self._cluster_log_priors()[np.newaxis, :]
        log_norm = logsumexp(log_joint, axis=1, keepdims=True)
        invalid = ~np.isfinite(log_norm[:, 0])
        if np.any(invalid):
            indices = np.where(invalid)[0].tolist()
            raise ValueError(
                "At least one sequence has zero likelihood under all clusters: "
                f"{indices}"
            )
        return np.exp(log_joint - log_norm), log_norm

    @staticmethod
    def _multichannel_sequence_log_likelihood(
            cluster: HMM,
            obs_list: List[np.ndarray],
    ) -> float:
        initial = np.asarray(cluster.initial_probs, dtype=float)
        transition = np.asarray(cluster.transition_probs, dtype=float)
        emission = cluster._multichannel_emission_matrix(obs_list)

        alpha = initial * emission[0]
        scale = float(np.sum(alpha))
        if not np.isfinite(scale) or scale <= 0.0:
            return -np.inf
        alpha = alpha / scale

        log_likelihood = np.log(scale)
        for t in range(1, emission.shape[0]):
            alpha = (alpha @ transition) * emission[t]
            scale = float(np.sum(alpha))
            if not np.isfinite(scale) or scale <= 0.0:
                return -np.inf
            alpha = alpha / scale
            log_likelihood += np.log(scale)

        return float(log_likelihood)

    @staticmethod
    def _sequence_groups_from_lengths(
            observations: List[np.ndarray],
            lengths: np.ndarray,
    ) -> List[Tuple[List[np.ndarray], np.ndarray]]:
        groups: Dict[Tuple[Tuple[int, ...], ...], List[int]] = {}
        start = 0
        for seq_idx, seq_length in enumerate(lengths):
            end = start + int(seq_length)
            key = tuple(tuple(obs[start:end].tolist()) for obs in observations)
            groups.setdefault(key, []).append(seq_idx)
            start = end

        return [
            (
                [np.asarray(channel, dtype=int) for channel in key],
                np.asarray(indices, dtype=int),
            )
            for key, indices in groups.items()
        ]

    @staticmethod
    def _should_compress_sequences(
            observations: List[np.ndarray],
            lengths: np.ndarray,
            n_iter: int = 1,
    ) -> bool:
        n_sequences = len(lengths)
        if n_sequences == 0:
            return False
        if n_iter > 1 or n_sequences >= 1000:
            return True

        n_unique = len(MHMM._sequence_groups_from_lengths(observations, lengths))
        duplicate_fraction = 1.0 - (n_unique / n_sequences)
        return duplicate_fraction >= 0.5

    def _sequence_log_likelihoods_compressed(
            self,
            sequences: Optional[Union[SequenceData, List[SequenceData]]] = None,
            groups: Optional[List[Tuple[List[np.ndarray], np.ndarray]]] = None,
    ) -> np.ndarray:
        self._check_fixed_inference_ready()
        sequences = sequences if sequences is not None else self._default_sequences()

        if self.n_channels > 1:
            observations, lengths = self.clusters[0]._multichannel_arrays(sequences)
            if groups is None:
                groups = self._sequence_groups_from_lengths(observations, lengths)
            log_likelihoods = np.zeros((len(lengths), self.n_clusters))
            for k, cluster in enumerate(self.clusters):
                for obs_list, indices in groups:
                    value = self._multichannel_sequence_log_likelihood(cluster, obs_list)
                    log_likelihoods[indices, k] = value
            return log_likelihoods

        self._check_single_channel_alphabet(sequences)
        X, lengths = sequence_data_to_hmmlearn_format(sequences)
        observations = [X[:, 0].astype(int, copy=False)]
        groups = self._sequence_groups_from_lengths(observations, lengths)
        log_likelihoods = np.zeros((len(lengths), self.n_clusters))
        for k, cluster in enumerate(self.clusters):
            for obs_list, indices in groups:
                sequence = obs_list[0].reshape(-1, 1)
                seq_len = np.array([sequence.shape[0]], dtype=np.int32)
                value = float(cluster._hmm_model.score(sequence, seq_len))
                log_likelihoods[indices, k] = value
        return log_likelihoods

    def _sequence_log_likelihoods(
            self,
            sequences: Optional[Union[SequenceData, List[SequenceData]]] = None,
            compress: bool = False,
            groups: Optional[List[Tuple[List[np.ndarray], np.ndarray]]] = None,
    ) -> np.ndarray:
        """Compute per-sequence, per-cluster log likelihoods."""
        self._check_fixed_inference_ready()
        if compress:
            return self._sequence_log_likelihoods_compressed(sequences, groups=groups)

        sequences = sequences if sequences is not None else self._default_sequences()

        if self.n_channels > 1:
            observations, lengths = self.clusters[0]._multichannel_arrays(sequences)
            log_likelihoods = np.zeros((len(lengths), self.n_clusters))
            for k, cluster in enumerate(self.clusters):
                start = 0
                for seq_idx, seq_length in enumerate(lengths):
                    end = start + int(seq_length)
                    obs_list = [obs[start:end] for obs in observations]
                    log_likelihoods[seq_idx, k] = self._multichannel_sequence_log_likelihood(
                        cluster, obs_list
                    )
                    start = end
            return log_likelihoods

        self._check_single_channel_alphabet(sequences)
        X, lengths = sequence_data_to_hmmlearn_format(sequences)
        log_likelihoods = np.zeros((len(lengths), self.n_clusters))
        cum_lengths = np.zeros(len(lengths) + 1, dtype=np.int32)
        cum_lengths[1:] = np.cumsum(lengths)

        for k, cluster in enumerate(self.clusters):
            for seq_idx in range(len(lengths)):
                start = cum_lengths[seq_idx]
                end = cum_lengths[seq_idx + 1]
                seq_X = X[start:end]
                seq_len = np.array([lengths[seq_idx]])
                log_likelihoods[seq_idx, k] = cluster._hmm_model.score(seq_X, seq_len)

        return log_likelihoods

    def compute_responsibilities(
            self,
            sequences: Optional[Union[SequenceData, List[SequenceData]]] = None,
            compress: bool = False,
    ) -> np.ndarray:
        """
        Compute posterior cluster probabilities from current component likelihoods.
        """
        log_likelihoods = self._sequence_log_likelihoods(sequences, compress=compress)
        responsibilities, _ = self._responsibilities_from_log_likelihoods(log_likelihoods)
        return responsibilities

    def score(
            self,
            sequences: Optional[Union[SequenceData, List[SequenceData]]] = None,
            compress: bool = False,
    ) -> float:
        """Compute the mixture log-likelihood under the current parameters."""
        log_likelihoods = self._sequence_log_likelihoods(sequences, compress=compress)
        log_joint = log_likelihoods + self._cluster_log_priors()[np.newaxis, :]
        return float(np.sum(logsumexp(log_joint, axis=1)))

    def _ensure_multichannel_cluster_parameters(self) -> None:
        assignments = self._initial_multichannel_cluster_assignments()
        rng = np.random.default_rng(self.random_state)

        for cluster_idx, cluster in enumerate(self.clusters):
            S = cluster.n_states
            if cluster.initial_probs is None:
                cluster.initial_probs = rng.dirichlet(np.ones(S))
            if cluster.transition_probs is None:
                sticky = np.full((S, S), 0.2 / max(S - 1, 1))
                np.fill_diagonal(sticky, 0.8)
                if S == 1:
                    sticky[:, :] = 1.0
                jitter = rng.dirichlet(np.ones(S), size=S)
                transition = 0.85 * sticky + 0.15 * jitter
                transition /= transition.sum(axis=1, keepdims=True)
                cluster.transition_probs = transition
            if cluster.emission_probs is None:
                cluster.emission_probs = self._initial_multichannel_emissions(
                    cluster,
                    assignments,
                    cluster_idx,
                    rng,
                )

    def _initial_multichannel_cluster_assignments(self) -> np.ndarray:
        """Create distinct deterministic starting groups from observation patterns."""
        features = self._multichannel_sequence_features()
        n_sequences = features.shape[0]
        if n_sequences == 0:
            return np.array([], dtype=int)

        if self.n_clusters == 1:
            return np.zeros(n_sequences, dtype=int)

        rng = np.random.default_rng(self.random_state)
        n_centers = min(self.n_clusters, n_sequences)
        first = int(rng.integers(n_sequences))
        centers = [features[first]]

        for _ in range(1, n_centers):
            distances = np.min(
                np.stack([np.sum((features - center) ** 2, axis=1) for center in centers]),
                axis=0,
            )
            next_idx = int(np.argmax(distances))
            centers.append(features[next_idx])

        while len(centers) < self.n_clusters:
            centers.append(features[len(centers) % n_sequences])

        centers = np.asarray(centers, dtype=float)
        assignments = np.arange(n_sequences, dtype=int) % self.n_clusters
        for _ in range(12):
            distances = np.sum(
                (features[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
                axis=2,
            )
            next_assignments = np.argmin(distances, axis=1)
            if np.array_equal(next_assignments, assignments):
                break
            assignments = next_assignments
            for k in range(self.n_clusters):
                mask = assignments == k
                if np.any(mask):
                    centers[k] = features[mask].mean(axis=0)

        empty = [k for k in range(self.n_clusters) if not np.any(assignments == k)]
        if empty:
            order = np.argsort(np.sum((features - features.mean(axis=0)) ** 2, axis=1))[::-1]
            for k, seq_idx in zip(empty, order):
                assignments[int(seq_idx)] = k

        return assignments

    def _multichannel_sequence_features(self) -> np.ndarray:
        observations, lengths = self.clusters[0]._multichannel_arrays(self._default_sequences())
        n_sequences = len(lengths)
        if n_sequences == 0:
            return np.empty((0, 0), dtype=float)

        if np.all(lengths == lengths[0]):
            seq_length = int(lengths[0])
            pieces = []
            for ch_idx, obs in enumerate(observations):
                denom = max(int(self.clusters[0].n_symbols[ch_idx]) - 1, 1)
                pieces.append(obs.reshape(n_sequences, seq_length) / denom)
            return np.hstack(pieces).astype(float, copy=False)

        features = np.zeros((n_sequences, int(np.sum(self.clusters[0].n_symbols))), dtype=float)
        offsets = np.cumsum([0] + list(self.clusters[0].n_symbols[:-1]))
        start = 0
        for seq_idx, seq_length in enumerate(lengths):
            end = start + int(seq_length)
            for ch_idx, obs in enumerate(observations):
                counts = np.bincount(
                    obs[start:end],
                    minlength=int(self.clusters[0].n_symbols[ch_idx]),
                ).astype(float)
                total = counts.sum()
                if total > 0.0:
                    counts /= total
                offset = offsets[ch_idx]
                features[seq_idx, offset:offset + len(counts)] = counts
            start = end
        return features

    def _initial_multichannel_emissions(
            self,
            cluster: HMM,
            assignments: np.ndarray,
            cluster_idx: int,
            rng: np.random.Generator,
    ) -> List[np.ndarray]:
        observations, lengths = cluster._multichannel_arrays(self._default_sequences())
        selected = np.where(assignments == cluster_idx)[0]
        if len(selected) == 0:
            selected = np.arange(len(lengths))

        starts = np.zeros(len(lengths) + 1, dtype=int)
        starts[1:] = np.cumsum(lengths)
        emissions = []

        for ch_idx, n_symbols in enumerate(cluster.n_symbols):
            counts = np.full(int(n_symbols), 0.5, dtype=float)
            obs = observations[ch_idx]
            for seq_idx in selected:
                start = starts[seq_idx]
                end = starts[seq_idx + 1]
                counts += np.bincount(obs[start:end], minlength=int(n_symbols))

            base = counts / counts.sum()
            rows = []
            for _ in range(cluster.n_states):
                rows.append(rng.dirichlet(25.0 * base + 0.25))
            emissions.append(np.vstack(rows))

        return emissions

    def _multichannel_forward_backward_for_cluster(
            self,
            cluster: HMM,
            obs_list: List[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        initial = np.asarray(cluster.initial_probs, dtype=float)
        transition = np.asarray(cluster.transition_probs, dtype=float)
        emission = cluster._multichannel_emission_matrix(obs_list)
        T = emission.shape[0]
        S = cluster.n_states

        alpha = np.empty((T, S), dtype=float)
        scales = np.empty(T, dtype=float)
        alpha[0], scales[0] = cluster._normalize_probabilities(initial * emission[0])
        if scales[0] <= np.finfo(float).tiny:
            gamma = np.ones((T, S), dtype=float) / S
            xi = np.zeros((max(T - 1, 0), S, S), dtype=float)
            return gamma, xi, -np.inf
        for t in range(1, T):
            alpha[t], scales[t] = cluster._normalize_probabilities(
                (alpha[t - 1] @ transition) * emission[t]
            )
            if scales[t] <= np.finfo(float).tiny:
                gamma = np.ones((T, S), dtype=float) / S
                xi = np.zeros((max(T - 1, 0), S, S), dtype=float)
                return gamma, xi, -np.inf

        beta = np.ones((T, S), dtype=float)
        for t in range(T - 2, -1, -1):
            beta[t] = transition @ (emission[t + 1] * beta[t + 1])
            beta[t] /= max(scales[t + 1], np.finfo(float).tiny)

        gamma = alpha * beta
        gamma_sums = gamma.sum(axis=1, keepdims=True)
        gamma_sums[gamma_sums <= 0.0] = 1.0
        gamma = gamma / gamma_sums

        xi = np.zeros((max(T - 1, 0), S, S), dtype=float)
        for t in range(T - 1):
            xi_t = (
                alpha[t, :, np.newaxis]
                * transition
                * emission[t + 1, np.newaxis, :]
                * beta[t + 1, np.newaxis, :]
            )
            denom = xi_t.sum()
            if denom > 0.0:
                xi[t] = xi_t / denom

        log_likelihood = float(np.sum(np.log(np.maximum(scales, np.finfo(float).tiny))))
        return gamma, xi, log_likelihood

    def _weighted_multichannel_mstep_for_cluster(
            self,
            cluster: HMM,
            observations: List[np.ndarray],
            lengths: np.ndarray,
            resp_k: np.ndarray,
            groups: Optional[List[Tuple[List[np.ndarray], np.ndarray]]] = None,
    ) -> None:
        S = cluster.n_states
        EPS = 1e-10

        acc_pi = np.zeros(S)
        acc_A = np.zeros((S, S))
        acc_A_den = np.zeros(S)
        acc_B = [np.zeros((S, n_symbols)) for n_symbols in cluster.n_symbols]
        acc_B_den = [np.zeros(S) for _ in range(cluster.n_channels)]
        total_weight = 0.0

        if groups is None:
            def iterable():
                start = 0
                for seq_idx, seq_length in enumerate(lengths):
                    end = start + int(seq_length)
                    yield (
                        [obs[start:end] for obs in observations],
                        np.array([seq_idx], dtype=int),
                    )
                    start = end
        else:
            iterable = lambda: iter(groups)

        for obs_list, indices in iterable():
            weight = float(np.sum(resp_k[indices]))
            if weight <= EPS:
                continue

            gamma, xi, _ = self._multichannel_forward_backward_for_cluster(cluster, obs_list)

            acc_pi += weight * gamma[0]
            total_weight += weight
            if xi.shape[0] > 0:
                acc_A += weight * xi.sum(axis=0)
                acc_A_den += weight * gamma[:-1].sum(axis=0)

            for ch_idx, obs_ch in enumerate(obs_list):
                for t, symbol in enumerate(obs_ch):
                    acc_B[ch_idx][:, symbol] += weight * gamma[t]
                acc_B_den[ch_idx] += weight * gamma.sum(axis=0)

        if total_weight <= EPS:
            return

        cluster.initial_probs = np.maximum(acc_pi / total_weight, EPS)
        cluster.initial_probs /= cluster.initial_probs.sum()

        new_A = np.zeros((S, S))
        for i in range(S):
            if acc_A_den[i] > EPS:
                new_A[i] = acc_A[i] / acc_A_den[i]
            else:
                new_A[i] = 1.0 / S
        new_A = np.maximum(new_A, EPS)
        new_A /= new_A.sum(axis=1, keepdims=True)
        cluster.transition_probs = new_A

        new_emissions = []
        for ch_idx, counts in enumerate(acc_B):
            n_symbols = counts.shape[1]
            emission = np.zeros_like(counts)
            for i in range(S):
                if acc_B_den[ch_idx][i] > EPS:
                    emission[i] = counts[i] / acc_B_den[ch_idx][i]
                else:
                    emission[i] = 1.0 / n_symbols
            emission = np.maximum(emission, EPS)
            emission /= emission.sum(axis=1, keepdims=True)
            new_emissions.append(emission)
        cluster.emission_probs = new_emissions

    def _fit_multichannel(
            self,
            n_iter: int = 100,
            tol: float = 1e-2,
            verbose: bool = False,
            compress: Optional[bool] = None,
    ) -> 'MHMM':
        self._ensure_multichannel_cluster_parameters()
        observations, lengths = self.clusters[0]._multichannel_arrays(self._default_sequences())
        if compress is None:
            compress = self._should_compress_sequences(observations, lengths, n_iter=n_iter)
        groups = self._sequence_groups_from_lengths(observations, lengths) if compress else None
        prev_log_likelihood = -np.inf

        for iteration in range(n_iter):
            log_likelihoods = self._sequence_log_likelihoods(
                compress=bool(compress),
                groups=groups,
            )
            responsibilities, _ = self._responsibilities_from_log_likelihoods(log_likelihoods)
            self.responsibilities = responsibilities
            self.cluster_probs = responsibilities.mean(axis=0)

            for k, cluster in enumerate(self.clusters):
                self._weighted_multichannel_mstep_for_cluster(
                    cluster,
                    observations,
                    lengths,
                    responsibilities[:, k],
                    groups=groups,
                )

            post_log_likelihoods = self._sequence_log_likelihoods(
                compress=bool(compress),
                groups=groups,
            )
            self.responsibilities, post_log_norm = self._responsibilities_from_log_likelihoods(
                post_log_likelihoods
            )
            log_likelihood = float(np.sum(post_log_norm))
            if verbose:
                print(f"Iteration {iteration + 1}: log-likelihood = {log_likelihood:.4f}")

            if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < tol:
                self.converged = True
                prev_log_likelihood = log_likelihood
                break

            prev_log_likelihood = log_likelihood

        final_log_likelihoods = self._sequence_log_likelihoods(
            compress=bool(compress),
            groups=groups,
        )
        self.responsibilities, final_log_norm = self._responsibilities_from_log_likelihoods(
            final_log_likelihoods
        )
        self.log_likelihood = float(np.sum(final_log_norm))
        self.n_iter = iteration + 1
        if self.converged is None:
            self.converged = False

        return self

    def fit(
            self,
            n_iter: int = 100,
            tol: float = 1e-2,
            verbose: bool = False,
            compress: Optional[bool] = None,
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
        if self.n_channels > 1:
            return self._fit_multichannel(
                n_iter=n_iter,
                tol=tol,
                verbose=verbose,
                compress=compress,
            )

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

            responsibilities, _ = self._responsibilities_from_log_likelihoods(log_likelihoods)
            self.responsibilities = responsibilities

            # M-step: Update cluster probabilities
            self.cluster_probs = np.mean(responsibilities, axis=0)

            # M-step: update each cluster with responsibility-weighted statistics.
            for k in range(self.n_clusters):
                self._weighted_mstep_for_cluster(
                    X, lengths, cum_lengths,
                    responsibilities[:, k],
                    self.clusters[k]
                )

            # Compute overall log-likelihood
            # log P(data) = sum_n log( sum_k P(k) * P(seq_n | theta_k) )
            post_log_likelihoods = self._sequence_log_likelihoods()
            self.responsibilities, post_log_norm = self._responsibilities_from_log_likelihoods(
                post_log_likelihoods
            )
            log_likelihood = float(np.sum(post_log_norm))

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

        self.log_likelihood = self.score()
        self.responsibilities = self.compute_responsibilities()
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

    def predict_cluster(
            self,
            sequences: Optional[Union[SequenceData, List[SequenceData]]] = None,
            compress: bool = False,
    ) -> np.ndarray:
        """
        Predict the most likely cluster for each sequence.

        Args:
            sequences: Optional SequenceData (uses self.observations if None)
            compress: Whether to reuse likelihoods for repeated sequences

        Returns:
            numpy array: Predicted cluster index for each sequence
        """
        responsibilities = self.compute_responsibilities(sequences, compress=compress)
        return np.argmax(responsibilities, axis=1)

    def __repr__(self) -> str:
        """String representation of the MHMM."""
        status = "fitted" if self.log_likelihood is not None else "unfitted"
        return (f"MHMM(n_clusters={self.n_clusters}, n_states={self.n_states}, "
                f"n_sequences={self.n_sequences}, status='{status}')")
