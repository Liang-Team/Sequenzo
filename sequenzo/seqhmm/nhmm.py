"""
@Author  : Yuqi Liang 梁彧祺
@File    : nhmm.py
@Time    : 2025-11-23 13:39
@Desc    : Non-homogeneous Hidden Markov Model (NHMM) for Sequenzo

A Non-homogeneous HMM allows transition and emission probabilities to vary
over time or with covariates. This is useful when the underlying process
changes over time or depends on external factors.

This is similar to seqHMM's nhmm class in R.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Tuple
from scipy.optimize import minimize
from sequenzo.define_sequence_data import SequenceData
from .utils import sequence_data_to_hmmlearn_format
from .nhmm_utils import (
    compute_transition_probs_with_covariates,
    compute_emission_probs_with_covariates,
    compute_initial_probs_with_covariates
)


class NHMM:
    """
    Non-homogeneous Hidden Markov Model for sequence analysis.
    
    In a Non-homogeneous HMM, transition and emission probabilities can vary
    over time or with covariates. This allows the model to capture time-varying
    or covariate-dependent patterns in the data.
    
    Attributes:
        observations: SequenceData object containing the observed sequences
        n_states: Number of hidden states
        n_symbols: Number of observed symbols
        alphabet: List of observed state symbols
        state_names: Optional names for hidden states
        X: Covariate matrix (n_sequences x n_timepoints x n_covariates)
        n_covariates: Number of covariates
        
        # Model parameters (coefficients)
        eta_pi: Coefficients for initial probabilities (n_covariates x n_states)
        eta_A: Coefficients for transition probabilities (n_covariates x n_states x n_states)
        eta_B: Coefficients for emission probabilities (n_covariates x n_states x n_symbols)
        
        # Fitting results
        log_likelihood: Log-likelihood of the fitted model
        n_iter: Number of optimization iterations
        converged: Whether optimization converged
    """
    
    def __init__(
        self,
        observations: SequenceData,
        n_states: int,
        X: np.ndarray,
        eta_pi: Optional[np.ndarray] = None,
        eta_A: Optional[np.ndarray] = None,
        eta_B: Optional[np.ndarray] = None,
        state_names: Optional[List[str]] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize a Non-homogeneous HMM model.
        
        Args:
            observations: SequenceData object containing the sequences
            n_states: Number of hidden states
            X: Covariate matrix of shape (n_sequences, n_timepoints, n_covariates)
               where X[i, t, c] is the value of covariate c at time t for sequence i
            eta_pi: Optional coefficients for initial probabilities (n_covariates x n_states)
            eta_A: Optional coefficients for transition probabilities (n_covariates x n_states x n_states)
            eta_B: Optional coefficients for emission probabilities (n_covariates x n_states x n_symbols)
            state_names: Optional names for hidden states
            random_state: Random seed for initialization
        """
        self.observations = observations
        self.alphabet = observations.alphabet
        self.n_symbols = len(self.alphabet)
        self.n_states = n_states
        self.n_sequences = len(observations.sequences)
        
        # Validate and store covariates
        if X.ndim != 3:
            raise ValueError("X must be 3-dimensional: (n_sequences, n_timepoints, n_covariates)")
        self.X = X
        self.n_covariates = X.shape[2]
        
        # Get sequence lengths
        self.sequence_lengths = np.array([len(seq) for seq in observations.sequences])
        self.length_of_sequences = int(self.sequence_lengths.max())
        
        # Validate X dimensions match sequences
        if X.shape[0] != self.n_sequences:
            raise ValueError(
                f"X first dimension ({X.shape[0]}) must equal n_sequences ({self.n_sequences})"
            )
        
        # Set names
        self.state_names = state_names or [f"State {i+1}" for i in range(n_states)]
        
        # Initialize coefficients if not provided
        rng = np.random.RandomState(random_state)
        
        if eta_pi is None:
            # Initialize with small random values
            self.eta_pi = rng.randn(self.n_covariates, n_states) * 0.1
        else:
            if eta_pi.shape != (self.n_covariates, n_states):
                raise ValueError(
                    f"eta_pi shape ({eta_pi.shape}) must be ({self.n_covariates}, {n_states})"
                )
            self.eta_pi = eta_pi
        
        if eta_A is None:
            # Initialize with small random values
            self.eta_A = rng.randn(self.n_covariates, n_states, n_states) * 0.1
        else:
            if eta_A.shape != (self.n_covariates, n_states, n_states):
                raise ValueError(
                    f"eta_A shape ({eta_A.shape}) must be ({self.n_covariates}, {n_states}, {n_states})"
                )
            self.eta_A = eta_A
        
        if eta_B is None:
            # Initialize with small random values
            self.eta_B = rng.randn(self.n_covariates, n_states, self.n_symbols) * 0.1
        else:
            if eta_B.shape != (self.n_covariates, n_states, self.n_symbols):
                raise ValueError(
                    f"eta_B shape ({eta_B.shape}) must be ({self.n_covariates}, {n_states}, {self.n_symbols})"
                )
            self.eta_B = eta_B
        
        # Fitting results
        self.log_likelihood = None
        self.n_iter = None
        self.converged = None
    
    def _compute_probs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute probabilities from coefficients and covariates.
        
        Returns:
            tuple: (initial_probs, transition_probs, emission_probs)
        """
        # Compute initial probabilities
        initial_probs = compute_initial_probs_with_covariates(
            self.eta_pi, self.X, self.n_states
        )
        
        # Compute transition probabilities
        transition_probs = compute_transition_probs_with_covariates(
            self.eta_A, self.X, self.n_states
        )
        
        # Compute emission probabilities
        emission_probs = compute_emission_probs_with_covariates(
            self.eta_B, self.X, self.n_states, self.n_symbols
        )
        
        return initial_probs, transition_probs, emission_probs
    
    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute negative log-likelihood (for minimization).
        
        Uses the forward-backward algorithm to compute the exact likelihood
        for time-varying probabilities.
        
        Args:
            params: Flattened parameter vector
            
        Returns:
            float: Negative log-likelihood
        """
        # Reshape parameters
        n_pi = self.n_covariates * self.n_states
        n_A = self.n_covariates * self.n_states * self.n_states
        n_B = self.n_covariates * self.n_states * self.n_symbols
        
        self.eta_pi = params[:n_pi].reshape(self.n_covariates, self.n_states)
        self.eta_A = params[n_pi:n_pi+n_A].reshape(self.n_covariates, self.n_states, self.n_states)
        self.eta_B = params[n_pi+n_A:].reshape(self.n_covariates, self.n_states, self.n_symbols)
        
        # Compute log-likelihood using forward-backward algorithm
        from .forward_backward_nhmm import log_likelihood_nhmm
        log_lik = log_likelihood_nhmm(self)
        
        return -log_lik  # Return negative for minimization
    
    def fit(
        self,
        n_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False
    ) -> 'NHMM':
        """
        Fit the NHMM model using numerical optimization.
        
        Note: This is a simplified implementation. A full implementation would
        use the forward-backward algorithm and proper gradient computation.
        
        Args:
            n_iter: Maximum number of optimization iterations
            tol: Convergence tolerance
            verbose: Whether to print progress
            
        Returns:
            self: Returns self for method chaining
        """
        # Flatten parameters
        params = np.concatenate([
            self.eta_pi.flatten(),
            self.eta_A.flatten(),
            self.eta_B.flatten()
        ])
        
        # Optimize using scipy with analytical gradients if available
        try:
            from .gradients_nhmm import compute_gradient_nhmm
            
            def objective_with_grad(params):
                """Objective function with gradient."""
                neg_log_lik = self._log_likelihood(params)
                grad = -compute_gradient_nhmm(self)  # Negative because we minimize
                return neg_log_lik, grad
            
            # Use L-BFGS-B with analytical gradients
            result = minimize(
                objective_with_grad,
                params,
                method='L-BFGS-B',
                jac=True,  # Indicate that gradient is provided
                options={'maxiter': n_iter, 'ftol': tol, 'disp': verbose}
            )
        except ImportError:
            # Fall back to numerical gradients if analytical not available
            result = minimize(
                self._log_likelihood,
                params,
                method='L-BFGS-B',
                options={'maxiter': n_iter, 'ftol': tol, 'disp': verbose}
            )
        
        # Store results
        self.n_iter = result.nit
        self.converged = result.success
        self.log_likelihood = -result.fun
        
        # Recompute log-likelihood using forward-backward for accuracy
        from .forward_backward_nhmm import log_likelihood_nhmm
        self.log_likelihood = log_likelihood_nhmm(self)
        
        if verbose:
            print(f"Optimization {'converged' if result.success else 'did not converge'}")
            print(f"Log-likelihood: {self.log_likelihood:.4f}")
            print(f"Iterations: {self.n_iter}")
        
        return self
    
    def __repr__(self) -> str:
        """String representation of the NHMM."""
        status = "fitted" if self.log_likelihood is not None else "unfitted"
        return (f"NHMM(n_states={self.n_states}, n_symbols={self.n_symbols}, "
                f"n_covariates={self.n_covariates}, n_sequences={self.n_sequences}, "
                f"status='{status}')")
