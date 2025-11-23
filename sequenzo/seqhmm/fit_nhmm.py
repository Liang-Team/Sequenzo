"""
@Author  : Yuqi Liang 梁彧祺
@File    : fit_nhmm.py
@Time    : 2025-11-23 13:38
@Desc    : Fit Non-homogeneous HMM models

This module provides the fit_nhmm function, which estimates NHMM parameters
using numerical optimization, similar to seqHMM's fit_nhmm() function in R.

Note: This is a simplified implementation. A full implementation would use
the forward-backward algorithm and proper gradient computation.
"""

from typing import Optional
from .nhmm import NHMM


def fit_nhmm(
    model: NHMM,
    n_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False
) -> NHMM:
    """
    Fit a Non-homogeneous HMM model to the observations.
    
    This function estimates the coefficients (eta_pi, eta_A, eta_B) that
    determine how covariates influence the initial, transition, and emission
    probabilities.
    
    Note: This is a simplified implementation. A full implementation would:
    1. Use the forward-backward algorithm to compute exact log-likelihood
    2. Compute analytical gradients
    3. Use more sophisticated optimization methods
    
    It is similar to seqHMM's fit_nhmm() function in R.
    
    Args:
        model: NHMM model object created by build_nhmm()
        n_iter: Maximum number of optimization iterations. Default is 100.
        tol: Convergence tolerance. Default is 1e-4.
        verbose: Whether to print progress information. Default is False.
        
    Returns:
        NHMM: The fitted model (same object, modified in place)
        
    Examples:
        >>> from sequenzo import SequenceData, load_dataset
        >>> from sequenzo.seqhmm import build_nhmm, fit_nhmm
        >>> import numpy as np
        >>> 
        >>> # Load and prepare data
        >>> df = load_dataset('mvad')
        >>> seq = SequenceData(df, time=range(15, 86), states=['EM', 'FE', 'HE', 'JL', 'SC', 'TR'])
        >>> 
        >>> # Create covariate matrix
        >>> n_sequences = len(seq.sequences)
        >>> n_timepoints = max(len(s) for s in seq.sequences)
        >>> X = np.zeros((n_sequences, n_timepoints, 1))
        >>> for i in range(n_sequences):
        ...     for t in range(len(seq.sequences[i])):
        ...         X[i, t, 0] = t
        >>> 
        >>> # Build and fit model
        >>> nhmm = build_nhmm(seq, n_states=4, X=X, random_state=42)
        >>> nhmm = fit_nhmm(nhmm, n_iter=100, tol=1e-4, verbose=True)
        >>> 
        >>> # Check results
        >>> print(f"Log-likelihood: {nhmm.log_likelihood:.2f}")
        >>> print(f"Iterations: {nhmm.n_iter}")
        >>> print(f"Converged: {nhmm.converged}")
    """
    # Fit the model
    model.fit(n_iter=n_iter, tol=tol, verbose=verbose)
    
    return model
