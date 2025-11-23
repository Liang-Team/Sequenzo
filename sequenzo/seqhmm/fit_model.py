"""
@Author  : Yuqi Liang 梁彧祺
@File    : fit_model.py
@Time    : 2025-11-22 22:57
@Desc    : Fit HMM models using EM algorithm

This module provides the fit_model function, which estimates HMM parameters
using the EM algorithm, similar to seqHMM's fit_model() function in R.
"""

from typing import Optional, Dict, Any
from .hmm import HMM


def fit_model(
    model: HMM,
    n_iter: int = 100,
    tol: float = 1e-2,
    verbose: bool = False
) -> HMM:
    """
    Fit an HMM model to the observations using EM algorithm.
    
    This function estimates the parameters (initial probabilities, transition
    probabilities, and emission probabilities) of an HMM model using the
    Expectation-Maximization (EM) algorithm.
    
    It is similar to seqHMM's fit_model() function in R, but currently only
    supports the EM algorithm step (not global or local optimization).
    
    Args:
        model: HMM model object created by build_hmm()
        n_iter: Maximum number of EM iterations. Default is 100.
        tol: Convergence tolerance. EM stops if the gain in log-likelihood
             is below this value. Default is 1e-2.
        verbose: Whether to print progress information. Default is False.
        
    Returns:
        HMM: The fitted model (same object, modified in place)
        
    Examples:
        >>> from sequenzo import SequenceData, load_dataset
        >>> from sequenzo.seqhmm import build_hmm, fit_model
        >>> 
        >>> # Load and prepare data
        >>> df = load_dataset('mvad')
        >>> seq = SequenceData(df, time=range(15, 86), states=['EM', 'FE', 'HE', 'JL', 'SC', 'TR'])
        >>> 
        >>> # Build and fit model
        >>> hmm = build_hmm(seq, n_states=4, random_state=42)
        >>> hmm = fit_model(hmm, n_iter=100, tol=1e-2, verbose=True)
        >>> 
        >>> # Check results
        >>> print(f"Log-likelihood: {hmm.log_likelihood:.2f}")
        >>> print(f"Iterations: {hmm.n_iter}")
        >>> print(f"Converged: {hmm.converged}")
    """
    # Fit the model
    model.fit(n_iter=n_iter, tol=tol, verbose=verbose)
    
    return model
