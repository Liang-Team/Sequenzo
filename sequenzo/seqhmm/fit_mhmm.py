"""
@Author  : Yuqi Liang 梁彧祺
@File    : fit_mhmm.py
@Time    : 2025-11-21 13:37
@Desc    : Fit Mixture HMM models using EM algorithm

This module provides the fit_mhmm function, which estimates Mixture HMM parameters
using the EM algorithm, similar to seqHMM's fit_model() function for mhmm objects.
"""

from typing import Optional
from .mhmm import MHMM


def fit_mhmm(
    model: MHMM,
    n_iter: int = 100,
    tol: float = 1e-2,
    verbose: bool = False
) -> MHMM:
    """
    Fit a Mixture HMM model to the observations using EM algorithm.
    
    This function estimates the parameters of a Mixture HMM model using the
    Expectation-Maximization (EM) algorithm. The EM algorithm alternates between:
    1. E-step: Compute responsibilities (posterior cluster probabilities)
    2. M-step: Update cluster probabilities and HMM parameters for each cluster
    
    It is similar to seqHMM's fit_model() function for mhmm objects in R.
    
    Args:
        model: MHMM model object created by build_mhmm()
        n_iter: Maximum number of EM iterations. Default is 100.
        tol: Convergence tolerance. EM stops if the gain in log-likelihood
             is below this value. Default is 1e-2.
        verbose: Whether to print progress information. Default is False.
        
    Returns:
        MHMM: The fitted model (same object, modified in place)
        
    Examples:
        >>> from sequenzo import SequenceData, load_dataset
        >>> from sequenzo.seqhmm import build_mhmm, fit_mhmm
        >>> 
        >>> # Load and prepare data
        >>> df = load_dataset('mvad')
        >>> seq = SequenceData(df, time=range(15, 86), states=['EM', 'FE', 'HE', 'JL', 'SC', 'TR'])
        >>> 
        >>> # Build and fit model
        >>> mhmm = build_mhmm(seq, n_clusters=3, n_states=4, random_state=42)
        >>> mhmm = fit_mhmm(mhmm, n_iter=100, tol=1e-2, verbose=True)
        >>> 
        >>> # Check results
        >>> print(f"Log-likelihood: {mhmm.log_likelihood:.2f}")
        >>> print(f"Iterations: {mhmm.n_iter}")
        >>> print(f"Converged: {mhmm.converged}")
        >>> print(f"Cluster probabilities: {mhmm.cluster_probs}")
    """
    # Fit the model
    model.fit(n_iter=n_iter, tol=tol, verbose=verbose)
    
    return model
