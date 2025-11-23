"""
@Author  : Yuqi Liang 梁彧祺
@File    : model_comparison.py
@Time    : 2025-10-08 14:32
@Desc    : Model comparison functions (AIC, BIC) for HMM models

This module provides functions for computing AIC and BIC to compare different
HMM models, similar to seqHMM's logLik() and summary() functions in R.
"""

import numpy as np
from typing import Optional
from .hmm import HMM
from .mhmm import MHMM
from .nhmm import NHMM


def compute_n_parameters(model) -> int:
    """
    Compute the number of free parameters in a model.
    
    This is used for computing AIC and BIC. The number of parameters
    (degrees of freedom) is the number of estimable parameters in the model.
    
    Args:
        model: HMM, MHMM, or NHMM model object
        
    Returns:
        int: Number of free parameters
    """
    if isinstance(model, HMM):
        # For basic HMM:
        # - Initial probabilities: n_states - 1 (sum to 1)
        # - Transition probabilities: n_states * (n_states - 1) (each row sums to 1)
        # - Emission probabilities: n_states * (n_symbols - 1) (each row sums to 1)
        n_init = model.n_states - 1
        n_trans = model.n_states * (model.n_states - 1)
        n_emiss = model.n_states * (model.n_symbols - 1)
        return n_init + n_trans + n_emiss
    
    elif isinstance(model, MHMM):
        # For Mixture HMM:
        # - Cluster probabilities: n_clusters - 1 (sum to 1)
        # - For each cluster: same as basic HMM
        # - Covariate coefficients (if any): n_covariates * (n_clusters - 1)
        n_cluster = model.n_clusters - 1
        
        # Parameters for each cluster
        n_per_cluster = 0
        for k in range(model.n_clusters):
            cluster = model.clusters[k]
            n_init = cluster.n_states - 1
            n_trans = cluster.n_states * (cluster.n_states - 1)
            n_emiss = cluster.n_states * (cluster.n_symbols - 1)
            n_per_cluster += n_init + n_trans + n_emiss
        
        # Covariate coefficients (if any)
        n_coefs = 0
        if model.coefficients is not None:
            n_coefs = model.coefficients.size - model.n_clusters  # First column is zero
        
        return n_cluster + n_per_cluster + n_coefs
    
    elif isinstance(model, NHMM):
        # For Non-homogeneous HMM:
        # - eta_pi: n_covariates * n_states
        # - eta_A: n_covariates * n_states * n_states
        # - eta_B: n_covariates * n_states * n_symbols
        # Note: We don't subtract constraints here because Softmax handles them
        n_pi = model.n_covariates * model.n_states
        n_A = model.n_covariates * model.n_states * model.n_states
        n_B = model.n_covariates * model.n_states * model.n_symbols
        return n_pi + n_A + n_B
    
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def compute_n_observations(model) -> int:
    """
    Compute the number of observations in a model.
    
    For multichannel models, each observed value in a single channel
    amounts to 1/n_channels observation, i.e., a fully observed time point
    for a single sequence amounts to one observation.
    
    Args:
        model: HMM, MHMM, or NHMM model object
        
    Returns:
        int: Number of observations
    """
    if isinstance(model, (HMM, MHMM, NHMM)):
        # For single-channel models, each time point is one observation
        # For multichannel models, we divide by number of channels
        n_channels = getattr(model, 'n_channels', 1)
        total_timepoints = sum(model.sequence_lengths)
        return int(total_timepoints / n_channels)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def aic(model, log_likelihood: Optional[float] = None) -> float:
    """
    Compute Akaike Information Criterion (AIC) for a model.
    
    AIC = -2 * log-likelihood + 2 * n_parameters
    
    Lower AIC values indicate better models (better fit with fewer parameters).
    
    This is similar to seqHMM's AIC computation via stats::AIC(logLik(model)).
    
    Args:
        model: Fitted HMM, MHMM, or NHMM model object
        log_likelihood: Optional log-likelihood value. If None, uses model.log_likelihood
        
    Returns:
        float: AIC value
        
    Examples:
        >>> from sequenzo.seqhmm import build_hmm, fit_model, aic
        >>> 
        >>> hmm = build_hmm(seq, n_states=4, random_state=42)
        >>> hmm = fit_model(hmm)
        >>> aic_value = aic(hmm)
        >>> print(f"AIC: {aic_value:.2f}")
    """
    if log_likelihood is None:
        if model.log_likelihood is None:
            raise ValueError("Model must be fitted before computing AIC. Use fit_model() first.")
        log_likelihood = model.log_likelihood
    
    n_params = compute_n_parameters(model)
    aic_value = -2 * log_likelihood + 2 * n_params
    
    return aic_value


def bic(model, log_likelihood: Optional[float] = None) -> float:
    """
    Compute Bayesian Information Criterion (BIC) for a model.
    
    BIC = -2 * log-likelihood + log(n_observations) * n_parameters
    
    Lower BIC values indicate better models. BIC penalizes complexity more
    than AIC, especially for large datasets.
    
    This is similar to seqHMM's BIC computation via stats::BIC(logLik(model)).
    
    Args:
        model: Fitted HMM, MHMM, or NHMM model object
        log_likelihood: Optional log-likelihood value. If None, uses model.log_likelihood
        
    Returns:
        float: BIC value
        
    Examples:
        >>> from sequenzo.seqhmm import build_hmm, fit_model, bic
        >>> 
        >>> hmm = build_hmm(seq, n_states=4, random_state=42)
        >>> hmm = fit_model(hmm)
        >>> bic_value = bic(hmm)
        >>> print(f"BIC: {bic_value:.2f}")
    """
    if log_likelihood is None:
        if model.log_likelihood is None:
            raise ValueError("Model must be fitted before computing BIC. Use fit_model() first.")
        log_likelihood = model.log_likelihood
    
    n_params = compute_n_parameters(model)
    n_obs = compute_n_observations(model)
    bic_value = -2 * log_likelihood + np.log(n_obs) * n_params
    
    return bic_value


def compare_models(models: list, criterion: str = 'BIC') -> dict:
    """
    Compare multiple models using AIC or BIC.
    
    This function computes AIC or BIC for multiple models and returns
    a comparison table, similar to comparing models in seqHMM.
    
    Args:
        models: List of fitted model objects (HMM, MHMM, or NHMM)
        criterion: Criterion to use ('AIC' or 'BIC'). Default is 'BIC'.
        
    Returns:
        dict: Dictionary with model names, log-likelihood, n_parameters, and criterion values
        
    Examples:
        >>> from sequenzo.seqhmm import build_hmm, fit_model, compare_models
        >>> 
        >>> # Fit models with different numbers of states
        >>> hmm3 = build_hmm(seq, n_states=3, random_state=42)
        >>> hmm4 = build_hmm(seq, n_states=4, random_state=42)
        >>> hmm5 = build_hmm(seq, n_states=5, random_state=42)
        >>> 
        >>> hmm3 = fit_model(hmm3)
        >>> hmm4 = fit_model(hmm4)
        >>> hmm5 = fit_model(hmm5)
        >>> 
        >>> # Compare models
        >>> comparison = compare_models([hmm3, hmm4, hmm5], criterion='BIC')
        >>> print(comparison)
    """
    if criterion not in ['AIC', 'BIC']:
        raise ValueError("criterion must be 'AIC' or 'BIC'")
    
    results = []
    for i, model in enumerate(models):
        if model.log_likelihood is None:
            raise ValueError(f"Model {i} must be fitted before comparison.")
        
        n_params = compute_n_parameters(model)
        n_obs = compute_n_observations(model)
        
        if criterion == 'AIC':
            criterion_value = aic(model)
        else:
            criterion_value = bic(model)
        
        results.append({
            'model': f"Model {i+1}",
            'log_likelihood': model.log_likelihood,
            'n_parameters': n_params,
            'n_observations': n_obs,
            criterion: criterion_value
        })
    
    # Sort by criterion value (lower is better)
    results.sort(key=lambda x: x[criterion])
    
    return {
        'criterion': criterion,
        'models': results,
        'best_model': results[0]['model']
    }
