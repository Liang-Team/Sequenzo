"""
@Author  : Yuqi Liang 梁彧祺
@File    : bootstrap.py
@Time    : 2025-10-05 08:15
@Desc    : Bootstrap confidence intervals for HMM model coefficients

This module provides functions for computing bootstrap confidence intervals
for model parameters, similar to seqHMM's bootstrap_coefs() function in R.
"""

import numpy as np
from typing import Optional, List, Dict, Callable, Union
from .hmm import HMM
from .mhmm import MHMM
from .nhmm import NHMM
from sequenzo.define_sequence_data import SequenceData

# Try to import tqdm for progress bar, but make it optional
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def bootstrap_model(
    model: Union[HMM, MHMM, NHMM],
    n_sim: int = 100,
    method: str = 'nonparametric',
    random_state: Optional[int] = None,
    verbose: bool = True,
    n_jobs: int = 1
) -> dict:
    """
    Bootstrap sampling for HMM model coefficients.
    
    This function performs bootstrap resampling to estimate confidence intervals
    for model parameters. For each bootstrap sample, the model is refitted and
    parameters are stored. This is similar to seqHMM's bootstrap_coefs() function.
    
    Args:
        model: Fitted HMM, MHMM, or NHMM model object
        n_sim: Number of bootstrap samples. Default is 100.
        method: Bootstrap method. Options:
            - 'nonparametric': Resample sequences with replacement (default)
            - 'parametric': Not yet implemented
        random_state: Random seed for reproducibility
        verbose: Whether to show progress bar
        n_jobs: Number of parallel jobs (not yet implemented, always uses 1)
        
    Returns:
        dict: Dictionary containing:
            - 'bootstrap_samples': List of bootstrap parameter estimates
            - 'original_model': Original model object
            - 'n_sim': Number of bootstrap samples
            - 'method': Bootstrap method used
            - 'summary': Summary statistics (mean, std, percentiles)
            
    Examples:
        >>> from sequenzo.seqhmm import build_hmm, fit_model, bootstrap_model
        >>> 
        >>> # Fit model
        >>> hmm = build_hmm(seq, n_states=4, random_state=42)
        >>> hmm = fit_model(hmm)
        >>> 
        >>> # Bootstrap
        >>> boot_results = bootstrap_model(hmm, n_sim=100, verbose=True)
        >>> 
        >>> # Get confidence intervals
        >>> ci = boot_results['summary']['ci_95']
        >>> print(f"95% CI for initial_probs: {ci['initial_probs']}")
    """
    if model.log_likelihood is None:
        raise ValueError("Model must be fitted before bootstrapping. Use fit_model() first.")
    
    rng = np.random.RandomState(random_state)
    n_sequences = model.n_sequences
    
    # Store bootstrap samples
    bootstrap_samples = []
    
    # Progress bar
    if verbose and HAS_TQDM:
        iterator = tqdm(range(n_sim), desc="Bootstrap sampling")
    else:
        iterator = range(n_sim)
        if verbose:
            print(f"Running {n_sim} bootstrap samples...")
    
    # Bootstrap loop
    for b in iterator:
        if method == 'nonparametric':
            # Resample sequences with replacement
            bootstrap_indices = rng.choice(n_sequences, size=n_sequences, replace=True)
            
            # Create bootstrap dataset
            bootstrap_obs = _resample_sequences(model.observations, bootstrap_indices)
            
            # Create and fit bootstrap model
            try:
                bootstrap_model_obj = _create_bootstrap_model(model, bootstrap_obs)
                bootstrap_model_obj = _fit_bootstrap_model(bootstrap_model_obj)
                
                # Extract parameters
                params = _extract_parameters(bootstrap_model_obj)
                bootstrap_samples.append(params)
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Bootstrap sample {b+1} failed: {e}")
                continue
        
        else:
            raise ValueError(f"Unknown bootstrap method: {method}")
    
    if len(bootstrap_samples) == 0:
        raise ValueError("All bootstrap samples failed. Check model fitting.")
    
    # Compute summary statistics
    summary = _compute_bootstrap_summary(bootstrap_samples, model)
    
    return {
        'bootstrap_samples': bootstrap_samples,
        'original_model': model,
        'n_sim': n_sim,
        'n_successful': len(bootstrap_samples),
        'method': method,
        'summary': summary
    }


def _resample_sequences(observations: SequenceData, indices: np.ndarray) -> SequenceData:
    """
    Resample sequences based on bootstrap indices.
    
    Args:
        observations: Original SequenceData object
        indices: Bootstrap indices (which sequences to include, with replacement)
        
    Returns:
        SequenceData: Resampled SequenceData object
    """
    import pandas as pd
    
    # Get the original DataFrame
    original_df = observations.to_dataframe()
    
    # Resample rows based on indices
    resampled_df = original_df.iloc[indices].copy()
    
    # Reset index to create new sequence IDs
    resampled_df = resampled_df.reset_index(drop=True)
    
    # Get time columns from original observations
    # We need to extract the time column names from the original data
    # This is a bit tricky - we'll use the values attribute
    time_cols = observations.values.columns.tolist() if hasattr(observations, 'values') else None
    
    # If we can't get time columns directly, try to infer from sequence length
    if time_cols is None:
        # Get max sequence length
        max_length = max(len(seq) for seq in observations.sequences)
        time_cols = list(range(1, max_length + 1))
    
    # Create new SequenceData object
    seq_data = SequenceData(
        resampled_df,
        time=time_cols,
        states=observations.states,
        labels=observations.labels,
        id_col=None
    )
    
    return seq_data


def _create_bootstrap_model(
    original_model: Union[HMM, MHMM, NHMM],
    bootstrap_obs: SequenceData
) -> Union[HMM, MHMM, NHMM]:
    """
    Create a new model object for bootstrap sample.
    
    Args:
        original_model: Original fitted model
        bootstrap_obs: Bootstrap resampled observations
        
    Returns:
        New model object with same structure as original
    """
    if isinstance(original_model, HMM):
        from .build_hmm import build_hmm
        return build_hmm(
            bootstrap_obs,
            n_states=original_model.n_states,
            initial_probs=original_model.initial_probs.copy(),
            transition_probs=original_model.transition_probs.copy(),
            emission_probs=original_model.emission_probs.copy(),
            state_names=original_model.state_names,
            random_state=None
        )
    
    elif isinstance(original_model, MHMM):
        from .build_mhmm import build_mhmm
        
        # Get cluster parameters
        initial_probs_list = [c.initial_probs.copy() for c in original_model.clusters]
        transition_probs_list = [c.transition_probs.copy() for c in original_model.clusters]
        emission_probs_list = [c.emission_probs.copy() for c in original_model.clusters]
        state_names_list = [c.state_names for c in original_model.clusters]
        
        return build_mhmm(
            bootstrap_obs,
            n_clusters=original_model.n_clusters,
            n_states=[c.n_states for c in original_model.clusters],
            initial_probs=initial_probs_list,
            transition_probs=transition_probs_list,
            emission_probs=emission_probs_list,
            cluster_probs=original_model.cluster_probs.copy(),
            cluster_names=original_model.cluster_names,
            state_names=state_names_list,
            random_state=None
        )
    
    elif isinstance(original_model, NHMM):
        from .build_nhmm import build_nhmm
        return build_nhmm(
            bootstrap_obs,
            n_states=original_model.n_states,
            X=original_model.X,  # Use same covariates (or resample if needed)
            eta_pi=original_model.eta_pi.copy(),
            eta_A=original_model.eta_A.copy(),
            eta_B=original_model.eta_B.copy(),
            state_names=original_model.state_names,
            random_state=None
        )
    
    else:
        raise ValueError(f"Unknown model type: {type(original_model)}")


def _fit_bootstrap_model(model: Union[HMM, MHMM, NHMM]) -> Union[HMM, MHMM, NHMM]:
    """
    Fit a bootstrap model.
    
    Args:
        model: Bootstrap model object
        
    Returns:
        Fitted model
    """
    if isinstance(model, HMM):
        from .fit_model import fit_model
        return fit_model(model, n_iter=50, tol=1e-2, verbose=False)
    
    elif isinstance(model, MHMM):
        from .fit_mhmm import fit_mhmm
        return fit_mhmm(model, n_iter=50, tol=1e-2, verbose=False)
    
    elif isinstance(model, NHMM):
        from .fit_nhmm import fit_nhmm
        return fit_nhmm(model, n_iter=50, tol=1e-3, verbose=False)
    
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def _extract_parameters(model: Union[HMM, MHMM, NHMM]) -> dict:
    """
    Extract parameters from a fitted model.
    
    Args:
        model: Fitted model object
        
    Returns:
        dict: Dictionary of parameters
    """
    if isinstance(model, HMM):
        return {
            'initial_probs': model.initial_probs.copy(),
            'transition_probs': model.transition_probs.copy(),
            'emission_probs': model.emission_probs.copy()
        }
    
    elif isinstance(model, MHMM):
        return {
            'cluster_probs': model.cluster_probs.copy(),
            'clusters': [
                {
                    'initial_probs': c.initial_probs.copy(),
                    'transition_probs': c.transition_probs.copy(),
                    'emission_probs': c.emission_probs.copy()
                }
                for c in model.clusters
            ],
            'coefficients': model.coefficients.copy() if model.coefficients is not None else None
        }
    
    elif isinstance(model, NHMM):
        return {
            'eta_pi': model.eta_pi.copy(),
            'eta_A': model.eta_A.copy(),
            'eta_B': model.eta_B.copy()
        }
    
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def _compute_bootstrap_summary(
    bootstrap_samples: List[dict],
    original_model: Union[HMM, MHMM, NHMM]
) -> dict:
    """
    Compute summary statistics from bootstrap samples.
    
    Args:
        bootstrap_samples: List of parameter dictionaries from bootstrap samples
        original_model: Original fitted model
        
    Returns:
        dict: Summary statistics including means, stds, and confidence intervals
    """
    summary = {}
    
    if isinstance(original_model, HMM):
        # Stack arrays
        initial_probs_stack = np.array([s['initial_probs'] for s in bootstrap_samples])
        transition_probs_stack = np.array([s['transition_probs'] for s in bootstrap_samples])
        emission_probs_stack = np.array([s['emission_probs'] for s in bootstrap_samples])
        
        # Compute statistics
        summary['initial_probs'] = {
            'mean': np.mean(initial_probs_stack, axis=0),
            'std': np.std(initial_probs_stack, axis=0),
            'ci_95': np.percentile(initial_probs_stack, [2.5, 97.5], axis=0)
        }
        
        summary['transition_probs'] = {
            'mean': np.mean(transition_probs_stack, axis=0),
            'std': np.std(transition_probs_stack, axis=0),
            'ci_95': np.percentile(transition_probs_stack, [2.5, 97.5], axis=0)
        }
        
        summary['emission_probs'] = {
            'mean': np.mean(emission_probs_stack, axis=0),
            'std': np.std(emission_probs_stack, axis=0),
            'ci_95': np.percentile(emission_probs_stack, [2.5, 97.5], axis=0)
        }
    
    elif isinstance(original_model, MHMM):
        # Cluster probabilities
        cluster_probs_stack = np.array([s['cluster_probs'] for s in bootstrap_samples])
        summary['cluster_probs'] = {
            'mean': np.mean(cluster_probs_stack, axis=0),
            'std': np.std(cluster_probs_stack, axis=0),
            'ci_95': np.percentile(cluster_probs_stack, [2.5, 97.5], axis=0)
        }
        
        # Cluster-specific parameters
        summary['clusters'] = []
        for k in range(original_model.n_clusters):
            cluster_params = {
                'initial_probs': np.array([s['clusters'][k]['initial_probs'] for s in bootstrap_samples]),
                'transition_probs': np.array([s['clusters'][k]['transition_probs'] for s in bootstrap_samples]),
                'emission_probs': np.array([s['clusters'][k]['emission_probs'] for s in bootstrap_samples])
            }
            
            summary['clusters'].append({
                'initial_probs': {
                    'mean': np.mean(cluster_params['initial_probs'], axis=0),
                    'std': np.std(cluster_params['initial_probs'], axis=0),
                    'ci_95': np.percentile(cluster_params['initial_probs'], [2.5, 97.5], axis=0)
                },
                'transition_probs': {
                    'mean': np.mean(cluster_params['transition_probs'], axis=0),
                    'std': np.std(cluster_params['transition_probs'], axis=0),
                    'ci_95': np.percentile(cluster_params['transition_probs'], [2.5, 97.5], axis=0)
                },
                'emission_probs': {
                    'mean': np.mean(cluster_params['emission_probs'], axis=0),
                    'std': np.std(cluster_params['emission_probs'], axis=0),
                    'ci_95': np.percentile(cluster_params['emission_probs'], [2.5, 97.5], axis=0)
                }
            })
    
    elif isinstance(original_model, NHMM):
        # Coefficients
        eta_pi_stack = np.array([s['eta_pi'] for s in bootstrap_samples])
        eta_A_stack = np.array([s['eta_A'] for s in bootstrap_samples])
        eta_B_stack = np.array([s['eta_B'] for s in bootstrap_samples])
        
        summary['eta_pi'] = {
            'mean': np.mean(eta_pi_stack, axis=0),
            'std': np.std(eta_pi_stack, axis=0),
            'ci_95': np.percentile(eta_pi_stack, [2.5, 97.5], axis=0)
        }
        
        summary['eta_A'] = {
            'mean': np.mean(eta_A_stack, axis=0),
            'std': np.std(eta_A_stack, axis=0),
            'ci_95': np.percentile(eta_A_stack, [2.5, 97.5], axis=0)
        }
        
        summary['eta_B'] = {
            'mean': np.mean(eta_B_stack, axis=0),
            'std': np.std(eta_B_stack, axis=0),
            'ci_95': np.percentile(eta_B_stack, [2.5, 97.5], axis=0)
        }
    
    return summary
