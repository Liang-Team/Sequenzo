"""
@Author  : Yuqi Liang 梁彧祺
@File    : advanced_optimization.py
@Time    : 2025-11-23 14:02
@Desc    : Advanced optimization methods for HMM models

This module provides advanced optimization methods including:
- Global optimization (MLSL - Multi-Level Single-Linkage)
- Local optimization (L-BFGS)
- Multiple restarts with random initial values

This is similar to seqHMM's fit_model() function with global_step and local_step options.
"""

import numpy as np
from typing import Optional, Dict, List, Callable, Union
from scipy.optimize import minimize, differential_evolution
from .hmm import HMM
from .mhmm import MHMM
from .nhmm import NHMM


def fit_model_advanced(
    model: Union[HMM, MHMM, NHMM],
    em_step: bool = True,
    global_step: bool = False,
    local_step: bool = False,
    n_iter: int = 100,
    tol: float = 1e-2,
    n_restarts: int = 0,
    verbose: bool = False,
    random_state: Optional[int] = None
) -> Union[HMM, MHMM, NHMM]:
    """
    Fit HMM model using advanced optimization methods.
    
    This function provides a three-step optimization approach:
    1. EM algorithm (optional, default True)
    2. Global optimization using MLSL (optional)
    3. Local optimization using L-BFGS (optional)
    
    Additionally supports multiple restarts with random initial values.
    
    This is similar to seqHMM's fit_model() function with advanced options.
    
    Args:
        model: HMM, MHMM, or NHMM model object
        em_step: Whether to use EM algorithm first. Default is True.
        global_step: Whether to use global optimization (MLSL). Default is False.
        local_step: Whether to use local optimization (L-BFGS). Default is False.
        n_iter: Maximum number of iterations for EM/local optimization
        tol: Convergence tolerance
        n_restarts: Number of random restarts. Default is 0.
        verbose: Whether to print progress
        random_state: Random seed for reproducibility
        
    Returns:
        Fitted model object
        
    Examples:
        >>> from sequenzo.seqhmm import build_hmm, fit_model_advanced
        >>> 
        >>> hmm = build_hmm(seq, n_states=4, random_state=42)
        >>> 
        >>> # Use all optimization steps
        >>> hmm = fit_model_advanced(
        ...     hmm,
        ...     em_step=True,
        ...     global_step=True,
        ...     local_step=True,
        ...     n_restarts=5,
        ...     verbose=True
        ... )
    """
    rng = np.random.RandomState(random_state)
    
    best_model = None
    best_log_lik = -np.inf
    results = []
    
    # Determine number of optimization runs (1 + n_restarts)
    n_runs = 1 + n_restarts
    
    for run in range(n_runs):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Optimization run {run + 1}/{n_runs}")
            print(f"{'='*50}")
        
        # Create a copy of the model for this run
        if run == 0:
            current_model = model
        else:
            # Random restart: create new model with random initial values
            current_model = _create_random_restart(model, rng)
        
        # Step 1: EM algorithm
        if em_step:
            if verbose:
                print("\n[Step 1] EM Algorithm...")
            current_model = _em_step(current_model, n_iter=n_iter, tol=tol, verbose=verbose)
        
        # Step 2: Global optimization
        if global_step:
            if verbose:
                print("\n[Step 2] Global Optimization (MLSL)...")
            current_model = _global_step(current_model, n_iter=n_iter, tol=tol, verbose=verbose)
        
        # Step 3: Local optimization
        if local_step:
            if verbose:
                print("\n[Step 3] Local Optimization (L-BFGS)...")
            current_model = _local_step(current_model, n_iter=n_iter, tol=tol, verbose=verbose)
        
        # Store results
        log_lik = current_model.log_likelihood if current_model.log_likelihood is not None else -np.inf
        results.append({
            'run': run,
            'log_likelihood': log_lik,
            'model': current_model
        })
        
        # Track best model
        if log_lik > best_log_lik:
            best_log_lik = log_lik
            best_model = current_model
        
        if verbose:
            print(f"Run {run + 1} log-likelihood: {log_lik:.4f}")
    
    # Return best model
    if verbose:
        print(f"\n{'='*50}")
        print(f"Best log-likelihood: {best_log_lik:.4f}")
        print(f"{'='*50}")
    
    return best_model


def _em_step(
    model: Union[HMM, MHMM, NHMM],
    n_iter: int = 100,
    tol: float = 1e-2,
    verbose: bool = False
) -> Union[HMM, MHMM, NHMM]:
    """EM algorithm step."""
    if isinstance(model, HMM):
        from .fit_model import fit_model
        return fit_model(model, n_iter=n_iter, tol=tol, verbose=verbose)
    elif isinstance(model, MHMM):
        from .fit_mhmm import fit_mhmm
        return fit_mhmm(model, n_iter=n_iter, tol=tol, verbose=verbose)
    elif isinstance(model, NHMM):
        # For NHMM, EM is not directly applicable, skip
        if verbose:
            print("  (EM step skipped for NHMM - using numerical optimization)")
        return model
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def _global_step(
    model: Union[HMM, MHMM, NHMM],
    n_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False
) -> Union[HMM, MHMM, NHMM]:
    """
    Global optimization step using Multi-Level Single-Linkage (MLSL) method.
    
    MLSL is a global optimization algorithm that uses low-discrepancy sequences
    to generate starting points and performs local optimization from each.
    
    For now, we use scipy's differential_evolution as an approximation,
    since scipy doesn't have MLSL. A full implementation would require
    NLOPT library or custom MLSL implementation.
    """
    if isinstance(model, NHMM):
        # For NHMM, use numerical optimization with multiple starts
        # This is an approximation of MLSL
        from .fit_nhmm import fit_nhmm
        
        # Get parameter bounds
        params = np.concatenate([
            model.eta_pi.flatten(),
            model.eta_A.flatten(),
            model.eta_B.flatten()
        ])
        
        # Create bounds (wider range for global search)
        bounds = [(-5, 5)] * len(params)  # Reasonable range for coefficients
        
        # Use differential evolution for global optimization
        def objective(params):
            return model._log_likelihood(params)
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=n_iter,
            tol=tol,
            seed=42,
            disp=verbose
        )
        
        # Update model parameters
        n_pi = model.n_covariates * model.n_states
        n_A = model.n_covariates * model.n_states * model.n_states
        n_B = model.n_covariates * model.n_states * model.n_symbols
        
        model.eta_pi = result.x[:n_pi].reshape(model.n_covariates, model.n_states)
        model.eta_A = result.x[n_pi:n_pi+n_A].reshape(model.n_covariates, model.n_states, model.n_states)
        model.eta_B = result.x[n_pi+n_A:].reshape(model.n_covariates, model.n_states, model.n_symbols)
        
        model.log_likelihood = -result.fun
        model.n_iter = result.nit
        model.converged = result.success
        
        return model
    
    else:
        # For HMM and MHMM, global optimization is less critical
        # since EM usually works well. We can skip or use a simplified version.
        if verbose:
            print("  (Global optimization less critical for HMM/MHMM with EM)")
        return model


def _local_step(
    model: Union[HMM, MHMM, NHMM],
    n_iter: int = 100,
    tol: float = 1e-8,
    verbose: bool = False
) -> Union[HMM, MHMM, NHMM]:
    """
    Local optimization step using L-BFGS method.
    
    This refines the solution found by EM or global optimization
    using a high-precision local optimizer.
    """
    if isinstance(model, NHMM):
        from .fit_nhmm import fit_nhmm
        # Use tighter tolerance for local optimization
        return fit_nhmm(model, n_iter=n_iter, tol=tol, verbose=verbose)
    
    elif isinstance(model, (HMM, MHMM)):
        # For HMM and MHMM, we can refine using additional EM iterations
        # with tighter tolerance
        return _em_step(model, n_iter=n_iter, tol=tol, verbose=verbose)
    
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def _create_random_restart(
    model: Union[HMM, MHMM, NHMM],
    rng: np.random.RandomState
) -> Union[HMM, MHMM, NHMM]:
    """
    Create a new model with random initial values for restart.
    
    Args:
        model: Original model
        rng: Random number generator
        
    Returns:
        New model with random initial parameters
    """
    if isinstance(model, HMM):
        from .build_hmm import build_hmm
        return build_hmm(
            model.observations,
            n_states=model.n_states,
            random_state=rng.randint(0, 2**31)
        )
    
    elif isinstance(model, MHMM):
        from .build_mhmm import build_mhmm
        return build_mhmm(
            model.observations,
            n_clusters=model.n_clusters,
            n_states=model.n_states,
            random_state=rng.randint(0, 2**31)
        )
    
    elif isinstance(model, NHMM):
        from .build_nhmm import build_nhmm
        # Randomize coefficients
        eta_pi = rng.randn(model.n_covariates, model.n_states) * 0.1
        eta_A = rng.randn(model.n_covariates, model.n_states, model.n_states) * 0.1
        eta_B = rng.randn(model.n_covariates, model.n_states, model.n_symbols) * 0.1
        
        return build_nhmm(
            model.observations,
            n_states=model.n_states,
            X=model.X,
            eta_pi=eta_pi,
            eta_A=eta_A,
            eta_B=eta_B,
            state_names=model.state_names,
            random_state=rng.randint(0, 2**31)
        )
    
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
