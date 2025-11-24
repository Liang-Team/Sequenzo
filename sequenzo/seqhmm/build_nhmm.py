"""
@Author  : Yuqi Liang 梁彧祺
@File    : build_nhmm.py
@Time    : 2025-11-22 19:30
@Desc    : Build Non-homogeneous HMM models

This module provides the build_nhmm function, which creates Non-homogeneous HMM
model objects similar to seqHMM's build_nhmm() function in R.

Note: This is a simplified implementation. A full implementation would require
more sophisticated handling of formulas and data structures.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union
from sequenzo.define_sequence_data import SequenceData
from .nhmm import NHMM
from .formulas import Formula


def build_nhmm(
    observations: SequenceData,
    n_states: int,
    X: Optional[np.ndarray] = None,
    emission_formula: Optional[Union[str, Formula]] = None,
    initial_formula: Optional[Union[str, Formula]] = None,
    transition_formula: Optional[Union[str, Formula]] = None,
    data: Optional[pd.DataFrame] = None,
    id_var: Optional[str] = None,
    time_var: Optional[str] = None,
    eta_pi: Optional[np.ndarray] = None,
    eta_A: Optional[np.ndarray] = None,
    eta_B: Optional[np.ndarray] = None,
    state_names: Optional[List[str]] = None,
    random_state: Optional[int] = None
) -> NHMM:
    """
    Build a Non-homogeneous Hidden Markov Model object.
    
    A Non-homogeneous HMM allows transition and emission probabilities to vary
    over time or with covariates. This function creates the model structure but
    does not fit it (use fit_nhmm() for that).
    
    It is similar to seqHMM's build_nhmm() function in R. Supports both
    direct covariate matrix input and formula-based specification.
    
    Args:
        observations: SequenceData object containing the sequences to model
        n_states: Number of hidden states
        X: Optional covariate matrix of shape (n_sequences, n_timepoints, n_covariates).
           If None, will be created from formulas.
        emission_formula: Optional formula string for emission probabilities (e.g., "~ x1 + x2")
        initial_formula: Optional formula string for initial probabilities
        transition_formula: Optional formula string for transition probabilities
        data: Optional DataFrame containing covariates (required if using formulas)
        id_var: Optional column name for sequence IDs in data (required if using formulas)
        time_var: Optional column name for time variable in data (required if using formulas)
        eta_pi: Optional coefficients for initial probabilities (n_covariates x n_states)
        eta_A: Optional coefficients for transition probabilities (n_covariates x n_states x n_states)
        eta_B: Optional coefficients for emission probabilities (n_covariates x n_states x n_symbols)
        state_names: Optional names for hidden states
        random_state: Random seed for initialization
        
    Returns:
        NHMM: A Non-homogeneous HMM model object (not yet fitted)
        
    Examples:
        >>> from sequenzo import SequenceData, load_dataset
        >>> from sequenzo.seqhmm import build_nhmm
        >>> import numpy as np
        >>> 
        >>> # Method 1: Direct covariate matrix
        >>> n_sequences = len(seq.sequences)
        >>> n_timepoints = max(len(s) for s in seq.sequences)
        >>> X = np.zeros((n_sequences, n_timepoints, 1))
        >>> for i in range(n_sequences):
        ...     for t in range(len(seq.sequences[i])):
        ...         X[i, t, 0] = t  # Time covariate
        >>> nhmm = build_nhmm(seq, n_states=4, X=X, random_state=42)
        >>> 
        >>> # Method 2: Formula-based (requires data DataFrame)
        >>> nhmm = build_nhmm(
        ...     seq, n_states=4,
        ...     emission_formula="~ time + age",
        ...     data=covariate_df,
        ...     id_var='id',
        ...     time_var='time',
        ...     random_state=42
        ... )
    """
    # Create covariate matrix from formulas if X is not provided
    if X is None:
        if data is None or id_var is None or time_var is None:
            raise ValueError(
                "If X is not provided, must provide data, id_var, and time_var for formula-based specification."
            )
        
        # Use emission_formula as default if others not specified
        formula = emission_formula or initial_formula or transition_formula
        if formula is None:
            raise ValueError("Must provide either X or at least one formula (emission_formula, initial_formula, or transition_formula).")
        
        # Create model matrix
        n_sequences = len(observations.sequences)
        n_timepoints = max(len(seq) for seq in observations.sequences)
        X = create_model_matrix(formula, data, id_var, time_var, n_sequences, n_timepoints)
    
    # Create and return NHMM object
    nhmm = NHMM(
        observations=observations,
        n_states=n_states,
        X=X,
        eta_pi=eta_pi,
        eta_A=eta_A,
        eta_B=eta_B,
        state_names=state_names,
        random_state=random_state
    )
    
    return nhmm
