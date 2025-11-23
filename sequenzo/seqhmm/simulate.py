"""
@Author  : Yuqi Liang 梁彧祺
@File    : simulate.py
@Time    : 2025-10-12 10:47
@Desc    : Model simulation functions for HMM models

This module provides functions for simulating sequences from HMM models,
similar to seqHMM's simulate_hmm() and simulate_mhmm() functions in R.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict
from sequenzo.define_sequence_data import SequenceData
from .hmm import HMM
from .mhmm import MHMM
from .formulas import create_model_matrix_time_constant


def simulate_hmm(
    n_sequences: int,
    initial_probs: np.ndarray,
    transition_probs: np.ndarray,
    emission_probs: np.ndarray,
    sequence_length: int,
    alphabet: Optional[List[str]] = None,
    state_names: Optional[List[str]] = None,
    random_state: Optional[int] = None
) -> dict:
    """
    Simulate sequences from a Hidden Markov Model.
    
    This function generates sequences of observed and hidden states given
    HMM parameters. It is similar to seqHMM's simulate_hmm() function in R.
    
    Args:
        n_sequences: Number of sequences to simulate
        initial_probs: Initial state probabilities (n_states,)
        transition_probs: Transition probability matrix (n_states x n_states)
        emission_probs: Emission probability matrix (n_states x n_symbols)
        sequence_length: Length of each simulated sequence
        alphabet: Optional list of observed state symbols. If None, uses integers.
        state_names: Optional list of hidden state names. If None, uses integers.
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Dictionary with keys:
            - 'observations': List of observed sequences (as lists)
            - 'states': List of hidden state sequences (as lists)
            - 'observations_df': DataFrame format (for creating SequenceData)
            
    Examples:
        >>> from sequenzo.seqhmm import simulate_hmm
        >>> import numpy as np
        >>> 
        >>> # Define HMM parameters
        >>> initial_probs = np.array([0.5, 0.5])
        >>> transition_probs = np.array([[0.7, 0.3], [0.3, 0.7]])
        >>> emission_probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        >>> 
        >>> # Simulate 10 sequences of length 20
        >>> sim = simulate_hmm(
        ...     n_sequences=10,
        ...     initial_probs=initial_probs,
        ...     transition_probs=transition_probs,
        ...     emission_probs=emission_probs,
        ...     sequence_length=20,
        ...     alphabet=['A', 'B'],
        ...     random_state=42
        ... )
        >>> 
        >>> print(f"Simulated {len(sim['observations'])} sequences")
        >>> print(f"First sequence: {sim['observations'][0]}")
    """
    rng = np.random.RandomState(random_state)
    
    n_states = len(initial_probs)
    n_symbols = emission_probs.shape[1]
    
    # Validate dimensions
    if transition_probs.shape != (n_states, n_states):
        raise ValueError(
            f"transition_probs shape ({transition_probs.shape}) must be ({n_states}, {n_states})"
        )
    if emission_probs.shape != (n_states, n_symbols):
        raise ValueError(
            f"emission_probs shape ({emission_probs.shape}) must be ({n_states}, {n_symbols})"
        )
    
    # Set default names
    if alphabet is None:
        alphabet = [str(i) for i in range(n_symbols)]
    if state_names is None:
        state_names = [str(i) for i in range(n_states)]
    
    # Initialize arrays
    states = []
    observations = []
    
    # Simulate sequences
    for seq_idx in range(n_sequences):
        seq_states = []
        seq_obs = []
        
        # Sample initial state
        initial_state_idx = rng.choice(n_states, p=initial_probs)
        seq_states.append(state_names[initial_state_idx])
        
        # Sample initial observation
        obs_idx = rng.choice(n_symbols, p=emission_probs[initial_state_idx, :])
        seq_obs.append(alphabet[obs_idx])
        
        # Simulate remaining time points
        current_state = initial_state_idx
        for t in range(1, sequence_length):
            # Sample next state
            current_state = rng.choice(n_states, p=transition_probs[current_state, :])
            seq_states.append(state_names[current_state])
            
            # Sample observation
            obs_idx = rng.choice(n_symbols, p=emission_probs[current_state, :])
            seq_obs.append(alphabet[obs_idx])
        
        states.append(seq_states)
        observations.append(seq_obs)
    
    # Create DataFrame format for easy conversion to SequenceData
    # This format: one row per sequence, columns are time points
    obs_dict = {}
    for t in range(sequence_length):
        obs_dict[f'time_{t+1}'] = [obs[t] for obs in observations]
    
    observations_df = pd.DataFrame(obs_dict)
    
    return {
        'observations': observations,
        'states': states,
        'observations_df': observations_df,
        'alphabet': alphabet,
        'state_names': state_names
    }


def compute_mixture_probs_from_covariates(
    X: np.ndarray,
    coefficients: np.ndarray
) -> np.ndarray:
    """
    Compute mixture probabilities from covariates and coefficients using multinomial logit.
    
    This function implements the softmax (multinomial logit) link function to convert
    linear predictors (X @ coefficients) into probabilities. The first column of
    coefficients is set to zero (reference category).
    
    Formula: P(cluster k | covariates) = exp(X @ coefficients[:, k]) / sum(exp(X @ coefficients))
    
    Args:
        X: Model matrix of shape (n_sequences, n_covariates) including intercept
        coefficients: Coefficient matrix of shape (n_covariates, n_clusters)
                     First column should be zeros (reference category)
        
    Returns:
        numpy array: Mixture probabilities of shape (n_sequences, n_clusters)
                    Each row sums to 1
                    
    Examples:
        >>> import numpy as np
        >>> from sequenzo.seqhmm.simulate import compute_mixture_probs_from_covariates
        >>> 
        >>> # Create model matrix (intercept + 1 covariate, 3 sequences)
        >>> X = np.array([[1, 0.5], [1, 1.0], [1, 1.5]])
        >>> 
        >>> # Create coefficients (2 covariates x 3 clusters, first column zeros)
        >>> coefs = np.array([
        ...     [0, -1.5, 0.5],      # intercepts (first cluster is reference)
        ...     [0, 3.0, -0.7]       # covariate effects
        ... ])
        >>> 
        >>> # Compute mixture probabilities
        >>> probs = compute_mixture_probs_from_covariates(X, coefs)
        >>> print(probs.shape)  # (3, 3) - 3 sequences, 3 clusters
        >>> print(probs.sum(axis=1))  # [1. 1. 1.] - each row sums to 1
    """
    # Compute linear predictors: X @ coefficients
    # Result shape: (n_sequences, n_clusters)
    linear_predictors = X @ coefficients
    
    # Apply softmax (multinomial logit) to convert to probabilities
    # For numerical stability, subtract the max before exponentiating
    # This doesn't change the result but prevents overflow
    max_vals = np.max(linear_predictors, axis=1, keepdims=True)
    exp_vals = np.exp(linear_predictors - max_vals)
    
    # Normalize so each row sums to 1
    probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    
    return probs


def simulate_mhmm(
    n_sequences: int,
    n_clusters: int,
    initial_probs: List[np.ndarray],
    transition_probs: List[np.ndarray],
    emission_probs: List[np.ndarray],
    cluster_probs: Optional[np.ndarray] = None,
    sequence_length: Optional[int] = None,
    alphabet: Optional[List[str]] = None,
    state_names: Optional[List[List[str]]] = None,
    cluster_names: Optional[List[str]] = None,
    formula: Optional[Union[str, None]] = None,
    data: Optional[pd.DataFrame] = None,
    coefficients: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> dict:
    """
    Simulate sequences from a Mixture Hidden Markov Model.
    
    This function generates sequences from a Mixture HMM, where each sequence
    is first assigned to a cluster, then simulated from that cluster's HMM.
    It is similar to seqHMM's simulate_mhmm() function in R.
    
    Cluster assignments can be done in two ways:
    1. Using fixed cluster probabilities (cluster_probs parameter)
    2. Using formula-based covariates (formula, data, coefficients parameters)
    
    When using formula-based covariates, mixture probabilities are computed
    using multinomial logit (softmax) from covariates and coefficients.
    
    Args:
        n_sequences: Number of sequences to simulate
        n_clusters: Number of clusters
        initial_probs: List of initial state probabilities, one per cluster
        transition_probs: List of transition matrices, one per cluster
        emission_probs: List of emission matrices, one per cluster
        cluster_probs: Optional fixed cluster probabilities (n_clusters,).
                      Either cluster_probs OR (formula + data + coefficients) must be provided.
        sequence_length: Length of each simulated sequence
        alphabet: Optional list of observed state symbols
        state_names: Optional list of state name lists, one per cluster
        cluster_names: Optional names for clusters
        formula: Optional formula string (e.g., "~ covariate_1 + covariate_2")
                for time-constant covariates. If provided, data and coefficients must also be provided.
        data: Optional DataFrame containing covariates (one row per sequence).
              Required if formula is provided.
        coefficients: Optional coefficient matrix of shape (n_covariates, n_clusters)
                     for formula-based covariates. First column should be zeros (reference category).
                     Required if formula is provided.
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Dictionary with keys:
            - 'observations': List of observed sequences
            - 'states': List of hidden state sequences
            - 'clusters': List of cluster assignments
            - 'observations_df': DataFrame format
            
    Examples:
        >>> from sequenzo.seqhmm import simulate_mhmm
        >>> import numpy as np
        >>> import pandas as pd
        >>> 
        >>> # Method 1: Fixed cluster probabilities
        >>> initial_probs = [np.array([0.5, 0.5]), np.array([0.3, 0.7])]
        >>> transition_probs = [
        ...     np.array([[0.7, 0.3], [0.3, 0.7]]),
        ...     np.array([[0.8, 0.2], [0.2, 0.8]])
        ... ]
        >>> emission_probs = [
        ...     np.array([[0.9, 0.1], [0.1, 0.9]]),
        ...     np.array([[0.7, 0.3], [0.3, 0.7]])
        ... ]
        >>> cluster_probs = np.array([0.6, 0.4])
        >>> 
        >>> sim = simulate_mhmm(
        ...     n_sequences=10,
        ...     n_clusters=2,
        ...     initial_probs=initial_probs,
        ...     transition_probs=transition_probs,
        ...     emission_probs=emission_probs,
        ...     cluster_probs=cluster_probs,
        ...     sequence_length=20,
        ...     alphabet=['A', 'B'],
        ...     random_state=42
        ... )
        >>> 
        >>> # Method 2: Formula-based covariates
        >>> # Create covariate data
        >>> data = pd.DataFrame({
        ...     'covariate_1': np.random.rand(30),
        ...     'covariate_2': np.random.choice(['A', 'B'], size=30)
        ... })
        >>> 
        >>> # Define coefficients (intercept + 2 covariates) x (2 clusters)
        >>> # First column is zeros (reference), second column has effects
        >>> coefs = np.array([
        ...     [0, -1.5],        # intercepts
        ...     [0, 3.0],         # covariate_1 effect
        ...     [0, -0.7]         # covariate_2_B effect (dummy for 'B')
        ... ])
        >>> 
        >>> sim = simulate_mhmm(
        ...     n_sequences=30,
        ...     n_clusters=2,
        ...     initial_probs=initial_probs,
        ...     transition_probs=transition_probs,
        ...     emission_probs=emission_probs,
        ...     sequence_length=20,
        ...     formula="~ covariate_1 + covariate_2",
        ...     data=data,
        ...     coefficients=coefs,
        ...     alphabet=['A', 'B'],
        ...     random_state=42
        ... )
    """
    rng = np.random.RandomState(random_state)
    
    # Validate sequence_length is provided
    if sequence_length is None:
        raise ValueError("sequence_length must be provided")
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be at least 1, got {sequence_length}")
    
    # Validate inputs
    if len(initial_probs) != n_clusters:
        raise ValueError(f"initial_probs length ({len(initial_probs)}) must equal n_clusters ({n_clusters})")
    if len(transition_probs) != n_clusters:
        raise ValueError(f"transition_probs length ({len(transition_probs)}) must equal n_clusters ({n_clusters})")
    if len(emission_probs) != n_clusters:
        raise ValueError(f"emission_probs length ({len(emission_probs)}) must equal n_clusters ({n_clusters})")
    
    # Validate that either cluster_probs OR formula-based approach is used
    use_formula = (formula is not None)
    use_fixed_probs = (cluster_probs is not None)
    
    if use_formula and use_fixed_probs:
        raise ValueError(
            "Cannot specify both cluster_probs and formula-based covariates. "
            "Use either cluster_probs OR (formula + data + coefficients)."
        )
    
    if not use_formula and not use_fixed_probs:
        raise ValueError(
            "Must specify either cluster_probs OR (formula + data + coefficients) "
            "for cluster assignment probabilities."
        )
    
    if use_formula:
        # Validate formula-based inputs
        if data is None:
            raise ValueError("If formula is provided, data must also be provided")
        if coefficients is None:
            raise ValueError("If formula is provided, coefficients must also be provided")
        
        # Create model matrix from formula and data
        # Step 1: Create model matrix X of shape (n_sequences, n_covariates)
        X = create_model_matrix_time_constant(formula, data, n_sequences)
        n_covariates = X.shape[1]
        
        # Step 2: Validate coefficients matrix
        if coefficients.shape != (n_covariates, n_clusters):
            raise ValueError(
                f"coefficients shape ({coefficients.shape}) must be "
                f"(n_covariates, n_clusters) = ({n_covariates}, {n_clusters}). "
                f"Note: n_covariates includes intercept and any dummy variables from categorical covariates."
            )
        
        # Step 3: Ensure first column of coefficients is zeros (reference category)
        coefficients = coefficients.copy()  # Don't modify original
        coefficients[:, 0] = 0.0
        
        # Step 4: Compute mixture probabilities from covariates
        # Result: (n_sequences, n_clusters) - probabilities for each sequence
        mixture_probs = compute_mixture_probs_from_covariates(X, coefficients)
        
    else:
        # Use fixed cluster probabilities
        if len(cluster_probs) != n_clusters:
            raise ValueError(f"cluster_probs length ({len(cluster_probs)}) must equal n_clusters ({n_clusters})")
        
        # Check that probabilities sum to approximately 1
        if not np.isclose(np.sum(cluster_probs), 1.0):
            raise ValueError(f"cluster_probs must sum to 1.0, but sum is {np.sum(cluster_probs)}")
        
        # Broadcast to (n_sequences, n_clusters) - same probabilities for all sequences
        mixture_probs = np.tile(cluster_probs, (n_sequences, 1))
    
    # Get alphabet from first cluster
    n_symbols = emission_probs[0].shape[1]
    if alphabet is None:
        alphabet = [str(i) for i in range(n_symbols)]
    
    # Set default names
    if cluster_names is None:
        cluster_names = [f"Cluster {i+1}" for i in range(n_clusters)]
    
    if state_names is None:
        state_names = []
        for k in range(n_clusters):
            n_states_k = len(initial_probs[k])
            state_names.append([str(i) for i in range(n_states_k)])
    
    # Initialize arrays
    observations = []
    states = []
    clusters = []
    
    # Simulate sequences
    for seq_idx in range(n_sequences):
        # Sample cluster assignment using probabilities for this specific sequence
        # If using formula-based covariates, each sequence has different probabilities
        # If using fixed cluster_probs, all sequences have the same probabilities
        cluster_idx = rng.choice(n_clusters, p=mixture_probs[seq_idx, :])
        clusters.append(cluster_names[cluster_idx])
        
        # Get parameters for this cluster
        cluster_initial = initial_probs[cluster_idx]
        cluster_transition = transition_probs[cluster_idx]
        cluster_emission = emission_probs[cluster_idx]
        cluster_state_names = state_names[cluster_idx]
        n_states_k = len(cluster_initial)
        
        # Simulate sequence from this cluster's HMM
        seq_states = []
        seq_obs = []
        
        # Sample initial state
        initial_state_idx = rng.choice(n_states_k, p=cluster_initial)
        seq_states.append(cluster_state_names[initial_state_idx])
        
        # Sample initial observation
        obs_idx = rng.choice(n_symbols, p=cluster_emission[initial_state_idx, :])
        seq_obs.append(alphabet[obs_idx])
        
        # Simulate remaining time points
        current_state = initial_state_idx
        for t in range(1, sequence_length):
            # Sample next state
            current_state = rng.choice(n_states_k, p=cluster_transition[current_state, :])
            seq_states.append(cluster_state_names[current_state])
            
            # Sample observation
            obs_idx = rng.choice(n_symbols, p=cluster_emission[current_state, :])
            seq_obs.append(alphabet[obs_idx])
        
        states.append(seq_states)
        observations.append(seq_obs)
    
    # Create DataFrame format
    obs_dict = {}
    for t in range(sequence_length):
        obs_dict[f'time_{t+1}'] = [obs[t] for obs in observations]
    
    observations_df = pd.DataFrame(obs_dict)
    observations_df['cluster'] = clusters
    
    return {
        'observations': observations,
        'states': states,
        'clusters': clusters,
        'observations_df': observations_df,
        'alphabet': alphabet,
        'state_names': state_names,
        'cluster_names': cluster_names
    }


def simulate_nhmm(
    n_states: int,
    emission_formula: Union[str, None],
    data: pd.DataFrame,
    id_var: str,
    time_var: str,
    initial_formula: Union[str, None] = None,
    transition_formula: Union[str, None] = None,
    coefs: Optional[Dict[str, np.ndarray]] = None,
    init_sd: Optional[float] = None,
    random_state: Optional[int] = None
) -> dict:
    """
    Simulate sequences from a Non-homogeneous Hidden Markov Model.
    
    This function generates sequences of observed and hidden states given the parameters
    of a non-homogeneous hidden Markov model. In an NHMM, transition and emission
    probabilities can vary over time or with covariates.
    
    It is similar to seqHMM's simulate_nhmm() function in R.
    
    Args:
        n_states: Number of hidden states (must be > 1)
        emission_formula: Formula string for emission probabilities (e.g., "~ x1 + x2").
                         The left-hand side should specify the response variable(s).
                         For multiple responses, use a list of formulas.
        data: DataFrame containing the variables used in model formulas.
              Must include the response variable(s) to define the number of observed
              symbols and sequence lengths. The actual values of response variables
              will be replaced by simulated values.
        id_var: Name of the ID variable in data identifying different sequences
        time_var: Name of the time index variable in data
        initial_formula: Optional formula string for initial state probabilities.
                        Default is "~ 1" (intercept only).
        transition_formula: Optional formula string for transition probabilities.
                           Default is "~ 1" (intercept only).
        coefs: Optional dictionary with keys 'initial_probs', 'transition_probs', 'emission_probs'
               containing coefficient matrices (etas). If None, coefficients are generated randomly.
        init_sd: Standard deviation for random coefficient generation.
                Default is 2.0 when coefs is None, 0.0 otherwise.
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Dictionary with keys:
            - 'observations': List of observed sequences (as lists)
            - 'states': List of hidden state sequences (as lists)
            - 'data': DataFrame with simulated response variables
            - 'model': Dictionary containing model information (coefficients, etc.)
            
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sequenzo.seqhmm import simulate_nhmm
        >>> 
        >>> # Create data with covariates and response variable
        >>> n_sequences = 10
        >>> sequence_length = 20
        >>> data = pd.DataFrame({
        ...     'id': np.repeat(range(1, n_sequences + 1), sequence_length),
        ...     'time': np.tile(range(1, sequence_length + 1), n_sequences),
        ...     'age': np.repeat(np.random.randint(20, 60, n_sequences), sequence_length),
        ...     'response': np.repeat(['A', 'B', 'C'], n_sequences * sequence_length // 3 + 1)[:n_sequences * sequence_length]
        ... })
        >>> 
        >>> # Simulate NHMM with time-varying probabilities
        >>> sim = simulate_nhmm(
        ...     n_states=3,
        ...     emission_formula="response ~ age",
        ...     data=data,
        ...     id_var='id',
        ...     time_var='time',
        ...     initial_formula="~ age",
        ...     transition_formula="~ age",
        ...     random_state=42
        ... )
        >>> 
        >>> print(f"Simulated {len(sim['observations'])} sequences")
        >>> print(f"First sequence: {sim['observations'][0]}")
    """
    rng = np.random.RandomState(random_state)
    
    # Step 1: Validate inputs
    if n_states < 2:
        raise ValueError(f"n_states must be at least 2, got {n_states}")
    
    if emission_formula is None:
        raise ValueError("emission_formula is required")
    
    if id_var not in data.columns:
        raise ValueError(f"id_var '{id_var}' not found in data columns: {list(data.columns)}")
    if time_var not in data.columns:
        raise ValueError(f"time_var '{time_var}' not found in data columns: {list(data.columns)}")
    
    # Step 2: Parse emission formula to get response variable(s)
    # For simplicity, we'll extract the response from the left-hand side
    # Format: "response ~ covariates" or "~ covariates" (response inferred from data)
    if isinstance(emission_formula, str):
        if '~' in emission_formula:
            parts = emission_formula.split('~')
            response_part = parts[0].strip()
            if response_part:
                # Response variable specified
                response_vars = [v.strip() for v in response_part.split('+')]
            else:
                # No response specified, try to infer from data
                # Look for categorical/object columns that might be responses
                response_vars = None
        else:
            raise ValueError("emission_formula must contain '~' separator")
    else:
        raise ValueError("emission_formula must be a string")
    
    # For now, we'll use a simplified approach: assume response is in data
    # and extract unique values to determine alphabet
    if response_vars is None:
        # Try to find a response column (categorical/object type)
        cat_cols = [col for col in data.columns 
                   if col not in [id_var, time_var] and 
                   (pd.api.types.is_categorical_dtype(data[col]) or 
                    pd.api.types.is_object_dtype(data[col]))]
        if cat_cols:
            response_vars = [cat_cols[0]]
        else:
            raise ValueError("Could not determine response variable. Please specify in emission_formula (e.g., 'response ~ x1')")
    
    # Get alphabet from response variable
    response_var = response_vars[0]
    if response_var not in data.columns:
        raise ValueError(f"Response variable '{response_var}' not found in data columns")
    
    # Extract unique values to form alphabet
    alphabet = sorted(data[response_var].dropna().unique().tolist())
    n_symbols = len(alphabet)
    
    # Step 3: Get sequence information from data
    # Group by id to get sequence lengths
    sequence_info = data.groupby(id_var).agg({
        time_var: ['min', 'max', 'count']
    }).reset_index()
    sequence_info.columns = [id_var, 'time_min', 'time_max', 'length']
    
    n_sequences = len(sequence_info)
    sequence_lengths = sequence_info['length'].values
    max_length = int(sequence_lengths.max())
    
    # Step 4: Create model matrices from formulas
    # Extract formula parts (right-hand side after ~)
    if isinstance(emission_formula, str):
        emission_rhs = emission_formula.split('~')[1].strip()
    else:
        emission_rhs = "1"
    
    if initial_formula is None:
        initial_formula = "~ 1"
    if isinstance(initial_formula, str):
        initial_rhs = initial_formula.split('~')[1].strip() if '~' in initial_formula else "1"
    else:
        initial_rhs = "1"
    
    if transition_formula is None:
        transition_formula = "~ 1"
    if isinstance(transition_formula, str):
        transition_rhs = transition_formula.split('~')[1].strip() if '~' in transition_formula else "1"
    else:
        transition_rhs = "1"
    
    # Create model matrices using the formulas module
    # For NHMM, we need time-varying covariates, so we create 3D matrices
    from .formulas import create_model_matrix
    
    # Create X matrices for initial, transition, and emission
    # Note: For initial probabilities, we only need the first time point
    # For transition and emission, we need all time points
    
    # Get unique sequence IDs
    unique_ids = sequence_info[id_var].values
    
    # Create a temporary SequenceData-like structure for create_model_matrix
    # We'll create model matrices directly from the data
    X_pi = _create_model_matrix_from_data(
        initial_rhs, data, id_var, time_var, n_sequences, max_length, rng
    )
    X_A = _create_model_matrix_from_data(
        transition_rhs, data, id_var, time_var, n_sequences, max_length, rng
    )
    X_B = _create_model_matrix_from_data(
        emission_rhs, data, id_var, time_var, n_sequences, max_length, rng
    )
    
    # Step 5: Generate or use provided coefficients
    n_covariates_pi = X_pi.shape[2]
    n_covariates_A = X_A.shape[2]
    n_covariates_B = X_B.shape[2]
    
    # Set default init_sd
    if init_sd is None:
        init_sd = 2.0 if coefs is None else 0.0
    
    if coefs is None:
        # Generate random coefficients
        eta_pi = rng.randn(n_covariates_pi, n_states) * init_sd
        eta_A = rng.randn(n_covariates_A, n_states, n_states) * init_sd
        eta_B = rng.randn(n_covariates_B, n_states, n_symbols) * init_sd
    else:
        # Use provided coefficients
        eta_pi = coefs.get('initial_probs')
        eta_A = coefs.get('transition_probs')
        eta_B = coefs.get('emission_probs')
        
        if eta_pi is None:
            eta_pi = rng.randn(n_covariates_pi, n_states) * init_sd
        if eta_A is None:
            eta_A = rng.randn(n_covariates_A, n_states, n_states) * init_sd
        if eta_B is None:
            eta_B = rng.randn(n_covariates_B, n_states, n_symbols) * init_sd
    
    # Step 6: Compute probabilities from coefficients using softmax
    # Import utility functions
    from .nhmm_utils import (
        compute_initial_probs_with_covariates,
        compute_transition_probs_with_covariates,
        compute_emission_probs_with_covariates
    )
    
    # Compute initial probabilities (one per sequence, using first time point)
    X_pi_first = X_pi[:, 0:1, :]  # Shape: (n_sequences, 1, n_covariates)
    initial_probs = compute_initial_probs_with_covariates(eta_pi, X_pi_first, n_states)
    # Result shape: (n_sequences, n_states)
    
    # Compute transition probabilities (for each sequence and time point)
    transition_probs = compute_transition_probs_with_covariates(eta_A, X_A, n_states)
    # Result shape: (n_sequences, n_timepoints, n_states, n_states)
    
    # Compute emission probabilities (for each sequence and time point)
    emission_probs = compute_emission_probs_with_covariates(eta_B, X_B, n_states, n_symbols)
    # Result shape: (n_sequences, n_timepoints, n_states, n_symbols)
    
    # Step 7: Simulate sequences
    state_names = [f"State {i+1}" for i in range(n_states)]
    observations = []
    states = []
    
    for seq_idx in range(n_sequences):
        seq_length = int(sequence_lengths[seq_idx])
        seq_states = []
        seq_obs = []
        
        # Sample initial state using initial probabilities for this sequence
        initial_state_idx = rng.choice(n_states, p=initial_probs[seq_idx, :])
        seq_states.append(state_names[initial_state_idx])
        
        # Sample initial observation using emission probabilities
        # Use first time point (t=0) for initial emission
        t = 0
        if t < seq_length:
            emission_probs_t = emission_probs[seq_idx, t, initial_state_idx, :]
            obs_idx = rng.choice(n_symbols, p=emission_probs_t)
            seq_obs.append(alphabet[obs_idx])
        
        # Simulate remaining time points
        current_state = initial_state_idx
        for t in range(1, seq_length):
            # Sample next state using transition probabilities
            # transition_probs[seq_idx, t-1, current_state, :] gives probabilities
            # for transitions from current_state at time t-1
            transition_probs_t = transition_probs[seq_idx, t-1, current_state, :]
            current_state = rng.choice(n_states, p=transition_probs_t)
            seq_states.append(state_names[current_state])
            
            # Sample observation using emission probabilities
            emission_probs_t = emission_probs[seq_idx, t, current_state, :]
            obs_idx = rng.choice(n_symbols, p=emission_probs_t)
            seq_obs.append(alphabet[obs_idx])
        
        states.append(seq_states)
        observations.append(seq_obs)
    
    # Step 8: Update data with simulated observations
    data_sim = data.copy()
    
    # Replace response variable values with simulated values
    # Create a mapping from (id, time) to observation
    obs_dict = {}
    for seq_idx, seq_id in enumerate(unique_ids):
        seq_obs = observations[seq_idx]
        seq_times = data[data[id_var] == seq_id][time_var].values
        for t_idx, time_val in enumerate(seq_times):
            if t_idx < len(seq_obs):
                obs_dict[(seq_id, time_val)] = seq_obs[t_idx]
    
    # Update data
    def get_obs(row):
        key = (row[id_var], row[time_var])
        return obs_dict.get(key, data.loc[row.name, response_var])
    
    data_sim[response_var] = data_sim.apply(get_obs, axis=1)
    
    # Step 9: Create states DataFrame
    states_list = []
    for seq_idx, seq_id in enumerate(unique_ids):
        seq_states = states[seq_idx]
        seq_times = data[data[id_var] == seq_id][time_var].values
        for t_idx, time_val in enumerate(seq_times):
            if t_idx < len(seq_states):
                states_list.append({
                    id_var: seq_id,
                    time_var: time_val,
                    'state': seq_states[t_idx]
                })
    
    states_df = pd.DataFrame(states_list)
    
    # Step 10: Return results
    return {
        'observations': observations,
        'states': states,
        'data': data_sim,
        'states_df': states_df,
        'model': {
            'n_states': n_states,
            'n_symbols': n_symbols,
            'alphabet': alphabet,
            'state_names': state_names,
            'eta_pi': eta_pi,
            'eta_A': eta_A,
            'eta_B': eta_B,
            'n_covariates_pi': n_covariates_pi,
            'n_covariates_A': n_covariates_A,
            'n_covariates_B': n_covariates_B
        }
    }


def _create_model_matrix_from_data(
    formula_rhs: str,
    data: pd.DataFrame,
    id_var: str,
    time_var: str,
    n_sequences: int,
    max_length: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Helper function to create model matrix from formula and data.
    
    This function creates a 3D covariate matrix of shape (n_sequences, n_timepoints, n_covariates)
    from a formula string and data DataFrame.
    
    Args:
        formula_rhs: Right-hand side of formula (e.g., "x1 + x2" or "1")
        data: DataFrame containing covariates
        id_var: Column name for sequence IDs
        time_var: Column name for time variable
        n_sequences: Number of sequences
        max_length: Maximum sequence length
        rng: Random number generator
        
    Returns:
        numpy array: Model matrix of shape (n_sequences, max_length, n_covariates)
    """
    # Parse formula terms
    if not formula_rhs or formula_rhs.strip() == "1":
        # Intercept only: return matrix of ones
        return np.ones((n_sequences, max_length, 1))
    
    # Split by + to get terms
    terms = [term.strip() for term in formula_rhs.split('+')]
    terms = [t for t in terms if t and t != '1']  # Remove empty and intercept (handled separately)
    
    # Always include intercept
    n_covariates = len(terms) + 1
    
    # Initialize matrix
    X = np.zeros((n_sequences, max_length, n_covariates))
    
    # First column is intercept (all ones)
    X[:, :, 0] = 1.0
    
    # Get unique sequence IDs
    unique_ids = sorted(data[id_var].unique())
    
    # Fill in covariates
    for term_idx, term in enumerate(terms):
        col_idx = term_idx + 1  # +1 because first column is intercept
        
        if term not in data.columns:
            raise ValueError(
                f"Variable '{term}' not found in data columns: {list(data.columns)}"
            )
        
        # For each sequence, extract covariate values
        for seq_idx, seq_id in enumerate(unique_ids):
            if seq_idx >= n_sequences:
                break
            
            # Get data for this sequence
            seq_data = data[data[id_var] == seq_id].sort_values(time_var)
            
            # Extract covariate values
            covar_values = seq_data[term].values
            
            # Fill matrix (pad with last value if sequence is shorter than max_length)
            seq_length = len(covar_values)
            for t in range(max_length):
                if t < seq_length:
                    X[seq_idx, t, col_idx] = covar_values[t]
                else:
                    # Pad with last value if sequence is shorter
                    X[seq_idx, t, col_idx] = covar_values[-1] if seq_length > 0 else 0.0
    
    return X
