"""
@Author  : Yuqi Liang 梁彧祺
@File    : build_nhmm.py
@Time    : 2025-11-22 19:30
@Desc    : Build Non-homogeneous HMM models

This module provides the build_nhmm function, which creates Non-homogeneous HMM
model objects similar to seqHMM's build_nhmm() function in R.

Formula inputs are expanded into separate initial, transition, and emission
design matrices, following seqHMM's family-level formula structure.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union
from sequenzo.define_sequence_data import SequenceData
from .nhmm import NHMM
from .formulas import Formula, create_model_matrix


def build_nhmm(
    observations: SequenceData,
    n_states: int,
    X: Optional[np.ndarray] = None,
    X_pi: Optional[np.ndarray] = None,
    X_A: Optional[np.ndarray] = None,
    X_B: Optional[np.ndarray] = None,
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
    n_sequences = len(observations.sequences)
    n_timepoints = max(len(seq) for seq in observations.sequences)

    def _formula_matrix(formula):
        return create_model_matrix(
            formula,
            data,
            id_var,
            time_var,
            n_sequences,
            n_timepoints,
            id_values=observations.ids,
            time_values=observations.time,
        )

    has_formula = any(
        formula is not None
        for formula in (emission_formula, initial_formula, transition_formula)
    )

    if has_formula:
        if data is None or id_var is None or time_var is None:
            raise ValueError(
                "Formula-based NHMM construction requires data, id_var, and time_var."
        )
        _validate_formula_scope(initial_formula, "initial_formula")
        _validate_formula_scope(transition_formula, "transition_formula")
        _validate_emission_formula_scope(emission_formula, data, observations, id_var, time_var)
        if X_pi is None:
            X_pi = _formula_matrix(initial_formula) if initial_formula is not None else (
                X if X is not None else _formula_matrix("~ 1")
            )
        if X_A is None:
            X_A = _formula_matrix(transition_formula) if transition_formula is not None else (
                X if X is not None else _formula_matrix("~ 1")
            )
        if X_B is None:
            X_B = _formula_matrix(emission_formula) if emission_formula is not None else (
                X if X is not None else _formula_matrix("~ 1")
            )
        if X is None:
            X = X_B
    elif X is None and any(matrix is not None for matrix in (X_pi, X_A, X_B)):
        if any(matrix is None for matrix in (X_pi, X_A, X_B)):
            raise ValueError(
                "When X is omitted, X_pi, X_A, and X_B must all be provided together."
            )
        X = next(matrix for matrix in (X_B, X_A, X_pi) if matrix is not None)
    elif X is None:
        raise ValueError(
            "Must provide either X or at least one formula "
            "(emission_formula, initial_formula, or transition_formula)."
        )
    
    # Create and return NHMM object
    nhmm = NHMM(
        observations=observations,
        n_states=n_states,
        X=X,
        X_pi=X_pi,
        X_A=X_A,
        X_B=X_B,
        eta_pi=eta_pi,
        eta_A=eta_A,
        eta_B=eta_B,
        state_names=state_names,
        random_state=random_state
    )
    
    return nhmm


def _validate_formula_scope(formula, name: str) -> None:
    if formula is None:
        return
    text = formula.raw_formula if isinstance(formula, Formula) else str(formula)
    stripped = text.strip()
    if "~" in stripped and not stripped.startswith("~"):
        lhs = formula.lhs if isinstance(formula, Formula) else stripped.split("~", 1)[0].strip()
        if lhs:
            raise ValueError(f"{name} must not contain a left-hand side")
    if name == "initial_formula" and "lag(" in stripped.replace(" ", "").lower():
        raise ValueError("initial_formula must not contain lag terms")


def _validate_emission_formula_scope(
    formula,
    data: pd.DataFrame,
    observations: SequenceData,
    id_var: str,
    time_var: str,
) -> None:
    if formula is None:
        return
    lhs = formula.lhs if isinstance(formula, Formula) else None
    if lhs is None and not isinstance(formula, Formula):
        stripped = str(formula).strip()
        if "~" in stripped and not stripped.startswith("~"):
            lhs = stripped.split("~", 1)[0].strip()
    if lhs and lhs not in data.columns:
        raise ValueError("emission_formula left-hand side must be a column in data")
    if not lhs:
        return

    expected = _sequence_observation_frame(observations, id_var, time_var, lhs)
    try:
        merged = expected.merge(
            data[[id_var, time_var, lhs]],
            on=[id_var, time_var],
            how="left",
            suffixes=("_expected", "_actual"),
            validate="one_to_one",
        )
    except pd.errors.MergeError as exc:
        raise ValueError("emission_formula left-hand side must match observations") from exc

    actual_col = f"{lhs}_actual"
    expected_col = f"{lhs}_expected"
    for expected_value, actual_value in zip(merged[expected_col], merged[actual_col]):
        if pd.isna(expected_value) and pd.isna(actual_value):
            continue
        if expected_value != actual_value:
            raise ValueError("emission_formula left-hand side must match observations")


def _sequence_observation_frame(
    observations: SequenceData,
    id_var: str,
    time_var: str,
    response_var: str,
) -> pd.DataFrame:
    rows = []
    ids = list(observations.ids)
    times = list(observations.time)
    alphabet = list(observations.alphabet)
    for seq_idx, sequence in enumerate(observations.sequences):
        for time_idx, state in enumerate(sequence):
            value = state
            if isinstance(state, (int, np.integer)):
                value = alphabet[state - 1] if 1 <= state <= len(alphabet) else np.nan
            rows.append(
                {
                    id_var: ids[seq_idx],
                    time_var: times[time_idx],
                    response_var: value,
                }
            )
    return pd.DataFrame(rows)
