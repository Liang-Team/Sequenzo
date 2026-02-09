"""
@Author  : Yuqi Liang 梁彧祺
@File    : integration_index.py
@Time    : 03/02/2026 11:08
@Desc    : Integration index for sequences

    This module implements integration index calculation, which measures
    the weighted proportion of time spent in a specific state, with weights
    increasing with position (later positions have higher weight).

    Reference: TraMineR R package
    - seqintegr: R/seqintegr.R
    https://github.com/cran/TraMineR/blob/master/R/seqintegr.R
"""

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData


def get_integration_index(seqdata: SequenceData, state: int = None, pow: float = 1.0, with_missing: bool = False) -> pd.DataFrame:
    """
    Calculate integration index for sequences.
    
    The integration index measures the weighted proportion of time spent
    in a specific state (or all states), where weights increase with position.
    Later positions in the sequence have higher weights, making them more
    important for integration.
    
    Formula: sum(integVector * (x == state)) / sum(integVector)
    where integVector = (1:ncol(seqdata))^pow
    
    Args:
        seqdata: SequenceData object containing sequence data
        state: Specific state to calculate integration for (1-indexed).
               If None, calculates integration for all states.
        pow: Power parameter for position weighting. Higher values give
             more weight to later positions. Default is 1.0 (linear weighting).
        with_missing: If True, include missing values in calculation.
                     If False, exclude positions with missing values.
    
    Returns:
        pd.DataFrame: DataFrame with columns for each state (or single column
                     if state is specified) containing integration values.
                     Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqintegr.R
        https://github.com/cran/TraMineR/blob/master/R/seqintegr.R
    
    Examples:
        >>> # Integration for all states
        >>> integr_all = get_integration_index(seqdata, state=None, pow=1.0)
        >>> print(integr_all.head())
                 State1    State2    State3
        ID_1      0.35      0.40      0.25
        
        >>> # Integration for specific state
        >>> integr_state = get_integration_index(seqdata, state=1, pow=1.0)
        >>> print(integr_state.head())
                 State1
        ID_1      0.35
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    # Get alphabet
    alph = seqdata.states.copy()
    if with_missing and hasattr(seqdata, 'ismissing') and seqdata.ismissing:
        # Missing state would be len(alph) + 1
        pass  # Handle missing if needed
    
    # Check state validity
    if state is not None:
        if isinstance(state, (list, np.ndarray)) and len(state) > 1:
            raise ValueError("[!] When non null, 'state' must be a single state")
        if isinstance(state, (list, np.ndarray)):
            state = state[0]
        if state < 1 or state > len(alph):
            raise ValueError(f"[!] state {state} not in the alphabet!")
    
    nbstat = 1 if state is not None else len(alph)
    nbseq = seqdata.seqdata.shape[0]
    
    # Create integration vector: (1, 2, 3, ..., ncol)^pow
    ncol = seqdata.seqdata.shape[1]
    integVector = np.power(np.arange(1, ncol + 1), pow)
    
    # Handle missing values
    # In SequenceData, missing values are typically NaN
    seq_matrix = seqdata.seqdata.values
    
    # Initialize result matrix
    if state is not None:
        colnames = [f'State{state}']
    else:
        colnames = [f'State{i+1}' for i in range(len(alph))]
    
    iseqtab = pd.DataFrame(np.zeros((nbseq, nbstat)), 
                           index=seqdata.seqdata.index,
                           columns=colnames)
    
    # Calculate integration for each sequence
    for i in range(nbseq):
        seq_row = seq_matrix[i, :]
        
        # Calculate suminteg for this sequence (accounting for missing)
        if not with_missing:
            # Exclude missing positions from suminteg
            missing_mask = pd.isna(seq_row)
            missing_positions = np.where(missing_mask)[0]
            if len(missing_positions) > 0:
                # Subtract weights of missing positions
                missing_weights = integVector[missing_positions]
                seq_suminteg = np.sum(integVector) - np.sum(missing_weights)
            else:
                seq_suminteg = np.sum(integVector)
        else:
            seq_suminteg = np.sum(integVector)
        
        if state is not None:
            # Single state calculation
            state_mask = (seq_row == state)
            if not with_missing:
                state_mask = state_mask & ~pd.isna(seq_row)
            # Get positions where state matches
            state_positions = np.where(state_mask)[0]
            if len(state_positions) > 0:
                weighted_sum = np.sum(integVector[state_positions])
            else:
                weighted_sum = 0.0
            iseqtab.iloc[i, 0] = weighted_sum / seq_suminteg if seq_suminteg > 0 else 0.0
        else:
            # All states calculation
            for j in range(len(alph)):
                state_val = j + 1  # States are 1-indexed
                state_mask = (seq_row == state_val)
                if not with_missing:
                    state_mask = state_mask & ~pd.isna(seq_row)
                # Get positions where state matches
                state_positions = np.where(state_mask)[0]
                if len(state_positions) > 0:
                    weighted_sum = np.sum(integVector[state_positions])
                else:
                    weighted_sum = 0.0
                iseqtab.iloc[i, j] = weighted_sum / seq_suminteg if seq_suminteg > 0 else 0.0
    
    # Reset index to include ID column
    iseqtab = iseqtab.reset_index().rename(columns={'index': 'ID'})
    
    return iseqtab
