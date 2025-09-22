"""
@Author  : 梁彧祺
@File    : simple_characteristics.py
@Time    : 22/09/2025 22:40
@Desc    : Simple sequence characteristics functions
"""

import numpy as np
import pandas as pd
from typing import Union, List


def get_subsequences_in_single_sequence(x: np.ndarray, nbstat: int, statlist: List, void=None, nr=None, with_missing: bool = False) -> int:
    """
    Internal helper function to count distinct subsequences in a single sequence.
    
    This is a low-level computational function that implements the dynamic programming 
    algorithm for counting subsequences. It's designed to be called by higher-level 
    functions like get_number_of_subsequences().
    
    Args:
        x (np.ndarray): Single sequence array (e.g., [1, 2, 1, 3])
        nbstat (int): Number of distinct states/symbols
        statlist (List): List of all possible states/symbols  
        void: Symbol representing void/empty elements (not used in current implementation)
        nr: Symbol representing missing values
        with_missing (bool): Whether to include missing values in the calculation
        
    Returns:
        int: Number of distinct subsequences in this one sequence
        
    Note:
        This is an internal function. Use get_number_of_subsequences() for analyzing
        sequence datasets. The algorithm uses dynamic programming for efficiency.
    """
    # Initialize state tracking array
    l = np.zeros(nbstat, dtype=int)
    
    # Remove void elements if specified
    if void is not None:
        x = x[x != void]
    
    # Remove missing values if not including them
    if not with_missing and nr is not None:
        x = x[x != nr]
    
    slength = len(x)
    
    # Empty sequence has one subsequence (the empty one)
    if slength == 0:
        return 1
    
    # Dynamic programming array
    N = np.zeros(slength + 1, dtype=int)
    N[0] = 1
    
    for i in range(1, slength + 1):
        N[i] = 2 * N[i-1]
        
        # Find the index of current state in statlist
        current_state = x[i-1]
        try:
            cidx = statlist.index(current_state)
        except ValueError:
            # If state not in statlist, skip this iteration
            continue
        
        # Subtract previously counted subsequences ending with this state
        if l[cidx] > 0:
            N[i] = N[i] - N[l[cidx]]
        
        # Update last position of this state
        l[cidx] = i - 1
    
    return N[slength]


def get_subsequences_all_sequences(seqdata, dss: bool = True, with_missing: bool = False) -> pd.DataFrame:
    """
    Calculate the number of distinct subsequences for all sequences in the dataset.
    
    This is the main function you'll use to analyze subsequence complexity across 
    multiple sequences. It processes your entire sequence dataset and returns a 
    summary table showing how many distinct subsequences exist in each sequence.
    
    Args:
        seqdata: SequenceData object or pandas DataFrame containing your sequence data
        dss (bool): Whether to apply distinct state sequence preprocessing. 
                   If True, consecutive identical states are compressed (e.g., [1,1,2,2] → [1,2])
        with_missing (bool): Whether to include missing values in the calculation
        
    Returns:
        pd.DataFrame: Results table with one column 'Subseq.' showing the subsequence 
                     count for each sequence. Row names match your sequence identifiers.
        
    Examples:
        >>> # Analyze subsequence complexity in your sequence dataset
        >>> result = get_number_of_subsequences(seq_data, dss=True, with_missing=False)
        >>> print(result.head())
                Subseq.
        seq_1        15
        seq_2        23
        seq_3         8
        
        >>> # Higher numbers = more complex sequences with more possible subsequences
        
    Note:
        This function works with SequenceData objects (recommended) or pandas DataFrames.
        Use this to understand the complexity and diversity patterns in your sequences.
    """
    # Check if input is a SequenceData object
    if hasattr(seqdata, 'seqdata'):
        # It is a SequenceData object
        sequences = seqdata.seqdata
        states = seqdata.states
        state_mapping = seqdata.state_mapping
        ids = sequences.index
        
        # Handle missing values
        nr_code = len(states) + 1 if hasattr(seqdata, 'ismissing') and seqdata.ismissing else None
        
    elif isinstance(seqdata, pd.DataFrame):
        # It's a DataFrame
        sequences = seqdata
        # Try to infer states from the data
        unique_vals = set()
        for col in sequences.columns:
            unique_vals.update(sequences[col].dropna().unique())
        states = sorted(list(unique_vals))
        state_mapping = {state: i+1 for i, state in enumerate(states)}
        ids = sequences.index
        nr_code = None
        
    else:
        raise ValueError("seqdata must be a SequenceData object or pandas DataFrame")
    
    # Apply DSS (Distinct State Sequences) if requested
    if dss:
        processed_sequences = sequences.copy()
        for idx in processed_sequences.index:
            row = processed_sequences.loc[idx].values
            # Remove consecutive duplicates
            if len(row) > 0:
                new_row = [row[0]]
                for i in range(1, len(row)):
                    if row[i] != row[i-1]:
                        new_row.append(row[i])
                # Pad with NaN if sequence got shorter
                while len(new_row) < len(row):
                    new_row.append(np.nan)
                processed_sequences.loc[idx] = new_row
    else:
        processed_sequences = sequences
    
    # Get state list
    if hasattr(seqdata, 'states'):
        # Use numeric codes from SequenceData
        statlist = list(range(1, len(states) + 1))
        if with_missing and nr_code is not None:
            statlist.append(nr_code)
    else:
        # Use original states  
        statlist = states
    
    nbstat = len(statlist)
    
    # Calculate subsequence count for each sequence
    results = []
    for idx in processed_sequences.index:
        seq_values = processed_sequences.loc[idx].values
        
        # Remove NaN values
        seq_values = seq_values[~pd.isna(seq_values)]
        
        if len(seq_values) == 0:
            result = 1  # Empty sequence has 1 subsequence
        else:
            result = get_subsequences_in_single_sequence(
                seq_values.astype(int), 
                nbstat, 
                statlist, 
                void=None, 
                nr=nr_code, 
                with_missing=with_missing
            )
        results.append(result)
    
    # Create result DataFrame
    result_df = pd.DataFrame(results, columns=['Subseq.'], index=ids)
    
    return result_df


def get_number_of_transitions(seqdata) -> pd.DataFrame:
    """
    Calculate how many state changes occur in each sequence.
    
    This function measures sequence instability by counting transitions (state changes).
    A transition happens whenever the sequence changes from one state to another.
    More transitions = more volatile/unstable sequences.
    
    Args:
        seqdata: SequenceData object or pandas DataFrame containing your sequence data
        
    Returns:
        pd.DataFrame: Results table with one column 'Transitions' showing the number of 
                     state changes for each sequence. Row names match your sequence identifiers.
        
    Examples:
        >>> # Count state changes in your sequences
        >>> result = get_number_of_transitions(seq_data)
        >>> print(result.head())
                Transitions
        seq_1            3
        seq_2            5  
        seq_3            2
        
        >>> # Example: sequence [1, 1, 2, 2, 1, 3] has 3 transitions:
        >>> # 1→2 (position 3), 2→1 (position 5), 1→3 (position 6)
        
    Note:
        Missing values are automatically ignored. Only counts actual state changes
        between valid sequence elements. Use this to measure sequence volatility.
    """
    # Check if input is a SequenceData object
    if hasattr(seqdata, 'seqdata'):
        sequences = seqdata.seqdata
        ids = sequences.index
    elif isinstance(seqdata, pd.DataFrame):
        sequences = seqdata
        ids = sequences.index
    else:
        raise ValueError("seqdata must be a SequenceData object or pandas DataFrame")
    
    results = []
    for idx in sequences.index:
        seq_values = sequences.loc[idx].values
        
        # Remove NaN values
        valid_values = seq_values[~pd.isna(seq_values)]
        
        if len(valid_values) <= 1:
            transitions = 0
        else:
            # Count transitions (state changes)
            transitions = 0
            for i in range(1, len(valid_values)):
                if valid_values[i] != valid_values[i-1]:
                    transitions += 1
        
        results.append(transitions)
    
    # Create result DataFrame
    result_df = pd.DataFrame(results, columns=['Transitions'], index=ids)
    
    return result_df