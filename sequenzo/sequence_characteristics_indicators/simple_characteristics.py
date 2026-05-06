"""
@Author  : 梁彧祺
@File    : simple_characteristics.py
@Time    : 22/09/2025 22:40
@Desc    : Simple sequence characteristics functions
"""

import numpy as np
import pandas as pd
from typing import Union, List

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from sequenzo.dissimilarity_measures.utils.seqlength import seqlength
from sequenzo.dissimilarity_measures.utils.get_sm_trate_substitution_cost_matrix import get_sm_trate_substitution_cost_matrix


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
    l = np.zeros(nbstat, dtype=int) - 1  # 必须是 -1（或其他负数）。避免 0-based 索引与 0 代表的无效值冲突
    
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
    N = np.zeros(slength + 1, dtype=object)  # Use object dtype to handle large integers
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
        if l[cidx] > -1:
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
                   If True, consecutive identical states are compressed (e.g., [1,1,2,2] -> [1,2])
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
    if isinstance(seqdata, np.ndarray):
        seqdata = pd.DataFrame(seqdata)

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

def cut_prefix(row, x=0):
    arr = row.to_numpy()
    if np.issubdtype(arr.dtype, np.number):
        pos_idx = np.where(arr < x)[0]
        if len(pos_idx) > 0:
            arr = arr[:pos_idx[0]]
    return arr

def seqsubsn(seqdata, DSS=True, with_missing=False) -> pd.DataFrame:
    if isinstance(seqdata, np.ndarray):
        sl = pd.unique(seqdata.ravel())
        seqdata = pd.DataFrame(seqdata)
        statelist = sl.tolist()
    elif isinstance(seqdata, pd.DataFrame):
        sl = pd.unique(seqdata.values.ravel())
        statelist = sl.tolist()
        pass
    elif isinstance(seqdata, SequenceData):
        sl = seqdata.states.copy()
        seqdata = seqdata.seqdata
        statelist = list(range(1, len(sl) + 1))
    else:
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")

    if DSS:
        seqdata = seqdss(seqdata)
        seqdata = pd.DataFrame(seqdata)

    ns = len(sl)

    result = seqdata.apply(lambda row: get_subsequences_in_single_sequence(
        cut_prefix(row),
        nbstat=ns,
        statlist=statelist
    ), axis=1)

    result = pd.DataFrame(result, columns=['Subseq.'], index=seqdata.index)
    return result

def get_number_of_transitions(seqdata, norm=False, pwight=False) -> pd.DataFrame:
    """
    Calculate how many state changes occur in each sequence.
    
    This function measures sequence instability by counting transitions (state changes).
    A transition happens whenever the sequence changes from one state to another.
    More transitions = more volatile/unstable sequences.
    
    Args:
        seqdata: SequenceData object or pandas DataFrame containing your sequence data
        norm:    If set as TRUE, the number of transitions is divided by its theoretical maximum, length of the sequence minus 1.
                 When the length of the sequence is 1, the normalized value is set as 0.
        pwight:  If set as TRUE, return count of transitions weighted
                 by their probability to not occur to give higher weights to rare transitions.
        
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
        >>> # 1->2 (position 3), 2->1 (position 5), 1->3 (position 6)
        
    Note:
        Missing values are automatically ignored. Only counts actual state changes
        between valid sequence elements. Use this to measure sequence volatility.
    """
    # Check if input is a SequenceData object
    if not hasattr(seqdata, 'seqdata'):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")

    dss = seqdss(seqdata)
    dss_length = seqlength(dss)
    number_seq = seqdata.seqdata.shape[0]

    if pwight:
        # 返回的是每个id序列在每个时间点下的各状态不发生概率的累加和
        tr = get_sm_trate_substitution_cost_matrix(seqdata)
        dss = dss + 1
        trans = np.zeros((number_seq, 1))

        for i in range(number_seq):
            if dss_length.iloc[i, 0] > 1:
                for j in range(1, dss_length.iloc[i, 0]):
                    state_from = dss.iloc[i, j-1]
                    state_to = dss.iloc[i, j]
                    trans[i, 0] += tr[state_from, state_to]

    else:
        # 返回的是每个id序列的转变次数，与上面的例子一致
        trans = dss_length - 1
        if any(dss_length==0):
            trans[dss_length==0] = 0

    if norm:
        seq_length = seqlength(seqdata)
        trans = trans / (seq_length-1)
        if any(seq_length<=1):
            trans[seq_length<=1] = 0

    trans = pd.DataFrame(trans, index=seqdata.seqdata.index, columns=['Transitions'])
    trans = trans.reset_index().rename(columns={'index': 'ID'})

    return trans
