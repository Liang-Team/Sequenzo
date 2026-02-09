"""
@Author  : Yuqi Liang 梁彧祺
@File    : binary_indicators.py
@Time    : 05/02/2026 20:17
@Desc    : Binary indicators for positive/negative state analysis

    This module implements binary indicators for analyzing sequences with
    positive and negative states, including:
    - Proportion of positive states (ppos)
    - Normative volatility (nvolat)
    - Volatility of positive-negative sequences (vpos)
    - Integrative potential (integr)

    Reference: TraMineR R package
    - seqipos: R/seqipos.R
    - seqrecode: R/seqrecode.R (for binary recoding)
    https://github.com/cran/TraMineR/blob/master/R/seqipos.R
"""

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from .volatility import get_volatility
from .integration_index import get_integration_index


def _recode_to_binary(seqdata: SequenceData, pos_states: list = None, neg_states: list = None, 
                     otherwise=None, with_missing: bool = False) -> pd.DataFrame:
    """
    Internal helper function to recode sequences to binary (positive/negative).
    
    Recodes states into 'p' (positive) and 'n' (negative) based on provided
    state lists. States not in either list are recoded according to 'otherwise'.
    
    Args:
        seqdata: SequenceData object
        pos_states: List of state indices (1-indexed) to recode as 'p'
        neg_states: List of state indices (1-indexed) to recode as 'n'
        otherwise: Value for states not in pos_states or neg_states.
                  If None, uses void value (typically NaN).
        with_missing: Whether to include missing values
    
    Returns:
        pd.DataFrame: Recoded binary sequences with 'p', 'n', or otherwise values
    """
    if pos_states is None and neg_states is None:
        raise ValueError("[!] At least one of pos_states and neg_states should be non null!")
    
    # Get alphabet
    alph = seqdata.states.copy()
    alph_size = len(alph)
    
    # Determine positive and negative states
    if pos_states is None:
        # All states not in neg_states are positive
        pos_states = [i+1 for i in range(alph_size) if (i+1) not in neg_states]
    if neg_states is None:
        # All states not in pos_states are negative
        neg_states = [i+1 for i in range(alph_size) if (i+1) not in pos_states]
    
    # Check for duplicates
    if len(pos_states) != len(set(pos_states)):
        raise ValueError("[!] Multiple occurrences of same state in pos_states")
    if len(neg_states) != len(set(neg_states)):
        raise ValueError("[!] Multiple occurrences of same state in neg_states")
    
    # Check validity - convert state labels to indices if needed
    # Check if pos_states/neg_states contain state labels or indices
    pos_indices = []
    neg_indices = []
    
    for state in pos_states:
        if isinstance(state, str):
            # State label - find index
            if state in alph:
                pos_indices.append(alph.index(state) + 1)
            else:
                raise ValueError(f"[!] Invalid state '{state}' in pos_states")
        else:
            # State index
            if 1 <= state <= alph_size:
                pos_indices.append(state)
            else:
                raise ValueError(f"[!] Invalid state {state} in pos_states")
    
    for state in neg_states:
        if isinstance(state, str):
            # State label - find index
            if state in alph:
                neg_indices.append(alph.index(state) + 1)
            else:
                raise ValueError(f"[!] Invalid state '{state}' in neg_states")
        else:
            # State index
            if 1 <= state <= alph_size:
                neg_indices.append(state)
            else:
                raise ValueError(f"[!] Invalid state {state} in neg_states")
    
    pos_states = pos_indices
    neg_states = neg_indices
    
    # Recode sequences
    seq_matrix = seqdata.seqdata.values.copy()
    binary_matrix = seq_matrix.copy().astype(object)
    
    # Initialize with otherwise value (or NaN)
    if otherwise is not None:
        binary_matrix[:] = otherwise
    else:
        binary_matrix[:] = np.nan
    
    # Recode positive states to 'p'
    for state in pos_states:
        binary_matrix[seq_matrix == state] = 'p'
    
    # Recode negative states to 'n'
    for state in neg_states:
        binary_matrix[seq_matrix == state] = 'n'
    
    # Convert to DataFrame
    binary_df = pd.DataFrame(binary_matrix, 
                            index=seqdata.seqdata.index,
                            columns=seqdata.seqdata.columns)
    
    return binary_df


def get_positive_negative_indicators(seqdata: SequenceData, pos_states: list = None, 
                                    neg_states: list = None, dss: bool = None,
                                    index: str = "share", pow: float = 1.0, w: float = 0.5,
                                    with_missing: bool = False) -> pd.DataFrame:
    """
    Calculate binary indicators for positive/negative state sequences.
    
    This function computes various indicators based on binary recoding of sequences
    into positive ('p') and negative ('n') states.
    
    Available indices:
    - "share": Proportion of positive states (ppos or nvolat)
    - "integr": Integrative potential (weighted proportion of positive states)
    - "volatility": Volatility of positive-negative sequences
    
    Args:
        seqdata: SequenceData object containing sequence data
        pos_states: List of state indices (1-indexed) representing positive states.
                   If None and neg_states provided, all others are positive.
        neg_states: List of state indices (1-indexed) representing negative states.
                   If None and pos_states provided, all others are negative.
        dss: Whether to use distinct state sequence (DSS). 
             If None, defaults to True for "share", False otherwise.
        index: Type of indicator to calculate:
              - "share": Proportion of positive states
              - "integr": Integrative potential
              - "volatility": Volatility measure
        pow: Power parameter for integration index (used when index="integr")
        w: Weight parameter for volatility (used when index="volatility")
        with_missing: If True, treat missing values as regular states.
    
    Returns:
        pd.DataFrame: DataFrame with one column containing the indicator values.
                     Column name matches the index type. Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqipos.R
        https://github.com/cran/TraMineR/blob/master/R/seqipos.R
    
    Examples:
        >>> # Proportion of positive states
        >>> ppos = get_positive_negative_indicators(seqdata, 
        ...                                        pos_states=[1, 2],
        ...                                        neg_states=[3, 4],
        ...                                        index="share",
        ...                                        dss=False)
        >>> print(ppos.head())
                 share
        ID_1     0.65
        ID_2     0.72
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    if pos_states is None and neg_states is None:
        raise ValueError("[!] At least one of pos_states and neg_states should be non null!")
    
    valid_indices = ["share", "integr", "volatility"]
    if index not in valid_indices:
        raise ValueError(f"[!] index should be one of {valid_indices}")
    
    # Determine dss usage
    if dss is None:
        dss = (index == "share")
    
    # Apply DSS first if requested (before recoding)
    if dss:
        s_seqdata = seqdss(seqdata)
        # Create temporary SequenceData from DSS
        temp_seqdata = SequenceData(
            data=pd.DataFrame(s_seqdata, index=seqdata.seqdata.index, columns=seqdata.seqdata.columns),
            time=list(seqdata.seqdata.columns),
            states=seqdata.states,
            id_col=None
        )
    else:
        temp_seqdata = seqdata
    
    # Recode to binary
    binary_seq = _recode_to_binary(temp_seqdata, pos_states, neg_states, 
                                   otherwise=None, with_missing=with_missing)
    s = binary_seq
    
    # Calculate indicator based on index type
    if index == "share":
        # Proportion of positive states
        npos = (s == 'p').sum(axis=1)
        nneg = (s == 'n').sum(axis=1)
        ret = npos / (nneg + npos)
        # Handle division by zero
        ret = ret.replace([np.inf, -np.inf], np.nan)
        col_name = "share"
    
    elif index == "integr":
        # Integrative potential using integration index
        # Convert binary sequence to numeric for integration calculation
        binary_numeric = s.copy()
        binary_numeric[binary_numeric == 'p'] = 1
        binary_numeric[binary_numeric == 'n'] = 2
        binary_numeric[pd.isna(binary_numeric)] = np.nan
        
        # Create temporary SequenceData
        # States must match numeric values: 1='p', 2='n'
        temp_states = [1, 2]
        temp_labels = ['p', 'n']
        binary_df = binary_numeric.reset_index()
        temp_seqdata = SequenceData(
            data=binary_df,
            time=list(binary_numeric.columns),
            states=temp_states,
            labels=temp_labels,
            id_col='index' if 'index' in binary_df.columns else None
        )
        
        # Calculate integration for state 1 (positive)
        integr_result = get_integration_index(temp_seqdata, state=1, pow=pow, with_missing=with_missing)
        # Get the integration values (skip ID column)
        ret = integr_result.iloc[:, 1].values if integr_result.shape[1] > 1 else integr_result.iloc[:, 0].values
        col_name = "integr"
    
    elif index == "volatility":
        # Volatility of positive-negative sequences
        # Convert binary sequence to numeric for volatility calculation
        binary_numeric = s.copy()
        binary_numeric[binary_numeric == 'p'] = 1
        binary_numeric[binary_numeric == 'n'] = 2
        binary_numeric[pd.isna(binary_numeric)] = np.nan
        
        # Create temporary SequenceData
        # States must match numeric values: 1='p', 2='n'
        temp_states = [1, 2]
        temp_labels = ['p', 'n']
        binary_df = binary_numeric.reset_index()
        temp_seqdata = SequenceData(
            data=binary_df,
            time=list(binary_numeric.columns),
            states=temp_states,
            labels=temp_labels,
            id_col='index' if 'index' in binary_df.columns else None
        )
        
        # Calculate volatility
        volat_result = get_volatility(temp_seqdata, w=w, with_missing=with_missing)
        ret = volat_result['Volat'].values
        col_name = "volatility"
    
    # Create result DataFrame
    result = pd.DataFrame({
        col_name: ret
    }, index=seqdata.seqdata.index)
    
    result = result.reset_index().rename(columns={'index': 'ID'})
    
    return result
