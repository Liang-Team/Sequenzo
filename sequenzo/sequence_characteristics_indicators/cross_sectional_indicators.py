"""
@Author  : Yuqi Liang 梁彧祺
@File    : cross_sectional_indicators.py
@Time    : 04/02/2026 19:35
@Desc    : Cross-sectional sequence characteristics

    This module implements cross-sectional characteristics including:
    - Mean time spent in each state (seqmeant)
    - Modal state sequence (seqmodst)

    Reference: TraMineR R package
    - seqmeant: R/seqmeant.R
    - seqmodst: R/seqmodst.R
    https://github.com/cran/TraMineR/blob/master/R/seqmeant.R
    https://github.com/cran/TraMineR/blob/master/R/seqmodst.R
"""

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from .state_frequencies_and_entropy_per_sequence import get_state_freq_and_entropy_per_seq


def get_mean_time_in_states(seqdata: SequenceData, weighted: bool = True, 
                           with_missing: bool = False, prop: bool = False,
                           serr: bool = False) -> pd.DataFrame:
    """
    Calculate mean time spent in each state across all sequences.
    
    This function computes the average duration spent in each state,
    optionally weighted by sequence weights.
    
    Args:
        seqdata: SequenceData object containing sequence data
        weighted: If True, use sequence weights. If False, treat all sequences equally.
        with_missing: If True, treat missing values as regular states.
        prop: If True, return proportions instead of absolute times.
        serr: If True, include standard error in output.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
                     - 'Mean': Mean time spent in each state
                     - 'Var', 'Stdev', 'SE' (if serr=True): Variance, standard deviation, standard error
                     Rows correspond to states.
    
    Reference:
        TraMineR: R/seqmeant.R
        https://github.com/cran/TraMineR/blob/master/R/seqmeant.R
    
    Examples:
        >>> mean_times = get_mean_time_in_states(seqdata, weighted=True)
        >>> print(mean_times)
                 Mean
        State1   2.5
        State2   3.2
        State3   1.8
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    # Get individual state distributions
    istatd = get_state_freq_and_entropy_per_seq(seqdata, prop=prop)
    
    # Get weights
    weights = seqdata.weights if hasattr(seqdata, 'weights') and seqdata.weights is not None else None
    
    if not weighted or weights is None:
        weights = np.ones(seqdata.seqdata.shape[0])
    
    # Check if all weights are 1 (effectively unweighted)
    if np.all(weights == 1):
        weighted = False
    
    wtot = np.sum(weights)
    
    # Exclude ID column
    state_cols = [col for col in istatd.columns if col != 'ID']
    istatd_values = istatd[state_cols].values
    
    # Calculate weighted mean for each state
    mtime = np.sum(istatd_values * weights[:, np.newaxis], axis=0) / wtot
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Mean': mtime
    }, index=state_cols)
    
    # Calculate standard error if requested
    if serr:
        w2tot = np.sum(weights ** 2)
        # Center the values (subtract mean)
        vcent = istatd_values - mtime
        var = np.sum(weights[:, np.newaxis] * (vcent ** 2), axis=0) * wtot / (wtot ** 2 - w2tot)
        sd = np.sqrt(var)
        SE = np.sqrt(var / wtot)
        
        result['Var'] = var
        result['Stdev'] = sd
        result['SE'] = SE
    
    return result


def get_modal_state_sequence(seqdata: SequenceData, weighted: bool = True,
                            with_missing: bool = False) -> pd.DataFrame:
    """
    Calculate the modal state sequence.
    
    The modal state sequence is constructed by selecting, at each position,
    the state that appears most frequently (optionally weighted) across
    all sequences.
    
    Args:
        seqdata: SequenceData object containing sequence data
        weighted: If True, use sequence weights. If False, treat all sequences equally.
        with_missing: If True, treat missing values as regular states.
    
    Returns:
        pd.DataFrame: DataFrame with one row representing the modal sequence.
                     Columns correspond to time positions, values are state indices (1-indexed).
                     Also includes 'Occurrences' attribute showing how many sequences match.
    
    Reference:
        TraMineR: R/seqmodst.R
        https://github.com/cran/TraMineR/blob/master/R/seqmodst.R
    
    Examples:
        >>> modal_seq = get_modal_state_sequence(seqdata, weighted=True)
        >>> print(modal_seq)
            T1  T2  T3  T4
        0   1   2   2   3
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    slength = seqdata.seqdata.shape[1]
    
    # Get state frequencies at each position
    # Use seqstatd equivalent - we'll compute it manually
    seq_matrix = seqdata.seqdata.values
    
    # Get weights
    weights = seqdata.weights if hasattr(seqdata, 'weights') and seqdata.weights is not None else None
    
    if not weighted or weights is None:
        weights = np.ones(seqdata.seqdata.shape[0])
    
    if np.all(weights == 1):
        weighted = False
    
    # Get alphabet
    states = seqdata.states.copy()
    if with_missing and hasattr(seqdata, 'ismissing') and seqdata.ismissing:
        # Missing state would be len(states) + 1
        pass
    
    # Calculate state frequencies at each position
    nstates = len(states)
    freq_matrix = np.zeros((nstates, slength))
    
    for t in range(slength):
        for s_idx, state_val in enumerate(range(1, nstates + 1)):
            # Count occurrences of this state at position t
            mask = (seq_matrix[:, t] == state_val)
            if not with_missing:
                mask = mask & ~pd.isna(seq_matrix[:, t])
            freq_matrix[s_idx, t] = np.sum(weights[mask])
    
    # Construct modal sequence: state with highest frequency at each position
    modal_seq = np.zeros(slength, dtype=int)
    stfreq = np.zeros(slength)
    
    for t in range(slength):
        smax_idx = np.argmax(freq_matrix[:, t])
        modal_seq[t] = smax_idx + 1  # Convert to 1-indexed
        stfreq[t] = freq_matrix[smax_idx, t]
    
    # Create result DataFrame
    time_cols = list(seqdata.seqdata.columns)
    result = pd.DataFrame([modal_seq], columns=time_cols)
    
    # Count occurrences: how many sequences match the modal sequence
    nbocc = 0
    for i in range(seqdata.seqdata.shape[0]):
        if np.array_equal(seq_matrix[i, :], modal_seq):
            nbocc += 1
    
    # Store occurrences as attribute (similar to TraMineR)
    result.attrs['Occurrences'] = nbocc
    result.attrs['Frequencies'] = stfreq
    result.attrs['nbseq'] = np.sum(weights)
    result.attrs['weighted'] = weighted
    
    return result
