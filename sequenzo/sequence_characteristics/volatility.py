"""
@Author  : Yuqi Liang 梁彧祺
@File    : volatility.py
@Time    : 02/02/2026 16:42
@Desc    : Objective volatility measure

    This module implements objective volatility calculation, which measures
    sequence volatility as a weighted combination of visited states and
    transition rates.

    Reference: TraMineR R package
    - seqivolatility: R/seqivolatility.R
    https://github.com/cran/TraMineR/blob/master/R/seqivolatility.R
"""

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from .simple_characteristics import get_number_of_transitions
from .state_frequencies_and_entropy_per_sequence import get_state_freq_and_entropy_per_seq


def get_volatility(seqdata: SequenceData, w: float = 0.5, with_missing: bool = False, adjust: bool = True) -> pd.DataFrame:
    """
    Calculate objective volatility for each sequence.
    
    Volatility measures how unstable a sequence is, combining:
    - The proportion of states visited (diversity component)
    - The transition rate (instability component)
    
    The formula is: w * (nvisit_adjusted) + (1-w) * transp
    where nvisit_adjusted accounts for the number of visited states
    relative to the alphabet size.
    
    Args:
        seqdata: SequenceData object containing sequence data
        w: Weight coefficient in range [0, 1]. 
           Higher w gives more weight to state diversity.
           Lower w gives more weight to transition rate.
        with_missing: If True, treat missing values as regular states.
                     If False, ignore missing values.
        adjust: If True, adjust visited states calculation:
                (nvisit - 1) / (alph.size - 1) when nvisit > 1, else 0
                If False, use: nvisit / alph.size
    
    Returns:
        pd.DataFrame: DataFrame with one column 'Volat' containing volatility
                     values. Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqivolatility.R
        https://github.com/cran/TraMineR/blob/master/R/seqivolatility.R
    
    Examples:
        >>> volatility = get_volatility(seqdata, w=0.5, adjust=True)
        >>> print(volatility.head())
                 Volat
        ID_1     0.65
        ID_2     0.78
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    if w < 0 or w > 1:
        raise ValueError("[!] w should be in the range [0, 1]!")
    
    # Get alphabet size
    alph = seqdata.states.copy()
    alph_size = len(alph)
    if with_missing and hasattr(seqdata, 'ismissing') and seqdata.ismissing:
        alph_size += 1  # Add 1 for missing state
    
    # Get normalized transition rate (proportion of transitions)
    transp_df = get_number_of_transitions(seqdata, norm=True)
    transp = transp_df['Transitions'].values
    
    # Get state distribution to count visited states
    sdist = get_state_freq_and_entropy_per_seq(seqdata, prop=False)
    state_cols = [col for col in sdist.columns if col != 'ID']
    nvisit = (sdist[state_cols] > 0).sum(axis=1).values
    
    # Calculate adjusted visited states proportion
    if adjust:
        # Adjustment: (nvisit - 1) / (alph_size - 1) when nvisit > 1, else 0
        ret = np.where(nvisit - 1 <= 0, 0, (nvisit - 1) / (alph_size - 1))
        ret = w * ret + (1 - w) * transp
    else:
        ret = w * (nvisit / alph_size) + (1 - w) * transp
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Volat': ret
    }, index=seqdata.seqdata.index)
    
    result = result.reset_index().rename(columns={'index': 'ID'})
    
    return result
