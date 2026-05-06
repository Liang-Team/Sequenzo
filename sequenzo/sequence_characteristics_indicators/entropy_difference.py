"""
@Author  : Yuqi Liang 梁彧祺
@File    : entropy_difference.py
@Time    : 01/02/2026 14:15
@Desc    : Entropy difference based on spell durations

    This module implements entropy difference calculation, which measures
    the entropy of spell durations within sequences.

    Reference: TraMineR R package
    - seqientdiff: R/seqientdiff.R
    - entropy helper: R/entropy.R
    https://github.com/cran/TraMineR/blob/master/R/seqientdiff.R
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur
from .simple_characteristics import cut_prefix


def get_entropy_difference(seqdata: SequenceData, norm: bool = True) -> pd.DataFrame:
    """
    Calculate entropy difference based on spell durations.
    
    This function computes the entropy of spell durations (consecutive periods
    in the same state) for each sequence. Higher entropy indicates more variability
    in spell lengths.
    
    Args:
        seqdata: SequenceData object containing sequence data
        norm: If True, normalize entropy by maximum possible entropy.
              Maximum entropy occurs when all spells have equal length.
    
    Returns:
        pd.DataFrame: DataFrame with one column 'Hdss' containing entropy
                     difference values. Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqientdiff.R
        https://github.com/cran/TraMineR/blob/master/R/seqientdiff.R
    
    Examples:
        >>> ent_diff = get_entropy_difference(seqdata, norm=True)
        >>> print(ent_diff.head())
                 Hdss
        ID_1     0.85
        ID_2     0.72
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    def entropydiff(dur, norm):
        """
        Calculate entropy difference for a single sequence's spell durations.
        
        Args:
            dur: Array of spell durations
            norm: Whether to normalize
        
        Returns:
            Entropy value
        """
        # Remove NaN values
        dur = dur[~np.isnan(dur)]
        dur = dur[dur > 0]  # Only positive durations
        
        len_seq = np.sum(dur)
        
        # If no valid durations, return 0
        if len(dur) == 0 or len_seq == 0:
            return 0.0
        
        # Calculate entropy
        ent = entropy(dur, base=np.e)
        
        if ent > 0 and norm:
            # Maximum entropy occurs when all spells have equal length
            # If length of DSS = length of sequence, each spell has length 1
            p = 1.0 / len_seq
            entmax = (-len_seq) * (p * np.log(p))
            ent = ent / entmax
        
        return ent
    
    # Get spell durations for all sequences
    iseqtab = pd.DataFrame(seqdur(seqdata))
    
    # Apply entropy calculation to each sequence
    # Use cut_prefix to remove trailing zeros/NaN
    ient = iseqtab.apply(lambda row: entropydiff(cut_prefix(row, 1), norm=norm), axis=1)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Hdss': ient.values
    }, index=seqdata.seqdata.index)
    
    result = result.reset_index().rename(columns={'index': 'ID'})
    
    return result
