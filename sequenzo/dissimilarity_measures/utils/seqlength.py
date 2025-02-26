"""
@Author  : 李欣怡
@File    : seqlength.py
@Time    : 2024/10/31 11:02
@Desc    : Returns a vector with the lengths of the sequences in seqdata
            (missing values count toward the sequence length, but invalid values do not)
            TraMineR_length evaluates the length of a sequence and returns a number
            Here, we compute the lengths of all the sequences and return a data frame
"""
import pandas as pd
from dissimilarity_measures.seqdef import SequenceData

def seqlength(seqdata):
    if isinstance(seqdata, SequenceData):
        seqdata = seqdata.seqdata

    # Get effective length after removing void elements
    non_nan_counts = seqdata.notna().sum(axis=1)

    seq_length = non_nan_counts.groupby(seqdata.index).sum()

    seq_length = pd.Series(seq_length, name="Length", index=seqdata.index)

    return seq_length