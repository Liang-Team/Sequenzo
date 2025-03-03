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
import numpy as np
from sequenzo.define_sequence_data import SequenceData


def seqlength(seqdata):
    if isinstance(seqdata, SequenceData):
        seqdata = seqdata.seqdata.replace(np.nan, -99)

    # Get effective length after removing void elements
    if isinstance(seqdata, pd.DataFrame):
        lengths = seqdata.apply(lambda row: row.dropna().shape[0], axis=1)
        lengths.name = 'Length'

    else:
        lengths = np.sum(~np.isnan(seqdata), axis=1)
        lengths = pd.Series(lengths, name="Length")

    return lengths