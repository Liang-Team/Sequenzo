"""
@Author  : Xinyi Li 李欣怡
@File    : seqdss.py
@Time    : 2024/11/19 10:11
@Desc    : Extracts distinct states from sequences
"""

import numpy as np
from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqlength import seqlength


def seqdss(seqdata, with_missing=False):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] data is NOT a sequence object, see SequenceData to create one.")

    number_seq = len(seqdata.seqdata)
    slen = seqlength(seqdata)
    maxsl = slen.max()

    trans = np.full((number_seq, maxsl), np.nan)

    # Converts character data to numeric values
    seqdatanum = seqdata.values

    if not with_missing:
        seqdatanum[np.isnan(seqdatanum)] = -99

    maxcol = 0
    for i in range(number_seq):
        idx = 0
        j = 0

        tmpseq = seqdatanum[i, :]

        while idx < slen.iloc[i]:
            current_code = tmpseq[idx]

            while idx < slen.iloc[i] - 1 and (tmpseq[idx + 1] == current_code or tmpseq[idx + 1] == -99):
                idx += 1

            if current_code != -99:
                trans[i, j] = current_code
                j += 1

            idx += 1

        if j > maxcol:
            maxcol = j

    trans = trans[:, :maxcol]

    return trans
