"""
@Author  : 李欣怡
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

    nbseq = len(seqdata.seqdata)
    sl = seqlength(seqdata)
    maxsl = sl.max()

    statl = np.arange(len(seqdata.alphabet))
    nr = seqdata.nr

    trans = np.full((nbseq, maxsl), 0)

    if with_missing:
        statl.append(nr)

    # Converts character data to numeric values
    seqdatanum = seqdata.values

    if not with_missing:
        seqdatanum[np.isnan(seqdatanum)] = -99

    maxcol = 0
    for i in range(nbseq):
        idx = 0
        j = 0

        tmpseq = seqdatanum[i, :]

        while idx < sl.iloc[i]:
            iseq = int(tmpseq[idx])

            while idx < sl.iloc[i] - 1 and (tmpseq[idx + 1] == iseq or tmpseq[idx + 1] == -99):
                idx += 1

            if iseq != -99:
                trans[i, j] = statl[iseq]
                j += 1

            idx += 1

        if j > maxcol:
            maxcol = j

    trans = trans[:, :maxcol]

    return trans
