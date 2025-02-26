"""
@Author  : 李欣怡
@File    : seqdss.py
@Time    : 2024/11/19 10:11
@Desc    : Extracts distinct states from sequences
"""

import os
from contextlib import redirect_stdout

import numpy as np

from dissimilarity_measures.seqdef import SequenceData
from seqlength import seqlength


def seqdss(seqdata, with_missing=False):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] data is NOT a sequence object, see seqdef function to create one.")

    nbseq = len(seqdata.seqdata)
    sl = seqlength(seqdata)
    maxsl = sl.max().max()

    statl = np.arange(len(seqdata.alphabet))
    nr = seqdata.nr

    trans = np.full((nbseq, maxsl), np.nan, dtype='U25')

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

        while idx < sl.iloc[i, 0]:
            iseq = tmpseq[idx]

            while idx < sl.iloc[i, 0] - 1 and (tmpseq[idx + 1] == iseq or tmpseq[idx + 1] == -99):
                idx += 1

            if iseq != -99:
                trans[i, j] = statl[iseq]
                j += 1

            idx += 1

        if j > maxcol:
            maxcol = j

    trans = trans[:, :maxcol]

    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            result = SequenceData(
                trans,
                var=seqdata.var,
                alphabet=seqdata.alphabet,
                states=seqdata.states,
                labels=seqdata.labels,
                id=seqdata.id,
                id_col=seqdata.id_col,
                weights=seqdata.weights,
                start=seqdata.start,
                missing_handling=seqdata.missing_handling,
                void=seqdata.void,
                nr=seqdata.nr,
                cpal=seqdata.cpal
            )

    return result
