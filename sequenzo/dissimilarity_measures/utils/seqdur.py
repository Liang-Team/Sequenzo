"""
@Author  : 李欣怡
@File    : seqdur.py
@Time    : 2024/11/12 00:20
@Desc    : Extracts states durations from sequences
"""
import pandas as pd
import numpy as np
from dissimilarity_measures.seqdef import SequenceData
from seqlength import seqlength

# example:
#     input:
#         A-A-A-B-B-C-D
#         A-B-B-B-B-B-B
#         B-B-B-B-C-C-C
#     output:
#         3-2-1-1
#         1-6
#         4-3
def seqdur(seqdata):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("data is not a sequence object, see seqdef function to create one")

    seq_length = seqlength(seqdata).iloc[:, 0]
    maxsl = max(seq_length)

    seqdata = seqdata.seqdata
    nbseq = seqdata.shape[0]

    trans = np.full((nbseq, maxsl), np.nan)
    trans_df = pd.DataFrame(trans, index=seqdata.index, columns=[f"DUR{i + 1}" for i in range(maxsl)])

    seqdatanum = seqdata.to_numpy()
    seqdatanum[np.isnan(seqdatanum)] = -99

    maxcol = 0
    for i in range(nbseq):
        idx = 0
        j = 0

        tmpseq = seqdatanum[i, :]

        # Skipping initial -99 values
        while idx < seq_length[i] and tmpseq[idx] == -99:
            idx += 1

        while idx < seq_length[i]:
            iseq = tmpseq[idx]
            dur = 1

            # calculate duration
            while idx < seq_length[i] - 1 and (tmpseq[idx + 1] == iseq or tmpseq[idx + 1] == -99):
                if tmpseq[idx + 1] != -99:
                    dur += 1
                idx += 1

            if iseq != -99:
                trans_df.iloc[i, j] = dur
                j += 1

            idx += 1

        if j > maxcol:
            maxcol = j

    # Remove redundant columns to ensure that
    # the matrix returned is of the appropriate dimension
    trans_df = trans_df.iloc[:, :maxcol]

    return trans_df