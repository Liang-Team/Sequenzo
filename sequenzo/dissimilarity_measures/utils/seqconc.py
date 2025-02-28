"""
@Author  : 李欣怡
@File    : seqconc.py
@Time    : 2024/11/2 18:02
@Desc    : Concatenates vectors of states or events into character strings
"""
import numpy as np
import pandas as pd
from sequenzo.define_sequence_data import SequenceData


def sconc_pd(seqdata, sep):
    vi = seqdata.notna()  # Choose values that are not NA
    return sep.join(seqdata[vi].astype(str))


def seqconc(data, sep="-", vname=['Sequence']):
    if isinstance(data, SequenceData):
        cseq = data.seqdata.apply(lambda row: sconc_pd(row, sep), axis=1)
        cseq.index = data.seqdata.index

        return cseq

    elif isinstance(data, pd.DataFrame):
        cseq = data.apply(lambda row: sconc_pd(row, sep), axis=1)
        cseq.index = data.index

        return cseq

    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            cseq = sconc_pd(pd.Series(data), sep)
        elif data.ndim == 2:
            # For 2D arrays, we treat each row as a sequence (similar to DataFrame rows)
            cseq = [sconc_pd(pd.Series(row), sep) for row in data]
        else:
            raise ValueError("Only 1D and 2D arrays are supported.")

        cseq = pd.DataFrame(cseq, columns=vname)

        return cseq

