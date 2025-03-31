import pandas as pd
import numpy as np
cimport numpy as cnp
from sequenzo.define_sequence_data import SequenceData
from libc.stdint cimport int32_t

def seqlength(seqdata):
    if isinstance(seqdata, SequenceData):
        seqdata = seqdata.seqdata.replace(np.nan, -99)

    cdef cnp.ndarray[int32_t, ndim=2] seqarray_long

    if isinstance(seqdata, pd.DataFrame):
        seqarray_long = seqdata.to_numpy(dtype=np.int32)
        return np.sum(seqarray_long > 0, axis=1)

    else:
        seqarray_long = seqdata
        return np.sum(seqarray_long > 0, axis=1)
