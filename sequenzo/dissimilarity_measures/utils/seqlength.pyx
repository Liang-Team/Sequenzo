# cython: np_import_array=True
import pandas as pd
import numpy as np
cimport numpy as cnp

from sequenzo.define_sequence_data import SequenceData
from libc.stdint cimport int32_t

def seqlength(seqdata):
    cdef cnp.ndarray[int32_t, ndim=2] seqarray_long

    # Handle SequenceData - use values property directly (similar to seqdur.pyx)
    if isinstance(seqdata, SequenceData):
        seqarray_long = seqdata.values.copy().astype(np.int32, copy=False)
        return np.sum(seqarray_long > 0, axis=1)

    # Handle DataFrame
    elif isinstance(seqdata, pd.DataFrame):
        seqarray_long = seqdata.replace(np.nan, -99).to_numpy(dtype=np.int32)
        return np.sum(seqarray_long > 0, axis=1)

    # Handle numpy array
    elif isinstance(seqdata, np.ndarray):
        seqarray_long = seqdata.astype(np.int32)
        return np.sum(seqarray_long > 0, axis=1)

    else:
        # Try to convert to numpy array as last resort
        seqarray_long = np.asarray(seqdata, dtype=np.int32)
        return np.sum(seqarray_long > 0, axis=1)
