import numpy as np
cimport numpy as cnp
from sequenzo.define_sequence_data import SequenceData
from libc.stdint cimport int32_t

def seqdur(seqdata):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("data is not a sequence object, see SequenceData to create one")

    cdef cnp.ndarray[int32_t, ndim=2] seqdatanum = seqdata.values.copy().astype(np.int32, copy=False)

    cdef int n = seqdatanum.shape[0], m = seqdatanum.shape[1]
    cdef int i, j

    cdef cnp.ndarray[int32_t, ndim=2] ffill = seqdatanum.copy()

    for j in range(1, m):
        ffill[:, j] = np.where(seqdatanum[:, j] < 0, ffill[:, j - 1], seqdatanum[:, j])

    cdef cnp.ndarray[char, ndim=2] boundaries = np.concatenate(
        [np.ones((n, 1), dtype=bool), ffill[:, 1:] != ffill[:, :-1]], axis=1
    )

    cdef cnp.ndarray[int32_t, ndim=2] group_ids = np.cumsum(boundaries, axis=1).astype(np.int32, copy=False)
    cdef cnp.ndarray[char, ndim=2] valid = seqdatanum >= 0

    cdef cnp.ndarray[int32_t, ndim=2] group_durations = np.zeros((n, m), dtype=np.int32)

    for i in range(n):
        counts = np.bincount(group_ids[i][valid[i]])
        counts = counts[1:] if counts.size > 0 else np.array([], dtype=int)
        group_durations[i, :len(counts)] = counts

    return group_durations
