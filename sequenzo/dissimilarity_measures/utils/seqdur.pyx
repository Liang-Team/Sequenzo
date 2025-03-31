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
        for i in range(n):
            if seqdatanum[i, j] < 0:
                ffill[i, j] = ffill[i, j - 1]

    cdef cnp.ndarray[char, ndim=2] boundaries = np.ones((n, 1), dtype=bool)
    boundaries = np.concatenate([boundaries, ffill[:, 1:] != ffill[:, :-1]], axis=1)

    cdef cnp.ndarray[int32_t, ndim=2] group_ids = np.cumsum(boundaries, axis=1).astype(np.int32, copy=False)
    cdef cnp.ndarray[char, ndim=2] valid = seqdatanum >= 0

    cdef list group_durations = []
    cdef int max_groups = 0

    for i in range(n):
        counts = np.bincount(group_ids[i][valid[i]])
        counts = counts[1:] if counts.size > 0 else np.array([], dtype=int)
        group_durations.append(counts)
        max_groups = max(max_groups, len(counts))

    cdef cnp.ndarray[int32_t, ndim=2] result = np.full((n, max_groups), np.nan, dtype=np.int32)

    for i in range(n):
        counts = group_durations[i]
        result[i, :len(counts)] = counts

    return result