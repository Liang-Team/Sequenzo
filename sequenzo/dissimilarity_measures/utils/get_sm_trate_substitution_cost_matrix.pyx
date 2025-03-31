import numpy as np
cimport numpy as cnp
from sequenzo.define_sequence_data import SequenceData
from libc.stdint cimport int32_t

def get_sm_trate_cost_matrix(seqdata, bint time_varying=False, bint weighted=True, int lag=1, bint count=False):
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[x] Seqdata must be a pandas DataFrame representing sequences.")

    cdef cnp.ndarray[double, ndim=1] weights
    if weighted:
        weights = seqdata.weights
    else:
        weights = np.ones(seqdata.seqdata.shape[0])

    cdef list states = seqdata.states
    cdef dict statesMapping = seqdata.state_mapping

    cdef int _size = len(states) + 1
    cdef int sdur = seqdata.seqdata.shape[1]

    cdef cnp.ndarray[int32_t, ndim=1] all_transition
    if lag < 0:
        all_transition = np.arange(abs(lag), sdur, dtype=np.int32)
    else:
        all_transition = np.arange(sdur - lag, dtype=np.int32)
    cdef int num_transition = len(all_transition)

    cdef cnp.ndarray[double, ndim=3] tmat_3d = np.zeros((num_transition, _size, _size))
    cdef cnp.ndarray[double, ndim=2] tmat_2d = np.zeros((_size, _size))

    cdef int sl, state_x, state_y
    cdef cnp.ndarray[char, ndim=2] missing_cond

    seqdata = seqdata.seqdata.to_numpy(dtype=np.int32)

    if time_varying:
        for sl in all_transition:
            missing_cond = np.not_equal(seqdata[:, [sl + lag]], np.nan)

            for state_x in statesMapping.values():
                colx_cond = (seqdata[:, sl] == state_x)
                PA = np.sum(weights * colx_cond * missing_cond)

                if PA == 0:
                    tmat_3d[sl, state_x, :] = 0
                else:
                    for state_y in statesMapping.values():
                        PAB = np.sum(weights * colx_cond * (seqdata[:, sl + lag] == state_y))
                        tmat_3d[sl, state_x, state_y] = PAB if count else PAB / PA

        return tmat_3d

    else:
        missing_cond = np.not_equal(seqdata[:, all_transition + lag], np.nan)

        for state_x in statesMapping.values():
            PA = np.sum(weights * np.sum((seqdata[:, all_transition] == state_x) & missing_cond, axis=1))

            if PA == 0:
                tmat_2d[state_x, :] = 0
            else:
                for state_y in statesMapping.values():
                    PAB = np.sum(weights * np.sum(
                        (seqdata[:, all_transition] == state_x) & (seqdata[:, all_transition + lag] == state_y),
                        axis=1))
                    tmat_2d[state_x, state_y] = PAB if count else PAB / PA

        return tmat_2d