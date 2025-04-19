# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np

import pandas as pd
from libc.math cimport isnan

def get_sm_trate_substitution_cost_matrix(
    object seqdata,
    bint time_varying=False,
    bint weighted=True,
    int lag=1,
    bint count=False
):
    """
    Compute substitution cost matrix (transition rate matrix)
    """

    from sequenzo.define_sequence_data import SequenceData
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[x] Seqdata must be a pandas DataFrame wrapped in a SequenceData object.")

    cdef np.ndarray[np.float64_t, ndim=1] weights
    if weighted:
        weights = np.asarray(seqdata.weights, dtype=np.float64)
    else:
        weights = np.ones(seqdata.seqdata.shape[0], dtype=np.float64)

    states = seqdata.states
    statesMapping = seqdata.state_mapping
    cdef int _size = len(states) + 1
    df = seqdata.seqdata
    cdef int n_rows = df.shape[0]
    cdef int sdur = df.shape[1]
    cdef int i, j, t, sl, state_x, state_y
    cdef double PA, PAB

    if lag < 0:
        all_transition = list(range(abs(lag), sdur))
    else:
        all_transition = list(range(sdur - lag))
    cdef int num_transition = len(all_transition)

    # convert df to NumPy 2D array of ints
    seq_mat = df.to_numpy(dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] seq_mat_mv = seq_mat

    if time_varying:
        tmat = np.zeros((num_transition, _size, _size), dtype=np.float64)

        for idx, sl in enumerate(all_transition):
            for state_x in statesMapping.values():
                PA = 0.0
                for i in range(n_rows):
                    if seq_mat_mv[i, sl] == state_x and not isnan(seq_mat_mv[i, sl + lag]):
                        PA += weights[i]

                if PA == 0:
                    tmat[idx, state_x, :] = 0
                else:
                    for state_y in statesMapping.values():
                        PAB = 0.0
                        for i in range(n_rows):
                            if (seq_mat_mv[i, sl] == state_x and
                                not isnan(seq_mat_mv[i, sl + lag]) and
                                seq_mat_mv[i, sl + lag] == state_y):
                                PAB += weights[i]

                        tmat[idx, state_x, state_y] = PAB if count else PAB / PA

    else:
        tmat = np.zeros((_size, _size), dtype=np.float64)

        for state_x in statesMapping.values():
            PA = 0.0
            for i in range(n_rows):
                for t in all_transition:
                    if (seq_mat_mv[i, t] == state_x and
                        not isnan(seq_mat_mv[i, t + lag])):
                        PA += weights[i]

            if PA == 0:
                tmat[state_x, :] = 0
            else:
                for state_y in statesMapping.values():
                    PAB = 0.0
                    for i in range(n_rows):
                        for t in all_transition:
                            if (seq_mat_mv[i, t] == state_x and
                                seq_mat_mv[i, t + lag] == state_y):
                                PAB += weights[i]

                    tmat[state_x, state_y] = PAB if count else PAB / PA

    return tmat