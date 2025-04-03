import numpy as np
cimport numpy as cnp

def get_weighted_diss(cnp.ndarray[double, ndim=2] diss,
                      cnp.ndarray[double, ndim=1] weights):
    cdef int n = weights.shape[0]
    cdef int i, j
    cdef double factor

    for i in range(n):
        for j in range(i + 1, n):
            factor = (weights[i] * weights[j]) ** 0.5
            diss[i, j] *= factor
            diss[j, i] = diss[i, j]

    return diss
