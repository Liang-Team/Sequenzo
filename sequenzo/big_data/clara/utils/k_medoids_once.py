"""
@Author  : 李欣怡
@File    : k_medoids_once.py
@Time    : 2025/2/8 11:53
@Desc    : 
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, cut_tree

import importlib
import sequenzo.dissimilarity_measures.c_code

c_code = importlib.import_module("sequenzo.dissimilarity_measures.c_code")

from sequenzo.clustering.utils.disscenter import disscentertrim

def k_medoids_once(diss, k, weights=None, npass=1, initialclust=None, method='PAMonce', cluster_only=False):

    # Lazily import the c_code module to avoid circular dependencies during installation
    # from .__init__ import _import_c_code
    # c_code = _import_c_code()

    if isinstance(method, str):
        method = method.lower()
        method_map = ["kmedoids", "pam", "pamonce"]
        if method in method_map:
            method = method_map.index(method) + 1  # 1-based index

    if not (isinstance(method, int) and method in {1, 2, 3}):
        raise ValueError(f"[!] Unknown clustering method {method}.")

    nelements = diss.shape[0]
    if nelements != diss.shape[1]:
        raise ValueError(f"[!] Dissipation matrix has {nelements} elements.")

    def internal_random_sample(nelements, k):
        return np.random.choice(nelements, k, replace=False)  # 0-based 直接适用

    if weights is None:
        weights = np.ones(diss.shape[1], dtype=float)

    if len(weights) != nelements:
        raise ValueError(f"[!] [!] weights should be a vector of length {nelements}.")

    if initialclust is None:
        initialclust = internal_random_sample(nelements, k)
    else:
        # initialclust = fcluster(initialclust, k, criterion='maxclust')  # 1-based 索引
        initialclust = cut_tree(initialclust, n_clusters=k).flatten() + 1  # 1-based 索引

        if len(initialclust) == nelements:
            initialclust = disscentertrim(diss=diss, group=initialclust, medoids_index="first", weights=weights)

            if len(initialclust) != k:
                raise ValueError(f"[!] 'initialclust' should be a vector of cluster membership with k={k}.")

        npass = 0

    if len(initialclust) != k:
        raise ValueError(f"[!] 'initialclust' should be a vector of medoids index of length :{k}.")

    if np.any((initialclust >= nelements) | (initialclust < 0)):
        raise ValueError(f" [!] Starting medoids should be in 1:{nelements}")

    if npass < 0:
        raise ValueError(" [!] 'npass' should be greater than 0")

    if k < 2 or k > nelements:
        raise ValueError(f" [!] 'k' should be in [2, {nelements}]")

    memb = c_code.PAMonce(nelements,
                           diss.astype(np.float64),
                           initialclust.astype(np.int32),
                           npass,
                           weights.astype(np.float64))
    memb_matrix = memb.runclusterloop()

    return memb_matrix


