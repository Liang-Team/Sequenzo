"""
@Author  : 李欣怡 Xinyi Li
@File    : KMedoids.py
@Time    : 2025/2/8 11:53
@Desc    : 
"""

import numpy as np
from scipy.cluster.hierarchy import cut_tree

import importlib
import sequenzo.clustering.clustering_c_code
clustering_c_code = importlib.import_module("sequenzo.clustering.clustering_c_code")

from sequenzo.clustering.utils.disscenter import disscentertrim

import glob
import os
import sys
import cffi

ffi = cffi.FFI()

if sys.platform.startswith("win"):
    files = glob.glob(os.path.join(os.path.dirname(__file__), "*.pyd"))
else:
    files = glob.glob(os.path.join(os.path.dirname(__file__), "*.so"))

if not files:
    raise FileNotFoundError("No compiled library found")

lib_file = files[0]

try:
    # 重定向 stderr 来抑制 cffi 的错误信息输出
    import io
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        lib = ffi.dlopen(lib_file)
    finally:
        # 恢复 stderr
        sys.stderr = old_stderr
except ImportError as e:
    if sys.platform.startswith("win") and 'cffi mode "ANY" is only "ABI"' in str(e):
        # Windows 降级到 ABI 模式，同样抑制错误信息
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            lib = ffi.dlopen(lib_file)
        finally:
            sys.stderr = old_stderr
    else:
        raise

def KMedoids(diss, k, weights=None, npass=1, initialclust=None, method='PAMonce', cluster_only=False):

    # Lazily import the c_code module to avoid circular dependencies during installation
    # from .__init__ import _import_c_code
    # c_code = _import_c_code()

    if isinstance(method, str):
        method = method.lower()
        method_map = ["kmedoids", "pam", "pamonce"]
        if method in method_map:
            method = method_map.index(method) + 1  # 1-based index

    if not (isinstance(method, int) and method in {1, 2, 3}):
        raise ValueError(f"[!] Unknown clustering method: {method}.")

    nelements = diss.shape[0]
    if nelements != diss.shape[1]:
        raise ValueError(f"[!] Dissipation matrix has {nelements} elements.")

    def internal_random_sample(nelements, k):
        return np.random.choice(nelements, k, replace=False)  # 0-based 直接适用

    if weights is None:
        weights = np.ones(diss.shape[1], dtype=float)

    if len(weights) != nelements:
        raise ValueError(f"[!] 'weights' should be a vector of length {nelements}.")

    if initialclust is None:
        initialclust = internal_random_sample(nelements, k)
    else:
        if _validate_linkage_matrix(initialclust):
            # initialclust = fcluster(initialclust, k, criterion='maxclust')  # 1-based 索引
            initialclust = cut_tree(initialclust, n_clusters=k).flatten() + 1  # 1-based 索引
        # TODO : 现在已经得到一个组了，为什么不用这个组当作 PAMonce/PAM 算法的初始化？反而利用这个组去选中心点？
        #  初始化中心点的必要性为什么大于组？初始化中心点无论好不好，最后经过不断迭代肯定能选出较好的
        # TODO : 就算想要从子样本扩展到全数据，入口参数的这个组也是可以的呀？
        if len(initialclust) == nelements:
            initialclust = disscentertrim(diss=diss, group=initialclust, medoids_index="first", weights=weights)

            if len(initialclust) != k:
                raise ValueError(f"[!] 'initialclust' should be a vector of cluster membership with k={k}.")

        npass = 0

    if len(initialclust) != k:
        raise ValueError(f"[!] 'initialclust' should be a vector of medoids index of length :{k}.")

    if isinstance(initialclust, list):
        initialclust = np.asarray(initialclust)
    if np.any((initialclust >= nelements) | (initialclust < 0)):
        raise ValueError(f"[!] Starting medoids should be in 1:{nelements}")

    if npass < 0:
        raise ValueError("[!] 'npass' should be greater than 0")

    if k < 2 or k > nelements:
        raise ValueError(f" [!] 'k' should be in [2, {nelements}]")

    if method == 1:   # KMedoid
        memb = clustering_c_code.KMedoid(nelements,
                                         diss.astype(np.float64),
                                         initialclust.astype(np.int32),
                                         npass,
                                         weights.astype(np.float64))
    elif method == 2:  # PAM
        memb = clustering_c_code.PAM(nelements,
                                     diss.astype(np.float64),
                                     initialclust.astype(np.int32),
                                     npass,
                                     weights.astype(np.float64))
    else:   # PAMonce
        memb = clustering_c_code.PAMonce(nelements,
                                         diss.astype(np.float64),
                                         initialclust.astype(np.int32),
                                         npass,
                                         weights.astype(np.float64))

    memb_matrix = memb.runclusterloop()

    print("[>] Computed successfully.")

    return memb_matrix

def _validate_linkage_matrix(initialclust):
    """
    Check that the passed matrix matches the linkage matrix type requirements
    """
    if not isinstance(initialclust, np.ndarray):
        return False    # Linkage matrix must be a NumPy array

    if initialclust.ndim != 2 or initialclust.shape[1] != 4:
        return False    # Linkage matrix must be a 2D array with 4 columns

    if initialclust.dtype != np.float64:
        return False    # Linkage matrix 'Z' must contain doubles (np.float64).

    return True


if __name__ == '__main__':
    # TODO : KMeodis 在 python3.11 里导包有 numpy 的问题
    # TODO : sequenzo 0.1.14 里找不到 KMeodis 模块（这是 init 的问题，现已修正）

    from sequenzo import *

    df = load_dataset('country_co2_emissions')

    time = list(df.columns)[1:]
    states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']

    sequence_data = SequenceData(df, time_type="age", time=time, id_col="country", states=states)

    om = get_distance_matrix(sequence_data, method="OM", sm="TRATE", indel="auto")

    centroid_indices = [0, 50, 100, 150, 190]
    n_pass = 10

    weights = np.ones(len(om))

    # Example 1: KMedoids algorithm without specifying the center point
    clustering = KMedoids(diss=om,
                          k=5,
                          method='KMedoids',
                          npass=n_pass,
                          weights=weights)

    # Example 2: PAM algorithm with a specified center point
    clustering = KMedoids(diss=om,
                          k=5,
                          method='PAM',
                          initialclust=centroid_indices,
                          npass=n_pass,
                          weights=weights)

    # Example 3: PAMonce algorithm with default parameters
    clustering = KMedoids(diss=om,
                          k=5,
                          method='PAMonce',
                          npass=n_pass,
                          weights=weights)