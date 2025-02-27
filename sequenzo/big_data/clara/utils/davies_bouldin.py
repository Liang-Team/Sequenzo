"""
@Author  : 李欣怡
@File    : davies_bouldin.py
@Time    : 2024/12/27 17:56
@Desc    :
    :param
        diss : numpy 2D, 距离矩阵
        clustering : numpy 1D, 每个数据点的隶属矩阵（一种可能：初始化时每个点构成一个簇，则每个数据点隶属自己）
        medoids : numpy 1D, 簇的中心点
"""
import numpy as np


def davies_bouldin_internal(diss, clustering, medoids, p=1, weights=None, medoidclust=False):
    # If weights are not provided, use uniform weights
    if weights is None:
        weights = np.ones(diss.shape[0])

    list_diam = np.zeros(len(medoids))

    # Calculate the diameter for each medoid
    for i in range(len(medoids)):
        medi = medoids[i] if medoidclust else i
        cond = (clustering == medi)

        # Calculate the diameter (weighted distance)
        list_diam[i] = (np.sum(weights[cond] * diss[cond, i] ** p) / np.sum(weights[cond])) ** (1 / p)

    maximum = np.zeros(len(medoids))

    # Calculate the maximum ratio for each medoid
    for i in range(len(medoids)):
        # Calculate the distance to other medoids
        maximum2 = (list_diam[i] + list_diam) / diss[medoids[i], :]

        # Take the maximum of the valid (finite) values
        # ensure values for "same" medoids
        maximum[i] = np.max(maximum2[np.isfinite(maximum2)])

    # Calculate the final Davies-Bouldin index (average of maximum values)
    final_db = np.mean(maximum)

    return {'db': final_db, 'per_cluster': maximum}


def fuzzy_davies_bouldin_internal(diss, memb, medoids, weights=None):
    if weights is None:
        weights = np.ones(diss.shape[0])

    # R 中定义后未使用，用另一个值赋值了
    list_diam = np.zeros(len(medoids))

    # R 中只定义未使用
    # n = np.sum(weights)

    mw = memb * weights[:, None]
    list_diam = np.sum(mw * diss, axis=0) / np.sum(mw, axis=0)

    # 初始化一个数组来存储每个簇的最大值
    maximum = np.zeros(len(medoids))

    # 对每个簇计算其与其他簇的相似度
    for i in range(len(medoids)):
        maximum2 = (list_diam[i] + list_diam) / diss[medoids[i], :]

        maximum[i] = np.max(maximum2[np.isfinite(maximum2)])

    final_db = np.mean(maximum)

    return {'db': final_db, 'per_cluster': maximum}


def adjpbm_internal(diss, clustering, medoids, p=1, weights=None, medoidclust=False):
    if weights is None:
        weights = np.ones(diss.shape[0])

    # Calculate internal distance
    internaldist = [
        (sum(weights[clustering == (medoids[i] if medoidclust else i)] * diss[
            clustering == (medoids[i] if medoidclust else i), i] ** p) /
         sum(weights[clustering == (medoids[i] if medoidclust else i)])) ** (1 / p)
        for i in range(len(medoids))
    ]

    # Calculate the minimum separation distance between medoids
    separation = np.nanmin(diss[medoids, :][:, medoids])

    # Calculate pbm (probabilistic cluster separation)
    pbm = (1 / len(medoids)) * (separation / np.sum(internaldist))

    return pbm
