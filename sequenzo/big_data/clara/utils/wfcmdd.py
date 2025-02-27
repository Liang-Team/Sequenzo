"""
@Author  : 李欣怡
@File    : wfcmdd.py
@Time    : 2024/12/28 13:38
@Desc    : 
"""

import numpy as np
import pandas as pd
import warnings


def wfcmdd(diss, memb, weights=None, method="FCMdd", m=2, dnoise=None, eta=None, alpha=0.001,
           iter_max=100, verbose=False, dlambda=None):
    # Setting and checking argument values
    METHODS = ["NCdd", "HNCdd", "FCMdd", "PCMdd"]

    if method not in METHODS:
        raise ValueError(f" [!] Method must be one of {METHODS}.")

    # TODO:源码中没有 weights = null 时的处理
    if weights is None:
        weights = np.ones(len(diss), dtype=int)

    # R 源码中只定义未使用
    # pweights = weights / np.sum(weights)
    d = np.array(diss)
    n = d.shape[0]

    if method == "NCdd":
        if dnoise is None and dlambda is None:
            raise ValueError(" [!] Must provide a value for dnoise or dlambda.")
        if dlambda is not None:
            dnoise = 1
    elif method == "HNCdd":
        if dnoise is None:
            raise ValueError(" [!] Must provide a value for dnoise.")
        m = 1
    elif method == "PCMdd":
        if eta is None:
            raise ValueError(" [!] Must provide a vector of values for eta.")

    # Checking the membership matrix (memb)
    if isinstance(memb, (pd.DataFrame, np.ndarray)):  # Check if memb is matrix or dataframe-like
        if memb.shape[0] != d.shape[1]:
            raise ValueError(" [!] The number of rows in memb must be the same as the number of rows and columns of d.")
        u = memb.to_numpy() if isinstance(memb, pd.DataFrame) else memb

    elif isinstance(memb, list) and all(isinstance(x, (int, float)) for x in memb):
    # else if (is.vector(memb) && is.numeric(memb))
        u = np.zeros((n, len(memb)))
        for k in range(len(memb)):
            u[memb[k], k] = 1

    else:
        raise ValueError("[!] Provide a number, a vector of seeds, or membership matrix for mobile clusters.")

    kMov = u.shape[1]
    med = np.full(kMov, np.nan)

    if method == "PCMdd" and len(eta) != kMov:
        raise ValueError(" [!] Vector of reference distances (eta) must have a length equal to the number of clusters.")

    if method in ["NCdd", "HNCdd"]:
        # u <- cbind(u, vector("numeric", length = n))
        u = np.hstack([u, np.zeros((n, 1))])

    kMovNC = u.shape[1]
    # print("kMovNC = ", kMovNC)
    uPrev = np.zeros((n, kMovNC))

    if dlambda is not None:
        kdiv = kMov * np.sum(weights)

    dist2med = np.zeros((n, kMovNC))
    # print("dist2med = ", dist2med)

    if method in ["NCdd", "HNCdd"]:
        dist2med[:, kMovNC - 1] = dnoise

    continue_flag = True
    iter_count = 1
    uPrev2 = 0
    # print("u = ", u)
    # print("d = ", d)
    # print("med = ", med)
    while continue_flag:
        # Finding centers
        for k in range(kMov):
            # candidates < - which(apply(u[, -k, drop=FALSE], 1, max) < 1 & (!1:n % in %med[0:(k - 1)]))
            # med[k] < - candidates[which.min((u[, k] ^ m * weights) % * % d[, candidates])]
            # dist2med[, k] < - d[, med[k]]

            u_removed_k = np.delete(u, k, axis=1)      # 去掉第 k 列
            max_per_row = np.max(u_removed_k, axis=1)  # 每行的最大值

            # 查找最大值小于 1 的行
            candidates = np.where((max_per_row < 1) & (~np.isin(np.arange(1, len(u) + 1), med[:k])))[0]
            # print("candidates = ", candidates)

            u_k_m = u[:, k] ** m
            # print("u_k_m = ", u_k_m)

            # 按照权重与距离矩阵进行矩阵乘法
            weighted_u_k_m = u_k_m * weights
            # print("weighted_u_k_m =", weighted_u_k_m)

            # 从 d 中选择 candidates 列
            d_candidates = d[:, candidates]
            # print("d_candidates =", d_candidates)

            # 进行矩阵乘法
            product = weighted_u_k_m @ d_candidates
            # print("product = ", product)
            # 选取最小值对应的索引
            min_index = np.argmin(product)
            # print("min_index = ", min_index)

            med[k] = candidates[min_index]  # 更新 med[k]
            # print("med[k] = ", med[k])

            dist2med[:, k] = d[:, int(med[k])]
            # print("dist2med[:, k] = ", dist2med[:, k])

        # Updating dnoise for adaptive dnoise clustering
        if dlambda is not None and method == "NCdd":
            dnoise = dlambda * np.sum(dist2med[:, :-1] * weights[:, None]) / (kMov * np.sum(weights))
            dist2med[:, kMovNC - 1] = dnoise


        # Updating membership
        if method == "HNCdd":
            d2cm = np.hstack([dist2med, np.full((dist2med.shape[0], 1), dnoise)])
            u = np.zeros_like(u)
            minC = np.argmin(d2cm, axis=1)
            for i in range(len(minC)):
                u[i, minC[i]] = 1

        elif method in ["FCMdd", "NCdd"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # dist2med_safe = np.where(dist2med == 0, 1e-10, dist2med)
                # TODO : 不显示中间报错
                u = (1 / dist2med) ** (1 / (m - 1))
                u /= np.sum(u, axis=1, keepdims=True)
                u[dist2med == 0] = 1

        elif method == "PCMdd":
            for k in range(kMov):
                u[:, k] = 1 / (1 + (dist2med[:, k] / eta[k]) ** (1 / (m - 1)))
            u[dist2med == 0] = 1

        # Checking convergence
        if iter_count > 2:
            continue_flag = np.max(np.abs(u - uPrev)) > alpha and iter_count <= iter_max \
                            and np.max(np.abs(u - uPrev2)) > alpha

        if continue_flag:
            uPrev2 = uPrev
            uPrev = u
            iter_count += 1
            if verbose:
                print(".", end="")

    # Calculate the functional value
    if method in ["NCdd", "FCMdd"]:
        functional = np.sum(dist2med * (u ** m) * weights[:, None])
    elif method == "HNCdd":
        functional = np.sum(dist2med * (u ** m) * weights[:, None])
    elif method == "PCMdd":
        functional = 0
        for k in range(kMov):
            functional += np.sum(dist2med[:, k] * (u[:, k] ** m) * weights) + np.sum(
                eta[k] * (1 - u[:, k]) ** m * weights)

    if verbose:
        print(f"\nIterations: {iter_count}, Functional: {functional}")

    mobile_centers = med[:kMov]

    return {
        "dnoise": dnoise,
        "memb": u,
        "mobileCenters": mobile_centers,
        "functional": functional
    }


if __name__ == "__main__":
    diss = np.array([[0.0, 1.0, 2.0],
                     [1.0, 0.0, 1.0],
                     [2.0, 1.0, 0.0]])
    diss = pd.DataFrame(diss)

    memb = np.array([[0.7, 0.3],
                     [0.2, 0.8],
                     [0.5, 0.5]])

    result = wfcmdd(diss=diss, memb=memb, method="FCMdd")

    print("result['dnoise'] = ", result['dnoise'])
    print("result['memb'] =")
    print(result['memb'])
    print("result['mobileCenters'] = ", result['mobileCenters'])
    print("result['functional'] = ", result['functional'])