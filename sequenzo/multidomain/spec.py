"""
@Author  : Yuqi Liang 梁彧祺
@File    : spec.py
@Time    : 19/04/2025 14:28
@Desc    : 
"""
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import hdbscan


def cmdscale(D):
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # Double centered distance matrix
    B = -0.5 * H @ (D ** 2) @ H

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Keep only positive eigvals
    w_positive = eigvals > 0
    L = np.diag(np.sqrt(eigvals[w_positive]))
    V = eigvecs[:, w_positive]

    # Coordinates
    X = V @ L

    return X, eigvals


def find_optimal_mds_dimension(cum_explained, title='MDS Dimensionality Selection', var_threshold=0.95, max_dim=100,
                               plot=True):
    """
    自动寻找 MDS 的最佳维度数（综合膝点、解释方差阈值、最大维度限制）。

    参数:
        cum_explained (list or np.ndarray): 累积解释方差（cumulative explained variance）
        title (str): 图表标题
        var_threshold (float): 累积解释方差阈值（如0.95表示95%）
        max_dim (int): 最大允许的维度数
        plot (bool): 是否绘图

    返回:
        dict:
            - 'knee': 膝点对应维度
            - 'var_cutoff': 达到解释方差阈值的维度
            - 'recommended': 综合考虑后的最终推荐维度
    """
    dimensions = list(range(1, len(cum_explained) + 1))

    # 1. 膝点检测
    knee_finder = KneeLocator(dimensions, cum_explained, curve='concave', direction='increasing')
    knee = knee_finder.knee or len(cum_explained)  # fallback if no knee found

    # 2. 解释方差阈值点
    var_cutoff = np.argmax(cum_explained >= var_threshold) + 1 if np.any(cum_explained >= var_threshold) else len(
        cum_explained)

    # 3. 综合推荐维度
    recommended = min(knee, var_cutoff, max_dim)

    # 4. 绘图
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(dimensions, cum_explained, marker='o', label='Cumulative explained variance')
        plt.axvline(knee, color='red', linestyle='--', label=f'Knee at {knee}')
        plt.axvline(var_cutoff, color='blue', linestyle='--', label=f'{int(var_threshold * 100)}% Var at {var_cutoff}')
        plt.axvline(recommended, color='green', linestyle='-', label=f'Recommended: {recommended}', linewidth=2)
        plt.xlabel('Number of dimensions')
        plt.ylabel('Cumulative explained variance')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'knee': knee,
        'var_cutoff': var_cutoff,
        'recommended': recommended
    }


if __name__ == '__main__':

    from sequenzo import *
    import pandas as pd

    family_df = pd.read_csv('/Users/lei/Documents/japan_romance/multidomain_algorithm/family_15_35.csv')
    happiness_df = pd.read_csv('/Users/lei/Documents/japan_romance/multidomain_algorithm/happiness_15_35.csv')

    time_cols = []

    for i in list(range(15, 36)):
        time_cols.append(str(i))

    family_sequence = SequenceData(data=family_df,
                                   time_type='age',
                                   time=time_cols,
                                   states=[1, 2, 3],
                                   labels=["Single", "Romantic Partner", "Married"])

    happiness_sequence = SequenceData(data=happiness_df,
                                      time_type='age',
                                      time=time_cols,
                                      states=[1, 2, 3, 4, 5],
                                      labels=["Unhappy", "Somewhat unhappy", "Neutral", "Somewhat happy", "Happy"])

    distance_matrix_family = get_distance_matrix(seqdata=family_sequence,
                                                 method='OM',
                                                 sm="CONSTANT",
                                                 indel=1)

    distance_matrix_happiness = get_distance_matrix(seqdata=happiness_sequence,
                                                    method='OM',
                                                    sm="CONSTANT",
                                                    indel=1)

    # ---------------------------

    # MDS 处理
    X_family, eigvals_family = cmdscale(distance_matrix_family)

    # 计算解释方差
    explained_ratio = eigvals_family / eigvals_family.sum()
    cum_explained = np.cumsum(explained_ratio)

    # 自动选择维度
    result_family = find_optimal_mds_dimension(
        cum_explained,
        title='Family MDS',
        var_threshold=0.90,
        max_dim=100
    )

    optimal_dim_family = result_family['recommended']
    print("最终推荐维度（Family）：", 154)

    # 获取最终嵌入矩阵
    X_family_optimal = X_family[:, :optimal_dim_family]

    # ----------
    # MDS 处理
    X_family, eigvals_family = cmdscale(distance_matrix_happiness)

    # 计算解释方差
    explained_ratio = eigvals_family / eigvals_family.sum()
    cum_explained = np.cumsum(explained_ratio)

    # 自动选择维度
    result_happiness = find_optimal_mds_dimension(
        cum_explained,
        title='Happiness MDS',
        var_threshold=0.90,
        max_dim=100
    )

    optimal_dim_happiness = result_family['recommended']
    print("最终推荐维度（Happiness）：", optimal_dim_happiness)

    # 获取最终嵌入矩阵
    X_happiness_optimal = X_family[:, :138]

    X_concat = np.hstack([X_family_optimal, X_happiness_optimal])

    # ----------------------------

    clusterer = hdbscan.HDBSCAN(min_cluster_size=30)
    cluster_labels = clusterer.fit_predict(X_concat)

    # 看一下聚了几个 cluster
    print(np.unique(cluster_labels, return_counts=True))

    # 构建 DataFrame 用于 plot
    hdbscan_df = pd.DataFrame({
        "ID": family_df["ID"],
        "Cluster ID": cluster_labels
    })

    # ---------------------------

    # 可视化 family sequence 按 cluster
    plot_sequence_index(seqdata=family_sequence,
                        id_group_df=hdbscan_df,
                        categories='Cluster ID',
                        save_as='hdbscan_family_plot')

    # 可视化 happiness sequence 按 cluster
    plot_sequence_index(seqdata=happiness_sequence,
                        id_group_df=hdbscan_df,
                        categories='Cluster ID',
                        save_as='hdbscan_happiness_plot')
