"""
仅获取 Ward 聚类簇划分的多种实现（不计算完整 linkage、树、CQI 等）

用于测试：若只关心簇标签，不同实现的耗时差异；
以及 Sequenzo 完整版（含 linkage）与最小版的对比。

实现：fastcluster、scipy、sklearn、Sequenzo（含 linkage / X_features / fast_path）

注意：
- fastcluster 和 scipy 的层次聚类算法本质上必须完成 n-1 次合并才能得到任意 k 的划分，
  因此无法避免 linkage 计算，只是这里不保留 linkage 作后续用途（树、CQI 等）。
- sklearn（与 TanaT 一致）：对原始特征矩阵直接 Ward，无需 MDS。需提供 X_features。
  compute_full_tree=False 可在得到 k 个簇后提前终止。
"""

import numpy as np
import time
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage


def _prepare_matrix(matrix):
    """清洗并准备距离矩阵（对称化、condensed 形式）"""
    m = np.asarray(matrix, dtype=np.float64)
    np.fill_diagonal(m, 0.0)
    if not np.allclose(m, m.T, rtol=1e-5, atol=1e-8):
        m = (m + m.T) / 2
    return squareform(m)


# =============================================================================
# 版本 1: fastcluster
# =============================================================================
def ward_labels_fastcluster(distance_matrix, num_clusters, ward_variant="ward_d2"):
    """
    使用 fastcluster 获取 Ward 簇标签。
    
    即使只需求簇划分，fastcluster 仍须计算完整 linkage（层次聚类算法的本质）。
    此处不保留 linkage 作他用，仅用 fcluster 切割得到标签。
    
    :param distance_matrix: 方形距离矩阵 (n x n)
    :param num_clusters: 簇数量 k
    :param ward_variant: "ward_d" 或 "ward_d2"
    :return: 耗时(秒)
    """
    condensed = _prepare_matrix(distance_matrix)
    
    try:
        from sequenzo.clustering.sequenzo_fastcluster.fastcluster import linkage
    except ImportError:
        try:
            from fastcluster import linkage
            if ward_variant == "ward_d2":
                # 标准 fastcluster 可能无 ward_d2，回退到 ward
                method = "ward"
        except ImportError:
            raise ImportError("需要 sequenzo_fastcluster 或 fastcluster")
    
    method = "ward" if ward_variant == "ward_d" else "ward_d2"
    t0 = time.perf_counter()
    Z = linkage(condensed, method=method)
    if ward_variant == "ward_d":
        Z = Z.copy()
        Z[:, 2] = Z[:, 2] / 2.0
    labels = fcluster(Z, t=num_clusters, criterion="maxclust")
    elapsed = time.perf_counter() - t0
    return elapsed


# =============================================================================
# 版本 2: scipy
# =============================================================================
def ward_labels_scipy(distance_matrix, num_clusters):
    """
    使用 scipy.cluster.hierarchy 获取 Ward 簇标签。
    
    scipy 的 'ward' 方法基于方差最小化（Ward D2 风格），与 ward_d2 接近。
    同样必须计算完整 linkage，此处仅用于 fcluster 切割。
    
    :param distance_matrix: 方形距离矩阵 (n x n)
    :param num_clusters: 簇数量 k
    :return: 耗时(秒)
    """
    condensed = _prepare_matrix(distance_matrix)
    t0 = time.perf_counter()
    Z = scipy_linkage(condensed, method="ward")
    labels = fcluster(Z, t=num_clusters, criterion="maxclust")
    elapsed = time.perf_counter() - t0
    return elapsed


# =============================================================================
# 版本 3: Sequenzo 完整版（含 linkage matrix）
# =============================================================================
def ward_labels_sequenzo_full(
    distance_matrix,
    num_clusters,
    entity_ids=None,
    ward_variant="ward_d2",
    fast_path=False,
):
    """
    使用 Sequenzo 的 Cluster 类，完整计算 linkage matrix 后获取簇标签。
    
    与「仅标签」版本对比，用于衡量 linkage 存储及后续可扩展性（树、CQI 等）的开销。
    
    :param distance_matrix: 方形距离矩阵 (n x n)
    :param num_clusters: 簇数量 k
    :param entity_ids: 可选，与矩阵行对应的实体 ID 列表；若 None，使用 0..n-1
    :param ward_variant: "ward_d" 或 "ward_d2"
    :param fast_path: 是否启用 Sequenzo fast_path（减少准备阶段开销）
    :return: 耗时(秒)
    """
    from sequenzo.clustering.hierarchical_clustering import Cluster
    
    n = distance_matrix.shape[0]
    if entity_ids is None:
        entity_ids = np.arange(n)
    t0 = time.perf_counter()
    cluster = Cluster(
        distance_matrix,
        entity_ids,
        clustering_method=ward_variant,
        fast_path=bool(fast_path),
    )
    # _ = cluster.get_cluster_labels(num_clusters)
    elapsed = time.perf_counter() - t0
    return elapsed


# =============================================================================
# 版本 4: Sequenzo（with X_features，向量路径）
# =============================================================================
def ward_labels_sequenzo_xfeatures(
    X_features,
    num_clusters,
    entity_ids=None,
    ward_variant="ward_d2",
    fast_path=False,
):
    """
    使用 Sequenzo 的 Cluster(X_features=...) 路径获取 Ward 簇标签。

    与 sklearn/TanaT 的「直接对原始特征做 Ward」思路一致；
    Sequenzo 内部使用 linkage_vector 的向量路径（更省内存）。

    :param X_features: 特征矩阵 (n x d)
    :param num_clusters: 簇数量 k
    :param entity_ids: 可选实体 ID；若 None，使用 0..n-1
    :param ward_variant: "ward_d" 或 "ward_d2"
    :param fast_path: 是否启用 Sequenzo fast_path（主要影响 matrix 预处理链路）
    :return: 耗时(秒)
    """
    from sequenzo.clustering.hierarchical_clustering import Cluster

    X = np.asarray(X_features, dtype=np.float64)
    n = X.shape[0]
    if entity_ids is None:
        entity_ids = np.arange(n)

    t0 = time.perf_counter()
    cluster = Cluster(
        matrix=None,
        entity_ids=entity_ids,
        clustering_method=ward_variant,
        X_features=X,
        fast_path=bool(fast_path),
    )
    _ = cluster.get_cluster_labels(num_clusters)
    elapsed = time.perf_counter() - t0
    return elapsed


# =============================================================================
# 版本 5: sklearn（与 TanaT 一致：直接 Ward 于原始特征，无需 MDS）
# =============================================================================
def ward_labels_sklearn(X_features, num_clusters, compute_full_tree=False):
    """
    使用 sklearn 获取 Ward 簇标签（与 run_tanat.py use_ward_from_features=True 一致）。
    
    直接对原始特征矩阵做 Ward 聚类，无需 MDS 嵌入。
    
    :param X_features: 特征矩阵 (n x d)，如序列网格 pdata（每行=一条序列在各时间点的状态）
    :param num_clusters: 簇数量 k
    :param compute_full_tree: 若 False，得到 k 个簇后提前终止
    :return: 耗时(秒)
    """
    from sklearn.cluster import AgglomerativeClustering
    
    X = np.asarray(X_features, dtype=np.float64)
    t0 = time.perf_counter()
    ac = AgglomerativeClustering(
        n_clusters=num_clusters,
        linkage="ward",
        metric="euclidean",
        compute_full_tree=compute_full_tree,
    )
    ac.fit(X)
    elapsed = time.perf_counter() - t0
    return elapsed


# =============================================================================
# 版本 6: TanaT（测试：时间网格 + OM 距离计算 + Ward on X）
# =============================================================================
def tanat_test_from_xfeatures(X_features, num_clusters, metric_name="edit"):
    """
    对 TanaT 进行最小测试：
    1) 将 X_features 视为时间网格序列，构建 TanaT 的 StateSequencePool
    2) 计算 TanaT 距离矩阵（如 OM/edit）
    3) 参考 run_tanat.py，用 sklearn 在 X 上执行 Ward 聚类

    :param X_features: 特征矩阵 (n x d)，每行一条序列、每列一个时间点状态
    :param num_clusters: 簇数量 k
    :param metric_name: 距离度量名称，目前支持 "edit"
    :return: dict, 包含 prep_time / metric_time / ward_time / total_time
    """
    from sklearn.cluster import AgglomerativeClustering

    try:
        import pandas as pd
        from tanat.sequence import StateSequencePool, StateSequenceSettings
        from tanat.metric.sequence import EditSequenceMetric, EditSequenceMetricSettings
    except ImportError as e:
        raise ImportError("需要安装 tanat（以及其依赖）才能运行 TanaT 测试。") from e

    X = np.asarray(X_features, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError("X_features 必须是 2D 矩阵 (n x d)。")

    n, seq_len = X.shape
    id_list = np.arange(n, dtype=np.int64)

    t0 = time.perf_counter()
    time_points = np.arange(1, seq_len + 1, dtype=np.int64)

    rows_grid = []
    for i in id_list:
        for t_idx, t in enumerate(time_points):
            rows_grid.append(
                {
                    "id": int(i),
                    "stime": int(t),
                    "etime": int(t),
                    "event": int(X[i, t_idx]),
                }
            )
    data_grid = pd.DataFrame(rows_grid)

    sdata = pd.DataFrame({"id": id_list, "c": np.zeros(n, dtype=np.int64)})

    settings = StateSequenceSettings(
        id_column="id",
        start_column="stime",
        end_column="etime",
        entity_features=["event"],
        static_features=["c"],
    )
    pool = StateSequencePool(data_grid, static_data=sdata, settings=settings)
    pool.update_entity_metadata(feature_name="event", feature_type="categorical")
    prep_time = time.perf_counter() - t0

    if metric_name != "edit":
        raise ValueError("当前 tanat_test_from_xfeatures 仅支持 metric_name='edit'。")

    metric = EditSequenceMetric(
        settings=EditSequenceMetricSettings(
            entity_metric="hamming",
            indel_cost=1.0,
            normalize=False,
        )
    )

    t1 = time.perf_counter()
    _ = metric.compute_matrix(pool)
    metric_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    ac = AgglomerativeClustering(
        n_clusters=num_clusters,
        linkage="ward",
        metric="euclidean",
    )
    ac.fit(np.asarray(X_features, dtype=np.float64))
    ward_time = time.perf_counter() - t2

    return {
        "prep_time": prep_time,
        "metric_time": metric_time,
        "ward_time": ward_time,
        "total_time": prep_time + metric_time + ward_time,
    }


# =============================================================================
# 简单耗时对比
# =============================================================================
def benchmark_ward_labels(
    distance_matrix,
    num_clusters=5,
    ward_variant="ward_d2",
    X_features=None,
    run_tanat_test=True,
):
    """
    对多种实现进行耗时对比。
    
    :param distance_matrix: 方形距离矩阵 (n x n)
    :param num_clusters: 簇数量
    :param ward_variant: fastcluster/Sequenzo 使用的 Ward 变体
    :param X_features: 原始特征矩阵 (n x d)，sklearn/sequenzo-xfeatures/tanat 测试所需
    :param run_tanat_test: 是否运行 TanaT 测试（默认 False，避免大样本时耗时过长）
    """
    n = distance_matrix.shape[0]
    print(f"[Benchmark] n={n}, k={num_clusters}")
    print("-" * 55)
    
    # Sequenzo 完整版（含 linkage matrix）
    t_sqz_full = ward_labels_sequenzo_full(distance_matrix, num_clusters, ward_variant=ward_variant)
    print(f"  Sequenzo (full+linkage): {t_sqz_full:.4f}s")
    t_sqz_full_fast = ward_labels_sequenzo_full(
        distance_matrix,
        num_clusters,
        ward_variant=ward_variant,
        fast_path=True,
    )
    print(f"  Sequenzo (full+linkage+fast_path): {t_sqz_full_fast:.4f}s")
    
    # fastcluster（仅标签，不保留 linkage 作他用）
    t_fc = ward_labels_fastcluster(distance_matrix, num_clusters, ward_variant)
    print(f"  fastcluster (labels):   {t_fc:.4f}s")
    
    # scipy
    t_sp = ward_labels_scipy(distance_matrix, num_clusters)
    print(f"  scipy (labels):         {t_sp:.4f}s")
    
    # Sequenzo（with X_features，向量路径）
    # if X_features is not None and X_features.shape[0] == n:
    #     t_sqz_x = ward_labels_sequenzo_xfeatures(X_features, num_clusters, ward_variant=ward_variant)
    #     print(f"  Sequenzo (with X_features): {t_sqz_x:.4f}s")
    #     t_sqz_x_fast = ward_labels_sequenzo_xfeatures(
    #         X_features,
    #         num_clusters,
    #         ward_variant=ward_variant,
    #         fast_path=True,
    #     )
    #     print(f"  Sequenzo (with X_features+fast_path): {t_sqz_x_fast:.4f}s")
    # else:
    #     print(f"  Sequenzo (with X_features): 跳过（需提供 X_features，且 shape[0]==n）")
    #     print(f"  Sequenzo (with X_features+fast_path): 跳过（需提供 X_features，且 shape[0]==n）")

    # sklearn（与 TanaT 一致：直接 Ward 于原始特征，无需 MDS）
    if X_features is not None and X_features.shape[0] == n:
        t_sk = ward_labels_sklearn(X_features, num_clusters, compute_full_tree=False)
        print(f"  sklearn (ward on X):    {t_sk:.4f}s  [compute_full_tree=False]")
        t_sk_full = ward_labels_sklearn(X_features, num_clusters, compute_full_tree=True)
        print(f"  sklearn (ward on X):    {t_sk_full:.4f}s  [compute_full_tree=True]")
    else:
        print(f"  sklearn (ward on X):    跳过（需提供 X_features，且 shape[0]==n）")
    
    # TanaT 测试（需要 tanat 包）
    # if run_tanat_test and X_features is not None and X_features.shape[0] == n:
    #     try:
    #         t_tanat = tanat_test_from_xfeatures(X_features, num_clusters=num_clusters, metric_name="edit")
    #         print(
    #             "  TanaT (prep+metric+ward): "
    #             f"{t_tanat['total_time']:.4f}s  "
    #             f"[prep={t_tanat['prep_time']:.4f}s, metric={t_tanat['metric_time']:.4f}s, ward={t_tanat['ward_time']:.4f}s]"
    #         )
    #     except ImportError as e:
    #         print(f"  TanaT (prep+metric+ward): 跳过（{e}）")
    # elif run_tanat_test:
    #     print(f"  TanaT (prep+metric+ward): 跳过（需提供 X_features，且 shape[0]==n）")
    # else:
    #     print("  TanaT (prep+metric+ward): 关闭（run_tanat_test=False）")

    print("-" * 55)
    print("注意：Sequenzo full 含 linkage；fastcluster/scipy 仅做 linkage+fcluster。")
    print("Sequenzo(with X_features)/sklearn/TanaT(ward on X) 都是直接在原始特征上做 Ward。")


if __name__ == "__main__":
    # 示例 1：随机数据，X=原始特征，D=欧氏距离矩阵
    np.random.seed(42)
    n, seq_len = 10000, 30
    X = np.random.randint(0, 10, (n, seq_len)).astype(np.float64)
    D = squareform(pdist(X, "euclidean"))
    benchmark_ward_labels(D, num_clusters=10, ward_variant="ward_d2", X_features=X, run_tanat_test=True)
    
    # 示例 2：若已有 Sequenzo 的距离矩阵 om 和序列网格 pdata：
    # om = get_distance_matrix(sequence_data, method="OM", ...)
    # pdata = sequence_data 的时序表示 (n x seq_len)
    # benchmark_ward_labels(om, num_clusters=5, X_features=pdata.values, run_tanat_test=True)
