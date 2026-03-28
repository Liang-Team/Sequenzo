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
    :return: float, 仅代表 ward_time 的耗时
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

    # metric = EditSequenceMetric(
    #     settings=EditSequenceMetricSettings(
    #         entity_metric="hamming",
    #         indel_cost=1.0,
    #         normalize=False,
    #     )
    # )

    # t1 = time.perf_counter()
    # _ = metric.compute_matrix(pool)
    # metric_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    ac = AgglomerativeClustering(
        n_clusters=num_clusters,
        linkage="ward",
        metric="euclidean",
    )
    ac.fit(np.asarray(X_features, dtype=np.float64))
    ward_time = time.perf_counter() - t2

    return ward_time


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


# =============================================================================
# 新版 fast_path 路径对比（含 condensed 输入）
# =============================================================================
def benchmark_fast_paths(
    sizes=None,
    seq_len=30,
    num_clusters=5,
    ward_variant="ward_d2",
    seed=42,
    repeats=5,
    run_tanat=True,
):
    """
    针对多个样本量，对比以下五种路径的耗时：
      1. fastcluster (baseline)
      2. Sequenzo full path  (fast_path=False, 方阵输入)
      3. Sequenzo fast path  (fast_path=True,  方阵输入)
      4. Sequenzo cond+fast  (fast_path=True,  condensed 1D 输入)
      5. Sequenzo cond,no fast (fast_path=False, condensed 1D 输入)

    计时方式：使用 timeit.repeat 而非手动 for 循环，取 min() 而非 median()。
    - timeit.repeat 在每次 repeat 前禁用 GC，避免 GC 暂停污染测量
    - min() 代表「无干扰的最快速度」，排除 OS 调度抖动、GC 等噪声
      （median 会把这些噪声保留进结果，导致系统性偏高）

    :param sizes:       要测试的样本量列表（默认 100~10000）
    :param seq_len:     随机数据的序列长度 / 特征维度
    :param num_clusters: 簇数
    :param ward_variant: Ward 变体（"ward_d"/"ward_d2"）
    :param seed:        随机种子
    :param repeats:     每个规模重复测量次数，取 min
    :param run_tanat:   是否测试 TanaT（需安装 tanat；False 则跳过）
    """
    import timeit
    from sequenzo.clustering.hierarchical_clustering import Cluster

    if sizes is None:
        sizes = [100, 200, 300, 500, 700, 1000, 1300, 2000, 3000, 5000, 10000]

    try:
        from sequenzo.clustering.sequenzo_fastcluster.fastcluster import linkage as _fc_linkage
    except ImportError:
        try:
            from fastcluster import linkage as _fc_linkage
        except ImportError:
            raise ImportError("需要 sequenzo_fastcluster 或 fastcluster")

    def _t(fn, repeats):
        """用 timeit.repeat 计时 fn()，返回最快一次的耗时（秒）。"""
        return min(timeit.repeat(fn, number=1, repeat=repeats))

    # 检测 TanaT 可用性
    _tanat_available = False
    if run_tanat:
        try:
            import tanat  # noqa: F401
            _tanat_available = True
        except ImportError:
            print("  [TanaT] tanat 未安装，TanaT 列将跳过。")

    width = 152 if _tanat_available else 135
    tanat_header = f"{'tanat(ward)':>14s}" if _tanat_available else ""
    print("=" * width)
    print(f"Fast-path benchmark  (ward_variant={ward_variant}, repeats={repeats}, seq_len={seq_len})")
    print(f"  计时方法: timeit.repeat(number=1, repeat={repeats}), 取 min")
    print(f"{'n':>7s}  {'fc(cond)':>10s}  {'fc(full)':>14s}  {'sqz_full':>14s}  "
          f"{'sqz_fast':>14s}  {'sqz_cond+fast':>16s}  {'sqz_cond(no fast)':>18s}  {tanat_header}")
    print("-" * width)

    results = []
    for n in sizes:
        np.random.seed(seed)
        X = np.random.randint(0, 10, (n, seq_len)).astype(np.float64)
        X_int = X.astype(np.int64)  # TanaT 需要整数型
        D = squareform(pdist(X, "euclidean"))
        condensed = pdist(X, "euclidean")
        entity_ids = np.arange(n)

        # --- fastcluster (condensed) ---
        t_fc = _t(
            lambda c=condensed: (
                lambda Z: fcluster(Z, t=num_clusters, criterion="maxclust")
            )(_fc_linkage(c.copy(), method=ward_variant)),
            repeats,
        )

        # --- fastcluster (full matrix) ---
        t_fc_full = _t(
            lambda m=D: (
                lambda Z: fcluster(Z, t=num_clusters, criterion="maxclust")
            )(_fc_linkage(squareform(m, checks=False), method=ward_variant)),
            repeats,
        )

        # --- Sequenzo full path ---
        t_full = _t(
            lambda: Cluster(D.copy(), entity_ids, clustering_method=ward_variant, fast_path=False),
            repeats,
        )

        # --- Sequenzo fast path (square matrix) ---
        t_fast = _t(
            lambda: Cluster(D.copy(), entity_ids, clustering_method=ward_variant, fast_path=True),
            repeats,
        )

        # --- Sequenzo fast path (condensed 1D) ---
        t_cond_fast = _t(
            lambda: Cluster(condensed.copy(), entity_ids, clustering_method=ward_variant, fast_path=True),
            repeats,
        )

        # --- Sequenzo condensed, no fast path (condensed 1D) ---
        t_cond_nf = _t(
            lambda: Cluster(condensed.copy(), entity_ids, clustering_method=ward_variant, fast_path=False),
            repeats,
        )

        results.append((t_fc, t_fc_full, t_full, t_fast, t_cond_fast, t_cond_nf))

        # --- TanaT (ward on X) ---
        t_tanat = None
        if _tanat_available:
            t_tanat = min(
                tanat_test_from_xfeatures(X_int, num_clusters=num_clusters, metric_name="edit")
                for _ in range(repeats)
            )

        results[-1] = (t_fc, t_fc_full, t_full, t_fast, t_cond_fast, t_cond_nf, t_tanat)

        tanat_col = (
            f"   {t_tanat:.4f}s ({t_tanat/t_fc:.2f}x)" if t_tanat is not None else ""
        )
        print(
            f"  n={n:6d}:  "
            f"{t_fc:.4f}s(c) "
            f"{t_fc_full:.4f}s ({t_fc_full/t_fc:.2f}x)   "
            f"{t_full:.4f}s ({t_full/t_fc:.2f}x)   "
            f"{t_fast:.4f}s ({t_fast/t_fc:.2f}x)   "
            f"{t_cond_fast:.4f}s ({t_cond_fast/t_fc:.2f}x)   "
            f"{t_cond_nf:.4f}s ({t_cond_nf/t_fc:.2f}x){tanat_col}"
        )

    print("-" * width)
    print("\nSummary — ratio to fastcluster (< 1.0 = faster than fastcluster):")
    tanat_hdr = f"  {'tanat':>10s}" if _tanat_available else ""
    print(f"{'n':>7s}  {'fc(full)':>10s}  {'full':>8s}  {'fast(sq)':>10s}  {'fast(cond)':>12s}  {'nf(cond)':>10s}{tanat_hdr}")
    for n, row in zip(sizes, results):
        t_fc, t_fc_full, t_full, t_fast, t_cond_fast, t_cond_nf, t_tanat = row
        tanat_cell = f"  {t_tanat/t_fc:10.2f}x" if t_tanat is not None else ""
        print(
            f"  {n:6d}  {t_fc_full/t_fc:10.2f}x  {t_full/t_fc:8.2f}x  {t_fast/t_fc:10.2f}x  "
            f"{t_cond_fast/t_fc:12.2f}x  {t_cond_nf/t_fc:10.2f}x{tanat_cell}"
        )

    print("\nSummary — absolute times in seconds (for plotting):")
    tanat_hdr2 = f"  {'tanat(s)':>12s}" if _tanat_available else ""
    print(
        f"{'n':>7s}  {'fc(cond)(s)':>13s}  {'fc(full)(s)':>13s}  {'full(s)':>10s}  "
        f"{'fast(sq)(s)':>12s}  {'fast(cond)(s)':>14s}  {'nf(cond)(s)':>13s}{tanat_hdr2}"
    )
    for n, row in zip(sizes, results):
        t_fc, t_fc_full, t_full, t_fast, t_cond_fast, t_cond_nf, t_tanat = row
        tanat_cell2 = f"  {t_tanat:12.4f}" if t_tanat is not None else ""
        print(
            f"  {n:6d}  {t_fc:13.4f}  {t_fc_full:13.4f}  {t_full:10.4f}  "
            f"{t_fast:12.4f}  {t_cond_fast:14.4f}  {t_cond_nf:13.4f}{tanat_cell2}"
        )
    print("=" * width)


if __name__ == "__main__":
    import time

    # ------------------------------------------------------------------
    # fast_path 路径对比（多规模，含 condensed 输入 + TanaT 竞品）
    # ------------------------------------------------------------------
    benchmark_fast_paths(
        sizes=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
        seq_len=30,
        num_clusters=5,
        ward_variant="ward_d2",
        seed=42,
        repeats=5,
        run_tanat=True,
    )

    # benchmark_fast_paths(
    #     sizes=[10000, 15000, 20000, 25000, 30000],
    #     # sizes=[10000],
    #     seq_len=30,
    #     num_clusters=5,
    #     ward_variant="ward_d2",
    #     seed=42,
    #     repeats=5,
    #     run_tanat=True,
    # )