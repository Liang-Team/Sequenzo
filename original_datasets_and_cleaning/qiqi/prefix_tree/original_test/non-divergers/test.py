"""
@Author  : Yuqi Liang 梁彧祺
@File    : test.py
@Time    : 02/05/2025 13:46
@Desc    : 
"""
from sequenzo import *

import pandas as pd
import numpy as np
import sys

df = pd.read_csv('/sequenzo/prefix_tree/df.csv')

# ================================
# Step 1: 筛选主流个体（non-diverged）
# ================================
mainstream_df = df[df["diverged"] == 0].copy()

# ================================
# Step 2: 构建 SequenceData 对象
# ================================
time_cols = [f"C{i}" for i in range(1, 11)]
states = sorted(mainstream_df[time_cols].stack().dropna().unique().tolist())

seq_obj = SequenceData(
    data=mainstream_df,
    time_type="year",
    time=time_cols,
    states=states,
    id_col="worker_id"
)

# ================================
# Step 3: 获取距离矩阵（用 OMspell）
# ================================
sys.setrecursionlimit(25000)

detailed_distance_matrix = get_distance_matrix(
    seqdata=seq_obj,
    method="OMspell",
    sm="CONSTANT",  # 因为你说转换太少
    indel=1,
    expcost=1
)

# ================================
# Step 4: 聚类 + 质量分析
# ================================
detailed_cluster = Cluster(
    detailed_distance_matrix,
    entity_ids=mainstream_df["worker_id"],
    clustering_method='ward'
)

detailed_cluster.plot_dendrogram(
    xlabel="Graduates",
    ylabel="Distance"
)

# 评估聚类质量
cluster_quality = ClusterQuality(detailed_cluster)
cluster_quality.compute_cluster_quality_scores()
cluster_quality.plot_combined_scores(norm='zscore', save_as='mainstream_combined_scores.png')
summary_table = cluster_quality.get_metrics_table()
print(summary_table)

# ================================
# Step 5: 获取 Cluster 结果
# ================================
cluster_results = ClusterResults(detailed_cluster)

membership_table = cluster_results.get_cluster_memberships(num_clusters=6)
print(membership_table.head())
membership_table.to_csv("mainstream_cluster_6_membership_table.csv", index=False)

distribution = cluster_results.get_cluster_distribution(num_clusters=6)
print(distribution)
cluster_results.plot_cluster_distribution(
    num_clusters=6,
    save_as="mainstream_cluster_distribution.png"
)

# ================================
# Step 6: 按 cluster 可视化序列 index plot
# ================================
plot_sequence_index(
    seqdata=seq_obj,
    id_group_df=membership_table,
    categories="Cluster ID",
    save_as="mainstream_cluster_index_plot_6.png",
    dpi=300
)
