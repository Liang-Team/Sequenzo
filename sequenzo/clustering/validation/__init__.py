"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 07/05/2025 09:10
@Desc    : 
Clustering validation and typology-regression diagnostics.

WeightedCluster counterparts: ``as.clustrange``, ``bootclustrange``,
``clustassoc``, and ``rarcat``.
"""
from .bootstrap_cluster_range import BootClusterRangeResult, boot_cluster_range
from .cluster_covariate_association import cluster_association
from .dissmfacw_factors import dissmfacw_table
from .observation_silhouette import observation_silhouette
from .partition_quality import (
    METRIC_ORDER,
    ClusterRangeResult,
    cluster_range_from_partitions,
    compute_partition_quality,
)
from .rarcat_typology_regression import RarcatResult, rarcat

__all__ = [
    "METRIC_ORDER",
    "ClusterRangeResult",
    "cluster_range_from_partitions",
    "compute_partition_quality",
    "dissmfacw_table",
    "observation_silhouette",
    "BootClusterRangeResult",
    "boot_cluster_range",
    "cluster_association",
    "RarcatResult",
    "rarcat",
]
