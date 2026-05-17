"""
@Author  : 梁彧祺 Yuqi Liang
@File    : __init__.py
@Time    : 06/05/2026 08:30
@Desc    :
    Clustering and typology for hierarchical (relational) sequence analysis.

    Submodules: ``pam`` (full matrix), ``clara`` (scalable), ``typology`` (user API).
"""

from .aggregate import aggregate_distance_matrix_by_level
from .clara import cluster_pair_typology_clara
from .pam import (
    cluster_level_1_profiles,
    cluster_level_2_profiles,
    cluster_pair_sequences,
)
from .results import HierarchicalClusterResult, PairTypologyResult
from .typology import cluster_pair_trajectories

__all__ = [
    "HierarchicalClusterResult",
    "PairTypologyResult",
    "aggregate_distance_matrix_by_level",
    "cluster_pair_sequences",
    "cluster_level_1_profiles",
    "cluster_level_2_profiles",
    "cluster_pair_typology_clara",
    "cluster_pair_trajectories",
]
