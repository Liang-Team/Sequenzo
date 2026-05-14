"""
@Author  : Yuqi Liang 梁彧祺
@File    : quality.py
@Time    : 13/05/2026 19:11
@Desc    :
Cluster-quality evaluation for property-based clustering trees.

Mirrors WeightedCluster ``as.clustrange`` methods for ``dtclust`` objects.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sequenzo.clustering.validation.partition_quality import (
    ClusterRangeResult,
    cluster_range_from_partitions,
)

from .tree_schedule import cut_tree


def property_clustering_quality(
    tree: Dict[str, Any],
    diss: np.ndarray,
    n_clusters: int = 20,
    weights: Optional[np.ndarray] = None,
    labels: bool = True,
) -> ClusterRangeResult:
    """
    Evaluate cluster-quality indicators for nested partitions of a property tree.

    Mirrors ``as.clustrange(pclust, diss=diss, ncluster=...)``.

    Parameters
    ----------
    tree : dict
        Scheduled property-based clustering tree.
    diss : np.ndarray
        Distance matrix used to build the tree.
    n_clusters : int, default 20
        Maximum number of groups to evaluate (must be >= 3 and <= number of leaves).
    weights : np.ndarray, optional
        Observation weights.
    labels : bool, default True
        If True, partitions use human-readable tree labels.

    Returns
    -------
    ClusterRangeResult
    """
    if n_clusters < 3:
        raise ValueError("n_clusters should be greater than 2.")

    fitted = tree["fitted"]["(fitted)"].to_numpy()
    max_k = len(np.unique(fitted))
    if n_clusters > max_k:
        raise ValueError(f"n_clusters should be less than {max_k + 1}.")

    partitions: Dict[str, pd.Series] = {}
    for k in range(2, n_clusters + 1):
        partitions[f"Split{k}"] = cut_tree(tree, n_clusters=k, labels=labels)
    partition_df = pd.DataFrame(partitions)
    return cluster_range_from_partitions(diss=diss, clustering=partition_df, weights=weights)


# WeightedCluster alias
as_clustrange_property_tree = property_clustering_quality
