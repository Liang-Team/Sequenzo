"""
@Author  : 梁彧祺 Yuqi Liang
@File    : pam.py
@Time    : 13/04/2026 20:17
@Desc    :
    Full-matrix PAM / K-medoids clustering at pair, level-1, or level-2 resolution.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from sequenzo.clustering.k_medoids import KMedoids
from sequenzo.clustering.sequences_to_variables.helpers import (
    cluster_labels_from_kmedoids_result,
)

from ..data import RelationalSequenceData
from ..distances import RelationalDistanceMatrix
from ..profiles import summarize_level_1_profiles, summarize_level_2_profiles

from .aggregate import aggregate_distance_matrix_by_level
from .results import HierarchicalClusterResult


def _run_kmedoids_on_matrix(
    matrix: np.ndarray,
    k: int,
    *,
    method: str = "PAMonce",
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> HierarchicalClusterResult:
    if k < 1:
        raise ValueError("k must be at least 1.")
    n = matrix.shape[0]
    if k >= n:
        labels = np.arange(n, dtype=int)
        return HierarchicalClusterResult(
            level="",
            cluster_labels=labels,
            unit_ids=np.arange(n),
            k=k,
            medoid_indices=np.arange(min(k, n)),
            distance_matrix=matrix,
            method=method,
            details={"note": "k >= n; each unit assigned its own cluster"},
        )

    medoids_1based = KMedoids(
        matrix,
        k=k,
        method=method,
        verbose=verbose,
        random_state=random_state,
    )
    labels = cluster_labels_from_kmedoids_result(medoids_1based, input_base=1)
    medoids_0 = medoids_1based.astype(int) - 1
    unique_medoids = np.unique(medoids_0)

    return HierarchicalClusterResult(
        level="",
        cluster_labels=labels,
        unit_ids=np.arange(n),
        k=len(unique_medoids),
        medoid_indices=unique_medoids,
        distance_matrix=matrix,
        method=method,
        details={"medoids_1based": medoids_1based},
    )


def cluster_pair_sequences(
    distance_matrix: RelationalDistanceMatrix,
    k: int,
    *,
    method: str = "PAMonce",
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> HierarchicalClusterResult:
    """Cluster all pair-level trajectories from a full distance matrix."""
    result = _run_kmedoids_on_matrix(
        distance_matrix.matrix,
        k,
        method=method,
        random_state=random_state,
        verbose=verbose,
    )
    result.level = "pair"
    result.unit_ids = distance_matrix.pair_ids
    return result


def cluster_level_1_profiles(
    sequence_data: RelationalSequenceData,
    distance_matrix: RelationalDistanceMatrix,
    k: int,
    *,
    method: str = "PAMonce",
    random_state: Optional[int] = None,
    verbose: bool = False,
    profile_features: Optional[List[str]] = None,
) -> HierarchicalClusterResult:
    """Cluster level-1 units by aggregated pair-distance portfolios."""
    agg, unit_ids = aggregate_distance_matrix_by_level(distance_matrix, level=1)

    if profile_features:
        profiles = summarize_level_1_profiles(sequence_data, distance_matrix)
        profiles = profiles.set_index("level_1_id").loc[unit_ids]
        feat = profiles[profile_features].to_numpy(dtype=float)
        feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0, ddof=1) + 1e-12)
        from scipy.spatial.distance import pdist

        feat_dist = squareform(pdist(feat, metric="euclidean"))
        matrix = 0.5 * agg + 0.5 * feat_dist
    else:
        matrix = agg

    result = _run_kmedoids_on_matrix(
        matrix, k, method=method, random_state=random_state, verbose=verbose
    )
    result.level = "level_1"
    result.unit_ids = unit_ids
    result.details["aggregated_distance_matrix"] = agg
    return result


def cluster_level_2_profiles(
    sequence_data: RelationalSequenceData,
    distance_matrix: RelationalDistanceMatrix,
    k: int,
    *,
    method: str = "PAMonce",
    random_state: Optional[int] = None,
    verbose: bool = False,
    profile_features: Optional[List[str]] = None,
) -> HierarchicalClusterResult:
    """Cluster level-2 units by cross-level-1 trajectory patterns."""
    agg, unit_ids = aggregate_distance_matrix_by_level(distance_matrix, level=2)

    if profile_features:
        profiles = summarize_level_2_profiles(sequence_data, distance_matrix)
        profiles = profiles.set_index("level_2_id").loc[unit_ids]
        feat = profiles[profile_features].to_numpy(dtype=float)
        feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0, ddof=1) + 1e-12)
        from scipy.spatial.distance import pdist

        feat_dist = squareform(pdist(feat, metric="euclidean"))
        matrix = 0.5 * agg + 0.5 * feat_dist
    else:
        matrix = agg

    result = _run_kmedoids_on_matrix(
        matrix, k, method=method, random_state=random_state, verbose=verbose
    )
    result.level = "level_2"
    result.unit_ids = unit_ids
    result.details["aggregated_distance_matrix"] = agg
    return result
