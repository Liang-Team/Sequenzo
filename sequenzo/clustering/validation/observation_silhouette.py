"""
@Author  : Yuqi Liang 梁彧祺
@File    : observation_silhouette.py
@Time    : 11/05/2025 10:01
@Desc    : 
Per-observation silhouette widths for weighted partitions.

Mirrors WeightedCluster ``wcSilhouetteObs``.
"""
from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np

from .partition_quality import _import_cpp, _labels_to_one_based, _prepare_weights

Measure = Literal["ASW", "ASWw"]


def observation_silhouette(
    diss: np.ndarray,
    clustering: np.ndarray,
    weights: Optional[np.ndarray] = None,
    measure: Measure = "ASW",
) -> np.ndarray:
    """
    Compute per-observation silhouette widths on a fixed partition.

    Parameters
    ----------
    diss
        Square symmetric distance matrix.
    clustering
        Cluster membership labels for each observation.
    weights
        Optional observation weights.
    measure
        ``"ASW"`` or ``"ASWw"`` (WeightedCluster naming).

    Returns
    -------
    np.ndarray
        One silhouette value per observation.
    """
    if measure not in ("ASW", "ASWw"):
        raise ValueError("measure must be 'ASW' or 'ASWw'.")

    cpp = _import_cpp()
    diss = np.asarray(diss, dtype=np.float64, order="C")
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square distance matrix.")

    n = diss.shape[0]
    clustering = np.asarray(clustering).reshape(-1)
    if clustering.shape[0] != n:
        raise ValueError("clustering must have one label per observation.")

    weights = _prepare_weights(weights, n)
    cluster_ids = _labels_to_one_based(clustering)
    n_clusters = int(np.max(cluster_ids))
    if n_clusters < 2:
        raise ValueError("clustering must contain at least two distinct groups.")

    selected_measure: Measure = measure
    if measure == "ASW":
        cluster_weights = np.bincount(cluster_ids - 1, weights=weights, minlength=n_clusters)
        if np.any(cluster_weights < 1.0):
            warnings.warn(
                "ASW cannot be computed because at least one cluster has less than "
                "one weighted observation. Returning ASWw instead.",
                UserWarning,
                stacklevel=2,
            )
            selected_measure = "ASWw"

    result = cpp.individual_asw(diss, cluster_ids, weights, n_clusters)
    measure_key = "asw_individual" if selected_measure == "ASW" else "asw_weighted"
    return np.asarray(result[measure_key], dtype=np.float64).reshape(-1)
