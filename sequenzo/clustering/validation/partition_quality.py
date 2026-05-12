"""
@Author  : Yuqi Liang 梁彧祺
@File    : partition_quality.py
@Time    : 08/05/2025 15:45
@Desc    : 
Utilities for evaluating clustering quality on fixed partitions.

Mirrors WeightedCluster ``as.clustrange.default`` when the clustering columns are
already known (for example bootstrap resamples in ``bootclustrange``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

METRIC_ORDER: List[str] = [
    "PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC",
]


def _import_cpp():
    try:
        from sequenzo.clustering import clustering_c_code as cpp
        return cpp
    except ImportError as exc:
        raise RuntimeError(
            "The clustering C++ extension is required for cluster-range quality "
            "evaluation. Rebuild the package before using this function."
        ) from exc


def _prepare_weights(weights: Optional[np.ndarray], n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != n:
        raise ValueError("weights must have one value per observation.")
    return weights


def _to_clustering_frame(clustering: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(clustering, pd.DataFrame):
        frame = clustering.copy()
    else:
        clustering = np.asarray(clustering)
        if clustering.ndim == 1:
            clustering = clustering.reshape(-1, 1)
        if clustering.ndim != 2:
            raise ValueError("clustering must be a 1D or 2D array-like object.")
        frame = pd.DataFrame(clustering)
    if frame.shape[0] == 0:
        raise ValueError("clustering must contain at least one observation.")
    return frame


def _labels_to_one_based(labels: np.ndarray) -> np.ndarray:
    """Convert arbitrary partition labels to contiguous 1-based cluster ids."""
    labels = np.asarray(labels)
    _, encoded = np.unique(labels, return_inverse=True)
    return (encoded + 1).astype(np.int32)


def compute_partition_quality(
    diss: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute WeightedCluster quality indicators for one clustering partition.

    Parameters
    ----------
    diss : np.ndarray
        Square symmetric distance matrix.
    labels : np.ndarray
        Cluster membership labels (any hashable values).
    weights : np.ndarray, optional
        Observation weights.

    Returns
    -------
    dict
        Mapping from metric name to raw quality value.
    """
    cpp = _import_cpp()
    diss = np.asarray(diss, dtype=np.float64, order="C")
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square distance matrix.")
    n = diss.shape[0]
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != n:
        raise ValueError("labels must have one value per observation.")
    weights = _prepare_weights(weights, n)
    cluster_ids = _labels_to_one_based(labels)
    n_clusters = int(np.max(cluster_ids))
    result = cpp.cluster_quality(diss, cluster_ids, weights, n_clusters)
    return {metric: float(result[metric]) for metric in METRIC_ORDER}


@dataclass
class ClusterRangeResult:
    """Container matching the core fields of R ``clustrange`` objects."""

    clustering: pd.DataFrame
    kvals: np.ndarray
    stats: pd.DataFrame
    boot: Optional[List[np.ndarray]] = None
    meant: Optional[pd.DataFrame] = None
    stderr: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "clustering": self.clustering,
            "kvals": self.kvals,
            "stats": self.stats,
        }
        if self.boot is not None:
            payload["boot"] = self.boot
        if self.meant is not None:
            payload["meant"] = self.meant
        if self.stderr is not None:
            payload["stderr"] = self.stderr
        return payload


def cluster_range_from_partitions(
    diss: np.ndarray,
    clustering: Union[np.ndarray, pd.DataFrame],
    weights: Optional[np.ndarray] = None,
) -> ClusterRangeResult:
    """
    Evaluate quality indicators for each column of a multi-k clustering table.

    This follows ``as.clustrange.default`` with ``R = 1``.
    """
    diss = np.asarray(diss, dtype=np.float64, order="C")
    frame = _to_clustering_frame(clustering)
    n = diss.shape[0]
    if frame.shape[0] != n:
        raise ValueError("clustering and diss must refer to the same observations.")
    weights = _prepare_weights(weights, n)

    kvals = np.empty(frame.shape[1], dtype=int)
    stats = np.empty((frame.shape[1], len(METRIC_ORDER)), dtype=np.float64)
    renamed_columns: List[str] = []

    for idx, column in enumerate(frame.columns):
        labels = frame.iloc[:, idx].to_numpy()
        kvals[idx] = len(np.unique(labels))
        stats[idx, :] = [compute_partition_quality(diss, labels, weights)[metric] for metric in METRIC_ORDER]
        renamed_columns.append(f"cluster{kvals[idx]}")

    clustering_out = frame.copy()
    clustering_out.columns = renamed_columns
    stats_df = pd.DataFrame(stats, index=renamed_columns, columns=METRIC_ORDER)
    stats_df.index.name = "Cluster"
    return ClusterRangeResult(clustering=clustering_out, kvals=kvals, stats=stats_df)
