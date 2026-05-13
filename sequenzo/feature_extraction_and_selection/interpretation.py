"""
@Author  : Yuqi Liang 梁彧祺
@File    : interpretation.py
@Time    : 20/03/2026 11:03
@Desc    :
    Interpretation utilities for selected features (Unterlerchner et al. 2023 workflow).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def interpret_selected_features(selection_result: Dict[str, Any]) -> pd.DataFrame:
    names = selection_result.get("selected_feature_names", [])
    indices = selection_result.get("selected_indices", [])
    if not indices and names:
        mask = selection_result.get("selected_mask")
        if mask is not None:
            indices = [i for i, keep in enumerate(mask) if keep]
    hit_counts = selection_result.get("hit_counts")

    if hit_counts is not None and len(indices) > 0:
        selected_hits = [int(hit_counts[i]) for i in indices]
    else:
        selected_hits = [None] * len(names if names else indices)

    feature_list = names if names else [f"X{i+1}" for i in indices]
    return pd.DataFrame(
        {
            "feature": feature_list,
            "index": indices,
            "hit_count": selected_hits,
        }
    ).sort_values(["hit_count", "feature"], ascending=[False, True], na_position="last")


def cluster_correlated_features(
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Sequence[str],
    *,
    abs_corr_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Group selected features by hierarchical clustering on ``1 - |corr|``.

    Features with absolute Pearson correlation at or above ``abs_corr_threshold``
    are merged into the same cluster (average-linkage, distance criterion).
    """
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=float)
        names = list(X.columns)
    else:
        arr = np.asarray(X, dtype=float)
        names = list(feature_names)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError("X columns must match feature_names length.")

    p = arr.shape[1]
    if p == 0:
        return pd.DataFrame(
            columns=[
                "feature",
                "cluster_id",
                "mean_abs_corr_with_cluster",
                "representative_feature",
            ]
        )

    if p == 1:
        return pd.DataFrame(
            {
                "feature": [names[0]],
                "cluster_id": [0],
                "mean_abs_corr_with_cluster": [0.0],
                "representative_feature": [names[0]],
            }
        )

    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    abs_corr = np.abs(corr)

    dist = 1.0 - abs_corr
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1.0 - abs_corr_threshold, criterion="distance") - 1

    representatives: Dict[int, str] = {}
    unique_clusters = sorted(set(int(c) for c in cluster_ids))
    for cid in unique_clusters:
        members = np.where(cluster_ids == cid)[0]
        if members.size == 1:
            representatives[cid] = names[int(members[0])]
            continue
        sub = abs_corr[np.ix_(members, members)]
        mean_link = sub.mean(axis=1)
        rep_idx = int(members[int(np.argmax(mean_link))])
        representatives[cid] = names[rep_idx]

    rows: List[Dict[str, Any]] = []
    for i, name in enumerate(names):
        cid = int(cluster_ids[i])
        members = np.where(cluster_ids == cid)[0]
        if members.size > 1:
            mean_abs = float(abs_corr[i, members].mean())
        else:
            mean_abs = 0.0
        rows.append(
            {
                "feature": name,
                "cluster_id": cid,
                "mean_abs_corr_with_cluster": mean_abs,
                "representative_feature": representatives[cid],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["cluster_id", "mean_abs_corr_with_cluster", "feature"],
        ascending=[True, False, True],
    )


__all__ = ["interpret_selected_features", "cluster_correlated_features"]
