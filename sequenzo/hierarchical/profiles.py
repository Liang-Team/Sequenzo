"""
@Author  : 梁彧祺 Yuqi Liang
@File    : profiles.py
@Time    : 17/04/2026 22:14
@Desc    :
    Higher-level and pair-specific summaries for hierarchical sequence analysis.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from .data import RelationalSequenceData
from .distances import RelationalDistanceMatrix
from .residuals import compute_pair_residuals, detect_pair_specific_outliers

__all__ = [
    "summarize_level_1_profiles",
    "summarize_level_2_profiles",
    "compute_pair_residuals",
    "detect_pair_specific_outliers",
]


def _mean_upper_triangle_distances(
    matrix: np.ndarray,
    indices: np.ndarray,
) -> float:
    """Mean distance among pairs with both indices in ``indices`` (upper triangle)."""
    if len(indices) < 2:
        return np.nan
    sub = matrix[np.ix_(indices, indices)]
    iu = np.triu_indices(len(indices), k=1)
    if len(iu[0]) == 0:
        return 0.0
    return float(np.mean(sub[iu]))


def summarize_level_1_profiles(
    sequence_data: RelationalSequenceData,
    distance_matrix: RelationalDistanceMatrix,
    *,
    target_state: Any = None,
) -> pd.DataFrame:
    """
    Region-style (level-1) profiles: internal diversity and cross-unit distances.

    Parameters
    ----------
    sequence_data : RelationalSequenceData
    distance_matrix : RelationalDistanceMatrix
    target_state : optional
        If given, compute the share of sequences reaching this state and mean
        first time index at which it appears.
    """
    df = sequence_data.to_dataframe()
    matrix = distance_matrix.matrix
    level_1 = distance_matrix.level_1_ids
    unique_l1 = pd.unique(level_1)

    rows = []
    all_indices = np.arange(len(level_1))

    for unit in unique_l1:
        mask = level_1 == unit
        idx = np.where(mask)[0]
        internal = _mean_upper_triangle_distances(matrix, idx)

        other_idx = np.where(~mask)[0]
        if len(other_idx) == 0:
            avg_to_others = np.nan
        else:
            cross = matrix[np.ix_(idx, other_idx)]
            avg_to_others = float(np.mean(cross))

        target_share = np.nan
        mean_first_target_index = np.nan
        mean_first_target_time = np.nan
        if target_state is not None:
            reached = []
            indices = []
            times = []
            for rec in sequence_data.records:
                if rec.level_1_id != unit:
                    continue
                seq = rec.sequence
                if target_state in seq:
                    reached.append(1)
                    idx = seq.index(target_state)
                    indices.append(idx)
                    times.append(rec.time_points[idx])
                else:
                    reached.append(0)
            if reached:
                target_share = float(np.mean(reached))
                mean_first_target_index = float(np.mean(indices)) if indices else np.nan
                mean_first_target_time = float(np.mean(times)) if times else np.nan

        rows.append(
            {
                "level_1_id": unit,
                "n_pairs": int(mask.sum()),
                "internal_diversity": internal,
                "avg_distance_to_others": avg_to_others,
                "target_state_share": target_share,
                "mean_first_target_index": mean_first_target_index,
                "mean_first_target_time": mean_first_target_time,
            }
        )

    return pd.DataFrame(rows)


def summarize_level_2_profiles(
    sequence_data: RelationalSequenceData,
    distance_matrix: RelationalDistanceMatrix,
    *,
    target_state: Any = None,
) -> pd.DataFrame:
    """
    CPC-style (level-2) profiles: cross-region diversity and diffusion metrics.
    """
    matrix = distance_matrix.matrix
    level_2 = distance_matrix.level_2_ids
    unique_l2 = pd.unique(level_2)

    rows = []
    for unit in unique_l2:
        mask = level_2 == unit
        idx = np.where(mask)[0]
        cross_region_diversity = _mean_upper_triangle_distances(matrix, idx)

        diffusion_share = np.nan
        mean_first_target_index = np.nan
        mean_first_target_time = np.nan
        if target_state is not None:
            indices = []
            times = []
            regions_seen = set()
            for rec in sequence_data.records:
                if rec.level_2_id != unit:
                    continue
                regions_seen.add(rec.level_1_id)
                if target_state in rec.sequence:
                    idx = rec.sequence.index(target_state)
                    indices.append(idx)
                    times.append(rec.time_points[idx])
            n_regions = len(regions_seen)
            if n_regions > 0:
                regions_with = len(
                    {
                        rec.level_1_id
                        for rec in sequence_data.records
                        if rec.level_2_id == unit and target_state in rec.sequence
                    }
                )
                diffusion_share = regions_with / n_regions
            mean_first_target_index = float(np.mean(indices)) if indices else np.nan
            mean_first_target_time = float(np.mean(times)) if times else np.nan

        rows.append(
            {
                "level_2_id": unit,
                "n_pairs": int(mask.sum()),
                "cross_region_diversity": cross_region_diversity,
                "diffusion_share": diffusion_share,
                "mean_first_target_index": mean_first_target_index,
                "mean_first_target_time": mean_first_target_time,
            }
        )

    return pd.DataFrame(rows)
