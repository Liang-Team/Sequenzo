"""
@Author  : 梁彧祺 Yuqi Liang
@File    : aggregate.py
@Time    : 16/04/2026 07:35
@Desc    :
    Aggregate pair-level distances to level-1 or level-2 unit distances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..distances import RelationalDistanceMatrix


def _aggregate_block_mean(matrix: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.nan
    block = matrix[np.ix_(idx_a, idx_b)]
    if np.array_equal(idx_a, idx_b):
        iu = np.triu_indices(len(idx_a), k=1)
        if len(iu[0]) == 0:
            return 0.0
        return float(np.mean(block[iu]))
    return float(np.mean(block))


def aggregate_distance_matrix_by_level(
    distance_matrix: RelationalDistanceMatrix,
    level: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a unit-level distance matrix by averaging pair distances across blocks.
    """
    if level not in (1, 2):
        raise ValueError("level must be 1 or 2.")

    ids = distance_matrix.level_1_ids if level == 1 else distance_matrix.level_2_ids
    unique = pd.unique(ids)
    n_units = len(unique)
    agg = np.zeros((n_units, n_units), dtype=float)

    index_lists = [np.where(ids == u)[0] for u in unique]
    matrix = distance_matrix.matrix
    for i, idx_i in enumerate(index_lists):
        agg[i, i] = _aggregate_block_mean(matrix, idx_i, idx_i)
        for j in range(i + 1, n_units):
            block_mean = _aggregate_block_mean(matrix, idx_i, index_lists[j])
            agg[i, j] = block_mean
            agg[j, i] = block_mean

    return agg, np.asarray(unique, dtype=object)
