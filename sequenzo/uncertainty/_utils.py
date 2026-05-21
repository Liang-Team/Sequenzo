"""
@Author  : Yuqi Liang 梁彧祺
@File    : _utils.py
@Time    : 03/05/2026 19:20
@Desc    : Distance indexing utilities for timing-uncertainty MC results.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import squareform


def get_ij_table(k: int) -> np.ndarray:
    """Pair indices (i, j) with i < j in dist-vector order."""
    rows: List[Tuple[int, int]] = []
    for i in range(1, k):
        for j in range(i + 1, k + 1):
            rows.append((i, j))
    return np.asarray(rows, dtype=int)


def dist_index(i: int, j: int, k: int) -> int:
    """Position in condensed distance vector (1-based i, j like R ``didx``)."""
    return k * (i - 1) - i * (i - 1) // 2 + j - i


def vector_to_dist(vec: np.ndarray, labels: List[str]) -> np.ndarray:
    """Build square symmetric distance matrix from lower-triangle vector."""
    k = len(labels)
    mat = np.zeros((k, k), dtype=float)
    idx = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            mat[i, j] = vec[idx]
            mat[j, i] = vec[idx]
            idx += 1
    return mat


def dist_to_condensed(mat: np.ndarray) -> np.ndarray:
    """Extract lower triangle as condensed vector."""
    k = mat.shape[0]
    return squareform(mat, checks=False)


def expand_unique_distances(
    mc_mean: np.ndarray,
    mc_se: np.ndarray,
    disagg_index: np.ndarray,
    labels: List[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand unique-sequence MC results to all rows (R ``MCexpand``).

    ``disagg_index`` is 1-based mapping from original row -> unique index.
    """
    sdx = np.asarray(disagg_index, dtype=int) - 1
    mean_full = mc_mean[np.ix_(sdx, sdx)]
    se_full = mc_se[np.ix_(sdx, sdx)]
    full_labels = list(labels)
    return mean_full, se_full


def n_transitions_from_sdur(sdur_row: np.ndarray) -> int:
    return int(np.sum(sdur_row > 0)) - 1


def single_spell_sequence(dss_row: np.ndarray, sdur_row: np.ndarray) -> bool:
    """True when the sequence has no transition (one spell only)."""
    return n_transitions_from_sdur(sdur_row) <= 0
