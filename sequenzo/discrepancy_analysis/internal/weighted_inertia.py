"""
@Author  : Yuqi Liang 梁彧祺
@File    : weighted_inertia.py
@Time    : 2026-02-10 19:19
@Desc    : 
Shared inertia / sum-of-squares helpers aligned with TraMineR C routines.

TraMineR uses two closely related quantities:
- dissvar(): weighted mean of pairwise dissimilarities (divide by W twice).
- dissassoc() / disstree(): sum-of-squares terms from C_tmrWeightedInertiaDist
  with var=FALSE (divide by W once) or C_tmrsubmatrixinertia in the
  unweighted case.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ...utils.core_distance_operations import weighted_inertia_contrib


def _pairwise_weighted_sum(
    distance_matrix: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Sum of w_i * w_j * d_ij over distinct pairs in ``indices``."""
    if len(indices) == 0:
        return 0.0

    sub_dist = distance_matrix[np.ix_(indices, indices)]
    sub_weights = weights[indices]
    total = 0.0
    n_group = len(indices)
    for i in range(n_group):
        for j in range(i + 1, n_group):
            total += sub_weights[i] * sub_weights[j] * sub_dist[i, j]
    return total


def weighted_inertia_sum(
    distance_matrix: np.ndarray,
    indices: np.ndarray,
    weights: Optional[np.ndarray] = None,
    squared: bool = False,
) -> float:
    """
    Return the inertia sum used by dissassoc() with var=FALSE.

    This matches TraMineR's C_tmrWeightedInertiaDist(..., var=FALSE) and the
    unweighted C_tmrsubmatrixinertia helper on the same index set.
    """
    if len(indices) == 0:
        return 0.0

    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if squared:
        matrix = matrix ** 2

    if weights is None:
        weights = np.ones(matrix.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    total_weight = float(np.sum(weights[indices]))
    if total_weight <= 0:
        return 0.0

    pair_sum = _pairwise_weighted_sum(matrix, indices, weights)
    return pair_sum / total_weight


def weighted_inertia_sum_from_submatrix(
    sub_dist: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Inertia sum on an already extracted submatrix and local weights."""
    total_weight = float(np.sum(weights))
    if total_weight <= 0 or sub_dist.shape[0] == 0:
        return 0.0

    total = 0.0
    n_group = sub_dist.shape[0]
    for i in range(n_group):
        for j in range(i + 1, n_group):
            total += weights[i] * weights[j] * sub_dist[i, j]
    return total / total_weight


def pseudo_variance_from_inertia_sum(inertia_sum: float, total_weight: float) -> float:
    """Convert an inertia sum into a dissvar()-style discrepancy."""
    if total_weight <= 0:
        return 0.0
    return inertia_sum / total_weight


def compute_pseudo_variance_from_matrix(
    distance_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
    squared: bool = False,
) -> float:
    """Compute dissvar() on a full square distance matrix."""
    if squared:
        distance_matrix = distance_matrix ** 2

    n = distance_matrix.shape[0]
    if weights is None:
        total_sum = float(np.sum(distance_matrix))
        return total_sum / (2.0 * (n ** 2))

    weights = np.asarray(weights, dtype=np.float64)
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return 0.0

    all_indices = np.arange(n, dtype=np.int32)
    contrib = weighted_inertia_contrib(distance_matrix, all_indices, weights)
    return float(np.sum(weights * contrib) / (2.0 * total_weight))


def distance_to_center_contributions(
    distance_matrix: np.ndarray,
    group_int: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> np.ndarray:
    """
  Compute centered distance-to-center contributions used by the Levene test.

  This mirrors TraMineR's disscenter() + group centering used inside
  dissassocweighted().
    """
    n = distance_matrix.shape[0]
    dist_center = np.zeros(n, dtype=np.float64)

    for group_id in range(1, k + 1):
        group_mask = group_int == group_id
        group_indices = np.where(group_mask)[0]
        if len(group_indices) == 0:
            continue

        group_contrib = weighted_inertia_contrib(
            distance_matrix,
            group_indices.astype(np.int32),
            weights,
        )
        dist_center[group_indices] = group_contrib
        group_weights = weights[group_indices]
        weighted_mean = float(np.sum(group_weights * group_contrib) / np.sum(group_weights))
        dist_center[group_indices] = group_contrib - weighted_mean / 2.0

    return dist_center


def weighted_residuals_z(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted residual sum of squares around the weighted mean."""
    if len(values) == 0:
        return 0.0
    weighted_mean = float(np.sum(weights * values) / np.sum(weights))
    return float(np.sum(weights * ((values - weighted_mean) ** 2)))


def unweighted_residuals_z(values: np.ndarray) -> float:
    """Unweighted residual sum of squares around the mean."""
    if len(values) == 0:
        return 0.0
    mean_value = float(np.mean(values))
    return float(np.sum((values - mean_value) ** 2))


def compute_dissassoc_test_values(
    distance_matrix: np.ndarray,
    group_int: np.ndarray,
    weights: np.ndarray,
    k: int,
    sc_tot: float,
    total_weight: float,
    dist_center: np.ndarray,
    dissc_sc_tot: float,
    group_index_lists: Sequence[np.ndarray],
    inertia_getter,
) -> np.ndarray:
    """
    Compute the five dissassoc() test statistics for one grouping.

    Parameters
    ----------
    inertia_getter
        Callable returning (group_weight, group_inertia_sum, levene_term)
        for each group id in ``1..k``.
    """
    sc_res = 0.0
    lns = 0.0
    nlnvi = 0.0
    fbf_denominator = 0.0
    sum_z = 0.0
    s1ni = 0.0

    for group_id in range(1, k + 1):
        group_weight, group_inertia, levene_term = inertia_getter(group_id)
        sc_res += group_inertia
        sum_z += levene_term

        if group_weight <= 0:
            continue

        group_variance = group_inertia / group_weight
        if group_weight > 1 and group_variance >= 0:
            lns += (group_weight - 1.0) * (group_variance / (total_weight - k))
            if group_variance == 0:
                nlnvi = -np.inf
            elif np.isfinite(nlnvi):
                nlnvi += (group_weight - 1.0) * np.log(group_variance)
            s1ni += 1.0 / (group_weight - 1.0)

        fbf_denominator += (1.0 - group_weight / total_weight) * group_variance

    sc_exp = sc_tot - sc_res
    if k > 1 and (total_weight - k) > 0 and sc_res > 0:
        pseudo_f = (sc_exp / (k - 1.0)) / (sc_res / (total_weight - k))
    else:
        pseudo_f = 0.0

    pseudo_r2 = sc_exp / sc_tot if sc_tot > 0 else 0.0
    pseudo_fbf = sc_exp / fbf_denominator if fbf_denominator > 0 else np.nan

    if lns > 0 and k > 1:
        t_calc = (total_weight - k) * np.log(lns) - nlnvi
        c_calc = 1.0 + (1.0 / (3.0 * (k - 1.0))) * (s1ni - 1.0 / (total_weight - k))
        bartlett = t_calc / c_calc if c_calc > 0 else np.nan
    else:
        bartlett = np.nan

    if sum_z > 0 and k > 1:
        levene = ((dissc_sc_tot - sum_z) / (k - 1.0)) / (sum_z / (total_weight - k))
    else:
        levene = np.nan

    return np.array([pseudo_f, pseudo_fbf, pseudo_r2, bartlett, levene], dtype=np.float64)
