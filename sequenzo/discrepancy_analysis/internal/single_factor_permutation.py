"""Permutation engine for dissassoc() (TraMineR dissassocweighted.*)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .weighted_inertia import (
    compute_dissassoc_test_values,
    distance_to_center_contributions,
    unweighted_residuals_z,
    weighted_inertia_sum,
    weighted_residuals_z,
)

def _group_index_lists(group_int: np.ndarray, k: int) -> List[np.ndarray]:
    """Precompute row indices for each 1-based group label."""
    return [np.where(group_int == group_id)[0] for group_id in range(1, k + 1)]


def _levene_group_term(
    dist_center: np.ndarray,
    group_indices: np.ndarray,
    weights: Optional[np.ndarray],
    unweighted: bool,
) -> float:
    """Within-group residual sum of z values for the Levene statistic."""
    if len(group_indices) == 0:
        return 0.0

    values = dist_center[group_indices]
    if unweighted:
        return unweighted_residuals_z(values)

    group_weights = weights[group_indices]
    return weighted_residuals_z(values, group_weights)


def association_permutation_test(
    distance_matrix: np.ndarray,
    group_int: np.ndarray,
    weights: np.ndarray,
    R: int,
    weight_permutation: str,
    squared: bool = False,
    sc_tot: Optional[float] = None,
    totweights: Optional[float] = None,
    k: Optional[int] = None,
    dist_center: Optional[np.ndarray] = None,
    dissc_sc_tot: Optional[float] = None,
) -> dict:
    """
    Permutation test for dissassoc() with TraMineR-compatible statistics.

    Returns the five observed statistics and their permutation p-values:
    Pseudo F, Pseudo Fbf, Pseudo R2, Bartlett, and Levene.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    if squared:
        distance_matrix = distance_matrix ** 2

    n = distance_matrix.shape[0]
    weights = np.asarray(weights, dtype=np.float64)
    unweighted = weight_permutation == "none"

    if totweights is None:
        totweights = float(np.sum(weights))
    if k is None:
        k = len(np.unique(group_int))
    if sc_tot is None:
        sc_tot = weighted_inertia_sum(
            distance_matrix,
            np.arange(n, dtype=np.int32),
            weights=weights,
            squared=False,
        )
    if dist_center is None:
        dist_center = distance_to_center_contributions(
            distance_matrix,
            group_int,
            weights,
            k,
        )
    if dissc_sc_tot is None:
        if unweighted:
            dissc_sc_tot = unweighted_residuals_z(dist_center)
        else:
            dissc_sc_tot = weighted_residuals_z(dist_center, weights)

    indgrp = _group_index_lists(group_int, k)

    def _values_from_perm(perm: np.ndarray, perm_weights: np.ndarray) -> np.ndarray:
        def inertia_getter(group_id: int) -> Tuple[float, float, float]:
            members = indgrp[group_id - 1]
            group_indices = perm[members]
            group_weight = float(np.sum(perm_weights[members]))
            group_inertia = weighted_inertia_sum(
                distance_matrix,
                group_indices,
                weights=perm_weights,
                squared=False,
            )
            levene_term = _levene_group_term(
                dist_center,
                group_indices,
                perm_weights,
                unweighted=unweighted,
            )
            return group_weight, group_inertia, levene_term

        return compute_dissassoc_test_values(
            distance_matrix=distance_matrix,
            group_int=group_int,
            weights=perm_weights,
            k=k,
            sc_tot=sc_tot,
            total_weight=totweights,
            dist_center=dist_center,
            dissc_sc_tot=dissc_sc_tot,
            group_index_lists=indgrp,
            inertia_getter=inertia_getter,
        )

    t0 = _values_from_perm(np.arange(n, dtype=np.int32), weights)

    if R <= 1:
        return {"R": R, "t0": t0, "t": None, "pval": np.full(5, np.nan)}

    t_matrix = np.zeros((R, 5))
    t_matrix[0, :] = t0

    if weight_permutation == "replicate":
        if not np.all(weights == np.round(weights)):
            raise ValueError("[!] For 'replicate' method, weights must be integers")

        expanded_indices = np.repeat(np.arange(n), weights.astype(int))
        expanded_group = group_int[expanded_indices]

        for replicate in range(1, R):
            perm = np.random.permutation(len(expanded_indices))
            perm_group = expanded_group[perm]

            def inertia_getter(group_id: int) -> Tuple[float, float, float]:
                members = np.where(perm_group == group_id)[0]
                if len(members) == 0:
                    return 0.0, 0.0, 0.0
                case_indices = expanded_indices[members]
                counts = np.bincount(case_indices, minlength=n).astype(np.float64)
                active = np.where(counts > 0)[0]
                group_weight = float(np.sum(counts[active]))
                group_inertia = weighted_inertia_sum(
                    distance_matrix,
                    active,
                    weights=counts,
                    squared=False,
                )
                levene_term = _levene_group_term(
                    dist_center,
                    active,
                    counts,
                    unweighted=False,
                )
                return group_weight, group_inertia, levene_term

            t_matrix[replicate, :] = compute_dissassoc_test_values(
                distance_matrix=distance_matrix,
                group_int=group_int,
                weights=weights,
                k=k,
                sc_tot=sc_tot,
                total_weight=totweights,
                dist_center=dist_center,
                dissc_sc_tot=dissc_sc_tot,
                group_index_lists=indgrp,
                inertia_getter=inertia_getter,
            )
    else:
        for replicate in range(1, R):
            perm = np.random.permutation(n)
            if weight_permutation == "group":
                perm_weights = weights[perm]
            else:
                perm_weights = weights
            t_matrix[replicate, :] = _values_from_perm(perm, perm_weights)

    pval = np.array([np.mean(t_matrix[:, j] >= t0[j]) for j in range(5)])
    return {"R": R, "t0": t0, "t": t_matrix, "pval": pval}


