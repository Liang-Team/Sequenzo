"""
@Author  : Yuqi Liang 梁彧祺
@File    : permutation_engine.py
@Time    : 2026-02-10
@Desc    : Permutation test implementation for tree analysis.

This module implements permutation tests matching TraMineR's behavior for
dissassoc() and disstree() functions. It supports various weight permutation
methods: none, replicate, diss, group.

Corresponds to TraMineR functions: DTNdissassocweighted(), dissassocweighted.*()
"""

import numpy as np
from typing import Callable, Optional

from .weighted_inertia import weighted_inertia_sum


def _compute_submatrix_inertia(
    distance_matrix: np.ndarray,
    indices: np.ndarray,
    weights: Optional[np.ndarray] = None,
    squared: bool = False,
    weights_are_local: bool = False,
) -> float:
    """TraMineR-compatible inertia sum for a node split."""
    if len(indices) == 0:
        return 0.0

    if weights is None:
        return weighted_inertia_sum(
            distance_matrix,
            indices,
            weights=None,
            squared=squared,
        )

    if weights_are_local:
        full_weights = np.zeros(distance_matrix.shape[0], dtype=np.float64)
        full_weights[indices] = weights
        return weighted_inertia_sum(
            distance_matrix,
            indices,
            weights=full_weights,
            squared=squared,
        )

    return weighted_inertia_sum(
        distance_matrix,
        indices,
        weights=weights,
        squared=squared,
    )


def permutation_test(
    data: np.ndarray,
    R: int,
    statistic: Callable,
    **kwargs
) -> dict:
    """
    General permutation test framework matching TraMineR.permutation().
    
    Parameters
    ----------
    data : np.ndarray
        Data to permute (usually group assignments)
    R : int
        Number of permutations
    statistic : Callable
        Function that computes test statistic: statistic(data, permuted_indices, **kwargs)
    **kwargs
        Additional arguments passed to statistic function
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'R': Number of permutations
        - 't0': Observed statistic value
        - 't': Array of permuted statistics (R x n_tests)
        - 'pval': P-values for each test statistic
    """
    n = len(data)
    
    # Compute observed statistic
    t0 = statistic(data, np.arange(n), **kwargs)
    n_tests = len(t0) if isinstance(t0, np.ndarray) else 1
    if n_tests == 1:
        t0 = np.array([t0])
    
    result = {
        'R': R,
        't0': t0,
        't': None,
        'pval': np.full(n_tests, np.nan)
    }
    
    if R <= 1:
        return result
    
    # Perform permutations
    t_matrix = np.zeros((R, n_tests))
    t_matrix[0, :] = t0
    
    # Set random seed for reproducibility (if needed)
    # Note: In TraMineR, seed is set externally
    
    for i in range(1, R):
        # Permute indices
        permuted_indices = np.random.permutation(n)
        t_matrix[i, :] = statistic(data, permuted_indices, **kwargs)
    
    result['t'] = t_matrix
    
    # Compute p-values: proportion of permuted stats >= observed
    for j in range(n_tests):
        result['pval'][j] = np.mean(t_matrix[:, j] >= t0[j])
    
    return result


def test_tree_split_significance(
    distance_matrix: np.ndarray,
    group: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    R: int,
    weight_permutation: str = "none",
    squared: bool = False
) -> float:
    """
    Test the statistical significance of a binary split in a distance tree using permutation tests.
    
    This function performs a permutation test to determine whether a proposed binary split
    of sequences into two groups is statistically significant. It randomly permutes group
    assignments and compares the observed split quality with the permuted distributions.
    
    **Corresponds to TraMineR function: `DTNdissassocweighted()`**
    
    This function is used internally by `distance_tree()` to test whether each
    potential split should be accepted or rejected based on statistical significance.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Full distance matrix
    group : np.ndarray
        Binary group assignment (True/False) for sequences in indices
    indices : np.ndarray
        Original indices of sequences in current node
    weights : np.ndarray
        Weights for all sequences (full length)
    R : int
        Number of permutations
    weight_permutation : str
        Method: "none", "replicate", "diss", "group"
    squared : bool
        Whether distances are squared
        
    Returns
    -------
    float
        P-value for the split
    """
    node_indices = np.asarray(indices, dtype=np.int32)
    group = np.asarray(group, dtype=bool)

    if len(node_indices) != len(group):
        raise ValueError("[!] 'group' and 'indices' must have the same length")

    if weight_permutation not in {"none", "replicate", "diss", "group"}:
        raise ValueError(f"[!] Unknown weight_permutation method: {weight_permutation}")

    if weight_permutation == "replicate":
        node_weights = weights[node_indices]
        if not np.all(node_weights == np.round(node_weights)):
            raise ValueError("[!] For 'replicate' method, weights must be integers")

        expanded_group = []
        expanded_indices = []
        for i, idx in enumerate(node_indices):
            w = int(node_weights[i])
            if w <= 0:
                continue
            expanded_group.extend([group[i]] * w)
            expanded_indices.extend([idx] * w)

        perm_base_group = np.asarray(expanded_group, dtype=bool)
        perm_base_indices = np.asarray(expanded_indices, dtype=np.int32)

        def statistic_func(grp: np.ndarray, perm_indices: np.ndarray) -> float:
            permuted = grp[perm_indices]
            g1_rep = perm_base_indices[permuted]
            g2_rep = perm_base_indices[~permuted]

            # Equivalent to R tabulate() -> unique + local weights
            w1 = np.bincount(g1_rep, minlength=distance_matrix.shape[0]).astype(float)
            w2 = np.bincount(g2_rep, minlength=distance_matrix.shape[0]).astype(float)
            i1 = np.where(w1 > 0)[0]
            i2 = np.where(w2 > 0)[0]

            r1 = _compute_submatrix_inertia(
                distance_matrix, i1, weights=w1[i1], squared=squared, weights_are_local=True
            )
            r2 = _compute_submatrix_inertia(
                distance_matrix, i2, weights=w2[i2], squared=squared, weights_are_local=True
            )
            return -(r1 + r2)

        result = permutation_test(perm_base_group, R, statistic_func)
    else:
        local_weights = np.asarray(weights[node_indices], dtype=float)

        def statistic_func(grp: np.ndarray, perm_indices: np.ndarray) -> float:
            permuted = grp[perm_indices]
            i1 = node_indices[permuted]
            i2 = node_indices[~permuted]

            if weight_permutation == "none":
                r1 = _compute_submatrix_inertia(distance_matrix, i1, weights=None, squared=squared)
                r2 = _compute_submatrix_inertia(distance_matrix, i2, weights=None, squared=squared)
                return -(r1 + r2)

            # diss: keep original local weights
            # group: permute local weights with the same permutation as group labels
            if weight_permutation == "group":
                permuted_local_weights = local_weights[perm_indices]
            else:
                permuted_local_weights = local_weights

            w1 = permuted_local_weights[permuted]
            w2 = permuted_local_weights[~permuted]

            r1 = _compute_submatrix_inertia(
                distance_matrix, i1, weights=w1, squared=squared, weights_are_local=True
            )
            r2 = _compute_submatrix_inertia(
                distance_matrix, i2, weights=w2, squared=squared, weights_are_local=True
            )
            return -(r1 + r2)

        result = permutation_test(group, R, statistic_func)
    
    # Return p-value for first test statistic
    return float(result['pval'][0])


# Public analysis helper re-exported as `test_tree_split`; mark it so pytest
# does not collect it when users or tests run `from sequenzo import *`.
test_tree_split_significance.__test__ = False
