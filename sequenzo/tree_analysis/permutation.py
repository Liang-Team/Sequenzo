"""
@Author  : Yuqi Liang 梁彧祺
@File    : permutation.py
@Time    : 2026-02-10
@Desc    : Permutation test implementation for tree analysis.

This module implements permutation tests matching TraMineR's behavior for
dissassoc() and disstree() functions. It supports various weight permutation
methods: none, replicate, diss, group.

Corresponds to TraMineR functions: DTNdissassocweighted(), dissassocweighted.*()
"""

import numpy as np
from typing import Callable, Optional
import random


def _compute_submatrix_inertia(
    distance_matrix: np.ndarray,
    indices: np.ndarray,
    weights: Optional[np.ndarray] = None,
    squared: bool = False
) -> float:
    """
    Compute weighted inertia (sum of squares) for a subset of sequences.
    
    This corresponds to TraMineR's C_tmrWeightedInertiaDist with var=FALSE
    or C_tmrsubmatrixinertia for unweighted case.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Full distance matrix (n x n)
    indices : np.ndarray
        Indices of sequences to include (0-based)
    weights : np.ndarray, optional
        Weights for sequences. If None, uses equal weights.
    squared : bool
        Whether distances are squared
        
    Returns
    -------
    float
        Sum of squares (SC) for the subset
    """
    if len(indices) == 0:
        return 0.0
    
    # Extract submatrix
    submatrix = distance_matrix[np.ix_(indices, indices)]
    
    if squared:
        submatrix = submatrix ** 2
    
    n = len(indices)
    
    if weights is None:
        # Unweighted: sum over all pairs
        ss = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                ss += submatrix[i, j]
        return ss
    else:
        # Weighted: sum(w_i * w_j * d_ij)
        sub_weights = weights[indices]
        ss = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                ss += sub_weights[i] * sub_weights[j] * submatrix[i, j]
        return ss


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
    
    This function is used internally by `build_distance_tree()` to test whether each
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
    n = len(group)
    dmatsize = distance_matrix.shape[0]
    
    # Define statistic functions matching TraMineR's internal functions
    def internal_unweighted(grp, perm_indices, indiv, dmat, grp2, use_sort):
        """Unweighted statistic: returns -(SC1 + SC2)"""
        groupe1 = indiv[perm_indices[grp]]
        groupe2 = indiv[perm_indices[grp2]]
        
        if use_sort:
            groupe1 = np.sort(groupe1)
            groupe2 = np.sort(groupe2)
        
        r1 = _compute_submatrix_inertia(dmat, groupe1, weights=None, squared=squared)
        r2 = _compute_submatrix_inertia(dmat, groupe2, weights=None, squared=squared)
        return -(r1 + r2)
    
    def internal_weighted(grp, perm_indices, indiv, dmat, grp2, use_sort, w, permut_group):
        """Weighted statistic: returns -(SC1 + SC2)"""
        groupe1 = indiv[perm_indices[grp]]
        groupe2 = indiv[perm_indices[grp2]]
        
        if use_sort:
            groupe1 = np.sort(groupe1)
            groupe2 = np.sort(groupe2)
        
        if permut_group:
            # Permute weights according to permuted indices
            # Map permuted indices back to original weight positions
            w_perm = w.copy()
            w_perm[indiv] = w[indiv[perm_indices]]
            w1 = w_perm[groupe1]
            w2 = w_perm[groupe2]
        else:
            # Use original weights
            w1 = w[groupe1]
            w2 = w[groupe2]
        
        r1 = _compute_submatrix_inertia(dmat, groupe1, weights=w1, squared=squared)
        r2 = _compute_submatrix_inertia(dmat, groupe2, weights=w2, squared=squared)
        return -(r1 + r2)
    
    def internal_replicate(grp, perm_indices, indiv, dmat, grp2):
        """Replicate statistic: expand groups by weights"""
        groupe1n = indiv[perm_indices[grp]]
        groupe2n = indiv[perm_indices[grp2]]
        
        # Count occurrences (like tabulate in R)
        wwt1 = np.bincount(groupe1n, minlength=dmatsize)
        wwt2 = np.bincount(groupe2n, minlength=dmatsize)
        
        groupe1 = np.where(wwt1 > 0)[0]
        groupe2 = np.where(wwt2 > 0)[0]
        
        r1 = _compute_submatrix_inertia(dmat, groupe1, weights=wwt1[groupe1], squared=squared)
        r2 = _compute_submatrix_inertia(dmat, groupe2, weights=wwt2[groupe2], squared=squared)
        return -(r1 + r2)
    
    # Map indices to original distance matrix indices
    indiv = indices  # Original indices (0-based, for accessing distance_matrix)
    
    # Determine which statistic function to use
    use_sort = len(group) > 750  # TraMineR optimization
    
    if weight_permutation == "none":
        def statistic_func(grp, perm_indices):
            grp2 = ~grp  # Complement group
            return internal_unweighted(grp, perm_indices, indiv, distance_matrix, grp2, use_sort)
    elif weight_permutation in ("diss", "group"):
        permut_group = (weight_permutation == "group")
        def statistic_func(grp, perm_indices):
            grp2 = ~grp  # Complement group
            return internal_weighted(grp, perm_indices, indiv, distance_matrix, grp2, use_sort, weights, permut_group)
    elif weight_permutation == "replicate":
        # Expand groups by weights
        node_weights = weights[indices]
        if not np.all(node_weights == np.round(node_weights)):
            raise ValueError("[!] For 'replicate' method, weights must be integers")
        
        # Replicate groups and indices
        expanded_group = []
        expanded_indices = []
        for i, idx in enumerate(indices):
            w = int(node_weights[i])
            expanded_group.extend([group[i]] * w)
            expanded_indices.extend([idx] * w)
        
        expanded_group = np.array(expanded_group, dtype=bool)
        expanded_indices = np.array(expanded_indices, dtype=np.int32)
        
        def statistic_func(grp, perm_indices):
            grp2 = ~grp  # Complement group
            return internal_replicate(grp, perm_indices, expanded_indices, distance_matrix, grp2)
        
        # Update group and indices for replicate case
        group = expanded_group
        indiv = expanded_indices
        n = len(group)
    else:
        raise ValueError(f"[!] Unknown weight_permutation method: {weight_permutation}")
    
    # Run permutation test
    result = permutation_test(group, R, statistic_func)
    
    # Return p-value for first test statistic
    return float(result['pval'][0])
