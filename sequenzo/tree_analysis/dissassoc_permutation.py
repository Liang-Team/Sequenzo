"""
@Author  : Yuqi Liang 梁彧祺
@File    : dissassoc_permutation.py
@Time    : 2026-02-10 10:02
@Desc    : Permutation test implementation for dissassoc (compute_distance_association).

This module implements permutation tests matching TraMineR's dissassocweighted.*()
functions for testing association between distance matrices and grouping variables.

Corresponds to TraMineR functions: dissassocweighted.unweighted(), 
dissassocweighted.replicate(), dissassocweighted.permdiss(), etc.
"""

import numpy as np
from typing import Callable, Optional, Tuple


def _compute_group_inertia(
    distance_matrix: np.ndarray,
    group_indices: np.ndarray,
    weights: Optional[np.ndarray] = None,
    squared: bool = False
) -> Tuple[float, float]:
    """
    Compute group inertia (sum of squares) and group size.
    
    Returns (SCresi, group_size) matching TraMineR's behavior.
    """
    if len(group_indices) == 0:
        return 0.0, 0.0
    
    if weights is None:
        # Unweighted: use C_tmrsubmatrixinertia equivalent
        group_dist = distance_matrix[np.ix_(group_indices, group_indices)]
        if squared:
            group_dist = group_dist ** 2
        
        group_ss = 0.0
        n_group = len(group_indices)
        for i in range(n_group):
            for j in range(i + 1, n_group):
                group_ss += group_dist[i, j]
        
        group_size = float(n_group)
    else:
        # Weighted: use C_tmrWeightedInertiaDist with var=FALSE
        group_dist = distance_matrix[np.ix_(group_indices, group_indices)]
        group_weights = weights[group_indices]
        
        if squared:
            group_dist = group_dist ** 2
        
        group_ss = 0.0
        n_group = len(group_indices)
        for i in range(n_group):
            for j in range(i + 1, n_group):
                group_ss += group_weights[i] * group_weights[j] * group_dist[i, j]
        
        group_size = float(np.sum(group_weights))
    
    return group_ss, group_size


def dissassoc_permutation_test(
    distance_matrix: np.ndarray,
    group_int: np.ndarray,
    weights: np.ndarray,
    R: int,
    weight_permutation: str,
    squared: bool = False,
    SCtot: float = None,
    totweights: float = None,
    k: int = None
) -> dict:
    """
    Permutation test for dissassoc matching TraMineR's dissassocweighted.*().
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Full distance matrix
    group_int : np.ndarray
        Group assignments as integers (1-based)
    weights : np.ndarray
        Weights for all sequences
    R : int
        Number of permutations
    weight_permutation : str
        Method: "none", "replicate", "diss", "group", "random-sampling"
    squared : bool
        Whether distances are squared
    SCtot : float, optional
        Precomputed total sum of squares (for efficiency)
    totweights : float, optional
        Precomputed total weights (for efficiency)
    k : int, optional
        Number of groups (for efficiency)
        
    Returns
    -------
    dict
        Permutation test results with keys: 'R', 't0', 't', 'pval'
    """
    n = distance_matrix.shape[0]
    
    if SCtot is None:
        # Compute SCtot using same logic as compute_pseudo_variance
        # (avoiding circular import)
        if squared:
            dist_sq = distance_matrix ** 2
        else:
            dist_sq = distance_matrix
        
        n = distance_matrix.shape[0]
        totweights = np.sum(weights) if totweights is None else totweights
        
        # Compute weighted sum of squares
        weighted_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                weighted_sum += weights[i] * weights[j] * dist_sq[i, j]
        
        # Discrepancy (variance) = weighted_sum / totweights^2
        discrepancy = weighted_sum / (totweights ** 2)
        # SCtot = discrepancy * totweights
        SCtot = discrepancy * totweights
    
    if totweights is None:
        totweights = np.sum(weights)
    
    if k is None:
        k = len(np.unique(group_int))
    
    # Build group index lists
    indgrp = []
    for i in range(1, k + 1):
        indgrp.append(np.where(group_int == i)[0])
    
    # Compute observed test values
    SCres = 0.0
    for i in range(k):
        group_indices = indgrp[i]
        group_ss, _ = _compute_group_inertia(
            distance_matrix, group_indices, weights=weights, squared=squared
        )
        SCres += group_ss
    
    SCexp = SCtot - SCres
    PseudoF = (SCexp / (k - 1)) / (SCres / (totweights - k)) if (k > 1 and SCres > 0) else 0.0
    PseudoR2 = SCexp / SCtot if SCtot > 0 else 0.0
    
    t0 = np.array([PseudoF, PseudoR2])
    
    # Define statistic function for permutations
    # Handle different weight_permutation methods
    if weight_permutation == "none":
        def compute_test_values(perm_indices):
            """Compute test statistics for unweighted permutation."""
            SCres_perm = 0.0
            
            for i in range(k):
                group_mask = (group_int[perm_indices] == (i + 1))
                group_indices_perm = np.where(group_mask)[0]
                
                if len(group_indices_perm) == 0:
                    continue
                
                # Map back to original indices
                group_indices_original = perm_indices[group_indices_perm]
                group_ss, _ = _compute_group_inertia(
                    distance_matrix, group_indices_original, weights=None, squared=squared
                )
                SCres_perm += group_ss
            
            SCexp_perm = SCtot - SCres_perm
            PseudoF_perm = (SCexp_perm / (k - 1)) / (SCres_perm / (totweights - k)) if (k > 1 and SCres_perm > 0) else 0.0
            PseudoR2_perm = SCexp_perm / SCtot if SCtot > 0 else 0.0
            
            return np.array([PseudoF_perm, PseudoR2_perm])
    elif weight_permutation == "replicate":
        # Expand by weights
        if not np.all(weights == np.round(weights)):
            raise ValueError("[!] For 'replicate' method, weights must be integers")
        
        expanded_indices = []
        expanded_group_int = []
        for i in range(n):
            w = int(weights[i])
            expanded_indices.extend([i] * w)
            expanded_group_int.extend([group_int[i]] * w)
        
        expanded_indices = np.array(expanded_indices, dtype=np.int32)
        expanded_group_int = np.array(expanded_group_int, dtype=np.int32)
        n_expanded = len(expanded_indices)
        
        def compute_test_values(perm_indices):
            """Compute test statistics for replicate permutation."""
            SCres_perm = 0.0
            
            for i in range(k):
                group_mask = (expanded_group_int[perm_indices] == (i + 1))
                group_indices_perm = expanded_indices[perm_indices[group_mask]]
                
                if len(group_indices_perm) == 0:
                    continue
                
                # Count occurrences
                wwt = np.bincount(group_indices_perm, minlength=n)
                group_indices_unique = np.where(wwt > 0)[0]
                
                if len(group_indices_unique) == 0:
                    continue
                
                group_ss, _ = _compute_group_inertia(
                    distance_matrix, group_indices_unique, weights=wwt[group_indices_unique], squared=squared
                )
                SCres_perm += group_ss
            
            SCexp_perm = SCtot - SCres_perm
            PseudoF_perm = (SCexp_perm / (k - 1)) / (SCres_perm / (totweights - k)) if (k > 1 and SCres_perm > 0) else 0.0
            PseudoR2_perm = SCexp_perm / SCtot if SCtot > 0 else 0.0
            
            return np.array([PseudoF_perm, PseudoR2_perm])
        
        # Update n for replicate case
        n = n_expanded
    else:
        # For other methods (diss, group), use standard permutation
        def compute_test_values(perm_indices):
            """Compute test statistics for weighted permutation."""
            SCres_perm = 0.0
            
            for i in range(k):
                group_mask = (group_int[perm_indices] == (i + 1))
                group_indices_perm = np.where(group_mask)[0]
                
                if len(group_indices_perm) == 0:
                    continue
                
                # Map back to original indices
                group_indices_original = perm_indices[group_indices_perm]
                group_ss, _ = _compute_group_inertia(
                    distance_matrix, group_indices_original, weights=weights[group_indices_original], squared=squared
                )
                SCres_perm += group_ss
            
            SCexp_perm = SCtot - SCres_perm
            PseudoF_perm = (SCexp_perm / (k - 1)) / (SCres_perm / (totweights - k)) if (k > 1 and SCres_perm > 0) else 0.0
            PseudoR2_perm = SCexp_perm / SCtot if SCtot > 0 else 0.0
            
            return np.array([PseudoF_perm, PseudoR2_perm])
    
    # Run permutations
    if R <= 1:
        return {
            'R': R,
            't0': t0,
            't': None,
            'pval': np.array([np.nan, np.nan])
        }
    
    t_matrix = np.zeros((R, 2))
    t_matrix[0, :] = t0
    
    for i in range(1, R):
        perm_indices = np.random.permutation(n)
        t_matrix[i, :] = compute_test_values(perm_indices)
    
    # Compute p-values
    pval = np.array([
        np.mean(t_matrix[:, 0] >= t0[0]),
        np.mean(t_matrix[:, 1] >= t0[1])
    ])
    
    return {
        'R': R,
        't0': t0,
        't': t_matrix,
        'pval': pval
    }
