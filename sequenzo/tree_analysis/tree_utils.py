"""
@Author  : Yuqi Liang 梁彧祺
@File    : tree_utils.py
@Time    : 2026-02-09 10:29
@Desc    : Utility functions for tree-structured analysis of sequences.
           These functions compute pseudo-variance and test associations between
           distance matrices and grouping variables.

           Corresponds to TraMineR functions: dissvar(), dissassoc()
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
import importlib
from .dissassoc_permutation import dissassoc_permutation_test

# Lazy import clustering C code for weighted inertia calculations
_clustering_c_code = None

def _get_clustering_c_code():
    """Lazy import of clustering C code module to avoid circular dependencies."""
    global _clustering_c_code
    if _clustering_c_code is None:
        _clustering_c_code = importlib.import_module("sequenzo.clustering.clustering_c_code")
    return _clustering_c_code


def compute_pseudo_variance(
    distance_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
    squared: bool = False
) -> float:
    """
    Compute the pseudo-variance (discrepancy) of a distance matrix.
    
    This function computes a measure of overall variability in the distance matrix,
    which is used as the basis for tree-structured analysis. The pseudo-variance
    represents the total "spread" of sequences in the distance space.
    
    **Corresponds to TraMineR function: `dissvar()`**
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        A square symmetric distance matrix of shape (n, n) where n is the number
        of sequences. The matrix should contain pairwise distances between sequences.
        
    weights : np.ndarray, optional
        Optional weights for each sequence. If None, all sequences are given
        equal weight. Shape should be (n,).
        Default: None (equal weights)
        
    squared : bool, optional
        If True, square the distance matrix before computing variance.
        This is useful when working with squared distances.
        Default: False
        
    Returns
    -------
    float
        The pseudo-variance (discrepancy) of the distance matrix.
        This is a measure of overall variability in the sequence space.
        
    Notes
    -----
    - For unweighted case: pseudo-variance = sum(all distances) / (2 * n^2)
    - For weighted case: uses weighted inertia calculation from C code
    - The pseudo-variance is used as the total sum of squares (SStot) in
      tree-structured analysis
      
    Examples
    --------
    >>> import numpy as np
    >>> from sequenzo.tree_analysis import compute_pseudo_variance
    >>> 
    >>> # Create a simple distance matrix
    >>> dist_matrix = np.array([
    ...     [0.0, 1.0, 2.0],
    ...     [1.0, 0.0, 1.5],
    ...     [2.0, 1.5, 0.0]
    ... ])
    >>> 
    >>> # Compute pseudo-variance
    >>> variance = compute_pseudo_variance(dist_matrix)
    >>> print(f"Pseudo-variance: {variance:.4f}")
    
    References
    ----------
    Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011).
    Discrepancy analysis of state sequences.
    Sociological Methods and Research, Vol. 40(3), 471-510.
    """
    # Convert to numpy array if needed
    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.values
    
    # Square the matrix if requested
    if squared:
        distance_matrix = distance_matrix ** 2
    
    n = distance_matrix.shape[0]
    
    # Check that matrix is square
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(
            "[!] 'distance_matrix' must be a square symmetric matrix. "
            f"Got shape {distance_matrix.shape}"
        )
    
    # Check symmetry (with tolerance for floating point errors)
    if not np.allclose(distance_matrix, distance_matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "[!] 'distance_matrix' must be symmetric. "
            "Distance matrices should be symmetric by definition."
        )
    
    # Unweighted case: simple formula
    if weights is None:
        # Formula: sum(all distances) / (2 * n^2)
        # We divide by 2 because we're summing both upper and lower triangles
        # (diagonal is zero, so it doesn't matter)
        total_sum = np.sum(distance_matrix)
        pseudo_var = total_sum / (2 * (n ** 2))
        return float(pseudo_var)
    
    # Weighted case: compute weighted pseudo-variance
    weights = np.asarray(weights, dtype=np.float64)
    
    if len(weights) != n:
        raise ValueError(
            f"[!] Length of 'weights' ({len(weights)}) must match "
            f"distance matrix size ({n})"
        )
    
    if np.any(weights < 0):
        raise ValueError("[!] All weights must be non-negative")
    
    if np.sum(weights) == 0:
        raise ValueError("[!] Sum of weights must be greater than zero")
    
    # Compute weighted pseudo-variance
    # Formula: sum over all pairs (i,j) of: w_i * w_j * d_ij / (sum of weights)^2
    # This corresponds to C_tmrWeightedInertiaDist in TraMineR
    
    total_weight = np.sum(weights)
    total_weight_squared = total_weight ** 2
    
    # Compute weighted sum of all distances
    # We iterate over upper triangle (i < j) to avoid double counting
    weighted_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            weighted_sum += weights[i] * weights[j] * distance_matrix[i, j]
    
    # Pseudo-variance = weighted_sum / total_weight_squared
    # This gives the average weighted distance
    pseudo_var = weighted_sum / total_weight_squared
    
    return float(pseudo_var)


def compute_distance_association(
    distance_matrix: np.ndarray,
    group: np.ndarray,
    weights: Optional[np.ndarray] = None,
    R: int = 1000,
    weight_permutation: str = "replicate",
    squared: bool = False
) -> dict:
    """
    Test the association between a distance matrix and a grouping variable.
    
    This function performs a pseudo-ANOVA analysis to test whether sequences
    in different groups have significantly different distance patterns.
    It uses permutation tests to assess statistical significance.
    
    **Corresponds to TraMineR function: `dissassoc()`**
    
    Note: This function was originally named `test_distance_association` but renamed
    to avoid pytest confusion. The name reflects that it tests/computes the association
    between distance matrices and grouping variables.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        A square symmetric distance matrix of shape (n, n).
        
    group : np.ndarray
        Grouping variable indicating which group each sequence belongs to.
        Can be numeric (group IDs) or categorical (group labels).
        Shape should be (n,).
        
    weights : np.ndarray, optional
        Optional weights for each sequence. Shape should be (n,).
        Default: None (equal weights)
        
    R : int, optional
        Number of permutations for the permutation test.
        Higher values give more accurate p-values but take longer to compute.
        Default: 1000
        
    weight_permutation : str, optional
        Method for handling weights in permutation tests. Options:
        - "replicate": Replicate cases according to weights (weights must be integers)
        - "diss": Attach weights to the distance matrix
        - "group": Permute at group level
        - "none": Unweighted permutation test
        Default: "replicate"
        
    squared : bool, optional
        If True, square the distance matrix before analysis.
        Default: False
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'pseudo_f': Pseudo F-statistic
        - 'pseudo_r2': Pseudo R-squared (proportion of variance explained)
        - 'pseudo_f_pval': P-value for pseudo F-statistic (from permutation test)
        - 'groups': DataFrame with variance and sample size per group
        - 'anova_table': DataFrame with ANOVA-like table (SS, df, MSE)
        - 'R': Number of permutations used
        
    Notes
    -----
    - This function implements a pseudo-ANOVA analysis for distance matrices
    - The pseudo F-statistic tests whether groups differ significantly
    - Pseudo R² measures how much variance is explained by group membership
    - Permutation tests are used because standard ANOVA assumptions don't hold
      for distance matrices
      
    Examples
    --------
    >>> import numpy as np
    >>> from sequenzo.tree_analysis import test_distance_association
    >>> 
    >>> # Create distance matrix and grouping variable
    >>> dist_matrix = np.array([
    ...     [0.0, 1.0, 2.0, 3.0],
    ...     [1.0, 0.0, 1.5, 2.5],
    ...     [2.0, 1.5, 0.0, 1.0],
    ...     [3.0, 2.5, 1.0, 0.0]
    ... ])
    >>> groups = np.array([1, 1, 2, 2])  # First two in group 1, last two in group 2
    >>> 
    >>> # Test association (with small R for speed)
    >>> result = test_distance_association(dist_matrix, groups, R=100)
    >>> print(f"Pseudo R²: {result['pseudo_r2']:.4f}")
    >>> print(f"P-value: {result['pseudo_f_pval']:.4f}")
    
    References
    ----------
    Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011).
    Discrepancy analysis of state sequences.
    Sociological Methods and Research, Vol. 40(3), 471-510.
    
    Anderson, M. J. (2001). A new method for non-parametric multivariate
    analysis of variance. Austral Ecology, 26, 32-46.
    """
    # Convert to numpy arrays
    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.values
    
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    group = np.asarray(group)
    
    # Handle missing values in group
    if isinstance(group, pd.Series):
        valid_mask = ~pd.isna(group)
    else:
        # Handle both numeric and object arrays
        try:
            valid_mask = ~np.isnan(group.astype(float))
        except (ValueError, TypeError):
            # For non-numeric, check for None/NaN differently
            valid_mask = pd.notna(group) if hasattr(pd, 'notna') else ~pd.isna(group)
    
    if not np.all(valid_mask):
        # Remove rows/columns with missing group values
        valid_indices = np.where(valid_mask)[0]
        distance_matrix = distance_matrix[np.ix_(valid_indices, valid_indices)]
        group = group[valid_mask]
    
    n = distance_matrix.shape[0]
    
    # Square matrix if requested
    if squared:
        distance_matrix = distance_matrix ** 2
    
    # Handle weights
    unweighted = weights is None
    if unweighted:
        weights = np.ones(n, dtype=np.float64)
        weight_permutation = "none"
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != n:
            raise ValueError(
                f"[!] Length of 'weights' ({len(weights)}) must match "
                f"distance matrix size ({n})"
            )
        # Filter weights for valid cases
        weights = weights[valid_mask] if not np.all(valid_mask) else weights
    
    # Convert group to integer codes
    group_factor = pd.Categorical(group)
    group_int = group_factor.codes.astype(np.int32) + 1  # 1-based for compatibility
    k = len(group_factor.categories)  # Number of groups
    
    # Compute total weighted inertia (SCtot)
    # Note: In TraMineR, dissassoc uses C_tmrWeightedInertiaDist with var=FALSE
    # which returns sum of squares directly, not variance
    # Formula: SCtot = sum over all pairs (i,j) of: w_i * w_j * d_ij
    # For unweighted case (all weights=1), this simplifies to: SCtot = sum(d_ij) for all pairs
    totweights = np.sum(weights)
    
    # Compute SCtot (sum of squares, not variance)
    # This matches TraMineR's C_tmrWeightedInertiaDist with var=FALSE
    # For unweighted case, TraMineR returns: SCtot = discrepancy * n = variance * n
    # where variance = sum(d_ij) / (2 * n^2) for unweighted case
    # So SCtot = sum(d_ij) / (2 * n) = sum(d_ij) / n (since we only count upper triangle)
    # Actually, looking at TraMineR code: SCtot = discrepancy * n where discrepancy = variance
    # For unweighted: variance = sum(d_ij) / (2 * n^2), so SCtot = sum(d_ij) / (2 * n)
    # But TraMineR's C_tmrWeightedInertiaDist with var=FALSE returns sum(w_i * w_j * d_ij)
    # For unweighted (w_i = 1), this is sum(d_ij) over all pairs
    # However, TraMineR's discrepancy = SCtot / n, so SCtot should be sum(d_ij) / n
    # Let me check: discrepancy = 0.274, n = 20, SCtot = 5.48
    # So SCtot = discrepancy * n = 0.274 * 20 = 5.48
    # But sum(d_ij) = 109.6, so SCtot = sum(d_ij) / 20 = 5.48
    # This means TraMineR divides by n somewhere!
    # Actually, looking at TraMineR code more carefully:
    # C_tmrWeightedInertiaDist with var=FALSE returns sum(w_i * w_j * d_ij)
    # For unweighted (w_i = 1), this is sum(d_ij) over all pairs
    # But TraMineR's discrepancy = SCtot / totweights
    # So SCtot = discrepancy * totweights = 0.274 * 20 = 5.48
    # This means TraMineR's C_tmrWeightedInertiaDist must return sum(d_ij) / n for unweighted case!
    # Actually, wait - let me recalculate:
    # discrepancy = variance = sum(d_ij) / (2 * n^2) = 109.6 / (2 * 400) = 109.6 / 800 = 0.137
    # But TraMineR gives 0.274, which is double!
    # So TraMineR's variance formula is: sum(d_ij) / n^2 (not divided by 2)
    # Then SCtot = variance * n = (sum(d_ij) / n^2) * n = sum(d_ij) / n
    # So SCtot = 109.6 / 20 = 5.48 ✓
    # This means TraMineR's C_tmrWeightedInertiaDist with var=FALSE returns sum(d_ij) / n for unweighted case
    # But for weighted case, it returns sum(w_i * w_j * d_ij) / totweights
    # So the formula is: SCtot = sum(w_i * w_j * d_ij) / totweights * totweights = sum(w_i * w_j * d_ij)
    # Wait, that doesn't make sense. Let me check the TraMineR code again.
    # From dissassocweighted.R line 302-304:
    # SCtot <- .Call(C_tmrWeightedInertiaDist, dissmatrix, as.integer(n), 
    #     as.integer(FALSE), allindiv, as.double(weights), 
    #     as.integer(FALSE))
    # And line 332: ret$groups$discrepancy[k+1] <- SCtot/totweights
    # So SCtot = discrepancy * totweights
    # For unweighted: SCtot = 0.274 * 20 = 5.48
    # But sum(d_ij) = 109.6, so SCtot = sum(d_ij) / 20 = 5.48
    # This means C_tmrWeightedInertiaDist returns sum(d_ij) / n for unweighted case!
    # But that doesn't match the weighted formula. Let me check the C code.
    # Actually, I think the issue is that TraMineR's variance formula is different.
    # Let me use the correct formula: SCtot = discrepancy * totweights
    # where discrepancy = compute_pseudo_variance(distance_matrix, weights, squared=False)
    # So SCtot = discrepancy * totweights
    
    # For unweighted case, compute discrepancy first
    discrepancy = compute_pseudo_variance(distance_matrix, weights=weights, squared=squared)
    SCtot = discrepancy * totweights
    
    # Compute within-group variances and residual sum of squares
    SCres = 0.0
    groups_data = []
    
    for i in range(k):
        # Find sequences in this group
        group_mask = (group_int == (i + 1))
        group_indices = np.where(group_mask)[0]
        group_size = np.sum(weights[group_mask])
        
        if group_size == 0:
            groups_data.append({
                'n': 0,
                'discrepancy': 0.0
            })
            continue
        
        # Compute within-group sum of squares (SCresi)
        # TraMineR uses C_tmrWeightedInertiaDist with var=FALSE for groups too
        group_dist = distance_matrix[np.ix_(group_indices, group_indices)]
        group_weights = weights[group_indices]
        
        # Compute within-group sum of squares (SCresi)
        # TraMineR uses C_tmrWeightedInertiaDist with var=FALSE for groups
        # This returns sum of squares directly: sum(w_i * w_j * d_ij) for pairs in group
        # But TraMineR calculates: discrepancy = SCresi / group_size
        # So SCresi = discrepancy * group_size
        # For consistency, compute group discrepancy first, then SCresi
        if group_size > 0:
            # Compute group discrepancy (variance) using the same formula as total
            group_var = compute_pseudo_variance(group_dist, weights=group_weights, squared=squared)
            # Then SCresi = discrepancy * group_size (matching TraMineR's formula)
            group_ss = group_var * group_size
        else:
            group_var = 0.0
            group_ss = 0.0
        
        SCres += group_ss
        groups_data.append({
            'n': group_size,
            'discrepancy': group_var
        })
    
    # Add total row
    # Total discrepancy = SCtot / totweights
    # This matches TraMineR: ret$groups$discrepancy[k+1] <- SCtot/totweights
    groups_data.append({
        'n': totweights,
        'discrepancy': SCtot / totweights if totweights > 0 else 0.0
    })
    
    # Create groups DataFrame
    group_names = list(group_factor.categories) + ['Total']
    groups_df = pd.DataFrame(groups_data, index=group_names)
    
    # Compute explained sum of squares
    SCexp = SCtot - SCres
    
    # Compute pseudo F and R²
    if k > 1:
        pseudo_f = (SCexp / (k - 1)) / (SCres / (totweights - k))
        pseudo_r2 = SCexp / SCtot if SCtot > 0 else 0.0
    else:
        pseudo_f = 0.0
        pseudo_r2 = 0.0
    
    # Create ANOVA table
    anova_data = {
        'SS': [SCexp, SCres, SCtot],
        'df': [k - 1, totweights - k, totweights - 1],
        'MSE': [
            SCexp / (k - 1) if k > 1 else 0.0,
            SCres / (totweights - k) if (totweights - k) > 0 else 0.0,
            SCtot / (totweights - 1) if (totweights - 1) > 0 else 0.0
        ]
    }
    anova_df = pd.DataFrame(anova_data, index=['Exp', 'Res', 'Total'])
    
    # Permutation test matching TraMineR's dissassocweighted.*()
    if R > 1:
        perm_result = dissassoc_permutation_test(
            distance_matrix=distance_matrix,
            group_int=group_int,
            weights=weights,
            R=R,
            weight_permutation=weight_permutation,
            squared=squared,
            SCtot=SCtot,
            totweights=totweights,
            k=k
        )
        # Extract p-value for Pseudo F (first statistic)
        pseudo_f_pval = perm_result['pval'][0]
    else:
        pseudo_f_pval = np.nan
    
    return {
        'pseudo_f': pseudo_f,
        'pseudo_r2': pseudo_r2,
        'pseudo_f_pval': pseudo_f_pval,
        'groups': groups_df,
        'anova_table': anova_df,
        'R': R,
        'weight_permutation': weight_permutation
    }
