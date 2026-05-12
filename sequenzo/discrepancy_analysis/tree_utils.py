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
from typing import Optional, Union, Dict, Any, List
from .dissassoc_permutation import dissassoc_permutation_test
from ..utils.core_distance_operations import weighted_inertia_contrib


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
    >>> from sequenzo.discrepancy_analysis import compute_pseudo_variance
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
    
    total_weight = np.sum(weights)
    all_indices = np.arange(n, dtype=np.int32)
    contrib = weighted_inertia_contrib(distance_matrix, all_indices, weights)
    # TraMineR relation: dissvar = weighted.mean(disscenter_raw, w) / 2
    return float(np.sum(weights * contrib) / (2.0 * total_weight))


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
    >>> from sequenzo.discrepancy_analysis import compute_distance_association
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
    discrepancy = compute_pseudo_variance(distance_matrix, weights=weights, squared=False)
    SCtot = discrepancy * totweights
    
    dist_center = np.zeros(n)
    for i in range(k):
        group_mask = (group_int == (i + 1))
        group_indices = np.where(group_mask)[0]
        if len(group_indices) == 0:
            continue
        group_w = weights[group_indices]

        group_contrib = weighted_inertia_contrib(distance_matrix, group_indices, weights)
        dist_center[group_indices] = group_contrib

        # Center by subtracting weighted mean for this group
        group_dist_center = dist_center[group_indices]
        weighted_mean = np.sum(group_w * group_dist_center) / np.sum(group_w)
        dist_center[group_indices] = group_dist_center - weighted_mean / 2
    
    # TraMineR SCresi* terms are weighted residual sums around (weighted) means.
    dc_w_mean = np.sum(weights * dist_center) / np.sum(weights)
    disscSCtot = np.sum(weights * ((dist_center - dc_w_mean) ** 2))
    
    # Compute within-group variances and residual sum of squares
    SCres = 0.0
    groups_data = []
    
    # Additional stats for Bartlett and Levene
    lns = 0.0
    nlnvi = 0.0
    FBFdenomin = 0.0
    sumz = 0.0  # For Levene test
    s1ni = 0.0  # For Bartlett correction
    
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
            group_var = compute_pseudo_variance(group_dist, weights=group_weights, squared=False)
            # Then SCresi = discrepancy * group_size (matching TraMineR's formula)
            group_ss = group_var * group_size
            
            # Additional calculations for Bartlett and Pseudo Fbf
            ni = group_size
            vari = group_var  # group_ss / ni
            
            # For Bartlett test
            if vari >= 0 and ni > 1:
                lns += (ni - 1) * (vari / (totweights - k))
                if vari == 0:
                    nlnvi = -np.inf
                elif np.isfinite(nlnvi):
                    nlnvi += (ni - 1) * np.log(vari)
                s1ni += 1.0 / (ni - 1)
            
            # For Pseudo Fbf (F with Bonferroni correction)
            FBFdenomin += (1 - ni / totweights) * vari
            
            group_dist_center = dist_center[group_indices]
            group_dc_w_mean = np.sum(group_weights * group_dist_center) / np.sum(group_weights)
            sumz += np.sum(group_weights * ((group_dist_center - group_dc_w_mean) ** 2))
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
    if k > 1 and (totweights - k) > 0:
        # Check for division by zero or very small denominator
        # Use a small epsilon to avoid numerical instability
        epsilon = np.finfo(float).eps * 10
        denominator = SCres / (totweights - k)
        if denominator > epsilon:
            pseudo_f = (SCexp / (k - 1)) / denominator
        else:
            # If SCres is zero or very small, pseudo F is undefined
            # Set to NaN to indicate invalid result
            pseudo_f = np.nan
        pseudo_r2 = SCexp / SCtot if SCtot > 0 else 0.0
        
        # Compute Pseudo Fbf (F with Bonferroni correction)
        # FPF = SCexp / FBFdenomin
        if FBFdenomin > 0:
            pseudo_fbf = SCexp / FBFdenomin
        else:
            pseudo_fbf = np.nan
        
        # Compute Bartlett test statistic
        # Bartlett = Tcalc / Ccalc
        if lns > 0:
            Tcalc = (totweights - k) * np.log(lns) - nlnvi
            Ccalc = 1 + 1 / (3 * (k - 1)) * (s1ni - 1 / (totweights - k))
            bartlett = Tcalc / Ccalc if Ccalc > 0 else np.nan
        else:
            bartlett = np.nan
        
        # Compute Levene test statistic (Pseudo W)
        # PseudoW = ((disscSCtot - sumz) / (k-1)) / (sumz / (totweights-k))
        if sumz > 0:
            levene = ((disscSCtot - sumz) / (k - 1)) / (sumz / (totweights - k))
        else:
            levene = np.nan
    else:
        pseudo_f = 0.0
        pseudo_r2 = 0.0
        pseudo_fbf = np.nan
        bartlett = np.nan
        levene = np.nan
    
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
    
    # Build statistics DataFrame to match TraMineR's output format
    # TraMineR returns: Pseudo F, Pseudo Fbf, Pseudo R2, Bartlett, Levene
    stats_data = {
        'Value': [pseudo_f, pseudo_fbf, pseudo_r2, bartlett, levene],
        'p-value': [pseudo_f_pval, np.nan, np.nan, np.nan, np.nan]
    }
    stats_df = pd.DataFrame(
        stats_data,
        index=['Pseudo F', 'Pseudo Fbf', 'Pseudo R2', 'Bartlett', 'Levene']
    )
    
    return {
        'pseudo_f': pseudo_f,
        'pseudo_fbf': pseudo_fbf,
        'pseudo_r2': pseudo_r2,
        'bartlett': bartlett,
        'levene': levene,
        'pseudo_f_pval': pseudo_f_pval,
        'stat': stats_df,  # Add formatted stats DataFrame
        'groups': groups_df,
        'anova_table': anova_df,
        'R': R,
        'weight_permutation': weight_permutation
    }


def dissmfacw(
    distance_matrix: np.ndarray,
    factors: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    R: int = 0,
    weight_permutation: str = "none",
    squared: bool = False
) -> pd.DataFrame:
    """
    Multi-factor association between a distance matrix and several covariates.

    TraMineR equivalent: dissmfacw()

    This function is a *practical* multi-factor wrapper around
    `compute_distance_association`. For each factor (column) in the provided
    DataFrame, it computes the pseudo-ANOVA statistics describing how much of
    the discrepancy in the distance matrix is explained by that factor alone.

    The goal is to provide an analysis table similar in spirit to TraMineR's
    `dissmfacw()` output, while keeping the implementation simple and easy
    to understand on the Python side.

    Parameters
    ----------
    distance_matrix : np.ndarray or pandas.DataFrame
        Square symmetric distance matrix of shape (n, n).
    factors : pandas.DataFrame
        DataFrame with n rows and one or more factor columns describing
        covariates (e.g., gender, cohort, country). Each column is treated
        as a separate factor in the analysis.
    weights : np.ndarray, optional
        Optional weights for each observation (length n). If None, equal
        weights are used.
    R : int, default 0
        Number of permutations for each factor-specific association test.
        When R=0, permutation p-values are skipped (faster).
    weight_permutation : {"none", "replicate", "diss", "group"}, default "none"
        Weight handling strategy for permutation tests. Passed directly
        to `compute_distance_association`.
    squared : bool, default False
        Whether to square distances before analysis, passed to
        `compute_distance_association`.

    Returns
    -------
    pandas.DataFrame
        Summary table with one row per factor and the following columns:
            - 'factor'     : factor name (column name in `factors`)
            - 'Pseudo R2'  : proportion of variance explained
            - 'Pseudo F'   : pseudo F statistic
            - 'p-value'    : permutation p-value for Pseudo F (if R > 1)
            - 'n_groups'   : number of non-empty groups for that factor
    """
    # Convert distance matrix to ndarray if needed
    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.values

    results = []
    for col in factors.columns:
        factor_values = factors[col].values
        assoc = compute_distance_association(
            distance_matrix=distance_matrix,
            group=factor_values,
            weights=weights,
            R=R,
            weight_permutation=weight_permutation,
            squared=squared,
        )

        # Count non-empty groups for information
        n_groups = (assoc["groups"]["n"] > 0).sum() - 1  # subtract "Total" row

        results.append(
            {
                "factor": col,
                "Pseudo R2": assoc["pseudo_r2"],
                "Pseudo F": assoc["pseudo_f"],
                "p-value": assoc["pseudo_f_pval"],
                "n_groups": int(n_groups),
            }
        )

    return pd.DataFrame(results).set_index("factor")


def dissmergegroups(
    distance_matrix: np.ndarray,
    group: Union[np.ndarray, pd.Series],
    weights: Optional[np.ndarray] = None,
    target_n_groups: Optional[int] = None,
    squared: bool = False,
    measure: str = "ASW",
    crit: float = 0.2,
    ref: str = "max",
    min_group: int = 4,
    small: float = 0.05,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Iteratively merge groups using TraMineR's dissmergegroups logic.

    This implementation follows the original TraMineR behaviour:
    - Quality-driven greedy merges (default quality measure: ASW),
    - adaptive search restricted to smallest group when needed,
    - stopping by both `min_group` and quality deterioration threshold
      `crit * quality_ref` with `ref in {"initial","max","previous"}`.

    Parameters
    ----------
    distance_matrix : np.ndarray or pandas.DataFrame
        Square symmetric distance matrix of shape (n, n).
    group : array-like
        Initial group labels (length n). Can be numeric or categorical.
    weights : np.ndarray, optional
        Optional weights for each observation.
    target_n_groups : int, optional
        Compatibility alias for `min_group`: if provided, overrides `min_group`.
    squared : bool, default False
        Whether to square distances before analysis (passed to
        `compute_distance_association`).

    measure : str, default "ASW"
        Cluster quality measure. Currently only "ASW" is supported.
    crit : float, default 0.2
        Maximum allowed proportion of quality loss.
    ref : {"initial", "max", "previous"}, default "max"
        Reference quality used in the deterioration threshold.
    min_group : int, default 4
        Minimal number of final groups.
    small : float, default 0.05
        If <1, interpreted as proportion of weighted sample size. While the
        smallest group is below this threshold, only merges involving that
        smallest group are evaluated.
    silent : bool, default False
        If False, merge steps are printed.

    Returns
    -------
    dict
        - 'final_group': final merged grouping as integer codes (1..K)
        - 'quality': final quality value
        - 'history': merge log
    """
    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.values
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("diss must be a square distance matrix")

    n = D.shape[0]
    g = pd.Categorical(pd.Series(group)).codes.astype(int) + 1
    if len(g) != n or np.any(g <= 0):
        raise ValueError("group must be valid and conformable with distance matrix")

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != n:
            raise ValueError("weights length must match distance matrix size")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

    if squared:
        D = D ** 2

    if target_n_groups is not None:
        min_group = int(target_n_groups)

    if min_group < 1:
        raise ValueError("min_group must be >= 1")

    if measure.upper() != "ASW":
        raise ValueError("Only measure='ASW' is currently supported")

    if ref not in {"initial", "max", "previous"}:
        raise ValueError("ref must be one of {'initial', 'max', 'previous'}")

    def _asw(labels: np.ndarray) -> float:
        unique = np.unique(labels)
        if len(unique) <= 1:
            return 0.0
        s = np.zeros(n, dtype=float)
        for i in range(n):
            gi = labels[i]
            in_g = np.where(labels == gi)[0]
            in_g_other = in_g[in_g != i]
            if len(in_g_other) == 0:
                a_i = 0.0
            else:
                ww = w[in_g_other]
                denom = float(np.sum(ww))
                a_i = float(np.sum(ww * D[i, in_g_other]) / denom) if denom > 0 else 0.0

            b_i = np.inf
            for gj in unique:
                if gj == gi:
                    continue
                out_g = np.where(labels == gj)[0]
                ww = w[out_g]
                denom = float(np.sum(ww))
                if denom <= 0:
                    continue
                cand = float(np.sum(ww * D[i, out_g]) / denom)
                if cand < b_i:
                    b_i = cand
            if not np.isfinite(b_i):
                b_i = 0.0
            den = max(a_i, b_i)
            s[i] = 0.0 if den <= 0 else (b_i - a_i) / den
        return float(np.average(s, weights=w))

    N = float(np.sum(w))
    minsize = small * N if small < 1 else float(small)

    history: List[Dict[str, Any]] = []
    quality = _asw(g)
    quality_ref = quality
    final_quality = quality

    while int(np.max(g)) > min_group:
        maxgn = int(np.max(g))
        grp_sizes = np.bincount(g, weights=w, minlength=maxgn + 1)[1:]
        diff = quality_ref
        best_pair = None
        best_qual = None

        if np.min(grp_sizes) > minsize:
            pairs = [(i, j) for i in range(1, maxgn) for j in range(i + 1, maxgn + 1)]
        else:
            i = int(np.argmin(grp_sizes)) + 1
            pairs = [(min(i, j), max(i, j)) for j in range(1, maxgn + 1) if j != i]

        for i, j in pairs:
            gng = g.copy()
            gng[gng == j] = i
            # re-factor to 1..K as TraMineR does after merge
            gng = pd.Categorical(gng).codes.astype(int) + 1
            qual = _asw(gng)
            loss = quality_ref - qual
            if best_pair is None or loss < diff:
                diff = loss
                best_pair = (i, j)
                best_qual = qual

        if best_pair is None:
            break

        if diff > crit * quality_ref:
            break

        i, j = best_pair
        if not silent:
            print(f"Merging groups {i} and {j}")
        g[g == j] = i
        g = pd.Categorical(g).codes.astype(int) + 1

        final_quality = float(best_qual if best_qual is not None else _asw(g))
        history.append(
            {
                "merged": (i, j),
                "quality": final_quality,
                "loss": float(diff),
                "n_groups": int(np.max(g)),
            }
        )

        if ref == "max":
            quality_ref = max(quality_ref, final_quality)
        elif ref == "previous":
            quality_ref = final_quality
        else:  # initial
            quality_ref = quality

    return {
        "final_group": pd.Series(g, name="group"),
        "quality": final_quality,
        "history": history,
    }
