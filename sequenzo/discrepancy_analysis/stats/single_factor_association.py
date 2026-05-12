"""Single-factor distance–covariate association (TraMineR: dissassoc)."""

import numpy as np
import pandas as pd
from typing import Optional
from ..internal.weighted_inertia import (
    weighted_inertia_sum,
    distance_to_center_contributions,
    weighted_residuals_z,
    unweighted_residuals_z,
)
from ..internal.single_factor_permutation import association_permutation_test
from ..internal.weight_permutation import resolve_weight_permutation


def single_factor_association(
    distance_matrix: np.ndarray,
    group: np.ndarray,
    weights: Optional[np.ndarray] = None,
    R: int = 1000,
    weight_permutation: Optional[str] = None,
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
        - "replicate": Replicate cases according to weights (weights must be integers;
          default when weights are supplied and ``weight_permutation`` is omitted)
        - "diss": Permute labels while weights enter the statistic (recommended for
          survey or calibration weights)
        - "group": Permute at group level
        - "none": Unweighted permutation test (used automatically when weights is None)
        Default: None (resolved to "none" without weights, otherwise "replicate")
        
    squared : bool, optional
        If True, use exponent v=2 on dissimilarities before analysis (generalized
        sum of squares). Default False follows Studer et al. (2011) and uses
        nonsquared dissimilarities (v=1), which is usually preferable for OM, LCS,
        and other non-Euclidean sequence distances.
        
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
    >>> from sequenzo.discrepancy_analysis import single_factor_association
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
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != n:
            raise ValueError(
                f"[!] Length of 'weights' ({len(weights)}) must match "
                f"distance matrix size ({n})"
            )
        # Filter weights for valid cases
        weights = weights[valid_mask] if not np.all(valid_mask) else weights

    weight_permutation = resolve_weight_permutation(
        None if unweighted else weights,
        weight_permutation,
    )
    
    # Convert group to integer codes
    group_factor = pd.Categorical(group)
    group_int = group_factor.codes.astype(np.int32) + 1  # 1-based for compatibility
    k = len(group_factor.categories)  # Number of groups
    
    # Total inertia sum (SS_T) and group-level residual sums (SS_W)
    all_indices = np.arange(n, dtype=np.int32)
    sc_tot = weighted_inertia_sum(
        distance_matrix,
        all_indices,
        weights=weights,
        squared=False,
    )
    totweights = float(np.sum(weights))

  # Centered distance-to-center contributions for the Levene statistic
    dist_center = distance_to_center_contributions(
        distance_matrix,
        group_int,
        weights,
        k,
    )
    if unweighted:
        dissc_sc_tot = unweighted_residuals_z(dist_center)
    else:
        dissc_sc_tot = weighted_residuals_z(dist_center, weights)

    sc_res = 0.0
    groups_data = []
    lns = 0.0
    nlnvi = 0.0
    fbf_denomin = 0.0
    sumz = 0.0
    s1ni = 0.0

    for i in range(k):
        group_mask = (group_int == (i + 1))
        group_indices = np.where(group_mask)[0]
        group_size = float(np.sum(weights[group_mask]))

        if group_size == 0:
            groups_data.append({"n": 0, "discrepancy": 0.0})
            continue

        group_ss = weighted_inertia_sum(
            distance_matrix,
            group_indices,
            weights=weights,
            squared=False,
        )
        group_var = group_ss / group_size if group_size > 0 else 0.0
        sc_res += group_ss

        if group_var >= 0 and group_size > 1:
            lns += (group_size - 1.0) * (group_var / (totweights - k))
            if group_var == 0:
                nlnvi = -np.inf
            elif np.isfinite(nlnvi):
                nlnvi += (group_size - 1.0) * np.log(group_var)
            s1ni += 1.0 / (group_size - 1.0)

        fbf_denomin += (1.0 - group_size / totweights) * group_var

        group_dist_center = dist_center[group_indices]
        if unweighted:
            group_dc_mean = float(np.mean(group_dist_center))
            sumz += float(np.sum((group_dist_center - group_dc_mean) ** 2))
        else:
            group_weights = weights[group_indices]
            group_dc_mean = float(np.sum(group_weights * group_dist_center) / np.sum(group_weights))
            sumz += float(np.sum(group_weights * ((group_dist_center - group_dc_mean) ** 2)))

        groups_data.append({"n": group_size, "discrepancy": group_var})
    
    # Add total row
    # Total discrepancy = SCtot / totweights
    # This matches TraMineR: ret$groups$discrepancy[k+1] <- SCtot/totweights
    groups_data.append({
        'n': totweights,
        'discrepancy': sc_tot / totweights if totweights > 0 else 0.0
    })
    
    # Create groups DataFrame
    group_names = list(group_factor.categories) + ['Total']
    groups_df = pd.DataFrame(groups_data, index=group_names)
    
    sc_exp = sc_tot - sc_res
    
    # Compute pseudo F and R²
    if k > 1 and (totweights - k) > 0:
        epsilon = np.finfo(float).eps * 10
        denominator = sc_res / (totweights - k)
        if denominator > epsilon:
            pseudo_f = (sc_exp / (k - 1)) / denominator
        else:
            pseudo_f = np.nan
        pseudo_r2 = sc_exp / sc_tot if sc_tot > 0 else 0.0
        
        if fbf_denomin > 0:
            pseudo_fbf = sc_exp / fbf_denomin
        else:
            pseudo_fbf = np.nan
        
        if lns > 0:
            tcalc = (totweights - k) * np.log(lns) - nlnvi
            ccalc = 1 + 1 / (3 * (k - 1)) * (s1ni - 1 / (totweights - k))
            bartlett = tcalc / ccalc if ccalc > 0 else np.nan
        else:
            bartlett = np.nan
        
        if sumz > 0:
            levene = ((dissc_sc_tot - sumz) / (k - 1)) / (sumz / (totweights - k))
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
        'SS': [sc_exp, sc_res, sc_tot],
        'df': [k - 1, totweights - k, totweights - 1],
        'MSE': [
            sc_exp / (k - 1) if k > 1 else 0.0,
            sc_res / (totweights - k) if (totweights - k) > 0 else 0.0,
            sc_tot / (totweights - 1) if (totweights - 1) > 0 else 0.0
        ]
    }
    anova_df = pd.DataFrame(anova_data, index=['Exp', 'Res', 'Total'])
    
    # Permutation test matching TraMineR's dissassocweighted.*()
    if R > 1:
        perm_result = association_permutation_test(
            distance_matrix=distance_matrix,
            group_int=group_int,
            weights=weights,
            R=R,
            weight_permutation=weight_permutation,
            squared=squared,
            sc_tot=sc_tot,
            totweights=totweights,
            k=k,
            dist_center=dist_center,
            dissc_sc_tot=dissc_sc_tot,
        )
        pseudo_f_pval = perm_result['pval'][0]
        stat_pvals = perm_result['pval']
    else:
        pseudo_f_pval = np.nan
        stat_pvals = np.full(5, np.nan)
    
    stats_data = {
        'Value': [pseudo_f, pseudo_fbf, pseudo_r2, bartlett, levene],
        'p-value': list(stat_pvals),
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
