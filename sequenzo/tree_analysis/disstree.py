"""
@Author  : Yuqi Liang 梁彧祺
@File    : disstree.py
@Time    : 2026-02-09 16:30
@Desc    : Core implementation of distance-based regression tree.

           Corresponds to TraMineR function: disstree()
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any
from itertools import combinations

from .tree_node import DissTreeNode, DissTreeSplit
from .tree_utils import compute_pseudo_variance, compute_distance_association
from .permutation import test_tree_split_significance

# Global counter for node IDs
_node_counter = 1


def _reset_node_counter():
    """Reset the global node counter (for testing)."""
    global _node_counter
    _node_counter = 1


def _get_next_node_id() -> int:
    """Get the next unique node ID."""
    global _node_counter
    node_id = _node_counter
    _node_counter += 1
    return node_id


def build_distance_tree(
    distance_matrix: np.ndarray,
    predictors: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    min_size: Union[float, int] = 0.05,
    max_depth: int = 5,
    R: int = 1000,
    pval: float = 0.01,
    weight_permutation: str = "replicate",
    squared: bool = False,
    first_split: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a distance-based regression tree.
    
    This function recursively partitions sequences based on covariates to
    explain differences in sequence patterns. It uses pseudo-variance reduction
    to select optimal splits and permutation tests to assess significance.
    
    **Corresponds to TraMineR function: `disstree()`**
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Square symmetric distance matrix of shape (n, n)
        
    predictors : pd.DataFrame
        DataFrame with covariates (predictor variables) for each sequence.
        Shape should be (n, p) where p is the number of predictors.
        
    weights : np.ndarray, optional
        Optional weights for each sequence. Shape should be (n,).
        Default: None (equal weights)
        
    min_size : float or int, optional
        Minimum number of cases in a node. If float < 1, treated as proportion.
        Default: 0.05 (5% of total)
        
    max_depth : int, optional
        Maximum depth of the tree.
        Default: 5
        
    R : int, optional
        Number of permutations for significance testing.
        Default: 1000
        
    pval : float, optional
        Maximum allowed p-value for a split to be retained.
        Default: 0.01
        
    weight_permutation : str, optional
        Method for handling weights in permutation tests. Options:
        - "replicate": Replicate cases according to weights
        - "diss": Attach weights to distance matrix
        - "group": Permute at group level
        - "none": Unweighted permutation test
        Default: "replicate"
        
    squared : bool, optional
        If True, square the distance matrix before analysis.
        Default: False
        
    first_split : str, optional
        Name of variable to force as the first split.
        Default: None
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'root': DissTreeNode object (root of the tree)
        - 'fitted': DataFrame with leaf membership for each sequence
        - 'info': Dictionary with tree metadata
        - 'data': Original predictor DataFrame
        - 'weights': Weights array
        - 'terms': Information about formula terms
        
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sequenzo.tree_analysis import build_distance_tree
    >>> 
    >>> # Create distance matrix and predictors
    >>> dist_matrix = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
    >>> predictors = pd.DataFrame({'gender': ['M', 'F', 'M'], 'age': [20, 25, 30]})
    >>> 
    >>> # Build tree
    >>> tree = build_distance_tree(dist_matrix, predictors, R=10, pval=0.1)
    >>> print(f"Tree depth: {tree['info']['max_depth']}")
    
    References
    ----------
    Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011).
    Discrepancy analysis of state sequences.
    Sociological Methods and Research, Vol. 40(3), 471-510.
    """
    # Reset node counter
    _reset_node_counter()
    
    # Convert to numpy array if needed
    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.values
    
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = distance_matrix.shape[0]
    
    # Validate inputs
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("[!] 'distance_matrix' must be square")
    
    if len(predictors) != n:
        raise ValueError(
            f"[!] Number of rows in 'predictors' ({len(predictors)}) "
            f"must match distance matrix size ({n})"
        )
    
    # Square matrix if requested
    if squared:
        distance_matrix = distance_matrix ** 2
    
    # Handle weights
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
        weight_permutation = "none"
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != n:
            raise ValueError(
                f"[!] Length of 'weights' ({len(weights)}) must match "
                f"distance matrix size ({n})"
            )
    
    # Convert min_size to absolute number if proportion
    total_weight = np.sum(weights)
    if min_size < 1:
        min_size = total_weight * min_size
    
    # Validate p-value
    if R <= 1:
        pval = 1.0
    elif pval < (1.0 / R):
        print(f"[!] Warning: Minimum possible p-value with R={R} is {1.0/R}. "
              f"Setting pval to {1.0/R}")
        pval = 1.0 / R
    
    # Handle first_split
    first_var_index = None
    if first_split is not None:
        if first_split not in predictors.columns:
            raise ValueError(
                f"[!] 'first_split' variable '{first_split}' not found in predictors"
            )
        first_var_index = predictors.columns.get_loc(first_split)
    
    # Compute overall variance
    # Note: For tree building, we need variance (not sum of squares)
    # This matches TraMineR's dissvar() which uses var=TRUE
    overall_variance = compute_pseudo_variance(
        distance_matrix, weights=weights, squared=False
    )
    
    # Build tree recursively
    # Reset index to ensure sequential 0-based indexing
    predictors_reset = predictors.reset_index(drop=True)
    all_indices = np.arange(n, dtype=np.int32)
    root = _build_node(
        distance_matrix=distance_matrix,
        predictors=predictors_reset,
        min_size=min_size,
        indices=all_indices,
        variance=overall_variance,
        depth=1,
        max_depth=max_depth,
        R=R,
        pval=pval,
        weights=weights,
        weight_permutation=weight_permutation,
        squared=squared,
        first_var_index=first_var_index
    )
    
    # Get leaf memberships
    leaf_memberships = _get_leaf_memberships(root, n)
    fitted_df = pd.DataFrame({'(fitted)': leaf_memberships})
    
    # Compute global statistics
    global_assoc = compute_distance_association(
        distance_matrix=distance_matrix,
        group=leaf_memberships,
        weights=weights,
        R=R,
        weight_permutation=weight_permutation,
        squared=False
    )
    
    # Build result dictionary
    result = {
        'root': root,
        'fitted': fitted_df,
        'info': {
            'method': 'disstree',
            'n': total_weight,
            'parameters': {
                'min_size': min_size,
                'max_depth': max_depth,
                'R': R,
                'pval': pval
            },
            'weight_permutation': weight_permutation,
            'adjustment': global_assoc
        },
        'data': predictors.copy(),
        'weights': weights.copy()
    }
    
    return result


def _build_node(
    distance_matrix: np.ndarray,
    predictors: pd.DataFrame,
    min_size: float,
    indices: np.ndarray,
    variance: float,
    depth: int,
    max_depth: int,
    R: int,
    pval: float,
    weights: np.ndarray,
    weight_permutation: str,
    squared: bool = False,
    first_var_index: Optional[int] = None
) -> DissTreeNode:
    """
    Recursively build a tree node.
    
    This is an internal function called by build_distance_tree().
    Corresponds to TraMineR function: DTNBuildNode()
    """
    n = len(indices)
    node_weight = np.sum(weights[indices])
    
    # Find medoid (sequence closest to center)
    # For now, use first sequence as placeholder
    # TODO: Implement proper medoid calculation
    medoid_idx = indices[0]
    
    # Create node
    node = DissTreeNode(
        node_id=_get_next_node_id(),
        indices=indices,
        variance=variance,
        depth=depth,
        medoid=medoid_idx
    )
    node.info['n'] = node_weight
    
    # Check stopping conditions
    # SCtot is sum of squares = variance * node_weight
    # This matches TraMineR's calculation: SCtot = vardis * node$info$n
    SCtot = variance * node_weight
    
    if depth >= max_depth:
        return node
    
    if node_weight < min_size * 2:  # Need at least min_size for each child
        return node
    
    # Find best split
    best_split = None
    best_SCres = SCtot
    
    if first_var_index is not None and depth == 1:
        # Force first split on specified variable
        best_split = _find_best_split(
            distance_matrix=distance_matrix,
            predictors=predictors,
            predictor_col=first_var_index,
            indices=indices,
            current_SCres=best_SCres,
            min_size=min_size,
            weights=weights
        )
        if best_split is not None:
            best_SCres = best_split['split'].info['SCres']
    else:
        # Try all predictors
        for col_idx in range(len(predictors.columns)):
            split_result = _find_best_split(
                distance_matrix=distance_matrix,
                predictors=predictors,
                predictor_col=col_idx,
                indices=indices,
                current_SCres=best_SCres,
                min_size=min_size,
                weights=weights
            )
            
            if split_result is not None:
                if best_split is None or split_result['split'].info['SCres'] < best_SCres:
                    best_split = split_result
                    best_SCres = split_result['split'].info['SCres']
    
    # If no valid split found, return terminal node
    if best_split is None:
        return node
    
    # Test significance with permutation test
    if R > 1:
        # Create binary grouping for permutation test
        # best_split['variable'] is already a boolean mask
        group_binary = best_split['variable']
        
        # Run permutation test matching TraMineR's DTNdissassocweighted()
        p_value = test_tree_split_significance(
            distance_matrix=distance_matrix,
            group=group_binary,
            indices=indices,
            weights=weights,
            R=R,
            weight_permutation=weight_permutation,
            squared=squared
        )
        
        if p_value > pval:
            return node  # Split not significant
        
        best_split['split'].info['pval'] = p_value
    
    # Compute R²
    R2 = 1.0 - (best_SCres / SCtot) if SCtot > 0 else 0.0
    best_split['split'].info['R2'] = R2
    
    # Set split on node
    node.split = best_split['split']
    
    # Recursively build children
    left_mask = best_split['variable']
    right_mask = ~left_mask
    
    left_indices = indices[left_mask]
    right_indices = indices[right_mask]
    
    left_variance = best_split['split'].info['lvar']
    right_variance = best_split['split'].info['rvar']
    
    # Get subset of predictors for left child
    # Reset index to ensure sequential 0-based indexing for recursive calls
    left_predictors = predictors.iloc[left_mask].reset_index(drop=True).copy()
    right_predictors = predictors.iloc[right_mask].reset_index(drop=True).copy()
    
    left_child = _build_node(
        distance_matrix=distance_matrix,
        predictors=left_predictors,
        min_size=min_size,
        indices=left_indices,
        variance=left_variance,
        depth=depth + 1,
        max_depth=max_depth,
        R=R,
        pval=pval,
        weights=weights,
        weight_permutation=weight_permutation,
        squared=squared,
        first_var_index=None
    )
    
    right_child = _build_node(
        distance_matrix=distance_matrix,
        predictors=right_predictors,
        min_size=min_size,
        indices=right_indices,
        variance=right_variance,
        depth=depth + 1,
        max_depth=max_depth,
        R=R,
        pval=pval,
        weights=weights,
        weight_permutation=weight_permutation,
        squared=squared,
        first_var_index=None
    )
    
    node.kids = [left_child, right_child]
    
    return node


def _find_best_split(
    distance_matrix: np.ndarray,
    predictors: pd.DataFrame,
    predictor_col: int,
    indices: np.ndarray,
    current_SCres: float,
    min_size: float,
    weights: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    Find the best binary split for a given predictor variable.
    
    Corresponds to TraMineR function: DTNGroupFactorBinary()
    """
    # Get predictor values
    # predictors is already subset and reset_index(drop=True) for the current node
    # indices contains the original indices into the full distance matrix (for distance matrix access)
    # Since predictors is reset_index(drop=True), it has sequential 0-based indexing
    # So we can directly use iloc with all rows (predictors already matches the current node)
    if len(predictors) != len(indices):
        raise ValueError(
            f"[!] Mismatch: predictors has {len(predictors)} rows, "
            f"but indices has {len(indices)} elements"
        )
    
    # Use iloc with all rows (predictors is already filtered and reset_index)
    predictor_values = predictors.iloc[:, predictor_col].values
    predictor_name = predictors.columns[predictor_col]
    
    # Handle missing values
    has_missing = pd.isna(predictor_values).any()
    
    # Convert to factor/categorical
    if pd.api.types.is_numeric_dtype(predictors.iloc[:, predictor_col]):
        # Continuous variable - try all possible cutpoints
        unique_vals = np.unique(predictor_values[~pd.isna(predictor_values)])
        if len(unique_vals) <= 1:
            return None
        
        best_split = None
        best_SCres = current_SCres
        
        for cutpoint in unique_vals[:-1]:  # Don't need last value
            left_mask = predictor_values <= cutpoint
            if has_missing:
                left_mask[pd.isna(predictor_values)] = False
            
            right_mask = ~left_mask
            if has_missing:
                right_mask[pd.isna(predictor_values)] = False
            
            # Check minimum size
            left_weight = np.sum(weights[indices[left_mask]])
            right_weight = np.sum(weights[indices[right_mask]])
            
            if left_weight < min_size or right_weight < min_size:
                continue
            
            # Compute variances for left and right groups
            # left_mask and right_mask are boolean masks over the current node's sequences
            # Map these to original indices for distance matrix access
            left_indices_original = indices[left_mask]
            right_indices_original = indices[right_mask]
            
            if len(left_indices_original) == 0 or len(right_indices_original) == 0:
                continue
            
            left_dist = distance_matrix[np.ix_(left_indices_original, left_indices_original)]
            right_dist = distance_matrix[np.ix_(right_indices_original, right_indices_original)]
            left_w = weights[left_indices_original]
            right_w = weights[right_indices_original]
            
            # Compute sum of squares for left and right groups
            # TraMineR uses C_tmrWeightedInertiaDist with var=FALSE for groups
            left_ss = 0.0
            for i_idx in range(len(left_indices_original)):
                for j_idx in range(i_idx + 1, len(left_indices_original)):
                    left_ss += left_w[i_idx] * left_w[j_idx] * left_dist[i_idx, j_idx]
            
            right_ss = 0.0
            for i_idx in range(len(right_indices_original)):
                for j_idx in range(i_idx + 1, len(right_indices_original)):
                    right_ss += right_w[i_idx] * right_w[j_idx] * right_dist[i_idx, j_idx]
            
            # SCres is sum of squares (not variance)
            SCres = left_ss + right_ss
            
            # Compute variances for info
            left_var = left_ss / left_weight if left_weight > 0 else 0.0
            right_var = right_ss / right_weight if right_weight > 0 else 0.0
            
            if SCres < best_SCres:
                best_SCres = SCres
                split_info = {
                    'lpop': left_weight,
                    'rpop': right_weight,
                    'lvar': left_var,
                    'rvar': right_var,
                    'SCres': SCres
                }
                
                split = DissTreeSplit(
                    varindex=predictor_col,
                    breaks=cutpoint,
                    prob=np.array([left_weight, right_weight]) / (left_weight + right_weight),
                    info=split_info
                )
                
                if has_missing:
                    # Assign missing to group with larger weight
                    split.naGroup = 1 if left_weight >= right_weight else 2
                    # Update masks to include missing values
                    if split.naGroup == 1:
                        left_mask[pd.isna(predictor_values)] = True
                        right_mask = ~left_mask
                    else:
                        right_mask[pd.isna(predictor_values)] = True
                        left_mask = ~right_mask
                
                best_split = {
                    'split': split,
                    'variable': left_mask
                }
        
        return best_split
    
    else:
        # Categorical variable - try all possible binary groupings
        factor = pd.Categorical(predictor_values)
        levels = factor.categories
        n_levels = len(levels)
        
        if n_levels <= 1:
            return None
        
        # Build group conditions
        group_conditions = []
        group_sizes = []
        
        for level in levels:
            mask = (factor == level)
            group_conditions.append(mask)
            group_sizes.append(np.sum(weights[indices[mask]]))
        
        if has_missing:
            missing_mask = pd.isna(predictor_values)
            group_conditions.append(missing_mask)
            group_sizes.append(np.sum(weights[indices[missing_mask]]))
            n_levels += 1
        
        # Try all binary combinations
        best_split = None
        best_SCres = current_SCres
        
        # For categorical, try combinations up to half the levels
        max_comb_size = (n_levels + 1) // 2
        
        for comb_size in range(1, max_comb_size + 1):
            for left_groups in combinations(range(n_levels), comb_size):
                left_groups = list(left_groups)
                right_groups = [i for i in range(n_levels) if i not in left_groups]
                
                # Build masks (over predictor_values, which matches indices length)
                left_mask = np.zeros(len(predictor_values), dtype=bool)
                for g in left_groups:
                    left_mask |= group_conditions[g]
                
                right_mask = ~left_mask
                
                # Check minimum size
                left_weight = np.sum(weights[indices[left_mask]])
                right_weight = np.sum(weights[indices[right_mask]])
                
                if left_weight < min_size or right_weight < min_size:
                    continue
                
                # Compute variances
                # Map boolean masks to original indices for distance matrix access
                left_indices_original = indices[left_mask]
                right_indices_original = indices[right_mask]
                
                if len(left_indices_original) == 0 or len(right_indices_original) == 0:
                    continue
                
                left_dist = distance_matrix[np.ix_(left_indices_original, left_indices_original)]
                right_dist = distance_matrix[np.ix_(right_indices_original, right_indices_original)]
                left_w = weights[left_indices_original]
                right_w = weights[right_indices_original]
                
                # Compute sum of squares for left and right groups
                left_ss = 0.0
                for i_idx in range(len(left_indices_original)):
                    for j_idx in range(i_idx + 1, len(left_indices_original)):
                        left_ss += left_w[i_idx] * left_w[j_idx] * left_dist[i_idx, j_idx]
                
                right_ss = 0.0
                for i_idx in range(len(right_indices_original)):
                    for j_idx in range(i_idx + 1, len(right_indices_original)):
                        right_ss += right_w[i_idx] * right_w[j_idx] * right_dist[i_idx, j_idx]
                
                # SCres is sum of squares (not variance)
                SCres = left_ss + right_ss
                
                # Compute variances for info
                left_var = left_ss / left_weight if left_weight > 0 else 0.0
                right_var = right_ss / right_weight if right_weight > 0 else 0.0
                
                if SCres < best_SCres:
                    best_SCres = SCres
                    split_info = {
                        'lpop': left_weight,
                        'rpop': right_weight,
                        'lvar': left_var,
                        'rvar': right_var,
                        'SCres': SCres
                    }
                    
                    # Create index array for which groups go left (1) or right (2)
                    index_array = np.zeros(n_levels, dtype=np.int32)
                    for g in left_groups:
                        index_array[g] = 1
                    for g in right_groups:
                        index_array[g] = 2
                    
                    split = DissTreeSplit(
                        varindex=predictor_col,
                        index=index_array,
                        labels=list(levels) + (['<Missing>'] if has_missing else []),
                        prob=np.array([left_weight, right_weight]) / (left_weight + right_weight),
                        info=split_info
                    )
                    
                    if has_missing:
                        # Missing group is the last one (n_levels - 1)
                        split.naGroup = 1 if (n_levels - 1) in left_groups else 2
                        # Update masks to include missing values
                        if split.naGroup == 1:
                            left_mask[pd.isna(predictor_values)] = True
                            right_mask = ~left_mask
                        else:
                            right_mask[pd.isna(predictor_values)] = True
                            left_mask = ~right_mask
                    
                    best_split = {
                        'split': split,
                        'variable': left_mask
                    }
        
        return best_split


def _simple_permutation_test(
    distance_matrix: np.ndarray,
    group: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    R: int,
    weight_permutation: str = "none",
    squared: bool = False
) -> float:
    """
    Internal wrapper for testing tree split significance with permutation tests.
    
    This is an internal helper function that calls the main permutation test
    function. It provides a simple interface for use within the tree building
    process.
    
    **Corresponds to TraMineR function: `DTNdissassocweighted()`**
    
    For the actual implementation, see `test_tree_split_significance()`.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Full distance matrix
    group : np.ndarray
        Binary group assignment (True/False) for sequences in current node
    indices : np.ndarray
        Original indices of sequences in current node (for accessing full distance matrix)
    weights : np.ndarray
        Weights for all sequences (full length, matching distance_matrix)
    R : int
        Number of permutations
    weight_permutation : str
        Method for handling weights: "none", "replicate", "diss", "group"
    squared : bool
        Whether distances are squared
        
    Returns
    -------
    float
        P-value for the split (proportion of permuted statistics >= observed)
    """
    if R <= 1:
        return np.nan
    
    return test_tree_split_significance(
        distance_matrix=distance_matrix,
        group=group,
        indices=indices,
        weights=weights,
        R=R,
        weight_permutation=weight_permutation,
        squared=squared
    )


def _get_leaf_memberships(root: DissTreeNode, n_total: int) -> np.ndarray:
    """
    Get leaf node membership for each sequence.
    
    Corresponds to TraMineR function: disstreeleaf()
    """
    memberships = np.full(n_total, -1, dtype=np.int32)
    _assign_leaf_ids(root, memberships)
    return memberships


def _assign_leaf_ids(node: DissTreeNode, memberships: np.ndarray):
    """Recursively assign leaf node IDs."""
    if node.is_terminal:
        memberships[node.info['ind']] = node.id
    else:
        _assign_leaf_ids(node.kids[0], memberships)
        _assign_leaf_ids(node.kids[1], memberships)
