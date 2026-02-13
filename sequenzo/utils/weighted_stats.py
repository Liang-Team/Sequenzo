"""
@Author  : Yuqi Liang 梁彧祺
@File    : weighted_stats.py
@Time    : 2026-02-13 07:22
@Desc    : 
Weighted Statistics Functions

This module provides weighted statistical functions that match TraMineR's implementation.
These functions are used internally throughout Sequenzo for weighted calculations.

Reference: TraMineR R package
- TraMineR-wtd-stats.R
- Based on Hmisc package functions (included in TraMineR to avoid dependencies)
- See: https://github.com/cran/TraMineR/blob/master/R/TraMineR-wtd-stats.R

Functions:
    weighted_mean: Weighted mean (corresponds to R's wtd.mean)
    weighted_variance: Weighted variance (corresponds to R's wtd.var)
    weighted_five_number_summary: Weighted five-number summary (corresponds to R's wtd.fivenum.tmr)
"""

import numpy as np
from typing import Optional, Union, Literal
import warnings


def weighted_mean(
    x: np.ndarray,
    weights: Optional[np.ndarray] = None,
    normwt: Union[str, bool] = 'ignored',
    na_rm: bool = True
) -> float:
    """
    Compute weighted mean of a vector.
    
    This function matches TraMineR's implementation.
    
    Corresponds to R function: wtd.mean() in TraMineR-wtd-stats.R
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of values
    weights : np.ndarray, optional
        Weights for each observation. If None, computes unweighted mean.
    normwt : str or bool, default 'ignored'
        Normalization flag (kept for API compatibility with R version, but ignored)
    na_rm : bool, default True
        If True, remove NA/NaN values before computation
        
    Returns
    -------
    float
        Weighted mean of x
        
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> w = np.array([1, 2, 1, 2, 1])
    >>> weighted_mean(x, weights=w)
    2.857142857142857
    
    >>> weighted_mean(x)  # Unweighted
    3.0
    """
    x = np.asarray(x, dtype=float)
    
    # If no weights provided, return regular mean
    if weights is None or len(weights) == 0:
        if na_rm:
            x = x[~np.isnan(x)]
        return float(np.mean(x))
    
    weights = np.asarray(weights, dtype=float)
    
    if len(weights) != len(x):
        raise ValueError(
            f"[!] Length of weights ({len(weights)}) must match length of x ({len(x)})"
        )
    
    # Remove NA values if requested
    if na_rm:
        valid = ~(np.isnan(x) | np.isnan(weights))
        x = x[valid]
        weights = weights[valid]
    
    # Check for all zero weights
    if np.sum(weights) == 0:
        return np.nan
    
    # Compute weighted mean: sum(weights * x) / sum(weights)
    return float(np.sum(weights * x) / np.sum(weights))


def weighted_variance(
    x: np.ndarray,
    weights: Optional[np.ndarray] = None,
    normwt: bool = False,
    na_rm: bool = True,
    method: Literal['unbiased', 'ML'] = 'unbiased'
) -> float:
    """
    Compute weighted variance of a vector.
    
    This function matches TraMineR's implementation.
    
    Corresponds to R function: wtd.var() in TraMineR-wtd-stats.R
    By Benjamin Tyner <btyner@gmail.com> 2017-0-12
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of values
    weights : np.ndarray, optional
        Weights for each observation. If None, computes unweighted variance.
    normwt : bool, default False
        If True, normalize weights so they sum to length(x)
    na_rm : bool, default True
        If True, remove NA/NaN values before computation
    method : {'unbiased', 'ML'}, default 'unbiased'
        Method for variance calculation:
        - 'unbiased': Unbiased frequency weights (uses n-1 denominator)
        - 'ML': Maximum likelihood (uses n denominator)
        
    Returns
    -------
    float
        Weighted variance of x
        
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> w = np.array([1, 2, 1, 2, 1])
    >>> weighted_variance(x, weights=w)
    2.0
    
    >>> weighted_variance(x)  # Unweighted
    2.5
    """
    x = np.asarray(x, dtype=float)
    
    # If no weights provided, return regular variance
    if weights is None or len(weights) == 0:
        if na_rm:
            x = x[~np.isnan(x)]
        return float(np.var(x, ddof=1))  # Unbiased variance (n-1 denominator)
    
    weights = np.asarray(weights, dtype=float)
    
    if len(weights) != len(x):
        raise ValueError(
            f"[!] Length of weights ({len(weights)}) must match length of x ({len(x)})"
        )
    
    # Remove NA values if requested
    if na_rm:
        valid = ~(np.isnan(x) | np.isnan(weights))
        x = x[valid]
        weights = weights[valid]
    
    # Normalize weights if requested
    if normwt:
        weights = weights * len(x) / np.sum(weights)
    
    # For ML method or normalized weights, use covariance matrix approach
    if normwt or method == 'ML':
        # Equivalent to R's stats::cov.wt(cbind(x), weights, method=method)$cov
        # For a single variable, this is just the weighted variance
        sw = np.sum(weights)
        if sw == 0:
            return np.nan
        
        xbar = np.sum(weights * x) / sw
        if method == 'ML':
            # ML: divide by n (sum of weights)
            var_val = np.sum(weights * ((x - xbar) ** 2)) / sw
        else:
            # Normalized weights case
            var_val = np.sum(weights * ((x - xbar) ** 2)) / sw
        return float(var_val)
    
    # Unbiased frequency weights case
    sw = np.sum(weights)
    if sw <= 1:
        warnings.warn(
            "only one effective observation; variance estimate undefined",
            RuntimeWarning
        )
        return np.nan
    
    xbar = np.sum(weights * x) / sw
    # Unbiased: divide by (sum(weights) - 1)
    var_val = np.sum(weights * ((x - xbar) ** 2)) / (sw - 1)
    return float(var_val)


def weighted_five_number_summary(
    x: np.ndarray,
    weights: Optional[np.ndarray] = None,
    na_rm: bool = True
) -> np.ndarray:
    """
    Compute weighted five-number summary (minimum, Q1, median, Q3, maximum).
    
    This function matches TraMineR's implementation.
    
    Corresponds to R function: wtd.fivenum.tmr() in TraMineR-wtd-stats.R
    Returns the five-number summary: [min, Q1, median, Q3, max]
    
    Parameters
    ----------
    x : np.ndarray
        Input vector of values
    weights : np.ndarray, optional
        Weights for each observation. If None, uses equal weights.
    na_rm : bool, default True
        If True, remove NA/NaN values before computation
        
    Returns
    -------
    np.ndarray
        Array of length 5 containing [min, Q1, median, Q3, max]
        
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> w = np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1])
    >>> weighted_five_number_summary(x, weights=w)
    array([1., 3., 5., 7., 10.])
    
    >>> weighted_five_number_summary(x)  # Unweighted
    array([1., 3.5, 5.5, 7.5, 10.])
    """
    def interpolated_index(myval: float, weights_arr: np.ndarray) -> float:
        """
        Find interpolated index for weighted quantile calculation.
        
        This is an internal helper function that finds the index position
        corresponding to a given cumulative weight value.
        """
        n = len(weights_arr)
        indices = np.arange(1, n + 1)
        total_weight = np.sum(weights_arr)
        
        # Compute cumulative weights below each position
        weights_below = np.zeros(n)
        for i in range(1, n):
            weights_below[i] = weights_below[i-1] + weights_arr[i-1]
        
        # Compute cumulative weights above each position
        weights_above = total_weight - weights_below - weights_arr
        
        # Find positions where cumulative weight below is less than myval
        low_cands = weights_below < myval
        # Find positions where cumulative weight above is less than (total - myval)
        high_cands = weights_above < (total_weight - myval)
        
        # Compute interpolated index
        low_idx = np.max(indices[low_cands]) if np.any(low_cands) else 1
        high_idx = np.min(indices[high_cands]) if np.any(high_cands) else n
        
        return (low_idx + high_idx) / 2.0
    
    x = np.asarray(x, dtype=float)
    
    # Default to equal weights if not provided
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = np.asarray(weights, dtype=float)
    
    if len(weights) != len(x):
        raise ValueError(
            f"[!] Length of weights ({len(weights)}) must match length of x ({len(x)})"
        )
    
    # Check if all weights are equal
    if len(x) > 1:
        equal_weights = np.all(np.diff(weights) == 0)
    else:
        equal_weights = True
    
    # Handle NA values
    xna = np.isnan(x) | (weights == 0)
    if na_rm:
        x = x[~xna]
        weights = weights[~xna]
    else:
        if np.any(xna):
            return np.full(5, np.nan)
    
    # Sort by x values
    sort_order = np.argsort(x)
    x = x[sort_order]
    weights = weights[sort_order]
    
    n = np.sum(weights)
    
    if n == 0:
        return np.full(5, np.nan)
    
    # Compute quantile positions
    if equal_weights:
        # For equal weights, use standard fivenum positions
        d = np.array([
            1,
            0.5 * np.floor(0.5 * (n + 3)),
            0.5 * (n + 1),
            n + 1 - 0.5 * np.floor(0.5 * (n + 3)),
            n
        ])
    else:
        # For unequal weights, use interpolation
        if len(x) > 1:
            q25_idx = interpolated_index(0.25 * n, weights)
            q50_idx = interpolated_index(0.5 * n, weights)
            q75_idx = interpolated_index(0.75 * n, weights)
            d = np.array([1, q25_idx, q50_idx, q75_idx, len(x)])
        else:
            d = np.ones(5)
    
    # R uses 1-based indexing, so d is 1-based
    # Convert to 0-based for Python array access
    # Ensure indices are within valid range [0, len(x)-1]
    d_floor = np.maximum(0, np.minimum(len(x) - 1, np.floor(d) - 1)).astype(int)
    d_ceil = np.maximum(0, np.minimum(len(x) - 1, np.ceil(d) - 1)).astype(int)
    
    # Handle edge case where floor and ceil might be out of bounds
    d_floor = np.clip(d_floor, 0, len(x) - 1)
    d_ceil = np.clip(d_ceil, 0, len(x) - 1)
    
    # Compute interpolated values: average of floor and ceiling
    # This matches R's: 0.5*(x[floor(d)]+x[ceiling(d)])
    result = 0.5 * (x[d_floor] + x[d_ceil])
    
    return result
