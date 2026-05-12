"""Overall sequence discrepancy from a distance matrix (TraMineR: dissvar)."""

import numpy as np
import pandas as pd
from typing import Optional

from ..internal.weighted_inertia import compute_pseudo_variance_from_matrix

def overall_discrepancy(
    distance_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
    squared: bool = False
) -> float:
    """
    Compute the overall discrepancy of a distance matrix.
    
    This function summarizes between-individual variability of trajectories from
    pairwise dissimilarities. It plays the role of a generalized variance or
    inertia, but it is called discrepancy because the input distances need not be
    Euclidean.
    
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
        If True, use exponent v=2 on dissimilarities before computing discrepancy.
        Default False uses nonsquared dissimilarities (v=1), as recommended by
        Studer et al. (2011) for non-Euclidean sequence distances.
        
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
    >>> from sequenzo.discrepancy_analysis import overall_discrepancy
    >>> 
    >>> # Create a simple distance matrix
    >>> dist_matrix = np.array([
    ...     [0.0, 1.0, 2.0],
    ...     [1.0, 0.0, 1.5],
    ...     [2.0, 1.5, 0.0]
    ... ])
    >>> 
    >>> # Compute pseudo-variance
    >>> variance = overall_discrepancy(dist_matrix)
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
    
    # Unweighted case: TraMineR dissvar() on a full matrix
    if weights is None:
        return compute_pseudo_variance_from_matrix(
            distance_matrix,
            weights=None,
            squared=False,
        )
    
    # Weighted case: TraMineR dissvar() with C_tmrWeightedInertiaDist(var=TRUE)
    return compute_pseudo_variance_from_matrix(
        distance_matrix,
        weights=weights,
        squared=False,
    )
