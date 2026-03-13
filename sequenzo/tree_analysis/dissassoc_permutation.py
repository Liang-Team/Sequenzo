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
import pandas as pd
from typing import Callable, Optional, Tuple, Sequence, Dict, Any

from .permutation import permutation_test


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


def gower_matrix(
    distance_matrix: np.ndarray,
    squared: bool = True,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute a centered Gower-like matrix from a distance matrix.

    This reproduces the behaviour of TraMineR's gower_matrix() used in
    dissmfacw():

    - Optionally square the distances.
    - Apply double-centering with respect to case weights.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix of shape (n, n).
    squared : bool, default True
        If True, square the distances before centering.
    weights : np.ndarray, optional
        Optional case weights of length n. If None, equal weights are used.

    Returns
    -------
    np.ndarray
        Centered Gower matrix of shape (n, n).
    """
    diss = np.asarray(distance_matrix, dtype=float)
    n = diss.shape[0]

    if diss.shape[0] != diss.shape[1]:
        raise ValueError("[gower_matrix] distance_matrix must be square.")

    if squared:
        diss = diss ** 2

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or len(weights) != n:
            raise ValueError("[gower_matrix] weights must be a 1D array of length n.")

    # Normalised weights
    s = weights / weights.sum()
    one = np.ones(n, dtype=float)

    # Double-centering: (I - 1 s^T) * (-diss/2) * (I - s 1^T)
    left = np.eye(n) - np.outer(one, s)
    right = np.eye(n) - np.outer(s, one)
    g_matrix = left @ (diss / -2.0) @ right
    return g_matrix


def _weighted_hat_matrix_qr(
    design: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Compute the weighted hat matrix using QR decomposition.

    This follows the logic of TraMineR's hatw_matrix_qr():

    - Multiply each row of the design matrix by sqrt(weights).
    - Compute QR decomposition: X_w = Q R.
    - Hat matrix H = Q Q^T.
    - Return the "weighted" hat matrix W^{1/2} H W^{1/2}, so that it can be
      multiplied element-wise with the Gower matrix.

    Parameters
    ----------
    design : np.ndarray
        Design matrix X of shape (n, p) (no intercept is added automatically).
    weights : np.ndarray
        Case weights of length n (must be non-negative).

    Returns
    -------
    np.ndarray
        Weighted hat matrix of shape (n, n).
    """
    X = np.asarray(design, dtype=float)
    w = np.asarray(weights, dtype=float)

    if X.ndim != 2:
        raise ValueError("[_weighted_hat_matrix_qr] design must be 2D.")
    n, _ = X.shape

    if w.ndim != 1 or len(w) != n:
        raise ValueError("[_weighted_hat_matrix_qr] weights must be a 1D array of length n.")

    if np.any(w < 0):
        raise ValueError("[_weighted_hat_matrix_qr] weights must be non-negative.")

    # If all weights are zero, fall back to equal weights.
    if np.allclose(w, 0.0):
        w = np.ones_like(w)

    w_sqrt = np.sqrt(w)
    Xw = X * w_sqrt[:, None]

    # Reduced QR decomposition: Q has orthonormal columns
    Q, _ = np.linalg.qr(Xw, mode="reduced")
    hat = Q @ Q.T  # n x n

    # Re-apply weights so that H can be combined element-wise with Gower matrix
    w_sqrt_mat = np.outer(w_sqrt, w_sqrt)
    return w_sqrt_mat * hat


def distance_multifactor_anova(
    distance_matrix: np.ndarray,
    design_matrix: np.ndarray,
    term_ids: Sequence[int],
    term_labels: Optional[Sequence[str]] = None,
    weights: Optional[np.ndarray] = None,
    gower: bool = False,
    squared: bool = False,
    R: int = 0,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Distance-based multi-factor ANOVA / regression (TraMineR's dissmfacw()).

    This function decomposes the variation in a distance matrix across several
    covariates (factors), using a Gower-centered distance matrix and weighted
    hat matrices from QR decomposition. It returns Pseudo-F and Pseudo-R²
    statistics for each term and for the complete model, and optional
    permutation-based p-values.

    Parameters
    ----------
    distance_matrix : np.ndarray
        N x N dissimilarity matrix.
    design_matrix : np.ndarray
        N x P design matrix (no intercept is added automatically). Columns can
        encode dummy variables for categorical covariates or continuous
        predictors. Missing values should be handled before calling.
    term_ids : sequence of int
        Length-P sequence that maps each column of design_matrix to a term
        index (0, 1, ..., n_terms-1). Columns with the same term index are
        treated as belonging to the same factor.
    term_labels : sequence of str, optional
        Optional human-readable labels for the terms, length n_terms.
        If None, labels will be "Term1", "Term2", ...
        An additional "Total" row is added for the complete model.
    weights : np.ndarray, optional
        Optional case weights of length N. If None, equal weights are used.
    gower : bool, default False
        If False, compute the Gower matrix internally from the distance matrix.
        If True, distance_matrix is assumed to already be a centered Gower-like
        matrix and is used as is.
    squared : bool, default False
        Whether to square distances before computing the Gower matrix (only
        relevant when gower=False).
    R : int, default 0
        Number of permutations for significance testing. If R <= 1, p-values
        are returned as NaN.
    random_state : int, optional
        Random seed for reproducibility of permutations.

    Returns
    -------
    dict
        Dictionary with:

        - 'summary': pandas.DataFrame with columns
          ['Variable', 'PseudoF', 'PseudoR2', 'p_value'].
        - 'g_matrix': the centered distance (Gower) matrix used.
        - 'weights': the weights vector used (after defaults).
    """
    D = np.asarray(distance_matrix, dtype=float)
    X = np.asarray(design_matrix, dtype=float)
    term_ids = np.asarray(term_ids, dtype=int)

    if D.shape[0] != D.shape[1]:
        raise ValueError("[distance_multifactor_anova] distance_matrix must be square.")
    n = D.shape[0]

    if X.ndim != 2:
        raise ValueError("[distance_multifactor_anova] design_matrix must be 2D.")
    if X.shape[0] != n:
        raise ValueError("[distance_multifactor_anova] design_matrix and distance_matrix must have the same number of rows.")
    if X.shape[1] != len(term_ids):
        raise ValueError("[distance_multifactor_anova] term_ids length must equal the number of columns in design_matrix.")

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or len(weights) != n:
            raise ValueError("[distance_multifactor_anova] weights must be a 1D array of length n.")

    # Prepare term information
    unique_terms = np.unique(term_ids)
    n_terms = len(unique_terms)

    if term_labels is None:
        term_labels = [f"Term{i+1}" for i in range(n_terms)]
    else:
        if len(term_labels) != n_terms:
            raise ValueError("[distance_multifactor_anova] term_labels length must match the number of unique term_ids.")

    # Map term_ids (possibly arbitrary ints) to 0..n_terms-1
    term_index_map = {tid: idx for idx, tid in enumerate(unique_terms)}
    var_list = np.array([term_index_map[t] for t in term_ids], dtype=int)
    var_list_index = np.arange(X.shape[1])

    # Centered distance (Gower) matrix
    if gower:
        G = D
    else:
        G = gower_matrix(D, squared=squared, weights=weights)

    total_weight = float(weights.sum())

    # Total sum of squares in Gower space (weighted diagonal)
    SCtot = float((weights * np.diag(G)).sum())

    # Pre-compute backward sums of squares for each term (leaving-one-out)
    p_list = np.zeros(n_terms, dtype=int)
    SSbv = np.zeros(n_terms, dtype=float)

    for term_idx in range(n_terms):
        # Keep all columns except those belonging to the current term
        cols_keep = var_list != term_idx
        pred = X[:, cols_keep]

        p_list[term_idx] = int((var_list == term_idx).sum())

        if pred.shape[1] == 0:
            # If no columns remain, SCexp is zero
            SSbv[term_idx] = 0.0
        else:
            hwm = _weighted_hat_matrix_qr(pred, weights=weights)
            # G is symmetric; element-wise product is enough
            SSbv[term_idx] = float((hwm * G).sum())

    # Helper to compute F and R2 for a given predictor matrix (complete + each term)
    def _compute_F_R2_for_predictor(pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Pseudo-F and Pseudo-R² for each term and the complete model.
        """
        m = pred.shape[1]

        # Complete model
        hwm_full = _weighted_hat_matrix_qr(pred, weights=weights)
        SCexpC = float((hwm_full * G).sum())
        SCresC = SCtot - SCexpC

        # Pseudo-F and R² for complete model (index n_terms)
        R2_list = np.zeros(n_terms + 1, dtype=float)
        F_list = np.zeros(n_terms + 1, dtype=float)

        R2_list[-1] = SCexpC / SCtot if SCtot > 0 else 0.0
        if m > 1 and SCresC > 0:
            F_list[-1] = (SCexpC / (m - 1.0)) / (SCresC / (total_weight - m))
        else:
            F_list[-1] = 0.0

        # If there is only one term, its stats equal the complete model
        if n_terms == 1:
            R2_list[0] = R2_list[-1]
            F_list[0] = F_list[-1]
            return F_list, R2_list

        # For each term, replace its columns with those of pred and recompute
        for term_idx in range(n_terms):
            # Copy full design
            pred_term = pred.copy()

            # Columns belonging to this term in the original X
            cols_term = (var_list == term_idx)
            if not np.any(cols_term):
                F_list[term_idx] = 0.0
                R2_list[term_idx] = 0.0
                continue

            # Replace those columns with the corresponding columns from pred
            pred_term[:, cols_term] = pred[:, cols_term]

            hwm_term = _weighted_hat_matrix_qr(pred_term, weights=weights)
            SCexp = float((hwm_term * G).sum())
            SCres = SCtot - SCexp

            # Backward SS for this term
            delta_SCexp = SCexp - SSbv[term_idx]

            if p_list[term_idx] > 0 and SCres > 0:
                F_list[term_idx] = (delta_SCexp / p_list[term_idx]) / (SCres / (total_weight - pred_term.shape[1]))
            else:
                F_list[term_idx] = 0.0

            R2_list[term_idx] = delta_SCexp / SCtot if SCtot > 0 else 0.0

        return F_list, R2_list

    # Observed statistics for original design matrix
    F_obs, R2_obs = _compute_F_R2_for_predictor(X)

    # Permutation-based p-values (if requested)
    p_values = np.full_like(F_obs, np.nan, dtype=float)
    if R is not None and R > 1:
        if random_state is not None:
            np.random.seed(random_state)

        def _statistic(data: np.ndarray, perm_indices: np.ndarray) -> np.ndarray:
            """
            Permutation statistic callback for permutation_test().

            - data is the original design matrix (N x P).
            - perm_indices is a permutation of row indices.
            """
            pred_perm = data[perm_indices, :]
            F_perm, R2_perm = _compute_F_R2_for_predictor(pred_perm)
            # Concatenate to align with observed statistics
            return np.concatenate([F_perm, R2_perm])

        perm_result = permutation_test(
            data=X,
            R=R,
            statistic=_statistic
        )

        t0 = perm_result["t0"]
        t = perm_result["t"]

        if t is not None:
            # First n_terms+1 entries of t0 are F, next n_terms+1 are R²
            F0 = t0[: n_terms + 1]
            F_perm = t[:, : n_terms + 1]
            # Right-tailed p-values for F
            for j in range(n_terms + 1):
                p_values[j] = float(np.mean(F_perm[:, j] >= F0[j]))

    # Build summary table
    var_names = list(term_labels) + ["Total"]
    summary_df = pd.DataFrame(
        {
            "Variable": var_names,
            "PseudoF": F_obs,
            "PseudoR2": R2_obs,
            "p_value": p_values,
        }
    )

    return {
        "summary": summary_df,
        "g_matrix": G,
        "weights": weights,
    }


def compute_distance_indicators(
    distance_matrix: np.ndarray,
    group: np.ndarray,
    weights: Optional[np.ndarray] = None,
    gower: bool = False,
    squared: bool = False
) -> pd.DataFrame:
    """
    Compute individual-level distance indicators: marginality and gain.

    This function mirrors TraMineRextras' dissindic() for the simple case of a
    single grouping variable. It returns, for each sequence:

    - marginality: residual distance under the complete model (including group).
    - gain: reduction in residual distance when moving from a null model
      (no group effect) to the complete model.

    Parameters
    ----------
    distance_matrix : np.ndarray
        N x N dissimilarity matrix.
    group : np.ndarray
        Group labels of length N. Can be numeric or string; used only for
        labelling the output.
    weights : np.ndarray, optional
        Optional case weights of length N. If None, equal weights are used.
    gower : bool, default False
        If False, compute the Gower matrix internally from the distance matrix.
        If True, distance_matrix is assumed to already be a centered Gower-like
        matrix and is used as is.
    squared : bool, default False
        Whether to square distances before computing the Gower matrix (only
        relevant when gower=False).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

        - 'group': the original group labels.
        - 'marginality': residual indicator under the complete model.
        - 'gain': improvement from null to complete model.
    """
    D = np.asarray(distance_matrix, dtype=float)
    g = np.asarray(group)
    n = D.shape[0]

    if D.shape[0] != D.shape[1]:
        raise ValueError("[compute_distance_indicators] distance_matrix must be square.")
    if g.shape[0] != n:
        raise ValueError("[compute_distance_indicators] group length must match number of rows in distance_matrix.")

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or len(weights) != n:
            raise ValueError("[compute_distance_indicators] weights must be a 1D array of length n.")

    # Build simple design matrix for the group (one-hot encoding, no intercept)
    group_series = pd.Series(g)
    X = pd.get_dummies(group_series, drop_first=False).to_numpy(dtype=float)

    # Centered distance matrix
    if gower:
        G = D
    else:
        G = gower_matrix(D, squared=squared, weights=weights)

    # Weight matrix W
    W_mat = np.zeros_like(G)
    np.fill_diagonal(W_mat, weights)

    # Residuals matrix for indicators: columns NullModels, Complete
    residuals = np.zeros((n, 2), dtype=float)

    # Null model: no covariates. Residual indicator is simply diag(G).
    residuals[:, 0] = np.diag(G)

    # Complete model: include group design matrix
    hwm = _weighted_hat_matrix_qr(X, weights=weights)
    # Residual operator: W - H_w
    resid_mat = (W_mat - hwm) @ G
    # Normalise by weights to keep scale similar to TraMineR's implementation
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals[:, 1] = np.diag(resid_mat) / weights

    # Indicators
    marginality = residuals[:, 1]
    gain = residuals[:, 0] - residuals[:, 1]

    return pd.DataFrame(
        {
            "group": g,
            "marginality": marginality,
            "gain": gain,
        }
    )

