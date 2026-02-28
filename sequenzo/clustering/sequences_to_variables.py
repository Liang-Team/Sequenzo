"""
@Author  : Yuqi Liang 梁彧祺
@File    : sequences_to_variables.py
@Time    : 01/03/2026 00:07
@Desc    :

Sequences to variables (Helske et al. 2024).

Construct regression-ready variables from sequence dissimilarities:
- Representativeness: R_i^k = 1 - d(i,k)/d_max (K continuous)
- Hard classification: cluster membership -> K-1 dummies (reference omitted)
- Soft classification: FANNY membership probabilities -> K-1 continuous (reference omitted)
- Pseudoclass: multiple draws from membership -> dummies -> fit -> Rubin combine

Reference: Helske, S., Helske, J., & Chihaya, G. K. (2024). From Sequences to Variables.
Sociological Methodology, 54(1), 27-51.

Example (representativeness + hard dummies):
    from sequenzo import get_distance_matrix, KMedoids, representativeness_matrix
    from sequenzo.clustering import (
        medoid_indices_from_kmedoids_result,
        cluster_labels_from_kmedoids_result,
        hard_classification_variables,
    )

    diss = get_distance_matrix(seqdata, method="OM", sm="TRATE", indel="auto")
    kmed_result = KMedoids(diss, k=5, method="PAMonce", verbose=False)
    medoids = medoid_indices_from_kmedoids_result(kmed_result)
    R = representativeness_matrix(diss, medoids, d_max=None, as_dataframe=True, ids=seqdata.ids)
    cluster_labels = cluster_labels_from_kmedoids_result(kmed_result)  # 0-based cluster id
    dummies = hard_classification_variables(cluster_labels, k=5, reference=0, as_dataframe=True, ids=seqdata.ids)
    # Use R or dummies as covariates in OLS/logistic regression.
"""
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple

from .seqs2vars_utils import max_distance, cluster_labels_to_dummies


# -----------------------------------------------------------------------------
# Phase 1: Representativeness (Helske et al. 2024 formula)
# -----------------------------------------------------------------------------

def representativeness_matrix(
    diss: Union[np.ndarray, pd.DataFrame],
    medoid_indices: Union[List[int], np.ndarray],
    d_max: Optional[float] = None,
    ids: Optional[Union[List, np.ndarray, pd.Index]] = None,
    as_dataframe: bool = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Representativeness of each sequence to each representative (medoid).

    R_i^k = 1 - (distance of sequence i to representative k) / (maximum distance between two sequences).
    Helske et al. (2024): "This leads to K continuous variables (which do not sum to 1)."

    Parameters
    ----------
    diss : np.ndarray or pd.DataFrame
        n x n distance matrix (same as used for PAM).
    medoid_indices : array-like of int
        Length K. Row/column indices of the K medoids in diss (0-based).
    d_max : float, optional
        Maximum distance between two sequences. If None, computed as max_distance(diss).
    ids : array-like, optional
        Sequence IDs for DataFrame index when as_dataframe=True.
    as_dataframe : bool, default False
        If True, return a DataFrame with columns R_1, R_2, ... R_K (and index=ids if provided).

    Returns
    -------
    np.ndarray of shape (n, K) or pd.DataFrame
        Representativeness values in [0, 1]; 1 = same as medoid, 0 = farthest.
    """
    diss = np.asarray(diss, dtype=float)
    medoid_indices = np.asarray(medoid_indices, dtype=int).ravel()
    n = diss.shape[0]
    k = len(medoid_indices)
    if diss.shape[0] != n or diss.shape[1] != n:
        raise ValueError("diss must be a square matrix")
    if np.any(medoid_indices < 0) or np.any(medoid_indices >= n):
        raise ValueError("medoid_indices must be in [0, n-1]")

    if d_max is None:
        d_max = max_distance(diss)
    if d_max <= 0:
        # All same distance: set R to 1 for same medoid, 0 otherwise (or all 1)
        R = np.zeros((n, k))
        for j, med in enumerate(medoid_indices):
            R[med, j] = 1.0
        if as_dataframe:
            return _to_rep_dataframe(R, ids, k)
        return R

    # R_i^k = 1 - diss[i, med_k] / d_max
    R = np.zeros((n, k))
    for j, med in enumerate(medoid_indices):
        R[:, j] = 1.0 - diss[:, med] / d_max
    # Clamp to [0, 1] in case of numerical noise
    R = np.clip(R, 0.0, 1.0)

    if as_dataframe:
        return _to_rep_dataframe(R, ids, k)
    return R


def _to_rep_dataframe(R: np.ndarray, ids, k: int) -> pd.DataFrame:
    cols = [f"R_{j+1}" for j in range(k)]
    df = pd.DataFrame(R, columns=cols)
    if ids is not None:
        df.index = pd.Index(ids)
    return df


# -----------------------------------------------------------------------------
# Helpers: medoid indices and cluster labels from KMedoids result
# -----------------------------------------------------------------------------

def medoid_indices_from_kmedoids_result(memb_matrix: np.ndarray) -> np.ndarray:
    """
    Get sorted medoid indices from KMedoids/PAM return value.

    sequenzo KMedoids returns for each row the *medoid index* of its cluster (not cluster id).
    So unique values are the K medoid indices.

    Parameters
    ----------
    memb_matrix : np.ndarray of int
        Return value of KMedoids(...).runclusterloop(); length n, values are medoid indices.

    Returns
    -------
    np.ndarray of shape (K,)
        Sorted medoid indices (0-based).
    """
    memb_matrix = np.asarray(memb_matrix, dtype=int).ravel()
    medoids = np.unique(memb_matrix)
    return np.sort(medoids)


def cluster_labels_from_kmedoids_result(memb_matrix: np.ndarray) -> np.ndarray:
    """
    Convert KMedoids return (medoid index per row) to 0-based cluster labels 0 .. K-1.

    Useful for hard_classification_variables and for consistent ordering with medoid_indices.

    Parameters
    ----------
    memb_matrix : np.ndarray of int
        Return value of KMedoids(...); each entry is the medoid index of that row's cluster.

    Returns
    -------
    np.ndarray of shape (n,)
        Cluster labels 0, 1, ..., K-1 (order by medoid index).
    """
    medoids = medoid_indices_from_kmedoids_result(memb_matrix)
    # label[i] = j iff memb_matrix[i] == medoids[j]
    return np.searchsorted(medoids, np.asarray(memb_matrix, dtype=int).ravel())


# -----------------------------------------------------------------------------
# Phase 2: Hard classification variables (cluster membership -> dummies)
# -----------------------------------------------------------------------------

def hard_classification_variables(
    labels: Union[np.ndarray, List[int]],
    k: Optional[int] = None,
    reference: int = 0,
    ids: Optional[Union[List, np.ndarray, pd.Index]] = None,
    as_dataframe: bool = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Turn cluster membership into dummy variables for regression (Helske et al. 2024 Table 1).

    "One cluster is typically chosen as a reference, and the respective (dummy or probability)
    variable is omitted from the model."

    Parameters
    ----------
    labels : array-like of int
        Cluster assignment per observation (0-based or 1-based).
    k : int, optional
        Number of clusters. If None, inferred from unique(labels).
    reference : int, optional
        Reference category index (0 = first in sorted order); that column is omitted.
    ids : array-like, optional
        Row identifiers for DataFrame index when as_dataframe=True.
    as_dataframe : bool, default False
        If True, return DataFrame with columns C_1, C_2, ... (K-1 columns).

    Returns
    -------
    np.ndarray of shape (n, K-1) or pd.DataFrame
        Dummy variables (0/1).
    """
    dummies = cluster_labels_to_dummies(labels, k=k, reference=reference)
    if as_dataframe:
        uniq = np.unique(labels)
        k_val = k if k is not None else len(uniq)
        ref_idx = min(reference, k_val - 1)
        col_indices = [i for i in range(k_val) if i != ref_idx]
        cols = [f"C_{uniq[c]+1}" for c in col_indices]
        df = pd.DataFrame(dummies, columns=cols)
        if ids is not None:
            df.index = pd.Index(ids)
        return df
    return dummies


# -----------------------------------------------------------------------------
# Phase 3: Soft classification (FANNY-style membership)
# -----------------------------------------------------------------------------

def fanny_membership(
    diss: np.ndarray,
    k: int,
    m: float = 1.4,
    max_iter: int = 100,
    tol: float = 1e-6,
    weights: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuzzy (FANNY-style) membership from distance matrix.

    Helske et al. (2024): "we fixed the membership exponent to 1.4".
    Uses PAM to get medoids, then computes membership u_ik from distances to medoids:
    u_ik proportional to (1/d_ik)^(1/(m-1)), normalized so each row sums to 1.

    Parameters
    ----------
    diss : np.ndarray
        n x n distance matrix.
    k : int
        Number of clusters.
    m : float, default 1.4
        Fuzziness exponent (>1). Higher = fuzzier.
    max_iter : int
        Maximum iterations for medoid updates (optional refinement).
    tol : float
        Convergence tolerance.
    weights : np.ndarray, optional
        Observation weights (length n).
    random_state : int, optional
        For reproducible PAM initialisation.

    Returns
    -------
    U : np.ndarray of shape (n, K)
        Membership matrix; each row sums to 1.
    medoid_indices : np.ndarray of shape (K,)
        Medoid indices (0-based).
    """
    from .KMedoids import KMedoids
    n = diss.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=float)
    if random_state is not None:
        np.random.seed(random_state)
    # Get medoids via PAM
    labels_pam = KMedoids(diss, k=k, weights=weights, method="PAMonce", verbose=False)
    medoid_indices = medoid_indices_from_kmedoids_result(labels_pam)
    # Distances from each point to each medoid: (n, K)
    d = diss[:, medoid_indices]
    # Avoid zero distance (point is medoid)
    eps = np.finfo(float).eps * (1 + np.max(d))
    d = np.maximum(d, eps)
    # u_ik \propto (1/d_ik)^(1/(m-1)); then normalize rows
    inv_exp = 1.0 / (m - 1.0)
    u = np.power(1.0 / d, inv_exp)
    row_sum = u.sum(axis=1, keepdims=True)
    u = u / np.maximum(row_sum, 1e-15)
    return u, medoid_indices


def soft_classification_variables(
    U: np.ndarray,
    reference: int = 0,
    ids: Optional[Union[List, np.ndarray, pd.Index]] = None,
    as_dataframe: bool = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Use membership matrix as K-1 continuous predictors (reference omitted).

    Helske et al. (2024) Table 1: Soft classification = "Membership degree", Continuous.
    "One cluster is typically chosen as a reference ... omitted from the model."

    Parameters
    ----------
    U : np.ndarray of shape (n, K)
        Membership matrix (each row sums to 1).
    reference : int
        Index of reference category (0-based) to omit.
    ids : array-like, optional
        Row IDs for DataFrame.
    as_dataframe : bool, default False
        If True, return DataFrame with columns P_1, P_2, ... (K-1 columns).

    Returns
    -------
    np.ndarray of shape (n, K-1) or pd.DataFrame
    """
    U = np.asarray(U, dtype=float)
    n, k = U.shape
    ref_idx = min(reference, k - 1)
    cols_keep = [j for j in range(k) if j != ref_idx]
    X = U[:, cols_keep].copy()
    if as_dataframe:
        col_names = [f"P_{j+1}" for j in cols_keep]
        df = pd.DataFrame(X, columns=col_names)
        if ids is not None:
            df.index = pd.Index(ids)
        return df
    return X


# -----------------------------------------------------------------------------
# Phase 4: Pseudoclass (multiple imputation style)
# -----------------------------------------------------------------------------

def pseudoclass_regression(
    y: np.ndarray,
    U: np.ndarray,
    X_fixed: Optional[np.ndarray] = None,
    M: int = 20,
    reference: int = 0,
    random_state: Optional[int] = None,
) -> dict:
    """
    Pseudoclass regression: draw M categorical memberships from U, fit each, combine with Rubin's rules.

    Helske et al. (2024): "individuals are randomly assigned to clusters multiple times
    on the basis of their membership probabilities ... combine the results across the models
    similarly to the multiple imputation technique (Rubin 2004)."

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        Outcome variable.
    U : np.ndarray of shape (n, K)
        Membership matrix (rows sum to 1).
    X_fixed : np.ndarray of shape (n, p), optional
        Other covariates (include intercept column if desired). If None, only dummies are used.
    M : int, default 20
        Number of pseudoclass replications.
    reference : int
        Reference category for dummies (0-based).
    random_state : int, optional
        For reproducible draws.

    Returns
    -------
    dict with keys
        beta_combined : np.ndarray
            Combined coefficient estimates (p + K-1 from dummies).
        se_combined : np.ndarray
            Combined standard errors.
        beta_list : list of np.ndarray
            Per-replication coefficient estimates (for diagnostics).
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("pseudoclass_regression requires statsmodels for OLS")

    rng = np.random.default_rng(random_state)
    n, K = U.shape
    X_fixed = np.asarray(X_fixed, dtype=float) if X_fixed is not None else np.empty((n, 0))
    if X_fixed.size > 0 and X_fixed.shape[0] != n:
        raise ValueError("X_fixed must have same number of rows as U")
    p_fixed = X_fixed.shape[1] if X_fixed.size > 0 else 0
    beta_list = []
    var_list = []  # variance (s.e.^2) per replication

    for _ in range(M):
        # Draw cluster for each individual from U
        labels_m = np.array([
            rng.choice(K, p=U[i, :]) for i in range(n)
        ])
        dummies_m = cluster_labels_to_dummies(labels_m, k=K, reference=reference)
        if p_fixed > 0:
            X_m = np.hstack([X_fixed, dummies_m])
        else:
            X_m = dummies_m
        if np.linalg.matrix_rank(X_m) < X_m.shape[1]:
            continue
        model = sm.OLS(y, X_m).fit()
        beta_list.append(model.params)
        var_list.append(model.bse ** 2)

    if len(beta_list) == 0:
        raise RuntimeError("All M replications had rank-deficient design matrix")
    beta_stack = np.array(beta_list)
    var_stack = np.array(var_list)
    # Ensure same parameter length (in case some runs dropped)
    n_par = beta_stack.shape[1]
    # Within-imputation variance (average)
    W = np.mean(var_stack, axis=0)
    # Between-imputation variance
    B = np.var(beta_stack, axis=0, ddof=1)
    # Rubin: T = W + (1 + 1/M)*B
    T = W + (1.0 + 1.0 / M) * B
    se_combined = np.sqrt(T)
    beta_combined = np.mean(beta_stack, axis=0)
    return {
        "beta_combined": beta_combined,
        "se_combined": se_combined,
        "beta_list": beta_list,
    }
