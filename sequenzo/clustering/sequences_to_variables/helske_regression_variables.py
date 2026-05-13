"""
@Author  : Yuqi Liang 梁彧祺
@File    : helske_regression_variables.py
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
    from sequenzo.clustering.sequences_to_variables import (
        medoid_indices_from_kmedoids_result,
        cluster_labels_from_kmedoids_result,
        hard_classification_variables,
    )

    diss = get_distance_matrix(seqdata, method="OM", sm="TRATE", indel="auto")
    kmed_result = KMedoids(diss, k=5, method="PAMonce", verbose=False)
    medoids = medoid_indices_from_kmedoids_result(kmed_result)
    R = representativeness_matrix(diss, medoids, d_max=None, as_dataframe=True, ids=seqdata.ids)
    cluster_labels = cluster_labels_from_kmedoids_result(kmed_result)
    dummies = hard_classification_variables(cluster_labels, k=5, reference=0, as_dataframe=True, ids=seqdata.ids)
"""
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple

from .helpers import (
    cluster_labels_from_kmedoids_result,
    cluster_labels_to_dummies,
    dummy_column_names,
    max_distance,
    medoid_indices_from_kmedoids_result,
    validate_diss_matrix,
    validate_membership_matrix,
)
from .fanny import (
    FannyResult,
    fanny,
    fanny_membership,
    medoid_membership_approximation,
    representative_indices_from_membership,
)

__all__ = [
    "representativeness_matrix",
    "medoid_indices_from_kmedoids_result",
    "cluster_labels_from_kmedoids_result",
    "hard_classification_variables",
    "fanny",
    "FannyResult",
    "fanny_membership",
    "medoid_membership_approximation",
    "representative_indices_from_membership",
    "soft_classification_variables",
    "pseudoclass_regression",
]


# -----------------------------------------------------------------------------
# Phase 1: Representativeness (Helske et al. 2024 formula)
# -----------------------------------------------------------------------------

def representativeness_matrix(
    diss: Union[np.ndarray, pd.DataFrame],
    medoid_indices: Union[List[int], np.ndarray],
    d_max: Optional[float] = None,
    ids: Optional[Union[List, np.ndarray, pd.Index]] = None,
    as_dataframe: bool = False,
    representative_names: Optional[List[str]] = None,
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
        Sequence IDs for DataFrame index when as_dataframe=True. When ``diss`` is a
        DataFrame and ``ids`` is None, the DataFrame index is used.
    as_dataframe : bool, default False
        If True, return a DataFrame with columns R_1, R_2, ... R_K (or ``representative_names``).
    representative_names : list of str, optional
        Column names for representativeness variables (length K).

    Returns
    -------
    np.ndarray of shape (n, K) or pd.DataFrame
        Representativeness values in [0, 1]; 1 = same as medoid, 0 = farthest.
    """
    if isinstance(diss, pd.DataFrame) and ids is None:
        ids = diss.index
    diss = np.asarray(diss, dtype=float)
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square matrix")
    validate_diss_matrix(diss)
    medoid_indices = np.asarray(medoid_indices, dtype=int).ravel()
    n = diss.shape[0]
    k = len(medoid_indices)
    if np.any(medoid_indices < 0) or np.any(medoid_indices >= n):
        raise ValueError("medoid_indices must be in [0, n-1]")
    if representative_names is not None and len(representative_names) != k:
        raise ValueError("representative_names must have length K")

    if d_max is None:
        d_max = max_distance(diss)
    if d_max <= 0:
        R = np.ones((n, k))
        if as_dataframe:
            return _to_rep_dataframe(R, ids, k, representative_names)
        return R

    R = np.zeros((n, k))
    for j, med in enumerate(medoid_indices):
        R[:, j] = 1.0 - diss[:, med] / d_max
    R = np.clip(R, 0.0, 1.0)

    if as_dataframe:
        return _to_rep_dataframe(R, ids, k, representative_names)
    return R


def _to_rep_dataframe(
    R: np.ndarray,
    ids,
    k: int,
    representative_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    if representative_names is not None:
        cols = list(representative_names)
    else:
        cols = [f"R_{j + 1}" for j in range(k)]
    df = pd.DataFrame(R, columns=cols)
    if ids is not None:
        df.index = pd.Index(ids)
    return df


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
        If True, return DataFrame with columns named from sorted unique labels.

    Returns
    -------
    np.ndarray of shape (n, K-1) or pd.DataFrame
        Dummy variables (0/1).
    """
    dummies = cluster_labels_to_dummies(labels, k=k, reference=reference)
    if as_dataframe:
        cols = dummy_column_names(labels, k=k, reference=reference, prefix="C")
        df = pd.DataFrame(dummies, columns=cols)
        if ids is not None:
            df.index = pd.Index(ids)
        return df
    return dummies


# -----------------------------------------------------------------------------
# Phase 3: Soft classification (FANNY membership)
# -----------------------------------------------------------------------------

def soft_classification_variables(
    U: np.ndarray,
    reference: int = 0,
    ids: Optional[Union[List, np.ndarray, pd.Index]] = None,
    as_dataframe: bool = False,
    cluster_names: Optional[List[str]] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Use membership matrix as K-1 continuous predictors (reference omitted).

    Helske et al. (2024) Table 1: Soft classification = "Membership degree", Continuous.

    Parameters
    ----------
    U : np.ndarray of shape (n, K)
        Membership matrix (each row sums to 1).
    reference : int
        Index of reference category (0-based) to omit.
    ids : array-like, optional
        Row IDs for DataFrame.
    as_dataframe : bool, default False
        If True, return DataFrame with columns P_1, P_2, ... (or ``cluster_names``).
    cluster_names : list of str, optional
        Names for clusters (length K); reference name is omitted.

    Returns
    -------
    np.ndarray of shape (n, K-1) or pd.DataFrame
    """
    U = validate_membership_matrix(U)
    n, k = U.shape
    if reference < 0 or reference >= k:
        raise ValueError(f"reference must be between 0 and {k - 1}; got {reference}")
    cols_keep = [j for j in range(k) if j != reference]
    X = U[:, cols_keep].copy()
    if as_dataframe:
        if cluster_names is not None:
            if len(cluster_names) != k:
                raise ValueError("cluster_names must have length K")
            col_names = [f"P_{cluster_names[j]}" for j in cols_keep]
        else:
            col_names = [f"P_{j + 1}" for j in cols_keep]
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
    model_type: str = "ols",
    add_intercept: bool = True,
) -> dict:
    """
    Pseudoclass regression: draw M categorical memberships from U, fit each, combine with Rubin's rules.

    Helske et al. (2024): individuals are randomly assigned to clusters on the basis of
    membership probabilities; results are combined with Rubin (2004) multiple-imputation rules.

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        Outcome variable (continuous for OLS, binary for logit).
    U : np.ndarray of shape (n, K)
        Membership matrix (rows sum to 1).
    X_fixed : np.ndarray of shape (n, p), optional
        Other covariates (intercept added separately when ``add_intercept=True``).
    M : int, default 20
        Number of pseudoclass replications.
    reference : int
        Reference category for dummies (0-based).
    random_state : int, optional
        For reproducible draws.
    model_type : {"ols", "logit"}, default "ols"
        Regression model type.
    add_intercept : bool, default True
        If True, prepend an intercept column to the design matrix.

    Returns
    -------
    dict with keys
        beta_combined, se_combined, cov_combined, beta_list, m_eff, failed
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError("pseudoclass_regression requires statsmodels") from exc

    model_type = model_type.lower()
    if model_type not in {"ols", "logit"}:
        raise ValueError("model_type must be 'ols' or 'logit'")
    if M < 1:
        raise ValueError("M must be at least 1")

    U = validate_membership_matrix(U)
    y = np.asarray(y, dtype=float).ravel()
    n, K = U.shape
    if y.shape[0] != n:
        raise ValueError("y must have the same number of rows as U")

    rng = np.random.default_rng(random_state)
    if reference < 0 or reference >= K:
        raise ValueError(f"reference must be between 0 and {K - 1}; got {reference}")

    X_fixed = np.asarray(X_fixed, dtype=float) if X_fixed is not None else np.empty((n, 0))
    if X_fixed.size > 0 and X_fixed.shape[0] != n:
        raise ValueError("X_fixed must have same number of rows as U")

    beta_list = []
    cov_list = []

    for _ in range(M):
        labels_m = np.array([rng.choice(K, p=U[i, :]) for i in range(n)])
        dummies_m = cluster_labels_to_dummies(labels_m, k=K, reference=reference)
        parts = []
        if X_fixed.size > 0:
            parts.append(X_fixed)
        parts.append(dummies_m)
        X_m = np.hstack(parts) if parts else dummies_m
        if add_intercept:
            X_m = sm.add_constant(X_m, has_constant="add")
        if np.linalg.matrix_rank(X_m) < X_m.shape[1]:
            continue
        try:
            if model_type == "ols":
                model = sm.OLS(y, X_m).fit()
            else:
                model = sm.Logit(y, X_m).fit(disp=False)
        except Exception:
            continue
        beta_list.append(model.params)
        cov_list.append(np.asarray(model.cov_params()))

    m_eff = len(beta_list)
    failed = M - m_eff
    if m_eff == 0:
        raise RuntimeError("All M replications had rank-deficient design matrix")

    beta_stack = np.array(beta_list)
    cov_stack = np.array(cov_list)

    beta_combined = np.mean(beta_stack, axis=0)
    W = np.mean(cov_stack, axis=0)
    if m_eff == 1:
        B = np.zeros_like(W)
    else:
        B = np.cov(beta_stack, rowvar=False, ddof=1)
    T = W + (1.0 + 1.0 / m_eff) * B
    se_combined = np.sqrt(np.diag(T))

    return {
        "beta_combined": beta_combined,
        "se_combined": se_combined,
        "cov_combined": T,
        "beta_list": beta_list,
        "m_eff": m_eff,
        "failed": failed,
    }
