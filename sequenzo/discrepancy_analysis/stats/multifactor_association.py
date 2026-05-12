"""Multifactor distance–covariate association (TraMineR: dissmfacw)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd

def gower_matrix(
    distance_matrix: np.ndarray,
    squared: bool = True,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Centered Gower matrix used by multifactor_association()."""
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

    s = weights / weights.sum()
    one = np.ones(n, dtype=float)
    left = np.eye(n) - np.outer(one, s)
    right = np.eye(n) - np.outer(s, one)
    return left @ (diss / -2.0) @ right


def _weighted_hat_matrix_qr(design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted hat matrix used by multifactor_association()."""
    X = np.asarray(design, dtype=float)
    w = np.asarray(weights, dtype=float)

    if X.ndim != 2:
        raise ValueError("[_weighted_hat_matrix_qr] design must be 2D.")
    n, _ = X.shape

    if w.ndim != 1 or len(w) != n:
        raise ValueError("[_weighted_hat_matrix_qr] weights must be a 1D array of length n.")
    if np.any(w < 0):
        raise ValueError("[_weighted_hat_matrix_qr] weights must be non-negative.")
    if np.allclose(w, 0.0):
        w = np.ones_like(w)

    w_sqrt = np.sqrt(w)
    Xw = X * w_sqrt[:, None]
    Q, _ = np.linalg.qr(Xw, mode="reduced")
    hat = Q @ Q.T
    return np.outer(w_sqrt, w_sqrt) * hat


def build_multifactor_design(
    factors: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build a TraMineR-style model matrix from a covariate data frame.

    The intercept is encoded with term id 0. Each original factor column is
    assigned one term id and treatment-coded dummy columns when needed.
    """
    if factors.shape[0] == 0:
        raise ValueError("[build_multifactor_design] factors must contain at least one row.")

    n = len(factors)
    columns: List[np.ndarray] = [np.ones(n, dtype=float)]
    term_ids: List[int] = [0]
    term_labels: List[str] = ["(Intercept)"]

    term_index = 1
    for column_name in factors.columns:
        series = factors[column_name]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 2:
            columns.append(series.astype(float).to_numpy())
            term_ids.extend([term_index])
        else:
            dummies = pd.get_dummies(series.astype(str), drop_first=True)
            if dummies.shape[1] == 0:
                columns.append(np.zeros(n, dtype=float))
                term_ids.append(term_index)
            else:
                for dummy_name in dummies.columns:
                    columns.append(dummies[dummy_name].to_numpy(dtype=float))
                    term_ids.append(term_index)
        term_labels.append(str(column_name))
        term_index += 1

    design = np.column_stack(columns)
    return design, np.asarray(term_ids, dtype=int), term_labels


def distance_multifactor_anova(
    distance_matrix: np.ndarray,
    design_matrix: np.ndarray,
    term_ids: Sequence[int],
    term_labels: Optional[Sequence[str]] = None,
    weights: Optional[np.ndarray] = None,
    gower: bool = False,
    squared: bool = False,
    R: int = 0,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Distance-based multifactor ANOVA matching TraMineR's multifactor_association()."""
    D = np.asarray(distance_matrix, dtype=float)
    X = np.asarray(design_matrix, dtype=float)
    var_list = np.asarray(term_ids, dtype=int)

    if D.shape[0] != D.shape[1]:
        raise ValueError("[distance_multifactor_anova] distance_matrix must be square.")
    n = D.shape[0]
    if X.ndim != 2 or X.shape[0] != n:
        raise ValueError("[distance_multifactor_anova] design_matrix must have n rows.")
    if X.shape[1] != len(var_list):
        raise ValueError("[distance_multifactor_anova] term_ids length must match design columns.")

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    unique_terms = np.unique(var_list)
    n_terms = len(unique_terms) - 1
    if term_labels is None:
        term_labels = [f"Term{idx}" for idx in range(1, n_terms + 1)]
    elif len(term_labels) != n_terms:
        raise ValueError("[distance_multifactor_anova] term_labels length must match the number of factors.")

    G = D if gower else gower_matrix(D, squared=squared, weights=weights)
    total_weight = float(weights.sum())
    sc_tot = float((weights * np.diag(G)).sum())

    var_list_index = np.arange(X.shape[1])
    p_list = np.zeros(n_terms, dtype=int)
    ss_backward = np.zeros(n_terms, dtype=float)
    for term_idx in range(1, n_terms + 1):
        cols_keep = var_list != term_idx
        pred = X[:, cols_keep]
        p_list[term_idx - 1] = int(np.sum(var_list == term_idx))
        if pred.shape[1] == 0:
            ss_backward[term_idx - 1] = 0.0
        else:
            hat = _weighted_hat_matrix_qr(pred, weights=weights)
            ss_backward[term_idx - 1] = float((hat * G).sum())

    def _statistics_for_predictor(
        base_predictor: np.ndarray,
        perm_predictor: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if perm_predictor is None:
            perm_predictor = base_predictor

        m = base_predictor.shape[1]
        hat_full = _weighted_hat_matrix_qr(perm_predictor, weights=weights)
        sc_exp_complete = float((hat_full * G).sum())
        sc_res_complete = sc_tot - sc_exp_complete

        r2_list = np.zeros(n_terms + 1, dtype=float)
        f_list = np.zeros(n_terms + 1, dtype=float)
        r2_list[-1] = sc_exp_complete / sc_tot if sc_tot > 0 else 0.0
        if m > 1 and sc_res_complete > 0:
            f_list[-1] = (sc_exp_complete / (m - 1.0)) / (sc_res_complete / (total_weight - m))
        else:
            f_list[-1] = 0.0

        if n_terms == 1:
            r2_list[0] = r2_list[-1]
            f_list[0] = f_list[-1]
            return f_list, r2_list

        for term_idx in range(1, n_terms + 1):
            cols_term = var_list == term_idx
            pred_term = base_predictor.copy()
            pred_term[:, cols_term] = perm_predictor[:, cols_term]
            hat_term = _weighted_hat_matrix_qr(pred_term, weights=weights)
            sc_exp = float((hat_term * G).sum())
            sc_res = sc_tot - sc_exp
            delta_sc_exp = sc_exp - ss_backward[term_idx - 1]
            if p_list[term_idx - 1] > 0 and sc_res > 0:
                f_list[term_idx - 1] = (
                    delta_sc_exp / p_list[term_idx - 1]
                ) / (sc_res / (total_weight - pred_term.shape[1]))
            else:
                f_list[term_idx - 1] = 0.0
            r2_list[term_idx - 1] = delta_sc_exp / sc_tot if sc_tot > 0 else 0.0

        return f_list, r2_list

    f_obs, r2_obs = _statistics_for_predictor(X)
    p_values = np.full_like(f_obs, np.nan, dtype=float)

    if R is not None and R > 1:
        if random_state is not None:
            np.random.seed(random_state)

        perm_f = np.zeros((R, n_terms + 1))
        perm_f[0, :] = f_obs

        for replicate in range(1, R):
            perm = np.random.permutation(n)
            perm_predictor = X[perm, :]
            f_perm, _ = _statistics_for_predictor(X, perm_predictor)
            perm_f[replicate, :] = f_perm

        for j in range(n_terms + 1):
            p_values[j] = float(np.mean(perm_f[:, j] >= f_obs[j]))

    var_names = list(term_labels) + ["Total"]
    summary_df = pd.DataFrame(
        {
            "Variable": var_names,
            "PseudoF": f_obs,
            "PseudoR2": r2_obs,
            "p_value": p_values,
        }
    )
    return {"summary": summary_df, "g_matrix": G, "weights": weights}


def multifactor_association(
    distance_matrix: np.ndarray,
    factors: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    R: int = 0,
    squared: bool = False,
    gower: bool = False,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """TraMineR-compatible wrapper around distance_multifactor_anova()."""
    design, term_ids, term_labels = build_multifactor_design(factors)
    result = distance_multifactor_anova(
        distance_matrix=distance_matrix,
        design_matrix=design,
        term_ids=term_ids,
        term_labels=term_labels[1:],
        weights=weights,
        gower=gower,
        squared=squared,
        R=R,
        random_state=random_state,
    )
    return result["summary"].set_index("Variable")


