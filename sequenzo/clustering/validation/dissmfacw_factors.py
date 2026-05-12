"""
@Author  : Yuqi Liang 梁彧祺
@File    : dissmfacw_factors.py
@Time    : 07/05/2025 20:29
@Desc    : 
Multi-factor discrepancy association (TraMineR ``dissmfacw`` core).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def _gower_matrix(diss: np.ndarray, squared: bool, weights: np.ndarray) -> np.ndarray:
    if squared:
        diss = np.square(diss)
    n = diss.shape[0]
    s = weights / np.sum(weights)
    one = np.ones(n, dtype=np.float64)
    left = np.eye(n) - np.outer(one, s)
    right = np.eye(n) - np.outer(s, one)
    return left @ (diss / -2.0) @ right


def _hatw_matrix_qr(predictor: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w_sqrt = np.sqrt(weights)
    weighted = predictor * w_sqrt[:, None]
    q_matrix, _ = np.linalg.qr(weighted, mode="reduced")
    hat = q_matrix @ q_matrix.T
    return np.outer(w_sqrt, w_sqrt) * hat


def _build_design_matrix(factors: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, List[str]]:
    n = len(factors)
    columns = [np.ones(n, dtype=np.float64)]
    assign = [0]
    term_names: List[str] = []
    term_id = 1
    for column in factors.columns:
        series = factors[column]
        term_names.append(str(column))
        if pd.api.types.is_numeric_dtype(series) and not isinstance(series.dtype, pd.CategoricalDtype):
            columns.append(series.to_numpy(dtype=np.float64))
            assign.append(term_id)
        else:
            dummies = pd.get_dummies(series.astype("category"), drop_first=False)
            for dummy_col in dummies.columns:
                columns.append(dummies[dummy_col].to_numpy(dtype=np.float64))
                assign.append(term_id)
        term_id += 1
    return np.column_stack(columns), np.asarray(assign, dtype=int), term_names


def dissmfacw_table(
    diss: np.ndarray,
    factors: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    squared: bool = False,
    gower: bool = False,
) -> pd.DataFrame:
    """
    Compute TraMineR-style pseudo-R² contributions for a multi-factor model.

    The returned table contains one row per factor plus a final ``Total`` row.
  Column ``PseudoR2`` matches ``dissmfacw(...)$mfac$PseudoR2`` used by
    WeightedCluster ``clustassoc``.
    """
    diss = np.asarray(diss, dtype=np.float64)
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square distance matrix.")
    if not isinstance(factors, pd.DataFrame):
        raise ValueError("factors must be a pandas DataFrame.")
    if factors.shape[0] != diss.shape[0]:
        raise ValueError("factors must have one row per observation.")

    valid = factors.notna().all(axis=1).to_numpy()
    diss = diss[np.ix_(valid, valid)]
    factors = factors.loc[valid].reset_index(drop=True)
    n = diss.shape[0]
    weights = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights[valid], dtype=np.float64)

    predictor, assign, term_names = _build_design_matrix(factors)
    unique_terms = sorted(set(assign.tolist()))
    n_terms = len(unique_terms) - 1
    g_matrix = diss if gower else _gower_matrix(diss, squared=squared, weights=weights)
    sc_tot = float(np.sum(weights * np.diag(g_matrix)))
    total_weight = float(np.sum(weights))

    ss_back = np.zeros(n_terms, dtype=np.float64)
    p_list = np.zeros(n_terms, dtype=int)
    for term_idx in range(1, n_terms + 1):
        keep = assign != term_idx
        reduced = predictor[:, keep]
        p_list[term_idx - 1] = int(np.sum(assign == term_idx))
        hat = _hatw_matrix_qr(reduced, weights)
        ss_back[term_idx - 1] = float(np.sum(hat * g_matrix))

    hat_full = _hatw_matrix_qr(predictor, weights)
    sc_exp_full = float(np.sum(hat_full * g_matrix))
    m = predictor.shape[1]
    sc_res_full = sc_tot - sc_exp_full

    pseudo_r2 = np.zeros(n_terms + 1, dtype=np.float64)
    pseudo_f = np.zeros(n_terms + 1, dtype=np.float64)
    pseudo_r2[-1] = sc_exp_full / sc_tot if sc_tot else np.nan
    pseudo_f[-1] = (sc_exp_full / (m - 1)) / (sc_res_full / (total_weight - m)) if sc_res_full else np.nan

    for term_idx in range(1, n_terms + 1):
        hat = _hatw_matrix_qr(predictor, weights)
        sc_exp = float(np.sum(hat * g_matrix))
        sc_res = sc_tot - sc_exp
        pseudo_r2[term_idx - 1] = (sc_exp - ss_back[term_idx - 1]) / sc_tot if sc_tot else np.nan
        pseudo_f[term_idx - 1] = ((sc_exp - ss_back[term_idx - 1]) / p_list[term_idx - 1]) / (sc_res / (total_weight - m)) if sc_res else np.nan

    return pd.DataFrame(
        {
            "Variable": term_names + ["Total"],
            "PseudoF": pseudo_f,
            "PseudoR2": pseudo_r2,
        }
    )
