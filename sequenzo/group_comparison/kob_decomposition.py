"""
@Author  : Yuqi Liang 梁彧祺
@File    : kob_decomposition.py
@Time    : 2026-02-17 19:31
@Desc    : 
Generic Kitagawa-Oaxaca-Blinder decomposition for group inequalities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Sequence

import numpy as np
import pandas as pd


@dataclass
class KOBDecompositionResult:
    total_gap: float
    explained: float
    unexplained_returns: float
    unexplained_intercept: float
    by_variable: pd.DataFrame
    group0_mean: float
    group1_mean: float


def _fit_ols(y: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray]:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if y.ndim != 1:
        raise ValueError("[_fit_ols] y must be a 1D array.")
    if X.ndim != 2:
        raise ValueError("[_fit_ols] X must be a 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("[_fit_ols] X and y must have the same number of rows.")
    X_design = np.column_stack([np.ones(X.shape[0], dtype=float), X])
    beta_all = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
    return float(beta_all[0]), beta_all[1:]


def _compute_reference_coefficients(
    beta0: np.ndarray,
    beta1: np.ndarray,
    term_ids: np.ndarray,
    reference: Literal["group0", "group1", "pooled"] = "group0",
    pooled_beta: Optional[np.ndarray] = None,
    majority_owner: Optional[np.ndarray] = None,
) -> np.ndarray:
    beta0 = np.asarray(beta0, dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    term_ids = np.asarray(term_ids)
    if beta0.shape != beta1.shape:
        raise ValueError("[_compute_reference_coefficients] beta0 and beta1 must have the same shape.")
    p = beta0.shape[0]
    if term_ids.shape[0] != p:
        raise ValueError("[_compute_reference_coefficients] term_ids length must equal number of coefficients.")
    if reference == "pooled":
        if pooled_beta is None:
            raise ValueError("[_compute_reference_coefficients] pooled_beta must be provided when reference='pooled'.")
        pooled_beta = np.asarray(pooled_beta, dtype=float)
        if pooled_beta.shape != beta0.shape:
            raise ValueError("[_compute_reference_coefficients] pooled_beta shape must match beta0.")

    unique_terms = np.unique(term_ids)
    term_index_map = {tid: idx for idx, tid in enumerate(unique_terms)}
    if majority_owner is not None:
        majority_owner = np.asarray(majority_owner, dtype=int)
        if majority_owner.shape[0] != len(unique_terms):
            raise ValueError("[_compute_reference_coefficients] majority_owner length must match number of unique term_ids.")
    else:
        majority_owner = np.full(len(unique_terms), -1, dtype=int)

    beta_star = np.zeros_like(beta0)
    for j in range(p):
        owner = majority_owner[term_index_map[term_ids[j]]]
        if owner == 0:
            beta_star[j] = beta0[j]
        elif owner == 1:
            beta_star[j] = beta1[j]
        else:
            if reference == "group0":
                beta_star[j] = beta0[j]
            elif reference == "group1":
                beta_star[j] = beta1[j]
            elif reference == "pooled":
                beta_star[j] = pooled_beta[j]
            else:
                raise ValueError(f"[_compute_reference_coefficients] Unknown reference='{reference}'.")
    return beta_star


def oaxaca_blinder_decomposition(
    y: np.ndarray,
    group: np.ndarray,
    X: np.ndarray,
    variable_names: Optional[Sequence[str]] = None,
    term_ids: Optional[Sequence[int]] = None,
    reference: Literal["group0", "group1", "pooled"] = "group0",
    majority_owner: Optional[Sequence[int]] = None,
) -> KOBDecompositionResult:
    y = np.asarray(y, dtype=float)
    g = np.asarray(group)
    X = np.asarray(X, dtype=float)
    if y.ndim != 1:
        raise ValueError("[oaxaca_blinder_decomposition] y must be a 1D array.")
    if X.ndim != 2:
        raise ValueError("[oaxaca_blinder_decomposition] X must be 2D.")
    if X.shape[0] != y.shape[0] or g.shape[0] != y.shape[0]:
        raise ValueError("[oaxaca_blinder_decomposition] y, group, and X must have the same length.")
    n, p = X.shape
    unique_groups = np.unique(g)
    if unique_groups.size != 2:
        raise ValueError("[oaxaca_blinder_decomposition] group must have exactly two distinct values.")
    g01 = (g == unique_groups[1]).astype(int)
    mask0 = g01 == 0
    mask1 = g01 == 1
    y0, y1 = y[mask0], y[mask1]
    X0, X1 = X[mask0, :], X[mask1, :]
    alpha0, beta0 = _fit_ols(y0, X0)
    alpha1, beta1 = _fit_ols(y1, X1)
    beta_pooled = None
    if reference == "pooled":
        _, beta_pooled = _fit_ols(y, X)

    if term_ids is None:
        term_ids = np.arange(p, dtype=int)
    else:
        term_ids = np.asarray(term_ids, dtype=int)
        if term_ids.shape[0] != p:
            raise ValueError("[oaxaca_blinder_decomposition] term_ids length must equal number of columns in X.")
    unique_terms = np.unique(term_ids)

    if variable_names is None:
        variable_names = [f"X{i+1}" for i in range(p)]
    elif len(variable_names) != p:
        raise ValueError("[oaxaca_blinder_decomposition] variable_names length must equal number of columns in X.")

    if majority_owner is not None:
        majority_owner_arr = np.asarray(majority_owner, dtype=int)
        if majority_owner_arr.shape[0] != len(unique_terms):
            raise ValueError("[oaxaca_blinder_decomposition] majority_owner length must match number of unique term_ids.")
    else:
        majority_owner_arr = np.full(len(unique_terms), -1, dtype=int)

    beta_star = _compute_reference_coefficients(
        beta0=beta0,
        beta1=beta1,
        term_ids=term_ids,
        reference=reference,
        pooled_beta=beta_pooled,
        majority_owner=majority_owner_arr,
    )

    Xbar0 = X0.mean(axis=0)
    Xbar1 = X1.mean(axis=0)
    ybar0 = float(y0.mean())
    ybar1 = float(y1.mean())
    total_gap = ybar0 - ybar1
    explained = float((Xbar0 - Xbar1) @ beta_star)
    returns = float(Xbar0 @ (beta0 - beta_star) + Xbar1 @ (beta_star - beta1))
    intercept_component = float(alpha0 - alpha1)
    explained_var = (Xbar0 - Xbar1) * beta_star
    returns_var = Xbar0 * (beta0 - beta_star) + Xbar1 * (beta_star - beta1)
    with np.errstate(divide="ignore", invalid="ignore"):
        explained_share = explained_var / total_gap if total_gap != 0 else np.nan
        returns_share = returns_var / total_gap if total_gap != 0 else np.nan
    by_variable = pd.DataFrame(
        {
            "variable": variable_names,
            "explained": explained_var,
            "returns": returns_var,
            "explained_share": explained_share,
            "returns_share": returns_share,
        }
    )
    return KOBDecompositionResult(
        total_gap=total_gap,
        explained=explained,
        unexplained_returns=returns,
        unexplained_intercept=intercept_component,
        by_variable=by_variable,
        group0_mean=ybar0,
        group1_mean=ybar1,
    )


def kob_decomposition(
    y: np.ndarray,
    group: np.ndarray,
    X: np.ndarray,
    variable_names: Optional[Sequence[str]] = None,
    term_ids: Optional[Sequence[int]] = None,
    reference: Literal["group0", "group1", "pooled"] = "group0",
    majority_owner: Optional[Sequence[int]] = None,
) -> KOBDecompositionResult:
    return oaxaca_blinder_decomposition(
        y=y,
        group=group,
        X=X,
        variable_names=variable_names,
        term_ids=term_ids,
        reference=reference,
        majority_owner=majority_owner,
    )
