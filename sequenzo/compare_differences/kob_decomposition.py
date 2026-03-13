"""
@Author  : Yuqi Liang 梁彧祺
@File    : oaxaca_blinder.py
@Time    : 2026-03-13
@Desc    : Generic Kitagawa–Oaxaca–Blinder (KOB) decomposition for group inequalities.

This module provides a general implementation of the Kitagawa–Oaxaca–Blinder
decomposition for differences in a numeric outcome between two groups.

It is designed to be *generic* and reusable:

- It can be applied to any continuous outcome and covariates (not only sequence data).
- It can be combined with sequence-derived typologies (e.g. life-course clusters)
  when used together with Sequenzo's sequence analysis and clustering tools.

The decomposition implementation follows:

- Jann, B. (2008). "The Blinder–Oaxaca decomposition for linear regression models."
  The Stata Journal, 8(4), 453–479.
- Fortin, N., Lemieux, T., & Firpo, S. (2011). "Decomposition methods in economics."
  In Handbook of Labor Economics, Vol. 4A, 1–102.

and is also motivated by the life-course-sensitive application in:

- Rowold, C., Struffolino, E., & Fasang, A. E. (2024).
  "Life-Course-Sensitive Analysis of Group Inequalities: Combining Sequence Analysis
  With the Kitagawa–Oaxaca–Blinder Decomposition."
  Sociological Methods & Research, 54(2), 646–705.

The key idea is to decompose the mean difference in an outcome between two groups
into:

- Explained (composition / endowments): differences in covariate means weighted
  by a reference coefficient vector.
- Unexplained (returns): differences in group-specific regression coefficients and
  intercepts, evaluated at the observed covariate means and reference coefficients.

This module focuses on the linear (OLS) case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class KOBDecompositionResult:
    """
    Container for Oaxaca–Blinder decomposition results.

    Attributes
    ----------
    total_gap : float
        Mean difference in outcome between group 0 and group 1.
    explained : float
        Explained (composition) component.
    unexplained_returns : float
        Returns component of the unexplained part (excludes intercept-only term).
    unexplained_intercept : float
        Intercept-only component of the unexplained part.
    by_variable : pandas.DataFrame
        Detailed decomposition by covariate (columns: 'variable', 'explained',
        'returns', 'explained_share', 'returns_share').
    group0_mean : float
        Mean outcome in group 0.
    group1_mean : float
        Mean outcome in group 1.
    """

    total_gap: float
    explained: float
    unexplained_returns: float
    unexplained_intercept: float
    by_variable: pd.DataFrame
    group0_mean: float
    group1_mean: float


def _fit_ols(
    y: np.ndarray,
    X: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Fit a simple OLS model y ~ 1 + X using normal equations.

    Parameters
    ----------
    y : np.ndarray
        Outcome vector of length n.
    X : np.ndarray
        Covariate matrix of shape (n, p) (no intercept column is included here).

    Returns
    -------
    alpha : float
        Intercept estimate.
    beta : np.ndarray
        Coefficient vector of length p.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if y.ndim != 1:
        raise ValueError("[_fit_ols] y must be a 1D array.")
    if X.ndim != 2:
        raise ValueError("[_fit_ols] X must be a 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("[_fit_ols] X and y must have the same number of rows.")

    n = X.shape[0]
    # Design matrix with intercept
    X_design = np.column_stack([np.ones(n, dtype=float), X])

    # Normal equations: beta_all = (X'X)^{-1} X'y
    XtX = X_design.T @ X_design
    Xty = X_design.T @ y
    beta_all = np.linalg.solve(XtX, Xty)

    alpha = float(beta_all[0])
    beta = beta_all[1:]
    return alpha, beta


def _compute_reference_coefficients(
    beta0: np.ndarray,
    beta1: np.ndarray,
    term_ids: np.ndarray,
    reference: Literal["group0", "group1", "pooled"] = "group0",
    pooled_beta: Optional[np.ndarray] = None,
    majority_owner: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build the reference coefficient vector beta* (non-discriminatory structure).

    Parameters
    ----------
    beta0 : np.ndarray
        Coefficient vector for group 0 (length p).
    beta1 : np.ndarray
        Coefficient vector for group 1 (length p).
    term_ids : np.ndarray
        Vector of length p assigning each column of X to a "term" (variable group),
        e.g., all dummies for one categorical variable can share the same id.
    reference : {"group0", "group1", "pooled"}, default "group0"
        Rule for building the reference coefficients:
        - "group0": use group 0 coefficients for all variables.
        - "group1": use group 1 coefficients for all variables.
        - "pooled": use pooled_beta (must be provided).
    pooled_beta : np.ndarray, optional
        Coefficient vector from pooled regression (required if reference="pooled").
    majority_owner : np.ndarray, optional
        Optional array of length equal to the number of unique term_ids, specifying
        which group is considered the "majority" or theoretically appropriate
        reference for each term.

        If provided, this allows cluster-specific or variable-specific reference
        coefficients, in the spirit of Rowold et al. (2024)'s Option III:
        - majority_owner[k] == 0: term k uses beta0.
        - majority_owner[k] == 1: term k uses beta1.
        - majority_owner[k] == -1: term k uses the global 'reference' rule.

    Returns
    -------
    np.ndarray
        Reference coefficient vector beta* of length p.
    """
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
    n_terms = len(unique_terms)
    term_index_map = {tid: idx for idx, tid in enumerate(unique_terms)}

    if majority_owner is not None:
        majority_owner = np.asarray(majority_owner, dtype=int)
        if majority_owner.shape[0] != n_terms:
            raise ValueError("[_compute_reference_coefficients] majority_owner length must match number of unique term_ids.")
    else:
        majority_owner = np.full(n_terms, -1, dtype=int)

    beta_star = np.zeros_like(beta0)

    for j in range(p):
        term = term_ids[j]
        term_idx = term_index_map[term]
        owner = majority_owner[term_idx]

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
) -> OaxacaBlinderResult:
    """
    Perform a Kitagawa–Oaxaca–Blinder decomposition for two groups.

    This function decomposes the mean difference in a continuous outcome y
    between two groups (group==0 vs group==1) into:

    - Explained (composition) part:
      (X̄_0 - X̄_1)' β*

    - Unexplained returns part:
      X̄_0'(β_0 - β*) + X̄_1'(β* - β_1)

    - Unexplained intercept part:
      (α_0 - α_1)

    where β* is a "reference" (or non-discriminatory) coefficient vector that
    can be defined in several ways (group 0, group 1, pooled, or term-wise
    majority).

    Parameters
    ----------
    y : np.ndarray
        Outcome vector of length n.
    group : np.ndarray
        Group indicator of length n. Must take exactly two distinct values,
        which are mapped internally to 0 and 1. By convention, group==0 is the
        "reference" or advantaged group and group==1 is the comparison group.
    X : np.ndarray
        Covariate matrix of shape (n, p). No intercept column should be
        included; this function adds its own intercept internally.
    variable_names : sequence of str, optional
        Optional names for each covariate (length p). Used in the detailed
        decomposition output. If None, variables are named "X1", "X2", ...
    term_ids : sequence of int, optional
        Optional grouping of coefficients into "terms" (length p). By default,
        each covariate is treated as its own term. Grouping is useful when
        several columns correspond to the same conceptual variable (e.g. dummy
        variables for one categorical predictor or a cluster typology).
    reference : {"group0", "group1", "pooled"}, default "group0"
        Global rule for constructing the reference coefficient vector β*:

        - "group0": use group 0 coefficients for all variables.
        - "group1": use group 1 coefficients for all variables.
        - "pooled": use pooled regression coefficients for all variables.

        When majority_owner is provided, this rule is applied only for terms
        where majority_owner[k] == -1.
    majority_owner : sequence of int, optional
        Optional array of length equal to the number of unique term_ids
        specifying, for each term, which group should define the reference
        coefficients:

        - 0: use group 0 coefficients for that term.
        - 1: use group 1 coefficients for that term.
        - -1: fall back to the global 'reference' rule.

        This allows cluster-specific reference coefficients similar to the
        "Option III" discussed in Rowold et al. (2024) for SA–KOB applications.

    Returns
    -------
    KOBDecompositionResult
        Dataclass with total gap, components, and detailed decomposition by
        covariate.
    """
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

    # Map arbitrary group labels to 0/1
    unique_groups = np.unique(g)
    if unique_groups.size != 2:
        raise ValueError("[oaxaca_blinder_decomposition] group must have exactly two distinct values.")

    group0_label, group1_label = unique_groups[0], unique_groups[1]
    g01 = (g == group1_label).astype(int)  # 0 for group0, 1 for group1

    mask0 = g01 == 0
    mask1 = g01 == 1

    y0, y1 = y[mask0], y[mask1]
    X0, X1 = X[mask0, :], X[mask1, :]

    # Fit separate OLS models for each group
    alpha0, beta0 = _fit_ols(y0, X0)
    alpha1, beta1 = _fit_ols(y1, X1)

    # Optionally fit pooled model for pooled reference coefficients
    alpha_pooled = None
    beta_pooled = None
    if reference == "pooled":
        alpha_pooled, beta_pooled = _fit_ols(y, X)

    # Term structure
    if term_ids is None:
        term_ids = np.arange(p, dtype=int)
    else:
        term_ids = np.asarray(term_ids, dtype=int)
        if term_ids.shape[0] != p:
            raise ValueError("[oaxaca_blinder_decomposition] term_ids length must equal number of columns in X.")

    unique_terms = np.unique(term_ids)
    n_terms = len(unique_terms)
    term_index_map = {tid: idx for idx, tid in enumerate(unique_terms)}

    # Variable names and term-level names
    if variable_names is None:
        variable_names = [f"X{i+1}" for i in range(p)]
    else:
        if len(variable_names) != p:
            raise ValueError("[oaxaca_blinder_decomposition] variable_names length must equal number of columns in X.")

    # Default majority_owner if not provided
    if majority_owner is not None:
        majority_owner_arr = np.asarray(majority_owner, dtype=int)
        if majority_owner_arr.shape[0] != n_terms:
            raise ValueError("[oaxaca_blinder_decomposition] majority_owner length must match number of unique term_ids.")
    else:
        majority_owner_arr = np.full(n_terms, -1, dtype=int)

    # Compute reference coefficients beta*
    beta_star = _compute_reference_coefficients(
        beta0=beta0,
        beta1=beta1,
        term_ids=term_ids,
        reference=reference,
        pooled_beta=beta_pooled,
        majority_owner=majority_owner_arr,
    )

    # Group-specific covariate means
    Xbar0 = X0.mean(axis=0)
    Xbar1 = X1.mean(axis=0)

    # Group-specific mean outcomes
    ybar0 = float(y0.mean())
    ybar1 = float(y1.mean())
    total_gap = ybar0 - ybar1

    # Composition (explained) component
    explained = float((Xbar0 - Xbar1) @ beta_star)

    # Returns component (excluding intercept)
    returns = float(
        Xbar0 @ (beta0 - beta_star) + Xbar1 @ (beta_star - beta1)
    )

    # Intercept-only component
    intercept_component = float(alpha0 - alpha1)

    # Detailed decomposition by variable
    explained_var = (Xbar0 - Xbar1) * beta_star
    returns_var = (
        Xbar0 * (beta0 - beta_star) + Xbar1 * (beta_star - beta1)
    )

    # Shares relative to the total gap
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
    """
    Thin wrapper around :func:`oaxaca_blinder_decomposition` with a shorter,
    method-oriented name (KOB = Kitagawa–Oaxaca–Blinder).

    This wrapper is provided for API clarity: users familiar with the
    life-course literature can directly call ``kob_decomposition`` to obtain
    a Kitagawa–Oaxaca–Blinder decomposition, while the underlying
    implementation remains the same.

    See :func:`oaxaca_blinder_decomposition` for full parameter
    documentation and references.
    """
    return oaxaca_blinder_decomposition(
        y=y,
        group=group,
        X=X,
        variable_names=variable_names,
        term_ids=term_ids,
        reference=reference,
        majority_owner=majority_owner,
    )

