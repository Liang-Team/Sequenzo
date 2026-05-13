"""
@Author  : Yuqi Liang 梁彧祺
@File    : oaxaca.py
@Time    : 2026-03-02 14:28
@Desc    :
Low-level twofold Oaxaca-Blinder / Kitagawa-Oaxaca-Blinder decomposition engine.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, Literal

import numpy as np
import pandas as pd

from .results import KOBDecompositionResult

_VALID_OWNERS = {-1, 0, 1}


def _fit_ols(
    y: np.ndarray,
    X: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if y.ndim != 1:
        raise ValueError("[_fit_ols] y must be a 1D array.")
    if X.ndim != 2:
        raise ValueError("[_fit_ols] X must be a 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("[_fit_ols] X and y must have the same number of rows.")

    X_design = np.column_stack([np.ones(X.shape[0], dtype=float), X])
    beta_all, _, rank, singular_values = np.linalg.lstsq(X_design, y, rcond=None)
    n_cols = X_design.shape[1]
    diagnostics: dict[str, Any] = {
        "rank": int(rank),
        "n_columns": n_cols,
        "rank_deficient": bool(rank < n_cols),
    }
    if singular_values.size > 0 and singular_values[-1] > 0:
        diagnostics["condition_number"] = float(singular_values[0] / singular_values[-1])
    else:
        diagnostics["condition_number"] = np.inf

    if diagnostics["rank_deficient"]:
        warnings.warn(
            "[_fit_ols] Design matrix is rank deficient; coefficients may not be uniquely identified.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(beta_all[0]), beta_all[1:], diagnostics


def _validate_finite_inputs(
    y: np.ndarray,
    group: np.ndarray,
    X: np.ndarray,
    *,
    drop_missing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    g = np.asarray(group)
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if not np.all(valid):
        if drop_missing:
            y = y[valid]
            g = g[valid]
            X = X[valid]
        else:
            raise ValueError(
                "[oaxaca_blinder_decomposition] y and X must not contain NaN or infinite values."
            )
    return y, g, X


def _resolve_binary_groups(
    group: np.ndarray,
    *,
    group0_value: Any = None,
    group1_value: Any = None,
) -> tuple[Any, Any, np.ndarray]:
    unique_groups = np.unique(group)
    if unique_groups.size != 2:
        raise ValueError(
            "[oaxaca_blinder_decomposition] group must have exactly two distinct values."
        )

    if group0_value is None and group1_value is None:
        group0_label, group1_label = unique_groups[0], unique_groups[1]
    elif group0_value is not None and group1_value is not None:
        group0_label, group1_label = group0_value, group1_value
        if group0_label not in unique_groups or group1_label not in unique_groups:
            raise ValueError(
                "[oaxaca_blinder_decomposition] group0_value and group1_value must appear in group."
            )
        if group0_label == group1_label:
            raise ValueError(
                "[oaxaca_blinder_decomposition] group0_value and group1_value must be different."
            )
    else:
        raise ValueError(
            "[oaxaca_blinder_decomposition] Provide both group0_value and group1_value, or neither."
        )

    g01 = (group == group1_label).astype(int)
    return group0_label, group1_label, g01


def _validate_owner_vector(
    owners: np.ndarray,
    *,
    name: str,
    expected_length: int,
) -> np.ndarray:
    owners = np.asarray(owners, dtype=int)
    if owners.shape[0] != expected_length:
        raise ValueError(f"[oaxaca_blinder_decomposition] {name} length must equal number of columns in X.")
    if not set(owners.tolist()).issubset(_VALID_OWNERS):
        raise ValueError(f"[oaxaca_blinder_decomposition] {name} must contain only -1, 0, or 1.")
    return owners


def _resolve_coefficient_owners(
    p: int,
    term_ids: np.ndarray,
    *,
    coefficient_owner_by_column: Optional[Sequence[int]] = None,
    majority_owner: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if coefficient_owner_by_column is not None:
        return _validate_owner_vector(
            np.asarray(coefficient_owner_by_column, dtype=int),
            name="coefficient_owner_by_column",
            expected_length=p,
        )

    if majority_owner is not None:
        majority_owner_arr = np.asarray(majority_owner, dtype=int)
        unique_terms = np.unique(term_ids)
        if majority_owner_arr.shape[0] != len(unique_terms):
            raise ValueError(
                "[oaxaca_blinder_decomposition] majority_owner length must match number of unique term_ids."
            )
        if not set(majority_owner_arr.tolist()).issubset(_VALID_OWNERS):
            raise ValueError(
                "[oaxaca_blinder_decomposition] majority_owner must contain only -1, 0, or 1."
            )
        term_index_map = {tid: idx for idx, tid in enumerate(unique_terms)}
        owners = np.array(
            [majority_owner_arr[term_index_map[tid]] for tid in term_ids],
            dtype=int,
        )
        warnings.warn(
            "[oaxaca_blinder_decomposition] majority_owner is deprecated; "
            "use coefficient_owner_by_column for cluster-specific reference coefficients.",
            DeprecationWarning,
            stacklevel=3,
        )
        return owners

    return np.full(p, -1, dtype=int)


def _default_category_ids(term_ids: np.ndarray) -> np.ndarray:
    category_ids = np.zeros(term_ids.shape[0], dtype=int)
    for term_id in np.unique(term_ids):
        columns = np.flatnonzero(term_ids == term_id)
        for offset, column in enumerate(columns, start=1):
            category_ids[column] = offset
    return category_ids


def _category_proportions(
    X: np.ndarray,
    columns: np.ndarray,
    category_ids_for_term: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    proportions = np.zeros(n_categories, dtype=float)
    for column, category in zip(columns, category_ids_for_term):
        proportions[int(category)] = X[:, column].mean()

    represented = {int(category) for category in category_ids_for_term}
    missing = [category for category in range(n_categories) if category not in represented]
    if len(missing) == 1:
        proportions[missing[0]] = 1.0 - sum(float(proportions[category]) for category in represented)
    return proportions


def _category_mu_from_beta(
    beta: np.ndarray,
    columns: np.ndarray,
    category_ids_for_term: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    mu = np.zeros(n_categories, dtype=float)
    for column, category in zip(columns, category_ids_for_term):
        mu[int(category)] = beta[column]
    return mu


def _yun_gamma(mu: np.ndarray) -> np.ndarray:
    return mu - mu.mean()


def _resolve_mu_star(
    mu0: np.ndarray,
    mu1: np.ndarray,
    owner_by_category: np.ndarray,
    *,
    reference: Literal["group0", "group1", "pooled"],
    mu_pooled: Optional[np.ndarray],
) -> np.ndarray:
    mu_star = np.zeros_like(mu0)
    for category, owner in enumerate(owner_by_category):
        if owner == 0:
            mu_star[category] = mu0[category]
        elif owner == 1:
            mu_star[category] = mu1[category]
        elif reference == "group0":
            mu_star[category] = mu0[category]
        elif reference == "group1":
            mu_star[category] = mu1[category]
        elif reference == "pooled":
            if mu_pooled is None:
                raise ValueError("[_resolve_mu_star] mu_pooled must be provided when reference='pooled'.")
            mu_star[category] = mu_pooled[category]
        else:
            raise ValueError(f"[_resolve_mu_star] Unknown reference='{reference}'.")
    return mu_star


def _detailed_normalized_term_contributions(
    *,
    columns: np.ndarray,
    category_ids_for_term: np.ndarray,
    n_categories: int,
    beta0: np.ndarray,
    beta1: np.ndarray,
    owner_by_category: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    reference: Literal["group0", "group1", "pooled"],
    mu_pooled: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu0 = _category_mu_from_beta(beta0, columns, category_ids_for_term, n_categories)
    mu1 = _category_mu_from_beta(beta1, columns, category_ids_for_term, n_categories)
    mu_star = _resolve_mu_star(
        mu0,
        mu1,
        owner_by_category,
        reference=reference,
        mu_pooled=mu_pooled,
    )

    gamma0 = _yun_gamma(mu0)
    gamma1 = _yun_gamma(mu1)
    gamma_star = _yun_gamma(mu_star)

    explained_columns = np.zeros(columns.shape[0], dtype=float)
    returns_columns = np.zeros(columns.shape[0], dtype=float)
    explained_categories = np.zeros(n_categories, dtype=float)
    returns_categories = np.zeros(n_categories, dtype=float)

    for category in range(n_categories):
        explained_categories[category] = (p0[category] - p1[category]) * gamma_star[category]
        returns_categories[category] = (
            p0[category] * (gamma0[category] - gamma_star[category])
            + p1[category] * (gamma_star[category] - gamma1[category])
        )

    for idx, (column, category) in enumerate(zip(columns, category_ids_for_term)):
        explained_columns[idx] = explained_categories[int(category)]
        returns_columns[idx] = returns_categories[int(category)]

    return explained_columns, returns_columns, explained_categories, returns_categories


def _build_normalized_by_term(
    term_id: int,
    n_categories: int,
    n_columns: int,
    explained_categories: np.ndarray,
    returns_categories: np.ndarray,
    total_gap: float,
) -> dict[str, Any]:
    explained_total = float(explained_categories.sum())
    returns_total = float(returns_categories.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        explained_share = explained_total / total_gap if total_gap != 0 else np.nan
        returns_share = returns_total / total_gap if total_gap != 0 else np.nan
    return {
        "term_id": term_id,
        "explained": explained_total,
        "returns": returns_total,
        "explained_share": explained_share,
        "returns_share": returns_share,
        "n_columns": n_columns,
        "n_categories": n_categories,
    }


def _owner_by_category_for_term(
    columns: np.ndarray,
    category_ids_for_term: np.ndarray,
    n_categories: int,
    owners_by_column: np.ndarray,
    owner_by_category_by_term: Optional[dict[int, np.ndarray]],
    term_id: int,
) -> np.ndarray:
    if owner_by_category_by_term is not None and term_id in owner_by_category_by_term:
        owners = np.asarray(owner_by_category_by_term[term_id], dtype=int)
        if owners.shape[0] != n_categories:
            raise ValueError(
                "[oaxaca_blinder_decomposition] owner_by_category length must equal n_categories for the term."
            )
        if not set(owners.tolist()).issubset(_VALID_OWNERS):
            raise ValueError(
                "[oaxaca_blinder_decomposition] owner_by_category must contain only -1, 0, or 1."
            )
        return owners

    owner_by_category = np.full(n_categories, -1, dtype=int)
    for column, category in zip(columns, category_ids_for_term):
        owner_by_category[int(category)] = int(owners_by_column[column])

    represented = {int(category) for category in category_ids_for_term}
    missing = [category for category in range(n_categories) if category not in represented]
    if len(missing) != 1:
        raise ValueError(
            "[oaxaca_blinder_decomposition] Each categorical term must omit exactly one baseline "
            "category. Pass owner_by_category_by_term to override owners for all categories."
        )
    return owner_by_category


def _resolve_categorical_terms(
    term_ids: np.ndarray,
    *,
    normalize_categorical: bool,
    categorical_terms: Optional[Sequence[int]],
) -> set[int]:
    if not normalize_categorical:
        return set()

    if categorical_terms is not None:
        return set(int(t) for t in categorical_terms)

    unique_terms, counts = np.unique(term_ids, return_counts=True)
    return set(int(t) for t, count in zip(unique_terms, counts) if count > 1)


def _compute_reference_coefficients(
    beta0: np.ndarray,
    beta1: np.ndarray,
    owners: np.ndarray,
    *,
    reference: Literal["group0", "group1", "pooled"] = "group0",
    pooled_beta: Optional[np.ndarray] = None,
) -> np.ndarray:
    beta0 = np.asarray(beta0, dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta0.shape != beta1.shape:
        raise ValueError("[_compute_reference_coefficients] beta0 and beta1 must have the same shape.")
    if owners.shape[0] != beta0.shape[0]:
        raise ValueError(
            "[_compute_reference_coefficients] owners length must equal number of coefficients."
        )
    if reference == "pooled":
        if pooled_beta is None:
            raise ValueError("[_compute_reference_coefficients] pooled_beta must be provided when reference='pooled'.")
        pooled_beta = np.asarray(pooled_beta, dtype=float)
        if pooled_beta.shape != beta0.shape:
            raise ValueError("[_compute_reference_coefficients] pooled_beta shape must match beta0.")

    beta_star = np.zeros_like(beta0)
    for j, owner in enumerate(owners):
        if owner == 0:
            beta_star[j] = beta0[j]
        elif owner == 1:
            beta_star[j] = beta1[j]
        elif reference == "group0":
            beta_star[j] = beta0[j]
        elif reference == "group1":
            beta_star[j] = beta1[j]
        elif reference == "pooled":
            beta_star[j] = pooled_beta[j]
        else:
            raise ValueError(f"[_compute_reference_coefficients] Unknown reference='{reference}'.")
    return beta_star


def _build_by_term(
    term_ids: np.ndarray,
    unique_terms: np.ndarray,
    explained_var: np.ndarray,
    returns_var: np.ndarray,
    total_gap: float,
) -> pd.DataFrame:
    explained_terms = []
    returns_terms = []
    n_columns_terms = []
    for term_id in unique_terms:
        mask = term_ids == term_id
        explained_terms.append(float(explained_var[mask].sum()))
        returns_terms.append(float(returns_var[mask].sum()))
        n_columns_terms.append(int(mask.sum()))

    explained_terms_arr = np.asarray(explained_terms, dtype=float)
    returns_terms_arr = np.asarray(returns_terms, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        explained_share = explained_terms_arr / total_gap if total_gap != 0 else np.nan
        returns_share = returns_terms_arr / total_gap if total_gap != 0 else np.nan

    return pd.DataFrame(
        {
            "term_id": unique_terms,
            "explained": explained_terms_arr,
            "returns": returns_terms_arr,
            "explained_share": explained_share,
            "returns_share": returns_share,
            "n_columns": np.asarray(n_columns_terms, dtype=int),
        }
    )


def oaxaca_blinder_decomposition(
    y: np.ndarray,
    group: np.ndarray,
    X: np.ndarray,
    variable_names: Optional[Sequence[str]] = None,
    term_ids: Optional[Sequence[int]] = None,
    reference: Literal["group0", "group1", "pooled"] = "group0",
    majority_owner: Optional[Sequence[int]] = None,
    *,
    coefficient_owner_by_column: Optional[Sequence[int]] = None,
    group0_value: Any = None,
    group1_value: Any = None,
    normalize_categorical: bool = False,
    categorical_terms: Optional[Sequence[int]] = None,
    category_ids: Optional[Sequence[int]] = None,
    n_categories_by_term: Optional[dict[int, int]] = None,
    owner_by_category_by_term: Optional[dict[int, Sequence[int]]] = None,
    drop_missing: bool = False,
) -> KOBDecompositionResult:
    y, g, X = _validate_finite_inputs(y, group, X, drop_missing=drop_missing)

    if y.ndim != 1:
        raise ValueError("[oaxaca_blinder_decomposition] y must be a 1D array.")
    if X.ndim != 2:
        raise ValueError("[oaxaca_blinder_decomposition] X must be 2D.")
    if X.shape[0] != y.shape[0] or g.shape[0] != y.shape[0]:
        raise ValueError("[oaxaca_blinder_decomposition] y, group, and X must have the same length.")

    group0_label, group1_label, g01 = _resolve_binary_groups(
        g,
        group0_value=group0_value,
        group1_value=group1_value,
    )

    n, p = X.shape
    mask0 = g01 == 0
    mask1 = g01 == 1
    y0, y1 = y[mask0], y[mask1]
    X0, X1 = X[mask0, :], X[mask1, :]

    alpha0, beta0, diag0 = _fit_ols(y0, X0)
    alpha1, beta1, diag1 = _fit_ols(y1, X1)

    if term_ids is None:
        term_ids_arr = np.arange(p, dtype=int)
    else:
        term_ids_arr = np.asarray(term_ids, dtype=int)
        if term_ids_arr.shape[0] != p:
            raise ValueError("[oaxaca_blinder_decomposition] term_ids length must equal number of columns in X.")
    unique_terms = np.unique(term_ids_arr)

    categorical_term_set = _resolve_categorical_terms(
        term_ids_arr,
        normalize_categorical=normalize_categorical,
        categorical_terms=categorical_terms,
    )

    if category_ids is None:
        category_ids_arr = _default_category_ids(term_ids_arr)
    else:
        category_ids_arr = np.asarray(category_ids, dtype=int)
        if category_ids_arr.shape[0] != p:
            raise ValueError("[oaxaca_blinder_decomposition] category_ids length must equal number of columns in X.")

    beta_pooled = None
    pooled_diag: dict[str, Any] = {}
    if reference == "pooled":
        _, beta_pooled, pooled_diag = _fit_ols(y, X)

    owners = _resolve_coefficient_owners(
        p,
        term_ids_arr,
        coefficient_owner_by_column=coefficient_owner_by_column,
        majority_owner=majority_owner,
    )
    beta_star = _compute_reference_coefficients(
        beta0=beta0,
        beta1=beta1,
        owners=owners,
        reference=reference,
        pooled_beta=beta_pooled,
    )

    if variable_names is None:
        variable_names_list = [f"X{i + 1}" for i in range(p)]
    elif len(variable_names) != p:
        raise ValueError("[oaxaca_blinder_decomposition] variable_names length must equal number of columns in X.")
    else:
        variable_names_list = list(variable_names)

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
    diagnostics_note: Optional[str] = None

    if categorical_term_set:
        mu_pooled_by_term: dict[int, np.ndarray] = {}
        if reference == "pooled":
            for term_id in sorted(categorical_term_set):
                columns = np.flatnonzero(term_ids_arr == term_id)
                category_ids_for_term = category_ids_arr[columns]
                if n_categories_by_term is not None and term_id in n_categories_by_term:
                    n_categories = int(n_categories_by_term[term_id])
                else:
                    n_categories = int(columns.size) + 1
                mu_pooled_by_term[term_id] = _category_mu_from_beta(
                    beta_pooled,
                    columns,
                    category_ids_for_term,
                    n_categories,
                )

        by_term_rows = []
        by_category_rows = []
        explained_raw = explained
        returns_raw = returns
        for term_id in unique_terms:
            columns = np.flatnonzero(term_ids_arr == term_id)
            if term_id in categorical_term_set:
                category_ids_for_term = category_ids_arr[columns]
                if n_categories_by_term is not None and term_id in n_categories_by_term:
                    n_categories = int(n_categories_by_term[term_id])
                else:
                    n_categories = int(columns.size) + 1
                owner_by_category = _owner_by_category_for_term(
                    columns,
                    category_ids_for_term,
                    n_categories,
                    owners,
                    owner_by_category_by_term,
                    int(term_id),
                )
                p0 = _category_proportions(X0, columns, category_ids_for_term, n_categories)
                p1 = _category_proportions(X1, columns, category_ids_for_term, n_categories)
                explained_columns, returns_columns, explained_categories, returns_categories = (
                    _detailed_normalized_term_contributions(
                        columns=columns,
                        category_ids_for_term=category_ids_for_term,
                        n_categories=n_categories,
                        beta0=beta0,
                        beta1=beta1,
                        owner_by_category=owner_by_category,
                        p0=p0,
                        p1=p1,
                        reference=reference,
                        mu_pooled=mu_pooled_by_term.get(int(term_id)),
                    )
                )
                explained_var[columns] = explained_columns
                returns_var[columns] = returns_columns
                by_term_rows.append(
                    _build_normalized_by_term(
                        term_id=int(term_id),
                        n_categories=n_categories,
                        n_columns=int(columns.size),
                        explained_categories=explained_categories,
                        returns_categories=returns_categories,
                        total_gap=total_gap,
                    )
                )
                for category in range(n_categories):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        by_category_rows.append(
                            {
                                "term_id": int(term_id),
                                "category_id": int(category),
                                "is_reference_category": bool(
                                    int(category) not in {int(c) for c in category_ids_for_term}
                                ),
                                "coefficient_owner": int(owner_by_category[category]),
                                "explained": float(explained_categories[category]),
                                "returns": float(returns_categories[category]),
                                "explained_share": (
                                    float(explained_categories[category] / total_gap)
                                    if total_gap != 0
                                    else np.nan
                                ),
                                "returns_share": (
                                    float(returns_categories[category] / total_gap)
                                    if total_gap != 0
                                    else np.nan
                                ),
                            }
                        )
            else:
                explained_term = float(explained_var[columns].sum())
                returns_term = float(returns_var[columns].sum())
                with np.errstate(divide="ignore", invalid="ignore"):
                    explained_share = explained_term / total_gap if total_gap != 0 else np.nan
                    returns_share = returns_term / total_gap if total_gap != 0 else np.nan
                by_term_rows.append(
                    {
                        "term_id": int(term_id),
                        "explained": explained_term,
                        "returns": returns_term,
                        "explained_share": explained_share,
                        "returns_share": returns_share,
                        "n_columns": int(columns.size),
                        "n_categories": np.nan,
                    }
                )
        by_term = pd.DataFrame(by_term_rows)
        by_category = pd.DataFrame(by_category_rows) if by_category_rows else pd.DataFrame()

        explained_columns_sum = float(explained_var.sum())
        returns_columns_sum = float(returns_var.sum())
        if not np.isclose(explained_raw, explained_columns_sum, atol=1e-8):
            diagnostics_note = (
                "Normalized categorical detailed contributions differ from the raw twofold "
                f"explained component ({explained_raw:.8f} vs {explained_columns_sum:.8f}). "
                "Scalar explained/returns keep the raw twofold totals; use by_category for "
                "reference-invariant category attribution."
            )
        elif not np.isclose(returns_raw, returns_columns_sum, atol=1e-8):
            diagnostics_note = (
                "Normalized categorical detailed returns differ from the raw twofold "
                f"returns component ({returns_raw:.8f} vs {returns_columns_sum:.8f}). "
                "Scalar explained/returns keep the raw twofold totals; use by_category for "
                "reference-invariant category attribution."
            )
        else:
            diagnostics_note = None
    else:
        by_term = _build_by_term(term_ids_arr, unique_terms, explained_var, returns_var, total_gap)
        by_term["n_categories"] = np.nan
        by_category = pd.DataFrame()
        diagnostics_note = None

    with np.errstate(divide="ignore", invalid="ignore"):
        explained_share = explained_var / total_gap if total_gap != 0 else np.nan
        returns_share = returns_var / total_gap if total_gap != 0 else np.nan

    by_column = pd.DataFrame(
        {
            "column": variable_names_list,
            "term_id": term_ids_arr,
            "category_id": category_ids_arr,
            "coefficient_owner": owners,
            "explained": explained_var,
            "returns": returns_var,
            "explained_share": explained_share,
            "returns_share": returns_share,
        }
    )

    gap_direction = f"{group0_label} minus {group1_label}"
    diagnostics = {
        "group0": {
            "n": int(mask0.sum()),
            "ols": diag0,
            "normalized_categorical_terms": sorted(categorical_term_set),
        },
        "group1": {
            "n": int(mask1.sum()),
            "ols": diag1,
            "normalized_categorical_terms": sorted(categorical_term_set),
        },
        "reference": reference,
        "normalize_categorical": normalize_categorical,
    }
    if pooled_diag:
        diagnostics["pooled_ols"] = pooled_diag
        diagnostics["pooled_reference_note"] = (
            "reference='pooled' uses coefficients from an OLS model fitted on the pooled "
            "sample without a group indicator."
        )
    if diagnostics_note is not None:
        diagnostics["normalized_component_note"] = diagnostics_note

    return KOBDecompositionResult(
        total_gap=total_gap,
        explained=explained,
        unexplained_returns=returns,
        unexplained_intercept=intercept_component,
        by_column=by_column,
        by_term=by_term,
        by_category=by_category,
        group0_mean=ybar0,
        group1_mean=ybar1,
        group0_label=group0_label,
        group1_label=group1_label,
        gap_direction=gap_direction,
        diagnostics=diagnostics,
    )


get_oaxaca_blinder_decomposition = oaxaca_blinder_decomposition

__all__ = [
    "oaxaca_blinder_decomposition",
    "get_oaxaca_blinder_decomposition",
]
