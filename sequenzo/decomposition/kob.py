"""
@Author  : Yuqi Liang 梁彧祺
@File    : kob.py
@Time    : 2026-03-01 13:26
@Desc    :
Kitagawa-Oaxaca-Blinder decomposition public API.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Literal

import numpy as np
import pandas as pd

from .oaxaca import oaxaca_blinder_decomposition
from .results import KOBDecompositionResult, KOBBootstrapResult


def get_kob_decomposition(
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
    """
    Twofold Kitagawa-Oaxaca-Blinder decomposition.

    Positive ``total_gap`` means ``group0`` has a higher mean outcome than ``group1``.
    Use ``group0_value`` / ``group1_value`` to control which labels map to each side.

    When ``reference='pooled'``, reference coefficients come from an OLS model fitted
    on the pooled sample without a group indicator.

    For SA-KOB with cluster dummies, set ``normalize_categorical=True`` and pass the
    cluster typology term in ``categorical_terms``. Use ``owner_by_category_by_term``
    to assign cluster-specific reference coefficients (Rowold, Struffolino, and Fasang
    2025, option III).
    """
    return oaxaca_blinder_decomposition(
        y=y,
        group=group,
        X=X,
        variable_names=variable_names,
        term_ids=term_ids,
        reference=reference,
        majority_owner=majority_owner,
        coefficient_owner_by_column=coefficient_owner_by_column,
        group0_value=group0_value,
        group1_value=group1_value,
        normalize_categorical=normalize_categorical,
        categorical_terms=categorical_terms,
        category_ids=category_ids,
        n_categories_by_term=n_categories_by_term,
        owner_by_category_by_term=owner_by_category_by_term,
        drop_missing=drop_missing,
    )


def _percentile_bounds(confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    lower = 100.0 * alpha / 2.0
    upper = 100.0 * (1.0 - alpha / 2.0)
    return lower, upper


def _validate_confidence_level(confidence_level: float) -> None:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")


def _percentile_ci(samples: np.ndarray, confidence_level: float) -> tuple[float, float]:
    _validate_confidence_level(confidence_level)
    lower, upper = _percentile_bounds(confidence_level)
    lo, hi = np.percentile(samples, [lower, upper])
    return float(lo), float(hi)


def _bootstrap_sample_indices(
    group: np.ndarray,
    *,
    group0_value: Any,
    group1_value: Any,
    rng: np.random.Generator,
    stratified: bool,
) -> np.ndarray:
    group = np.asarray(group)
    if stratified:
        unique_groups = np.unique(group)
        if unique_groups.size != 2:
            raise ValueError(
                "[_bootstrap_sample_indices] group must have exactly two distinct values "
                "for stratified resampling."
            )
        if group0_value is None and group1_value is None:
            group0_label, group1_label = unique_groups[0], unique_groups[1]
        else:
            group0_label, group1_label = group0_value, group1_value
        idx0 = np.flatnonzero(group == group0_label)
        idx1 = np.flatnonzero(group == group1_label)
        if idx0.size == 0 or idx1.size == 0:
            raise ValueError("[get_kob_decomposition_bootstrap] Both groups must be non-empty for stratified resampling.")
        boot0 = rng.choice(idx0, size=idx0.size, replace=True)
        boot1 = rng.choice(idx1, size=idx1.size, replace=True)
        return np.concatenate([boot0, boot1])
    n = group.shape[0]
    return rng.integers(0, n, size=n)


def get_kob_decomposition_bootstrap(
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
    n_boot: int = 500,
    random_state: Optional[int] = None,
    confidence_level: float = 0.95,
    stratified: bool = True,
) -> KOBBootstrapResult:
    if n_boot < 2:
        raise ValueError("[get_kob_decomposition_bootstrap] n_boot must be at least 2.")
    _validate_confidence_level(confidence_level)

    kwargs = dict(
        y=y,
        group=group,
        X=X,
        variable_names=variable_names,
        term_ids=term_ids,
        reference=reference,
        majority_owner=majority_owner,
        coefficient_owner_by_column=coefficient_owner_by_column,
        group0_value=group0_value,
        group1_value=group1_value,
        normalize_categorical=normalize_categorical,
        categorical_terms=categorical_terms,
        category_ids=category_ids,
        n_categories_by_term=n_categories_by_term,
        owner_by_category_by_term=owner_by_category_by_term,
        drop_missing=drop_missing,
    )
    point = get_kob_decomposition(**kwargs)

    y_arr = np.asarray(y, dtype=float)
    g_arr = np.asarray(group)
    X_arr = np.asarray(X, dtype=float)
    if drop_missing:
        valid = np.isfinite(y_arr) & np.all(np.isfinite(X_arr), axis=1)
        y_arr = y_arr[valid]
        g_arr = g_arr[valid]
        X_arr = X_arr[valid]

    rng = np.random.default_rng(random_state)

    boot_total_gap = np.empty(n_boot, dtype=float)
    boot_explained = np.empty(n_boot, dtype=float)
    boot_returns = np.empty(n_boot, dtype=float)
    boot_intercept = np.empty(n_boot, dtype=float)
    boot_by_column = np.empty((n_boot, point.by_column.shape[0], 2), dtype=float)
    boot_by_term = np.empty((n_boot, point.by_term.shape[0], 2), dtype=float)

    for b in range(n_boot):
        idx = _bootstrap_sample_indices(
            g_arr,
            group0_value=group0_value,
            group1_value=group1_value,
            rng=rng,
            stratified=stratified,
        )
        res = get_kob_decomposition(
            y=y_arr[idx],
            group=g_arr[idx],
            X=X_arr[idx],
            **{k: v for k, v in kwargs.items() if k not in {"y", "group", "X"}},
        )
        boot_total_gap[b] = res.total_gap
        boot_explained[b] = res.explained
        boot_returns[b] = res.unexplained_returns
        boot_intercept[b] = res.unexplained_intercept
        boot_by_column[b, :, 0] = res.by_column["explained"].to_numpy()
        boot_by_column[b, :, 1] = res.by_column["returns"].to_numpy()
        boot_by_term[b, :, 0] = res.by_term["explained"].to_numpy()
        boot_by_term[b, :, 1] = res.by_term["returns"].to_numpy()

    standard_errors = {
        "total_gap": float(np.std(boot_total_gap, ddof=1)),
        "explained": float(np.std(boot_explained, ddof=1)),
        "unexplained_returns": float(np.std(boot_returns, ddof=1)),
        "unexplained_intercept": float(np.std(boot_intercept, ddof=1)),
    }
    confidence_intervals = {
        "total_gap": _percentile_ci(boot_total_gap, confidence_level),
        "explained": _percentile_ci(boot_explained, confidence_level),
        "unexplained_returns": _percentile_ci(boot_returns, confidence_level),
        "unexplained_intercept": _percentile_ci(boot_intercept, confidence_level),
    }

    by_column_se = pd.DataFrame(
        {
            "column": point.by_column["column"],
            "term_id": point.by_column["term_id"],
            "explained_se": np.std(boot_by_column[:, :, 0], axis=0, ddof=1),
            "returns_se": np.std(boot_by_column[:, :, 1], axis=0, ddof=1),
        }
    )
    ci_lower, ci_upper = _percentile_bounds(confidence_level)
    by_column_ci = pd.DataFrame(
        {
            "column": point.by_column["column"],
            "term_id": point.by_column["term_id"],
            "explained_ci_lower": np.percentile(boot_by_column[:, :, 0], ci_lower, axis=0),
            "explained_ci_upper": np.percentile(boot_by_column[:, :, 0], ci_upper, axis=0),
            "returns_ci_lower": np.percentile(boot_by_column[:, :, 1], ci_lower, axis=0),
            "returns_ci_upper": np.percentile(boot_by_column[:, :, 1], ci_upper, axis=0),
        }
    )
    by_term_se = pd.DataFrame(
        {
            "term_id": point.by_term["term_id"],
            "explained_se": np.std(boot_by_term[:, :, 0], axis=0, ddof=1),
            "returns_se": np.std(boot_by_term[:, :, 1], axis=0, ddof=1),
        }
    )
    by_term_ci = pd.DataFrame(
        {
            "term_id": point.by_term["term_id"],
            "explained_ci_lower": np.percentile(boot_by_term[:, :, 0], ci_lower, axis=0),
            "explained_ci_upper": np.percentile(boot_by_term[:, :, 0], ci_upper, axis=0),
            "returns_ci_lower": np.percentile(boot_by_term[:, :, 1], ci_lower, axis=0),
            "returns_ci_upper": np.percentile(boot_by_term[:, :, 1], ci_upper, axis=0),
        }
    )

    return KOBBootstrapResult(
        point_estimate=point,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        by_column_standard_errors=by_column_se,
        by_column_confidence_intervals=by_column_ci,
        by_term_standard_errors=by_term_se,
        by_term_confidence_intervals=by_term_ci,
        n_boot=n_boot,
        confidence_level=confidence_level,
    )


__all__ = [
    "KOBDecompositionResult",
    "KOBBootstrapResult",
    "get_kob_decomposition",
    "get_kob_decomposition_bootstrap",
]
