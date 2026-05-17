"""
@Author  : 梁彧祺 Yuqi Liang, Jan Meyerhoff-Liang
@File    : residuals.py
@Time    : 10/04/2026 21:06
@Desc    :
    Pair-specific residual models for hierarchical sequence analysis.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from .data import RelationalSequenceData
from .distances import RelationalDistanceMatrix

ResidualMethod = Literal["simple", "additive", "crossed_anova"]


def _mean_distance_to_others(matrix: np.ndarray) -> np.ndarray:
    """For each row, mean distance to all other sequences."""
    n = matrix.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(n):
        others = np.concatenate([matrix[i, :i], matrix[i, i + 1 :]])
        out[i] = float(np.mean(others)) if len(others) else 0.0
    return out


def _fit_additive_expected(
    level_1_ids: np.ndarray,
    level_2_ids: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """
    Additive model on mean distances: E[m_i] = mu + alpha[l1_i] + beta[l2_i].

    Effects are estimated by iterative centering (alternating projections).
    """
    n = len(observed)
    grand = float(np.mean(observed))
    l1 = pd.Categorical(level_1_ids)
    l2 = pd.Categorical(level_2_ids)

    alpha = np.zeros(len(l1.categories), dtype=float)
    beta = np.zeros(len(l2.categories), dtype=float)

    for _ in range(50):
        for k, cat in enumerate(l1.categories):
            mask = l1.codes == k
            if mask.any():
                alpha[k] = float(np.mean(observed[mask] - grand - beta[l2.codes[mask]]))

        for k, cat in enumerate(l2.categories):
            mask = l2.codes == k
            if mask.any():
                beta[k] = float(np.mean(observed[mask] - grand - alpha[l1.codes[mask]]))

    fitted = grand + alpha[l1.codes] + beta[l2.codes]
    return fitted


def _fit_crossed_expected(
    level_1_ids: np.ndarray,
    level_2_ids: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """
    Cell-means model: expected mean distance for each (level_1, level_2) cell,
    shrunk toward additive mains when cells are empty.
    """
    df = pd.DataFrame(
        {
            "l1": level_1_ids,
            "l2": level_2_ids,
            "y": observed,
        }
    )
    cell_mean = df.groupby(["l1", "l2"], observed=True)["y"].transform("mean")
    additive = _fit_additive_expected(level_1_ids, level_2_ids, observed)
    # Blend: use cell mean when cell has multiple observations, else additive
    counts = df.groupby(["l1", "l2"], observed=True)["y"].transform("count")
    weight = np.clip((counts - 1) / 2.0, 0.0, 1.0)
    return weight * cell_mean.to_numpy() + (1.0 - weight) * additive


def compute_pair_residuals(
    sequence_data: RelationalSequenceData,
    distance_matrix: RelationalDistanceMatrix,
    *,
    method: ResidualMethod = "additive",
) -> pd.DataFrame:
    """
    Residualized pair-level distances after accounting for level structure.

    Scores reflect whether a pair is unusually similar or distant relative to
    additive level-1 and level-2 structure — not substantive labels such as
    "early RCA" (those require separate sequence features).

    Parameters
    ----------
    sequence_data : RelationalSequenceData
    distance_matrix : RelationalDistanceMatrix
    method : str
        - ``"additive"`` (default): mu + alpha(level_1) + beta(level_2)
        - ``"simple"``: mean distance minus average of same-level-1 and same-level-2 means
        - ``"crossed_anova"``: cell-means / additive blend (experimental)

    Returns
    -------
    pandas.DataFrame
        One row per pair with observed, expected, residual, standardized residual.
    """
    matrix = distance_matrix.matrix
    l1 = distance_matrix.level_1_ids
    l2 = distance_matrix.level_2_ids
    observed = _mean_distance_to_others(matrix)

    if method == "simple":
        n = len(observed)
        expected = np.zeros(n, dtype=float)
        for i in range(n):
            same_l1 = (l1 == l1[i]) & (np.arange(n) != i)
            same_l2 = (l2 == l2[i]) & (np.arange(n) != i)
            m1 = float(np.mean(matrix[i, same_l1])) if same_l1.any() else observed[i]
            m2 = float(np.mean(matrix[i, same_l2])) if same_l2.any() else observed[i]
            expected[i] = 0.5 * (m1 + m2)
    elif method == "additive":
        expected = _fit_additive_expected(l1, l2, observed)
    elif method == "crossed_anova":
        expected = _fit_crossed_expected(l1, l2, observed)
    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'simple', 'additive', or 'crossed_anova'."
        )

    residual = observed - expected
    std = float(np.std(residual, ddof=1)) if len(residual) > 1 else 1.0
    if std <= 0:
        std = 1.0
    standardized = residual / std

    base = sequence_data.to_dataframe()
    base["observed_mean_distance"] = observed
    base["expected_mean_distance"] = expected
    base["residual"] = residual
    base["standardized_residual"] = standardized
    return base


def detect_pair_specific_outliers(
    sequence_data: RelationalSequenceData,
    distance_matrix: RelationalDistanceMatrix,
    level_1_ids: Optional[np.ndarray] = None,
    level_2_ids: Optional[np.ndarray] = None,
    *,
    method: ResidualMethod = "additive",
    top_n: int = 20,
    z_threshold: float = 1.96,
) -> pd.DataFrame:
    """
    Identify unusual region–CPC pairs using residualized mean distances.

    Parameters
    ----------
    method : str
        Residual model (see :func:`compute_pair_residuals`).
    top_n : int
        Return the top ``top_n`` pairs by absolute standardized residual.
    z_threshold : float
        Flag ``is_outlier`` when |standardized residual| exceeds this value.
    """
    residuals = compute_pair_residuals(
        sequence_data,
        distance_matrix,
        method=method,
    )
    residuals["abs_standardized_residual"] = residuals["standardized_residual"].abs()
    residuals = residuals.sort_values("abs_standardized_residual", ascending=False)

    def _interpret(row: pd.Series) -> str:
        z = row["standardized_residual"]
        if abs(z) < z_threshold:
            return "typical given level-1 and level-2 structure"
        if z > 0:
            return "unusually distant (pair-specific deviation above expectation)"
        return "unusually similar (pair-specific convergence below expectation)"

    residuals["is_outlier"] = residuals["abs_standardized_residual"] >= z_threshold
    residuals["interpretation"] = residuals.apply(_interpret, axis=1)
    residuals["outlier_score"] = residuals["standardized_residual"]

    cols = [
        "pair_id",
        "level_1_id",
        "level_2_id",
        "observed_mean_distance",
        "expected_mean_distance",
        "residual",
        "standardized_residual",
        "outlier_score",
        "is_outlier",
        "interpretation",
    ]
    present = [c for c in cols if c in residuals.columns]
    return residuals[present].head(top_n).reset_index(drop=True)
