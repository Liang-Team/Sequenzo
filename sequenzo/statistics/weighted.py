"""
@Author  : Yuqi Liang 梁彧祺
@File    : weighted.py
@Time    : 2026-04-28 07:08
@Desc    : 
User-facing weighted statistics functions.
"""

from __future__ import annotations

import numpy as np

from sequenzo.utils.weighted_stats import (
    weighted_five_number_summary,
    weighted_mean,
    weighted_variance,
)


def get_weighted_mean(values: np.ndarray, weights: np.ndarray | None = None, remove_missing: bool = True) -> float:
    """
    Equivalent to TraMineR::weighted.mean() helper behavior.
    """
    return float(weighted_mean(values, weights=weights, na_rm=remove_missing))


def get_weighted_variance(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    remove_missing: bool = True,
    method: str = "unbiased",
) -> float:
    """
    Equivalent to TraMineR::weighted.var() helper behavior.
    """
    return float(weighted_variance(values, weights=weights, na_rm=remove_missing, method=method))


def get_weighted_five_number_summary(
    values: np.ndarray, weights: np.ndarray | None = None, remove_missing: bool = True
) -> np.ndarray:
    """
    Equivalent to TraMineR::weighted.fivenum() helper behavior.
    """
    return weighted_five_number_summary(values, weights=weights, na_rm=remove_missing)
