"""
@Author  : Yuqi Liang 梁彧祺
@File    : results.py
@Time    : 2026-05-13 07:52
@Desc    :
Result containers for Kitagawa-Oaxaca-Blinder decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .sa_kob import ClusterCovariates


@dataclass
class KOBDecompositionResult:
    total_gap: float
    explained: float
    unexplained_returns: float
    unexplained_intercept: float
    by_column: pd.DataFrame
    by_term: pd.DataFrame
    by_category: pd.DataFrame
    group0_mean: float
    group1_mean: float
    group0_label: Any
    group1_label: Any
    gap_direction: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def by_variable(self) -> pd.DataFrame:
        """Backward-compatible alias for :attr:`by_column`."""
        return self.by_column


@dataclass
class KOBBootstrapResult:
    point_estimate: KOBDecompositionResult
    standard_errors: Dict[str, float]
    confidence_intervals: Dict[str, tuple[float, float]]
    by_column_standard_errors: pd.DataFrame
    by_column_confidence_intervals: pd.DataFrame
    by_term_standard_errors: pd.DataFrame
    by_term_confidence_intervals: pd.DataFrame
    n_boot: int
    confidence_level: float


@dataclass
class SAKOBDecompositionResult:
    """SA-KOB result wrapping the generic KOB output and cluster metadata."""

    kob: KOBDecompositionResult
    cluster_composition: pd.DataFrame
    cluster_owners: pd.DataFrame
    by_cluster: pd.DataFrame
    cluster_covariates: ClusterCovariates
    common_support_table: pd.DataFrame

    @property
    def total_gap(self) -> float:
        return self.kob.total_gap

    @property
    def explained(self) -> float:
        return self.kob.explained

    @property
    def unexplained_returns(self) -> float:
        return self.kob.unexplained_returns

    @property
    def unexplained_intercept(self) -> float:
        return self.kob.unexplained_intercept

    @property
    def by_column(self) -> pd.DataFrame:
        return self.kob.by_column

    @property
    def by_term(self) -> pd.DataFrame:
        return self.kob.by_term

    @property
    def by_category(self) -> pd.DataFrame:
        return self.kob.by_category

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return self.kob.diagnostics

    @property
    def explained_detailed(self) -> float:
        """Sum of Yun-normalized explained contributions in ``by_cluster``."""
        return float(self.by_cluster["explained"].sum())

    @property
    def returns_detailed(self) -> float:
        """Sum of Yun-normalized returns contributions in ``by_cluster``."""
        return float(self.by_cluster["returns"].sum())

    @property
    def explained_difference(self) -> float:
        """``explained_detailed`` minus raw twofold ``explained``."""
        return self.explained_detailed - self.explained

    @property
    def returns_difference(self) -> float:
        """``returns_detailed`` minus raw twofold ``unexplained_returns``."""
        return self.returns_detailed - self.unexplained_returns


@dataclass
class SAKOBBootstrapResult:
    point_estimate: SAKOBDecompositionResult
    standard_errors: Dict[str, float]
    confidence_intervals: Dict[str, tuple[float, float]]
    by_cluster_standard_errors: pd.DataFrame
    by_cluster_confidence_intervals: pd.DataFrame
    n_boot: int
    confidence_level: float
    recompute_owners_each_draw: bool


__all__ = [
    "KOBDecompositionResult",
    "KOBBootstrapResult",
    "SAKOBDecompositionResult",
    "SAKOBBootstrapResult",
]
