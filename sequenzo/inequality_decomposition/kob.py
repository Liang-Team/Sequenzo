"""
@Author  : Yuqi Liang 梁彧祺
@File    : kob.py
@Time    : 2026-03-01 13:26
@Desc    : 
Kitagawa-Oaxaca-Blinder decomposition public API.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Literal

import numpy as np
import pandas as pd

from .oaxaca import oaxaca_blinder_decomposition


@dataclass
class KOBDecompositionResult:
    total_gap: float
    explained: float
    unexplained_returns: float
    unexplained_intercept: float
    by_variable: pd.DataFrame
    group0_mean: float
    group1_mean: float


def get_kob_decomposition(
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


__all__ = [
    "KOBDecompositionResult",
    "get_kob_decomposition",
]
